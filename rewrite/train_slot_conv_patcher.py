#!/usr/bin/env python
"""Train the rewrite slot-conv patcher autoencoder.

This is intentionally split from rewrite/train_patcher.py so slot-conv
experiments can run independently from the transformer patcher pipeline.
For now this entrypoint is keyed off second-patcher config blocks
(`patcher2` / `patcher2_train`).
"""

from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rewrite.patcher_models import SlotConvAutoencoder


class HiddenSequenceDataset(Dataset):
    """Sliding windows over patcher2 hidden-state cache.

    Expects a 2D float array shaped [N, D] (stream of token-level hidden states).
    Returns windows shaped [T, D].
    """

    def __init__(self, hidden: np.ndarray, seq_len: int):
        if hidden.ndim != 2:
            raise ValueError(f"Expected hidden states with rank 2 [N, D], got rank {hidden.ndim}")
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if hidden.shape[0] <= seq_len:
            raise ValueError("Not enough cached hidden states for sequence length")
        self.hidden = torch.from_numpy(hidden.astype(np.float32))
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.hidden.shape[0] - self.seq_len

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.hidden[idx : idx + self.seq_len]


def _seq_len_from_cfg(cfg: dict) -> int:
    model_cfg = cfg["model"]
    p1 = int(cfg.get("patcher", {}).get("patch_size", 1))
    p2_enabled = bool(cfg.get("patcher2", {}).get("enabled", True))
    p2 = int(cfg.get("patcher2", {}).get("patch_size", 2)) if p2_enabled else 1
    default_seq_len = int(model_cfg["seq_len"]) * p1 * p2
    return int(cfg.get("patcher2_train", {}).get("seq_len_tokens", default_seq_len))


def _validate_known_keys(section_name: str, section_cfg: dict, allowed_keys: set[str]) -> None:
    unknown = sorted(k for k in section_cfg if k not in allowed_keys)
    if not unknown:
        return

    messages = []
    for key in unknown:
        suggestion = difflib.get_close_matches(key, sorted(allowed_keys), n=1)
        hint = f" (did you mean '{suggestion[0]}'?)" if suggestion else ""
        messages.append(f"{section_name}.{key}{hint}")
    raise ValueError(
        "Unknown config key(s) detected for slot-conv training: "
        + ", ".join(messages)
    )


def _resolve_slot_conv_dims(patcher_cfg: dict, d_model: int) -> tuple[int, int]:
    groups = patcher_cfg.get("groups", patcher_cfg.get("group_count", patcher_cfg.get("num_groups")))
    d_chunk = patcher_cfg.get("d_chunk", patcher_cfg.get("chunk_dim", patcher_cfg.get("group_width")))

    if groups is None and d_chunk is None:
        raise ValueError(
            "slot_conv requires patcher2.groups and patcher2.d_chunk (or aliases group_count/chunk_dim)."
        )
    if groups is None:
        d_chunk = int(d_chunk)
        if d_chunk <= 0 or d_model % d_chunk != 0:
            raise ValueError(f"Cannot infer groups: d_model={d_model} must be divisible by d_chunk={d_chunk}")
        groups = d_model // d_chunk
    if d_chunk is None:
        groups = int(groups)
        if groups <= 0 or d_model % groups != 0:
            raise ValueError(f"Cannot infer d_chunk: d_model={d_model} must be divisible by groups={groups}")
        d_chunk = d_model // groups

    groups = int(groups)
    d_chunk = int(d_chunk)
    if groups * d_chunk != d_model:
        raise ValueError(f"slot_conv expects groups*d_chunk == d_model, got {groups}*{d_chunk} != {d_model}")
    return groups, d_chunk


def _maybe_prepare_stage1_cache(cfg_path: str) -> None:
    cmd = [sys.executable, "scripts/prepare_data_patcher2.py", "--config", cfg_path]
    print(f"Stage1 hidden cache missing; running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rewrite slot-conv patcher")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument(
        "--prepare-data-if-missing",
        action="store_true",
        help="If train_stage1_hidden.npy is missing, run scripts/prepare_data_patcher2.py automatically.",
    )
    parser.add_argument(
        "--print-effective-config",
        action="store_true",
        help="Print resolved slot-conv wiring (groups/chunk/seq_len) before training.",
    )
    args = parser.parse_args()

    from blt_lite.utils import get_device, load_config, set_seed

    cfg = load_config(args.config)
    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    device = get_device()

    model_cfg = cfg["model"]
    patcher_cfg = cfg.get("patcher2", {})
    train_cfg = cfg.get("patcher2_train", {})

    _validate_known_keys(
        "patcher2",
        patcher_cfg,
        {
            "enabled",
            "type",
            "patch_size",
            "latent_dim",
            "pos_encoding",
            "encoder_layers",
            "decoder_layers",
            "n_heads",
            "dropout",
            "grad_checkpointing",
            "flash_attention",
            "block_attention",
            "block_size",
            "pretrained_path",
            "groups",
            "d_chunk",
            "kernel_size",
            "hidden_mult",
            "use_residual",
            "group_count",
            "num_groups",
            "chunk_dim",
            "group_width",
        },
    )
    _validate_known_keys(
        "patcher2_train",
        train_cfg,
        {"batch_size", "max_steps", "log_every", "lr", "seq_len_tokens"},
    )

    patcher_type = str(patcher_cfg.get("type", "slot_conv")).lower()
    if patcher_type != "slot_conv":
        raise ValueError("rewrite/train_slot_conv_patcher.py requires patcher2.type=slot_conv")

    d_model = int(model_cfg["d_model"])
    groups, d_chunk = _resolve_slot_conv_dims(patcher_cfg, d_model)

    processed_root = Path(cfg["data"]["processed_dir_patcher2"])
    train_hidden_path = processed_root / "train_stage1_hidden.npy"
    if not train_hidden_path.exists():
        if args.prepare_data_if_missing:
            _maybe_prepare_stage1_cache(args.config)
        if not train_hidden_path.exists():
            raise FileNotFoundError(
                f"Missing patcher2 hidden cache: {train_hidden_path}. "
                "Run scripts/prepare_data_patcher2.py first."
            )

    hidden = np.load(train_hidden_path)
    if hidden.shape[1] != d_model:
        raise ValueError(f"Expected hidden width {d_model}, got {hidden.shape[1]} from {train_hidden_path}")
    seq_len = _seq_len_from_cfg(cfg)
    batch_size = int(train_cfg.get("batch_size", cfg.get("train", {}).get("batch_size", 8)))
    train_loader = DataLoader(HiddenSequenceDataset(hidden, seq_len), batch_size=batch_size, shuffle=True, drop_last=True)
    if args.print_effective_config:
        print(
            "slot_conv_wiring "
            f"type=slot_conv d_model={d_model} groups={groups} d_chunk={d_chunk} "
            f"seq_len_tokens={seq_len} hidden_cache={train_hidden_path}"
        )

    patcher = SlotConvAutoencoder(
        groups=groups,
        d_chunk=d_chunk,
        d_model=d_model,
        kernel_size=int(patcher_cfg.get("kernel_size", 3)),
        hidden_mult=int(patcher_cfg.get("hidden_mult", 1)),
        dropout=float(patcher_cfg.get("dropout", 0.0)),
        use_residual=bool(patcher_cfg.get("use_residual", True)),
    ).to(device)

    optimizer = AdamW(patcher.parameters(), lr=float(train_cfg.get("lr", 3e-4)))
    max_steps = int(train_cfg.get("max_steps", 100))
    log_every = int(train_cfg.get("log_every", 10))

    patcher.train()
    step = 0

    while step < max_steps:
        for batch in train_loader:
            token_hidden = batch.to(device)
            slots = token_hidden.view(token_hidden.size(0), token_hidden.size(1), groups, d_chunk)
            recon_hidden, _ = patcher(slots)
            loss = torch.nn.functional.mse_loss(recon_hidden, slots)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            if step % log_every == 0 or step == 1:
                print(f"rewrite_slot_conv_step={step} loss={loss.item():.8f}")

            if step >= max_steps:
                break


if __name__ == "__main__":
    main()
