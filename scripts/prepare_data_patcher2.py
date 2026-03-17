#!/usr/bin/env python
"""
prepare_data_patcher2.py

Runs the frozen patcher1 + token embedding over the full corpus and caches
the resulting hidden states to disk as .npy files.

These cached states are the training data for patcher2 (train_patcher2.py).

Without this step, patcher2 has nothing to train on.

Usage:
    python scripts/prepare_data_patcher2.py --config configs/tiny.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from blt_lite.dataset import TokenSequenceDataset
from blt_lite.model import PatcherAutoencoder
from blt_lite.tokenizer import FixedPatchTokenizer
from blt_lite.utils import ensure_dir, get_device, load_config, set_seed


def _token_seq_len_from_cfg(cfg: dict) -> int:
    model_cfg = cfg["model"]
    p1 = int(cfg.get("patcher", {}).get("patch_size", 1))
    default_seq_len = int(model_cfg["seq_len"]) * p1
    return int(cfg.get("patcher_train", {}).get("seq_len_tokens", default_seq_len))


def _build_patcher_and_embed(cfg: dict, tokenizer: FixedPatchTokenizer, device: torch.device):
    model_cfg = cfg["model"]
    patcher_cfg = cfg.get("patcher", {})
    seq_len = _token_seq_len_from_cfg(cfg)
    d_model = int(model_cfg["d_model"])

    emb = torch.nn.Embedding(tokenizer.vocab_len, d_model).to(device)
    patcher = PatcherAutoencoder(
        in_dim=d_model,
        latent_dim=int(patcher_cfg.get("latent_dim", d_model)),
        out_dim=d_model,
        patch_size=int(patcher_cfg.get("patch_size", 1)),
        seq_len=seq_len,
        encoder_layers=int(patcher_cfg.get("encoder_layers", 2)),
        decoder_layers=int(patcher_cfg.get("decoder_layers", 2)),
        n_heads=int(patcher_cfg.get("n_heads", model_cfg["n_heads"])),
        dropout=0.0,   # no dropout during caching
        pos_encoding=str(patcher_cfg.get("pos_encoding", "learned")),
        grad_checkpointing=False,
        flash_attention=bool(patcher_cfg.get("flash_attention", True)),
        block_attention=bool(patcher_cfg.get("block_attention", False)),
        block_size=int(patcher_cfg.get("block_size", 8)),
    ).to(device)
    return emb, patcher


def _load_patcher_checkpoint(emb, patcher, path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    patcher.load_state_dict(ckpt["patcher"])
    emb.load_state_dict(ckpt["token_emb"])
    print(f"Loaded patcher checkpoint from {path}")


@torch.no_grad()
def _cache_hidden_states(
    emb: torch.nn.Embedding,
    patcher: PatcherAutoencoder,
    tokens: np.ndarray,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    """
    Run token_emb + patcher over the full token stream.
    Returns a 2D float32 array of shape (N, d_model) — one vector per token position,
    after patcher reconstruction.
    """
    dataset = TokenSequenceDataset(tokens, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    emb.eval()
    patcher.eval()

    all_hidden = []
    for batch_idx, (x, _) in enumerate(loader):
        x = x.to(device)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and amp_enabled), dtype=amp_dtype):
            h = emb(x)               # (B, T, D)
            recon, _ = patcher(h)    # (B, T, D) — reconstructed hidden states
        # Flatten batch and time into (B*T, D), keep as float32
        all_hidden.append(recon.float().cpu().numpy().reshape(-1, recon.shape[-1]))

        if batch_idx % 50 == 0:
            print(f"  cached batch {batch_idx}/{len(loader)}")

    return np.concatenate(all_hidden, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Cache patcher1 hidden states for patcher2 training."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--patcher-checkpoint",
        default="",
        help="Path to patcher checkpoint. Overrides patcher.pretrained_path in config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if not bool(cfg.get("patcher2", {}).get("enabled", True)):
        raise ValueError("patcher2 is disabled in config; nothing to prepare.")

    set_seed(int(cfg["train"].get("seed", 42)))
    device = get_device()

    amp_enabled = bool(cfg.get("patcher_train", {}).get("amp_enabled", True))
    amp_dtype = (
        torch.float16
        if str(cfg.get("patcher_train", {}).get("amp_dtype", "float16")) == "float16"
        else torch.bfloat16
    )

    # --- resolve patcher checkpoint ---
    ckpt_path_str = args.patcher_checkpoint or cfg.get("patcher", {}).get("pretrained_path", "")
    if not ckpt_path_str:
        raise ValueError(
            "Provide a patcher checkpoint via --patcher-checkpoint or patcher.pretrained_path in config."
        )
    ckpt_path = Path(ckpt_path_str)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Patcher checkpoint not found: {ckpt_path}")

    # --- load tokenizer and tokens ---
    patcher_dir = Path(cfg["data"]["processed_dir_patcher"])
    tokenizer = FixedPatchTokenizer.load(patcher_dir / "tokenizer.json")
    train_tokens = np.load(patcher_dir / "train_tokens.npy")
    val_tokens   = np.load(patcher_dir / "val_tokens.npy")

    seq_len    = _token_seq_len_from_cfg(cfg)
    batch_size = int(cfg.get("patcher_train", {}).get("batch_size", 8))

    # --- build and load patcher ---
    emb, patcher = _build_patcher_and_embed(cfg, tokenizer, device)
    _load_patcher_checkpoint(emb, patcher, ckpt_path, device)

    # Freeze — we are only caching, not training
    for p in list(emb.parameters()) + list(patcher.parameters()):
        p.requires_grad_(False)

    # --- cache ---
    out_dir = ensure_dir(cfg["data"]["processed_dir_patcher2"])

    print("Caching train hidden states...")
    train_hidden = _cache_hidden_states(
        emb, patcher, train_tokens, seq_len, batch_size, device, amp_enabled, amp_dtype
    )
    np.save(out_dir / "train_stage1_hidden.npy", train_hidden)
    print(f"  saved train_stage1_hidden.npy  shape={train_hidden.shape}")

    print("Caching val hidden states...")
    val_hidden = _cache_hidden_states(
        emb, patcher, val_tokens, seq_len, batch_size, device, amp_enabled, amp_dtype
    )
    np.save(out_dir / "val_stage1_hidden.npy", val_hidden)
    print(f"  saved val_stage1_hidden.npy    shape={val_hidden.shape}")

    # Mirror tokenizer so train_patcher2.py can find it
    import shutil
    shutil.copy2(patcher_dir / "tokenizer.json", out_dir / "tokenizer.json")

    print(f"\nDone. Cached hidden states written to {out_dir}")
    print("Next step: python scripts/train_patcher2.py --config <config>")


if __name__ == "__main__":
    main()
