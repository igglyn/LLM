#!/usr/bin/env python
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
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from blt_lite.model import PatcherAutoencoder
from blt_lite.tokenizer import FixedPatchTokenizer
from blt_lite.utils import ensure_dir, get_device, load_config, set_seed


class HiddenReconstructionDataset(Dataset):
    def __init__(self, hidden_path: Path, seq_len: int):
        hidden = np.load(hidden_path, mmap_mode="r")
        if hidden.ndim != 2:
            raise ValueError(f"Expected 2D hidden cache at {hidden_path}, got shape={hidden.shape}")
        if hidden.shape[0] <= seq_len:
            raise ValueError("Not enough cached hidden states for sequence length")
        self.hidden = torch.from_numpy(np.asarray(hidden, dtype=np.float32))
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.hidden.shape[0] - self.seq_len

    def __getitem__(self, idx: int):
        x = self.hidden[idx : idx + self.seq_len]
        return x, x


def _token_seq_len_from_cfg(cfg: dict) -> int:
    model_cfg = cfg["model"]
    p1 = int(cfg.get("patcher", {}).get("patch_size", 1))
    p2 = int(cfg.get("patcher2", {}).get("patch_size", 2)) if bool(cfg.get("patcher2", {}).get("enabled", True)) else 1
    default_seq_len = int(model_cfg["seq_len"]) * p1 * p2
    return int(cfg.get("patcher2_train", {}).get("seq_len_tokens", default_seq_len))


def build_stage2(cfg: dict, tokenizer: FixedPatchTokenizer, device: torch.device):
    model_cfg = cfg["model"]
    p2cfg = cfg.get("patcher2", {})
    seq_len = _token_seq_len_from_cfg(cfg)
    d_model = int(model_cfg["d_model"])
    p2 = PatcherAutoencoder(
        in_dim=d_model,
        latent_dim=int(p2cfg.get("latent_dim", d_model)),
        out_dim=d_model,
        patch_size=int(p2cfg.get("patch_size", 2)),
        seq_len=seq_len,
        encoder_layers=int(p2cfg.get("encoder_layers", 2)),
        decoder_layers=int(p2cfg.get("decoder_layers", 2)),
        n_heads=int(p2cfg.get("n_heads", model_cfg["n_heads"])),
        dropout=float(p2cfg.get("dropout", model_cfg["dropout"])),
        pos_encoding=str(p2cfg.get("pos_encoding", "learned")),
        grad_checkpointing=bool(p2cfg.get("grad_checkpointing", False)),
        flash_attention=bool(p2cfg.get("flash_attention", True)),
        block_attention=bool(p2cfg.get("block_attention", False)),
        block_size=int(p2cfg.get("block_size", 8)),
    ).to(device)
    return p2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not bool(cfg.get("patcher2", {}).get("enabled", True)):
        raise ValueError("patcher2 is disabled in config; skip train_patcher2.py or enable patcher2.enabled")
    set_seed(int(cfg["train"].get("seed", 42)))

    device = get_device()
    processed_dir = Path(cfg["data"]["processed_dir_patcher2"])
    tokenizer = FixedPatchTokenizer.load(processed_dir / "tokenizer.json")

    seq_len = _token_seq_len_from_cfg(cfg)
    train_hidden = processed_dir / "train_stage1_hidden.npy"
    val_hidden = processed_dir / "val_stage1_hidden.npy"
    if not train_hidden.exists() or not val_hidden.exists():
        raise FileNotFoundError("Missing cached stage1 hidden states. Run scripts/prepare_data_patcher2.py first.")

    p2train = cfg.get("patcher2_train", {})
    train_loader = DataLoader(
        HiddenReconstructionDataset(train_hidden, seq_len),
        batch_size=int(p2train.get("batch_size", cfg["train"]["batch_size"])),
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        HiddenReconstructionDataset(val_hidden, seq_len),
        batch_size=int(p2train.get("batch_size", cfg["train"]["batch_size"])),
        shuffle=False,
        drop_last=False,
    )

    patcher2 = build_stage2(cfg, tokenizer, device)

    optimizer = AdamW(
        patcher2.parameters(),
        lr=float(p2train.get("lr", 3e-4)),
        weight_decay=float(p2train.get("weight_decay", 0.01)),
    )
    amp_enabled = bool(p2train.get("amp_enabled", cfg.get("train", {}).get("amp_enabled", True)))
    amp_dtype = torch.float16 if str(p2train.get("amp_dtype", cfg.get("train", {}).get("amp_dtype", "float16"))) == "float16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and amp_enabled))

    out_dir = ensure_dir(p2train.get("out_dir", "outputs/patcher2"))
    max_steps = int(p2train.get("max_steps", 3000))
    eval_every = int(p2train.get("eval_every", 100))
    eval_batches = int(p2train.get("eval_batches", 50))
    save_every = int(p2train.get("save_every", 200))

    def batch_loss(h: torch.Tensor) -> torch.Tensor:
        recon, _ = patcher2(h)
        return torch.nn.functional.mse_loss(recon, h)

    best_val = float("inf")
    step = 0
    patcher2.train()
    while step < max_steps:
        for h, _ in train_loader:
            h = h.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and amp_enabled), dtype=amp_dtype):
                loss = batch_loss(h)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(patcher2.parameters(), float(p2train.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()

            if step % 20 == 0:
                print(f"patcher2_step={step} recon_loss={loss.item():.6f}")

            if step % eval_every == 0 and step > 0:
                patcher2.eval()
                losses = []
                with torch.no_grad():
                    for vh, _ in val_loader:
                        vh = vh.to(device)
                        with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and amp_enabled), dtype=amp_dtype):
                            losses.append(batch_loss(vh).item())
                        if len(losses) >= eval_batches:
                            break
                val_loss = float(sum(losses) / max(1, len(losses)))
                print(f"patcher2_step={step} val_recon_loss={val_loss:.6f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({"patcher2": patcher2.state_dict(), "config": cfg}, out_dir / "best.pt")
                patcher2.train()

            if step % save_every == 0 and step > 0:
                torch.save({"patcher2": patcher2.state_dict(), "config": cfg}, out_dir / f"step_{step}.pt")

            step += 1
            if step >= max_steps:
                break

    torch.save({"patcher2": patcher2.state_dict(), "config": cfg}, out_dir / "last.pt")
    print(f"Second patcher pretraining complete. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
