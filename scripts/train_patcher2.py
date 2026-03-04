#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

import torch
from torch.optim import AdamW

from blt_lite.model import PatcherAutoencoder
from blt_lite.tokenizer import FixedPatchTokenizer
from blt_lite.train import build_dataloaders
from blt_lite.utils import ensure_dir, get_device, load_config, set_seed


def build_stage1(cfg: dict, tokenizer: FixedPatchTokenizer, device: torch.device):
    model_cfg = cfg["model"]
    patcher_cfg = cfg["patcher"]
    seq_len = int(cfg["data"]["seq_len"])
    patch_size = int(cfg.get("patcher", {}).get("patch_size", getattr(tokenizer, "patch_size", 1)))
    d_model = int(model_cfg["d_model"])

    emb = torch.nn.Embedding(tokenizer.vocab_len, d_model).to(device)
    p1 = PatcherAutoencoder(
        in_dim=d_model,
        latent_dim=int(patcher_cfg.get("latent_dim", d_model)),
        out_dim=d_model,
        patch_size=patch_size,
        seq_len=seq_len,
        encoder_layers=int(patcher_cfg.get("encoder_layers", 2)),
        decoder_layers=int(patcher_cfg.get("decoder_layers", 2)),
        n_heads=int(patcher_cfg.get("n_heads", model_cfg["n_heads"])),
        dropout=float(patcher_cfg.get("dropout", model_cfg["dropout"])),
        pos_encoding=str(patcher_cfg.get("pos_encoding", "learned")),
    ).to(device)
    return emb, p1


def build_stage2(cfg: dict, tokenizer: FixedPatchTokenizer, device: torch.device):
    model_cfg = cfg["model"]
    p2cfg = cfg.get("patcher2", {})
    seq_len = int(cfg["data"]["seq_len"])
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
    ).to(device)
    return p2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["train"].get("seed", 42)))

    device = get_device()
    processed_dir = Path(cfg["data"]["processed_dir"])
    tokenizer = FixedPatchTokenizer.load(processed_dir / "tokenizer.json")

    p2train = cfg.get("patcher2_train", {})
    train_loader, val_loader = build_dataloaders(
        processed_dir / "train_tokens.npy",
        processed_dir / "val_tokens.npy",
        seq_len=int(cfg["data"]["seq_len"]),
        batch_size=int(p2train.get("batch_size", cfg["train"]["batch_size"])),
    )

    token_emb, patcher1 = build_stage1(cfg, tokenizer, device)
    patcher2 = build_stage2(cfg, tokenizer, device)

    p1_path = cfg.get("patcher", {}).get("pretrained_path", "")
    if not p1_path:
        raise ValueError("patcher.pretrained_path must be set before training patcher2")
    ckpt = torch.load(p1_path, map_location=device)
    if "token_emb" in ckpt:
        token_emb.load_state_dict(ckpt["token_emb"])
    state = ckpt["patcher"] if isinstance(ckpt, dict) and "patcher" in ckpt else ckpt
    patcher1.load_state_dict(state)
    token_emb.eval(); patcher1.eval()
    for p in token_emb.parameters():
        p.requires_grad = False
    for p in patcher1.parameters():
        p.requires_grad = False

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

    def batch_loss(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            token_hidden = token_emb(x)
            stage1_hidden, _ = patcher1(token_hidden)
        stage2_hidden, _ = patcher2(stage1_hidden)
        return torch.nn.functional.mse_loss(stage2_hidden, stage1_hidden)

    best_val = float("inf")
    step = 0
    patcher2.train()
    while step < max_steps:
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and amp_enabled), dtype=amp_dtype):
                loss = batch_loss(x)
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
                    for vx, _ in val_loader:
                        vx = vx.to(device)
                        with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and amp_enabled), dtype=amp_dtype):
                            losses.append(batch_loss(vx).item())
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
