#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import math

import torch
from torch.optim import AdamW

from blt_lite.model import TinyPatchLM
from blt_lite.tokenizer import FixedPatchTokenizer
from blt_lite.train import build_dataloaders, evaluate
from blt_lite.utils import ensure_dir, get_device, load_config, set_seed


def warmup_cosine_lr(step: int, max_steps: int, warmup_steps: int, lr_max: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)

    decay_total = max(1, max_steps - warmup_steps)
    decay_step = min(step - warmup_steps, decay_total)
    cosine = 0.5 * (1 + math.cos(math.pi * decay_step / decay_total))
    return lr_min + (lr_max - lr_min) * cosine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    set_seed(int(tcfg.get("seed", 42)))

    device = get_device()
    processed_dir = Path(cfg["data"]["processed_dir"])
    tokenizer = FixedPatchTokenizer.load(processed_dir / "tokenizer.json")
    seq_len = int(cfg["data"]["seq_len"])

    train_loader, val_loader = build_dataloaders(
        processed_dir / "train_tokens.npy",
        processed_dir / "val_tokens.npy",
        seq_len=seq_len,
        batch_size=int(tcfg["batch_size"]),
    )

    model = TinyPatchLM(
        vocab_size=tokenizer.vocab_len,
        seq_len=seq_len,
        patch_size=int(getattr(tokenizer, "patch_size", cfg.get("tokenizer", {}).get("patch_size", 1))),
        d_model=int(cfg["model"]["d_model"]),
        n_layers=int(cfg["model"]["n_layers"]),
        n_heads=int(cfg["model"]["n_heads"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=float(tcfg.get("lr_max", 3e-4)), weight_decay=float(tcfg["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    out_dir = ensure_dir(tcfg.get("out_dir", "outputs"))
    max_steps = int(tcfg["max_steps"])
    eval_every = int(tcfg.get("eval_every", 100))
    save_every = int(tcfg.get("save_every", 200))
    grad_accum_steps = int(tcfg.get("grad_accum_steps", 1))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    lr_max = float(tcfg.get("lr_max", 3e-4))
    lr_min = float(tcfg.get("lr_min", 3e-5))
    warmup_steps = int(tcfg.get("warmup_steps", 1000))
    eval_batches = int(tcfg.get("eval_batches", 50))

    step = 0
    best_val = float("inf")
    model.train()
    while step < max_steps:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            lr = warmup_cosine_lr(step, max_steps, warmup_steps, lr_max, lr_min)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                _, loss = model(x, y)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if step % 20 == 0:
                print(f"step={step} train_loss={loss.item() * grad_accum_steps:.4f} lr={lr:.6f}")

            if step % eval_every == 0 and step > 0:
                val_loss = evaluate(model, val_loader, device, max_batches=eval_batches)
                print(f"step={step} val_loss={val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": cfg,
                            "val_loss": val_loss,
                        },
                        out_dir / "best.pt",
                    )

            if step % save_every == 0 and step > 0:
                torch.save({"model": model.state_dict(), "config": cfg}, out_dir / f"step_{step}.pt")

            step += 1
            if step >= max_steps:
                break

    torch.save({"model": model.state_dict(), "config": cfg}, out_dir / "last.pt")
    print(f"Training complete. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
