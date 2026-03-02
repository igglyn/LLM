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
import re

import torch
from torch.optim import AdamW

from blt_lite.model import TinyPatchLM
from blt_lite.tokenizer import FixedPatchTokenizer
from blt_lite.train import build_dataloaders, evaluate
from blt_lite.utils import ensure_dir, get_device, load_config, set_seed


_STEP_RE = re.compile(r"step_(\d+)\.pt$")


def warmup_cosine_lr(step: int, max_steps: int, warmup_steps: int, lr_max: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)

    decay_total = max(1, max_steps - warmup_steps)
    decay_step = min(step - warmup_steps, decay_total)
    cosine = 0.5 * (1 + math.cos(math.pi * decay_step / decay_total))
    return lr_min + (lr_max - lr_min) * cosine


def parse_step_from_checkpoint_name(path: Path) -> int:
    match = _STEP_RE.search(path.name)
    if match:
        return int(match.group(1))
    if path.name in {"best.pt", "last.pt"}:
        return 0
    raise ValueError(f"Checkpoint filename must match step_<N>.pt, best.pt, or last.pt; got: {path.name}")


def _resolve_checkpoint_path(raw_path: str, out_dir: Path) -> Path:
    ckpt_path = Path(raw_path)
    if not ckpt_path.is_absolute() and not ckpt_path.exists():
        ckpt_path = out_dir / ckpt_path
    return ckpt_path


def _build_model_from_cfg(cfg: dict, tokenizer: FixedPatchTokenizer, device: torch.device) -> TinyPatchLM:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    patcher_cfg = cfg.get("patcher", {})
    return TinyPatchLM(
        vocab_size=tokenizer.vocab_len,
        seq_len=int(data_cfg["seq_len"]),
        patch_size=int(getattr(tokenizer, "patch_size", cfg.get("tokenizer", {}).get("patch_size", 1))),
        d_model=int(model_cfg["d_model"]),
        n_layers=int(model_cfg["n_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        dropout=float(model_cfg["dropout"]),
        patcher_latent_dim=int(patcher_cfg.get("latent_dim", model_cfg["d_model"])),
        patcher_encoder_layers=int(patcher_cfg.get("encoder_layers", 2)),
        patcher_decoder_layers=int(patcher_cfg.get("decoder_layers", 2)),
        patcher_heads=int(patcher_cfg.get("n_heads", model_cfg["n_heads"])),
        patcher_dropout=float(patcher_cfg.get("dropout", model_cfg["dropout"])),
    ).to(device)




def _maybe_load_pretrained_patcher(model: TinyPatchLM, cfg: dict, device: torch.device) -> None:
    patcher_cfg = cfg.get("patcher", {})
    path = patcher_cfg.get("pretrained_path", "")
    if path:
        model.load_patcher_checkpoint(path, map_location=device)
        print(f"Loaded pretrained patcher from {path}")
    if bool(patcher_cfg.get("freeze", False)):
        for p in model.patcher.parameters():
            p.requires_grad = False
        print("Froze patcher parameters")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional checkpoint filename or path to resume from (expects step_<N>.pt naming).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    set_seed(int(tcfg.get("seed", 42)))

    device = get_device()
    out_dir = ensure_dir(tcfg.get("out_dir", "outputs"))
    processed_dir = Path(cfg["data"]["processed_dir"])
    tokenizer = FixedPatchTokenizer.load(processed_dir / "tokenizer.json")

    resume_ckpt = None
    step = 0
    if args.checkpoint:
        ckpt_path = _resolve_checkpoint_path(args.checkpoint, out_dir)
        resume_ckpt = torch.load(ckpt_path, map_location=device)
        if "config" in resume_ckpt and isinstance(resume_ckpt["config"], dict):
            cfg = resume_ckpt["config"]
            tcfg = cfg["train"]
        step = parse_step_from_checkpoint_name(ckpt_path)

    seq_len = int(cfg["data"]["seq_len"])
    train_loader, val_loader = build_dataloaders(
        processed_dir / "train_tokens.npy",
        processed_dir / "val_tokens.npy",
        seq_len=seq_len,
        batch_size=int(tcfg["batch_size"]),
    )

    model = _build_model_from_cfg(cfg, tokenizer, device)
    _maybe_load_pretrained_patcher(model, cfg, device)
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        print(f"Resumed from checkpoint {ckpt_path} at step={step}")

    optimizer = AdamW(model.parameters(), lr=float(tcfg.get("lr_max", 3e-4)), weight_decay=float(tcfg["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    max_steps = int(tcfg["max_steps"])
    eval_every = int(tcfg.get("eval_every", 100))
    save_every = int(tcfg.get("save_every", 200))
    grad_accum_steps = int(tcfg.get("grad_accum_steps", 1))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    lr_max = float(tcfg.get("lr_max", 3e-4))
    lr_min = float(tcfg.get("lr_min", 3e-5))
    warmup_steps = int(tcfg.get("warmup_steps", 1000))
    eval_batches = int(tcfg.get("eval_batches", 50))

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
