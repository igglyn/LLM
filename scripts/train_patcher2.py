#!/usr/bin/env python
"""train_patcher2.py — Train second patcher with inline patcher1 forward on CPU.

Patcher1 runs on CPU to avoid contention with patcher2 training on GPU.
No cache required — patcher1 embeddings computed on-the-fly each batch.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

import torch
from torch.optim import AdamW

from blt_lite.model import PatcherAutoencoder
from blt_lite.tokenizer import FixedPatchTokenizer
from blt_lite.train import build_dataloaders
from blt_lite.utils import ensure_dir, get_device, load_config, set_seed


def _token_seq_len_from_cfg(cfg: dict) -> int:
    model_cfg = cfg["model"]
    p1 = int(cfg.get("patcher",  {}).get("patch_size", 1))
    p2 = int(cfg.get("patcher2", {}).get("patch_size", 2)) if bool(cfg.get("patcher2", {}).get("enabled", True)) else 1
    default_seq_len = int(model_cfg["seq_len"]) * p1 * p2
    return int(cfg.get("patcher2_train", {}).get("seq_len_tokens", default_seq_len))


def _build_patcher1_cpu(cfg: dict, tokenizer: FixedPatchTokenizer):
    """Build and load patcher1 + token_emb on CPU."""
    model_cfg   = cfg["model"]
    patcher_cfg = cfg["patcher"]
    seq_len     = _token_seq_len_from_cfg(cfg)
    d_model     = int(model_cfg["d_model"])

    emb = torch.nn.Embedding(tokenizer.vocab_len, d_model)
    p1  = PatcherAutoencoder(
        in_dim               = d_model,
        latent_dim           = int(patcher_cfg.get("latent_dim", d_model)),
        out_dim              = d_model,
        patch_size           = int(patcher_cfg.get("patch_size", 1)),
        seq_len              = seq_len,
        encoder_layers       = int(patcher_cfg.get("encoder_layers", 2)),
        decoder_layers       = int(patcher_cfg.get("decoder_layers", 2)),
        n_heads              = int(patcher_cfg.get("n_heads", model_cfg["n_heads"])),
        dropout              = 0.0,
        pos_encoding         = str(patcher_cfg.get("pos_encoding", "learned")),
        grad_checkpointing   = False,
        flash_attention      = bool(patcher_cfg.get("flash_attention", True)),
        block_attention      = bool(patcher_cfg.get("block_attention", False)),
        block_size           = int(patcher_cfg.get("block_size", 8)),
    )

    ckpt_path = cfg.get("patcher", {}).get("pretrained_path", "")
    if not ckpt_path:
        raise ValueError("patcher.pretrained_path must be set in config")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    emb.load_state_dict(ckpt["token_emb"])
    p1.load_state_dict(ckpt["patcher"] if "patcher" in ckpt else ckpt)

    emb.eval()
    p1.eval()
    for param in list(emb.parameters()) + list(p1.parameters()):
        param.requires_grad_(False)

    print(f"Loaded patcher1 from {ckpt_path} (CPU, frozen)")
    return emb, p1


def _build_patcher2(cfg: dict, tokenizer: FixedPatchTokenizer, device: torch.device):
    model_cfg = cfg["model"]
    p2cfg     = cfg.get("patcher2", {})
    seq_len   = _token_seq_len_from_cfg(cfg)
    d_model   = int(model_cfg["d_model"])

    return PatcherAutoencoder(
        in_dim             = d_model,
        latent_dim         = int(p2cfg.get("latent_dim", d_model)),
        out_dim            = d_model,
        patch_size         = int(p2cfg.get("patch_size", 2)),
        seq_len            = seq_len,
        encoder_layers     = int(p2cfg.get("encoder_layers", 2)),
        decoder_layers     = int(p2cfg.get("decoder_layers", 2)),
        n_heads            = int(p2cfg.get("n_heads", model_cfg["n_heads"])),
        dropout            = float(p2cfg.get("dropout", model_cfg["dropout"])),
        pos_encoding       = str(p2cfg.get("pos_encoding", "learned")),
        grad_checkpointing = bool(p2cfg.get("grad_checkpointing", False)),
        flash_attention    = bool(p2cfg.get("flash_attention", True)),
        block_attention    = bool(p2cfg.get("block_attention", False)),
        block_size         = int(p2cfg.get("block_size", 8)),
    ).to(device)


def maybe_reduce_lr_by_thresholds(optimizer, val_loss: float, train_cfg: dict, reduction_state: dict) -> dict:
    thresholds = [
        ("lr_reduce_threshold",   "lr_reduce_factor"),
        ("lr_reduce_threshold_2", "lr_reduce_factor_2"),
    ]
    lr_min = float(train_cfg.get("lr_min", 1e-6))
    for idx, (thr_key, fac_key) in enumerate(thresholds):
        if reduction_state.get(idx, False):
            continue
        threshold = train_cfg.get(thr_key)
        if threshold is None or val_loss > float(threshold):
            continue
        factor = float(train_cfg.get(fac_key, train_cfg.get("lr_reduce_factor", 0.5)))
        for group in optimizer.param_groups:
            old_lr        = float(group["lr"])
            group["lr"]   = max(lr_min, old_lr * factor)
            print(f"Reduced patcher2 LR via {thr_key}: {old_lr:.8f} -> {group['lr']:.8f}")
        reduction_state[idx] = True
    return reduction_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not bool(cfg.get("patcher2", {}).get("enabled", True)):
        raise ValueError("patcher2 is disabled in config")
    set_seed(int(cfg["train"].get("seed", 42)))

    device         = get_device()
    processed_dir  = Path(cfg["data"]["processed_dir_patcher"])
    tokenizer      = FixedPatchTokenizer.load(processed_dir / "tokenizer.json")
    seq_len        = _token_seq_len_from_cfg(cfg)
    p2train        = cfg.get("patcher2_train", {})

    train_loader, val_loader = build_dataloaders(
        processed_dir / "train_tokens.npy",
        processed_dir / "val_tokens.npy",
        seq_len    = seq_len,
        batch_size = int(p2train.get("batch_size", cfg["train"]["batch_size"])),
    )

    # Patcher1 on CPU — runs inline, no cache
    emb, p1 = _build_patcher1_cpu(cfg, tokenizer)

    # Patcher2 on GPU
    patcher2  = _build_patcher2(cfg, tokenizer, device)
    optimizer = AdamW(
        patcher2.parameters(),
        lr           = float(p2train.get("lr", 3e-4)),
        weight_decay = float(p2train.get("weight_decay", 0.01)),
    )

    amp_enabled = bool(p2train.get("amp_enabled", cfg.get("train", {}).get("amp_enabled", True)))
    amp_dtype   = torch.float16 if str(p2train.get("amp_dtype", "float16")) == "float16" else torch.bfloat16
    scaler      = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and amp_enabled))

    out_dir   = ensure_dir(p2train.get("out_dir", "outputs/patcher2"))
    max_steps = int(p2train.get("max_steps", 3000))
    eval_every  = int(p2train.get("eval_every",  100))
    eval_batches = int(p2train.get("eval_batches", 50))
    save_every  = int(p2train.get("save_every",  200))

    @torch.no_grad()
    def get_p1_hidden(x_cpu: torch.Tensor) -> torch.Tensor:
        """Run patcher1 on CPU, return hidden states on GPU."""
        h  = emb(x_cpu)
        h1, _ = p1(h)
        return h1.to(device)

    def batch_loss(x_cpu: torch.Tensor) -> torch.Tensor:
        h1 = get_p1_hidden(x_cpu)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and amp_enabled), dtype=amp_dtype):
            recon, _ = patcher2(h1)
            return torch.nn.functional.mse_loss(recon, h1)

    best_val = float("inf")
    step     = 0
    lr_reduction_state: dict[int, bool] = {}
    patcher2.train()

    while step < max_steps:
        for x, _ in train_loader:
            # x stays on CPU for patcher1 forward
            optimizer.zero_grad(set_to_none=True)
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
                        losses.append(batch_loss(vx).item())
                        if len(losses) >= eval_batches:
                            break
                val_loss = float(sum(losses) / max(1, len(losses)))
                print(f"patcher2_step={step} val_recon_loss={val_loss:.6f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({"patcher2": patcher2.state_dict(), "config": cfg}, out_dir / "best.pt")
                lr_reduction_state = maybe_reduce_lr_by_thresholds(optimizer, val_loss, p2train, lr_reduction_state)
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
