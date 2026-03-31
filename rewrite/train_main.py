#!/usr/bin/env python
"""Train rewrite main model with both patchers loaded/frozen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blt_lite.tokenizer import FixedPatchTokenizer
from blt_lite.utils import ensure_dir, get_device, load_config, set_seed
from rewrite.dataloader import build_rewrite_dataloaders
from rewrite.main_model import RewriteMainModel, RewriteMainModelConfig
from rewrite.patcher_models import EmbeddedPatcherConfig


def _build_model(cfg: dict, tokenizer: FixedPatchTokenizer, device: torch.device) -> RewriteMainModel:
    model_cfg = cfg.get("model", {})
    p1_cfg = cfg.get("patcher", {})
    p2_cfg = cfg.get("patcher2", {})

    main_cfg = RewriteMainModelConfig(
        vocab_size=tokenizer.vocab_len,
        d_model=int(model_cfg.get("d_model", 384)),
        patch_size=int(p1_cfg.get("patch_size", getattr(tokenizer, "patch_size", 1))),
        n_layers=int(model_cfg.get("n_layers", 4)),
        n_heads=int(model_cfg.get("n_heads", 6)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_patcher2=bool(p2_cfg.get("enabled", True)),
        patcher2_patch_size=int(p2_cfg.get("patch_size", 2)),
    )

    patcher1_cfg = EmbeddedPatcherConfig(
        d_model=int(p1_cfg.get("latent_dim", model_cfg.get("d_model", 384))),
        encoder_layers=int(p1_cfg.get("encoder_layers", 2)),
        decoder_layers=int(p1_cfg.get("decoder_layers", 2)),
        n_heads=int(p1_cfg.get("n_heads", model_cfg.get("n_heads", 6))),
        dropout=float(p1_cfg.get("dropout", model_cfg.get("dropout", 0.1))),
        pos_encoding=str(p1_cfg.get("pos_encoding", "rope")),
        grad_checkpointing=bool(p1_cfg.get("grad_checkpointing", False)),
        flash_attention=bool(p1_cfg.get("flash_attention", True)),
        block_attention=bool(p1_cfg.get("block_attention", False)),
        block_size=int(p1_cfg.get("block_size", 8)),
    )

    patcher2_cfg = None
    if bool(p2_cfg.get("enabled", True)):
        patcher2_cfg = EmbeddedPatcherConfig(
            d_model=int(p2_cfg.get("latent_dim", model_cfg.get("d_model", 384))),
            encoder_layers=int(p2_cfg.get("encoder_layers", 2)),
            decoder_layers=int(p2_cfg.get("decoder_layers", 2)),
            n_heads=int(p2_cfg.get("n_heads", model_cfg.get("n_heads", 6))),
            dropout=float(p2_cfg.get("dropout", model_cfg.get("dropout", 0.1))),
            pos_encoding=str(p2_cfg.get("pos_encoding", "rope")),
            grad_checkpointing=bool(p2_cfg.get("grad_checkpointing", False)),
            flash_attention=bool(p2_cfg.get("flash_attention", True)),
            block_attention=bool(p2_cfg.get("block_attention", False)),
            block_size=int(p2_cfg.get("block_size", 8)),
        )

    return RewriteMainModel(cfg=main_cfg, patcher_cfg=patcher1_cfg, patcher2_cfg=patcher2_cfg).to(device)


def _load_and_freeze_patchers(model: RewriteMainModel, cfg: dict, device: torch.device) -> None:
    p1_path = str(cfg.get("patcher", {}).get("pretrained_path", "")).strip()
    if not p1_path:
        raise ValueError("patcher.pretrained_path must be set for rewrite main-model training")
    model.load_patcher_checkpoint(p1_path, map_location=device)
    for p in model.patcher.parameters():
        p.requires_grad = False
    print(f"Loaded and froze patcher1: {p1_path}")

    if model.patcher2 is None:
        return

    p2_path = str(cfg.get("patcher2", {}).get("pretrained_path", "")).strip()
    if not p2_path:
        raise ValueError("patcher2.pretrained_path must be set when patcher2 is enabled")
    model.load_patcher2_checkpoint(p2_path, map_location=device)
    for p in model.patcher2.parameters():
        p.requires_grad = False
    print(f"Loaded and froze patcher2: {p2_path}")


def _evaluate(model: RewriteMainModel, val_loader, device: torch.device, max_batches: int | None = None) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rewrite main model using both patchers")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    device = get_device()

    train_cfg = cfg.get("train", {})
    processed_root = Path(cfg["data"].get("processed_dir_patcher", cfg["data"]["processed_dir"])) / "rewrite_sources"
    batch_size = int(train_cfg.get("batch_size", 8))

    loaders, tokenizer_path = build_rewrite_dataloaders(cfg, output_root=processed_root, batch_size=batch_size)
    tokenizer = FixedPatchTokenizer.load(tokenizer_path)

    model = _build_model(cfg, tokenizer, device)
    _load_and_freeze_patchers(model, cfg, device)

    train_loader = loaders["main_train"]
    val_loader = loaders["main_val"]

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        params,
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    max_steps = int(train_cfg.get("max_steps", 1000))
    eval_every = int(train_cfg.get("eval_every", 200))
    log_every = int(train_cfg.get("log_every", 20))
    out_dir = ensure_dir(str(train_cfg.get("out_dir", "outputs/rewrite_main")))

    step = 0
    best_val = float("inf")
    while step < max_steps:
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(train_cfg.get("grad_clip", 1.0)))
            optimizer.step()

            step += 1
            if step % log_every == 0 or step == 1:
                print(f"rewrite_main_step={step} loss={loss.item():.6f}")

            if step % eval_every == 0 or step == max_steps:
                val_loss = _evaluate(model, val_loader, device, max_batches=int(train_cfg.get("val_max_batches", 50)))
                print(f"rewrite_main_step={step} val_loss={val_loss:.6f}")
                payload = {"model": model.state_dict(), "config": cfg, "step": step, "val_loss": val_loss}
                torch.save(payload, Path(out_dir) / "last.pt")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(payload, Path(out_dir) / "best.pt")

            if step >= max_steps:
                break


if __name__ == "__main__":
    main()
