#!/usr/bin/env python
"""Simple rewrite patcher training loop.

Usage mirrors existing scripts by accepting a `--config <path>` argument.
This rewrite version intentionally keeps projection glue out of the patcher.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blt_lite.tokenizer import FixedPatchTokenizer
from rewrite.dataloader import build_first_patcher_dataloaders
from rewrite.patcher_models import EmbeddedPatcherConfig, RewritePatcherAutoencoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple rewrite patcher training loop")
    parser.add_argument("--config", required=True, help="Path to config YAML (same style as scripts/train_patcher.py)")
    args = parser.parse_args()

    from blt_lite.utils import get_device, load_config, set_seed
    cfg = load_config(args.config)
    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    device = get_device()

    model_cfg = cfg["model"]
    patcher_cfg = cfg.get("patcher", {})
    train_cfg = cfg.get("patcher_train", {})

    processed_root = Path(cfg["data"].get("processed_dir_patcher", cfg["data"]["processed_dir"])) / "rewrite_sources"
    batch_size = int(train_cfg.get("batch_size", cfg.get("train", {}).get("batch_size", 8)))
    train_loader, _, tokenizer_path = build_first_patcher_dataloaders(cfg, output_root=processed_root, batch_size=batch_size)

    tokenizer = FixedPatchTokenizer.load(tokenizer_path)
    d_model = int(model_cfg["d_model"])
    emb = torch.nn.Embedding(tokenizer.vocab_len, d_model).to(device)

    embedded_cfg = EmbeddedPatcherConfig(
        d_model=d_model,
        encoder_layers=int(patcher_cfg.get("encoder_layers", 2)),
        decoder_layers=int(patcher_cfg.get("decoder_layers", 2)),
        n_heads=int(patcher_cfg.get("n_heads", model_cfg.get("n_heads", 6))),
        dropout=float(patcher_cfg.get("dropout", model_cfg.get("dropout", 0.1))),
        pos_encoding=str(patcher_cfg.get("pos_encoding", "rope")),
        grad_checkpointing=bool(patcher_cfg.get("grad_checkpointing", False)),
        flash_attention=bool(patcher_cfg.get("flash_attention", True)),
        block_attention=bool(patcher_cfg.get("block_attention", False)),
        block_size=int(patcher_cfg.get("block_size", 8)),
    )

    patcher_type = str(patcher_cfg.get("type", "transformer")).lower()
    if patcher_type != "transformer":
        raise ValueError(
            "rewrite/train_patcher.py trains only transformer patchers. "
            "Use rewrite/train_slot_conv_patcher.py for slot_conv."
        )

    patcher = RewritePatcherAutoencoder(
        patch_size=int(patcher_cfg.get("patch_size", getattr(tokenizer, "patch_size", 1))),
        cfg=embedded_cfg,
    ).to(device)

    optimizer = AdamW(list(emb.parameters()) + list(patcher.parameters()), lr=float(train_cfg.get("lr", 3e-4)))
    max_steps = int(train_cfg.get("max_steps", 100))
    log_every = int(train_cfg.get("log_every", 10))

    patcher.train()
    emb.train()
    step = 0

    while step < max_steps:
        for batch in train_loader:
            x = batch.to(device)
            token_hidden = emb(x)
            recon_hidden, _ = patcher(token_hidden)
            loss = torch.nn.functional.mse_loss(recon_hidden, token_hidden)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            if step % log_every == 0 or step == 1:
                print(f"rewrite_patcher_step={step} loss={loss.item():.8f}")

            if step >= max_steps:
                break


if __name__ == "__main__":
    main()
