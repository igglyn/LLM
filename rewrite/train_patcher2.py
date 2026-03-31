#!/usr/bin/env python
"""Rewrite second patcher training loop.

This stage trains patcher2 over first-layer patch states (not token states).
The dataloader emits explicit first-layer token patches, grouped into windows
that will be combined into meta-patches.
"""

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
from rewrite.dataloader import build_second_patcher_dataloaders
from rewrite.patcher_models import EmbeddedPatcherConfig, RewritePatcherAutoencoder


def _embedded_cfg(cfg: dict, section: str) -> EmbeddedPatcherConfig:
    model_cfg = cfg.get("model", {})
    patcher_cfg = cfg.get(section, {})
    return EmbeddedPatcherConfig(
        d_model=int(model_cfg["d_model"]),
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


def _load_stage1_frozen(cfg: dict, tokenizer: FixedPatchTokenizer, device: torch.device):
    patcher_cfg = cfg.get("patcher", {})
    ckpt_path = patcher_cfg.get("pretrained_path", "")
    if not ckpt_path:
        raise ValueError("patcher.pretrained_path must be set for rewrite patcher2 training")

    d_model = int(cfg["model"]["d_model"])
    emb = torch.nn.Embedding(tokenizer.vocab_len, d_model).to(device)
    p1 = RewritePatcherAutoencoder(
        patch_size=int(patcher_cfg.get("patch_size", getattr(tokenizer, "patch_size", 1))),
        cfg=_embedded_cfg(cfg, "patcher"),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    emb.load_state_dict(ckpt["token_emb"])
    p1.load_state_dict(ckpt["patcher"])

    emb.eval()
    p1.eval()
    for param in list(emb.parameters()) + list(p1.parameters()):
        param.requires_grad_(False)

    print(f"Loaded frozen rewrite patcher1 from {ckpt_path}")
    return emb, p1


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite second patcher training loop")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    from blt_lite.utils import ensure_dir, get_device, load_config, set_seed

    cfg = load_config(args.config)
    if not bool(cfg.get("patcher2", {}).get("enabled", True)):
        raise ValueError("patcher2 is disabled in config")

    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    device = get_device()

    train_cfg = cfg.get("patcher2_train", {})
    processed_root = Path(cfg["data"].get("processed_dir_patcher", cfg["data"]["processed_dir"])) / "rewrite_sources"
    batch_size = int(train_cfg.get("batch_size", cfg.get("train", {}).get("batch_size", 8)))
    train_loader, val_loader, tokenizer_path = build_second_patcher_dataloaders(
        cfg, output_root=processed_root, batch_size=batch_size
    )

    tokenizer = FixedPatchTokenizer.load(tokenizer_path)
    emb, patcher1 = _load_stage1_frozen(cfg, tokenizer, device)

    patcher2 = RewritePatcherAutoencoder(
        patch_size=int(cfg.get("patcher2", {}).get("patch_size", 2)),
        cfg=_embedded_cfg(cfg, "patcher2"),
    ).to(device)

    optimizer = AdamW(
        patcher2.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    max_steps = int(train_cfg.get("max_steps", 3000))
    eval_every = int(train_cfg.get("eval_every", 100))
    eval_batches = int(train_cfg.get("eval_batches", 50))
    log_every = int(train_cfg.get("log_every", 20))
    save_every = int(train_cfg.get("save_every", 200))
    out_dir = ensure_dir(train_cfg.get("out_dir", "outputs/patcher2"))

    @torch.no_grad()
    def stage1_patches(token_patches: torch.Tensor) -> torch.Tensor:
        bsz, patch_count, patch_size = token_patches.shape
        flat_tokens = token_patches.reshape(bsz, patch_count * patch_size)
        hidden = emb(flat_tokens)
        _, patch_states = patcher1(hidden)
        if patch_states.size(1) != patch_count:
            raise ValueError(
                "Unexpected patch count from stage1 patcher. "
                f"expected={patch_count} got={patch_states.size(1)}"
            )
        return patch_states

    def batch_loss(token_patches: torch.Tensor) -> torch.Tensor:
        p1_states = stage1_patches(token_patches)
        recon_states, _ = patcher2(p1_states)
        return F.mse_loss(recon_states, p1_states)

    step = 0
    best_val = float("inf")
    patcher2.train()

    while step < max_steps:
        for batch in train_loader:
            token_patches = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(token_patches)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(patcher2.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            optimizer.step()

            if step % log_every == 0 or step == 1:
                print(f"rewrite_patcher2_step={step} recon_loss={loss.item():.8f}")

            if step % eval_every == 0 and step > 0:
                patcher2.eval()
                losses = []
                with torch.no_grad():
                    for vbatch in val_loader:
                        losses.append(batch_loss(vbatch.to(device)).item())
                        if len(losses) >= eval_batches:
                            break
                val_loss = float(sum(losses) / max(1, len(losses)))
                print(f"rewrite_patcher2_step={step} val_recon_loss={val_loss:.8f}")
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


if __name__ == "__main__":
    main()
