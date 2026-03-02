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


def build_patcher_and_embed(cfg: dict, tokenizer: FixedPatchTokenizer, device: torch.device):
    model_cfg = cfg["model"]
    patcher_cfg = cfg["patcher"]
    seq_len = int(cfg["data"]["seq_len"])
    patch_size = int(getattr(tokenizer, "patch_size", cfg.get("tokenizer", {}).get("patch_size", 1)))
    d_model = int(model_cfg["d_model"])

    emb = torch.nn.Embedding(tokenizer.vocab_len, d_model).to(device)
    patcher = PatcherAutoencoder(
        in_dim=d_model,
        latent_dim=int(patcher_cfg.get("latent_dim", d_model)),
        out_dim=d_model,
        patch_size=patch_size,
        seq_len=seq_len,
        encoder_layers=int(patcher_cfg.get("encoder_layers", 2)),
        decoder_layers=int(patcher_cfg.get("decoder_layers", 2)),
        n_heads=int(patcher_cfg.get("n_heads", model_cfg["n_heads"])),
        dropout=float(patcher_cfg.get("dropout", model_cfg["dropout"])),
    ).to(device)
    return emb, patcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["train"].get("seed", 42)))

    device = get_device()
    processed_dir = Path(cfg["data"]["processed_dir"])
    tokenizer = FixedPatchTokenizer.load(processed_dir / "tokenizer.json")

    patcher_train = cfg.get("patcher_train", {})
    train_loader, val_loader = build_dataloaders(
        processed_dir / "train_tokens.npy",
        processed_dir / "val_tokens.npy",
        seq_len=int(cfg["data"]["seq_len"]),
        batch_size=int(patcher_train.get("batch_size", cfg["train"]["batch_size"])),
    )

    token_emb, patcher = build_patcher_and_embed(cfg, tokenizer, device)
    params = list(token_emb.parameters()) + list(patcher.parameters())
    optimizer = AdamW(
        params,
        lr=float(patcher_train.get("lr", 3e-4)),
        weight_decay=float(patcher_train.get("weight_decay", 0.01)),
    )

    out_dir = ensure_dir(patcher_train.get("out_dir", "outputs/patcher"))
    max_steps = int(patcher_train.get("max_steps", 3000))
    eval_every = int(patcher_train.get("eval_every", 100))
    save_every = int(patcher_train.get("save_every", 200))

    def batch_loss(x: torch.Tensor) -> torch.Tensor:
        token_hidden = token_emb(x)
        recon_hidden, _ = patcher(token_hidden)
        return torch.nn.functional.mse_loss(recon_hidden, token_hidden)

    best_val = float("inf")
    step = 0
    patcher.train()
    token_emb.train()
    while step < max_steps:
        for x, _ in train_loader:
            x = x.to(device)
            loss = batch_loss(x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(patcher_train.get("grad_clip", 1.0)))
            optimizer.step()

            if step % 20 == 0:
                print(f"patcher_step={step} recon_loss={loss.item():.6f}")

            if step % eval_every == 0 and step > 0:
                patcher.eval(); token_emb.eval()
                losses = []
                with torch.no_grad():
                    for vx, _ in val_loader:
                        vx = vx.to(device)
                        losses.append(batch_loss(vx).item())
                        if len(losses) >= int(patcher_train.get("eval_batches", 50)):
                            break
                val_loss = float(sum(losses) / max(1, len(losses)))
                print(f"patcher_step={step} val_recon_loss={val_loss:.6f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(
                        {"patcher": patcher.state_dict(), "token_emb": token_emb.state_dict(), "config": cfg},
                        out_dir / "best.pt",
                    )
                patcher.train(); token_emb.train()

            if step % save_every == 0 and step > 0:
                torch.save(
                    {"patcher": patcher.state_dict(), "token_emb": token_emb.state_dict(), "config": cfg},
                    out_dir / f"step_{step}.pt",
                )

            step += 1
            if step >= max_steps:
                break

    torch.save(
        {"patcher": patcher.state_dict(), "token_emb": token_emb.state_dict(), "config": cfg},
        out_dir / "last.pt",
    )
    print(f"Patcher pretraining complete. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
