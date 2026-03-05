#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import random

import numpy as np
import torch

from blt_lite.model import TinyPatchLM
from blt_lite.tokenizer import FixedPatchTokenizer
from blt_lite.utils import get_device, load_config, set_seed


def _build_model_from_cfg(cfg: dict, tokenizer: FixedPatchTokenizer, device: torch.device) -> TinyPatchLM:
    model_cfg = cfg["model"]
    patcher_cfg = cfg.get("patcher", {})
    patcher2_cfg = cfg.get("patcher2", {})
    use_patcher2 = bool(patcher2_cfg.get("enabled", True))
    model = TinyPatchLM(
        vocab_size=tokenizer.vocab_len,
        seq_len=int(model_cfg["seq_len"]),
        patch_size=int(patcher_cfg.get("patch_size", getattr(tokenizer, "patch_size", 1))),
        d_model=int(model_cfg["d_model"]),
        n_layers=int(model_cfg["n_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        dropout=float(model_cfg["dropout"]),
        patcher_latent_dim=int(patcher_cfg.get("latent_dim", model_cfg["d_model"])),
        patcher_encoder_layers=int(patcher_cfg.get("encoder_layers", 2)),
        patcher_decoder_layers=int(patcher_cfg.get("decoder_layers", 2)),
        patcher_heads=int(patcher_cfg.get("n_heads", model_cfg["n_heads"])),
        patcher_dropout=float(patcher_cfg.get("dropout", model_cfg["dropout"])),
        patcher_pos_encoding=str(patcher_cfg.get("pos_encoding", "learned")),
        use_patcher2=use_patcher2,
        patcher2_patch_size=int(patcher2_cfg.get("patch_size", 2)),
        patcher2_latent_dim=int(patcher2_cfg.get("latent_dim", model_cfg["d_model"])),
        patcher2_encoder_layers=int(patcher2_cfg.get("encoder_layers", 2)),
        patcher2_decoder_layers=int(patcher2_cfg.get("decoder_layers", 2)),
        patcher2_heads=int(patcher2_cfg.get("n_heads", model_cfg["n_heads"])),
        patcher2_dropout=float(patcher2_cfg.get("dropout", model_cfg["dropout"])),
        patcher2_pos_encoding=str(patcher2_cfg.get("pos_encoding", "learned")),
        use_amp=bool(cfg.get("train", {}).get("amp_enabled", True)),
        amp_dtype=str(cfg.get("train", {}).get("amp_dtype", "float16")),
        pos_encoding=str(model_cfg.get("pos_encoding", "learned")),
        grad_checkpointing=bool(model_cfg.get("grad_checkpointing", False)),
        flash_attention=bool(model_cfg.get("flash_attention", True)),
        patcher_grad_checkpointing=bool(patcher_cfg.get("grad_checkpointing", False)),
        patcher2_grad_checkpointing=bool(patcher2_cfg.get("grad_checkpointing", False)),
        patcher_flash_attention=bool(patcher_cfg.get("flash_attention", True)),
        patcher2_flash_attention=bool(patcher2_cfg.get("flash_attention", True)),
        patcher_block_attention=bool(patcher_cfg.get("block_attention", False)),
        patcher2_block_attention=bool(patcher2_cfg.get("block_attention", False)),
        patcher_block_size=int(patcher_cfg.get("block_size", 8)),
        patcher2_block_size=int(patcher2_cfg.get("block_size", 8)),
    ).to(device)
    return model


def _collect_online_hidden(model: TinyPatchLM, tokens: torch.Tensor) -> torch.Tensor:
    h = model.token_emb(tokens)
    if model.token_pos_emb is not None:
        pos = torch.arange(0, tokens.shape[1], device=tokens.device).unsqueeze(0)
        h = h + model.token_pos_emb(pos)
    h, _ = model.patcher(h)
    if model.patcher2 is not None:
        h, _ = model.patcher2(h)
    return h


def main():
    parser = argparse.ArgumentParser(description="Compare precomputed patcher hidden caches vs online frozen patcher outputs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--samples", type=int, default=4, help="Number of random windows to compare")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--allow-patcher2", action="store_true", help="Allow running when patcher2.enabled=true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    random.seed(args.seed)

    patcher2_enabled = bool(cfg.get("patcher2", {}).get("enabled", True))
    if patcher2_enabled and not args.allow_patcher2:
        raise ValueError("This script is intended to start with one patcher -> main model. Disable patcher2 or pass --allow-patcher2.")

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    processed_dir = Path(cfg["data"]["processed_dir_tiny"])
    tokens_path = processed_dir / f"{args.split}_tokens.npy"
    hidden_path = processed_dir / f"{args.split}_stage2_hidden.npy"
    tok_path = processed_dir / "tokenizer.json"

    for p in (tokens_path, hidden_path, tok_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    tokenizer = FixedPatchTokenizer.load(tok_path)
    model = _build_model_from_cfg(cfg, tokenizer, device)

    p1_path = cfg.get("patcher", {}).get("pretrained_path", "")
    if not p1_path:
        raise ValueError("patcher.pretrained_path is required")
    ckpt1 = torch.load(p1_path, map_location=device)
    if "token_emb" not in ckpt1:
        raise ValueError("patcher checkpoint must include token_emb state")
    model.token_emb.load_state_dict(ckpt1["token_emb"])
    model.patcher.load_state_dict(ckpt1["patcher"] if isinstance(ckpt1, dict) and "patcher" in ckpt1 else ckpt1)

    if model.patcher2 is not None:
        p2_path = cfg.get("patcher2", {}).get("pretrained_path", "")
        if not p2_path:
            raise ValueError("patcher2.pretrained_path is required when patcher2 is enabled")
        ckpt2 = torch.load(p2_path, map_location=device)
        model.patcher2.load_state_dict(ckpt2["patcher2"] if isinstance(ckpt2, dict) and "patcher2" in ckpt2 else ckpt2)

    model.eval()

    tokens = np.load(tokens_path, mmap_mode="r")
    cached_hidden = np.load(hidden_path, mmap_mode="r")

    seq_len = int(cfg["model"]["seq_len"])
    p1 = int(cfg.get("patcher", {}).get("patch_size", 1))
    p2 = int(cfg.get("patcher2", {}).get("patch_size", 2)) if patcher2_enabled else 1
    token_seq_len = seq_len * p1 * p2

    max_start = len(tokens) - token_seq_len - 1
    if max_start <= 0:
        raise ValueError(f"Not enough tokens ({len(tokens)}) for required token_seq_len={token_seq_len}")

    picks = [random.randint(0, max_start) for _ in range(args.samples)]

    print(f"split={args.split} samples={args.samples} token_seq_len={token_seq_len} patcher2_enabled={patcher2_enabled} device={device}")

    with torch.no_grad():
        for i, start in enumerate(picks):
            end = start + token_seq_len
            x_np = np.asarray(tokens[start:end], dtype=np.int64)
            h_cache_np = np.asarray(cached_hidden[start:end], dtype=np.float32)
            y_np = np.asarray(tokens[start + 1 : end + 1], dtype=np.int64)

            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            h_cache = torch.from_numpy(h_cache_np).unsqueeze(0).to(device)
            y = torch.from_numpy(y_np).unsqueeze(0).to(device)

            h_online = _collect_online_hidden(model, x)
            h_diff = (h_online - h_cache).float()
            h_mse = torch.mean(h_diff * h_diff).item()
            h_max_abs = torch.max(torch.abs(h_diff)).item()

            logits_online, loss_online = model(x, y)
            logits_cache, loss_cache = model.forward_from_hidden(h_cache, y)
            l_diff = (logits_online - logits_cache).float()
            l_mse = torch.mean(l_diff * l_diff).item()
            l_max_abs = torch.max(torch.abs(l_diff)).item()
            loss_delta = abs(float(loss_online.item()) - float(loss_cache.item()))

            print(
                f"sample={i} start={start} "
                f"hidden_mse={h_mse:.6e} hidden_max_abs={h_max_abs:.6e} "
                f"logits_mse={l_mse:.6e} logits_max_abs={l_max_abs:.6e} "
                f"loss_online={loss_online.item():.6f} loss_cache={loss_cache.item():.6f} loss_abs_delta={loss_delta:.6e}"
            )


if __name__ == "__main__":
    main()
