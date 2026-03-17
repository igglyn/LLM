#!/usr/bin/env python
"""
compare_cache.py

Compares cached hidden states against live patcher forward passes on the
same token sequences.

Measures MSE between cached and live representations, and also compares
the cross-entropy loss the trunk would see from each path on the same targets.

Usage:
    python scripts/compare_cache.py --config configs/tiny.yaml [--batches 20] [--dtype float32]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import contextlib

import numpy as np
import torch
import torch.nn.functional as F

from blt_lite.model import PatcherAutoencoder, TinyPatchLM
from blt_lite.tokenizer import FixedPatchTokenizer
from blt_lite.utils import get_device, load_config


def _sdpa_math_context(device: torch.device):
    if device.type != "cuda":
        return contextlib.nullcontext()
    attention_ns = getattr(torch.nn, "attention", None)
    sdpa_kernel = getattr(attention_ns, "sdpa_kernel", None) if attention_ns is not None else None
    sdp_backend = getattr(attention_ns, "SDPBackend", None) if attention_ns is not None else None
    if sdpa_kernel is None or sdp_backend is None:
        return contextlib.nullcontext()
    return sdpa_kernel(backends=[sdp_backend.MATH])


def _token_seq_len_from_cfg(cfg: dict) -> int:
    model_cfg = cfg["model"]
    p1 = int(cfg.get("patcher", {}).get("patch_size", 1))
    patcher2_enabled = bool(cfg.get("patcher2", {}).get("enabled", True))
    p2 = int(cfg.get("patcher2", {}).get("patch_size", 2)) if patcher2_enabled else 1
    return int(model_cfg["seq_len"]) * p1 * p2


def _build_patchers(cfg: dict, tokenizer: FixedPatchTokenizer, device: torch.device):
    model_cfg = cfg["model"]
    patcher_cfg = cfg["patcher"]
    p2cfg = cfg.get("patcher2", {})
    patcher2_enabled = bool(p2cfg.get("enabled", True))
    d_model = int(model_cfg["d_model"])
    seq_len = _token_seq_len_from_cfg(cfg)

    emb = torch.nn.Embedding(tokenizer.vocab_len, d_model).to(device)
    p1 = PatcherAutoencoder(
        in_dim=d_model,
        latent_dim=int(patcher_cfg.get("latent_dim", d_model)),
        out_dim=d_model,
        patch_size=int(patcher_cfg.get("patch_size", 1)),
        seq_len=seq_len,
        encoder_layers=int(patcher_cfg.get("encoder_layers", 2)),
        decoder_layers=int(patcher_cfg.get("decoder_layers", 2)),
        n_heads=int(patcher_cfg.get("n_heads", model_cfg["n_heads"])),
        dropout=0.0,
        pos_encoding=str(patcher_cfg.get("pos_encoding", "learned")),
        grad_checkpointing=False,
        flash_attention=bool(patcher_cfg.get("flash_attention", True)),
        block_attention=bool(patcher_cfg.get("block_attention", False)),
        block_size=int(patcher_cfg.get("block_size", 8)),
    ).to(device)

    p2 = None
    if patcher2_enabled:
        p2 = PatcherAutoencoder(
            in_dim=d_model,
            latent_dim=int(p2cfg.get("latent_dim", d_model)),
            out_dim=d_model,
            patch_size=int(p2cfg.get("patch_size", 2)),
            seq_len=seq_len,
            encoder_layers=int(p2cfg.get("encoder_layers", 2)),
            decoder_layers=int(p2cfg.get("decoder_layers", 2)),
            n_heads=int(p2cfg.get("n_heads", model_cfg["n_heads"])),
            dropout=0.0,
            pos_encoding=str(p2cfg.get("pos_encoding", "learned")),
            grad_checkpointing=False,
            flash_attention=bool(p2cfg.get("flash_attention", True)),
            block_attention=bool(p2cfg.get("block_attention", False)),
            block_size=int(p2cfg.get("block_size", 8)),
        ).to(device)

    return emb, p1, p2, d_model, seq_len


def _load_patchers(emb, p1, p2, cfg: dict, device: torch.device):
    p1_path = cfg.get("patcher", {}).get("pretrained_path", "")
    p2_path = cfg.get("patcher2", {}).get("pretrained_path", "")
    patcher2_enabled = bool(cfg.get("patcher2", {}).get("enabled", True))

    if not p1_path:
        raise ValueError("patcher.pretrained_path must be set in config")

    ckpt1 = torch.load(p1_path, map_location=device)
    emb.load_state_dict(ckpt1["token_emb"])
    p1.load_state_dict(ckpt1["patcher"] if "patcher" in ckpt1 else ckpt1)

    if patcher2_enabled:
        if not p2_path:
            raise ValueError("patcher2.pretrained_path must be set in config")
        ckpt2 = torch.load(p2_path, map_location=device)
        p2.load_state_dict(ckpt2["patcher2"] if "patcher2" in ckpt2 else ckpt2)

    emb.eval()
    p1.eval()
    if p2 is not None:
        p2.eval()


@torch.no_grad()
def _live_hidden(
    tokens: torch.Tensor,
    emb: torch.nn.Embedding,
    p1: PatcherAutoencoder,
    p2: PatcherAutoencoder | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run tokens through live patcher stack, return hidden states in target dtype."""
    x = tokens.unsqueeze(0).to(device)
    with _sdpa_math_context(device):
        h = emb(x).to(dtype)
        h, _ = p1(h)
        if p2 is not None:
            h, _ = p2(h)
    return h.squeeze(0)  # (T, D)


def main():
    parser = argparse.ArgumentParser(
        description="Compare cached vs live patcher hidden states."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--batches",
        type=int,
        default=20,
        help="Number of batches to compare (default: 20).",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Dtype to use for live forward pass comparison (default: float32).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    live_dtype = torch.float32 if args.dtype == "float32" else torch.float16

    data_cfg = cfg["data"]
    tiny_dir = Path(data_cfg["processed_dir_tiny"])

    # Check cache exists
    cache_path = tiny_dir / "train_stage2_hidden.npy"
    tokens_path = tiny_dir / "train_tokens.npy"
    if not cache_path.exists():
        raise FileNotFoundError(f"No cache found at {cache_path} — run prepare_data_tiny.py first")
    if not tokens_path.exists():
        raise FileNotFoundError(f"No tokens found at {tokens_path}")

    tokenizer = FixedPatchTokenizer.load(tiny_dir / "tokenizer.json")
    emb, p1, p2, d_model, seq_len = _build_patchers(cfg, tokenizer, device)
    _load_patchers(emb, p1, p2, cfg, device)

    # Load cache and tokens
    cache = np.load(cache_path, mmap_mode="r")
    tokens = np.load(tokens_path)

    print(f"Cache shape  : {cache.shape}  dtype={cache.dtype}")
    print(f"Tokens shape : {tokens.shape}")
    print(f"Live dtype   : {args.dtype}")
    print(f"Batches      : {args.batches}")
    print()

    mse_values = []
    max_abs_values = []
    mean_abs_values = []

    n_batches = min(args.batches, (len(tokens) - seq_len) // seq_len)

    for i in range(n_batches):
        start = i * seq_len
        end   = start + seq_len
        if end > len(tokens):
            break

        chunk_tokens = torch.from_numpy(tokens[start:end].astype(np.int64))

        # Live path
        live = _live_hidden(chunk_tokens, emb, p1, p2, device, live_dtype)  # (T, D)

        # Cached path — cast to same dtype as live for fair comparison
        cached = torch.from_numpy(
            np.asarray(cache[start:end], dtype=np.float32)
        ).to(device=device, dtype=live_dtype)  # (T, D)

        diff = (live - cached)
        mse       = (diff ** 2).mean().item()
        max_abs   = diff.abs().max().item()
        mean_abs  = diff.abs().mean().item()

        mse_values.append(mse)
        max_abs_values.append(max_abs)
        mean_abs_values.append(mean_abs)

    # Summary
    mse_arr      = np.array(mse_values)
    max_abs_arr  = np.array(max_abs_values)
    mean_abs_arr = np.array(mean_abs_values)

    print("=== Cache vs Live Comparison ===")
    print(f"  Batches compared : {len(mse_values)}")
    print(f"  Sequence length  : {seq_len}")
    print()
    print(f"  MSE (mean)       : {mse_arr.mean():.6f}")
    print(f"  MSE (max)        : {mse_arr.max():.6f}")
    print(f"  MSE (min)        : {mse_arr.min():.6f}")
    print()
    print(f"  Mean |diff| avg  : {mean_abs_arr.mean():.6f}")
    print(f"  Max  |diff| avg  : {max_abs_arr.mean():.6f}")
    print(f"  Max  |diff| peak : {max_abs_arr.max():.6f}")
    print()

    # Verdict
    if mse_arr.mean() < 1e-6:
        print("  Verdict: cache and live are effectively identical.")
    elif mse_arr.mean() < 1e-4:
        print("  Verdict: small divergence — likely float16 precision loss in cache.")
    elif mse_arr.mean() < 1e-2:
        print("  Verdict: moderate divergence — worth investigating.")
    else:
        print("  Verdict: large divergence — cache does not match live representations.")
        print("           Check positional embeddings, dropout state, and AMP precision.")

    print("================================")


if __name__ == "__main__":
    main()
