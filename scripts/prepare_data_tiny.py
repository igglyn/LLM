#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import contextlib
import json
import shutil

import numpy as np
import torch

from blt_lite.model import PatcherAutoencoder
from blt_lite.tokenizer import FixedPatchTokenizer
from blt_lite.utils import ensure_dir, get_device, load_config


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
    p2 = int(cfg.get("patcher2", {}).get("patch_size", 2)) if bool(cfg.get("patcher2", {}).get("enabled", True)) else 1
    return int(model_cfg["seq_len"]) * p1 * p2


def _build_patchers(cfg: dict, tokenizer: FixedPatchTokenizer, device: torch.device):
    model_cfg = cfg["model"]
    patcher_cfg = cfg["patcher"]
    p2cfg = cfg["patcher2"]
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
        dropout=float(patcher_cfg.get("dropout", model_cfg["dropout"])),
        pos_encoding=str(patcher_cfg.get("pos_encoding", "learned")),
        grad_checkpointing=bool(patcher_cfg.get("grad_checkpointing", False)),
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
            dropout=float(p2cfg.get("dropout", model_cfg["dropout"])),
            pos_encoding=str(p2cfg.get("pos_encoding", "learned")),
            grad_checkpointing=bool(p2cfg.get("grad_checkpointing", False)),
            flash_attention=bool(p2cfg.get("flash_attention", True)),
            block_attention=bool(p2cfg.get("block_attention", False)),
            block_size=int(p2cfg.get("block_size", 8)),
        ).to(device)
    return emb, p1, p2, d_model, seq_len


def _encode_stream(
    tokens: np.ndarray,
    emb: torch.nn.Embedding,
    p1: PatcherAutoencoder,
    p2: PatcherAutoencoder | None,
    seq_len: int,
    device: torch.device,
    out_path: Path,
    d_model: int,
):
    hidden = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(len(tokens), d_model))
    chunk = max(seq_len, 1)
    with torch.no_grad():
        for start in range(0, len(tokens), chunk):
            end = min(len(tokens), start + chunk)
            ctx_start = max(0, end - seq_len)
            chunk_tokens = tokens[ctx_start:end].astype(np.int64)
            if chunk_tokens.size > seq_len:
                raise ValueError(f"Internal error: chunk length {chunk_tokens.size} exceeds seq_len {seq_len}")
            x = torch.from_numpy(chunk_tokens).unsqueeze(0).to(device)
            with _sdpa_math_context(device):
                token_hidden = emb(x)
                h1, _ = p1(token_hidden)
                h2 = h1 if p2 is None else p2(h1)[0]
            take_from = start - ctx_start
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            hidden[start:end] = h2[:, take_from:, :].squeeze(0).to(torch.float16).cpu().numpy()
    hidden.flush()


# ---------------------------------------------------------------------------
# Reconstruction accuracy testing
# ---------------------------------------------------------------------------

@torch.no_grad()
def _test_patcher_reconstruction(
    tokens: np.ndarray,
    emb: torch.nn.Embedding,
    p1: PatcherAutoencoder,
    seq_len: int,
    device: torch.device,
    n_batches: int | None = None,
) -> dict:
    """
    Feed bytes through token_emb → patcher encoder → patcher decoder,
    project back to token logits via the embedding matrix (tied weights),
    and compare argmax predictions to the original token ids.

    Returns a dict with match rate, total tokens checked, and a few
    decoded examples so you can eyeball the quality.
    """
    total_tokens = 0
    total_matches = 0
    examples = []

    # Use embedding matrix as a tied projection: (D,) → vocab logits
    # dot product with all embeddings, argmax gives nearest token
    embed_weight = emb.weight  # (vocab, D)

    chunk = max(seq_len, 1)
    batches_run = 0

    for start in range(0, len(tokens) - seq_len, chunk):
        if n_batches is not None and batches_run >= n_batches:
            break

        end = min(len(tokens), start + seq_len)
        chunk_tokens = tokens[start:end].astype(np.int64)
        x = torch.from_numpy(chunk_tokens).unsqueeze(0).to(device)  # (1, T)

        with _sdpa_math_context(device):
            token_hidden = emb(x)              # (1, T, D)
            recon, _ = p1(token_hidden)        # (1, T, D)

        # Project reconstructed hidden states back to token ids
        # logits: (1, T, vocab)
        logits = torch.matmul(recon.float(), embed_weight.T)
        predicted = logits.argmax(dim=-1).squeeze(0).cpu().numpy()   # (T,)
        original  = chunk_tokens[:predicted.shape[0]]

        matches = (predicted == original).sum()
        total_matches += int(matches)
        total_tokens  += len(original)

        # Keep a couple of decoded examples
        if len(examples) < 3:
            orig_text  = bytes([t for t in original[:64] if t < 256]).decode("utf-8", errors="replace")
            recon_text = bytes([t for t in predicted[:64] if t < 256]).decode("utf-8", errors="replace")
            examples.append({"original": orig_text, "reconstructed": recon_text})

        batches_run += 1

    match_rate = total_matches / max(1, total_tokens)
    return {
        "match_rate":    match_rate,
        "total_tokens":  total_tokens,
        "total_matches": total_matches,
        "batches_tested": batches_run,
        "examples":      examples,
    }


def _print_reconstruction_report(stats: dict):
    print("\n=== Patcher Reconstruction Report ===")
    print(f"  Batches tested : {stats['batches_tested']}")
    print(f"  Tokens checked : {stats['total_tokens']:,}")
    print(f"  Matches        : {stats['total_matches']:,}")
    print(f"  Match rate     : {stats['match_rate']*100:.2f}%")
    print()
    for i, ex in enumerate(stats["examples"]):
        print(f"  Example {i+1}:")
        print(f"    Original     : {repr(ex['original'])}")
        print(f"    Reconstructed: {repr(ex['reconstructed'])}")
    print("=====================================\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare TinyPatchLM hidden state cache with patcher reconstruction testing."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="Minimum patcher reconstruction match rate (0.0-1.0). Errors out if not met. "
             "If not set, prints stats but does not block.",
    )
    parser.add_argument(
        "--recon-batches",
        type=int,
        default=None,
        help="Number of batches to use for reconstruction testing (default: all).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    patcher2_enabled = bool(cfg.get("patcher2", {}).get("enabled", True))
    source_dir = Path(data_cfg["processed_dir_patcher2"] if patcher2_enabled else data_cfg["processed_dir_patcher"])
    out_dir = ensure_dir(data_cfg["processed_dir_tiny"])

    if not patcher2_enabled:
        print(f"patcher2 disabled; building tiny-stage caches from {source_dir}")

    for fname in ("train_tokens.npy", "val_tokens.npy", "tokenizer.json"):
        src = source_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"Missing required source artifact: {src}")
        shutil.copy2(src, out_dir / fname)

    device = get_device()
    tokenizer = FixedPatchTokenizer.load(out_dir / "tokenizer.json")
    emb, p1, p2, d_model, seq_len = _build_patchers(cfg, tokenizer, device)

    p1_path = cfg.get("patcher", {}).get("pretrained_path", "")
    p2_path = cfg.get("patcher2", {}).get("pretrained_path", "")
    if not p1_path:
        raise ValueError("patcher.pretrained_path must be set before preparing tiny hidden states")
    if patcher2_enabled and not p2_path:
        raise ValueError("patcher2.pretrained_path must be set before preparing tiny hidden states when patcher2 is enabled")

    ckpt1 = torch.load(p1_path, map_location=device)
    if "token_emb" not in ckpt1:
        raise ValueError("patcher checkpoint must include token_emb state for tiny preprocessing")
    emb.load_state_dict(ckpt1["token_emb"])
    p1.load_state_dict(ckpt1["patcher"] if isinstance(ckpt1, dict) and "patcher" in ckpt1 else ckpt1)

    if patcher2_enabled:
        ckpt2 = torch.load(p2_path, map_location=device)
        if p2 is None:
            raise ValueError("patcher2 enabled but patcher2 model was not constructed")
        p2.load_state_dict(ckpt2["patcher2"] if isinstance(ckpt2, dict) and "patcher2" in ckpt2 else ckpt2)

    emb.eval()
    p1.eval()
    if p2 is not None:
        p2.eval()

    # --- Reconstruction test ---
    train_tokens = np.load(out_dir / "train_tokens.npy")
    val_tokens   = np.load(out_dir / "val_tokens.npy")

    print("Testing patcher reconstruction accuracy...")
    recon_stats = _test_patcher_reconstruction(
        train_tokens, emb, p1, seq_len, device, n_batches=args.recon_batches
    )
    _print_reconstruction_report(recon_stats)

    if args.min_accuracy is not None:
        if recon_stats["match_rate"] < args.min_accuracy:
            raise RuntimeError(
                f"Patcher reconstruction accuracy {recon_stats['match_rate']*100:.2f}% "
                f"is below required minimum {args.min_accuracy*100:.2f}%. "
                f"Aborting — do not train trunk on these representations."
            )
        print(f"Reconstruction accuracy OK ({recon_stats['match_rate']*100:.2f}% >= {args.min_accuracy*100:.2f}%)")

    # --- Cache hidden states ---
    print("Caching train hidden states...")
    _encode_stream(train_tokens, emb, p1, p2, seq_len, device, out_dir / "train_stage2_hidden.npy", d_model)

    print("Caching val hidden states...")
    _encode_stream(val_tokens, emb, p1, p2, seq_len, device, out_dir / "val_stage2_hidden.npy", d_model)

    summary = {
        "source_dir": str(source_dir),
        "stage_dir": str(out_dir),
        "train_tokens": int(train_tokens.shape[0]),
        "val_tokens": int(val_tokens.shape[0]),
        "train_stage2_hidden": "train_stage2_hidden.npy",
        "val_stage2_hidden": "val_stage2_hidden.npy",
        "model_seq_len_large_patches": int(cfg["model"]["seq_len"]),
        "effective_token_seq_len": seq_len,
        "patcher_reconstruction": {
            "match_rate": recon_stats["match_rate"],
            "total_tokens": recon_stats["total_tokens"],
            "batches_tested": recon_stats["batches_tested"],
        },
    }

    with open(out_dir / "stage_info.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Prepared TinyPatchLM stage data in {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
