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
    default_seq_len = int(model_cfg["seq_len"]) * p1
    return int(cfg.get("patcher_train", {}).get("seq_len_tokens", default_seq_len))


def _build_stage1(cfg: dict, tokenizer: FixedPatchTokenizer, device: torch.device):
    model_cfg = cfg["model"]
    patcher_cfg = cfg["patcher"]
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
    return emb, p1, d_model, seq_len


def _encode_stream(tokens: np.ndarray, emb: torch.nn.Embedding, patcher: PatcherAutoencoder, seq_len: int, device: torch.device, out_path: Path, d_model: int, debug_token_match: bool = False) -> tuple[int, int]:
    hidden = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(len(tokens), d_model))
    chunk = max(seq_len, 1)
    matched_total = 0
    unmatched_total = 0
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
                recon_hidden, _ = patcher(token_hidden)
            take_from = start - ctx_start
            recon_slice = recon_hidden[:, take_from:, :]
            if debug_token_match:
                logits = torch.matmul(recon_slice, emb.weight.transpose(0, 1))
                pred = logits.argmax(dim=-1)
                target = x[:, take_from:]
                matched = int((pred == target).sum().item())
                total = int(pred.numel())
                matched_total += matched
                unmatched_total += (total - matched)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            hidden[start:end] = recon_slice.squeeze(0).to(torch.float16).cpu().numpy()
    hidden.flush()
    return matched_total, unmatched_total


def main():
    parser = argparse.ArgumentParser(description="Prepare explicit stage-2 (patcher2) training artifacts with cached stage-1 hidden states.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug-token-match", action="store_true", help="Print matched/unmatched token ratio while generating stage1 hidden cache")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    source_dir = Path(data_cfg["processed_dir_patcher"])
    out_dir = ensure_dir(data_cfg["processed_dir_patcher2"])

    for fname in ("train_tokens.npy", "val_tokens.npy", "tokenizer.json"):
        src = source_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"Missing required source artifact: {src}")
        shutil.copy2(src, out_dir / fname)

    device = get_device()
    tokenizer = FixedPatchTokenizer.load(out_dir / "tokenizer.json")
    emb, patcher1, d_model, seq_len = _build_stage1(cfg, tokenizer, device)

    ckpt_path = cfg.get("patcher", {}).get("pretrained_path", "")
    if not ckpt_path:
        raise ValueError("patcher.pretrained_path must be set before preparing stage2 hidden states")
    ckpt = torch.load(ckpt_path, map_location=device)
    if "token_emb" not in ckpt:
        raise ValueError("patcher checkpoint must include token_emb state for stage2 preprocessing")
    emb.load_state_dict(ckpt["token_emb"])
    patcher1.load_state_dict(ckpt["patcher"] if isinstance(ckpt, dict) and "patcher" in ckpt else ckpt)
    emb.eval(); patcher1.eval()

    train_tokens = np.load(out_dir / "train_tokens.npy")
    val_tokens = np.load(out_dir / "val_tokens.npy")
    train_match, train_unmatch = _encode_stream(train_tokens, emb, patcher1, seq_len, device, out_dir / "train_stage1_hidden.npy", d_model, debug_token_match=args.debug_token_match)
    val_match, val_unmatch = _encode_stream(val_tokens, emb, patcher1, seq_len, device, out_dir / "val_stage1_hidden.npy", d_model, debug_token_match=args.debug_token_match)

    summary = {
        "source_dir": str(source_dir),
        "stage_dir": str(out_dir),
        "train_tokens": int(train_tokens.shape[0]),
        "val_tokens": int(val_tokens.shape[0]),
        "train_stage1_hidden": "train_stage1_hidden.npy",
        "val_stage1_hidden": "val_stage1_hidden.npy",
        "seq_len_tokens": int(cfg.get("patcher2_train", {}).get("seq_len_tokens", 0)),
    }
    if args.debug_token_match:
        summary["train_token_match"] = int(train_match)
        summary["train_token_unmatch"] = int(train_unmatch)
        summary["val_token_match"] = int(val_match)
        summary["val_token_unmatch"] = int(val_unmatch)
        summary["train_match_unmatch_ratio"] = float(train_match) / float(max(train_unmatch, 1))
        summary["val_match_unmatch_ratio"] = float(val_match) / float(max(val_unmatch, 1))

    with open(out_dir / "stage_info.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Prepared explicit patcher2 stage data in {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
