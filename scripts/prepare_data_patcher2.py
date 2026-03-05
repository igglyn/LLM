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


def _decode_to_tokens(hidden: torch.Tensor, emb_weight: torch.Tensor) -> torch.Tensor:
    logits = torch.matmul(hidden, emb_weight.transpose(0, 1))
    return torch.argmax(logits, dim=-1)


def _encode_stream(
    tokens: np.ndarray,
    emb: torch.nn.Embedding,
    patcher: PatcherAutoencoder,
    seq_len: int,
    device: torch.device,
    out_path: Path,
    d_model: int,
    verify_token_identity: bool,
):
    hidden = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(len(tokens), d_model))
    chunk = max(seq_len, 1)
    mismatch_count = 0
    checked_tokens = 0
    emb_weight = emb.weight
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
            out_hidden = recon_hidden[:, take_from:, :]
            if verify_token_identity:
                decoded = _decode_to_tokens(out_hidden, emb_weight)
                target = x[:, take_from:]
                mismatch_count += int((decoded != target).sum().item())
                checked_tokens += int(target.numel())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            hidden[start:end] = out_hidden.squeeze(0).to(torch.float16).cpu().numpy()
    hidden.flush()
    return {"checked_tokens": checked_tokens, "mismatch_tokens": mismatch_count}


def main():
    parser = argparse.ArgumentParser(description="Prepare explicit stage-2 (patcher2) training artifacts with cached stage-1 hidden states.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--verify-token-identity", action="store_true", help="Decode reconstructed hidden states back to tokens and assert exact token identity.")
    parser.add_argument(
        "--verify-token-identity-fail-rate",
        type=float,
        default=1e-8,
        help="Maximum tolerated mismatch rate before failing verification (default: 1e-8, i.e. ~1 in 100M).",
    )
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
    emb.eval()
    patcher1.eval()

    train_tokens = np.load(out_dir / "train_tokens.npy")
    val_tokens = np.load(out_dir / "val_tokens.npy")
    train_verify = _encode_stream(
        train_tokens,
        emb,
        patcher1,
        seq_len,
        device,
        out_dir / "train_stage1_hidden.npy",
        d_model,
        verify_token_identity=args.verify_token_identity,
    )
    val_verify = _encode_stream(
        val_tokens,
        emb,
        patcher1,
        seq_len,
        device,
        out_dir / "val_stage1_hidden.npy",
        d_model,
        verify_token_identity=args.verify_token_identity,
    )

    if args.verify_token_identity:
        total_checked = int(train_verify["checked_tokens"] + val_verify["checked_tokens"])
        total_mismatch = int(train_verify["mismatch_tokens"] + val_verify["mismatch_tokens"])
        mismatch_rate = (float(total_mismatch) / float(total_checked)) if total_checked > 0 else 0.0
        fail_rate = float(args.verify_token_identity_fail_rate)
        if mismatch_rate > fail_rate:
            raise RuntimeError(
                "Token identity check failed for stage1 patcher reconstruction: "
                f"{total_mismatch}/{total_checked} mismatched (rate={mismatch_rate:.12g}, threshold={fail_rate:.12g})"
            )
    else:
        mismatch_rate = None
        fail_rate = float(args.verify_token_identity_fail_rate)

    summary = {
        "source_dir": str(source_dir),
        "stage_dir": str(out_dir),
        "train_tokens": int(train_tokens.shape[0]),
        "val_tokens": int(val_tokens.shape[0]),
        "train_stage1_hidden": "train_stage1_hidden.npy",
        "val_stage1_hidden": "val_stage1_hidden.npy",
        "seq_len_tokens": int(cfg.get("patcher2_train", {}).get("seq_len_tokens", 0)),
        "token_identity_check": {
            "enabled": bool(args.verify_token_identity),
            "fail_rate_threshold": fail_rate,
            "total_mismatch_rate": mismatch_rate,
            "train": train_verify,
            "val": val_verify,
        },
    }
    with open(out_dir / "stage_info.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Prepared explicit patcher2 stage data in {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
