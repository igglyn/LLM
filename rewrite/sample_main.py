#!/usr/bin/env python
"""Sample text from a trained rewrite main model checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blt_lite.tokenizer import FixedPatchTokenizer
from blt_lite.utils import get_device, load_config
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


def _sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    next_logits = logits[:, -1, :]
    if temperature <= 0:
        return torch.argmax(next_logits, dim=-1)

    scaled = next_logits / temperature
    if top_k > 0:
        top_vals, top_idx = torch.topk(scaled, min(top_k, scaled.shape[-1]), dim=-1)
        probs = torch.softmax(top_vals, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return top_idx.gather(-1, sampled).squeeze(-1)

    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample text from rewrite main model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--checkpoint", default="best.pt", help="Path to model checkpoint")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    processed_root = Path(cfg["data"].get("processed_dir_patcher", cfg["data"]["processed_dir"])) / "rewrite_sources"
    tokenizer = FixedPatchTokenizer.load(processed_root / "tokenizer.json")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute() and not ckpt_path.exists():
        ckpt_path = Path(cfg.get("train", {}).get("out_dir", "outputs/rewrite_main")) / ckpt_path

    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    model_cfg_source = ckpt_cfg if isinstance(ckpt_cfg, dict) else cfg

    model = _build_model(model_cfg_source, tokenizer, device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    token_ids = tokenizer.encode(args.prompt, add_bos=True, add_eos=False)
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max(0, int(args.max_new_tokens))):
            logits, _ = model(idx)
            next_token = _sample_next_token(logits, temperature=float(args.temperature), top_k=int(args.top_k))
            idx = torch.cat([idx, next_token.unsqueeze(0)], dim=1)

    out_ids = idx[0].tolist()
    print("Generated IDs:", out_ids)
    print("Generated text:\n", tokenizer.decode(out_ids))


if __name__ == "__main__":
    main()
