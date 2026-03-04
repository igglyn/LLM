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

from blt_lite.model import TinyPatchLM
from blt_lite.tokenizer import FixedPatchTokenizer
from blt_lite.utils import get_device, load_config


def _build_model_from_cfg(cfg: dict, tokenizer: FixedPatchTokenizer, device: torch.device) -> TinyPatchLM:
    model_cfg = cfg["model"]
    patcher_cfg = cfg.get("patcher", {})
    patcher2_cfg = cfg.get("patcher2", {})
    use_patcher2 = bool(patcher2_cfg.get("enabled", True))
    return TinyPatchLM(
        vocab_size=tokenizer.vocab_len,
        seq_len=int(model_cfg["seq_len"]),
        patch_size=int(cfg.get("patcher", {}).get("patch_size", getattr(tokenizer, "patch_size", 1))),
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
        patcher2_patch_size=int(cfg.get("patcher2", {}).get("patch_size", 2)),
        patcher2_latent_dim=int(cfg.get("patcher2", {}).get("latent_dim", model_cfg["d_model"])),
        patcher2_encoder_layers=int(cfg.get("patcher2", {}).get("encoder_layers", 2)),
        patcher2_decoder_layers=int(cfg.get("patcher2", {}).get("decoder_layers", 2)),
        patcher2_heads=int(cfg.get("patcher2", {}).get("n_heads", model_cfg["n_heads"])),
        patcher2_dropout=float(cfg.get("patcher2", {}).get("dropout", model_cfg["dropout"])),
        patcher2_pos_encoding=str(cfg.get("patcher2", {}).get("pos_encoding", "learned")),
        use_amp=bool(cfg.get("train", {}).get("amp_enabled", True)),
        amp_dtype=str(cfg.get("train", {}).get("amp_dtype", "float16")),
        pos_encoding=str(model_cfg.get("pos_encoding", "learned")),
        grad_checkpointing=bool(model_cfg.get("grad_checkpointing", False)),
        flash_attention=bool(model_cfg.get("flash_attention", True)),
        patcher_grad_checkpointing=bool(patcher_cfg.get("grad_checkpointing", False)),
        patcher2_grad_checkpointing=bool(cfg.get("patcher2", {}).get("grad_checkpointing", False)),
        patcher_flash_attention=bool(patcher_cfg.get("flash_attention", True)),
        patcher2_flash_attention=bool(cfg.get("patcher2", {}).get("flash_attention", True)),
        patcher_block_attention=bool(patcher_cfg.get("block_attention", False)),
        patcher_block_size=int(patcher_cfg.get("block_size", 8)),
        patcher2_block_attention=bool(cfg.get("patcher2", {}).get("block_attention", False)),
        patcher2_block_size=int(cfg.get("patcher2", {}).get("block_size", 8)),
    ).to(device)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    processed_dir = Path(cfg["data"].get("processed_dir_tiny", cfg["data"]["processed_dir"]))
    tokenizer = FixedPatchTokenizer.load(processed_dir / "tokenizer.json")

    train_cfg = cfg["train"]
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute() and not ckpt_path.exists():
        ckpt_path = Path(train_cfg.get("out_dir", "outputs")) / ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device)

    ckpt_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    model_cfg_source = ckpt_cfg if isinstance(ckpt_cfg, dict) else cfg

    model = _build_model_from_cfg(model_cfg_source, tokenizer, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    token_ids = tokenizer.encode(args.prompt, add_bos=True, add_eos=False)
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)

    patcher2_enabled = bool(cfg.get("patcher2", {}).get("enabled", True))
    p2 = int(cfg.get("patcher2", {}).get("patch_size", 2)) if patcher2_enabled else 1
    large_patch_size = int(cfg.get("patcher", {}).get("patch_size", getattr(tokenizer, "patch_size", 1))) * p2
    max_new_patches = int(cfg["sample"]["max_new_patches"])
    out = model.generate(
        idx,
        max_new_tokens=max_new_patches * large_patch_size,
        temperature=float(cfg["sample"]["temperature"]),
        top_k=int(cfg["sample"]["top_k"]),
    )
    out_ids = out[0].tolist()
    text = tokenizer.decode(out_ids)

    print("Generated IDs:", out_ids)
    print("Generated text:\n", text)


if __name__ == "__main__":
    main()
