#!/usr/bin/env python
"""Validate rewrite patcher reconstruction accuracy without creating caches."""

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
from blt_lite.utils import get_device, load_config, set_seed
from rewrite.dataloader import build_first_patcher_dataloaders
from rewrite.patcher_models import EmbeddedPatcherConfig, RewritePatcherAutoencoder


@torch.no_grad()
def _test_reconstruction(loader, emb, patcher, device: torch.device, n_batches: int | None = None) -> dict:
    total_tokens = 0
    total_matches = 0
    examples = []
    batches_run = 0

    embed_weight = emb.weight

    for batch in loader:
        if n_batches is not None and batches_run >= n_batches:
            break

        x = batch.to(device)
        token_hidden = emb(x)
        recon, _ = patcher(token_hidden)

        logits = torch.matmul(recon.float(), embed_weight.T)
        predicted = logits.argmax(dim=-1).cpu()
        original = x.cpu()

        matches = (predicted == original).sum().item()
        total_matches += int(matches)
        total_tokens += int(original.numel())

        if len(examples) < 3 and original.shape[0] > 0:
            orig_row = original[0, :64].tolist()
            pred_row = predicted[0, :64].tolist()
            orig_text = bytes([t for t in orig_row if 0 <= t < 256]).decode("utf-8", errors="replace")
            pred_text = bytes([t for t in pred_row if 0 <= t < 256]).decode("utf-8", errors="replace")
            examples.append({"original": orig_text, "reconstructed": pred_text})

        batches_run += 1

    match_rate = total_matches / max(1, total_tokens)
    return {
        "match_rate": match_rate,
        "total_tokens": total_tokens,
        "total_matches": total_matches,
        "batches_tested": batches_run,
        "examples": examples,
    }


def _print_report(stats: dict) -> None:
    print("\n=== Rewrite Patcher Reconstruction Report ===")
    print(f"  Batches tested : {stats['batches_tested']}")
    print(f"  Tokens checked : {stats['total_tokens']:,}")
    print(f"  Matches        : {stats['total_matches']:,}")
    print(f"  Match rate     : {stats['match_rate']*100:.2f}%")
    print()
    for i, ex in enumerate(stats["examples"]):
        print(f"  Example {i+1}:")
        print(f"    Original     : {repr(ex['original'])}")
        print(f"    Reconstructed: {repr(ex['reconstructed'])}")
    print("============================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate rewrite patcher reconstruction accuracy")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--patcher-checkpoint", default="", help="Path to rewrite patcher checkpoint")
    parser.add_argument("--split", choices=["train", "val"], default="val", help="Which split to evaluate")
    parser.add_argument("--recon-batches", type=int, default=None, help="Limit number of batches")
    parser.add_argument("--min-accuracy", type=float, default=None, help="Minimum required match rate in [0,1]")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    device = get_device()

    model_cfg = cfg["model"]
    patcher_cfg = cfg.get("patcher", {})
    train_cfg = cfg.get("patcher_train", {})

    processed_root = Path(cfg["data"].get("processed_dir_patcher", cfg["data"]["processed_dir"])) / "rewrite_sources"
    batch_size = int(train_cfg.get("batch_size", cfg.get("train", {}).get("batch_size", 8)))
    train_loader, val_loader, tokenizer_path = build_first_patcher_dataloaders(
        cfg, output_root=processed_root, batch_size=batch_size
    )
    loader = train_loader if args.split == "train" else val_loader

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
    patcher = RewritePatcherAutoencoder(
        patch_size=int(patcher_cfg.get("patch_size", getattr(tokenizer, "patch_size", 1))),
        cfg=embedded_cfg,
    ).to(device)

    ckpt_path = args.patcher_checkpoint or cfg.get("patcher", {}).get("pretrained_path", "")
    if not ckpt_path:
        raise ValueError("Provide --patcher-checkpoint or set patcher.pretrained_path in config")

    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "token_emb" not in ckpt or "patcher" not in ckpt:
        raise ValueError("Rewrite checkpoint must contain 'token_emb' and 'patcher' entries")
    emb.load_state_dict(ckpt["token_emb"])
    patcher.load_state_dict(ckpt["patcher"])

    emb.eval()
    patcher.eval()

    stats = _test_reconstruction(loader, emb, patcher, device, n_batches=args.recon_batches)
    _print_report(stats)

    if args.min_accuracy is not None:
        if stats["match_rate"] < args.min_accuracy:
            raise RuntimeError(
                f"Reconstruction accuracy {stats['match_rate']*100:.2f}% is below required minimum "
                f"{args.min_accuracy*100:.2f}%"
            )
        print(f"Reconstruction accuracy OK ({stats['match_rate']*100:.2f}% >= {args.min_accuracy*100:.2f}%)")


if __name__ == "__main__":
    main()
