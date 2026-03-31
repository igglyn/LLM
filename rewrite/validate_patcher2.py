#!/usr/bin/env python
"""Validate rewrite second patcher by reconstructing bytes through both patchers."""

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


@torch.no_grad()
def _test_joint_reconstruction(
    loader,
    emb: torch.nn.Embedding,
    patcher1: RewritePatcherAutoencoder,
    patcher2: RewritePatcherAutoencoder,
    device: torch.device,
    patch_size: int,
    n_batches: int | None = None,
) -> dict:
    total_tokens = 0
    total_matches = 0
    total_initial_patch_bytes = 0
    total_initial_patch_matches = 0
    examples = []
    batches_run = 0

    embed_weight = emb.weight

    for batch in loader:
        if n_batches is not None and batches_run >= n_batches:
            break

        token_patches = batch.to(device)
        bsz, patch_count, patch_sz = token_patches.shape
        if patch_sz != patch_size:
            raise ValueError(f"Unexpected patch size in batch. expected={patch_size} got={patch_sz}")

        flat_tokens = token_patches.reshape(bsz, patch_count * patch_size)
        token_hidden = emb(flat_tokens)

        stage1_states = patcher1.encoder(token_hidden)
        stage2_recon_states, _ = patcher2(stage1_states)
        joint_recon_hidden = patcher1.decoder(token_hidden, stage2_recon_states)

        logits = torch.matmul(joint_recon_hidden.float(), embed_weight.T)
        predicted = logits.argmax(dim=-1)

        orig_cpu = flat_tokens.cpu()
        pred_cpu = predicted.cpu()

        token_matches = (pred_cpu == orig_cpu).sum().item()
        total_matches += int(token_matches)
        total_tokens += int(orig_cpu.numel())

        orig_first = token_patches[:, :, 0].cpu()
        pred_first = pred_cpu.view(bsz, patch_count, patch_size)[:, :, 0].cpu()
        first_matches = (pred_first == orig_first).sum().item()
        total_initial_patch_matches += int(first_matches)
        total_initial_patch_bytes += int(orig_first.numel())

        if len(examples) < 3 and orig_cpu.shape[0] > 0:
            orig_row = orig_cpu[0, :64].tolist()
            pred_row = pred_cpu[0, :64].tolist()
            orig_text = bytes([t for t in orig_row if 0 <= t < 256]).decode("utf-8", errors="replace")
            pred_text = bytes([t for t in pred_row if 0 <= t < 256]).decode("utf-8", errors="replace")
            examples.append({"original": orig_text, "reconstructed": pred_text})

        batches_run += 1

    match_rate = total_matches / max(1, total_tokens)
    first_byte_match_rate = total_initial_patch_matches / max(1, total_initial_patch_bytes)
    return {
        "match_rate": match_rate,
        "first_byte_match_rate": first_byte_match_rate,
        "total_tokens": total_tokens,
        "total_matches": total_matches,
        "total_initial_patch_bytes": total_initial_patch_bytes,
        "total_initial_patch_matches": total_initial_patch_matches,
        "batches_tested": batches_run,
        "examples": examples,
    }


def _print_report(stats: dict) -> None:
    print("\n=== Rewrite Patcher2 Joint Reconstruction Report ===")
    print(f"  Batches tested                 : {stats['batches_tested']}")
    print(f"  Tokens checked (running total) : {stats['total_tokens']:,}")
    print(f"  Token matches (running total)  : {stats['total_matches']:,}")
    print(f"  Token match rate               : {stats['match_rate']*100:.2f}%")
    print(f"  Initial patch bytes checked    : {stats['total_initial_patch_bytes']:,}")
    print(f"  Initial patch byte matches     : {stats['total_initial_patch_matches']:,}")
    print(f"  Initial patch byte match rate  : {stats['first_byte_match_rate']*100:.2f}%")
    print()
    for i, ex in enumerate(stats["examples"]):
        print(f"  Example {i+1}:")
        print(f"    Original     : {repr(ex['original'])}")
        print(f"    Reconstructed: {repr(ex['reconstructed'])}")
    print("====================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate rewrite patcher2 by reconstructing through patcher1+patcher2")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--patcher1-checkpoint", default="", help="Path to rewrite patcher1 checkpoint")
    parser.add_argument("--patcher2-checkpoint", default="", help="Path to rewrite patcher2 checkpoint")
    parser.add_argument("--split", choices=["train", "val"], default="val", help="Which split to evaluate")
    parser.add_argument("--recon-batches", type=int, default=None, help="Limit number of batches")
    parser.add_argument("--min-accuracy", type=float, default=None, help="Minimum required token match rate in [0,1]")
    parser.add_argument(
        "--min-first-byte-accuracy",
        type=float,
        default=None,
        help="Minimum required initial-patch-byte match rate in [0,1]",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    device = get_device()

    train_cfg = cfg.get("patcher2_train", {})
    processed_root = Path(cfg["data"].get("processed_dir_patcher", cfg["data"]["processed_dir"])) / "rewrite_sources"
    batch_size = int(train_cfg.get("batch_size", cfg.get("train", {}).get("batch_size", 8)))
    train_loader, val_loader, tokenizer_path = build_second_patcher_dataloaders(
        cfg,
        output_root=processed_root,
        batch_size=batch_size,
    )
    loader = train_loader if args.split == "train" else val_loader

    tokenizer = FixedPatchTokenizer.load(tokenizer_path)
    d_model = int(cfg["model"]["d_model"])
    patch_size = int(cfg.get("patcher", {}).get("patch_size", getattr(tokenizer, "patch_size", 1)))

    emb = torch.nn.Embedding(tokenizer.vocab_len, d_model).to(device)
    patcher1 = RewritePatcherAutoencoder(patch_size=patch_size, cfg=_embedded_cfg(cfg, "patcher")).to(device)
    patcher2 = RewritePatcherAutoencoder(
        patch_size=int(cfg.get("patcher2", {}).get("patch_size", 2)),
        cfg=_embedded_cfg(cfg, "patcher2"),
    ).to(device)

    p1_ckpt_path = args.patcher1_checkpoint or cfg.get("patcher", {}).get("pretrained_path", "")
    p2_ckpt_path = args.patcher2_checkpoint or cfg.get("patcher2", {}).get("pretrained_path", "")
    if not p1_ckpt_path:
        raise ValueError("Provide --patcher1-checkpoint or set patcher.pretrained_path in config")
    if not p2_ckpt_path:
        raise ValueError("Provide --patcher2-checkpoint or set patcher2.pretrained_path in config")

    p1_ckpt = torch.load(p1_ckpt_path, map_location=device)
    if not isinstance(p1_ckpt, dict) or "token_emb" not in p1_ckpt or "patcher" not in p1_ckpt:
        raise ValueError("Rewrite patcher1 checkpoint must contain 'token_emb' and 'patcher' entries")
    emb.load_state_dict(p1_ckpt["token_emb"])
    patcher1.load_state_dict(p1_ckpt["patcher"])

    p2_ckpt = torch.load(p2_ckpt_path, map_location=device)
    if not isinstance(p2_ckpt, dict) or "patcher2" not in p2_ckpt:
        raise ValueError("Rewrite patcher2 checkpoint must contain a 'patcher2' entry")
    patcher2.load_state_dict(p2_ckpt["patcher2"])

    emb.eval()
    patcher1.eval()
    patcher2.eval()

    stats = _test_joint_reconstruction(loader, emb, patcher1, patcher2, device, patch_size, n_batches=args.recon_batches)
    _print_report(stats)

    if args.min_accuracy is not None and stats["match_rate"] < args.min_accuracy:
        raise RuntimeError(
            f"Token reconstruction accuracy {stats['match_rate']*100:.2f}% is below required minimum "
            f"{args.min_accuracy*100:.2f}%"
        )
    if args.min_accuracy is not None:
        print(f"Token reconstruction accuracy OK ({stats['match_rate']*100:.2f}% >= {args.min_accuracy*100:.2f}%)")

    if args.min_first_byte_accuracy is not None and stats["first_byte_match_rate"] < args.min_first_byte_accuracy:
        raise RuntimeError(
            f"Initial patch-byte accuracy {stats['first_byte_match_rate']*100:.2f}% is below required minimum "
            f"{args.min_first_byte_accuracy*100:.2f}%"
        )
    if args.min_first_byte_accuracy is not None:
        print(
            "Initial patch-byte accuracy OK "
            f"({stats['first_byte_match_rate']*100:.2f}% >= {args.min_first_byte_accuracy*100:.2f}%)"
        )


if __name__ == "__main__":
    main()
