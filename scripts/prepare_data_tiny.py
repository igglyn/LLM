#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json
import shutil

import numpy as np

from blt_lite.utils import ensure_dir, load_config


def _copy_stage(source_dir: Path, dest_dir: Path) -> dict:
    ensure_dir(dest_dir)
    for fname in ("train_tokens.npy", "val_tokens.npy", "tokenizer.json"):
        src = source_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"Missing required source artifact: {src}")
        shutil.copy2(src, dest_dir / fname)

    train_tokens = np.load(dest_dir / "train_tokens.npy", mmap_mode="r")
    val_tokens = np.load(dest_dir / "val_tokens.npy", mmap_mode="r")
    return {
        "source_dir": str(source_dir),
        "stage_dir": str(dest_dir),
        "train_tokens": int(train_tokens.shape[0]),
        "val_tokens": int(val_tokens.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare explicit TinyPatchLM token artifacts.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    source_dir = Path(data_cfg.get("processed_dir_patcher2", data_cfg.get("processed_dir_patcher", data_cfg["processed_dir"])))
    out_dir = Path(data_cfg.get("processed_dir_tiny", "data/processed/tiny"))

    summary = _copy_stage(source_dir, out_dir)
    p1 = int(cfg.get("patcher", {}).get("patch_size", 1))
    p2 = int(cfg.get("patcher2", {}).get("patch_size", 2))
    summary["model_seq_len_large_patches"] = int(cfg["model"]["seq_len"])
    summary["effective_token_seq_len"] = int(cfg["model"]["seq_len"]) * p1 * p2

    with open(out_dir / "stage_info.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Prepared explicit TinyPatchLM stage data in {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
