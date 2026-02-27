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

import numpy as np

from blt_lite.dataset import TextFolderProvider
from blt_lite.tokenizer import FixedPatchTokenizer
from blt_lite.utils import ensure_dir, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    provider = TextFolderProvider(cfg["data"]["raw_path"], cfg["data"].get("pattern", "*.txt"))
    texts = list(provider.iter_texts())
    if not texts:
        raise RuntimeError("No text files found in data/raw path.")

    tok_cfg = cfg.get("tokenizer", {})
    tokenizer = FixedPatchTokenizer()
    tokenizer.fit(texts)

    add_bos = tok_cfg.get("add_bos", True)
    add_eos = tok_cfg.get("add_eos", True)
    all_ids = []
    for text in texts:
        all_ids.extend(tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos))
    all_ids = np.asarray(all_ids, dtype=np.int32)

    split = int(len(all_ids) * float(cfg["data"].get("train_split", 0.95)))
    train_ids = all_ids[:split]
    val_ids = all_ids[split:]

    out_dir = ensure_dir(cfg["data"]["processed_dir"])
    np.save(out_dir / "train_tokens.npy", train_ids)
    np.save(out_dir / "val_tokens.npy", val_ids)
    tokenizer.save(out_dir / "tokenizer.json")

    diagnostics = tokenizer.diagnostics(texts, add_bos=add_bos, add_eos=add_eos)
    with open(out_dir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"Prepared data in {out_dir}")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
