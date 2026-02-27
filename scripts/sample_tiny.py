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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    processed_dir = Path(cfg["data"]["processed_dir"])
    tokenizer = FixedPatchTokenizer.load(processed_dir / "tokenizer.json")

    train_cfg = cfg["train"]
    ckpt = torch.load(Path(train_cfg.get("out_dir", "outputs")) / args.checkpoint, map_location=device)

    model = TinyPatchLM(
        vocab_size=tokenizer.vocab_len,
        seq_len=int(cfg["data"]["seq_len"]),
        d_model=int(cfg["model"]["d_model"]),
        n_layers=int(cfg["model"]["n_layers"]),
        n_heads=int(cfg["model"]["n_heads"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    token_ids = tokenizer.encode(args.prompt, add_bos=True, add_eos=False)
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)

    out = model.generate(
        idx,
        max_new_tokens=int(cfg["sample"]["max_new_tokens"]),
        temperature=float(cfg["sample"]["temperature"]),
        top_k=int(cfg["sample"]["top_k"]),
    )
    out_ids = out[0].tolist()
    text = tokenizer.decode(out_ids)

    print("Generated IDs:", out_ids)
    print("Generated text:\n", text)


if __name__ == "__main__":
    main()
