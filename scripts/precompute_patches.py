"""Offline precompute utility for first-stage patcher outputs."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute patcher1 outputs to disk")
    parser.add_argument("--config", default="configs/examples/tiny_baseline.toml")
    parser.add_argument("--checkpoint", default="", help="optional checkpoint with model state")
    parser.add_argument("--output", default="precomputed_patcher1.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        print(f"missing dependency: {exc}. install project dependencies to precompute patches.")
        return

    from llm_lab.config.defaults import load_config
    from llm_lab.data.byte_dataset import ByteDataset
    from llm_lab.data.collate import collate_batch
    from llm_lab.models.assemble import assemble_model

    cfg = load_config(args.config)
    model = assemble_model(cfg)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise ValueError(f"checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)

    patcher1 = model.patcher1
    patcher1.eval()
    for p in patcher1.parameters():
        p.requires_grad = False

    dataset = ByteDataset(
        data_dir=cfg.data.path,
        seq_len=cfg.data.seq_len,
        seed=cfg.train.seed,
    )
    if len(dataset) == 0:
        print("dataset is empty; provide .txt files in data.path")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        drop_last=False,
    )

    hidden_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for x_u8 in loader:
            h = patcher1(x_u8)
            hidden_batches.append(h.cpu())

    hidden = torch.cat(hidden_batches, dim=0)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hidden, out)
    print(f"saved precomputed patches: {tuple(hidden.shape)} -> {out}")


if __name__ == "__main__":
    main()
