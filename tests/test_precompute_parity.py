"""Parity tests for precomputed hidden states vs online downstream inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


pytest.importorskip("torch")
import torch
from torch.utils.data import DataLoader

from llm_lab.config.defaults import load_config
from llm_lab.data.byte_dataset import ByteDataset
from llm_lab.data.collate import collate_batch
from llm_lab.models.assemble import assemble_model
from scripts.precompute_patches import main as precompute_main


def test_precompute_matches_online_downstream_hidden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Ensure multiple full chunks at seq_len=16.
    (data_dir / "sample.txt").write_text("abcdefghijklmnopqrstuvwxyz0123456789")

    config_path = tmp_path / "cfg.toml"
    output_path = tmp_path / "precomputed.pt"
    config_path.write_text(
        f"""
[data]
path = "{data_dir}"
seq_len = 16
batch_size = 2

[model.codec]
name = "fsq"
levels_per_dim = 8
d_model = 16

[model.patcher1]
name = "chunk"
patch_size = 4
d_model = 16

[[model.mixers]]
name = "transformer"
d_model = 16
n_heads = 4
n_layers = 1

[model.head]
name = "byte"
d_model = 16

[train]
seed = 7
""".strip()
    )

    torch.manual_seed(123)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "precompute_patches.py",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--batch-size",
            "2",
        ],
    )
    precompute_main()

    saved = torch.load(output_path, map_location="cpu")

    cfg = load_config(str(config_path))
    torch.manual_seed(123)
    model = assemble_model(cfg)
    model.eval()

    dataset = ByteDataset(data_dir=cfg.data.path, seq_len=cfg.data.seq_len, seed=cfg.train.seed)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_batch, drop_last=False)

    online_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for x_u8 in loader:
            _, h_downstream = model.downstream_hidden_from_bytes(x_u8, apply_stop_gradient=False)
            online_batches.append(h_downstream.cpu())

    online = torch.cat(online_batches, dim=0)
    assert saved.shape == online.shape
    assert torch.allclose(saved, online, atol=1e-6, rtol=1e-6)
