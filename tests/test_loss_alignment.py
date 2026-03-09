"""Tests for byte-level loss/logit alignment checks in training loop."""

from __future__ import annotations

from copy import deepcopy

import pytest


def test_train_loop_accepts_byte_aligned_logits() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn
    from torch.utils.data import DataLoader

    from llm_lab.train.loop import train_loop

    class _ByteAlignedToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, x_u8: torch.Tensor) -> torch.Tensor:
            # Emit one logit position per input byte.
            one_hot = torch.nn.functional.one_hot(x_u8.to(torch.long), num_classes=256).to(torch.float32)
            return one_hot * self.scale

    model = _ByteAlignedToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    samples = [torch.randint(0, 256, (12,), dtype=torch.uint8) for _ in range(4)]
    dataloader = DataLoader(samples, batch_size=2, shuffle=False)

    metrics = train_loop(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        steps=1,
    )

    assert metrics["step"] == 1
    assert float(metrics["loss"]) == pytest.approx(float(metrics["loss"]))


def test_train_loop_raises_on_compressed_patch_space_logits() -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from llm_lab.config.defaults import DEFAULT_CONFIG
    from llm_lab.config.schema import ComponentCfg
    from llm_lab.models.assemble import assemble_model
    from llm_lab.train.loop import train_loop

    cfg = deepcopy(DEFAULT_CONFIG)
    cfg.model.codec.name = "identity"
    cfg.model.codec.kwargs = {}
    cfg.model.patcher1.name = "chunk"
    cfg.model.patcher1.kwargs = {"patch_size": 4, "d_model": 16}
    cfg.model.mixers = [
        ComponentCfg(name="transformer", kwargs={"d_model": 16, "n_heads": 4, "n_layers": 1})
    ]
    cfg.model.mixer = cfg.model.mixers[0]
    cfg.model.head.name = "byte"
    cfg.model.head.kwargs = {"d_model": 16}

    model = assemble_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    samples = [torch.randint(0, 256, (16,), dtype=torch.uint8) for _ in range(4)]
    dataloader = DataLoader(samples, batch_size=2, shuffle=False)

    with pytest.raises(ValueError, match="byte-aligned logits"):
        train_loop(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            steps=1,
        )
