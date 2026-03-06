"""Tests for runtime shape validator helpers and integration."""

from copy import deepcopy

import pytest

from llm_lab.config.defaults import DEFAULT_CONFIG
from llm_lab.debug.shapes import assert_bytes, assert_hidden, assert_ids, assert_logits


def test_assert_bytes_raises_on_invalid_dtype() -> None:
    torch = pytest.importorskip("torch")
    x = torch.zeros(2, 4, dtype=torch.int64)
    with pytest.raises(ValueError, match="expected dtype torch.uint8"):
        assert_bytes(x, name="x")


def test_assert_ids_raises_on_invalid_rank() -> None:
    torch = pytest.importorskip("torch")
    x = torch.zeros(2, 4, 3, dtype=torch.int64)
    with pytest.raises(ValueError, match="rank 2"):
        assert_ids(x, name="ids")


def test_assert_hidden_raises_on_invalid_dtype() -> None:
    torch = pytest.importorskip("torch")
    x = torch.zeros(2, 4, 8, dtype=torch.int64)
    with pytest.raises(ValueError, match="expected floating dtype"):
        assert_hidden(x, name="h")


def test_assert_logits_raises_on_vocab_mismatch() -> None:
    torch = pytest.importorskip("torch")
    x = torch.zeros(2, 4, 128, dtype=torch.float32)
    with pytest.raises(ValueError, match="expected last dim V=256"):
        assert_logits(x, vocab_size=256, name="logits")


def test_shape_validation_smoke_passes_when_enabled() -> None:
    torch = pytest.importorskip("torch")
    from llm_lab.config.schema import ComponentCfg, DebugCfg
    from llm_lab.models.assemble import assemble_model

    cfg = deepcopy(DEFAULT_CONFIG)
    cfg.model.codec.name = "identity"
    cfg.model.codec.kwargs = {}
    cfg.model.patcher1.name = "chunk"
    cfg.model.patcher1.kwargs = {"patch_size": 8, "d_model": 16}
    cfg.model.mixers = [
        ComponentCfg(name="transformer", kwargs={"d_model": 16, "n_heads": 4, "n_layers": 1})
    ]
    cfg.model.mixer = cfg.model.mixers[0]
    cfg.model.head.name = "byte"
    cfg.model.head.kwargs = {"d_model": 16}
    cfg.debug = DebugCfg(validate_shapes=True)

    model = assemble_model(cfg)
    x = torch.randint(0, 256, (2, 17), dtype=torch.uint8)
    y = model(x)
    assert y.shape[0] == 2
