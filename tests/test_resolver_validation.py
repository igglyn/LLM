from __future__ import annotations

from pathlib import Path

from shared.config import parse_config, resolve_config


def test_resolver_smoke_on_canonical_config() -> None:
    config = parse_config(Path("examples/config.example.xml"))
    resolved = resolve_config(config)

    assert resolved.model.patchers[0].transformer_blocks[0].d_model == 4096
    assert resolved.model.patchers[0].transformer_blocks[0].n_heads == 32
    assert resolved.model.trunk is not None
    assert resolved.model.trunk.transformer_blocks[0].d_model == 4096
