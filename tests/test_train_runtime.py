from __future__ import annotations

from pathlib import Path

from shared.config import parse_config, resolve_config
from train.blocks import MixOfExpertsBlock
from train.builder import build_model_runtime
from train.metrics import summarize_model_runtime


EXAMPLE_CONFIG_PATH = Path("examples/config.example.xml")


def test_patcher_order_preserved_from_xml() -> None:
    runtime = _build_runtime(EXAMPLE_CONFIG_PATH)
    assert [patcher.name for patcher in runtime.patchers] == ["patcher_text", "patcher_code"]


def test_trunk_child_block_order_preserved_from_xml() -> None:
    runtime = _build_runtime(EXAMPLE_CONFIG_PATH)
    assert [block.block_name for block in runtime.trunk.blocks] == ["DRope", "MixOfExperts", "Transformer"]


def test_expert_child_block_order_preserved_from_xml() -> None:
    runtime = _build_runtime(EXAMPLE_CONFIG_PATH)
    moe = next(block for block in runtime.trunk.blocks if isinstance(block, MixOfExpertsBlock))

    assert [expert.name for expert in moe.experts] == ["expert_1"]
    assert [block.block_name for block in moe.experts[0].blocks] == ["Transformer"]


def test_summary_smoke_path_exposes_train_fields() -> None:
    runtime = _build_runtime(EXAMPLE_CONFIG_PATH)
    summary = summarize_model_runtime(runtime)
    smoke_state = runtime.smoke("summary smoke")

    assert summary["patchers"][0]["optimizer"] == "adamw"
    assert summary["patchers"][0]["scheduler_count"] == 2
    assert summary["trunk"]["train_mode"] == "full"
    assert "TrunkEnd(main_trunk)" in smoke_state.execution_trace


def _build_runtime(config_path: Path):
    return build_model_runtime(resolve_config(parse_config(config_path)))
