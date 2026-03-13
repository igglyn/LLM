from __future__ import annotations

from pathlib import Path

from shared.config import parse_config, resolve_config
from train.runtime.model_runtime import MixOfExpertsBlock, TransformerBlock, build_model_runtime, summarize_model_runtime


EXAMPLE_CONFIG_PATH = Path("examples/config.example.xml")


def test_build_model_from_canonical_xml() -> None:
    model_runtime = _build_runtime(EXAMPLE_CONFIG_PATH)

    assert len(model_runtime.patchers) == 2
    assert model_runtime.trunk.name == "main_trunk"


def test_patcher_order_is_xml_order() -> None:
    model_runtime = _build_runtime(EXAMPLE_CONFIG_PATH)

    assert [patcher.name for patcher in model_runtime.patchers] == ["patcher_text", "patcher_code"]


def test_trunk_block_order_is_preserved() -> None:
    model_runtime = _build_runtime(EXAMPLE_CONFIG_PATH)

    assert [block.block_name for block in model_runtime.trunk.blocks] == ["DRope", "MixOfExperts", "Transformer"]


def test_moe_subtree_build_and_probe() -> None:
    model_runtime = _build_runtime(EXAMPLE_CONFIG_PATH)

    moe_blocks = [block for block in model_runtime.trunk.blocks if isinstance(block, MixOfExpertsBlock)]
    assert len(moe_blocks) == 1
    assert moe_blocks[0].name == "moe_main"
    assert len(moe_blocks[0].experts) == 1
    assert moe_blocks[0].experts[0].name == "expert_1"

    state = model_runtime.smoke("probe moe")
    assert "moe_main" in state.moe_metrics
    assert state.moe_metrics["moe_main"]["selected_expert"] == "expert_1"


def test_inherited_defaults_resolve_into_instantiated_transformers() -> None:
    model_runtime = _build_runtime(EXAMPLE_CONFIG_PATH)

    first_patcher_transformer = next(
        block for block in model_runtime.patchers[0].blocks if isinstance(block, TransformerBlock)
    )
    assert first_patcher_transformer.d_model == 4096
    assert first_patcher_transformer.n_heads == 32

    trunk_transformer = next(block for block in model_runtime.trunk.blocks if isinstance(block, TransformerBlock))
    assert trunk_transformer.d_model == 4096
    assert trunk_transformer.n_heads == 32


def test_train_blocks_exposed_in_summary() -> None:
    model_runtime = _build_runtime(EXAMPLE_CONFIG_PATH)
    summary = summarize_model_runtime(model_runtime)

    assert summary["patchers"][0]["train_mode"] == "finetune"
    assert summary["patchers"][0]["optimizer"] == "adamw"
    assert summary["patchers"][0]["scheduler_count"] == 2
    assert summary["trunk"]["train_mode"] == "full"
    assert summary["trunk"]["optimizer"] == "adamw"


def _build_runtime(config_path: Path):
    raw = parse_config(config_path)
    resolved = resolve_config(raw)
    return build_model_runtime(resolved)
