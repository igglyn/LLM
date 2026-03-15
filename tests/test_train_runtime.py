from __future__ import annotations

from pathlib import Path

from shared.config import parse_config, resolve_config
from train.blocks import MixOfExpertsBlock
from train.builder import build_model_runtime
from train.metrics import summarize_model_runtime
from train.runtime import _apply_loss_threshold_decay, _is_offset_step
from train.specs import RuntimeSchedulerConfig


EXAMPLE_CONFIG_PATH = Path("examples/config.example.xml")


def test_patcher_order_preserved_from_xml() -> None:
    runtime = _build_runtime(EXAMPLE_CONFIG_PATH)
    assert [patcher.name for patcher in runtime.patchers] == ["patcher_text", "patcher_code"]


def test_trunk_child_block_order_preserved_from_xml() -> None:
    runtime = _build_runtime(EXAMPLE_CONFIG_PATH)
    block_names = [block.block_name for block in runtime.trunk.blocks]

    assert block_names[:2] == ["DRope", "MixOfExperts"]
    assert block_names[2:] == ["Transformer"] * 24


def test_expert_child_block_order_preserved_from_xml() -> None:
    runtime = _build_runtime(EXAMPLE_CONFIG_PATH)
    moe = next(block for block in runtime.trunk.blocks if isinstance(block, MixOfExpertsBlock))

    assert [expert.name for expert in moe.experts] == ["expert_1"]
    assert [block.block_name for block in moe.experts[0].blocks] == ["Transformer"] * 8


def test_summary_smoke_path_exposes_train_fields() -> None:
    runtime = _build_runtime(EXAMPLE_CONFIG_PATH)
    summary = summarize_model_runtime(runtime)
    smoke_state = runtime.smoke("summary smoke")

    assert summary["patchers"][0]["train"]["optimizer"]["type"] == "adamw"
    assert summary["patchers"][0]["train"]["optimizer"]["dropout"] == 0.0
    assert summary["patchers"][0]["train"]["optimizer"]["grad_clip"] is None
    assert summary["patchers"][0]["train"]["batch_size"] == 8
    assert summary["patchers"][0]["train"]["save_every"] == 5000
    assert summary["patchers"][0]["train"]["device"] == "cuda"
    assert len(summary["patchers"][0]["train"]["schedulers"]) == 3
    assert summary["trunk"]["train"]["steps"] == 50000
    assert summary["has_moe"] is True
    assert "TrunkEnd(main_trunk)" in smoke_state.execution_trace


def test_transformer_layers_expand_runtime_block_counts() -> None:
    runtime = _build_runtime(EXAMPLE_CONFIG_PATH)

    assert len(runtime.patchers[0].blocks) == 5
    assert len(runtime.patchers[1].blocks) == 6
    assert len(runtime.trunk.blocks) == 26


def _build_runtime(config_path: Path):
    return build_model_runtime(resolve_config(parse_config(config_path)))


def test_offset_scheduler_marks_skip_range() -> None:
    schedulers = [RuntimeSchedulerConfig(scheduler_type="offset", attributes={"start_step": "2", "end_step": "5"})]

    assert _is_offset_step(schedulers, step=1) is False
    assert _is_offset_step(schedulers, step=2) is True
    assert _is_offset_step(schedulers, step=4) is True
    assert _is_offset_step(schedulers, step=5) is False


def test_loss_threshold_decay_is_one_shot_per_threshold() -> None:
    thresholds = [
        {
            "start": 0,
            "end": 10,
            "threshold": 2.0,
            "decay_factor": 0.5,
            "monitor": "train_loss",
            "triggered": False,
        }
    ]

    lr_multiplier = 1.0
    lr_multiplier = _apply_loss_threshold_decay(
        step=0,
        loss_value=1.0,
        configured_thresholds=thresholds,
        lr_multiplier=lr_multiplier,
    )
    assert lr_multiplier == 0.5

    # Loss remains below threshold, but the one-shot trigger should not re-apply decay.
    lr_multiplier = _apply_loss_threshold_decay(
        step=1,
        loss_value=0.8,
        configured_thresholds=thresholds,
        lr_multiplier=lr_multiplier,
    )
    assert lr_multiplier == 0.5

