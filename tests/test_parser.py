from __future__ import annotations

from pathlib import Path

import pytest

from shared.config import ConfigParseError, ConfigResolutionError, parse_config, resolve_config
from shared.config.specs import (
    ConfigSpec,
    DatasetSpec,
    DefaultsSpec,
    DistillationSpec,
    MixtureBuildSpec,
    ModelSpec,
    PatcherSpec,
    SourceExtractionSpec,
    TrainSpec,
    OptimizerSpec,
    TransformerBlockSpec,
)


EXAMPLE_CONFIG_PATH = Path("examples/config.example.xml")
HF_EXAMPLE_CONFIG_PATH = Path("examples/config.hf.example.xml")


def test_parse_example_config_smoke() -> None:
    config = parse_config(EXAMPLE_CONFIG_PATH)

    assert len(config.dataset.source_extraction.dataset_entries) == 3
    assert len(config.dataset.mixture_build.groups) == 3
    assert len(config.dataset.distillation.teachers.teachers) == 1
    assert len(config.model.patchers) == 2
    assert config.model.trunk is not None
    assert len(config.model.trunk.mix_of_experts_blocks) == 1
    assert len(config.model.trunk.mix_of_experts_blocks[0].experts) == 1


def test_parse_hf_example_config_smoke() -> None:
    config = parse_config(HF_EXAMPLE_CONFIG_PATH)

    assert len(config.dataset.source_extraction.dataset_entries) == 1
    assert config.dataset.source_extraction.dataset_entries[0].source.source_type == "huggingface"
    assert config.dataset.distillation.teachers.teachers[0].backend.backend_type == "huggingface"
    assert config.model.patchers[0].rope_blocks[0].base == 10000.0
    assert config.model.patchers[0].rope_blocks[0].scale == 1.0
    assert config.model.trunk.drope_blocks[0].base == 10000.0
    assert config.model.trunk.drope_blocks[0].scale == 1.0


def test_scheduler_list_preserves_order() -> None:
    config = parse_config(EXAMPLE_CONFIG_PATH)

    first_patcher_schedulers = config.model.patchers[0].train.optimizer.schedulers
    assert [scheduler.scheduler_type for scheduler in first_patcher_schedulers] == ["warmup", "cosine", "loss_threshold"]
    assert config.model.patchers[0].train.steps == 20000




def test_train_requires_scheduler_step_coverage(tmp_path: Path) -> None:
    xml = _minimal_valid_xml(
        patcher_attrs='name="p1" patch_size="128"',
        patcher_transformer='<Transformer />',
        trunk_attrs='name="t1" context="1024"',
        trunk_transformer='<Transformer />',
    ).replace(
        '<Scheduler type="cosine" start_step="0" end_step="100" />',
        '<Scheduler type="cosine" start_step="10" end_step="100" />',
        1,
    )
    path = tmp_path / "uncovered_steps.xml"
    path.write_text(xml, encoding="utf-8")

    with pytest.raises(ConfigParseError, match="uncovered steps"):
        parse_config(path)


def test_train_requires_scheduler_ranges(tmp_path: Path) -> None:
    xml = _minimal_valid_xml(
        patcher_attrs='name="p1" patch_size="128"',
        patcher_transformer='<Transformer />',
        trunk_attrs='name="t1" context="1024"',
        trunk_transformer='<Transformer />',
    ).replace(
        '<Scheduler type="cosine" start_step="0" end_step="100" />',
        '<Scheduler type="cosine" />',
        1,
    )
    path = tmp_path / "missing_scheduler_range.xml"
    path.write_text(xml, encoding="utf-8")

    with pytest.raises(ConfigParseError, match="start_step"):
        parse_config(path)



def test_cosine_scheduler_rejects_t_max(tmp_path: Path) -> None:
    xml = _minimal_valid_xml(
        patcher_attrs='name="p1" patch_size="128"',
        patcher_transformer='<Transformer />',
        trunk_attrs='name="t1" context="1024"',
        trunk_transformer='<Transformer />',
    ).replace(
        '<Scheduler type="cosine" start_step="0" end_step="100" />',
        '<Scheduler type="cosine" start_step="0" end_step="100" t_max="100" />',
        1,
    )
    path = tmp_path / "cosine_tmax.xml"
    path.write_text(xml, encoding="utf-8")

    with pytest.raises(ConfigParseError, match="should not include"):
        parse_config(path)

def test_patcher_child_block_order_preserves_order() -> None:
    config = parse_config(EXAMPLE_CONFIG_PATH)

    assert config.model.patchers[0].block_order == ["RoPE", "Transformer", "PosEmbedding"]
    assert config.model.patchers[1].block_order == ["PosEmbedding", "RoPE", "Transformer"]


def test_model_level_defaults_resolve_for_transformer() -> None:
    config = parse_config(EXAMPLE_CONFIG_PATH)
    resolved = resolve_config(config)

    first_transformer = resolved.model.patchers[0].transformer_blocks[0]
    assert first_transformer.d_model == 4096
    assert first_transformer.n_heads == 32


def test_patcher_level_override_beats_model_default(tmp_path: Path) -> None:
    xml = _minimal_valid_xml(
        patcher_attrs='name="p1" patch_size="128" d_model="2048"',
        patcher_transformer='<Transformer n_heads="16" />',
        trunk_attrs='name="t1" context="1024"',
        trunk_transformer='<Transformer />',
    )
    path = tmp_path / "patcher_override.xml"
    path.write_text(xml, encoding="utf-8")

    resolved = resolve_config(parse_config(path))
    block = resolved.model.patchers[0].transformer_blocks[0]
    assert block.d_model == 2048
    assert block.n_heads == 16


def test_trunk_level_override_beats_model_default(tmp_path: Path) -> None:
    xml = _minimal_valid_xml(
        patcher_attrs='name="p1" patch_size="128"',
        patcher_transformer='<Transformer />',
        trunk_attrs='name="t1" context="1024" d_model="8192" n_heads="64"',
        trunk_transformer='<Transformer />',
    )
    path = tmp_path / "trunk_override.xml"
    path.write_text(xml, encoding="utf-8")

    resolved = resolve_config(parse_config(path))
    block = resolved.model.trunk.transformer_blocks[0]
    assert block.d_model == 8192
    assert block.n_heads == 64


def test_explicit_block_override_beats_inherited_values(tmp_path: Path) -> None:
    xml = _minimal_valid_xml(
        patcher_attrs='name="p1" patch_size="128" d_model="2048" n_heads="16"',
        patcher_transformer='<Transformer d_model="3072" n_heads="24" />',
        trunk_attrs='name="t1" context="1024"',
        trunk_transformer='<Transformer />',
    )
    path = tmp_path / "explicit_override.xml"
    path.write_text(xml, encoding="utf-8")

    resolved = resolve_config(parse_config(path))
    block = resolved.model.patchers[0].transformer_blocks[0]
    assert block.d_model == 3072
    assert block.n_heads == 24


def test_rope_drope_base_and_scale_defaults_and_overrides(tmp_path: Path) -> None:
    xml = """
<Config>
  <Dataset>
    <SourceExtraction />
    <MixtureBuild />
    <Distillation>
      <Teachers>
        <Teacher name="t">
          <Backend type="x">
            <ModelRef name_or_path="m" />
            <Execution device="cpu" precision="fp32" />
          </Backend>
        </Teacher>
      </Teachers>
      <Stage name="StageA" enabled="true" teacher_ref="t"><TopKLogits k="16" /></Stage>
      <Stage name="StageB" enabled="true" teacher_ref="t"><LongContext max_tokens="1024" /></Stage>
      <Stage name="StageC" enabled="true" teacher_ref="t"><StructuredOutputs schema="json" /></Stage>
    </Distillation>
  </Dataset>
  <Model>
    <Defaults d_model="1024" n_heads="8" />
    <Patcher name="p1" patch_size="128">
      <Train steps="100"><Optimizer type="adamw" lr="0.001" weight_decay="0.1"><Scheduler type="cosine" start_step="0" end_step="100" /></Optimizer></Train>
      <RoPE base="32000" scale="0.8" />
      <Transformer />
    </Patcher>
    <Trunk name="t1" context="1024">
      <Train steps="100"><Optimizer type="adamw" lr="0.0005" weight_decay="0.1"><Scheduler type="cosine" start_step="0" end_step="100" /></Optimizer></Train>
      <DRope />
      <Transformer />
    </Trunk>
  </Model>
</Config>
"""
    path = tmp_path / "rope_drope_defaults.xml"
    path.write_text(xml, encoding="utf-8")

    resolved = resolve_config(parse_config(path))
    rope = resolved.model.patchers[0].rope_blocks[0]
    drope = resolved.model.trunk.drope_blocks[0]

    assert rope.base == 32000.0
    assert rope.scale == 0.8
    assert drope.base == 10000.0
    assert drope.scale == 1.0


def test_resolution_raises_when_transformer_unresolved() -> None:
    invalid = ConfigSpec(
        dataset=DatasetSpec(
            source_extraction=SourceExtractionSpec(),
            mixture_build=MixtureBuildSpec(),
            distillation=DistillationSpec(),
        ),
        model=ModelSpec(
            defaults=DefaultsSpec(d_model=None, n_heads=None),  # type: ignore[arg-type]
            patchers=[
                PatcherSpec(
                    name="p",
                    patch_size=1,
                    train=TrainSpec(steps=1, optimizer=OptimizerSpec(optimizer_type="adamw", lr=0.1, weight_decay=0.0)),
                    transformer_blocks=[TransformerBlockSpec()],
                    block_order=["Transformer"],
                )
            ],
            trunk=None,
        ),
    )

    with pytest.raises(ConfigResolutionError, match="missing d_model or n_heads"):
        resolve_config(invalid)


def test_config_requires_dataset_and_model(tmp_path: Path) -> None:
    xml = """<Config><Dataset /></Config>"""
    path = tmp_path / "bad.xml"
    path.write_text(xml, encoding="utf-8")

    with pytest.raises(ConfigParseError, match="<Config> must contain exactly <Dataset> and <Model> children"):
        parse_config(path)


def test_stage_mode_validation(tmp_path: Path) -> None:
    xml = _minimal_valid_xml(
        patcher_attrs='name="p1" patch_size="128"',
        patcher_transformer='<Transformer />',
        trunk_attrs='name="t1" context="1024"',
        trunk_transformer='<Transformer />',
    ).replace(
        '<Stage name="StageA" enabled="true" teacher_ref="t"><TopKLogits k="16" /></Stage>',
        '<Stage name="StageA" enabled="true" teacher_ref="t"><TopKLogits k="16" /><LongContext max_tokens="100" /></Stage>',
    )
    path = tmp_path / "bad_stage.xml"
    path.write_text(xml, encoding="utf-8")

    with pytest.raises(ConfigParseError, match='must contain exactly one mode block'):
        parse_config(path)


def test_teacher_requires_backend_children(tmp_path: Path) -> None:
    xml = _minimal_valid_xml(
        patcher_attrs='name="p1" patch_size="128"',
        patcher_transformer='<Transformer />',
        trunk_attrs='name="t1" context="1024"',
        trunk_transformer='<Transformer />',
    ).replace('<ModelRef name_or_path="m" />', '')
    path = tmp_path / "bad_teacher.xml"
    path.write_text(xml, encoding="utf-8")

    with pytest.raises(ConfigParseError, match="<Backend> missing required child <ModelRef>"):
        parse_config(path)


def _minimal_valid_xml(*, patcher_attrs: str, patcher_transformer: str, trunk_attrs: str, trunk_transformer: str) -> str:
    return f"""
<Config>
  <Dataset>
    <SourceExtraction />
    <MixtureBuild />
    <Distillation>
      <Teachers>
        <Teacher name="t">
          <Backend type="x">
            <ModelRef name_or_path="m" />
            <Execution device="cpu" precision="fp32" />
          </Backend>
        </Teacher>
      </Teachers>
      <Stage name="StageA" enabled="true" teacher_ref="t"><TopKLogits k="16" /></Stage>
      <Stage name="StageB" enabled="true" teacher_ref="t"><LongContext max_tokens="1024" /></Stage>
      <Stage name="StageC" enabled="true" teacher_ref="t"><StructuredOutputs schema="json" /></Stage>
    </Distillation>
  </Dataset>
  <Model>
    <Defaults d_model="1024" n_heads="8" />
    <Patcher {patcher_attrs}>
      <Train steps="100">
        <Optimizer type="adamw" lr="0.001" weight_decay="0.1">
          <Scheduler type="cosine" start_step="0" end_step="100" />
        </Optimizer>
      </Train>
      {patcher_transformer}
    </Patcher>
    <Trunk {trunk_attrs}>
      <Train steps="100">
        <Optimizer type="adamw" lr="0.0005" weight_decay="0.1">
          <Scheduler type="cosine" start_step="0" end_step="100" />
        </Optimizer>
      </Train>
      {trunk_transformer}
    </Trunk>
  </Model>
</Config>
"""
