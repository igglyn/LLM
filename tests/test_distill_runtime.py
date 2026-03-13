from __future__ import annotations

import json
from pathlib import Path

import pytest

from distill.runtime.io_utils import read_jsonl
from distill.runtime.pipeline import run_extract, run_mix, run_stage_a_command
from distill.runtime.teacher_runtime import TeacherRuntimeError, find_teacher_by_name
from shared.config import parse_config


def test_end_to_end_distill_pipeline_with_local_dummy_backend(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("hello world", encoding="utf-8")
    (docs_dir / "b.txt").write_text("goodbye world", encoding="utf-8")

    config_path = tmp_path / "config.xml"
    config_path.write_text(_runtime_config_xml(str(docs_dir / "*.txt")), encoding="utf-8")

    extracted_path = tmp_path / "extracted.jsonl"
    mixed_path = tmp_path / "mixed.jsonl"
    stage_a_path = tmp_path / "stage_a.jsonl"

    assert run_extract(config_path, extracted_path) == 2
    assert run_mix(config_path, extracted_path, mixed_path) == 2
    assert run_stage_a_command(config_path, mixed_path, stage_a_path) == 2

    rows = read_jsonl(stage_a_path)
    assert len(rows) == 2
    assert set(rows[0].keys()) == {
        "document_id",
        "teacher_name",
        "group",
        "dataset_entry",
        "split",
        "text",
        "predictions",
    }


def test_teacher_lookup_by_name() -> None:
    config = parse_config(Path("examples/config.example.xml"))

    teacher = find_teacher_by_name(config.dataset.distillation, "teacher_main")
    assert teacher.name == "teacher_main"

    with pytest.raises(TeacherRuntimeError, match="not found"):
        find_teacher_by_name(config.dataset.distillation, "missing_teacher")


def test_stage_a_mode_validation(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("hello", encoding="utf-8")

    xml = _runtime_config_xml(str(docs_dir / "*.txt")).replace(
        "<StageA enabled=\"true\" teacher_ref=\"teacher_local\"><TopKLogits k=\"3\" /></StageA>",
        "<StageA enabled=\"true\" teacher_ref=\"teacher_local\"><LongContext max_tokens=\"1024\" /></StageA>",
    )
    config_path = tmp_path / "bad_stage_a.xml"
    config_path.write_text(xml, encoding="utf-8")

    from shared.config import ConfigParseError
    with pytest.raises(ConfigParseError, match="StageA"):
        parse_config(config_path)


def test_stage_a_jsonl_output_schema(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("alpha beta gamma", encoding="utf-8")

    config_path = tmp_path / "config.xml"
    config_path.write_text(_runtime_config_xml(str(docs_dir / "*.txt")), encoding="utf-8")

    extracted_path = tmp_path / "extracted.jsonl"
    mixed_path = tmp_path / "mixed.jsonl"
    stage_a_path = tmp_path / "stage_a.jsonl"

    run_extract(config_path, extracted_path)
    run_mix(config_path, extracted_path, mixed_path)
    run_stage_a_command(config_path, mixed_path, stage_a_path)

    with stage_a_path.open("r", encoding="utf-8") as handle:
        row = json.loads(handle.readline())

    assert isinstance(row["document_id"], str)
    assert isinstance(row["teacher_name"], str)
    assert isinstance(row["predictions"], list)
    assert {"token", "score", "rank"}.issubset(row["predictions"][0].keys())


def _runtime_config_xml(glob_path: str) -> str:
    return f"""
<Config>
  <Dataset>
    <SourceExtraction>
      <DatasetEntry name="set_local">
        <Source name="local" type="local_text_glob" uri="{glob_path}" />
        <SplitMapping>
          <Map from="train" to="train" />
        </SplitMapping>
        <Filter type="min_bytes" value="1" />
      </DatasetEntry>
    </SourceExtraction>
    <MixtureBuild target_documents="2" random_seed="7" min_bytes="1" depletion_policy="rebalance">
      <Group name="g1" percentage="100">
        <DatasetRef name="set_local" />
      </Group>
    </MixtureBuild>
    <Distillation>
      <Teachers>
        <Teacher name="teacher_local">
          <Backend type="dummy_local">
            <ModelRef name_or_path="dummy" />
            <Execution device="cpu" precision="fp32" />
          </Backend>
        </Teacher>
      </Teachers>
      <StageA enabled="true" teacher_ref="teacher_local"><TopKLogits k="3" /></StageA>
      <StageB enabled="false" teacher_ref="teacher_local"><LongContext max_tokens="1024" /></StageB>
      <StageC enabled="false" teacher_ref="teacher_local"><StructuredOutputs schema="json" /></StageC>
    </Distillation>
  </Dataset>
  <Model>
    <Defaults d_model="1024" n_heads="8" />
    <Patcher name="p1" patch_size="128"><Train mode="x"><Optimizer type="adamw" lr="0.001" weight_decay="0.0" /></Train><Transformer /></Patcher>
    <Trunk name="t1" context="1024"><Train mode="x"><Optimizer type="adamw" lr="0.001" weight_decay="0.0" /></Train><Transformer /></Trunk>
  </Model>
</Config>
"""
