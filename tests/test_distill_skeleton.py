from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from distill.io_jsonl import read_jsonl, write_jsonl
from distill.schemas import MixtureEntry, NormalizedDocument
from distill.stages.stage_a import StageAError, run_stage_a
from distill.teachers import teacher_by_ref
from shared.config import parse_config, resolve_config


def test_cli_summary(tmp_path: Path) -> None:
    config = tmp_path / "config.xml"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello", encoding="utf-8")
    config.write_text(_config_xml(str(docs / "*.txt")), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "distill", "summary", "--config", str(config)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "dataset_entries=1" in completed.stdout
    assert "groups=1" in completed.stdout
    assert "teachers=1" in completed.stdout


def test_teacher_lookup_by_teacher_ref() -> None:
    resolved = resolve_config(parse_config(Path("examples/config.example.xml")))
    teacher = teacher_by_ref(resolved.dataset.distillation, "teacher_main")
    assert teacher.name == "teacher_main"


def test_stage_a_mode_validation_rejects_non_positive_k(tmp_path: Path) -> None:
    config = tmp_path / "config.xml"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello", encoding="utf-8")
    config.write_text(_config_xml(str(docs / "*.txt"), topk_k="0"), encoding="utf-8")

    resolved = resolve_config(parse_config(config))
    with pytest.raises(StageAError, match="positive k"):
        run_stage_a(
            resolved,
            [MixtureEntry(document_id="d", group="g", dataset_entry="set_local", split="train", text="hello", byte_length=5)],
        )


def test_jsonl_writer_schema_shape(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [
        NormalizedDocument(
            document_id="d1",
            dataset_entry="entry",
            split="train",
            text="abc",
            byte_length=3,
            metadata={"path": "x"},
        )
    ]
    write_jsonl(path, rows)
    parsed = read_jsonl(path)

    assert set(parsed[0].keys()) == {"document_id", "dataset_entry", "split", "text", "byte_length", "metadata"}


def _config_xml(glob_path: str, *, topk_k: str = "3") -> str:
    return f"""
<Config>
  <Dataset>
    <SourceExtraction>
      <DatasetEntry name=\"set_local\">
        <Source name=\"local\" type=\"local_text_glob\" uri=\"{glob_path}\" />
        <SplitMapping><Map from=\"train\" to=\"train\" /></SplitMapping>
      </DatasetEntry>
    </SourceExtraction>
    <MixtureBuild target_documents=\"1\" random_seed=\"7\"> 
      <Group name=\"g1\" percentage=\"100\"><DatasetRef name=\"set_local\" /></Group>
    </MixtureBuild>
    <Distillation>
      <Teachers>
        <Teacher name=\"teacher_local\">
          <Backend type=\"dummy_local\">
            <ModelRef name_or_path=\"dummy\" />
            <Execution device=\"cpu\" precision=\"fp32\" />
          </Backend>
        </Teacher>
      </Teachers>
      <StageA enabled=\"true\" teacher_ref=\"teacher_local\"><TopKLogits k=\"{topk_k}\" /></StageA>
      <StageB enabled=\"false\" teacher_ref=\"teacher_local\"><LongContext max_tokens=\"1024\" /></StageB>
      <StageC enabled=\"false\" teacher_ref=\"teacher_local\"><StructuredOutputs schema=\"json\" /></StageC>
    </Distillation>
  </Dataset>
  <Model>
    <Defaults d_model=\"1024\" n_heads=\"8\" />
    <Patcher name=\"p1\" patch_size=\"128\"><Train mode=\"x\"><Optimizer type=\"adamw\" lr=\"0.001\" weight_decay=\"0.0\" /></Train><Transformer /></Patcher>
    <Trunk name=\"t1\" context=\"1024\"><Train mode=\"x\"><Optimizer type=\"adamw\" lr=\"0.001\" weight_decay=\"0.0\" /></Train><Transformer /></Trunk>
  </Model>
</Config>
"""
