from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from distill.io_jsonl import read_jsonl, write_jsonl
from distill.extraction import run_extraction
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


def test_huggingface_source_type_is_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_datasets = ModuleType("datasets")

    def fake_load_dataset(name: str, *, split: str):
        assert name == "my-dataset"
        assert split == "train"
        return [{"body": "hello hf"}, {"body": "world hf"}]

    fake_datasets.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    config = tmp_path / "hf_config.xml"
    config.write_text(_hf_config_xml(), encoding="utf-8")
    resolved = resolve_config(parse_config(config))

    rows = run_extraction(resolved)
    assert len(rows) == 2
    assert rows[0].split == "train"
    assert rows[0].metadata["hf_split"] == "train"
    assert rows[0].text == "hello hf"






def test_huggingface_source_with_config_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_datasets = ModuleType("datasets")

    def fake_load_dataset(name: str, config: str, *, split: str):
        assert name == "my-dataset"
        assert config == "wikitext-103-v1"
        assert split == "train"
        return [{"body": "hello cfg"}]

    fake_datasets.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    config = tmp_path / "hf_config.xml"
    config.write_text(_hf_config_xml(dataset_config="wikitext-103-v1"), encoding="utf-8")
    resolved = resolve_config(parse_config(config))

    rows = run_extraction(resolved)
    assert len(rows) == 1
    assert rows[0].metadata["hf_config"] == "wikitext-103-v1"

def test_huggingface_max_entries_filter_limits_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_datasets = ModuleType("datasets")

    def fake_load_dataset(name: str, *, split: str):
        assert name == "my-dataset"
        assert split == "train"
        return [{"body": "one"}, {"body": "two"}, {"body": "three"}]

    fake_datasets.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    config = tmp_path / "hf_config.xml"
    config.write_text(_hf_config_xml(max_entries="2"), encoding="utf-8")
    resolved = resolve_config(parse_config(config))

    rows = run_extraction(resolved)
    assert len(rows) == 2


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
      <Stage name="StageA" enabled=\"true\" teacher_ref=\"teacher_local\"><TopKLogits k=\"{topk_k}\" /></Stage>
      <Stage name="StageB" enabled=\"false\" teacher_ref=\"teacher_local\"><LongContext max_tokens=\"1024\" /></Stage>
      <Stage name="StageC" enabled=\"false\" teacher_ref=\"teacher_local\"><StructuredOutputs schema=\"json\" /></Stage>
    </Distillation>
  </Dataset>
  <Model>
    <Defaults d_model=\"1024\" n_heads=\"8\" />
    <Patcher name=\"p1\" patch_size=\"128\"><Train steps=\"10\"><Optimizer type=\"adamw\" weight_decay=\"0.0\"><Scheduler type=\"cosine\" start_step=\"0\" end_step=\"10\" /></Optimizer></Train><Transformer /></Patcher>
    <Trunk name=\"t1\" context=\"1024\"><Train steps=\"10\"><Optimizer type=\"adamw\" weight_decay=\"0.0\"><Scheduler type=\"cosine\" start_step=\"0\" end_step=\"10\" /></Optimizer></Train><Transformer /></Trunk>
  </Model>
</Config>
"""


def _hf_config_xml(*, max_entries: str | None = None, dataset_config: str | None = None) -> str:
    return f"""
<Config>
  <Dataset>
    <SourceExtraction>
      <DatasetEntry name="set_hf">
        <Source name="hf" type="huggingface" uri="my-dataset" split="train" text_column="body" {f'config="{dataset_config}"' if dataset_config is not None else ''} />
        {f'<Filter type="max_entries" value="{max_entries}" />' if max_entries is not None else ''}
      </DatasetEntry>
    </SourceExtraction>
    <MixtureBuild target_documents="1" random_seed="7">
      <Group name="g1" percentage="100"><DatasetRef name="set_hf" /></Group>
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
      <Stage name="StageA" enabled="true" teacher_ref="teacher_local"><TopKLogits k="3" /></Stage>
      <Stage name="StageB" enabled="false" teacher_ref="teacher_local"><LongContext max_tokens="1024" /></Stage>
      <Stage name="StageC" enabled="false" teacher_ref="teacher_local"><StructuredOutputs schema="json" /></Stage>
    </Distillation>
  </Dataset>
  <Model>
    <Defaults d_model="1024" n_heads="8" />
    <Patcher name="p1" patch_size="128"><Train steps="10"><Optimizer type="adamw" weight_decay="0.0"><Scheduler type="cosine" start_step="0" end_step="10" /></Optimizer></Train><Transformer /></Patcher>
    <Trunk name="t1" context="1024"><Train steps="10"><Optimizer type="adamw" weight_decay="0.0"><Scheduler type="cosine" start_step="0" end_step="10" /></Optimizer></Train><Transformer /></Trunk>
  </Model>
</Config>
"""
