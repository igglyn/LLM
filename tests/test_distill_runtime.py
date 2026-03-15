from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from distill.runtime.io_utils import read_jsonl
from distill.runtime.mixture_build import run_mixture_build
from distill.runtime.pipeline import run_extract, run_mix, run_stage_a_command
from distill.runtime.source_extraction import run_source_extraction
from distill.runtime.teacher_runtime import HuggingFaceBackend
from distill.runtime.types import ExtractedDocument
from shared.config import parse_config


def test_local_source_extraction_with_filter_and_split_mapping(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("hello", encoding="utf-8")
    (docs_dir / "b.txt").write_text("x", encoding="utf-8")

    config_path = tmp_path / "config.xml"
    config_path.write_text(_runtime_config_xml(str(docs_dir / "*.txt")), encoding="utf-8")
    config = parse_config(config_path)

    docs = run_source_extraction(config)
    assert len(docs) == 1
    assert docs[0].split == "train_mapped"
    assert docs[0].metadata["path"].endswith("a.txt")




def test_local_source_extraction_max_entries_filter_limits_output(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("aaaa", encoding="utf-8")
    (docs_dir / "b.txt").write_text("bbbb", encoding="utf-8")
    (docs_dir / "c.txt").write_text("cccc", encoding="utf-8")

    config_path = tmp_path / "config.xml"
    config_path.write_text(_runtime_config_xml(str(docs_dir / "*.txt"), min_bytes="1", max_entries="2"), encoding="utf-8")
    config = parse_config(config_path)

    docs = run_source_extraction(config)
    assert len(docs) == 2


def test_mixture_grouping_and_rebalance_selection(tmp_path: Path) -> None:
    config_path = tmp_path / "mix_config.xml"
    config_path.write_text(_mixture_config_xml(), encoding="utf-8")
    config = parse_config(config_path)

    docs = [
        ExtractedDocument("d1", "set_a", "train", "aaa", 3),
        ExtractedDocument("d2", "set_a", "train", "bbb", 3),
        ExtractedDocument("d3", "set_b", "train", "ccc", 3),
    ]

    mixed = run_mixture_build(config, docs)
    assert len(mixed) == 3
    assert {row.dataset_entry for row in mixed} == {"set_a", "set_b"}


def test_stage_a_end_to_end_smoke_with_dummy_teacher(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("alpha beta gamma", encoding="utf-8")

    config_path = tmp_path / "config.xml"
    config_path.write_text(_runtime_config_xml(str(docs_dir / "*.txt"), min_bytes="1"), encoding="utf-8")

    extracted_path = tmp_path / "extracted.jsonl"
    mixed_path = tmp_path / "mixed.jsonl"
    stage_a_path = tmp_path / "stage_a.jsonl"

    assert run_extract(config_path, extracted_path) == 1
    assert run_mix(config_path, extracted_path, mixed_path) == 1
    assert run_stage_a_command(config_path, mixed_path, stage_a_path) == 1

    row = read_jsonl(stage_a_path)[0]
    assert {"record_id", "doc_id", "prompt_text", "target_text", "top_k_predictions", "metadata"} <= set(row)




def test_runtime_huggingface_source_with_config_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_datasets = ModuleType("datasets")

    def fake_load_dataset(name: str, config: str, *, split: str):
        assert name == "my-dataset"
        assert config == "wikitext-2-raw-v1"
        assert split == "train"
        return [{"body": "runtime hf"}]

    fake_datasets.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    config_path = tmp_path / "config.xml"
    config_path.write_text(_runtime_hf_config_xml(dataset_config="wikitext-2-raw-v1"), encoding="utf-8")
    config = parse_config(config_path)

    docs = run_source_extraction(config)
    assert len(docs) == 1
    assert docs[0].metadata["hf_config"] == "wikitext-2-raw-v1"

def test_hf_backend_interface_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_transformers = ModuleType("transformers")
    fake_torch = ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *_):
            return False

    class _TensorRow:
        def __init__(self, values):
            self._values = values

        def to(self, _device):
            return self

        def tolist(self):
            return list(self._values)

    class _Tensor2D:
        def __init__(self, values):
            self._values = values

        def __getitem__(self, idx):
            return _TensorRow(self._values[idx])

    class _Logits:
        def __getitem__(self, _idx):
            return self

    class _ModelOutput:
        logits = _Logits()

    class _FakeTokenizer:
        @staticmethod
        def from_pretrained(_name):
            return _FakeTokenizer()

        def __call__(self, _text, return_tensors="pt"):
            assert return_tensors == "pt"
            return {"input_ids": _TensorRow([1, 2, 3])}

        def convert_ids_to_tokens(self, ids):
            return [f"tok_{i}" for i in ids]

    class _FakeModel:
        @staticmethod
        def from_pretrained(_name, **_kwargs):
            return _FakeModel()

        def to(self, _device):
            return self

        def eval(self):
            return None

        def __call__(self, **_encoded):
            return _ModelOutput()

    def _softmax(_logits, dim=-1):
        assert dim == -1
        return "probs"

    def _topk(_probs, k):
        return _Tensor2D([[0.9, 0.1][:k]]), _Tensor2D([[42, 7][:k]])

    fake_torch.no_grad = _NoGrad
    fake_torch.softmax = _softmax
    fake_torch.topk = _topk
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"

    fake_transformers.AutoTokenizer = _FakeTokenizer
    fake_transformers.AutoModelForCausalLM = _FakeModel

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    backend = HuggingFaceBackend("dummy", {"device": "cpu", "precision": "fp32"})
    preds = backend.top_k_next_tokens("hello", 2)
    assert len(preds) == 2
    assert {"token", "score", "rank"} <= preds[0].__dict__.keys()


def test_jsonl_schema_validation(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("hello world", encoding="utf-8")
    config_path = tmp_path / "config.xml"
    config_path.write_text(_runtime_config_xml(str(docs_dir / "*.txt"), min_bytes="1"), encoding="utf-8")

    output_path = tmp_path / "stage_a.jsonl"
    completed = subprocess.run(
        [sys.executable, "-m", "distill", "stage-a", "--config", str(config_path), "--output", str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "stage-a completed" in completed.stdout

    with output_path.open("r", encoding="utf-8") as handle:
        row = json.loads(handle.readline())

    assert isinstance(row["record_id"], str)
    assert isinstance(row["doc_id"], str)
    assert isinstance(row["prompt_text"], str)
    assert isinstance(row["target_text"], str)
    assert isinstance(row["top_k_predictions"], list)
    assert isinstance(row["metadata"], dict)


def _runtime_config_xml(glob_path: str, *, min_bytes: str = "4", max_entries: str | None = None) -> str:
    return f"""
<Config>
  <Dataset>
    <SourceExtraction>
      <DatasetEntry name="set_local">
        <Source name="local" type="local_text_glob" uri="{glob_path}" split="train" />
        <SplitMapping>
          <Map from="train" to="train_mapped" />
        </SplitMapping>
        <Filter type="min_bytes" value="{min_bytes}" />
        {f'<Filter type="max_entries" value="{max_entries}" />' if max_entries is not None else ''}
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


def _mixture_config_xml() -> str:
    return """
<Config>
  <Dataset>
    <SourceExtraction />
    <MixtureBuild target_documents="5" random_seed="0" min_bytes="1" max_bytes="9999" depletion_policy="rebalance">
      <Group name="ga" percentage="50"><DatasetRef name="set_a" /></Group>
      <Group name="gb" percentage="50"><DatasetRef name="set_b" /></Group>
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
      <Stage name="StageA" enabled="false" teacher_ref="teacher_local"><TopKLogits k="3" /></Stage>
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


def _runtime_hf_config_xml(*, dataset_config: str | None = None) -> str:
    return f"""
<Config>
  <Dataset>
    <SourceExtraction>
      <DatasetEntry name="set_hf">
        <Source name="hf" type="huggingface" uri="my-dataset" split="train" text_column="body" {f'config="{dataset_config}"' if dataset_config is not None else ''} />
      </DatasetEntry>
    </SourceExtraction>
    <MixtureBuild target_documents="1" random_seed="7" min_bytes="1" depletion_policy="rebalance">
      <Group name="g1" percentage="100">
        <DatasetRef name="set_hf" />
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
