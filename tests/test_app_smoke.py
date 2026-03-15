from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_canonical_config_smoke_through_both_apps(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("hello smoke", encoding="utf-8")

    config_path = tmp_path / "smoke_config.xml"
    config_path.write_text(_smoke_config_xml(str(docs_dir / "*.txt")), encoding="utf-8")

    extracted = tmp_path / "extracted"
    mixed = tmp_path / "mixed.jsonl"
    stage_a_dir = tmp_path / "stage_a"

    _run([sys.executable, "-m", "distill", "extract", "--config", str(config_path), "--output-dir", str(extracted)])
    _run([sys.executable, "-m", "distill", "mix", "--config", str(config_path), "--input", str(extracted), "--output", str(mixed)])
    _run([sys.executable, "-m", "distill", "stage-a", "--config", str(config_path), "--input", str(mixed), "--output", str(stage_a_dir)])

    _dir = tmp_path / "model_out"
    model_file = _dir / "model.json"

    distill_dir = tmp_path / "distill_train"
    distill_dir.mkdir()
    (distill_dir / "stage_a.jsonl").write_text(stage_a.read_text(encoding="utf-8"), encoding="utf-8")

    _run([sys.executable, "-m", "train", "build", "--config", str(config_path), "--distill-dir", str(distill_dir), "--output-dir", str(_dir)])
    _run([sys.executable, "-m", "train", "summary", "--model-file", str(model_file)])
    _run([sys.executable, "-m", "train", "smoke", "--config", str(config_path), "--text", "hello app smoke"])

    assert (stage_a_dir / "stage_a.jsonl").exists()
    assert (stage_a_dir / "token_mapping.jsonl").exists()
    assert model_file.exists()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _smoke_config_xml(glob_path: str) -> str:
    return f"""
<Config>
  <Dataset>
    <SourceExtraction>
      <DatasetEntry name="set_local">
        <Source name="local" type="local_text_glob" uri="{glob_path}" />
      </DatasetEntry>
    </SourceExtraction>
    <MixtureBuild target_documents="1" random_seed="7" depletion_policy="rebalance">
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
