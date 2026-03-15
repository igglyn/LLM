from __future__ import annotations

from pathlib import Path

import torch

from shared.config import parse_config, resolve_config
from train.blocks import MixOfExpertsBlock
from train.builder import build_model_runtime
from train.metrics import summarize_model_runtime


def test_resolved_defaults_and_overrides(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    resolved = resolve_config(parse_config(config_path))

    patcher = resolved.model.patchers[0]
    assert patcher.transformer_blocks[0].d_model == 1024
    assert patcher.transformer_blocks[0].n_heads == 8
    assert patcher.transformer_blocks[1].d_model == 1536
    assert patcher.transformer_blocks[1].n_heads == 12
    assert patcher.rope_blocks[0].d_model == 2048
    assert patcher.rope_blocks[0].n_heads == 16
    assert patcher.rope_blocks[0].base == 16000.0
    assert patcher.rope_blocks[0].scale == 0.5
    assert resolved.model.trunk.rope_blocks[0].base == 12000.0
    assert resolved.model.trunk.rope_blocks[0].scale == 0.75
    assert resolved.model.trunk.pos_embedding_blocks[0].attributes["type"] == "learned"
    assert resolved.model.trunk.drope_blocks[0].base == 8000.0
    assert resolved.model.trunk.drope_blocks[0].scale == 1.25


def test_patcher_forward_order(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path)
    trace = runtime.smoke("hello").execution_trace
    patcher_trace = [t for t in trace if t.startswith(("Patcher", "RoPE", "Transformer", "PosEmbedding", "LayerNorm"))]
    assert patcher_trace[:6] == [
        "PatcherStart(p1)",
        "RoPE(d_model=2048,n_heads=16,base=16000.0,scale=0.5)",
        "Transformer(d_model=1024,n_heads=8)",
        "PosEmbedding",
        "Transformer(d_model=1536,n_heads=12)",
        "LayerNorm(eps=1e-05)",
    ]


def test_trunk_forward_order(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path)
    trace = runtime.smoke("hello").execution_trace
    trunk_start = trace.index("TrunkStart(t1)")
    assert trace[trunk_start : trunk_start + 5] == [
        "TrunkStart(t1)",
        "RoPE(d_model=1024,n_heads=8,base=12000.0,scale=0.75)",
        "PosEmbedding",
        "DRope(d_model=1024,n_heads=8,base=8000.0,scale=1.25)",
        "MoERoute(moe1->expert_b)",
    ]


def test_mix_of_experts_build(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path)
    moe = next(b for b in runtime.trunk.blocks if isinstance(b, MixOfExpertsBlock))

    assert len(moe.experts) == 2
    assert [e.name for e in moe.experts] == ["expert_a", "expert_b"]


def test_smoke_forward_dummy(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path)
    out = runtime.forward_dummy(batch_size=2, seq_len=4, d_model=1024)
    initial = torch.arange(2 * 4 * 1024, dtype=torch.float32).reshape(2, 4, 1024)

    assert out.tensor_shape == (2, 4, 1024)
    assert out.tensor is not None
    assert not torch.allclose(out.tensor, initial)
    assert "TrunkEnd(t1)" in out.execution_trace
    assert out.moe_metrics["moe1"]["route_calls"] == 1






def test_vocab_embedding_runtime_trace(tmp_path: Path) -> None:
    path = tmp_path / "vocab_runtime.xml"
    path.write_text(
        """
<Config>
  <Dataset>
    <SourceExtraction />
    <MixtureBuild />
    <Distillation>
      <Teachers>
        <Teacher name="t">
          <Backend type="x"><ModelRef name_or_path="m" /><Execution device="cpu" precision="fp32" /></Backend>
        </Teacher>
      </Teachers>
      <Stage name="StageA" enabled="true" teacher_ref="t"><TopKLogits k="16" /></Stage>
      <Stage name="StageB" enabled="false" teacher_ref="t"><LongContext max_tokens="256" /></Stage>
      <Stage name="StageC" enabled="false" teacher_ref="t"><StructuredOutputs schema="json" /></Stage>
    </Distillation>
  </Dataset>
  <Model>
    <Defaults d_model="128" n_heads="8" />
    <Patcher name="p1" patch_size="64">
      <Train steps="10"><Optimizer type="adamw" weight_decay="0.0"><Scheduler type="cosine" start_step="0" end_step="10" /></Optimizer></Train>
      <VocabEmbedding vocab_size="50000" />
      <Transformer />
    </Patcher>
    <Trunk name="t1" context="128">
      <Train steps="10"><Optimizer type="adamw" weight_decay="0.0"><Scheduler type="cosine" start_step="0" end_step="10" /></Optimizer></Train>
      <Transformer />
    </Trunk>
  </Model>
</Config>
        """.strip(),
        encoding="utf-8",
    )

    runtime = build_model_runtime(resolve_config(parse_config(path)))
    trace = runtime.smoke("hello").execution_trace

    assert "VocabEmbedding(vocab_size=50000)" in trace

def test_scheduler_ordering_preserved_and_summarized(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path)
    summary = summarize_model_runtime(runtime)
    schedulers = summary["patchers"][0]["train"]["schedulers"]

    assert [s["type"] for s in schedulers] == ["warmup", "cosine", "loss_threshold"]


def _build_runtime(tmp_path: Path):
    config_path = _write_config(tmp_path)
    return build_model_runtime(resolve_config(parse_config(config_path)))


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "train_model.xml"
    path.write_text(
        """
<Config>
  <Dataset>
    <SourceExtraction>
      <DatasetEntry name="set_local">
        <Source name="local" type="local_text_glob" uri="/tmp/*.txt" />
      </DatasetEntry>
    </SourceExtraction>
    <MixtureBuild target_documents="1" random_seed="7" depletion_policy="rebalance">
      <Group name="g1" percentage="100"><DatasetRef name="set_local" /></Group>
    </MixtureBuild>
    <Distillation>
      <Teachers>
        <Teacher name="teacher_local">
          <Backend type="dummy_local"><ModelRef name_or_path="dummy" /><Execution device="cpu" precision="fp32" /></Backend>
        </Teacher>
      </Teachers>
      <Stage name="StageA" enabled="true" teacher_ref="teacher_local"><TopKLogits k="3" /></Stage>
      <Stage name="StageB" enabled="false" teacher_ref="teacher_local"><LongContext max_tokens="256" /></Stage>
      <Stage name="StageC" enabled="false" teacher_ref="teacher_local"><StructuredOutputs schema="json" /></Stage>
    </Distillation>
  </Dataset>
  <Model>
    <Defaults d_model="1024" n_heads="8" />
    <Patcher name="p1" patch_size="64" d_model="2048" n_heads="16">
      <Train steps="100">
        <Optimizer type="adamw" weight_decay="0.0">
          <Scheduler type="warmup" start_step="0" end_step="10" />
          <Scheduler type="cosine" start_step="10" end_step="100" />
          <Scheduler type="loss_threshold" start_step="0" end_step="100" threshold="1.0" />
        </Optimizer>
      </Train>
      <RoPE base="16000" scale="0.5" />
      <Transformer d_model="1024" n_heads="8" />
      <PosEmbedding type="learned" />
      <Transformer d_model="1536" n_heads="12" />
      <LayerNorm />
    </Patcher>
    <Trunk name="t1" context="2048">
      <Train steps="100"><Optimizer type="adamw" weight_decay="0.1"><Scheduler type="cosine" start_step="0" end_step="100" /></Optimizer></Train>
      <RoPE base="12000" scale="0.75" />
      <PosEmbedding type="learned" />
      <DRope base="8000" scale="1.25" />
      <MixOfExperts name="moe1">
        <Expert name="expert_a"><Transformer /></Expert>
        <Expert name="expert_b"><Transformer d_model="1152" n_heads="9" /></Expert>
      </MixOfExperts>
      <Transformer />
    </Trunk>
  </Model>
</Config>
        """.strip(),
        encoding="utf-8",
    )
    return path
