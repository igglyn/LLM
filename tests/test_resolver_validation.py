from __future__ import annotations

from pathlib import Path

import pytest

from shared.config import parse_config, resolve_config
from shared.config.resolver import ConfigResolutionError


def test_resolver_smoke_on_canonical_config() -> None:
    config = parse_config(Path("examples/config.example.xml"))
    resolved = resolve_config(config)

    assert resolved.model.patchers[0].transformer_blocks[0].d_model == 4096
    assert resolved.model.patchers[0].transformer_blocks[0].n_heads == 32
    assert resolved.model.trunk is not None
    assert resolved.model.trunk.transformer_blocks[0].d_model == 4096


def test_resolver_rejects_incompatible_transformer_heads(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_heads.xml"
    config_path.write_text(
        """
<Config>
  <Dataset>
    <SourceExtraction>
      <DatasetEntry name="set_local">
        <Source name="local" type="local_text_glob" uri="/tmp/*.txt" />
      </DatasetEntry>
    </SourceExtraction>
    <MixtureBuild><Group name="g" percentage="100"><DatasetRef name="set_local" /></Group></MixtureBuild>
    <Distillation>
      <Teachers>
        <Teacher name="t"><Backend type="dummy_local"><ModelRef name_or_path="dummy" /><Execution device="cpu" precision="fp32" /></Backend></Teacher>
      </Teachers>
      <Stage name="StageA" enabled="true" teacher_ref="t"><TopKLogits k="8" /></Stage>
    </Distillation>
  </Dataset>
  <Model>
    <Defaults d_model="130" n_heads="8" />
    <Patcher name="p1" patch_size="64">
      <Train steps="5"><Optimizer type="adamw" weight_decay="0.0"><Scheduler type="cosine" start_step="0" end_step="5" /></Optimizer></Train>
      <Transformer />
    </Patcher>
    <Trunk name="t1" context="128">
      <Train steps="5"><Optimizer type="adamw" weight_decay="0.0"><Scheduler type="cosine" start_step="0" end_step="5" /></Optimizer></Train>
      <Transformer />
    </Trunk>
  </Model>
</Config>
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigResolutionError, match="divisible by n_heads"):
        resolve_config(parse_config(config_path))
