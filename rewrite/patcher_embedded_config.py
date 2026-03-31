"""Embedded rewrite patcher definitions.

These definitions codify rewrite constraints:
- patcher `seq_len` is derived at runtime from token length + patch size,
  not configured as a separate constructor field.
- projection glue (linear adapters) is removed from patcher internals and is
  expected to live in the rewritten main model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PatcherEncoderDefinition:
    component: str = "RewritePatchEncoder"
    patch_aggregation: str = "mean_pool"
    seq_len_mode: str = "derived_from_runtime_token_length"
    contains_projection_glue: bool = False
    projection_glue_owner: str = "main_model_rewrite"


@dataclass(frozen=True)
class PatcherDecoderDefinition:
    component: str = "RewritePatchDecoder"
    token_expansion: str = "repeat_interleave"
    seq_len_mode: str = "derived_from_runtime_token_length"
    contains_projection_glue: bool = False
    projection_glue_owner: str = "main_model_rewrite"


PATCHER_ENCODER_DEFINITION = PatcherEncoderDefinition()
PATCHER_DECODER_DEFINITION = PatcherDecoderDefinition()


EMBEDDED_PATCHER_DEFINITIONS = {
    "encoder": asdict(PATCHER_ENCODER_DEFINITION),
    "decoder": asdict(PATCHER_DECODER_DEFINITION),
}
