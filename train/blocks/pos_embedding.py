from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from train.specs import RuntimeState


@dataclass(frozen=True)
class PosEmbeddingBlock:
    attributes: Dict[str, str]

    @property
    def block_name(self) -> str:
        return "PosEmbedding"

    def run(self, state: RuntimeState) -> RuntimeState:
        return RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, "PosEmbedding"],
            moe_metrics=dict(state.moe_metrics),
            tensor_shape=state.tensor_shape,
        )
