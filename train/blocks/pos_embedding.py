from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from train.specs import RuntimeState


@dataclass(frozen=True)
class PosEmbeddingBlock:
    attributes: Dict[str, str]

    @property
    def block_name(self) -> str:
        return "PosEmbedding"

    def run(self, state: RuntimeState) -> RuntimeState:
        tensor = state.tensor
        if tensor is not None:
            seq_len = tensor.shape[1]
            position = torch.arange(seq_len, dtype=tensor.dtype, device=tensor.device).view(1, seq_len, 1)
            tensor = tensor + position

        return RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, "PosEmbedding"],
            moe_metrics=dict(state.moe_metrics),
            tensor_shape=tuple(tensor.shape) if tensor is not None else state.tensor_shape,
            tensor=tensor,
        )
