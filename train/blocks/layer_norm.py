from __future__ import annotations

from dataclasses import dataclass

import torch

from train.specs import RuntimeState


@dataclass(frozen=True)
class LayerNormBlock:
    eps: float = 1e-5

    @property
    def block_name(self) -> str:
        return "LayerNorm"

    def run(self, state: RuntimeState) -> RuntimeState:
        tensor = state.tensor
        if tensor is not None:
            mean = tensor.mean(dim=-1, keepdim=True)
            variance = ((tensor - mean) ** 2).mean(dim=-1, keepdim=True)
            tensor = (tensor - mean) / torch.sqrt(variance + self.eps)

        return RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, f"LayerNorm(eps={self.eps})"],
            moe_metrics=dict(state.moe_metrics),
            tensor_shape=tuple(tensor.shape) if tensor is not None else state.tensor_shape,
            tensor=tensor,
        )
