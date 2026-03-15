from __future__ import annotations

from dataclasses import dataclass

import torch

from train.specs import RuntimeState


@dataclass(frozen=True)
class DRopeBlock:
    d_model: int
    n_heads: int
    base: float
    scale: float

    @property
    def block_name(self) -> str:
        return "DRope"

    def run(self, state: RuntimeState) -> RuntimeState:
        tensor = state.tensor
        if tensor is not None:
            decay = 1.0 / (1.0 + (self.scale / self.base))
            positions = torch.arange(tensor.shape[1], device=tensor.device, dtype=tensor.dtype)
            mask = torch.exp(-positions * (1.0 - decay)).view(1, -1, 1)
            tensor = tensor * mask

        return RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, f"DRope(d_model={self.d_model},n_heads={self.n_heads},base={self.base},scale={self.scale})"],
            moe_metrics=dict(state.moe_metrics),
            tensor_shape=tuple(tensor.shape) if tensor is not None else state.tensor_shape,
            tensor=tensor,
        )
