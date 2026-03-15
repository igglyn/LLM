from __future__ import annotations

from dataclasses import dataclass

import torch

from train.specs import RuntimeState


@dataclass(frozen=True)
class RoPEBlock:
    d_model: int
    n_heads: int
    base: float
    scale: float

    @property
    def block_name(self) -> str:
        return "RoPE"

    def run(self, state: RuntimeState) -> RuntimeState:
        tensor = state.tensor
        if tensor is not None and tensor.shape[-1] >= 2:
            even = tensor[..., 0::2]
            odd = tensor[..., 1::2]
            angle = (1.0 / self.base) * self.scale
            cos = torch.cos(torch.tensor(angle, dtype=tensor.dtype, device=tensor.device))
            sin = torch.sin(torch.tensor(angle, dtype=tensor.dtype, device=tensor.device))
            rotated_even = even * cos - odd * sin
            rotated_odd = even * sin + odd * cos
            tensor = tensor.clone()
            tensor[..., 0::2] = rotated_even
            tensor[..., 1::2] = rotated_odd

        return RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, f"RoPE(d_model={self.d_model},n_heads={self.n_heads},base={self.base},scale={self.scale})"],
            moe_metrics=dict(state.moe_metrics),
            tensor_shape=tuple(tensor.shape) if tensor is not None else state.tensor_shape,
            tensor=tensor,
        )
