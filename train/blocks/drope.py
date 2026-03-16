from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from train.specs import RuntimeState
from .rope import RoPEModule


class DroPEModule(nn.Module):
    def __init__(self, d_model: int, n_heads: int, base: float = 10000.0, scale: float = 1.0, max_seq_len: int = 4096):
        super().__init__()
        self.rope = RoPEModule(d_model, n_heads, base, scale, max_seq_len)
        self.dropped = False

    def drop(self):
        self.dropped = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropped:
            return x
        return self.rope(x)


@dataclass(frozen=True)
class DroPEBlock:
    d_model: int
    n_heads: int
    base: float = 10000.0
    scale: float = 1.0

    @property
    def block_name(self) -> str:
        return "DroPE"

    def build(self) -> DroPEModule:
        return DroPEModule(
            d_model=self.d_model,
            n_heads=self.n_heads,
            base=self.base,
            scale=self.scale,
        )

    def run(self, state: RuntimeState) -> RuntimeState:
        tensor = state.tensor
        return RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, f"DroPE(d_model={self.d_model},n_heads={self.n_heads},base={self.base},scale={self.scale})"],
            moe_metrics=dict(state.moe_metrics),
            tensor_shape=tuple(tensor.shape) if tensor is not None else state.tensor_shape,
            tensor=tensor,
        )
