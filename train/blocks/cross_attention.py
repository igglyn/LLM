from __future__ import annotations

from dataclasses import dataclass

import torch

from train.specs import RuntimeState


@dataclass(frozen=True)
class CrossAttentionBlock:
    d_model: int
    n_heads: int

    @property
    def block_name(self) -> str:
        return "CrossAttention"

    def run(self, state: RuntimeState) -> RuntimeState:
        tensor = state.tensor
        if tensor is not None:
            head_dim = max(1, self.d_model // max(1, self.n_heads))
            context = torch.flip(tensor, dims=(1,))
            tensor = tensor + context * (1.0 / float(head_dim))

        return RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, f"CrossAttention(d_model={self.d_model},n_heads={self.n_heads})"],
            moe_metrics=dict(state.moe_metrics),
            tensor_shape=tuple(tensor.shape) if tensor is not None else state.tensor_shape,
            tensor=tensor,
        )
