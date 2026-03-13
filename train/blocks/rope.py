from __future__ import annotations

from dataclasses import dataclass

from train.specs import RuntimeState


@dataclass(frozen=True)
class RoPEBlock:
    d_model: int
    n_heads: int

    @property
    def block_name(self) -> str:
        return "RoPE"

    def run(self, state: RuntimeState) -> RuntimeState:
        return RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, f"RoPE(d_model={self.d_model},n_heads={self.n_heads})"],
            moe_metrics=dict(state.moe_metrics),
            tensor_shape=state.tensor_shape,
        )
