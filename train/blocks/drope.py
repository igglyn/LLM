from __future__ import annotations

from dataclasses import dataclass

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
        return RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, f"DRope(d_model={self.d_model},n_heads={self.n_heads},base={self.base},scale={self.scale})"],
            moe_metrics=dict(state.moe_metrics),
            tensor_shape=state.tensor_shape,
        )
