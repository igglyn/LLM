from __future__ import annotations

from dataclasses import dataclass

from train.specs import RuntimeState


@dataclass(frozen=True)
class DRopeBlock:
    d_model: int | None
    n_heads: int | None

    @property
    def block_name(self) -> str:
        return "DRope"

    def run(self, state: RuntimeState) -> RuntimeState:
        return RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, f"DRope(d_model={self.d_model},n_heads={self.n_heads})"],
            moe_metrics=dict(state.moe_metrics),
        )
