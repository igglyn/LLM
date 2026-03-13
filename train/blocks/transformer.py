from __future__ import annotations

from dataclasses import dataclass

from train.specs import RuntimeState


@dataclass(frozen=True)
class TransformerBlock:
    d_model: int
    n_heads: int

    @property
    def block_name(self) -> str:
        return "Transformer"

    def run(self, state: RuntimeState) -> RuntimeState:
        return RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, f"Transformer(d_model={self.d_model},n_heads={self.n_heads})"],
            moe_metrics=dict(state.moe_metrics),
        )
