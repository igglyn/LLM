from __future__ import annotations

from dataclasses import dataclass

from train.specs import ExpertRuntime, RuntimeState


@dataclass(frozen=True)
class MixOfExpertsBlock:
    name: str
    experts: list[ExpertRuntime]

    @property
    def block_name(self) -> str:
        return "MixOfExperts"

    def run(self, state: RuntimeState) -> RuntimeState:
        if not self.experts:
            return state

        route_index = len(state.text) % len(self.experts)
        selected = self.experts[route_index]
        metrics = dict(state.moe_metrics)
        usage = dict(metrics.get(self.name, {}))
        counts = dict(usage.get("expert_counts", {}))
        counts[selected.name] = counts.get(selected.name, 0) + 1
        usage.update(
            {
                "selected_expert": selected.name,
                "selected_index": route_index,
                "num_experts": len(self.experts),
                "expert_counts": counts,
                "route_calls": usage.get("route_calls", 0) + 1,
            }
        )
        metrics[self.name] = usage

        routed = RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, f"MoERoute({self.name}->{selected.name})"],
            moe_metrics=metrics,
            tensor_shape=state.tensor_shape,
        )
        return selected.run(routed)
