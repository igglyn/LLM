from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

from shared.config.specs import SchedulerSpec


@dataclass(frozen=True)
class RuntimeState:
    text: str
    execution_trace: List[str] = field(default_factory=list)
    moe_metrics: Dict[str, Any] = field(default_factory=dict)


class RuntimeBlock(Protocol):
    @property
    def block_name(self) -> str: ...

    def run(self, state: RuntimeState) -> RuntimeState: ...


@dataclass(frozen=True)
class RuntimeTrainConfig:
    mode: str
    optimizer_type: str
    lr: float
    weight_decay: float
    schedulers: List[SchedulerSpec]


@dataclass(frozen=True)
class ExpertRuntime:
    name: str
    blocks: List[RuntimeBlock]

    def run(self, state: RuntimeState) -> RuntimeState:
        current = RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, f"ExpertStart({self.name})"],
            moe_metrics=dict(state.moe_metrics),
        )
        for block in self.blocks:
            current = block.run(current)
        return RuntimeState(
            text=current.text,
            execution_trace=[*current.execution_trace, f"ExpertEnd({self.name})"],
            moe_metrics=dict(current.moe_metrics),
        )


@dataclass(frozen=True)
class PatcherRuntime:
    name: str
    patch_size: int
    train_config: RuntimeTrainConfig
    blocks: List[RuntimeBlock]

    def run(self, state: RuntimeState) -> RuntimeState:
        current = RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, f"PatcherStart({self.name})"],
            moe_metrics=dict(state.moe_metrics),
        )
        for block in self.blocks:
            current = block.run(current)
        return RuntimeState(
            text=current.text,
            execution_trace=[*current.execution_trace, f"PatcherEnd({self.name})"],
            moe_metrics=dict(current.moe_metrics),
        )


@dataclass(frozen=True)
class TrunkRuntime:
    name: str
    context: int
    train_config: RuntimeTrainConfig
    blocks: List[RuntimeBlock]

    def run(self, state: RuntimeState) -> RuntimeState:
        current = RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, f"TrunkStart({self.name})"],
            moe_metrics=dict(state.moe_metrics),
        )
        for block in self.blocks:
            current = block.run(current)
        return RuntimeState(
            text=current.text,
            execution_trace=[*current.execution_trace, f"TrunkEnd({self.name})"],
            moe_metrics=dict(current.moe_metrics),
        )


@dataclass(frozen=True)
class ModelRuntime:
    patchers: List[PatcherRuntime]
    trunk: TrunkRuntime

    def smoke(self, text: str) -> RuntimeState:
        state = RuntimeState(text=text)
        for patcher in self.patchers:
            state = patcher.run(state)
        return self.trunk.run(state)
