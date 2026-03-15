from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

import torch


@dataclass(frozen=True)
class RuntimeState:
    text: str
    execution_trace: List[str] = field(default_factory=list)
    moe_metrics: Dict[str, Any] = field(default_factory=dict)
    tensor_shape: tuple[int, int, int] | None = None
    tensor: torch.Tensor | None = None


class RuntimeBlock(Protocol):
    @property
    def block_name(self) -> str: ...

    def run(self, state: RuntimeState) -> RuntimeState: ...


@dataclass(frozen=True)
class RuntimeSchedulerConfig:
    scheduler_type: str
    attributes: Dict[str, str]


@dataclass(frozen=True)
class RuntimeTrainConfig:
    steps: int
    batch_size: int
    save_every: int
    optimizer_type: str
    weight_decay: float
    schedulers: List[RuntimeSchedulerConfig]


@dataclass(frozen=True)
class ExpertRuntime:
    name: str
    blocks: List[RuntimeBlock]

    def run(self, state: RuntimeState) -> RuntimeState:
        current = RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, f"ExpertStart({self.name})"],
            moe_metrics=dict(state.moe_metrics),
            tensor_shape=state.tensor_shape,
            tensor=state.tensor,
        )
        for block in self.blocks:
            current = block.run(current)
        return RuntimeState(
            text=current.text,
            execution_trace=[*current.execution_trace, f"ExpertEnd({self.name})"],
            moe_metrics=dict(current.moe_metrics),
            tensor_shape=current.tensor_shape,
            tensor=current.tensor,
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
            tensor_shape=state.tensor_shape,
            tensor=state.tensor,
        )
        for block in self.blocks:
            current = block.run(current)
        return RuntimeState(
            text=current.text,
            execution_trace=[*current.execution_trace, f"PatcherEnd({self.name})"],
            moe_metrics=dict(current.moe_metrics),
            tensor_shape=current.tensor_shape,
            tensor=current.tensor,
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
            tensor_shape=state.tensor_shape,
            tensor=state.tensor,
        )
        for block in self.blocks:
            current = block.run(current)
        return RuntimeState(
            text=current.text,
            execution_trace=[*current.execution_trace, f"TrunkEnd({self.name})"],
            moe_metrics=dict(current.moe_metrics),
            tensor_shape=current.tensor_shape,
            tensor=current.tensor,
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

    def forward_dummy(self, batch_size: int, seq_len: int, d_model: int) -> RuntimeState:
        tensor = torch.arange(batch_size * seq_len * d_model, dtype=torch.float32).reshape(batch_size, seq_len, d_model)
        state = RuntimeState(text="<dummy>", tensor_shape=(batch_size, seq_len, d_model), tensor=tensor)
        for patcher in self.patchers:
            state = patcher.run(state)
        return self.trunk.run(state)
