from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from shared.config.specs import (
    PosEmbeddingBlockSpec,
    ResolvedConfigSpec,
    ResolvedMixOfExpertsSpec,
    ResolvedPatcherSpec,
    ResolvedTrunkSpec,
    SchedulerSpec,
    TrainSpec,
)


@dataclass(frozen=True)
class RuntimeState:
    text: str
    execution_trace: List[str] = field(default_factory=list)
    moe_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeTrainConfig:
    mode: str
    optimizer_type: str
    lr: float
    weight_decay: float
    schedulers: List[SchedulerSpec]


class RuntimeBlock:
    block_name: str

    def run(self, state: RuntimeState) -> RuntimeState:
        raise NotImplementedError


@dataclass(frozen=True)
class RoPEBlock(RuntimeBlock):
    d_model: int | None
    n_heads: int | None

    @property
    def block_name(self) -> str:
        return "RoPE"

    def run(self, state: RuntimeState) -> RuntimeState:
        trace = list(state.execution_trace)
        trace.append(f"RoPE(d_model={self.d_model},n_heads={self.n_heads})")
        return RuntimeState(text=state.text, execution_trace=trace, moe_metrics=dict(state.moe_metrics))


@dataclass(frozen=True)
class DRopeBlock(RuntimeBlock):
    d_model: int | None
    n_heads: int | None

    @property
    def block_name(self) -> str:
        return "DRope"

    def run(self, state: RuntimeState) -> RuntimeState:
        trace = list(state.execution_trace)
        trace.append(f"DRope(d_model={self.d_model},n_heads={self.n_heads})")
        return RuntimeState(text=state.text, execution_trace=trace, moe_metrics=dict(state.moe_metrics))


@dataclass(frozen=True)
class PosEmbeddingBlock(RuntimeBlock):
    attributes: Dict[str, str]

    @property
    def block_name(self) -> str:
        return "PosEmbedding"

    def run(self, state: RuntimeState) -> RuntimeState:
        trace = list(state.execution_trace)
        trace.append("PosEmbedding")
        return RuntimeState(text=state.text, execution_trace=trace, moe_metrics=dict(state.moe_metrics))


@dataclass(frozen=True)
class TransformerBlock(RuntimeBlock):
    d_model: int
    n_heads: int

    @property
    def block_name(self) -> str:
        return "Transformer"

    def run(self, state: RuntimeState) -> RuntimeState:
        trace = list(state.execution_trace)
        trace.append(f"Transformer(d_model={self.d_model},n_heads={self.n_heads})")
        return RuntimeState(text=state.text, execution_trace=trace, moe_metrics=dict(state.moe_metrics))


@dataclass(frozen=True)
class ExpertRuntime:
    name: str
    blocks: List[RuntimeBlock]

    def run(self, state: RuntimeState) -> RuntimeState:
        current = state
        trace = list(current.execution_trace)
        trace.append(f"ExpertStart({self.name})")
        current = RuntimeState(text=current.text, execution_trace=trace, moe_metrics=dict(current.moe_metrics))
        for block in self.blocks:
            current = block.run(current)
        trace = list(current.execution_trace)
        trace.append(f"ExpertEnd({self.name})")
        return RuntimeState(text=current.text, execution_trace=trace, moe_metrics=dict(current.moe_metrics))


@dataclass(frozen=True)
class MixOfExpertsBlock(RuntimeBlock):
    name: str
    experts: List[ExpertRuntime]

    @property
    def block_name(self) -> str:
        return "MixOfExperts"

    def run(self, state: RuntimeState) -> RuntimeState:
        if not self.experts:
            return state
        route_index = len(state.text) % len(self.experts)
        chosen = self.experts[route_index]
        metrics = dict(state.moe_metrics)
        metrics[self.name] = {
            "selected_expert": chosen.name,
            "selected_index": route_index,
            "num_experts": len(self.experts),
        }
        trace = list(state.execution_trace)
        trace.append(f"MoERoute({self.name}->{chosen.name})")
        routed_state = RuntimeState(text=state.text, execution_trace=trace, moe_metrics=metrics)
        return chosen.run(routed_state)


@dataclass(frozen=True)
class PatcherRuntime:
    name: str
    patch_size: int
    train_config: RuntimeTrainConfig
    blocks: List[RuntimeBlock]

    def run(self, state: RuntimeState) -> RuntimeState:
        current = state
        trace = list(current.execution_trace)
        trace.append(f"PatcherStart({self.name})")
        current = RuntimeState(text=current.text, execution_trace=trace, moe_metrics=dict(current.moe_metrics))
        for block in self.blocks:
            current = block.run(current)
        trace = list(current.execution_trace)
        trace.append(f"PatcherEnd({self.name})")
        return RuntimeState(text=current.text, execution_trace=trace, moe_metrics=dict(current.moe_metrics))


@dataclass(frozen=True)
class TrunkRuntime:
    name: str
    context: int
    train_config: RuntimeTrainConfig
    blocks: List[RuntimeBlock]

    def run(self, state: RuntimeState) -> RuntimeState:
        current = state
        trace = list(current.execution_trace)
        trace.append(f"TrunkStart({self.name})")
        current = RuntimeState(text=current.text, execution_trace=trace, moe_metrics=dict(current.moe_metrics))
        for block in self.blocks:
            current = block.run(current)
        trace = list(current.execution_trace)
        trace.append(f"TrunkEnd({self.name})")
        return RuntimeState(text=current.text, execution_trace=trace, moe_metrics=dict(current.moe_metrics))


@dataclass(frozen=True)
class ModelRuntime:
    patchers: List[PatcherRuntime]
    trunk: TrunkRuntime

    def smoke(self, text: str) -> RuntimeState:
        state = RuntimeState(text=text)
        for patcher in self.patchers:
            state = patcher.run(state)
        state = self.trunk.run(state)
        return state


def build_model_runtime(resolved_config: ResolvedConfigSpec) -> ModelRuntime:
    if resolved_config.model.trunk is None:
        raise ValueError("Resolved config must include a trunk.")

    patchers = [_build_patcher_runtime(patcher) for patcher in resolved_config.model.patchers]
    trunk = _build_trunk_runtime(resolved_config.model.trunk)
    return ModelRuntime(patchers=patchers, trunk=trunk)


def _build_patcher_runtime(patcher: ResolvedPatcherSpec) -> PatcherRuntime:
    blocks = _compose_patcher_blocks(patcher)
    return PatcherRuntime(
        name=patcher.name,
        patch_size=patcher.patch_size,
        train_config=_to_runtime_train_config(patcher.train),
        blocks=blocks,
    )


def _build_trunk_runtime(trunk: ResolvedTrunkSpec) -> TrunkRuntime:
    blocks = _compose_trunk_blocks(trunk)
    return TrunkRuntime(
        name=trunk.name,
        context=trunk.context,
        train_config=_to_runtime_train_config(trunk.train),
        blocks=blocks,
    )


def _compose_patcher_blocks(patcher: ResolvedPatcherSpec) -> List[RuntimeBlock]:
    # Extension hook: add new patcher block mappings here.
    rope_index = 0
    transformer_index = 0
    pos_index = 0
    blocks: List[RuntimeBlock] = []
    for block_name in patcher.block_order:
        if block_name == "RoPE":
            block = patcher.rope_blocks[rope_index]
            rope_index += 1
            blocks.append(RoPEBlock(d_model=block.d_model, n_heads=block.n_heads))
        elif block_name == "Transformer":
            block = patcher.transformer_blocks[transformer_index]
            transformer_index += 1
            blocks.append(TransformerBlock(d_model=block.d_model, n_heads=block.n_heads))
        elif block_name == "PosEmbedding":
            block = patcher.pos_embedding_blocks[pos_index]
            pos_index += 1
            blocks.append(PosEmbeddingBlock(attributes=dict(block.attributes)))
        else:
            raise ValueError(f"Unsupported patcher block '{block_name}'.")
    return blocks


def _compose_trunk_blocks(trunk: ResolvedTrunkSpec) -> List[RuntimeBlock]:
    # Extension hook: add new trunk block mappings here.
    drope_index = 0
    transformer_index = 0
    moe_index = 0
    blocks: List[RuntimeBlock] = []
    for block_name in trunk.block_order:
        if block_name == "DRope":
            block = trunk.drope_blocks[drope_index]
            drope_index += 1
            blocks.append(DRopeBlock(d_model=block.d_model, n_heads=block.n_heads))
        elif block_name == "Transformer":
            block = trunk.transformer_blocks[transformer_index]
            transformer_index += 1
            blocks.append(TransformerBlock(d_model=block.d_model, n_heads=block.n_heads))
        elif block_name == "MixOfExperts":
            block = trunk.mix_of_experts_blocks[moe_index]
            moe_index += 1
            blocks.append(_build_moe_runtime(block))
        else:
            raise ValueError(f"Unsupported trunk block '{block_name}'.")
    return blocks


def _build_moe_runtime(moe: ResolvedMixOfExpertsSpec) -> MixOfExpertsBlock:
    experts: List[ExpertRuntime] = []
    for expert in moe.experts:
        expert_blocks: List[RuntimeBlock] = []
        transformer_index = 0
        for block_name in expert.block_order:
            if block_name != "Transformer":
                raise ValueError(f"Unsupported expert block '{block_name}'.")
            block = expert.transformer_blocks[transformer_index]
            transformer_index += 1
            expert_blocks.append(TransformerBlock(d_model=block.d_model, n_heads=block.n_heads))
        experts.append(ExpertRuntime(name=expert.name, blocks=expert_blocks))
    return MixOfExpertsBlock(name=moe.name, experts=experts)


def _to_runtime_train_config(train_spec: TrainSpec) -> RuntimeTrainConfig:
    # Extension hook: wire scheduler semantics/executors without changing parser contracts.
    return RuntimeTrainConfig(
        mode=train_spec.mode,
        optimizer_type=train_spec.optimizer.optimizer_type,
        lr=train_spec.optimizer.lr,
        weight_decay=train_spec.optimizer.weight_decay,
        schedulers=list(train_spec.optimizer.schedulers),
    )


def summarize_model_runtime(model_runtime: ModelRuntime) -> dict[str, Any]:
    return {
        "patchers": [
            {
                "name": patcher.name,
                "patch_size": patcher.patch_size,
                "train_mode": patcher.train_config.mode,
                "optimizer": patcher.train_config.optimizer_type,
                "scheduler_count": len(patcher.train_config.schedulers),
                "block_order": [block.block_name for block in patcher.blocks],
            }
            for patcher in model_runtime.patchers
        ],
        "trunk": {
            "name": model_runtime.trunk.name,
            "context": model_runtime.trunk.context,
            "train_mode": model_runtime.trunk.train_config.mode,
            "optimizer": model_runtime.trunk.train_config.optimizer_type,
            "scheduler_count": len(model_runtime.trunk.train_config.schedulers),
            "block_order": [block.block_name for block in model_runtime.trunk.blocks],
            "moe_blocks": [block.name for block in model_runtime.trunk.blocks if isinstance(block, MixOfExpertsBlock)],
        },
    }
