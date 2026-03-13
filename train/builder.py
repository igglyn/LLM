from __future__ import annotations

from shared.config.specs import ResolvedConfigSpec, ResolvedMixOfExpertsSpec, ResolvedPatcherSpec, ResolvedTrunkSpec
from train.blocks import DRopeBlock, MixOfExpertsBlock, PosEmbeddingBlock, RoPEBlock, TransformerBlock
from train.optim import runtime_train_config_from_spec
from train.specs import ExpertRuntime, ModelRuntime, PatcherRuntime, RuntimeBlock, TrunkRuntime


def build_model_runtime(resolved_config: ResolvedConfigSpec) -> ModelRuntime:
    if resolved_config.model.trunk is None:
        raise ValueError("Resolved config must include <Trunk>.")

    patchers = [_build_patcher_runtime(p) for p in resolved_config.model.patchers]
    trunk = _build_trunk_runtime(resolved_config.model.trunk)
    return ModelRuntime(patchers=patchers, trunk=trunk)


def _build_patcher_runtime(patcher: ResolvedPatcherSpec) -> PatcherRuntime:
    return PatcherRuntime(
        name=patcher.name,
        patch_size=patcher.patch_size,
        train_config=runtime_train_config_from_spec(patcher.train),
        blocks=_compose_patcher_blocks(patcher),
    )


def _build_trunk_runtime(trunk: ResolvedTrunkSpec) -> TrunkRuntime:
    return TrunkRuntime(
        name=trunk.name,
        context=trunk.context,
        train_config=runtime_train_config_from_spec(trunk.train),
        blocks=_compose_trunk_blocks(trunk),
    )


def _compose_patcher_blocks(patcher: ResolvedPatcherSpec) -> list[RuntimeBlock]:
    indexes = {"RoPE": 0, "PosEmbedding": 0, "Transformer": 0}
    blocks: list[RuntimeBlock] = []

    for block_name in patcher.block_order:
        if block_name == "RoPE":
            block = patcher.rope_blocks[indexes[block_name]]
            indexes[block_name] += 1
            blocks.append(RoPEBlock(d_model=block.d_model, n_heads=block.n_heads))
        elif block_name == "PosEmbedding":
            block = patcher.pos_embedding_blocks[indexes[block_name]]
            indexes[block_name] += 1
            blocks.append(PosEmbeddingBlock(attributes=dict(block.attributes)))
        elif block_name == "Transformer":
            block = patcher.transformer_blocks[indexes[block_name]]
            indexes[block_name] += 1
            blocks.append(TransformerBlock(d_model=block.d_model, n_heads=block.n_heads))
        else:
            raise ValueError(f"Unsupported patcher child block '{block_name}'.")

    return blocks


def _compose_trunk_blocks(trunk: ResolvedTrunkSpec) -> list[RuntimeBlock]:
    indexes = {"DRope": 0, "Transformer": 0, "MixOfExperts": 0}
    blocks: list[RuntimeBlock] = []

    for block_name in trunk.block_order:
        if block_name == "DRope":
            block = trunk.drope_blocks[indexes[block_name]]
            indexes[block_name] += 1
            blocks.append(DRopeBlock(d_model=block.d_model, n_heads=block.n_heads))
        elif block_name == "Transformer":
            block = trunk.transformer_blocks[indexes[block_name]]
            indexes[block_name] += 1
            blocks.append(TransformerBlock(d_model=block.d_model, n_heads=block.n_heads))
        elif block_name == "MixOfExperts":
            block = trunk.mix_of_experts_blocks[indexes[block_name]]
            indexes[block_name] += 1
            blocks.append(_build_moe_runtime(block))
        else:
            raise ValueError(f"Unsupported trunk child block '{block_name}'.")

    return blocks


def _build_moe_runtime(moe: ResolvedMixOfExpertsSpec) -> MixOfExpertsBlock:
    experts: list[ExpertRuntime] = []
    for expert in moe.experts:
        transformer_index = 0
        expert_blocks: list[RuntimeBlock] = []
        for block_name in expert.block_order:
            if block_name != "Transformer":
                raise ValueError(f"Unsupported expert child block '{block_name}'.")
            block = expert.transformer_blocks[transformer_index]
            transformer_index += 1
            expert_blocks.append(TransformerBlock(d_model=block.d_model, n_heads=block.n_heads))
        experts.append(ExpertRuntime(name=expert.name, blocks=expert_blocks))

    return MixOfExpertsBlock(name=moe.name, experts=experts)
