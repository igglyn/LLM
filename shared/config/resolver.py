from __future__ import annotations

from dataclasses import replace

from .specs import (
    ConfigSpec,
    DRopeBlockSpec,
    DefaultsSpec,
    ExpertSpec,
    MixOfExpertsSpec,
    PatcherSpec,
    ResolvedConfigSpec,
    ResolvedDRopeBlockSpec,
    ResolvedExpertSpec,
    ResolvedMixOfExpertsSpec,
    ResolvedModelSpec,
    ResolvedPatcherSpec,
    ResolvedRoPEBlockSpec,
    ResolvedTransformerBlockSpec,
    ResolvedTrunkSpec,
    RoPEBlockSpec,
    TransformerBlockSpec,
    TrunkSpec,
)


class ConfigResolutionError(ValueError):
    pass


def resolve_config(raw_config: ConfigSpec) -> ResolvedConfigSpec:
    model_defaults = raw_config.model.defaults

    resolved_patchers = [
        _resolve_patcher(raw_patcher, model_defaults=model_defaults)
        for raw_patcher in raw_config.model.patchers
    ]

    resolved_trunk = None
    if raw_config.model.trunk is not None:
        resolved_trunk = _resolve_trunk(raw_config.model.trunk, model_defaults=model_defaults)

    return ResolvedConfigSpec(
        dataset=raw_config.dataset,
        model=ResolvedModelSpec(
            defaults=raw_config.model.defaults,
            patchers=resolved_patchers,
            trunk=resolved_trunk,
        ),
    )


def _resolve_patcher(raw_patcher: PatcherSpec, model_defaults: DefaultsSpec) -> ResolvedPatcherSpec:
    patcher_defaults = _merged_defaults(model_defaults, raw_patcher.d_model, raw_patcher.n_heads)

    resolved_transformers = [
        _resolve_transformer_block(block, container_defaults=patcher_defaults, context=f"Patcher '{raw_patcher.name}'")
        for block in raw_patcher.transformer_blocks
    ]
    resolved_ropes = [
        _resolve_rope_block(block, container_defaults=patcher_defaults) for block in raw_patcher.rope_blocks
    ]

    if raw_patcher.train is None:
        raise ConfigResolutionError(f"Patcher '{raw_patcher.name}' is missing <Train> block.")

    return ResolvedPatcherSpec(
        name=raw_patcher.name,
        patch_size=raw_patcher.patch_size,
        train=raw_patcher.train,
        rope_blocks=resolved_ropes,
        pos_embedding_blocks=raw_patcher.pos_embedding_blocks,
        transformer_blocks=resolved_transformers,
        block_order=list(raw_patcher.block_order),
    )


def _resolve_trunk(raw_trunk: TrunkSpec, model_defaults: DefaultsSpec) -> ResolvedTrunkSpec:
    trunk_defaults = _merged_defaults(model_defaults, raw_trunk.d_model, raw_trunk.n_heads)

    resolved_transformers = [
        _resolve_transformer_block(block, container_defaults=trunk_defaults, context=f"Trunk '{raw_trunk.name}'")
        for block in raw_trunk.transformer_blocks
    ]
    resolved_dropes = [
        _resolve_drope_block(block, container_defaults=trunk_defaults) for block in raw_trunk.drope_blocks
    ]
    resolved_moes = [
        _resolve_moe_block(moe_block, trunk_defaults=trunk_defaults) for moe_block in raw_trunk.mix_of_experts_blocks
    ]

    if raw_trunk.train is None:
        raise ConfigResolutionError(f"Trunk '{raw_trunk.name}' is missing <Train> block.")

    return ResolvedTrunkSpec(
        name=raw_trunk.name,
        context=raw_trunk.context,
        train=raw_trunk.train,
        drope_blocks=resolved_dropes,
        transformer_blocks=resolved_transformers,
        mix_of_experts_blocks=resolved_moes,
        block_order=list(raw_trunk.block_order),
    )


def _resolve_moe_block(raw_moe: MixOfExpertsSpec, trunk_defaults: DefaultsSpec) -> ResolvedMixOfExpertsSpec:
    resolved_experts = [_resolve_expert(expert, trunk_defaults=trunk_defaults) for expert in raw_moe.experts]
    return ResolvedMixOfExpertsSpec(name=raw_moe.name, experts=resolved_experts)


def _resolve_expert(raw_expert: ExpertSpec, trunk_defaults: DefaultsSpec) -> ResolvedExpertSpec:
    expert_defaults = _merged_defaults(trunk_defaults, raw_expert.d_model, raw_expert.n_heads)
    resolved_transformers = [
        _resolve_transformer_block(
            block,
            container_defaults=expert_defaults,
            context=f"Expert '{raw_expert.name}'",
        )
        for block in raw_expert.transformer_blocks
    ]
    return ResolvedExpertSpec(
        name=raw_expert.name,
        transformer_blocks=resolved_transformers,
        block_order=list(raw_expert.block_order),
    )


def _resolve_transformer_block(
    raw_block: TransformerBlockSpec,
    container_defaults: DefaultsSpec,
    context: str,
) -> ResolvedTransformerBlockSpec:
    d_model = raw_block.d_model if raw_block.d_model is not None else container_defaults.d_model
    n_heads = raw_block.n_heads if raw_block.n_heads is not None else container_defaults.n_heads

    if d_model is None or n_heads is None:
        raise ConfigResolutionError(f"{context} has <Transformer> missing d_model or n_heads after resolution.")

    return ResolvedTransformerBlockSpec(d_model=d_model, n_heads=n_heads, attributes=dict(raw_block.attributes))


def _resolve_rope_block(raw_block: RoPEBlockSpec, container_defaults: DefaultsSpec) -> ResolvedRoPEBlockSpec:
    d_model = raw_block.d_model if raw_block.d_model is not None else container_defaults.d_model
    n_heads = raw_block.n_heads if raw_block.n_heads is not None else container_defaults.n_heads
    if d_model is None or n_heads is None:
        raise ConfigResolutionError("RoPE block missing d_model or n_heads after resolution.")
    return ResolvedRoPEBlockSpec(d_model=d_model, n_heads=n_heads, attributes=dict(raw_block.attributes))


def _resolve_drope_block(raw_block: DRopeBlockSpec, container_defaults: DefaultsSpec) -> ResolvedDRopeBlockSpec:
    d_model = raw_block.d_model if raw_block.d_model is not None else container_defaults.d_model
    n_heads = raw_block.n_heads if raw_block.n_heads is not None else container_defaults.n_heads
    if d_model is None or n_heads is None:
        raise ConfigResolutionError("DRope block missing d_model or n_heads after resolution.")
    return ResolvedDRopeBlockSpec(d_model=d_model, n_heads=n_heads, attributes=dict(raw_block.attributes))


def _merged_defaults(base_defaults: DefaultsSpec, d_model: int | None, n_heads: int | None) -> DefaultsSpec:
    return replace(
        base_defaults,
        d_model=base_defaults.d_model if d_model is None else d_model,
        n_heads=base_defaults.n_heads if n_heads is None else n_heads,
    )
