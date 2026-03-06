"""Model assembly utilities."""

from __future__ import annotations

import torch
from torch import nn

# Import concrete components for registration side effects.
from llm_lab.components.codecs.fsq import FSQCodec as _FSQCodec
from llm_lab.components.codecs.identity import IdentityCodec as _IdentityCodec
from llm_lab.components.codecs.vq_lite import VQLiteCodec as _VQLiteCodec
from llm_lab.components.heads.byte_lm import ByteLMHead as _ByteLMHead
from llm_lab.components.heads.reconstruct_bytes import ReconstructBytesHead as _ReconstructBytesHead
from llm_lab.components.memory.context_pyramid import ContextPyramidMemory as _ContextPyramidMemory
from llm_lab.components.memory.knn_fifo import KNNFIFOMemory as _KNNFIFOMemory
from llm_lab.components.mixers.block_transformer import (
    BlockTransformerMixer as _BlockTransformerMixer,
)
from llm_lab.components.mixers.hyena import HyenaMixer as _HyenaMixer
from llm_lab.components.mixers.mamba import MambaMixer as _MambaMixer
from llm_lab.components.mixers.perceiver_io import PerceiverIOMixer as _PerceiverIOMixer
from llm_lab.components.mixers.retnet import RetNetMixer as _RetNetMixer
from llm_lab.components.mixers.rwkv import RWKVMixer as _RWKVMixer
from llm_lab.components.mixers.transformer import TransformerMixer as _TransformerMixer
from llm_lab.components.patchers.chunk import ChunkPatcher as _ChunkPatcher
from llm_lab.components.patchers.dynamic_blt import DynamicBLTPatcher as _DynamicBLTPatcher
from llm_lab.components.patchers.hidden_chunk import HiddenChunkPatcher as _HiddenChunkPatcher
from llm_lab.config.schema import ComponentCfg, ExperimentConfig
from llm_lab.debug.shapes import assert_bytes, assert_hidden, assert_ids, assert_logits
from llm_lab.registry import build_component, build_mixer_stack


class AssembledModel(nn.Module):
    """Minimal assembled model: bytes -> patch embeddings -> logits."""

    def __init__(
        self,
        codec: object,
        patcher1: nn.Module,
        patcher2: nn.Module | None,
        mixers: list[nn.Module],
        head: nn.Module,
        reconstruction_head: nn.Module | None = None,
        memory: object | None = None,
        patcher1_stop_gradient: bool = False,
        patcher2_stop_gradient: bool = False,
        validate_shapes: bool = False,
    ) -> None:
        super().__init__()
        self.codec = codec
        self.patcher1 = patcher1
        self.patcher2 = patcher2
        self.mixers = nn.ModuleList(mixers)
        self.head = head
        self.reconstruction_head = reconstruction_head
        self.memory = memory
        self.patcher1_stop_gradient = patcher1_stop_gradient
        self.patcher2_stop_gradient = patcher2_stop_gradient
        self.validate_shapes = validate_shapes
        self.last_codec_codes: torch.Tensor | None = None

    def component_param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Return parameters grouped by major model components."""
        return {
            "patcher1": list(self.patcher1.parameters()),
            "patcher2": list(self.patcher2.parameters()) if self.patcher2 is not None else [],
            "mixers": list(self.mixers.parameters()),
            "head": list(self.head.parameters()),
            "reconstruction_head": (
                list(self.reconstruction_head.parameters())
                if self.reconstruction_head is not None
                else []
            ),
            "memory": (
                list(self.memory.parameters())
                if isinstance(self.memory, nn.Module)
                else []
            ),
        }

    def component_param_counts(self) -> dict[str, dict[str, int]]:
        """Return per-component total/trainable parameter counts."""
        counts: dict[str, dict[str, int]] = {}
        for name, params in self.component_param_groups().items():
            total = sum(p.numel() for p in params)
            trainable = sum(p.numel() for p in params if p.requires_grad)
            counts[name] = {"total": total, "trainable": trainable}
        return counts


    def forward_with_aux(self, x_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass returning main logits and optional reconstruction logits."""
        if self.validate_shapes:
            assert_bytes(x_u8, name="model_input")

        token_ids = self.codec.encode(x_u8)
        if self.validate_shapes:
            assert_ids(token_ids, name="codec_output")

        x_bytes = self.codec.decode(token_ids)
        assert_bytes(x_bytes, name="patcher1_input")

        h1 = self.patcher1.forward(x_bytes)
        assert_hidden(h1, name="patcher1_output")

        # Optional codec-side hidden quantization hook (e.g., FSQ / VQ).
        h1, code_ids = self.codec.quantize_hidden_with_codes(h1)
        self.last_codec_codes = code_ids
        if self.validate_shapes:
            assert_hidden(h1, name="codec_quantized_hidden")

        patcher1_stop_gradient = bool(getattr(self, "patcher1_stop_gradient", False))
        h1_downstream = h1.detach() if patcher1_stop_gradient else h1

        h2 = None
        if self.patcher2 is not None:
            assert_hidden(h1_downstream, name="patcher2_input")
            h2 = self.patcher2.forward(h1_downstream)
            assert_hidden(h2, name="patcher2_output")
            if bool(getattr(self, "patcher2_stop_gradient", False)):
                h2 = h2.detach()

        uses_context_pyramid = self.memory is not None and hasattr(self.memory, "combine")
        if uses_context_pyramid:
            h_mix, recent_len = self.memory.combine(h1_downstream, h2)
            if self.validate_shapes:
                assert_hidden(h_mix, name="memory_output")
        else:
            h_mix = h2 if h2 is not None else h1_downstream
            recent_len = h_mix.shape[1]

        for i, mixer in enumerate(self.mixers):
            h_mix = mixer(h_mix)
            if self.validate_shapes:
                assert_hidden(h_mix, name=f"mixer_output[{i}]")

        h_head = h_mix[:, -recent_len:, :] if uses_context_pyramid else h_mix
        logits = self.head(h_head)
        if self.validate_shapes:
            assert_logits(logits, name="head_output")

        recon_logits = None
        if self.reconstruction_head is not None:
            recon_logits = self.reconstruction_head(h1)

        return logits, recon_logits

    def forward(self, x_u8: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_aux(x_u8)
        return logits


def _freeze_module(module: nn.Module | None) -> None:
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = False


def _resolve_mixers(config: ExperimentConfig) -> list[ComponentCfg]:
    mixers = getattr(config.model, "mixers", None)
    if mixers:
        # Preferred path when both legacy mixer and new mixers exist.
        return mixers
    # Backward compatibility for configs that only provide single mixer.
    return [config.model.mixer]


def _validate_mixer_hidden_dim(mixers: list[ComponentCfg]) -> None:
    target_d: int | None = None
    for m in mixers:
        d = m.kwargs.get("d_model") if isinstance(m.kwargs, dict) else None
        if d is None:
            continue
        if target_d is None:
            target_d = d
        elif d != target_d:
            raise ValueError("All mixers in stack must use the same d_model.")


def _clean_component_kwargs(cfg: ComponentCfg) -> dict[str, object]:
    """Remove control flags that should not be forwarded to constructors."""
    kwargs = dict(cfg.kwargs) if isinstance(cfg.kwargs, dict) else {}
    kwargs.pop("freeze", None)
    kwargs.pop("stop_gradient", None)
    return kwargs


def assemble_model(config: ExperimentConfig) -> AssembledModel:
    """Assemble baseline model from codec, patcher, mixer stack, and head config."""
    _ = (
        _IdentityCodec,
        _FSQCodec,
        _VQLiteCodec,
        _ChunkPatcher,
        _DynamicBLTPatcher,
        _HiddenChunkPatcher,
        _TransformerMixer,
        _BlockTransformerMixer,
        _RetNetMixer,
        _RWKVMixer,
        _HyenaMixer,
        _MambaMixer,
        _PerceiverIOMixer,
        _KNNFIFOMemory,
        _ContextPyramidMemory,
        _ByteLMHead,
        _ReconstructBytesHead,
    )
    codec = build_component("codec", config.model.codec.name, **config.model.codec.kwargs)
    patcher1 = build_component(
        "patcher", config.model.patcher1.name, **_clean_component_kwargs(config.model.patcher1)
    )
    patcher2 = None
    if config.model.patcher2 is not None:
        patcher2 = build_component(
            "patcher", config.model.patcher2.name, **_clean_component_kwargs(config.model.patcher2)
        )

    mixer_cfgs = _resolve_mixers(config)
    _validate_mixer_hidden_dim(mixer_cfgs)
    mixers = build_mixer_stack(mixer_cfgs)

    head = build_component("head", config.model.head.name, **config.model.head.kwargs)

    memory = None
    if config.model.memory is not None:
        memory = build_component("memory", config.model.memory.name, **config.model.memory.kwargs)

    reconstruction_head = None
    if bool(getattr(config.train, "enable_aux_reconstruction", False)):
        patch_size = int(getattr(patcher1, "patch_size", config.model.patcher1.kwargs.get("patch_size", 1)))
        d_model = int(getattr(patcher1, "d_model", config.model.patcher1.kwargs.get("d_model", 16)))
        reconstruction_head = _ReconstructBytesHead(d_model=d_model, patch_size=patch_size)

    if getattr(config.model.patcher1, "freeze", False):
        _freeze_module(patcher1)
    if config.model.patcher2 is not None and getattr(config.model.patcher2, "freeze", False):
        _freeze_module(patcher2)

    validate_shapes = bool(getattr(getattr(config, "debug", None), "validate_shapes", False))
    return AssembledModel(
        codec=codec,
        patcher1=patcher1,
        patcher2=patcher2,
        mixers=mixers,
        head=head,
        reconstruction_head=reconstruction_head,
        memory=memory,
        patcher1_stop_gradient=bool(getattr(config.model.patcher1, "stop_gradient", False)),
        patcher2_stop_gradient=bool(
            getattr(config.model.patcher2, "stop_gradient", False)
            if config.model.patcher2 is not None
            else False
        ),
        validate_shapes=validate_shapes,
    )
