"""Structured configuration dataclasses for llm_lab."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ComponentCfg:
    """Named component configuration.

    Attributes:
        name: Registry name used to resolve the component implementation.
        kwargs: Optional constructor keyword arguments.
        freeze: If True, disable gradients for this component's parameters.
        stop_gradient: If True, detach this component's outputs before
            downstream components.
    """

    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    freeze: bool = False
    stop_gradient: bool = False


@dataclass
class DataCfg:
    """Data pipeline configuration."""

    dataset: str = "byte_dataset"
    dataset_type: Literal["bytes_txt", "jsonl_text"] = "bytes_txt"
    path: str = ""
    jsonl_path: str = ""
    jsonl_text_field: str | None = None
    jsonl_group_size: int = 1
    jsonl_shuffle_buffer: int = 0
    batch_size: int = 4
    seq_len: int = 128
    use_precomputed_patches: bool = False
    precomputed_patches_path: str = ""


@dataclass
class ModelCfg:
    """Model component wiring configuration.

    Required subconfigs:
    - codec (e.g., vq_lite supports ``codebook_size``, ``d_model``,
      ``use_ema_updates``, ``ema_decay``)
    - patcher1
    - head

    Hierarchical patchers:
    - ``patcher1`` required (raw bytes -> hidden), supports ``patcher1.freeze``
    - ``patcher2`` optional (hidden -> hidden), supports ``patcher2.freeze``

    Mixer configuration:
    - Legacy: ``mixer`` (single mixer)
    - New: ``mixers`` (sequential stack)
      If both are provided, ``mixers`` is preferred.

    Optional subconfigs:
    - patcher2
    - memory (e.g. context_pyramid with keep_recent_tokens,
      keep_lowres_history, lowres_source)
    """

    codec: ComponentCfg = field(default_factory=lambda: ComponentCfg(name="identity"))
    patcher1: ComponentCfg = field(default_factory=lambda: ComponentCfg(name="chunk"))
    patcher2: ComponentCfg | None = None
    mixer: ComponentCfg = field(default_factory=lambda: ComponentCfg(name="transformer"))
    mixers: list[ComponentCfg] | None = None
    head: ComponentCfg = field(default_factory=lambda: ComponentCfg(name="byte"))
    memory: ComponentCfg | None = None


@dataclass
class TrainCfg:
    """Training hyperparameter configuration."""

    steps: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: Literal["adamw", "ademamix_fused"] = "adamw"
    seed: int = 0
    mode: str = "full"

    ademamix_betas: tuple[float, float, float] = (0.9, 0.999, 0.9999)
    ademamix_alpha: float = 5.0
    ademamix_t_alpha: int | None = None
    ademamix_t_beta3: int | None = None
    ademamix_slow_ema_reset_steps: int | None = None
    ademamix_use_foreach: bool = True

    enable_aux_reconstruction: bool = False
    aux_reconstruction_weight: float = 0.0
    preserve_memory_across_batches: bool = True
    reset_memory_on_document_boundary: bool = False
    truncate_bptt_segments: int | None = None


@dataclass
class DebugCfg:
    """Runtime debug/validation toggles."""

    validate_shapes: bool = False


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    debug: DebugCfg = field(default_factory=DebugCfg)
