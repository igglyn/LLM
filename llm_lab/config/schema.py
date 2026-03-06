"""Structured configuration dataclasses for llm_lab."""

from dataclasses import dataclass, field
from typing import Any


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
    path: str = ""
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
    seed: int = 0
    mode: str = "full"
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
