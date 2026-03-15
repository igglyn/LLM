from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SourceSpec:
    name: str
    source_type: str
    uri: str
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitMapEntrySpec:
    from_split: str
    to_split: str


@dataclass(frozen=True)
class SplitMappingSpec:
    entries: List[SplitMapEntrySpec] = field(default_factory=list)


@dataclass(frozen=True)
class FilterSpec:
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetEntrySpec:
    name: str
    source: SourceSpec
    split_mapping: Optional[SplitMappingSpec] = None
    filters: List[FilterSpec] = field(default_factory=list)


@dataclass(frozen=True)
class SourceExtractionSpec:
    dataset_entries: List[DatasetEntrySpec] = field(default_factory=list)


@dataclass(frozen=True)
class DatasetRefSpec:
    name: str


@dataclass(frozen=True)
class GroupSpec:
    name: str
    percentage: float
    dataset_refs: List[DatasetRefSpec] = field(default_factory=list)


@dataclass(frozen=True)
class MixtureBuildSpec:
    attributes: Dict[str, str] = field(default_factory=dict)
    groups: List[GroupSpec] = field(default_factory=list)


@dataclass(frozen=True)
class ModelRefSpec:
    name_or_path: str


@dataclass(frozen=True)
class ExecutionSpec:
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendSpec:
    backend_type: str
    model_ref: ModelRefSpec
    execution: ExecutionSpec


@dataclass(frozen=True)
class TeacherSpec:
    name: str
    backend: BackendSpec


@dataclass(frozen=True)
class TeachersSpec:
    teachers: List[TeacherSpec] = field(default_factory=list)


@dataclass(frozen=True)
class TopKLogitsSpec:
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LongContextSpec:
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuredOutputsSpec:
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StageSpec:
    name: str
    enabled: bool
    teacher_ref: str
    mode_type: str
    mode_attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DistillationSpec:
    attributes: Dict[str, str] = field(default_factory=dict)
    teachers: TeachersSpec = field(default_factory=TeachersSpec)
    stages: List[StageSpec] = field(default_factory=list)

    def stage_by_name(self, stage_name: str) -> StageSpec | None:
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        return None


@dataclass(frozen=True)
class DefaultsSpec:
    d_model: int
    n_heads: int


@dataclass(frozen=True)
class SchedulerSpec:
    scheduler_type: str
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizerSpec:
    optimizer_type: str
    weight_decay: float
    dropout: float = 0.0
    schedulers: List[SchedulerSpec] = field(default_factory=list)


@dataclass(frozen=True)
class TrainSpec:
    steps: int
    batch_size: int = 1
    save_every: int = 0
    optimizer: OptimizerSpec = field(default_factory=lambda: OptimizerSpec(optimizer_type="adamw", weight_decay=0.0))


@dataclass(frozen=True)
class RoPEBlockSpec:
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    base: Optional[float] = None
    scale: Optional[float] = None
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DRopeBlockSpec:
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    base: Optional[float] = None
    scale: Optional[float] = None
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PosEmbeddingBlockSpec:
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VocabEmbeddingBlockSpec:
    vocab_size: int
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LayerNormBlockSpec:
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TransformerBlockSpec:
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    transformer_blocks: List[TransformerBlockSpec] = field(default_factory=list)
    block_order: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class MixOfExpertsSpec:
    name: str
    experts: List[ExpertSpec] = field(default_factory=list)


@dataclass(frozen=True)
class PatcherSpec:
    name: str
    patch_size: int
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    train: TrainSpec | None = None
    rope_blocks: List[RoPEBlockSpec] = field(default_factory=list)
    pos_embedding_blocks: List[PosEmbeddingBlockSpec] = field(default_factory=list)
    vocab_embedding_blocks: List[VocabEmbeddingBlockSpec] = field(default_factory=list)
    layer_norm_blocks: List[LayerNormBlockSpec] = field(default_factory=list)
    transformer_blocks: List[TransformerBlockSpec] = field(default_factory=list)
    block_order: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrunkSpec:
    name: str
    context: int
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    train: TrainSpec | None = None
    rope_blocks: List[RoPEBlockSpec] = field(default_factory=list)
    pos_embedding_blocks: List[PosEmbeddingBlockSpec] = field(default_factory=list)
    vocab_embedding_blocks: List[VocabEmbeddingBlockSpec] = field(default_factory=list)
    drope_blocks: List[DRopeBlockSpec] = field(default_factory=list)
    transformer_blocks: List[TransformerBlockSpec] = field(default_factory=list)
    mix_of_experts_blocks: List[MixOfExpertsSpec] = field(default_factory=list)
    block_order: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelSpec:
    defaults: DefaultsSpec
    patchers: List[PatcherSpec] = field(default_factory=list)
    trunk: TrunkSpec | None = None


@dataclass(frozen=True)
class DatasetSpec:
    source_extraction: SourceExtractionSpec
    mixture_build: MixtureBuildSpec
    distillation: DistillationSpec


@dataclass(frozen=True)
class ConfigSpec:
    dataset: DatasetSpec
    model: ModelSpec


# Resolved layer


@dataclass(frozen=True)
class ResolvedTransformerBlockSpec:
    d_model: int
    n_heads: int
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedRoPEBlockSpec:
    d_model: int
    n_heads: int
    base: float
    scale: float
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedDRopeBlockSpec:
    d_model: int
    n_heads: int
    base: float
    scale: float
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedExpertSpec:
    name: str
    transformer_blocks: List[ResolvedTransformerBlockSpec] = field(default_factory=list)
    block_order: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResolvedMixOfExpertsSpec:
    name: str
    experts: List[ResolvedExpertSpec] = field(default_factory=list)


@dataclass(frozen=True)
class ResolvedPatcherSpec:
    name: str
    patch_size: int
    train: TrainSpec
    rope_blocks: List[ResolvedRoPEBlockSpec] = field(default_factory=list)
    pos_embedding_blocks: List[PosEmbeddingBlockSpec] = field(default_factory=list)
    vocab_embedding_blocks: List[VocabEmbeddingBlockSpec] = field(default_factory=list)
    layer_norm_blocks: List[LayerNormBlockSpec] = field(default_factory=list)
    transformer_blocks: List[ResolvedTransformerBlockSpec] = field(default_factory=list)
    block_order: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResolvedTrunkSpec:
    name: str
    context: int
    train: TrainSpec
    rope_blocks: List[ResolvedRoPEBlockSpec] = field(default_factory=list)
    pos_embedding_blocks: List[PosEmbeddingBlockSpec] = field(default_factory=list)
    vocab_embedding_blocks: List[VocabEmbeddingBlockSpec] = field(default_factory=list)
    drope_blocks: List[ResolvedDRopeBlockSpec] = field(default_factory=list)
    transformer_blocks: List[ResolvedTransformerBlockSpec] = field(default_factory=list)
    mix_of_experts_blocks: List[ResolvedMixOfExpertsSpec] = field(default_factory=list)
    block_order: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResolvedModelSpec:
    defaults: DefaultsSpec
    patchers: List[ResolvedPatcherSpec] = field(default_factory=list)
    trunk: ResolvedTrunkSpec | None = None


@dataclass(frozen=True)
class ResolvedConfigSpec:
    dataset: DatasetSpec
    model: ResolvedModelSpec
