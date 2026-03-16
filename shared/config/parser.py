from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List

from .specs import (
    BackendSpec,
    ConfigSpec,
    DRopeBlockSpec,
    DatasetEntrySpec,
    DatasetRefSpec,
    DatasetSpec,
    DefaultsSpec,
    DistillationSpec,
    ExecutionSpec,
    ExpertSpec,
    FilterSpec,
    GroupSpec,
    LayerNormBlockSpec,
    MixtureBuildSpec,
    MixOfExpertsSpec,
    ModelRefSpec,
    ModelSpec,
    OptimizerSpec,
    PatcherSpec,
    PosEmbeddingBlockSpec,
    VocabEmbeddingBlockSpec,
    RoPEBlockSpec,
    SchedulerSpec,
    SourceExtractionSpec,
    SourceSpec,
    SplitMapEntrySpec,
    SplitMappingSpec,
    StageSpec,
    TeacherSpec,
    TeachersSpec,
    TrainSpec,
    TransformerBlockSpec,
    CrossAttentionBlockSpec,
    TrunkSpec,
)


class ConfigParseError(ValueError):
    pass


def parse_config(path: str | Path) -> ConfigSpec:
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "Config":
        raise ConfigParseError("Root element must be <Config>.")

    children = list(root)
    if len(children) != 2:
        raise ConfigParseError("<Config> must contain exactly <Dataset> and <Model> children.")

    dataset_elem = _find_exact_one(children, "Dataset", "<Config>")
    model_elem = _find_exact_one(children, "Model", "<Config>")

    return ConfigSpec(dataset=_parse_dataset(dataset_elem), model=_parse_model(model_elem))


def _parse_dataset(elem: ET.Element) -> DatasetSpec:
    return DatasetSpec(
        source_extraction=_parse_source_extraction(_required_child(elem, "SourceExtraction")),
        mixture_build=_parse_mixture_build(_required_child(elem, "MixtureBuild")),
        distillation=_parse_distillation(_required_child(elem, "Distillation")),
    )


def _parse_source_extraction(elem: ET.Element) -> SourceExtractionSpec:
    entries: List[DatasetEntrySpec] = []
    for dataset_elem in elem.findall("DatasetEntry"):
        source_elem = _required_child(dataset_elem, "Source")
        split_mapping_elem = dataset_elem.find("SplitMapping")

        split_mapping = None
        if split_mapping_elem is not None:
            split_mapping_entries = [
                SplitMapEntrySpec(
                    from_split=_required_attr(map_elem, "from"),
                    to_split=_required_attr(map_elem, "to"),
                )
                for map_elem in split_mapping_elem.findall("Map")
            ]
            split_mapping = SplitMappingSpec(entries=split_mapping_entries)

        filters = [FilterSpec(attributes=dict(filter_elem.attrib)) for filter_elem in dataset_elem.findall("Filter")]

        entries.append(
            DatasetEntrySpec(
                name=_required_attr(dataset_elem, "name"),
                source=SourceSpec(
                    name=_required_attr(source_elem, "name"),
                    source_type=_source_type(source_elem),
                    uri=_required_attr(source_elem, "uri"),
                    attributes=dict(source_elem.attrib),
                ),
                split_mapping=split_mapping,
                filters=filters,
            )
        )

    return SourceExtractionSpec(dataset_entries=entries)


def _parse_mixture_build(elem: ET.Element) -> MixtureBuildSpec:
    groups: List[GroupSpec] = []
    for group_elem in elem.findall("Group"):
        dataset_refs = [
            DatasetRefSpec(name=_required_attr(ref_elem, "name")) for ref_elem in group_elem.findall("DatasetRef")
        ]
        groups.append(
            GroupSpec(
                name=_required_attr(group_elem, "name"),
                percentage=float(_required_attr(group_elem, "percentage")),
                dataset_refs=dataset_refs,
            )
        )

    return MixtureBuildSpec(attributes=dict(elem.attrib), groups=groups)


def _parse_distillation(elem: ET.Element) -> DistillationSpec:
    teachers = _parse_teachers(_required_child(elem, "Teachers"))
    stages = [_parse_stage(stage_elem) for stage_elem in elem.findall("Stage")]

    if not stages:
        raise ConfigParseError("<Distillation> must contain at least one <Stage> child.")

    return DistillationSpec(
        attributes=dict(elem.attrib),
        teachers=teachers,
        stages=stages,
    )


def _parse_teachers(elem: ET.Element) -> TeachersSpec:
    teachers: List[TeacherSpec] = []
    for teacher_elem in elem.findall("Teacher"):
        backend_elem = _required_child(teacher_elem, "Backend")
        model_ref_elem = _required_child(backend_elem, "ModelRef")
        execution_elem = _required_child(backend_elem, "Execution")
        teachers.append(
            TeacherSpec(
                name=_required_attr(teacher_elem, "name"),
                backend=BackendSpec(
                    backend_type=_required_attr(backend_elem, "type"),
                    model_ref=ModelRefSpec(name_or_path=_required_attr(model_ref_elem, "name_or_path")),
                    execution=ExecutionSpec(attributes=dict(execution_elem.attrib)),
                ),
            )
        )
    return TeachersSpec(teachers=teachers)


def _parse_stage(elem: ET.Element) -> StageSpec:
    stage_name = _required_attr(elem, "name")
    children = list(elem)
    if len(children) != 1:
        raise ConfigParseError(f"<Stage name=\"{stage_name}\"> must contain exactly one mode block.")
    mode_elem = children[0]

    return StageSpec(
        name=stage_name,
        enabled=_parse_bool_attr(elem, "enabled"),
        teacher_ref=_required_attr(elem, "teacher_ref"),
        mode_type=mode_elem.tag,
        mode_attributes=dict(mode_elem.attrib),
    )


def _parse_model(elem: ET.Element) -> ModelSpec:
    defaults_elem = _required_child(elem, "Defaults")
    patchers = [_parse_patcher(patcher_elem) for patcher_elem in elem.findall("Patcher")]
    trunk = _parse_trunk(_required_child(elem, "Trunk"))

    return ModelSpec(
        defaults=DefaultsSpec(
            d_model=int(_required_attr(defaults_elem, "d_model")),
            n_heads=int(_required_attr(defaults_elem, "n_heads")),
        ),
        patchers=patchers,
        trunk=trunk,
    )


def _parse_patcher(elem: ET.Element) -> PatcherSpec:
    train = _parse_train(_required_child(elem, "Train"))

    rope_blocks: List[RoPEBlockSpec] = []
    pos_embedding_blocks: List[PosEmbeddingBlockSpec] = []
    vocab_embedding_blocks: List[VocabEmbeddingBlockSpec] = []
    layer_norm_blocks: List[LayerNormBlockSpec] = []
    transformer_blocks: List[TransformerBlockSpec] = []
    cross_attention_blocks: List[CrossAttentionBlockSpec] = []
    block_order: List[str] = []

    for child in list(elem):
        if child.tag == "Train":
            continue
        if child.tag == "RoPE":
            rope_blocks.append(
                RoPEBlockSpec(
                    d_model=_optional_int_attr(child, "d_model"),
                    n_heads=_optional_int_attr(child, "n_heads"),
                    base=_optional_float_attr(child, "base"),
                    scale=_optional_float_attr(child, "scale"),
                    attributes=dict(child.attrib),
                )
            )
            block_order.append("RoPE")
        elif child.tag == "PosEmbedding":
            pos_embedding_blocks.append(PosEmbeddingBlockSpec(attributes=dict(child.attrib)))
            block_order.append("PosEmbedding")
        elif child.tag == "VocabEmbedding":
            vocab_embedding_blocks.append(
                VocabEmbeddingBlockSpec(vocab_size=int(_required_attr(child, "vocab_size")), attributes=dict(child.attrib))
            )
            block_order.append("VocabEmbedding")
        elif child.tag == "LayerNorm":
            layer_norm_blocks.append(LayerNormBlockSpec(attributes=dict(child.attrib)))
            block_order.append("LayerNorm")
        elif child.tag == "Transformer":
            transformer_blocks.append(
                TransformerBlockSpec(
                    d_model=_optional_int_attr(child, "d_model"),
                    n_heads=_optional_int_attr(child, "n_heads"),
                    attributes=dict(child.attrib),
                )
            )
            block_order.append("Transformer")
        elif child.tag == "CrossAttention":
            cross_attention_blocks.append(
                CrossAttentionBlockSpec(
                    d_model=_optional_int_attr(child, "d_model"),
                    n_heads=_optional_int_attr(child, "n_heads"),
                    attributes=dict(child.attrib),
                )
            )
            block_order.append("CrossAttention")
        else:
            raise ConfigParseError(f"<Patcher> contains unsupported child <{child.tag}>.")

    return PatcherSpec(
        name=_required_attr(elem, "name"),
        patch_size=int(_required_attr(elem, "patch_size")),
        d_model=_optional_int_attr(elem, "d_model"),
        n_heads=_optional_int_attr(elem, "n_heads"),
        train=train,
        rope_blocks=rope_blocks,
        pos_embedding_blocks=pos_embedding_blocks,
        vocab_embedding_blocks=vocab_embedding_blocks,
        layer_norm_blocks=layer_norm_blocks,
        transformer_blocks=transformer_blocks,
        cross_attention_blocks=cross_attention_blocks,
        block_order=block_order,
    )


def _parse_trunk(elem: ET.Element) -> TrunkSpec:
    train = _parse_train(_required_child(elem, "Train"))

    rope_blocks: List[RoPEBlockSpec] = []
    pos_embedding_blocks: List[PosEmbeddingBlockSpec] = []
    vocab_embedding_blocks: List[VocabEmbeddingBlockSpec] = []
    drope_blocks: List[DRopeBlockSpec] = []
    transformer_blocks: List[TransformerBlockSpec] = []
    cross_attention_blocks: List[CrossAttentionBlockSpec] = []
    moe_blocks: List[MixOfExpertsSpec] = []
    block_order: List[str] = []

    for child in list(elem):
        if child.tag == "Train":
            continue
        if child.tag == "RoPE":
            rope_blocks.append(
                RoPEBlockSpec(
                    d_model=_optional_int_attr(child, "d_model"),
                    n_heads=_optional_int_attr(child, "n_heads"),
                    base=_optional_float_attr(child, "base"),
                    scale=_optional_float_attr(child, "scale"),
                    attributes=dict(child.attrib),
                )
            )
            block_order.append("RoPE")
        elif child.tag == "PosEmbedding":
            pos_embedding_blocks.append(PosEmbeddingBlockSpec(attributes=dict(child.attrib)))
            block_order.append("PosEmbedding")
        elif child.tag == "VocabEmbedding":
            vocab_embedding_blocks.append(
                VocabEmbeddingBlockSpec(vocab_size=int(_required_attr(child, "vocab_size")), attributes=dict(child.attrib))
            )
            block_order.append("VocabEmbedding")
        elif child.tag == "DRope":
            drope_blocks.append(
                DRopeBlockSpec(
                    d_model=_optional_int_attr(child, "d_model"),
                    n_heads=_optional_int_attr(child, "n_heads"),
                    base=_optional_float_attr(child, "base"),
                    scale=_optional_float_attr(child, "scale"),
                    attributes=dict(child.attrib),
                )
            )
            block_order.append("DRope")
        elif child.tag == "Transformer":
            transformer_blocks.append(
                TransformerBlockSpec(
                    d_model=_optional_int_attr(child, "d_model"),
                    n_heads=_optional_int_attr(child, "n_heads"),
                    attributes=dict(child.attrib),
                )
            )
            block_order.append("Transformer")
        elif child.tag == "CrossAttention":
            cross_attention_blocks.append(
                CrossAttentionBlockSpec(
                    d_model=_optional_int_attr(child, "d_model"),
                    n_heads=_optional_int_attr(child, "n_heads"),
                    attributes=dict(child.attrib),
                )
            )
            block_order.append("CrossAttention")
        elif child.tag == "MixOfExperts":
            moe_blocks.append(_parse_moe(child))
            block_order.append("MixOfExperts")
        else:
            raise ConfigParseError(f"<Trunk> contains unsupported child <{child.tag}>.")

    return TrunkSpec(
        name=_required_attr(elem, "name"),
        context=int(_required_attr(elem, "context")),
        d_model=_optional_int_attr(elem, "d_model"),
        n_heads=_optional_int_attr(elem, "n_heads"),
        train=train,
        rope_blocks=rope_blocks,
        pos_embedding_blocks=pos_embedding_blocks,
        vocab_embedding_blocks=vocab_embedding_blocks,
        drope_blocks=drope_blocks,
        transformer_blocks=transformer_blocks,
        cross_attention_blocks=cross_attention_blocks,
        mix_of_experts_blocks=moe_blocks,
        block_order=block_order,
    )


def _parse_moe(elem: ET.Element) -> MixOfExpertsSpec:
    experts: List[ExpertSpec] = []
    for expert_elem in elem.findall("Expert"):
        transformer_blocks: List[TransformerBlockSpec] = []
        cross_attention_blocks: List[CrossAttentionBlockSpec] = []
        block_order: List[str] = []
        for child in list(expert_elem):
            if child.tag == "Transformer":
                transformer_blocks.append(
                    TransformerBlockSpec(
                        d_model=_optional_int_attr(child, "d_model"),
                        n_heads=_optional_int_attr(child, "n_heads"),
                        attributes=dict(child.attrib),
                    )
                )
                block_order.append("Transformer")
            elif child.tag == "CrossAttention":
                cross_attention_blocks.append(
                    CrossAttentionBlockSpec(
                        d_model=_optional_int_attr(child, "d_model"),
                        n_heads=_optional_int_attr(child, "n_heads"),
                        attributes=dict(child.attrib),
                    )
                )
                block_order.append("CrossAttention")
            else:
                raise ConfigParseError(f"<Expert> contains unsupported child <{child.tag}>.")

        experts.append(
            ExpertSpec(
                name=_required_attr(expert_elem, "name"),
                d_model=_optional_int_attr(expert_elem, "d_model"),
                n_heads=_optional_int_attr(expert_elem, "n_heads"),
                transformer_blocks=transformer_blocks,
                cross_attention_blocks=cross_attention_blocks,
                block_order=block_order,
            )
        )
    return MixOfExpertsSpec(name=_required_attr(elem, "name"), experts=experts)


def _parse_train(elem: ET.Element) -> TrainSpec:
    optimizer_elem = _required_child(elem, "Optimizer")
    train_steps = int(_required_attr(elem, "steps"))
    batch_size = int(elem.attrib.get("batch_size", "1"))
    save_every = int(elem.attrib.get("save_every", "0"))
    device = elem.attrib.get("device", "cpu")

    if batch_size <= 0:
        raise ConfigParseError("<Train> attribute 'batch_size' must be positive.")
    if save_every < 0:
        raise ConfigParseError("<Train> attribute 'save_every' must be >= 0.")

    schedulers: list[SchedulerSpec] = []
    for optimizer_child in list(optimizer_elem):
        if optimizer_child.tag == "Scheduler":
            schedulers.append(
                SchedulerSpec(
                    scheduler_type=_required_attr(optimizer_child, "type"),
                    attributes=dict(optimizer_child.attrib),
                )
            )
            continue
        if optimizer_child.tag == "Offset":
            start_step = _required_attr(optimizer_child, "start_step")
            end_step = _required_attr(optimizer_child, "end_step")
            attributes = dict(optimizer_child.attrib)
            attributes.setdefault("start_step", start_step)
            attributes.setdefault("end_step", end_step)
            schedulers.append(SchedulerSpec(scheduler_type="offset", attributes=attributes))

    _validate_scheduler_step_coverage(train_steps, schedulers)

    optimizer = OptimizerSpec(
        optimizer_type=_required_attr(optimizer_elem, "type"),
        weight_decay=float(_required_attr(optimizer_elem, "weight_decay")),
        dropout=float(optimizer_elem.attrib.get("dropout", "0.0")),
        grad_clip=(float(optimizer_elem.attrib["grad_clip"]) if "grad_clip" in optimizer_elem.attrib else None),
        schedulers=schedulers,
    )
    return TrainSpec(steps=train_steps, batch_size=batch_size, save_every=save_every, device=device, optimizer=optimizer)


def _validate_scheduler_step_coverage(train_steps: int, schedulers: list[SchedulerSpec]) -> None:
    if train_steps <= 0:
        raise ConfigParseError("<Train> attribute 'steps' must be positive.")
    if not schedulers:
        raise ConfigParseError("<Train> requires at least one <Scheduler> to cover all training steps.")

    intervals: list[tuple[int, int, str]] = []
    for scheduler in schedulers:
        scheduler_type = scheduler.scheduler_type
        if scheduler_type == "offset":
            start_raw = scheduler.attributes.get("start_step")
            end_raw = scheduler.attributes.get("end_step")
            if start_raw is None or end_raw is None:
                raise ConfigParseError("<Offset> must include 'start_step' and 'end_step'.")
            start_step = int(start_raw)
            end_step = int(end_raw)
            if start_step < 0 or end_step <= start_step:
                raise ConfigParseError("<Offset> must satisfy 0 <= start_step < end_step.")
            if end_step > train_steps:
                raise ConfigParseError(
                    f"<Offset> end_step={end_step} exceeds <Train steps=\"{train_steps}\">."
                )
            intervals.append((start_step, end_step, scheduler_type))
            continue

        start_raw = scheduler.attributes.get("start_step")
        end_raw = scheduler.attributes.get("end_step")
        if start_raw is None or end_raw is None:
            raise ConfigParseError(
                f"<Scheduler type=\"{scheduler_type}\"> must include 'start_step' and 'end_step'."
            )
        if scheduler_type == "cosine" and "t_max" in scheduler.attributes:
            raise ConfigParseError(
                "<Scheduler type=\"cosine\"> should not include \"t_max\"; use start_step/end_step range instead."
            )

        start_step = int(start_raw)
        end_step = int(end_raw)
        if start_step < 0 or end_step <= start_step:
            raise ConfigParseError(
                f"<Scheduler type=\"{scheduler_type}\"> must satisfy 0 <= start_step < end_step."
            )
        if end_step > train_steps:
            raise ConfigParseError(
                f"<Scheduler type=\"{scheduler_type}\"> end_step={end_step} exceeds <Train steps=\"{train_steps}\">."
            )
        intervals.append((start_step, end_step, scheduler_type))

    if not intervals:
        return

    intervals.sort(key=lambda x: (x[0], x[1]))
    covered_until = 0
    for start_step, end_step, scheduler_type in intervals:
        if start_step > covered_until:
            raise ConfigParseError(
                f"<Train> has uncovered steps in [{covered_until}, {start_step}); add a scheduler to cover this range."
            )
        if start_step <= covered_until:
            covered_until = max(covered_until, end_step)

    if covered_until < train_steps:
        raise ConfigParseError(
            f"<Train> has uncovered steps in [{covered_until}, {train_steps}); add a scheduler to cover this range."
        )


def _source_type(source_elem: ET.Element) -> str:
    source_type = source_elem.attrib.get("type")
    if source_type is not None:
        return source_type
    legacy_format = source_elem.attrib.get("format")
    if legacy_format is not None:
        return legacy_format
    raise ConfigParseError("<Source> must include either 'type' or 'format'.")

def _validate_single_mode_child(elem: ET.Element, expected_tag: str) -> None:
    children = list(elem)
    if len(children) != 1 or children[0].tag != expected_tag:
        raise ConfigParseError(f"<{elem.tag}> must contain exactly one <{expected_tag}> mode block.")


def _parse_bool_attr(elem: ET.Element, name: str) -> bool:
    value = _required_attr(elem, name)
    if value == "true":
        return True
    if value == "false":
        return False
    raise ConfigParseError(f"<{elem.tag}> attribute '{name}' must be 'true' or 'false'.")


def _optional_int_attr(elem: ET.Element, attr: str) -> int | None:
    value = elem.attrib.get(attr)
    if value is None:
        return None
    return int(value)


def _optional_float_attr(elem: ET.Element, attr: str) -> float | None:
    value = elem.attrib.get(attr)
    if value is None:
        return None
    return float(value)


def _find_exact_one(children: Iterable[ET.Element], tag: str, parent: str) -> ET.Element:
    matches = [child for child in children if child.tag == tag]
    if len(matches) != 1:
        raise ConfigParseError(f"{parent} must contain exactly one <{tag}>.")
    return matches[0]


def _required_child(elem: ET.Element, tag: str) -> ET.Element:
    child = elem.find(tag)
    if child is None:
        raise ConfigParseError(f"<{elem.tag}> missing required child <{tag}>.")
    return child


def _required_attr(elem: ET.Element, attr: str) -> str:
    value = elem.attrib.get(attr)
    if value is None:
        raise ConfigParseError(f"<{elem.tag}> missing required attribute '{attr}'.")
    return value
