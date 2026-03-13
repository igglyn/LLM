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
    LongContextSpec,
    MixtureBuildSpec,
    MixOfExpertsSpec,
    ModelRefSpec,
    ModelSpec,
    OptimizerSpec,
    PatcherSpec,
    PosEmbeddingBlockSpec,
    RoPEBlockSpec,
    SchedulerSpec,
    SourceExtractionSpec,
    SourceSpec,
    SplitMapEntrySpec,
    SplitMappingSpec,
    StageASpec,
    StageBSpec,
    StageCSpec,
    StructuredOutputsSpec,
    TeacherSpec,
    TeachersSpec,
    TopKLogitsSpec,
    TrainSpec,
    TransformerBlockSpec,
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
    return DistillationSpec(
        attributes=dict(elem.attrib),
        teachers=_parse_teachers(_required_child(elem, "Teachers")),
        stage_a=_parse_stage_a(_required_child(elem, "StageA")),
        stage_b=_parse_stage_b(_required_child(elem, "StageB")),
        stage_c=_parse_stage_c(_required_child(elem, "StageC")),
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


def _parse_stage_a(elem: ET.Element) -> StageASpec:
    _validate_single_mode_child(elem, "TopKLogits")
    mode_elem = _required_child(elem, "TopKLogits")
    return StageASpec(
        enabled=_parse_bool_attr(elem, "enabled"),
        teacher_ref=_required_attr(elem, "teacher_ref"),
        top_k_logits=TopKLogitsSpec(attributes=dict(mode_elem.attrib)),
    )


def _parse_stage_b(elem: ET.Element) -> StageBSpec:
    _validate_single_mode_child(elem, "LongContext")
    mode_elem = _required_child(elem, "LongContext")
    return StageBSpec(
        enabled=_parse_bool_attr(elem, "enabled"),
        teacher_ref=_required_attr(elem, "teacher_ref"),
        long_context=LongContextSpec(attributes=dict(mode_elem.attrib)),
    )


def _parse_stage_c(elem: ET.Element) -> StageCSpec:
    _validate_single_mode_child(elem, "StructuredOutputs")
    mode_elem = _required_child(elem, "StructuredOutputs")
    return StageCSpec(
        enabled=_parse_bool_attr(elem, "enabled"),
        teacher_ref=_required_attr(elem, "teacher_ref"),
        structured_outputs=StructuredOutputsSpec(attributes=dict(mode_elem.attrib)),
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
    transformer_blocks: List[TransformerBlockSpec] = []
    block_order: List[str] = []

    for child in list(elem):
        if child.tag == "Train":
            continue
        if child.tag == "RoPE":
            rope_blocks.append(
                RoPEBlockSpec(
                    d_model=_optional_int_attr(child, "d_model"),
                    n_heads=_optional_int_attr(child, "n_heads"),
                    attributes=dict(child.attrib),
                )
            )
            block_order.append("RoPE")
        elif child.tag == "PosEmbedding":
            pos_embedding_blocks.append(PosEmbeddingBlockSpec(attributes=dict(child.attrib)))
            block_order.append("PosEmbedding")
        elif child.tag == "Transformer":
            transformer_blocks.append(
                TransformerBlockSpec(
                    d_model=_optional_int_attr(child, "d_model"),
                    n_heads=_optional_int_attr(child, "n_heads"),
                    attributes=dict(child.attrib),
                )
            )
            block_order.append("Transformer")
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
        transformer_blocks=transformer_blocks,
        block_order=block_order,
    )


def _parse_trunk(elem: ET.Element) -> TrunkSpec:
    train = _parse_train(_required_child(elem, "Train"))

    drope_blocks: List[DRopeBlockSpec] = []
    transformer_blocks: List[TransformerBlockSpec] = []
    moe_blocks: List[MixOfExpertsSpec] = []
    block_order: List[str] = []

    for child in list(elem):
        if child.tag == "Train":
            continue
        if child.tag == "DRope":
            drope_blocks.append(
                DRopeBlockSpec(
                    d_model=_optional_int_attr(child, "d_model"),
                    n_heads=_optional_int_attr(child, "n_heads"),
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
        drope_blocks=drope_blocks,
        transformer_blocks=transformer_blocks,
        mix_of_experts_blocks=moe_blocks,
        block_order=block_order,
    )


def _parse_moe(elem: ET.Element) -> MixOfExpertsSpec:
    experts: List[ExpertSpec] = []
    for expert_elem in elem.findall("Expert"):
        transformer_blocks: List[TransformerBlockSpec] = []
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
            else:
                raise ConfigParseError(f"<Expert> contains unsupported child <{child.tag}>.")

        experts.append(
            ExpertSpec(
                name=_required_attr(expert_elem, "name"),
                d_model=_optional_int_attr(expert_elem, "d_model"),
                n_heads=_optional_int_attr(expert_elem, "n_heads"),
                transformer_blocks=transformer_blocks,
                block_order=block_order,
            )
        )
    return MixOfExpertsSpec(name=_required_attr(elem, "name"), experts=experts)


def _parse_train(elem: ET.Element) -> TrainSpec:
    optimizer_elem = _required_child(elem, "Optimizer")

    schedulers = [
        SchedulerSpec(scheduler_type=_required_attr(scheduler_elem, "type"), attributes=dict(scheduler_elem.attrib))
        for scheduler_elem in optimizer_elem.findall("Scheduler")
    ]

    optimizer = OptimizerSpec(
        optimizer_type=_required_attr(optimizer_elem, "type"),
        lr=float(_required_attr(optimizer_elem, "lr")),
        weight_decay=float(_required_attr(optimizer_elem, "weight_decay")),
        schedulers=schedulers,
    )
    return TrainSpec(mode=_required_attr(elem, "mode"), optimizer=optimizer)



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
