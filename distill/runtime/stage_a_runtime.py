from __future__ import annotations

from shared.config.specs import ConfigSpec

from .teacher_runtime import TeacherRuntimeError, build_teacher_runtime, find_teacher_by_name
from .types import MixtureDocument, StageAOutputRow


class StageARuntimeError(ValueError):
    pass


def run_stage_a(config: ConfigSpec, mixed_docs: list[MixtureDocument]) -> list[StageAOutputRow]:
    distillation = config.dataset.distillation
    stage_a = distillation.stage_a
    if stage_a is None:
        raise StageARuntimeError("StageA is not configured.")
    if not stage_a.enabled:
        return []
    if stage_a.top_k_logits is None:
        raise StageARuntimeError("StageA mode must be <TopKLogits>.")

    teacher_spec = find_teacher_by_name(distillation, stage_a.teacher_ref)
    teacher_runtime = build_teacher_runtime(teacher_spec)

    k_value = int(stage_a.top_k_logits.attributes.get("k", "0"))
    if k_value <= 0:
        raise StageARuntimeError("StageA TopKLogits requires positive 'k'.")

    rows: list[StageAOutputRow] = []
    for doc in mixed_docs:
        predictions = teacher_runtime.backend.top_k_next_tokens(doc.text, k_value)
        rows.append(
            StageAOutputRow(
                document_id=doc.document_id,
                teacher_name=teacher_spec.name,
                group=doc.group,
                dataset_entry=doc.dataset_entry,
                split=doc.split,
                text=doc.text,
                predictions=predictions,
            )
        )
    return rows


def validate_stage_a_mode(config: ConfigSpec) -> None:
    stage_a = config.dataset.distillation.stage_a
    if stage_a is None or stage_a.top_k_logits is None:
        raise StageARuntimeError("StageA mode must be <TopKLogits>.")
