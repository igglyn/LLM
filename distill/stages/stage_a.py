from __future__ import annotations

from shared.config.specs import ResolvedConfigSpec

from distill.schemas import MixtureEntry, StageARecord
from distill.teachers import build_teacher_runtime, teacher_by_ref


class StageAError(ValueError):
    pass


def validate_stage_a_mode(config: ResolvedConfigSpec) -> None:
    stage_a = config.dataset.distillation.stage_a
    if stage_a is None or stage_a.top_k_logits is None:
        raise StageAError("StageA mode must be TopKLogits.")


def run_stage_a(config: ResolvedConfigSpec, mixture_entries: list[MixtureEntry]) -> list[StageARecord]:
    stage_a = config.dataset.distillation.stage_a
    if stage_a is None:
        raise StageAError("Missing StageA config.")
    if not stage_a.enabled:
        return []
    validate_stage_a_mode(config)

    k = int(stage_a.top_k_logits.attributes.get("k", "0"))
    if k <= 0:
        raise StageAError("StageA TopKLogits requires positive k.")

    teacher = teacher_by_ref(config.dataset.distillation, stage_a.teacher_ref)
    runtime = build_teacher_runtime(teacher)

    rows: list[StageARecord] = []
    for entry in mixture_entries:
        rows.append(
            StageARecord(
                document_id=entry.document_id,
                teacher_name=teacher.name,
                group=entry.group,
                dataset_entry=entry.dataset_entry,
                split=entry.split,
                text=entry.text,
                predictions=runtime.backend.top_k_next_tokens(entry.text, k),
            )
        )
    return rows
