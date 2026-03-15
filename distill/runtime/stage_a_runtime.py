from __future__ import annotations

from shared.config.specs import ConfigSpec

from .teacher_runtime import build_teacher_runtime, find_teacher_by_name
from .types import MixtureDocument, StageAOutputRow


class StageARuntimeError(ValueError):
    pass


def run_stage_a(config: ConfigSpec, mixed_docs: list[MixtureDocument]) -> list[StageAOutputRow]:
    distillation = config.dataset.distillation
    stage_a = distillation.stage_by_name("StageA")
    if stage_a is None:
        raise StageARuntimeError("StageA is not configured.")
    if not stage_a.enabled:
        return []
    if stage_a.mode_type != "TopKLogits":
        raise StageARuntimeError("StageA mode must be <TopKLogits>.")

    teacher_spec = find_teacher_by_name(distillation, stage_a.teacher_ref)
    teacher_runtime = build_teacher_runtime(teacher_spec)

    k_value = int(stage_a.mode_attributes.get("k", "0"))
    if k_value <= 0:
        raise StageARuntimeError("StageA TopKLogits requires positive 'k'.")

    rows: list[StageAOutputRow] = []
    for record_index, doc in enumerate(mixed_docs):
        prompt_text, target_text = _prompt_target_slice(doc.text)
        predictions = teacher_runtime.backend.top_k_next_tokens(prompt_text, k_value)
        rows.append(
            StageAOutputRow(
                record_id=f"stageA-{record_index}",
                doc_id=doc.document_id,
                prompt_text=prompt_text,
                target_text=target_text,
                top_k_predictions=predictions,
                metadata={
                    "teacher_name": teacher_spec.name,
                    "group": doc.group,
                    "dataset_entry": doc.dataset_entry,
                    "split": doc.split,
                    **doc.metadata,
                },
            )
        )
    return rows


def validate_stage_a_mode(config: ConfigSpec) -> None:
    stage_a = config.dataset.distillation.stage_by_name("StageA")
    if stage_a is None or stage_a.mode_type != "TopKLogits":
        raise StageARuntimeError("StageA mode must be <TopKLogits>.")


def _prompt_target_slice(text: str) -> tuple[str, str]:
    words = text.split()
    if len(words) <= 1:
        return text, ""
    return " ".join(words[:-1]), words[-1]
