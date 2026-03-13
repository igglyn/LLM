from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from shared.config.specs import DistillationSpec, TeacherSpec

from distill.schemas import TopKPrediction


class TeacherError(ValueError):
    pass


class TeacherBackend(Protocol):
    def top_k_next_tokens(self, text: str, k: int) -> list[TopKPrediction]:
        ...


@dataclass(frozen=True)
class TeacherRuntime:
    teacher_spec: TeacherSpec
    backend: TeacherBackend


def teacher_by_ref(distillation: DistillationSpec, teacher_ref: str) -> TeacherSpec:
    for teacher in distillation.teachers.teachers:
        if teacher.name == teacher_ref:
            return teacher
    raise TeacherError(f"Teacher ref '{teacher_ref}' not found.")
