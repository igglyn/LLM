from __future__ import annotations

from .base import TeacherError, TeacherRuntime, teacher_by_ref
from .hf_backend import DummyLocalBackend, HuggingFaceBackend


def build_teacher_runtime(teacher_spec) -> TeacherRuntime:
    backend_type = teacher_spec.backend.backend_type
    if backend_type in {"dummy_local", "local_dummy"}:
        backend = DummyLocalBackend()
    elif backend_type in {"huggingface", "hf"}:
        backend = HuggingFaceBackend(model_name_or_path=teacher_spec.backend.model_ref.name_or_path)
    else:
        raise TeacherError(f"Unsupported backend type '{backend_type}'.")
    return TeacherRuntime(teacher_spec=teacher_spec, backend=backend)


__all__ = [
    "TeacherError",
    "TeacherRuntime",
    "teacher_by_ref",
    "build_teacher_runtime",
]
