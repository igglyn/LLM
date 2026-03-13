from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from shared.config.specs import DistillationSpec, TeacherSpec

from .types import TopKPrediction


class TeacherRuntimeError(ValueError):
    pass


class TeacherBackend(Protocol):
    def top_k_next_tokens(self, text: str, k: int) -> list[TopKPrediction]:
        ...


@dataclass
class DummyLocalBackend:
    def top_k_next_tokens(self, text: str, k: int) -> list[TopKPrediction]:
        seed_tokens = text.split()[-k:] if text.split() else ["<empty>"]
        tokens = list(reversed(seed_tokens))
        while len(tokens) < k:
            tokens.append(f"tok_{len(tokens)}")
        return [TopKPrediction(token=token, score=float(1.0 / (index + 1)), rank=index + 1) for index, token in enumerate(tokens[:k])]


class HuggingFaceBackend:
    def __init__(self, model_name_or_path: str, execution: dict[str, str]) -> None:
        self.model_name_or_path = model_name_or_path
        self.execution = execution
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise TeacherRuntimeError("HF backend requires transformers package.") from exc

        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    def top_k_next_tokens(self, text: str, k: int) -> list[TopKPrediction]:
        import torch  # type: ignore

        encoded = self._tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**encoded).logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            values, indices = torch.topk(probs, k)

        tokens = self._tokenizer.convert_ids_to_tokens(indices[0].tolist())
        scores = values[0].tolist()
        return [
            TopKPrediction(token=str(token), score=float(score), rank=index + 1)
            for index, (token, score) in enumerate(zip(tokens, scores))
        ]


@dataclass
class TeacherRuntime:
    teacher_spec: TeacherSpec
    backend: TeacherBackend


def build_teacher_runtime(teacher_spec: TeacherSpec) -> TeacherRuntime:
    # Extension hook: register additional backend types here with explicit adapters.
    backend_type = teacher_spec.backend.backend_type
    if backend_type in {"dummy_local", "local_dummy"}:
        backend = DummyLocalBackend()
    elif backend_type in {"huggingface", "hf"}:
        backend = HuggingFaceBackend(
            model_name_or_path=teacher_spec.backend.model_ref.name_or_path,
            execution=teacher_spec.backend.execution.attributes,
        )
    else:
        raise TeacherRuntimeError(f"Unsupported backend type '{backend_type}'.")

    return TeacherRuntime(teacher_spec=teacher_spec, backend=backend)


def find_teacher_by_name(distillation: DistillationSpec, teacher_name: str) -> TeacherSpec:
    for teacher in distillation.teachers.teachers:
        if teacher.name == teacher_name:
            return teacher
    raise TeacherRuntimeError(f"Teacher '{teacher_name}' not found.")
