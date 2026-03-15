from __future__ import annotations

from dataclasses import dataclass

from shared.config.specs import TeacherSpec


@dataclass(frozen=True)
class TeacherVocabResult:
    teacher_name: str
    backend_type: str
    model_name_or_path: str
    vocab_size: int | None
    source: str


def teacher_vocab_size(teacher: TeacherSpec) -> TeacherVocabResult:
    backend_type = teacher.backend.backend_type
    model_name_or_path = teacher.backend.model_ref.name_or_path

    if backend_type in {"dummy_local", "local_dummy"}:
        return TeacherVocabResult(
            teacher_name=teacher.name,
            backend_type=backend_type,
            model_name_or_path=model_name_or_path,
            vocab_size=None,
            source="unsupported_backend",
        )

    if backend_type not in {"huggingface", "hf"}:
        return TeacherVocabResult(
            teacher_name=teacher.name,
            backend_type=backend_type,
            model_name_or_path=model_name_or_path,
            vocab_size=None,
            source="unsupported_backend",
        )

    try:
        from transformers import AutoConfig, AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("HF vocab inspection requires transformers package.") from exc

    config = AutoConfig.from_pretrained(model_name_or_path)
    config_vocab_size = getattr(config, "vocab_size", None)
    if isinstance(config_vocab_size, int) and config_vocab_size > 0:
        return TeacherVocabResult(
            teacher_name=teacher.name,
            backend_type=backend_type,
            model_name_or_path=model_name_or_path,
            vocab_size=config_vocab_size,
            source="config.vocab_size",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(tokenizer_vocab_size, int) and tokenizer_vocab_size > 0:
        return TeacherVocabResult(
            teacher_name=teacher.name,
            backend_type=backend_type,
            model_name_or_path=model_name_or_path,
            vocab_size=tokenizer_vocab_size,
            source="tokenizer.vocab_size",
        )

    tokenizer_len = len(tokenizer)
    if tokenizer_len > 0:
        return TeacherVocabResult(
            teacher_name=teacher.name,
            backend_type=backend_type,
            model_name_or_path=model_name_or_path,
            vocab_size=tokenizer_len,
            source="len(tokenizer)",
        )

    return TeacherVocabResult(
        teacher_name=teacher.name,
        backend_type=backend_type,
        model_name_or_path=model_name_or_path,
        vocab_size=None,
        source="unavailable",
    )
