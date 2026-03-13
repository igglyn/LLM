from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from shared.config.specs import ResolvedConfigSpec


@dataclass(frozen=True)
class NormalizedDocument:
    document_id: str
    dataset_entry: str
    split: str
    text: str
    byte_length: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MixtureEntry:
    document_id: str
    group: str
    dataset_entry: str
    split: str
    text: str
    byte_length: int


@dataclass(frozen=True)
class TopKPrediction:
    token: str
    score: float
    rank: int


@dataclass(frozen=True)
class StageARecord:
    document_id: str
    teacher_name: str
    group: str
    dataset_entry: str
    split: str
    text: str
    predictions: list[TopKPrediction]


@dataclass(frozen=True)
class DistillRuntimeConfig:
    resolved: ResolvedConfigSpec


def to_json_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
