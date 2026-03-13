from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class ExtractedDocument:
    document_id: str
    dataset_entry: str
    split: str
    text: str
    byte_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MixtureDocument:
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
class StageAOutputRow:
    document_id: str
    teacher_name: str
    group: str
    dataset_entry: str
    split: str
    text: str
    predictions: List[TopKPrediction]


def dataclass_to_json_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj)
