from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TypeVar

from .types import dataclass_to_json_dict

T = TypeVar("T")


def write_jsonl(path: str | Path, rows: Iterable[T]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dataclass_to_json_dict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    output: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                output.append(json.loads(line))
    return output
