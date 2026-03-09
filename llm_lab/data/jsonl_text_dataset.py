"""JSONL text dataset that yields fixed-length byte chunks."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


class JsonlTextDataset(Dataset[Tensor]):
    """Dataset that reads text from JSONL records and returns byte chunks.

    Supports either a single JSONL file path or a directory containing ``*.jsonl``
    files.

    Text selection priority per record:

    1. ``jsonl_text_field`` override, when provided.
    2. ``structured_output.completion_text`` (Stage C distill-factory records).
       If ``include_prompt`` is True and ``structured_output.prompt_text`` exists,
       include ``prompt_text + "\\n" + completion_text``.
    3. Top-level ``completion_text``.
       If ``include_prompt`` is True and top-level ``prompt_text`` exists,
       include ``prompt_text + "\\n" + completion_text``.
    4. Top-level ``text``.

    Selected text is encoded with UTF-8 using ``errors=\"replace\"``, concatenated,
    and split into non-overlapping ``seq_len`` chunks. Each item is returned as a
    ``torch.uint8`` tensor with shape ``[T]`` where ``T == seq_len``.
    """

    def __init__(
        self,
        jsonl_path: str,
        seq_len: int,
        seed: int = 0,
        jsonl_text_field: str | None = None,
        include_prompt: bool = False,
        jsonl_group_size: int = 1,
        jsonl_shuffle_buffer: int = 0,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.seq_len = seq_len
        self.seed = seed
        self.jsonl_text_field = jsonl_text_field
        self.include_prompt = include_prompt
        self.jsonl_group_size = jsonl_group_size
        self.jsonl_shuffle_buffer = jsonl_shuffle_buffer

        if self.seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if self.jsonl_group_size <= 0:
            raise ValueError("jsonl_group_size must be > 0")
        if self.jsonl_shuffle_buffer < 0:
            raise ValueError("jsonl_shuffle_buffer must be >= 0")

        rng = random.Random(self.seed)
        selected_texts = list(self._iter_selected_texts())
        selected_texts = list(self._apply_shuffle_buffer(selected_texts, rng))
        grouped_texts = list(self._group_texts(selected_texts))

        raw_parts = [text.encode("utf-8", errors="replace") for text in grouped_texts]
        raw = b"".join(raw_parts)
        n_chunks = len(raw) // self.seq_len
        self._chunks = [
            raw[i * self.seq_len : (i + 1) * self.seq_len] for i in range(n_chunks)
        ]

        rng.shuffle(self._chunks)

    def _resolve_paths(self, path: Path) -> list[Path]:
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(path.glob("*.jsonl"))
        return []

    def _join_prompt_completion(
        self,
        prompt: object | None,
        completion: object,
    ) -> str:
        if self.include_prompt and prompt is not None:
            return f"{prompt}\n{completion}"
        return str(completion)

    def _extract_text(self, record: dict[str, object]) -> str | None:
        if self.jsonl_text_field:
            value = record.get(self.jsonl_text_field)
            return None if value is None else str(value)

        structured = record.get("structured_output")
        if isinstance(structured, dict):
            completion = structured.get("completion_text")
            if completion is not None:
                return self._join_prompt_completion(
                    structured.get("prompt_text"),
                    completion,
                )

        completion = record.get("completion_text")
        if completion is not None:
            return self._join_prompt_completion(record.get("prompt_text"), completion)

        text = record.get("text")
        if text is not None:
            return str(text)

        return None

    def _iter_selected_texts(self) -> list[str]:
        texts: list[str] = []
        paths = self._resolve_paths(self.jsonl_path)
        for path in paths:
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
                if not isinstance(record, dict):
                    continue
                text = self._extract_text(record)
                if text is not None:
                    texts.append(text)
        return texts

    def _apply_shuffle_buffer(self, texts: list[str], rng: random.Random) -> list[str]:
        if self.jsonl_shuffle_buffer <= 0:
            return texts

        shuffled: list[str] = []
        buffer: list[str] = []
        for text in texts:
            buffer.append(text)
            if len(buffer) >= self.jsonl_shuffle_buffer:
                idx = rng.randrange(len(buffer))
                shuffled.append(buffer.pop(idx))

        while buffer:
            idx = rng.randrange(len(buffer))
            shuffled.append(buffer.pop(idx))

        return shuffled

    def _group_texts(self, texts: list[str]) -> list[str]:
        if self.jsonl_group_size == 1:
            return texts

        grouped: list[str] = []
        current: list[str] = []
        for text in texts:
            current.append(text)
            if len(current) == self.jsonl_group_size:
                grouped.append("\n\n".join(current))
                current = []

        if current:
            grouped.append("\n\n".join(current))

        return grouped

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, index: int) -> Tensor:
        chunk = self._chunks[index]
        return torch.tensor(list(chunk), dtype=torch.uint8)
