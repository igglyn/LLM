"""Tests for JSONL text dataset ingestion and chunking."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from llm_lab.data.jsonl_text_dataset import JsonlTextDataset


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )


def _decode_chunk(t: torch.Tensor) -> str:
    return bytes(t.tolist()).decode("utf-8", errors="replace")


def test_flat_jsonl_prefers_completion_then_text_and_is_deterministic(tmp_path: Path) -> None:
    jsonl = tmp_path / "flat.jsonl"
    _write_jsonl(
        jsonl,
        [
            {"completion_text": "ABCD"},
            {"text": "EFGH"},
            {"prompt_text": "ignored", "completion_text": "IJKL"},
        ],
    )

    ds1 = JsonlTextDataset(str(jsonl), seq_len=4, seed=17)
    ds2 = JsonlTextDataset(str(jsonl), seq_len=4, seed=17)

    assert len(ds1) == 3
    assert ds1[0].dtype == torch.uint8
    assert ds1[0].shape == (4,)
    assert torch.equal(ds1[0], ds2[0])
    assert torch.equal(ds1[1], ds2[1])
    assert torch.equal(ds1[2], ds2[2])


def test_structured_output_is_preferred_and_prompt_is_optional(tmp_path: Path) -> None:
    jsonl = tmp_path / "structured.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "completion_text": "TOPLVL",
                "structured_output": {
                    "task_type": "qa",
                    "prompt_text": "Q",
                    "completion_text": "A",
                },
            }
        ],
    )

    no_prompt = JsonlTextDataset(str(jsonl), seq_len=1, seed=0, include_prompt=False)
    with_prompt = JsonlTextDataset(str(jsonl), seq_len=3, seed=0, include_prompt=True)

    no_prompt_bytes = b"".join(bytes(no_prompt[i].tolist()) for i in range(len(no_prompt)))

    assert no_prompt_bytes == b"A"
    assert len(with_prompt) == 1
    assert _decode_chunk(with_prompt[0]) == "Q\nA"


def test_jsonl_text_field_override_still_works(tmp_path: Path) -> None:
    jsonl = tmp_path / "override.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "text": "ignored",
                "my_field": "WXYZ",
                "structured_output": {"completion_text": "also_ignored"},
            }
        ],
    )

    ds = JsonlTextDataset(str(jsonl), seq_len=4, seed=0, jsonl_text_field="my_field")
    assert len(ds) == 1
    assert _decode_chunk(ds[0]) == "WXYZ"
