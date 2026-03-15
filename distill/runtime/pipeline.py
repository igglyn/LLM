from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from shared.config import parse_config

from .io_utils import read_jsonl, write_jsonl
from .mixture_build import run_mixture_build
from .source_extraction import run_source_extraction
from .stage_a_runtime import run_stage_a, validate_stage_a_mode
from .types import ExtractedDocument, MixtureDocument, StageAOutputRow, TokenMappingRow


def run_extract(config_path: str | Path, output_dir: str | Path) -> int:
    config = parse_config(config_path)
    docs = run_source_extraction(config)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    docs_by_dataset: dict[str, list[ExtractedDocument]] = defaultdict(list)
    for doc in docs:
        docs_by_dataset[doc.dataset_entry].append(doc)

    for dataset_name, rows in docs_by_dataset.items():
        write_jsonl(output_root / f"{dataset_name}.jsonl", rows)
    return len(docs)


def run_mix(config_path: str | Path, extracted_path: str | Path, output_path: str | Path) -> int:
    config = parse_config(config_path)
    extracted = [ExtractedDocument(**row) for row in _read_extracted_rows(extracted_path)]
    mixed = run_mixture_build(config, extracted)
    write_jsonl(output_path, mixed)
    return len(mixed)


def run_stage_a_command(config_path: str | Path, mixed_path: str | Path, output_dir: str | Path) -> int:
    config = parse_config(config_path)
    validate_stage_a_mode(config)
    mixed = [MixtureDocument(**row) for row in read_jsonl(mixed_path)]
    rows = run_stage_a(config, mixed)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "stage_a.jsonl", rows)
    write_jsonl(output_root / "token_mapping.jsonl", _build_token_mapping(rows))
    return len(rows)


def _read_extracted_rows(extracted_path: str | Path) -> list[dict]:
    source = Path(extracted_path)
    if source.is_dir():
        rows: list[dict] = []
        for jsonl_path in sorted(source.glob("*.jsonl")):
            rows.extend(read_jsonl(jsonl_path))
        return rows
    return read_jsonl(source)


def _build_token_mapping(stage_a_rows: list[StageAOutputRow]) -> list[TokenMappingRow]:
    seen: dict[str, int] = {}

    def _maybe_add(token: str) -> None:
        if token and token not in seen:
            seen[token] = len(seen)

    for row in stage_a_rows:
        target_text = getattr(row, "target_text", "")
        if isinstance(target_text, str):
            _maybe_add(target_text)

        predictions = getattr(row, "top_k_predictions", [])
        for prediction in predictions:
            token = getattr(prediction, "token", "")
            if isinstance(token, str):
                _maybe_add(token)

    return [TokenMappingRow(token=token, mapped_id=mapped_id) for token, mapped_id in seen.items()]
