from __future__ import annotations

from pathlib import Path

from shared.config import parse_config

from .io_utils import read_jsonl, write_jsonl
from .mixture_build import run_mixture_build
from .source_extraction import run_source_extraction
from .stage_a_runtime import run_stage_a, validate_stage_a_mode
from .types import ExtractedDocument, MixtureDocument


def run_extract(config_path: str | Path, output_path: str | Path) -> int:
    config = parse_config(config_path)
    docs = run_source_extraction(config)
    write_jsonl(output_path, docs)
    return len(docs)


def run_mix(config_path: str | Path, extracted_path: str | Path, output_path: str | Path) -> int:
    config = parse_config(config_path)
    extracted = [ExtractedDocument(**row) for row in read_jsonl(extracted_path)]
    mixed = run_mixture_build(config, extracted)
    write_jsonl(output_path, mixed)
    return len(mixed)


def run_stage_a_command(config_path: str | Path, mixed_path: str | Path, output_path: str | Path) -> int:
    config = parse_config(config_path)
    validate_stage_a_mode(config)
    mixed = [MixtureDocument(**row) for row in read_jsonl(mixed_path)]
    rows = run_stage_a(config, mixed)
    write_jsonl(output_path, rows)
    return len(rows)
