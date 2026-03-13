from __future__ import annotations

import argparse
import os
from pathlib import Path

from shared.config import parse_config, resolve_config

from .extraction import run_extraction
from .io_jsonl import read_jsonl, write_jsonl
from .mixture import run_mixture
from .schemas import MixtureEntry, NormalizedDocument
from .stages import run_stage_a


def _required_path(value: str | None, env_name: str, label: str) -> str:
    resolved = value or os.environ.get(env_name)
    if resolved is None:
        raise SystemExit(f"Missing {label}. Provide CLI flag or set {env_name}.")
    return resolved


def _load_resolved(config_path: str | Path):
    raw = parse_config(config_path)
    return resolve_config(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset/Distill runtime CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract")
    extract.add_argument("--config", required=True)
    extract.add_argument("--output")

    mix = sub.add_parser("mix")
    mix.add_argument("--config", required=True)
    mix.add_argument("--input")
    mix.add_argument("--output")

    stage_a = sub.add_parser("stage-a")
    stage_a.add_argument("--config", required=True)
    stage_a.add_argument("--input")
    stage_a.add_argument("--output")

    summary = sub.add_parser("summary")
    summary.add_argument("--config", required=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    resolved = _load_resolved(args.config)

    if args.command == "summary":
        dataset = resolved.dataset
        print(
            "summary",
            f"dataset_entries={len(dataset.source_extraction.dataset_entries)}",
            f"groups={len(dataset.mixture_build.groups)}",
            f"teachers={len(dataset.distillation.teachers.teachers)}",
            sep=" ",
        )
        return

    if args.command == "extract":
        output = _required_path(args.output, "DISTILL_EXTRACT_OUTPUT", "extract output path")
        docs = run_extraction(resolved)
        write_jsonl(output, docs)
        print(f"extract completed: rows={len(docs)}, output={output}")
        return

    if args.command == "mix":
        input_path = _required_path(args.input, "DISTILL_EXTRACT_INPUT", "extract input path")
        output = _required_path(args.output, "DISTILL_MIX_OUTPUT", "mix output path")
        docs = [NormalizedDocument(**row) for row in read_jsonl(input_path)]
        mixed = run_mixture(resolved, docs)
        write_jsonl(output, mixed)
        print(f"mix completed: rows={len(mixed)}, output={output}")
        return

    if args.command == "stage-a":
        input_path = _required_path(args.input, "DISTILL_MIX_INPUT", "mixture input path")
        output = _required_path(args.output, "DISTILL_STAGE_A_OUTPUT", "stage-a output path")
        mixed = [MixtureEntry(**row) for row in read_jsonl(input_path)]
        rows = run_stage_a(resolved, mixed)
        write_jsonl(output, rows)
        print(f"stage-a completed: rows={len(rows)}, output={output}")
        return


if __name__ == "__main__":
    main()
