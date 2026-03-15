from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from shared.config import parse_config

from .runtime.pipeline import run_extract, run_mix, run_stage_a_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset/Distill runtime CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract")
    extract.add_argument("--config", required=True)
    extract.add_argument("--output-dir", required=True)

    mix = sub.add_parser("mix")
    mix.add_argument("--config", required=True)
    mix.add_argument("--input", required=True)
    mix.add_argument("--output", required=True)

    stage_a = sub.add_parser("stage-a")
    stage_a.add_argument("--config", required=True)
    stage_a.add_argument("--input")
    stage_a.add_argument("--output", required=True)

    summary = sub.add_parser("summary")
    summary.add_argument("--config", required=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "summary":
        config = parse_config(args.config)
        print(
            "summary",
            f"dataset_entries={len(config.dataset.source_extraction.dataset_entries)}",
            f"groups={len(config.dataset.mixture_build.groups)}",
            f"teachers={len(config.dataset.distillation.teachers.teachers)}",
            sep=" ",
        )
        return

    if args.command == "extract":
        count = run_extract(args.config, args.output_dir)
        print(f"extract completed: rows={count}, output_dir={args.output_dir}")
        return

    if args.command == "mix":
        count = run_mix(args.config, args.input, args.output)
        print(f"mix completed: rows={count}, output={args.output}")
        return

    if args.command == "stage-a":
        if args.input:
            count = run_stage_a_command(args.config, args.input, args.output)
            print(f"stage-a completed: rows={count}, output={args.output}")
            return

        with tempfile.TemporaryDirectory(prefix="distill-stage-a-") as tmp_dir:
            tmp_root = Path(tmp_dir)
            extracted = tmp_root / "extracted"
            mixed = tmp_root / "mixed.jsonl"
            run_extract(args.config, extracted)
            run_mix(args.config, extracted, mixed)
            count = run_stage_a_command(args.config, mixed, args.output)
        print(f"stage-a completed: rows={count}, output={args.output}")
        return


if __name__ == "__main__":
    main()
