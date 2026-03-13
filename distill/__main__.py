from __future__ import annotations

import argparse
import os

from .runtime.pipeline import run_extract, run_mix, run_stage_a_command


def _required_output(arg_value: str | None, env_name: str) -> str:
    output = arg_value or os.environ.get(env_name)
    if output is None:
        raise SystemExit(f"Missing output path. Provide --output or set {env_name}.")
    return output


def _required_input(arg_value: str | None, env_name: str, purpose: str) -> str:
    value = arg_value or os.environ.get(env_name)
    if value is None:
        raise SystemExit(f"Missing {purpose} path. Provide CLI arg or set {env_name}.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="run source extraction")
    extract.add_argument("--config", required=True, help="Path to XML config")
    extract.add_argument("--output", help="Path for extracted JSONL")

    mix = subparsers.add_parser("mix", help="run mixture build")
    mix.add_argument("--config", required=True, help="Path to XML config")
    mix.add_argument("--input", help="Extracted JSONL path")
    mix.add_argument("--output", help="Mixed JSONL output path")

    stage_a = subparsers.add_parser("stage-a", help="run StageA distillation")
    stage_a.add_argument("--config", required=True, help="Path to XML config")
    stage_a.add_argument("--input", help="Mixed JSONL path")
    stage_a.add_argument("--output", help="StageA JSONL output path")

    args = parser.parse_args()

    if args.command == "extract":
        output = _required_output(args.output, "DISTILL_EXTRACT_OUTPUT")
        count = run_extract(args.config, output)
        print(f"extract completed: rows={count}, output={output}")
        return

    if args.command == "mix":
        input_path = _required_input(args.input, "DISTILL_EXTRACT_INPUT", "extract input")
        output = _required_output(args.output, "DISTILL_MIX_OUTPUT")
        count = run_mix(args.config, input_path, output)
        print(f"mix completed: rows={count}, output={output}")
        return

    if args.command == "stage-a":
        input_path = _required_input(args.input, "DISTILL_MIX_INPUT", "mix input")
        output = _required_output(args.output, "DISTILL_STAGE_A_OUTPUT")
        count = run_stage_a_command(args.config, input_path, output)
        print(f"stage-a completed: rows={count}, output={output}")
        return


if __name__ == "__main__":
    main()
