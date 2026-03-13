from __future__ import annotations

import argparse
import json

from train.metrics import summarize_model_runtime
from train.runtime import load_model_runtime, run_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("build", "summary", "smoke"):
        cmd_parser = subparsers.add_parser(command)
        cmd_parser.add_argument("--config", required=True, help="Path to XML config")
    subparsers.choices["smoke"].add_argument("--text", default="hello train runtime")

    args = parser.parse_args()
    model_runtime = load_model_runtime(args.config)

    if args.command == "build":
        print(f"build completed: patchers={len(model_runtime.patchers)}, trunk_blocks={len(model_runtime.trunk.blocks)}")
    elif args.command == "summary":
        print(json.dumps(summarize_model_runtime(model_runtime), indent=2))
    else:
        state = run_smoke(model_runtime, args.text)
        print(json.dumps({"trace": state.execution_trace, "moe_metrics": state.moe_metrics}, indent=2))
