from __future__ import annotations

import argparse
import json

from shared.config import parse_config, resolve_config
from train.runtime import build_model_runtime, summarize_model_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="build runtime model from config")
    build_parser.add_argument("--config", required=True, help="Path to XML config")

    summary_parser = subparsers.add_parser("summary", help="print runtime summary")
    summary_parser.add_argument("--config", required=True, help="Path to XML config")

    smoke_parser = subparsers.add_parser("smoke", help="run smoke execution")
    smoke_parser.add_argument("--config", required=True, help="Path to XML config")
    smoke_parser.add_argument("--text", default="hello train runtime", help="Smoke input text")

    args = parser.parse_args()

    raw_config = parse_config(args.config)
    resolved_config = resolve_config(raw_config)
    model_runtime = build_model_runtime(resolved_config)

    if args.command == "build":
        print(
            f"build completed: patchers={len(model_runtime.patchers)}, "
            f"trunk_blocks={len(model_runtime.trunk.blocks)}"
        )
        return

    if args.command == "summary":
        print(json.dumps(summarize_model_runtime(model_runtime), indent=2))
        return

    if args.command == "smoke":
        state = model_runtime.smoke(args.text)
        print(
            json.dumps(
                {
                    "trace": state.execution_trace,
                    "moe_metrics": state.moe_metrics,
                },
                indent=2,
            )
        )
        return


if __name__ == "__main__":
    main()
