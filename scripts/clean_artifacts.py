#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import shutil

from blt_lite.utils import load_config


def _existing_unique(paths: list[Path]) -> list[Path]:
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        k = str(p.resolve())
        if k in seen:
            continue
        seen.add(k)
        if p.exists():
            uniq.append(p)
    return uniq


def _delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove generated processed/output artifacts for a config.")
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/tiny.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be deleted")
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Also delete data.raw_path (disabled by default for safety)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})

    targets = [
        Path(data_cfg.get("processed_dir", "data/processed")),
        Path(data_cfg.get("processed_dir_patcher", "data/processed/patcher")),
        Path(data_cfg.get("processed_dir_patcher2", "data/processed/patcher2")),
        Path(data_cfg.get("processed_dir_tiny", "data/processed/tiny")),
        Path(cfg.get("train", {}).get("out_dir", "outputs")),
        Path(cfg.get("patcher_train", {}).get("out_dir", "outputs/patcher")),
        Path(cfg.get("patcher2_train", {}).get("out_dir", "outputs/patcher2")),
    ]

    if args.include_raw:
        targets.append(Path(data_cfg.get("raw_path", "data/raw")))

    existing = _existing_unique(targets)

    if not existing:
        print("No artifact paths found to clean.")
        return

    mode = "[DRY RUN]" if args.dry_run else "[DELETE]"
    print(f"{mode} Removing {len(existing)} path(s):")
    for p in existing:
        print(f" - {p}")

    if args.dry_run:
        return

    for p in existing:
        _delete_path(p)

    print("Cleanup complete.")


if __name__ == "__main__":
    main()
