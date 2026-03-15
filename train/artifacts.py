from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from train.metrics import summarize_model_runtime
from train.specs import ModelRuntime


def write_model_artifact(model_runtime: ModelRuntime, dataset_file: str, output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_file = output_path / "model.json"
    model_file.write_text(
        json.dumps(
            {
                "dataset_file": dataset_file,
                "summary": summarize_model_runtime(model_runtime),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model_file


def read_model_artifact(model_file: str) -> dict[str, Any]:
    return json.loads(Path(model_file).read_text(encoding="utf-8"))

