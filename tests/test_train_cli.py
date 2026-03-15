from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_writes_model_artifact_and_summary_reads_it(tmp_path: Path) -> None:
    config_path = Path("examples/config.example.xml")
    dataset_file = tmp_path / "dataset.jsonl"
    dataset_file.write_text('{"text":"sample"}\n', encoding="utf-8")
    output_dir = tmp_path / "model"
    model_file = output_dir / "model.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "train",
            "build",
            "--config",
            str(config_path),
            "--dataset-file",
            str(dataset_file),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    artifact = json.loads(model_file.read_text(encoding="utf-8"))
    assert artifact["dataset_file"] == str(dataset_file)

    summary = subprocess.check_output(
        [sys.executable, "-m", "train", "summary", "--model-file", str(model_file)],
        text=True,
    )
    summary_obj = json.loads(summary)

    assert summary_obj["patcher_count"] == 2
    assert summary_obj["trunk"]["name"] == "main_trunk"
