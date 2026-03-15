from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_trains_and_summary_reads_artifact(tmp_path: Path) -> None:
    config_path = Path('examples/config.example.xml')
    dataset_file = tmp_path / 'dataset.jsonl'
    dataset_file.write_text('{"text":"sample one"}\n{"text":"sample two"}\n', encoding='utf-8')
    output_dir = tmp_path / 'model'
    model_file = output_dir / 'model.json'

    subprocess.run(
        [
            sys.executable,
            '-m',
            'train',
            'build',
            '--config',
            str(config_path),
            '--dataset-file',
            str(dataset_file),
            '--output-dir',
            str(output_dir),
        ],
        check=True,
    )

    artifact = json.loads(model_file.read_text(encoding='utf-8'))
    assert artifact['dataset_file'] == str(dataset_file)
    assert Path(artifact['weights_file']).exists()
    assert artifact['training']['configured_steps'] == 50000
    assert artifact['training']['steps'] == 1000
    assert artifact['training']['final_loss'] <= artifact['training']['initial_loss']

    summary = subprocess.check_output(
        [sys.executable, '-m', 'train', 'summary', '--model-file', str(model_file)],
        text=True,
    )
    summary_obj = json.loads(summary)

    assert summary_obj['patcher_count'] == 2
    assert summary_obj['trunk']['name'] == 'main_trunk'


def test_run_uses_trained_artifact(tmp_path: Path) -> None:
    config_path = Path('examples/config.example.xml')
    dataset_file = tmp_path / 'dataset.jsonl'
    dataset_file.write_text('{"text":"sample one"}\n{"text":"sample two"}\n', encoding='utf-8')
    output_dir = tmp_path / 'model'
    model_file = output_dir / 'model.json'

    subprocess.run(
        [
            sys.executable,
            '-m',
            'train',
            'build',
            '--config',
            str(config_path),
            '--dataset-file',
            str(dataset_file),
            '--output-dir',
            str(output_dir),
        ],
        check=True,
    )

    output = subprocess.check_output(
        [
            sys.executable,
            '-m',
            'train',
            'run',
            '--config',
            str(config_path),
            '--model-file',
            str(model_file),
            '--text',
            'run payload',
        ],
        text=True,
    )
    result = json.loads(output)

    assert result['text'] == 'run payload'
    assert isinstance(result['score'], float)
    assert result['d_model'] > 0
    assert 'TrunkEnd(main_trunk)' in result['trace']


def test_package_writes_manifest_only(tmp_path: Path) -> None:
    config_path = Path('examples/config.example.xml')
    dataset_file = tmp_path / 'dataset.jsonl'
    dataset_file.write_text('{"text":"sample"}\n', encoding='utf-8')
    output_dir = tmp_path / 'model'
    model_file = output_dir / 'model.json'

    subprocess.run(
        [
            sys.executable,
            '-m',
            'train',
            'package',
            '--config',
            str(config_path),
            '--dataset-file',
            str(dataset_file),
            '--output-dir',
            str(output_dir),
        ],
        check=True,
    )

    artifact = json.loads(model_file.read_text(encoding='utf-8'))
    assert artifact['dataset_file'] == str(dataset_file)
    assert 'weights_file' not in artifact
    assert 'training' not in artifact
