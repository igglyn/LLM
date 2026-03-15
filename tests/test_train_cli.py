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
    token_mapping_file = tmp_path / 'token_mapping.jsonl'
    token_mapping_file.write_text('{"token":"sample","mapped_id":0}\n', encoding='utf-8')

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
            '--token-mapping-file',
            str(token_mapping_file),
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
    token_mapping_file = tmp_path / 'token_mapping.jsonl'
    token_mapping_file.write_text('{"token":"sample","mapped_id":0}\n', encoding='utf-8')

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
            '--token-mapping-file',
            str(token_mapping_file),
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


def test_build_fails_when_token_mapping_size_mismatches_vocab(tmp_path: Path) -> None:
    config_path = tmp_path / 'config_vocab.xml'
    config_path.write_text(
        """
<Config>
  <Dataset>
    <SourceExtraction />
    <MixtureBuild />
    <Distillation>
      <Teachers>
        <Teacher name="t">
          <Backend type="dummy_local">
            <ModelRef name_or_path="dummy" />
            <Execution device="cpu" precision="fp32" />
          </Backend>
        </Teacher>
      </Teachers>
      <Stage name="StageA" enabled="true" teacher_ref="t"><TopKLogits k="2" /></Stage>
      <Stage name="StageB" enabled="false" teacher_ref="t"><LongContext max_tokens="64" /></Stage>
      <Stage name="StageC" enabled="false" teacher_ref="t"><StructuredOutputs schema="json" /></Stage>
    </Distillation>
  </Dataset>
  <Model>
    <Defaults d_model="64" n_heads="8" />
    <Patcher name="p1" patch_size="8">
      <Train steps="5"><Optimizer type="adamw" weight_decay="0.0"><Scheduler type="cosine" start_step="0" end_step="5" /></Optimizer></Train>
      <VocabEmbedding vocab_size="3" />
      <Transformer />
    </Patcher>
    <Trunk name="t1" context="64">
      <Train steps="5"><Optimizer type="adamw" weight_decay="0.0"><Scheduler type="cosine" start_step="0" end_step="5" /></Optimizer></Train>
      <Transformer />
    </Trunk>
  </Model>
</Config>
        """.strip(),
        encoding='utf-8',
    )

    dataset_file = tmp_path / 'dataset.jsonl'
    dataset_file.write_text('{"text":"sample one"}\n', encoding='utf-8')
    token_mapping_file = tmp_path / 'token_mapping.jsonl'
    token_mapping_file.write_text('{"token":"a","mapped_id":0}\n{"token":"b","mapped_id":1}\n', encoding='utf-8')
    output_dir = tmp_path / 'model'

    completed = subprocess.run(
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
            '--token-mapping-file',
            str(token_mapping_file),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert 'dictionary size mismatch' in (completed.stderr + completed.stdout)
