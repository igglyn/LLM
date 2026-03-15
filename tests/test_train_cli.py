from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


def test_build_trains_and_summary_reads_artifact(tmp_path: Path) -> None:
    config_path = Path('examples/config.example.xml')
    distill_dir = _write_distill_dir(tmp_path)
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
            '--distill-dir',
            str(distill_dir),
            '--output-dir',
            str(output_dir),
            '--max-steps',
            '60',
        ],
        check=True,
    )

    artifact = json.loads(model_file.read_text(encoding='utf-8'))
    assert artifact['dataset_file'] == str(distill_dir)
    assert Path(artifact['weights_file']).exists()
    assert artifact['training']['configured_steps'] == 50000
    assert artifact['training']['steps'] == 60
    assert artifact['training']['batch_size'] == 16
    assert artifact['training']['save_every'] == 10000
    assert artifact['training']['checkpoint_files'] == []
    assert artifact['training']['used_vocab_embedding'] is False
    assert artifact['training']['config_d_model'] == 4096
    assert artifact['training']['config_n_heads'] == 32
    assert artifact['training']['used_positional_embedding'] is True
    assert artifact['training']['token_mapping_file'] == str(distill_dir / 'token_mappings.json')
    assert any(path.endswith('stage_a.jsonl') for path in artifact['training']['data_files'])
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
    distill_dir = _write_distill_dir(tmp_path)
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
            '--distill-dir',
            str(distill_dir),
            '--output-dir',
            str(output_dir),
            '--max-steps',
            '60',
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
            '--max-new-tokens',
            '3',
        ],
        text=True,
    )
    result = json.loads(output)

    assert result['text'] == 'run payload'
    assert isinstance(result['score'], float)
    assert result['d_model'] > 0
    assert isinstance(result['predicted_token'], str)
    assert isinstance(result['predicted_token_id'], int)
    assert 'TrunkEnd(main_trunk)' in result['trace']
    assert result['generation']['max_new_tokens'] == 3
    assert len(result['generation']['generated_tokens']) == 3
    assert len(result['generation']['generated_token_ids']) == 3


def test_build_with_vocab_and_pos_embedding_sets_both_flags(tmp_path: Path) -> None:
    config_path = _write_vocab_and_pos_config(tmp_path)
    distill_dir = _write_distill_dir(tmp_path)
    output_dir = tmp_path / 'model_both_embed'
    model_file = output_dir / 'model.json'

    subprocess.run(
        [
            sys.executable,
            '-m',
            'train',
            'build',
            '--config',
            str(config_path),
            '--distill-dir',
            str(distill_dir),
            '--output-dir',
            str(output_dir),
            '--max-steps',
            '5',
        ],
        check=True,
    )

    artifact = json.loads(model_file.read_text(encoding='utf-8'))
    assert artifact['training']['used_vocab_embedding'] is True
    assert artifact['training']['used_positional_embedding'] is True


def test_build_preserves_small_configured_d_model_and_n_heads(tmp_path: Path) -> None:
    config_path = _write_small_dim_config(tmp_path, d_model=384, n_heads=6)
    distill_dir = _write_distill_dir(tmp_path)
    output_dir = tmp_path / 'model_small_dims'
    model_file = output_dir / 'model.json'

    subprocess.run(
        [
            sys.executable,
            '-m',
            'train',
            'build',
            '--config',
            str(config_path),
            '--distill-dir',
            str(distill_dir),
            '--output-dir',
            str(output_dir),
            '--max-steps',
            '5',
        ],
        check=True,
    )

    artifact = json.loads(model_file.read_text(encoding='utf-8'))
    payload = torch.load(artifact['weights_file'], map_location='cpu')
    meta = payload['meta']

    assert (meta['d_model'], meta['n_heads']) == (384, 6)
    assert meta['prediction_target'] == 'input_reconstruction_plus_next_token'
    assert meta['latent_dim'] == 384
    assert meta['decoder_head'] == 'token_logits_for_inference'


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


def test_build_reads_jsonl_token_mapping_from_distill_dir(tmp_path: Path) -> None:
    config_path = Path('examples/config.example.xml')
    distill_dir = tmp_path / 'distill'
    distill_dir.mkdir(parents=True, exist_ok=True)
    (distill_dir / 'stage_a.jsonl').write_text(
        '\n'.join(
            [
                json.dumps({'prompt_text': 'hello world', 'target_text': 'again'}),
                json.dumps({'prompt_text': 'code completion', 'target_text': 'rocks'}),
            ]
        )
        + '\n',
        encoding='utf-8',
    )
    (distill_dir / 'token_mapping.jsonl').write_text(
        '\n'.join(
            [
                json.dumps({'token': '<pad>', 'mapped_id': 0}),
                json.dumps({'token': '<unk>', 'mapped_id': 1}),
                json.dumps({'token': 'hello', 'mapped_id': 2}),
                json.dumps({'token': 'world', 'mapped_id': 3}),
            ]
        )
        + '\n',
        encoding='utf-8',
    )

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
            '--distill-dir',
            str(distill_dir),
            '--output-dir',
            str(output_dir),
            '--max-steps',
            '10',
        ],
        check=True,
    )

    artifact = json.loads(model_file.read_text(encoding='utf-8'))
    assert artifact['training']['token_mapping_file'] == str(distill_dir / 'token_mapping.jsonl')



def test_build_log_every_prints_step_loss_and_lr(tmp_path: Path) -> None:
    config_path = _write_small_dim_config(tmp_path, d_model=64, n_heads=4)
    distill_dir = _write_distill_dir(tmp_path)
    output_dir = tmp_path / 'model_logs'

    completed = subprocess.run(
        [
            sys.executable,
            '-m',
            'train',
            'build',
            '--config',
            str(config_path),
            '--distill-dir',
            str(distill_dir),
            '--output-dir',
            str(output_dir),
            '--max-steps',
            '4',
            '--log-every',
            '2',
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert 'step=2 loss=' in completed.stdout
    assert 'lr=' in completed.stdout
    assert 'step=4 loss=' in completed.stdout

def _write_distill_dir(tmp_path: Path) -> Path:
    distill_dir = tmp_path / 'distill'
    distill_dir.mkdir(parents=True, exist_ok=True)
    (distill_dir / 'stage_a.jsonl').write_text(
        '\n'.join(
            [
                json.dumps({'prompt_text': 'hello world', 'target_text': 'again'}),
                json.dumps({'prompt_text': 'code completion', 'target_text': 'rocks'}),
            ]
        )
        + '\n',
        encoding='utf-8',
    )
    (distill_dir / 'token_mappings.json').write_text(
        json.dumps({'token_to_id': {'<pad>': 0, '<unk>': 1, 'hello': 2, 'world': 3, 'again': 4, 'code': 5, 'completion': 6, 'rocks': 7}}),
        encoding='utf-8',
    )
    return distill_dir


def _write_vocab_and_pos_config(tmp_path: Path) -> Path:
    path = tmp_path / 'vocab_pos.xml'
    path.write_text(
        """
<Config>
  <Dataset>
    <SourceExtraction>
      <DatasetEntry name="set_local">
        <Source name="local" type="local_text_glob" uri="/tmp/*.txt" />
      </DatasetEntry>
    </SourceExtraction>
    <MixtureBuild><Group name="g" percentage="100"><DatasetRef name="set_local" /></Group></MixtureBuild>
    <Distillation>
      <Teachers>
        <Teacher name="t"><Backend type="dummy_local"><ModelRef name_or_path="dummy" /><Execution device="cpu" precision="fp32" /></Backend></Teacher>
      </Teachers>
      <Stage name="StageA" enabled="true" teacher_ref="t"><TopKLogits k="8" /></Stage>
    </Distillation>
  </Dataset>
  <Model>
    <Defaults d_model="128" n_heads="8" />
    <Patcher name="p1" patch_size="64">
      <Train steps="10"><Optimizer type="adamw" weight_decay="0.0"><Scheduler type="cosine" start_step="0" end_step="10" /></Optimizer></Train>
      <VocabEmbedding vocab_size="32000" />
      <PosEmbedding type="learned" />
      <Transformer layers="1" />
    </Patcher>
    <Trunk name="t1" context="128">
      <Train steps="10"><Optimizer type="adamw" weight_decay="0.0"><Scheduler type="cosine" start_step="0" end_step="10" /></Optimizer></Train>
      <Transformer layers="1" />
    </Trunk>
  </Model>
</Config>
        """.strip(),
        encoding='utf-8',
    )
    return path


def _write_small_dim_config(tmp_path: Path, d_model: int, n_heads: int) -> Path:
    path = tmp_path / 'small_dims.xml'
    path.write_text(
        f"""
<Config>
  <Dataset>
    <SourceExtraction>
      <DatasetEntry name="set_local">
        <Source name="local" type="local_text_glob" uri="/tmp/*.txt" />
      </DatasetEntry>
    </SourceExtraction>
    <MixtureBuild><Group name="g" percentage="100"><DatasetRef name="set_local" /></Group></MixtureBuild>
    <Distillation>
      <Teachers>
        <Teacher name="t"><Backend type="dummy_local"><ModelRef name_or_path="dummy" /><Execution device="cpu" precision="fp32" /></Backend></Teacher>
      </Teachers>
      <Stage name="StageA" enabled="true" teacher_ref="t"><TopKLogits k="8" /></Stage>
    </Distillation>
  </Dataset>
  <Model>
    <Defaults d_model="{d_model}" n_heads="{n_heads}" />
    <Patcher name="p1" patch_size="64">
      <Train steps="10"><Optimizer type="adamw" weight_decay="0.0"><Scheduler type="cosine" start_step="0" end_step="10" /></Optimizer></Train>
      <Transformer layers="1" />
    </Patcher>
    <Trunk name="t1" context="128">
      <Train steps="10"><Optimizer type="adamw" weight_decay="0.0"><Scheduler type="cosine" start_step="0" end_step="10" /></Optimizer></Train>
      <Transformer layers="1" />
    </Trunk>
  </Model>
</Config>
        """.strip(),
        encoding='utf-8',
    )
    return path
