from __future__ import annotations

import argparse
import json

from train.artifacts import read_model_artifact, write_build_artifact, write_training_artifact
from train.runtime import load_model_runtime, run_smoke, run_trained_model, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description='Train CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    build_parser = subparsers.add_parser('build')
    build_parser.add_argument('--config', required=True, help='Path to XML config')
    build_parser.add_argument('--distill-dir', required=True, help='Directory containing distill JSONL data and token mappings')
    build_parser.add_argument('--output-dir', required=True, help='Directory where trained model artifacts are written')
    build_parser.add_argument('--max-steps', type=int, default=None, help='Optional cap on training steps (defaults to config steps)')
    build_parser.add_argument('--resume-from', default=None, help='Optional checkpoint file to resume build training from')

    package_parser = subparsers.add_parser('package')
    package_parser.add_argument('--config', required=True, help='Path to XML config')
    package_parser.add_argument('--dataset-file', required=True, help='Path to prepared dataset file')
    package_parser.add_argument('--output-dir', required=True, help='Directory where the model manifest artifact is written')

    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--config', required=True, help='Path to XML config')
    run_parser.add_argument('--model-file', required=True, help='Path to trained model artifact containing weights_file')
    run_parser.add_argument('--text', default='hello train runtime', help='Text payload for runtime routing context')

    summary_parser = subparsers.add_parser('summary')
    summary_parser.add_argument('--model-file', required=True, help='Path to a built model artifact')

    smoke_parser = subparsers.add_parser('smoke')
    smoke_parser.add_argument('--config', required=True, help='Path to XML config')
    smoke_parser.add_argument('--text', default='hello train runtime')

    args = parser.parse_args()

    if args.command == 'build':
        model_runtime = load_model_runtime(args.config)
        training = train_model(model_runtime=model_runtime, distill_dir=args.distill_dir, output_dir=args.output_dir, max_steps=args.max_steps, resume_from=args.resume_from)
        model_file = write_training_artifact(
            model_runtime=model_runtime,
            dataset_file=args.distill_dir,
            output_dir=args.output_dir,
            weights_file=training['weights_file'],
            training={
                'configured_steps': training['configured_steps'],
                'steps': training['steps'],
                'start_step': training['start_step'],
                'batch_size': training['configured_batch_size'],
                'save_every': training['save_every'],
                'checkpoint_files': training['checkpoint_files'],
                'optimizer_type': training['optimizer_type'],
                'transformer_layers': training['transformer_layers'],
                'moe_expert_count': training['moe_expert_count'],
                'dataset_examples': training['dataset_examples'],
                'data_files': training['data_files'],
                'token_mapping_file': training['token_mapping_file'],
                'used_positional_embedding': training['used_positional_embedding'],
                'initial_loss': training['initial_loss'],
                'final_loss': training['final_loss'],
            },
        )
        print(
            'build completed: '
            f"model_file={model_file} "
            f"weights_file={training['weights_file']} "
            f"final_loss={training['final_loss']:.6f}"
        )
    elif args.command == 'package':
        model_runtime = load_model_runtime(args.config)
        model_file = write_build_artifact(
            model_runtime=model_runtime,
            dataset_file=args.dataset_file,
            output_dir=args.output_dir,
        )
        print(f'package completed: model_file={model_file}')
    elif args.command == 'run':
        model_runtime = load_model_runtime(args.config)
        model_artifact = read_model_artifact(args.model_file)
        weights_file = model_artifact.get('weights_file')
        if not isinstance(weights_file, str):
            raise ValueError('model artifact does not contain a weights_file. Use `train build` output.')
        result = run_trained_model(model_runtime=model_runtime, weights_file=weights_file, text=args.text)
        print(json.dumps(result, indent=2))
    elif args.command == 'summary':
        model_artifact = read_model_artifact(args.model_file)
        print(json.dumps(model_artifact['summary'], indent=2))
    else:
        model_runtime = load_model_runtime(args.config)
        state = run_smoke(model_runtime, args.text)
        print(json.dumps({'trace': state.execution_trace, 'moe_metrics': state.moe_metrics}, indent=2))


if __name__ == '__main__':
    main()
