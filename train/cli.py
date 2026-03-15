from __future__ import annotations

import argparse
import json

from train.artifacts import read_model_artifact, write_build_artifact, write_training_artifact
from train.runtime import load_model_runtime, run_smoke, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description='Train CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    build_parser = subparsers.add_parser('build')
    build_parser.add_argument('--config', required=True, help='Path to XML config')
    build_parser.add_argument('--dataset-file', required=True, help='Path to prepared dataset file')
    build_parser.add_argument('--output-dir', required=True, help='Directory where trained model artifacts are written')

    package_parser = subparsers.add_parser('package')
    package_parser.add_argument('--config', required=True, help='Path to XML config')
    package_parser.add_argument('--dataset-file', required=True, help='Path to prepared dataset file')
    package_parser.add_argument('--output-dir', required=True, help='Directory where the model manifest artifact is written')

    summary_parser = subparsers.add_parser('summary')
    summary_parser.add_argument('--model-file', required=True, help='Path to a built model artifact')

    smoke_parser = subparsers.add_parser('smoke')
    smoke_parser.add_argument('--config', required=True, help='Path to XML config')
    smoke_parser.add_argument('--text', default='hello train runtime')

    args = parser.parse_args()

    if args.command == 'build':
        model_runtime = load_model_runtime(args.config)
        training = train_model(model_runtime=model_runtime, dataset_file=args.dataset_file, output_dir=args.output_dir)
        model_file = write_training_artifact(
            model_runtime=model_runtime,
            dataset_file=args.dataset_file,
            output_dir=args.output_dir,
            weights_file=training['weights_file'],
            training={
                'configured_steps': training['configured_steps'],
                'steps': training['steps'],
                'dataset_examples': training['dataset_examples'],
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
    elif args.command == 'summary':
        model_artifact = read_model_artifact(args.model_file)
        print(json.dumps(model_artifact['summary'], indent=2))
    else:
        model_runtime = load_model_runtime(args.config)
        state = run_smoke(model_runtime, args.text)
        print(json.dumps({'trace': state.execution_trace, 'moe_metrics': state.moe_metrics}, indent=2))


if __name__ == '__main__':
    main()
