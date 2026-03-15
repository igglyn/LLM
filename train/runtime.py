from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from shared.config import parse_config, resolve_config
from train.builder import build_model_runtime
from train.specs import ModelRuntime, RuntimeState


def load_model_runtime(config_path: str) -> ModelRuntime:
    return build_model_runtime(resolve_config(parse_config(config_path)))


def run_smoke(model_runtime: ModelRuntime, text: str) -> RuntimeState:
    return model_runtime.smoke(text)


def run_dummy_forward(model_runtime: ModelRuntime, batch_size: int, seq_len: int, d_model: int) -> RuntimeState:
    return model_runtime.forward_dummy(batch_size=batch_size, seq_len=seq_len, d_model=d_model)


def train_model(model_runtime: ModelRuntime, dataset_file: str, output_dir: str, max_steps: int = 1000) -> dict[str, Any]:
    texts = _load_dataset_texts(dataset_file)
    configured_steps = max([model_runtime.trunk.train_config.steps, *[p.train_config.steps for p in model_runtime.patchers], 1])
    steps = min(configured_steps, max_steps)
    d_model = _infer_d_model(model_runtime)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    weights = torch.nn.Parameter(torch.zeros(d_model, dtype=torch.float32))
    optimizer = torch.optim.AdamW([weights], lr=0.05, weight_decay=model_runtime.trunk.train_config.weight_decay)

    losses: list[float] = []
    for step in range(steps):
        text = texts[step % len(texts)]
        state = model_runtime.forward_dummy(batch_size=1, seq_len=1, d_model=d_model)
        features = state.tensor.reshape(-1)
        target = _target_from_text(text, d_model)

        prediction = features * torch.tanh(weights)
        loss = torch.mean((prediction - target) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))

    weights_file = output_path / 'weights.pt'
    torch.save({'weights': weights.detach().cpu()}, weights_file)

    return {
        'configured_steps': configured_steps,
        'steps': steps,
        'dataset_examples': len(texts),
        'final_loss': losses[-1],
        'initial_loss': losses[0],
        'weights_file': str(weights_file),
    }


def run_trained_model(model_runtime: ModelRuntime, weights_file: str, text: str) -> dict[str, Any]:
    d_model = _infer_d_model(model_runtime)
    saved = torch.load(weights_file, map_location='cpu')
    weights = saved.get('weights')
    if not isinstance(weights, torch.Tensor):
        raise ValueError(f'weights file is missing tensor: {weights_file}')
    if weights.numel() != d_model:
        raise ValueError(f'weights dimension mismatch: expected={d_model} got={weights.numel()}')

    state = model_runtime.forward_dummy(batch_size=1, seq_len=1, d_model=d_model)
    features = state.tensor.reshape(-1)
    logits = features * torch.tanh(weights.reshape(-1))
    score = float(torch.mean(logits).detach().cpu())

    return {
        'text': text,
        'score': score,
        'd_model': d_model,
        'trace': state.execution_trace,
        'moe_metrics': state.moe_metrics,
    }


def _infer_d_model(model_runtime: ModelRuntime) -> int:
    for patcher in model_runtime.patchers:
        for block in patcher.blocks:
            d_model = getattr(block, 'd_model', None)
            if isinstance(d_model, int) and d_model > 0:
                return d_model
    for block in model_runtime.trunk.blocks:
        d_model = getattr(block, 'd_model', None)
        if isinstance(d_model, int) and d_model > 0:
            return d_model
    return 128


def _load_dataset_texts(dataset_file: str) -> list[str]:
    texts: list[str] = []
    for line in Path(dataset_file).read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and isinstance(parsed.get('text'), str):
            texts.append(parsed['text'])
        else:
            texts.append(str(parsed))
    if not texts:
        raise ValueError(f'dataset file has no usable rows: {dataset_file}')
    return texts


def _target_from_text(text: str, d_model: int) -> torch.Tensor:
    values = [((ord(ch) % 97) / 96.0) for ch in text]
    if not values:
        values = [0.0]
    repeated = (values * ((d_model // len(values)) + 1))[:d_model]
    return torch.tensor(repeated, dtype=torch.float32)


__all__ = [
    'ModelRuntime',
    'RuntimeState',
    'build_model_runtime',
    'load_model_runtime',
    'run_smoke',
    'run_dummy_forward',
    'train_model',
    'run_trained_model',
]
