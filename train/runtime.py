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


def train_model(
    model_runtime: ModelRuntime,
    dataset_file: str,
    output_dir: str,
    token_mapping_file: str,
    max_steps: int = 1000,
) -> dict[str, Any]:
    texts = _load_dataset_texts(dataset_file)
    configured_steps = max([model_runtime.trunk.train_config.steps, *[p.train_config.steps for p in model_runtime.patchers], 1])
    steps = min(configured_steps, max_steps)
    d_model = _infer_d_model(model_runtime)

    token_to_id = _load_token_mapping(token_mapping_file)
    _validate_vocab_size(model_runtime, token_to_id)
    _tokenize = build_tokenizer(token_to_id)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    weights = torch.nn.Parameter(torch.zeros(d_model, dtype=torch.float32))
    optimizer = torch.optim.AdamW([weights], lr=0.05, weight_decay=model_runtime.trunk.train_config.weight_decay)

    _ = _tokenize

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


def _load_token_mapping(token_mapping_file: str) -> dict[str, int]:
    rows: list[dict[str, Any]] = []
    for line in Path(token_mapping_file).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError(f"token mapping row must be an object: {token_mapping_file}")
        rows.append(parsed)

    if not rows:
        raise ValueError(f"token mapping file has no rows: {token_mapping_file}")

    token_to_id: dict[str, int] = {}
    for row in rows:
        token = row.get("token")
        mapped_id = row.get("mapped_id")
        if not isinstance(token, str) or not token:
            raise ValueError(f"token mapping row missing non-empty 'token': {token_mapping_file}")
        if not isinstance(mapped_id, int) or mapped_id < 0:
            raise ValueError(f"token mapping row missing non-negative integer 'mapped_id': {token_mapping_file}")
        if token in token_to_id and token_to_id[token] != mapped_id:
            raise ValueError(f"token mapping has conflicting ids for token '{token}'.")
        token_to_id[token] = mapped_id

    return token_to_id


def _validate_vocab_size(model_runtime: ModelRuntime, token_to_id: dict[str, int]) -> None:
    vocab_sizes: set[int] = set()

    for patcher in model_runtime.patchers:
        for block in patcher.blocks:
            vocab_size = getattr(block, "vocab_size", None)
            if isinstance(vocab_size, int) and vocab_size > 0:
                vocab_sizes.add(vocab_size)

    for block in model_runtime.trunk.blocks:
        vocab_size = getattr(block, "vocab_size", None)
        if isinstance(vocab_size, int) and vocab_size > 0:
            vocab_sizes.add(vocab_size)

    if not vocab_sizes:
        return
    if len(vocab_sizes) != 1:
        raise ValueError(f"model has conflicting vocab_size values: {sorted(vocab_sizes)}")

    expected_vocab_size = next(iter(vocab_sizes))
    if len(token_to_id) != expected_vocab_size:
        raise ValueError(
            f"token mapping dictionary size mismatch: expected={expected_vocab_size} got={len(token_to_id)}"
        )


def build_tokenizer(token_to_id: dict[str, int]):
    def tokenize(text: str) -> list[int]:
        token_ids: list[int] = []
        for token in text.split():
            if token not in token_to_id:
                raise ValueError(f"token '{token}' missing from token mapping dictionary")
            token_ids.append(token_to_id[token])
        return token_ids

    return tokenize


__all__ = [
    'ModelRuntime',
    'RuntimeState',
    'build_model_runtime',
    'load_model_runtime',
    'run_smoke',
    'run_dummy_forward',
    'train_model',
    'run_trained_model',
    'build_tokenizer',
]
