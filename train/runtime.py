from __future__ import annotations

from shared.config import parse_config, resolve_config
from train.builder import build_model_runtime
from train.specs import ModelRuntime, RuntimeState


def load_model_runtime(config_path: str) -> ModelRuntime:
    return build_model_runtime(resolve_config(parse_config(config_path)))


def run_smoke(model_runtime: ModelRuntime, text: str) -> RuntimeState:
    return model_runtime.smoke(text)


def run_dummy_forward(model_runtime: ModelRuntime, batch_size: int, seq_len: int, d_model: int) -> RuntimeState:
    return model_runtime.forward_dummy(batch_size=batch_size, seq_len=seq_len, d_model=d_model)


__all__ = [
    "ModelRuntime",
    "RuntimeState",
    "build_model_runtime",
    "load_model_runtime",
    "run_smoke",
    "run_dummy_forward",
]
