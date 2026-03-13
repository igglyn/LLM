from __future__ import annotations

from shared.config import parse_config, resolve_config
from train.builder import build_model_runtime
from train.metrics import summarize_model_runtime
from train.specs import ModelRuntime, RuntimeState


def load_model_runtime(config_path: str) -> ModelRuntime:
    return build_model_runtime(resolve_config(parse_config(config_path)))


def run_smoke(model_runtime: ModelRuntime, text: str) -> RuntimeState:
    return model_runtime.smoke(text)


__all__ = ["ModelRuntime", "RuntimeState", "build_model_runtime", "load_model_runtime", "run_smoke", "summarize_model_runtime"]
