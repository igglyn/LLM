"""Default configuration values and TOML loading."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any
import warnings

from llm_lab.config.schema import (
    ComponentCfg,
    DataCfg,
    DebugCfg,
    ExperimentConfig,
    ModelCfg,
    TrainCfg,
)

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

DEFAULT_CONFIG = ExperimentConfig()


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _component_from_dict(value: Any, *, required: bool) -> ComponentCfg | None:
    if value is None:
        if required:
            raise ValueError("Missing required component config.")
        return None
    if not isinstance(value, dict):
        raise ValueError("Component config must be a table/dict.")

    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Component config requires non-empty 'name'.")

    kwargs = value.get("kwargs", {})
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise ValueError("Component 'kwargs' must be a table/dict when provided.")

    inline_kwargs = {
        k: v
        for k, v in value.items()
        if k not in {"name", "kwargs", "freeze", "stop_gradient"}
    }
    kwargs = {**kwargs, **inline_kwargs}

    freeze = bool(value.get("freeze", False))
    stop_gradient = bool(value.get("stop_gradient", False))
    return ComponentCfg(name=name, kwargs=kwargs, freeze=freeze, stop_gradient=stop_gradient)


def _mixers_from_model_dict(raw_model: dict[str, Any], merged_model: dict[str, Any]) -> list[ComponentCfg]:
    """Resolve mixer stack from config with legacy compatibility.

    Preference order:
    1) explicit ``model.mixers``
    2) explicit legacy ``model.mixer``
    3) default legacy ``model.mixer``
    """
    has_mixers = "mixers" in raw_model
    has_mixer = "mixer" in raw_model

    if has_mixers:
        mixers_raw = raw_model.get("mixers")
        if not isinstance(mixers_raw, list):
            raise ValueError("model.mixers must be an array of tables.")
        if len(mixers_raw) == 0:
            raise ValueError("model.mixers must not be empty.")
        if has_mixer:
            warnings.warn(
                "Both model.mixer and model.mixers were provided; using model.mixers.",
                stacklevel=2,
            )
        mixers: list[ComponentCfg] = []
        for item in mixers_raw:
            comp = _component_from_dict(item, required=True)
            assert comp is not None
            mixers.append(comp)
        return mixers

    if has_mixer:
        single = _component_from_dict(raw_model.get("mixer"), required=True)
        assert single is not None
        return [single]

    # Fallback to merged defaults.
    single_default = _component_from_dict(merged_model.get("mixer"), required=True)
    assert single_default is not None
    return [single_default]


def load_config(path: str) -> ExperimentConfig:
    """Load TOML config, merge with defaults, and return structured config."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    merged = _merge_dict(asdict(DEFAULT_CONFIG), raw)

    data_dict = merged.get("data", {})
    model_dict = merged.get("model", {})
    train_dict = merged.get("train", {})
    debug_dict = merged.get("debug", {})
    raw_model_dict = raw.get("model", {}) if isinstance(raw.get("model", {}), dict) else {}

    mixers = _mixers_from_model_dict(raw_model_dict, model_dict)

    data_cfg = DataCfg(**data_dict)
    model_cfg = ModelCfg(
        codec=_component_from_dict(model_dict.get("codec"), required=True),
        patcher1=_component_from_dict(model_dict.get("patcher1"), required=True),
        # patcher2 is optional for hierarchical patch pipelines.
        patcher2=_component_from_dict(model_dict.get("patcher2"), required=False),
        mixer=mixers[0],
        mixers=mixers,
        head=_component_from_dict(model_dict.get("head"), required=True),
        memory=_component_from_dict(model_dict.get("memory"), required=False),
    )
    train_cfg = TrainCfg(**train_dict)
    debug_cfg = DebugCfg(**debug_dict)

    return ExperimentConfig(data=data_cfg, model=model_cfg, train=train_cfg, debug=debug_cfg)
