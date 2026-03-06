"""Component registry for pluggable llm_lab modules."""

from collections.abc import Callable
from typing import Any

_CODEC_REGISTRY: dict[str, type] = {}
_PATCHER_REGISTRY: dict[str, type] = {}
_MIXER_REGISTRY: dict[str, type] = {}
_HEAD_REGISTRY: dict[str, type] = {}
_MEMORY_REGISTRY: dict[str, type] = {}


def _register(registry: dict[str, type], kind: str, name: str) -> Callable[[type], type]:
    """Return a decorator that registers a class under ``name``."""

    def decorator(cls: type) -> type:
        registry[name] = cls
        return cls

    return decorator


def _get(registry: dict[str, type], kind: str, name: str) -> type:
    """Get a registered class by name or raise a clear error."""
    try:
        return registry[name]
    except KeyError as exc:
        available = ", ".join(sorted(registry)) or "<none>"
        raise ValueError(
            f"Unknown {kind} '{name}'. Available {kind}s: {available}."
        ) from exc


def register_codec(name: str) -> Callable[[type], type]:
    """Decorator to register a codec class."""
    return _register(_CODEC_REGISTRY, "codec", name)


def register_patcher(name: str) -> Callable[[type], type]:
    """Decorator to register a patcher class."""
    return _register(_PATCHER_REGISTRY, "patcher", name)


def register_mixer(name: str) -> Callable[[type], type]:
    """Decorator to register a mixer class."""
    return _register(_MIXER_REGISTRY, "mixer", name)


def register_head(name: str) -> Callable[[type], type]:
    """Decorator to register a head class."""
    return _register(_HEAD_REGISTRY, "head", name)


def register_memory(name: str) -> Callable[[type], type]:
    """Decorator to register a memory class."""
    return _register(_MEMORY_REGISTRY, "memory", name)


def get_codec(name: str) -> type:
    """Get a registered codec class by name."""
    return _get(_CODEC_REGISTRY, "codec", name)


def get_patcher(name: str) -> type:
    """Get a registered patcher class by name."""
    return _get(_PATCHER_REGISTRY, "patcher", name)


def get_mixer(name: str) -> type:
    """Get a registered mixer class by name."""
    return _get(_MIXER_REGISTRY, "mixer", name)


def get_head(name: str) -> type:
    """Get a registered head class by name."""
    return _get(_HEAD_REGISTRY, "head", name)


def get_memory(name: str) -> type:
    """Get a registered memory class by name."""
    return _get(_MEMORY_REGISTRY, "memory", name)


def build_component(kind: str, name: str, **kwargs: Any) -> Any:
    """Instantiate a registered component by kind and name.

    Args:
        kind: One of ``codec``, ``patcher``, ``mixer``, ``head``, ``memory``.
        name: Registered component name.
        **kwargs: Constructor keyword arguments.

    Returns:
        An instance of the requested component class.

    Raises:
        ValueError: If ``kind`` or ``name`` is unknown.
    """
    getters: dict[str, Callable[[str], type]] = {
        "codec": get_codec,
        "patcher": get_patcher,
        "mixer": get_mixer,
        "head": get_head,
        "memory": get_memory,
    }
    try:
        cls = getters[kind](name)
    except KeyError as exc:
        valid = ", ".join(sorted(getters))
        raise ValueError(f"Unknown component kind '{kind}'. Valid kinds: {valid}.") from exc
    return cls(**kwargs)


def build_mixer_stack(mixers: list[Any]) -> list[Any]:
    """Instantiate a list of mixer components in order."""
    built = []
    for mixer_cfg in mixers:
        name = getattr(mixer_cfg, "name", None)
        kwargs = getattr(mixer_cfg, "kwargs", {})
        if not isinstance(name, str) or not name:
            raise ValueError("Each mixer config must define a non-empty 'name'.")
        if not isinstance(kwargs, dict):
            raise ValueError("Each mixer config must define dict 'kwargs'.")
        built.append(build_component("mixer", name, **kwargs))
    return built
