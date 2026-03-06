"""Registry behavior tests."""

import pytest

from llm_lab.registry import (
    build_component,
    get_codec,
    get_head,
    get_memory,
    get_mixer,
    get_patcher,
    register_codec,
    register_head,
    register_memory,
    register_mixer,
    register_patcher,
)


@register_codec("dummy_codec")
class DummyCodec:
    def __init__(self, value: int = 1) -> None:
        self.value = value


@register_patcher("dummy_patcher")
class DummyPatcher:
    pass


@register_mixer("dummy_mixer")
class DummyMixer:
    pass


@register_head("dummy_head")
class DummyHead:
    pass


@register_memory("dummy_memory")
class DummyMemory:
    pass


def test_get_registered_components() -> None:
    assert get_codec("dummy_codec") is DummyCodec
    assert get_patcher("dummy_patcher") is DummyPatcher
    assert get_mixer("dummy_mixer") is DummyMixer
    assert get_head("dummy_head") is DummyHead
    assert get_memory("dummy_memory") is DummyMemory


def test_build_component_instantiates_class() -> None:
    instance = build_component("codec", "dummy_codec", value=7)
    assert isinstance(instance, DummyCodec)
    assert instance.value == 7


def test_missing_component_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown codec"):
        get_codec("missing_codec")


def test_unknown_kind_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown component kind"):
        build_component("unknown_kind", "anything")
