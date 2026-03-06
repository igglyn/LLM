"""Tests for TOML config loading into dataclass-based config objects."""

from pathlib import Path

from llm_lab.config.defaults import load_config


def _write_config(tmp_path: Path, text: str) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(text)
    return config_path


def test_loads_patcher1_stop_gradient_from_toml(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
[data]
path = "dummy.txt"

[model.codec]
name = "identity"

[model.patcher1]
name = "chunk"
stop_gradient = true

[model.head]
name = "byte"
""".strip(),
    )

    config = load_config(str(config_path))

    assert config.model.patcher1.stop_gradient is True


def test_loads_patcher2_stop_gradient_from_toml(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
[data]
path = "dummy.txt"

[model.codec]
name = "identity"

[model.patcher1]
name = "chunk"

[model.patcher2]
name = "hidden_chunk"
stop_gradient = true

[model.head]
name = "byte"
""".strip(),
    )

    config = load_config(str(config_path))

    assert config.model.patcher2 is not None
    assert config.model.patcher2.stop_gradient is True
