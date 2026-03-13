from __future__ import annotations

import ast
from pathlib import Path


def test_distill_does_not_import_train_runtime() -> None:
    for path in _python_files(Path("distill")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = _imports(tree)
        assert not any(name.startswith("train.runtime") for name in imports), f"{path} imports train.runtime"


def test_train_does_not_import_distill_runtime() -> None:
    for path in _python_files(Path("train")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = _imports(tree)
        assert not any(name.startswith("distill.runtime") for name in imports), f"{path} imports distill.runtime"


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _imports(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.append(node.module)
    return names
