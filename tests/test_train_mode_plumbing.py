"""Tests for train.mode plumbing through training entrypoint."""

from __future__ import annotations

import sys
import types
from copy import deepcopy

import pytest

import scripts.train as train_script
from llm_lab.config.defaults import DEFAULT_CONFIG


class _FakeDataLoader:
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs

    def __iter__(self):
        return iter([])


def _install_import_stubs(monkeypatch: pytest.MonkeyPatch, captures: dict[str, object]) -> None:
    torch_mod = types.ModuleType("torch")
    torch_mod.manual_seed = lambda seed: None
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_mod.nn = types.SimpleNamespace(Module=object)

    torch_utils_mod = types.ModuleType("torch.utils")
    torch_utils_data_mod = types.ModuleType("torch.utils.data")
    torch_utils_data_mod.DataLoader = _FakeDataLoader

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils_mod)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_utils_data_mod)

    byte_dataset_mod = types.ModuleType("llm_lab.data.byte_dataset")

    class _ByteDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __len__(self):
            return 2

    byte_dataset_mod.ByteDataset = _ByteDataset
    monkeypatch.setitem(sys.modules, "llm_lab.data.byte_dataset", byte_dataset_mod)

    collate_mod = types.ModuleType("llm_lab.data.collate")
    collate_mod.collate_batch = lambda batch: batch
    monkeypatch.setitem(sys.modules, "llm_lab.data.collate", collate_mod)

    precomputed_mod = types.ModuleType("llm_lab.data.precomputed_patch_dataset")

    class _PrecomputedPatchDataset:
        def __init__(self, path):
            self.path = path

        def __len__(self):
            return 2

    precomputed_mod.PrecomputedPatchDataset = _PrecomputedPatchDataset
    monkeypatch.setitem(sys.modules, "llm_lab.data.precomputed_patch_dataset", precomputed_mod)

    models_mod = types.ModuleType("llm_lab.models.assemble")

    class _FakeModel:
        def to(self, device):
            return self

    models_mod.assemble_model = lambda cfg: _FakeModel()
    monkeypatch.setitem(sys.modules, "llm_lab.models.assemble", models_mod)

    optim_mod = types.ModuleType("llm_lab.train.optim")

    def _build_optimizer(model, lr, weight_decay=0.0, mode="full"):
        captures["optimizer_mode"] = mode
        return object()

    optim_mod.build_optimizer = _build_optimizer
    monkeypatch.setitem(sys.modules, "llm_lab.train.optim", optim_mod)

    loop_mod = types.ModuleType("llm_lab.train.loop")

    def _train_loop(**kwargs):
        captures["loop_mode"] = kwargs.get("mode")
        return {"step": 1, "loss": 0.0}

    loop_mod.train_loop = _train_loop
    monkeypatch.setitem(sys.modules, "llm_lab.train.loop", loop_mod)


def test_train_mode_patcher_only_is_plumbed_to_optimizer_and_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    captures: dict[str, object] = {}
    _install_import_stubs(monkeypatch, captures)

    cfg = deepcopy(DEFAULT_CONFIG)
    cfg.train.mode = "patcher_only"
    cfg.train.steps = 1
    cfg.data.path = "dummy"
    cfg.data.use_precomputed_patches = False
    cfg.data.batch_size = 1

    monkeypatch.setattr(train_script, "load_config", lambda path: cfg)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "dummy.toml"])

    train_script.main()

    assert captures["optimizer_mode"] == "patcher_only"
    assert captures["loop_mode"] == "patcher_only"


def test_precomputed_path_rejects_patcher_only_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    captures: dict[str, object] = {}
    _install_import_stubs(monkeypatch, captures)

    cfg = deepcopy(DEFAULT_CONFIG)
    cfg.train.mode = "patcher_only"
    cfg.train.steps = 1
    cfg.data.use_precomputed_patches = True
    cfg.data.precomputed_patches_path = "dummy.npz"

    monkeypatch.setattr(train_script, "load_config", lambda path: cfg)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "dummy.toml"])

    with pytest.raises(ValueError, match="patcher_only.*precomputed patches"):
        train_script.main()

    assert "optimizer_mode" not in captures
