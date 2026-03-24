"""Optimizer utilities."""

from __future__ import annotations

import torch
from torch import nn


def _set_requires_grad(params: list[nn.Parameter], enabled: bool) -> None:
    for p in params:
        p.requires_grad = enabled


def _apply_train_mode(model: nn.Module, mode: str) -> None:
    if mode not in {"full", "patcher_only"}:
        raise ValueError("train mode must be 'full' or 'patcher_only'.")
    if mode == "full":
        return

    if not hasattr(model, "component_param_groups") or not callable(model.component_param_groups):
        raise ValueError("patcher_only mode requires model.component_param_groups().")

    groups = model.component_param_groups()
    patcher_params = groups.get("patcher1", []) + groups.get("patcher2", [])
    if len(patcher_params) == 0:
        raise ValueError("patcher_only mode requires at least one patcher parameter.")

    # Keep patchers trainable; freeze everything else for dedicated pretraining.
    _set_requires_grad(groups.get("patcher1", []), True)
    _set_requires_grad(groups.get("patcher2", []), True)
    for name, params in groups.items():
        if name not in {"patcher1", "patcher2"}:
            _set_requires_grad(params, False)


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.0,
    mode: str = "full",
) -> torch.optim.Optimizer:
    """Build AdamW with explicit train mode control and logging."""
    _apply_train_mode(model, mode)
    all_params = list(model.parameters())

    print(f"train mode: {mode}")

    # Prefer explicit component grouping when available.
    if hasattr(model, "component_param_counts") and callable(model.component_param_counts):
        counts = model.component_param_counts()
        for name in ["patcher1", "patcher2", "mixers", "head", "reconstruction_head", "memory"]:
            if name in counts:
                c = counts[name]
                print(f"{name} trainable: {c['trainable']}/{c['total']}")

    trainable_params = [p for p in all_params if p.requires_grad]

    total_count = sum(p.numel() for p in all_params)
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"trainable parameters: {trainable_count}/{total_count}")

    return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
