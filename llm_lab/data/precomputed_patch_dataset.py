"""Dataset for offline precomputed patcher1 hidden states."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


class PrecomputedPatchDataset(Dataset[Tensor]):
    """Load precomputed patcher1 outputs from disk.

    Expected file format (``torch.save``):
      - Tensor of shape ``[N, T, D]`` (float), or
      - list of ``[T, D]`` tensors.

    Each dataset item is a hidden sequence tensor with shape ``[T, D]``.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise ValueError(f"precomputed patch file not found: {self.path}")

        payload = torch.load(self.path, map_location="cpu")

        if isinstance(payload, torch.Tensor):
            if payload.ndim != 3:
                raise ValueError("precomputed tensor must have shape [N, T, D].")
            self._items = [payload[i] for i in range(payload.shape[0])]
        elif isinstance(payload, list):
            self._items = payload
        else:
            raise ValueError("precomputed patch file must contain a tensor or list of tensors.")

        if len(self._items) == 0:
            raise ValueError("precomputed patch dataset is empty.")

        for i, x in enumerate(self._items):
            if not isinstance(x, torch.Tensor):
                raise ValueError(f"item {i} is not a tensor.")
            if x.ndim != 2:
                raise ValueError(f"item {i} must have shape [T, D], got {tuple(x.shape)}")
            if not torch.is_floating_point(x):
                raise ValueError(f"item {i} must be floating point, got {x.dtype}")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Tensor:
        return self._items[index]
