"""Batch collation helpers."""

from __future__ import annotations

import torch
from torch import Tensor


def collate_batch(items: list[Tensor]) -> Tensor:
    """Stack sequence tensors into a batch tensor.

    Args:
        items: List of tensors each with shape ``[T]`` and dtype ``torch.uint8``.

    Returns:
        Tensor with shape ``[B, T]`` and dtype ``torch.uint8``.
    """
    if not items:
        raise ValueError("items must be non-empty")
    return torch.stack(items, dim=0)
