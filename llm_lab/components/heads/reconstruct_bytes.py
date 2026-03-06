"""Auxiliary reconstruction head for patcher1 hidden states."""

from __future__ import annotations

import torch
from torch import nn

from llm_lab.registry import register_head


@register_head("reconstruct_bytes")
class ReconstructBytesHead(nn.Module):
    """Project patcher1 hidden states to per-byte logits inside each patch.

    Input:
        h1: ``[B, T1, D]``

    Output:
        byte logits: ``[B, T1, patch_size, 256]``
    """

    def __init__(self, d_model: int = 16, patch_size: int = 4) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        self.d_model = d_model
        self.patch_size = patch_size
        self.proj = nn.Linear(d_model, patch_size * 256)

    def forward(self, h1: torch.Tensor) -> torch.Tensor:
        if h1.ndim != 3:
            raise ValueError("ReconstructBytesHead.forward expects rank-3 [B, T1, D].")
        bsz, t1, _ = h1.shape
        out = self.proj(h1)
        return out.view(bsz, t1, self.patch_size, 256)
