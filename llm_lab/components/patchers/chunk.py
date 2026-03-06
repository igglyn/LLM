"""Chunk patcher implementation."""

from __future__ import annotations

import torch
from torch import nn

from llm_lab.interfaces.patcher import Patcher
from llm_lab.registry import register_patcher


@register_patcher("chunk")
class ChunkPatcher(nn.Module, Patcher):
    """Groups byte ids into fixed-size patches and projects to embeddings.

    Input:
        ``x_u8`` with shape ``[B, T]`` and dtype ``torch.uint8``.

    Process:
        1. Convert bytes to token ids (``torch.int64``).
        2. Pad length to a multiple of ``patch_size``.
        3. Reshape into ``[B, Tp, patch_size]``.
        4. Convert to float and project to ``[B, Tp, D]``.

    Output:
        Tensor with shape ``[B, Tp, D]``.
    """

    def __init__(self, patch_size: int = 4, d_model: int = 16) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        self.patch_size = patch_size
        self.d_model = d_model
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, x_u8: torch.Tensor) -> torch.Tensor:
        if x_u8.ndim != 2:
            raise ValueError("ChunkPatcher.forward expects rank-2 tensor [B, T].")
        if x_u8.dtype != torch.uint8:
            raise ValueError("ChunkPatcher.forward expects torch.uint8 input.")

        token_ids = x_u8.to(torch.int64)
        bsz, t = token_ids.shape
        rem = t % self.patch_size
        if rem:
            pad = self.patch_size - rem
            token_ids = torch.nn.functional.pad(token_ids, (0, pad), value=0)

        tp = token_ids.shape[1] // self.patch_size
        patch_ids = token_ids.view(bsz, tp, self.patch_size)

        patch_values = patch_ids.to(torch.float32) / 255.0
        return self.proj(patch_values)
