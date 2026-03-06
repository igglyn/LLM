"""Second-stage hidden-state chunk patcher."""

from __future__ import annotations

import torch
from torch import nn

from llm_lab.interfaces.patcher import HiddenPatcher
from llm_lab.registry import register_patcher


@register_patcher("hidden_chunk")
class HiddenChunkPatcher(nn.Module, HiddenPatcher):
    """Group hidden states into fixed chunks and pool to one vector per chunk.

    Input:
        Hidden states ``h`` with shape ``[B, T, D]``.

    Output:
        Hidden states with shape ``[B, Tp, D]`` where ``Tp = T // patch_size``.

    Notes:
        Incomplete trailing tokens are dropped.
    """

    def __init__(self, patch_size: int = 2, d_model: int = 16, use_mlp: bool = False) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        self.patch_size = patch_size
        self.d_model = d_model
        self.use_mlp = use_mlp
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.mlp = nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError("HiddenChunkPatcher.forward expects rank-3 tensor [B, T, D].")
        if not torch.is_floating_point(h):
            raise ValueError("HiddenChunkPatcher.forward expects floating-point input.")

        bsz, t, d = h.shape
        if d != self.d_model:
            raise ValueError(
                f"HiddenChunkPatcher.forward expected last dim d_model={self.d_model}, got {d}."
            )

        tp = t // self.patch_size
        if tp == 0:
            pooled = h[:, :0, :]
        else:
            trimmed = h[:, : tp * self.patch_size, :]
            pooled = trimmed.view(bsz, tp, self.patch_size, d).mean(dim=2)
        return self.mlp(pooled)
