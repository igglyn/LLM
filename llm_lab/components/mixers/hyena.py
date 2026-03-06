"""Hyena-style long convolution mixer."""

from __future__ import annotations

import torch
from torch import nn

from llm_lab.interfaces.mixer import Mixer
from llm_lab.registry import register_mixer


@register_mixer("hyena")
class HyenaMixer(nn.Module, Mixer):
    """Minimal Hyena-style mixer using depthwise long convolution + gating.

    Input/output shape: ``[B, T, D]``.
    """

    def __init__(self, d_model: int = 16, kernel_size: int = 31) -> None:
        super().__init__()
        if d_model <= 0 or kernel_size <= 0:
            raise ValueError("d_model and kernel_size must be > 0")
        self.d_model = d_model
        self.kernel_size = kernel_size

        # Depthwise long convolution over sequence length.
        self.dw_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model,
            bias=False,
        )
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError("HyenaMixer.forward expects [B, T, D]")

        x = self.norm(h)
        uv = self.in_proj(x)
        u, v = torch.chunk(uv, chunks=2, dim=-1)

        # [B, T, D] -> [B, D, T]
        v_t = v.transpose(1, 2)
        y = self.dw_conv(v_t)
        # trim causal padding tail to preserve T
        y = y[:, :, : h.size(1)]
        y = y.transpose(1, 2)

        y = torch.sigmoid(u) * y
        y = self.out_proj(y)
        return h + y
