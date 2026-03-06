"""Baseline transformer mixer."""

from __future__ import annotations

import torch
from torch import nn

from llm_lab.components.norms.rmsnorm import RMSNorm
from llm_lab.interfaces.mixer import Mixer
from llm_lab.registry import register_mixer


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm2 = RMSNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(h)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        h = h + attn_out
        h = h + self.ff(self.norm2(h))
        return h


@register_mixer("transformer")
class TransformerMixer(nn.Module, Mixer):
    """Stack of transformer blocks preserving shape ``[B, T, D]``."""

    def __init__(self, d_model: int = 16, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or n_layers <= 0:
            raise ValueError("d_model, n_heads, and n_layers must be > 0")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.layers = nn.ModuleList(
            [_TransformerBlock(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = h
        for layer in self.layers:
            out = layer(out)
        return out
