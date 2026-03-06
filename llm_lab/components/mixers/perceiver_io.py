"""Perceiver-style latent bottleneck mixer."""

from __future__ import annotations

import torch
from torch import nn

from llm_lab.interfaces.mixer import Mixer
from llm_lab.registry import register_mixer


class _LatentBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        a_in = self.norm1(z)
        a_out, _ = self.self_attn(a_in, a_in, a_in, need_weights=False)
        z = z + a_out
        z = z + self.ff(self.norm2(z))
        return z


@register_mixer("perceiver_io")
class PerceiverIOMixer(nn.Module, Mixer):
    """Compress token sequence into latents, process, and project back.

    Input/output: ``[B, T, D]``.
    """

    def __init__(
        self,
        d_model: int = 16,
        num_latents: int = 8,
        n_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_latents <= 0 or n_layers <= 0 or n_heads <= 0:
            raise ValueError("d_model, num_latents, n_layers, n_heads must be > 0")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.latents = nn.Parameter(torch.randn(num_latents, d_model) * 0.02)
        self.cross_in = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.blocks = nn.ModuleList([_LatentBlock(d_model, n_heads) for _ in range(n_layers)])
        self.cross_out = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError("PerceiverIOMixer.forward expects [B, T, D]")

        bsz, _, _ = h.shape
        z = self.latents.unsqueeze(0).expand(bsz, -1, -1)

        z, _ = self.cross_in(z, h, h, need_weights=False)
        for block in self.blocks:
            z = block(z)

        out, _ = self.cross_out(h, z, z, need_weights=False)
        return h + out
