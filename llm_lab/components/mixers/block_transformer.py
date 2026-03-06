"""Block-local transformer mixer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from llm_lab.components.norms.rmsnorm import RMSNorm
from llm_lab.interfaces.mixer import Mixer
from llm_lab.registry import register_mixer


class _BlockTransformerLayer(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_in = self.norm1(x)
        a_out, _ = self.attn(a_in, a_in, a_in, need_weights=False)
        x = x + a_out
        x = x + self.ff(self.norm2(x))
        return x


@register_mixer("block_transformer")
class BlockTransformerMixer(nn.Module, Mixer):
    """Transformer mixer with attention restricted to local blocks.

    Input/output shape: ``[B, T, D]``.
    Attention is computed independently inside each non-overlapping block of
    size ``block_size``.
    """

    def __init__(
        self,
        d_model: int = 16,
        n_heads: int = 4,
        n_layers: int = 2,
        block_size: int = 8,
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or n_layers <= 0 or block_size <= 0:
            raise ValueError("d_model, n_heads, n_layers, and block_size must be > 0")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.block_size = block_size
        self.layers = nn.ModuleList(
            [_BlockTransformerLayer(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)]
        )

    def _reshape_blocks(self, h: torch.Tensor) -> tuple[torch.Tensor, int]:
        bsz, t, d_model = h.shape
        rem = t % self.block_size
        pad = (self.block_size - rem) % self.block_size
        if pad:
            h = F.pad(h, (0, 0, 0, pad), value=0.0)

        t_pad = h.shape[1]
        n_blocks = t_pad // self.block_size
        h_blocks = h.view(bsz, n_blocks, self.block_size, d_model)
        h_blocks = h_blocks.reshape(bsz * n_blocks, self.block_size, d_model)
        return h_blocks, pad

    def _restore_shape(self, h_blocks: torch.Tensor, bsz: int, t_orig: int) -> torch.Tensor:
        d_model = h_blocks.shape[-1]
        n_blocks = h_blocks.shape[0] // bsz
        h = h_blocks.view(bsz, n_blocks, self.block_size, d_model)
        h = h.reshape(bsz, n_blocks * self.block_size, d_model)
        return h[:, :t_orig, :]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError("BlockTransformerMixer.forward expects [B, T, D]")

        bsz, t_orig, _ = h.shape
        blocks, _ = self._reshape_blocks(h)

        out = blocks
        for layer in self.layers:
            out = layer(out)

        return self._restore_shape(out, bsz=bsz, t_orig=t_orig)
