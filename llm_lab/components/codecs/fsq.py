"""Finite Scalar Quantization (FSQ) codec."""

from __future__ import annotations

import torch

from llm_lab.interfaces.codec import Codec
from llm_lab.registry import register_codec


@register_codec("fsq")
class FSQCodec(Codec):
    """FSQ codec with deterministic per-dimension scalar quantization.

    This codec preserves byte/token compatibility via ``encode``/``decode`` and
    additionally exposes hidden-state quantization through ``quantize_hidden``.
    """

    def __init__(self, levels_per_dim: int = 8, d_model: int = 16) -> None:
        if levels_per_dim < 2:
            raise ValueError("levels_per_dim must be >= 2")
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        self.levels_per_dim = levels_per_dim
        self.d_model = d_model

    def encode(self, x_u8: torch.Tensor) -> torch.Tensor:
        if x_u8.dtype != torch.uint8:
            raise ValueError("FSQCodec.encode expects torch.uint8 input.")
        return x_u8.to(torch.int64)

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype != torch.int64:
            raise ValueError("FSQCodec.decode expects torch.int64 input.")
        return token_ids.clamp(0, 255).to(torch.uint8)

    def _levels(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Evenly spaced scalar levels in fixed range [-1, 1].
        return torch.linspace(-1.0, 1.0, self.levels_per_dim, device=device, dtype=dtype)

    def quantize_hidden(self, h: torch.Tensor) -> torch.Tensor:
        """Quantize hidden states with STE.

        Input/Output: ``[B, T, D]`` floating tensor.
        """
        if h.ndim != 3:
            raise ValueError("FSQCodec.quantize_hidden expects rank-3 tensor [B, T, D].")
        if not torch.is_floating_point(h):
            raise ValueError("FSQCodec.quantize_hidden expects floating-point input.")
        if h.shape[-1] != self.d_model:
            raise ValueError(
                f"FSQCodec.quantize_hidden expected last dim d_model={self.d_model}, got {h.shape[-1]}."
            )

        x = torch.clamp(h, -1.0, 1.0)
        levels = self._levels(device=x.device, dtype=x.dtype)

        # Nearest-level quantization.
        dist = torch.abs(x.unsqueeze(-1) - levels)
        idx = torch.argmin(dist, dim=-1)
        q = levels[idx]

        # Straight-through estimator for gradients.
        return x + (q - x).detach()
