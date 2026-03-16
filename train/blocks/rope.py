from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from train.specs import RuntimeState


class RoPEModule(nn.Module):
    def __init__(self, d_model: int, n_heads: int, base: float = 10000.0, scale: float = 1.0, max_seq_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = scale

        # precompute inverse frequencies per dimension pair
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq)

        # precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float() * self.scale
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cache', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cache', emb.sin()[None, None, :, :])

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        B, T, C = x.shape

        if T > self.cos_cache.shape[2]:
            self._build_cache(T)

        x = x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        cos = self.cos_cache[:, :, :T, :]
        sin = self.sin_cache[:, :, :T, :]
        x = (x * cos) + (self._rotate_half(x) * sin)
        return x.transpose(1, 2).contiguous().view(B, T, C)


@dataclass(frozen=True)
class RoPEBlock:
    d_model: int
    n_heads: int
    base: float
    scale: float

    @property
    def block_name(self) -> str:
        return "RoPE"

    def build(self) -> RoPEModule:
        return RoPEModule(
            d_model=self.d_model,
            n_heads=self.n_heads,
            base=self.base,
            scale=self.scale,
        )

    def run(self, state: RuntimeState) -> RuntimeState:
        tensor = state.tensor
        return RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, f"RoPE(d_model={self.d_model},n_heads={self.n_heads},base={self.base},scale={self.scale})"],
            moe_metrics=dict(state.moe_metrics),
            tensor_shape=tuple(tensor.shape) if tensor is not None else state.tensor_shape,
            tensor=tensor,
        )
