"""RetNet-style chunkwise retention mixer."""

from __future__ import annotations

import math

import torch
from torch import nn

from llm_lab.interfaces.mixer import Mixer
from llm_lab.registry import register_mixer


class _RetentionLayer(nn.Module):
    """Single retention layer with chunkwise state updates."""

    def __init__(self, d_model: int, decay: float = 0.95) -> None:
        super().__init__()
        if not (0.0 < decay < 1.0):
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, h: torch.Tensor, chunk_size: int) -> torch.Tensor:
        bsz, seq_len, d_model = h.shape
        q = self.to_q(h)
        k = self.to_k(h)
        v = self.to_v(h)

        n_chunks = math.ceil(seq_len / chunk_size)
        state = torch.zeros(bsz, d_model, d_model, dtype=h.dtype, device=h.device)
        out_chunks: list[torch.Tensor] = []

        for c in range(n_chunks):
            start = c * chunk_size
            end = min((c + 1) * chunk_size, seq_len)
            q_c = q[:, start:end, :]
            k_c = k[:, start:end, :]
            v_c = v[:, start:end, :]

            kv_c = torch.einsum("btd,bte->bde", k_c, v_c) / max(1, (end - start))
            state = (self.decay * state) + (1.0 - self.decay) * kv_c

            ret_c = torch.einsum("btd,bde->bte", q_c, state)
            gate_c = self.gate(h[:, start:end, :])
            out_chunks.append(ret_c * gate_c)

        out = torch.cat(out_chunks, dim=1)
        return self.to_out(out)


@register_mixer("retnet")
class RetNetMixer(nn.Module, Mixer):
    """Chunkwise retention mixer with stable residual + norm updates.

    Input/output shape: ``[B, T, D]``.
    """

    def __init__(
        self,
        d_model: int = 16,
        n_layers: int = 2,
        chunk_size: int = 8,
        decay: float = 0.95,
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_layers <= 0 or chunk_size <= 0:
            raise ValueError("d_model, n_layers, and chunk_size must be > 0")
        self.chunk_size = chunk_size
        self.layers = nn.ModuleList([_RetentionLayer(d_model=d_model, decay=decay) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError("RetNetMixer.forward expects [B, T, D]")

        out = h
        for layer, norm in zip(self.layers, self.norms):
            upd = layer(norm(out), chunk_size=self.chunk_size)
            out = out + upd
        return out
