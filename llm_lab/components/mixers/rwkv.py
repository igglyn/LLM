"""RWKV-style recurrent mixer."""

from __future__ import annotations

import torch
from torch import nn

from llm_lab.interfaces.mixer import Mixer
from llm_lab.registry import register_mixer


@register_mixer("rwkv")
class RWKVMixer(nn.Module, Mixer):
    """Minimal RWKV-style time-mixing recurrent block stack.

    Processes sequence step-by-step with exponential moving state updates while
    preserving mixer interface shape ``[B, T, D]``.
    """

    def __init__(self, d_model: int = 16, n_layers: int = 2, decay: float = 0.95) -> None:
        super().__init__()
        if d_model <= 0 or n_layers <= 0:
            raise ValueError("d_model and n_layers must be > 0")
        if not (0.0 < decay < 1.0):
            raise ValueError("decay must be in (0, 1)")

        self.d_model = d_model
        self.n_layers = n_layers
        self.decay = decay

        self.in_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.key = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(n_layers)])
        self.value = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(n_layers)])
        self.receptance = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(n_layers)]
        )
        self.out = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(n_layers)])

    def _time_mix_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        state = torch.zeros(bsz, d_model, device=x.device, dtype=x.dtype)
        outs: list[torch.Tensor] = []

        normed = self.in_norms[layer_idx](x)
        k = self.key[layer_idx](normed)
        v = self.value[layer_idx](normed)
        r = torch.sigmoid(self.receptance[layer_idx](normed))

        for t in range(seq_len):
            kv_t = k[:, t, :] * v[:, t, :]
            state = (self.decay * state) + ((1.0 - self.decay) * kv_t)
            y_t = r[:, t, :] * state
            outs.append(y_t.unsqueeze(1))

        y = torch.cat(outs, dim=1)
        y = self.out[layer_idx](y)
        return x + y

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError("RWKVMixer.forward expects [B, T, D]")

        out = h
        for i in range(self.n_layers):
            out = self._time_mix_layer(out, i)
        return out
