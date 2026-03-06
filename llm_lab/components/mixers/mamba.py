"""Mamba-style mixer with optional real backend and explicit fallback."""

from __future__ import annotations

import importlib

import torch
from torch import nn

from llm_lab.interfaces.mixer import Mixer
from llm_lab.registry import register_mixer


class _FallbackMambaLayer(nn.Module):
    """Lightweight gated recurrent fallback (not true Mamba)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.h_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        h_state = torch.zeros(bsz, d_model, dtype=x.dtype, device=x.device)
        outputs: list[torch.Tensor] = []

        x_n = self.norm(x)
        for t in range(seq_len):
            uv = self.in_proj(x_n[:, t, :])
            u, v = torch.chunk(uv, chunks=2, dim=-1)
            gate = torch.sigmoid(u)
            cand = torch.tanh(v + self.h_proj(h_state))
            h_state = gate * cand + (1.0 - gate) * h_state
            outputs.append(self.out_proj(h_state).unsqueeze(1))

        return x + torch.cat(outputs, dim=1)


@register_mixer("mamba")
class MambaMixer(nn.Module, Mixer):
    """Mamba mixer wrapper with backend selection.

    backend:
      - ``auto``: use real backend if available, else fallback
      - ``real``: require real backend and fail clearly if unavailable
      - ``fallback``: always use lightweight fallback path
    """

    def __init__(
        self,
        d_model: int = 16,
        n_layers: int = 2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_layers <= 0:
            raise ValueError("d_model and n_layers must be > 0")
        if backend not in {"auto", "real", "fallback"}:
            raise ValueError("backend must be one of: auto, real, fallback")

        self.d_model = d_model
        self.n_layers = n_layers
        self.backend = backend

        real_cls = self._load_real_mamba_cls()
        if backend == "real":
            if real_cls is None:
                raise NotImplementedError(
                    "Mamba real backend requested but dependency is unavailable. "
                    "Install `mamba-ssm` to enable backend='real'."
                )
            self.backend_used = "real"
            self.layers = nn.ModuleList([real_cls(d_model=d_model) for _ in range(n_layers)])
        elif backend == "fallback":
            self.backend_used = "fallback"
            self.layers = nn.ModuleList([_FallbackMambaLayer(d_model=d_model) for _ in range(n_layers)])
        else:  # auto
            if real_cls is not None:
                self.backend_used = "real"
                self.layers = nn.ModuleList([real_cls(d_model=d_model) for _ in range(n_layers)])
            else:
                self.backend_used = "fallback"
                self.layers = nn.ModuleList(
                    [_FallbackMambaLayer(d_model=d_model) for _ in range(n_layers)]
                )

    @staticmethod
    def _load_real_mamba_cls():
        """Try to load a real Mamba implementation class from mamba-ssm."""
        candidates = [
            ("mamba_ssm.modules.mamba_simple", "Mamba"),
            ("mamba_ssm", "Mamba"),
        ]
        for module_name, class_name in candidates:
            try:
                mod = importlib.import_module(module_name)
            except Exception:
                continue
            cls = getattr(mod, class_name, None)
            if cls is not None:
                return cls
        return None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError("MambaMixer.forward expects [B, T, D]")

        out = h
        for layer in self.layers:
            out = layer(out)
        return out
