"""Mixed-resolution context pyramid memory."""

from __future__ import annotations

import torch

from llm_lab.interfaces.memory import Memory
from llm_lab.registry import register_memory


@register_memory("context_pyramid")
class ContextPyramidMemory(Memory):
    """Keep recent high-resolution context and older low-resolution history."""

    def __init__(
        self,
        keep_recent_tokens: int = 32,
        keep_lowres_history: int = 64,
        lowres_source: str = "pooled_patcher1",
    ) -> None:
        if keep_recent_tokens <= 0:
            raise ValueError("keep_recent_tokens must be > 0")
        if keep_lowres_history < 0:
            raise ValueError("keep_lowres_history must be >= 0")
        if lowres_source not in {"patcher2", "pooled_patcher1"}:
            raise ValueError("lowres_source must be 'patcher2' or 'pooled_patcher1'")

        self.keep_recent_tokens = keep_recent_tokens
        self.keep_lowres_history = keep_lowres_history
        self.lowres_source = lowres_source
        self._lowres_history: torch.Tensor | None = None

    def read(self) -> torch.Tensor:
        if self._lowres_history is None:
            return torch.zeros(1, 0, 1)
        return self._lowres_history

    def write(self, h: torch.Tensor) -> None:
        if h.ndim != 3:
            raise ValueError("ContextPyramidMemory.write expects [B, T, D]")

        if self.keep_lowres_history == 0:
            self._lowres_history = h[:, :0, :].detach()
            return

        if self._lowres_history is None or self._lowres_history.shape[0] != h.shape[0]:
            self._lowres_history = h.detach()
        else:
            self._lowres_history = torch.cat([self._lowres_history, h.detach()], dim=1)

        if self._lowres_history.shape[1] > self.keep_lowres_history:
            self._lowres_history = self._lowres_history[:, -self.keep_lowres_history :, :]

    @staticmethod
    def _pool_patcher1(h1: torch.Tensor) -> torch.Tensor:
        bsz, t, d = h1.shape
        tp = t // 2
        if tp == 0:
            return h1[:, :0, :]
        trimmed = h1[:, : tp * 2, :]
        return trimmed.view(bsz, tp, 2, d).mean(dim=2)

    def combine(
        self,
        high_res: torch.Tensor,
        low_res_candidate: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int]:
        """Return mixed-resolution mixer input and recent high-res output length."""
        if high_res.ndim != 3:
            raise ValueError("high_res must have shape [B, T, D]")

        recent_len = min(self.keep_recent_tokens, high_res.shape[1])
        recent = high_res[:, -recent_len:, :]

        if self.lowres_source == "patcher2" and low_res_candidate is not None:
            current_lowres = low_res_candidate
        else:
            current_lowres = self._pool_patcher1(high_res)

        history = self.read()
        if history.shape[-1] == 1 and history.shape[1] == 0:
            history = current_lowres[:, :0, :]
        else:
            history = history.to(device=high_res.device, dtype=high_res.dtype)

        if self.keep_lowres_history == 0:
            mixed = recent
        else:
            if history.shape[1] > 0:
                lowres = torch.cat([history, current_lowres], dim=1)
            else:
                lowres = current_lowres
            if lowres.shape[1] > self.keep_lowres_history:
                lowres = lowres[:, -self.keep_lowres_history :, :]
            mixed = torch.cat([lowres, recent], dim=1)

        self.write(current_lowres)
        return mixed, recent_len
