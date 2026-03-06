"""Lightweight vector-quantization codec."""

from __future__ import annotations

import torch
from torch import nn

from llm_lab.interfaces.codec import Codec
from llm_lab.registry import register_codec


@register_codec("vq_lite")
class VQLiteCodec(nn.Module, Codec):
    """Minimal VQ codec with trainable codebook and optional EMA updates."""

    def __init__(
        self,
        codebook_size: int = 64,
        d_model: int = 16,
        use_ema_updates: bool = False,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        if codebook_size <= 1:
            raise ValueError("codebook_size must be > 1")
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")

        self.codebook_size = codebook_size
        self.d_model = d_model
        self.use_ema_updates = use_ema_updates
        self.ema_decay = ema_decay

        self.codebook = nn.Embedding(codebook_size, d_model)
        nn.init.uniform_(self.codebook.weight, -1.0, 1.0)

        self.register_buffer("ema_counts", torch.zeros(codebook_size))
        self.register_buffer("ema_sums", self.codebook.weight.detach().clone())

        if self.use_ema_updates:
            self.codebook.weight.requires_grad = False

    def encode(self, x_u8: torch.Tensor) -> torch.Tensor:
        if x_u8.dtype != torch.uint8:
            raise ValueError("VQLiteCodec.encode expects torch.uint8 input.")
        return x_u8.to(torch.int64)

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype != torch.int64:
            raise ValueError("VQLiteCodec.decode expects torch.int64 input.")
        return token_ids.clamp(0, 255).to(torch.uint8)

    def _maybe_ema_update(self, flat: torch.Tensor, idx_flat: torch.Tensor) -> None:
        if not self.use_ema_updates or not self.training:
            return

        one_hot = torch.nn.functional.one_hot(idx_flat, num_classes=self.codebook_size).to(flat.dtype)
        batch_counts = one_hot.sum(dim=0)
        batch_sums = one_hot.t() @ flat

        decay = self.ema_decay
        self.ema_counts.mul_(decay).add_(batch_counts * (1.0 - decay))
        self.ema_sums.mul_(decay).add_(batch_sums * (1.0 - decay))

        # Keep codebook numerically stable and avoid collapse to zeros.
        eps = 1e-5
        denom = self.ema_counts.unsqueeze(1).clamp_min(eps)
        new_codebook = self.ema_sums / denom

        # For entries with no support yet, keep current vectors.
        active = (self.ema_counts > eps).unsqueeze(1)
        kept = torch.where(active, new_codebook, self.codebook.weight.data)
        self.codebook.weight.data.copy_(kept)

    def quantize_hidden_with_codes(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize hidden vectors and return quantized outputs and code indices.

        Args:
            h: Hidden tensor of shape ``[B, T, D]``.

        Returns:
            q: Quantized hidden tensor ``[B, T, D]``.
            idx: Code indices ``[B, T]``.
        """
        if h.ndim != 3:
            raise ValueError("VQLiteCodec.quantize_hidden_with_codes expects rank-3 [B, T, D].")
        if not torch.is_floating_point(h):
            raise ValueError("VQLiteCodec.quantize_hidden_with_codes expects floating input.")
        if h.shape[-1] != self.d_model:
            raise ValueError(
                f"VQLiteCodec expected last dim d_model={self.d_model}, got {h.shape[-1]}."
            )

        bsz, t, d = h.shape
        flat = h.reshape(-1, d)
        codebook = self.codebook.weight

        # Squared L2 nearest-neighbor search.
        x2 = (flat ** 2).sum(dim=1, keepdim=True)
        c2 = (codebook ** 2).sum(dim=1).unsqueeze(0)
        xc = flat @ codebook.t()
        dist = x2 + c2 - (2.0 * xc)

        idx_flat = torch.argmin(dist, dim=1)
        self._maybe_ema_update(flat.detach(), idx_flat)

        q_flat = self.codebook(idx_flat)
        q = q_flat.view(bsz, t, d)
        idx = idx_flat.view(bsz, t)

        # Straight-through estimator.
        q_ste = h + (q - h).detach()
        return q_ste, idx

    def quantize_hidden(self, h: torch.Tensor) -> torch.Tensor:
        q, _ = self.quantize_hidden_with_codes(h)
        return q
