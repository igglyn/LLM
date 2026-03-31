"""Rewrite patcher models with no in-patcher projection glue.

Design constraints for rewrite:
- `seq_len` is not a constructor argument for patcher modules.
- Patcher modules do not own token<->latent projection glue (linear adapters).
- Dimensional adaptation should live in the main model rewrite layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from blt_lite.model import TransformerBlock, _build_rope_cache


@dataclass(frozen=True)
class EmbeddedPatcherConfig:
    """Hardcoded rewrite config for patcher internals only."""

    d_model: int = 384
    encoder_layers: int = 2
    decoder_layers: int = 2
    n_heads: int = 6
    dropout: float = 0.1
    pos_encoding: str = "rope"
    grad_checkpointing: bool = False
    flash_attention: bool = True
    block_attention: bool = False
    block_size: int = 8


DEFAULT_PATCHER_CONFIG = EmbeddedPatcherConfig()


class RewritePatchEncoder(nn.Module):
    """Patch encoder without linear projection glue.

    Input/output channel size is fixed to `d_model`; patch aggregation is mean-pool.
    """

    def __init__(self, patch_size: int, cfg: EmbeddedPatcherConfig = DEFAULT_PATCHER_CONFIG):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        self.patch_size = patch_size
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.pos_encoding = cfg.pos_encoding
        self.grad_checkpointing = cfg.grad_checkpointing
        self.block_attention = cfg.block_attention
        self.block_size = cfg.block_size

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    dropout=cfg.dropout,
                    use_flash_attention=cfg.flash_attention,
                )
                for _ in range(max(1, cfg.encoder_layers))
            ]
        )

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        bsz, token_t, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(
                f"RewritePatchEncoder expects d_model={self.d_model}, got {d_model}. "
                "Move projection glue to the main model."
            )
        padded_t = math.ceil(token_t / self.patch_size) * self.patch_size
        if padded_t == token_t:
            return x
        pad = torch.zeros((bsz, padded_t - token_t, d_model), device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad(x)
        bsz, padded_t, d_model = x.shape
        patch_t = padded_t // self.patch_size
        grouped = x.view(bsz, patch_t, self.patch_size, d_model)
        out = grouped.mean(dim=2)

        rope_cos = rope_sin = None
        if self.pos_encoding == "rope":
            head_dim = self.d_model // self.n_heads
            rope_cos, rope_sin = _build_rope_cache(patch_t, head_dim)
            rope_cos = rope_cos.to(device=x.device, dtype=x.dtype)
            rope_sin = rope_sin.to(device=x.device, dtype=x.dtype)

        for block in self.blocks:
            out = block(
                out,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                block_attention=self.block_attention,
                block_size=self.block_size,
            )
        return out


class RewritePatchDecoder(nn.Module):
    """Patch decoder without query/out linear glue.

    Expands patch states back to token resolution via repeat-interleave.
    """

    def __init__(self, patch_size: int, cfg: EmbeddedPatcherConfig = DEFAULT_PATCHER_CONFIG):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        self.patch_size = patch_size
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.pos_encoding = cfg.pos_encoding
        self.grad_checkpointing = cfg.grad_checkpointing
        self.block_attention = cfg.block_attention
        self.block_size = cfg.block_size

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    dropout=cfg.dropout,
                    use_flash_attention=cfg.flash_attention,
                )
                for _ in range(max(1, cfg.decoder_layers))
            ]
        )

    def forward(self, token_hidden: torch.Tensor, patch_hidden: torch.Tensor) -> torch.Tensor:
        if token_hidden.size(-1) != self.d_model or patch_hidden.size(-1) != self.d_model:
            raise ValueError(
                f"RewritePatchDecoder expects d_model={self.d_model}. "
                "Move projection glue to the main model."
            )

        token_t = token_hidden.size(1)
        expanded = patch_hidden.repeat_interleave(self.patch_size, dim=1)
        expanded = expanded[:, :token_t, :]
        out = token_hidden + expanded

        rope_cos = rope_sin = None
        if self.pos_encoding == "rope":
            head_dim = self.d_model // self.n_heads
            rope_cos, rope_sin = _build_rope_cache(token_t, head_dim)
            rope_cos = rope_cos.to(device=token_hidden.device, dtype=token_hidden.dtype)
            rope_sin = rope_sin.to(device=token_hidden.device, dtype=token_hidden.dtype)

        for block in self.blocks:
            out = block(
                out,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                block_attention=self.block_attention,
                block_size=self.block_size,
            )
        return out


class RewritePatcherAutoencoder(nn.Module):
    """Rewrite patcher autoencoder with no in-module linear glue."""

    def __init__(self, patch_size: int, cfg: EmbeddedPatcherConfig = DEFAULT_PATCHER_CONFIG):
        super().__init__()
        self.encoder = RewritePatchEncoder(patch_size=patch_size, cfg=cfg)
        self.decoder = RewritePatchDecoder(patch_size=patch_size, cfg=cfg)

    def forward(self, token_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lat = self.encoder(token_hidden)
        recon = self.decoder(token_hidden, lat)
        return recon, lat
