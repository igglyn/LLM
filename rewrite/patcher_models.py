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


@dataclass(frozen=True)
class SequenceBatchShape:
    """Shape metadata for adapting token and patch tensors."""

    batch: int
    seq: int
    token_t: int
    padded_t: int


class SequenceBatchAdapter:
    """Adapter for explicit sequence axis handling during patch reshaping.

    Supports token_hidden inputs of:
    - [B, T, D] (implicit sequence axis of size 1)
    - [B, S, T, D] (explicit sequence axis)
    """

    def __init__(self, patch_size: int, d_model: int):
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        self.patch_size = patch_size
        self.d_model = d_model

    def _normalize_tokens(self, token_hidden: torch.Tensor) -> tuple[torch.Tensor, SequenceBatchShape, bool]:
        if token_hidden.ndim == 3:
            bsz, token_t, d_model = token_hidden.shape
            seq = 1
            has_explicit_seq_dim = False
            token_hidden = token_hidden.unsqueeze(1)
        elif token_hidden.ndim == 4:
            bsz, seq, token_t, d_model = token_hidden.shape
            has_explicit_seq_dim = True
        else:
            raise ValueError(f"Expected token_hidden with rank 3 or 4, got rank {token_hidden.ndim}")

        if d_model != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d_model}. Move projection glue to main model.")

        padded_t = math.ceil(token_t / self.patch_size) * self.patch_size
        if padded_t != token_t:
            pad = torch.zeros(
                (bsz, seq, padded_t - token_t, d_model),
                dtype=token_hidden.dtype,
                device=token_hidden.device,
            )
            token_hidden = torch.cat([token_hidden, pad], dim=2)

        shape = SequenceBatchShape(batch=bsz, seq=seq, token_t=token_t, padded_t=padded_t)
        return token_hidden, shape, has_explicit_seq_dim

    def tokens_to_patches(self, token_hidden: torch.Tensor) -> tuple[torch.Tensor, SequenceBatchShape, bool]:
        normalized, shape, has_explicit_seq_dim = self._normalize_tokens(token_hidden)
        grouped = normalized.view(shape.batch, shape.seq, shape.padded_t // self.patch_size, self.patch_size, self.d_model)
        return grouped.mean(dim=3), shape, has_explicit_seq_dim

    def flatten_patch_batch(self, patch_hidden: torch.Tensor) -> torch.Tensor:
        bsz, seq, patch_t, d_model = patch_hidden.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected patch d_model={self.d_model}, got {d_model}")
        return patch_hidden.reshape(bsz * seq, patch_t, d_model)

    def unflatten_patch_batch(self, flat_patch_hidden: torch.Tensor, shape: SequenceBatchShape) -> torch.Tensor:
        _, patch_t, d_model = flat_patch_hidden.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected patch d_model={self.d_model}, got {d_model}")
        return flat_patch_hidden.view(shape.batch, shape.seq, patch_t, d_model)

    def patches_to_tokens(self, patch_hidden: torch.Tensor, token_shape: SequenceBatchShape, *, crop_to_token_t: bool = True) -> torch.Tensor:
        expanded = patch_hidden.repeat_interleave(self.patch_size, dim=2)
        max_t = token_shape.token_t if crop_to_token_t else token_shape.padded_t
        return expanded[:, :, :max_t, :]


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
        self.adapter = SequenceBatchAdapter(patch_size=patch_size, d_model=cfg.d_model)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, shape, has_explicit_seq_dim = self.adapter.tokens_to_patches(x)
        out = self.adapter.flatten_patch_batch(out)
        patch_t = out.size(1)

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
        out = self.adapter.unflatten_patch_batch(out, shape)
        return out if has_explicit_seq_dim else out.squeeze(1)


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
        self.adapter = SequenceBatchAdapter(patch_size=patch_size, d_model=cfg.d_model)

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
        norm_tokens, token_shape, has_explicit_seq_dim = self.adapter._normalize_tokens(token_hidden)
        if patch_hidden.ndim == 3:
            patch_hidden = patch_hidden.unsqueeze(1)
        if patch_hidden.ndim != 4:
            raise ValueError(f"Expected patch_hidden with rank 3 or 4, got rank {patch_hidden.ndim}")
        if patch_hidden.shape[:2] != norm_tokens.shape[:2]:
            raise ValueError(
                f"token_hidden and patch_hidden batch/seq dims must match, got "
                f"{tuple(norm_tokens.shape[:2])} vs {tuple(patch_hidden.shape[:2])}"
            )
        out = norm_tokens + self.adapter.patches_to_tokens(patch_hidden, token_shape, crop_to_token_t=False)
        out = self.adapter.flatten_patch_batch(out)
        token_t = token_shape.padded_t

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
        out = out.view(token_shape.batch, token_shape.seq, token_t, self.d_model)
        out = out[:, :, : token_shape.token_t, :]
        return out if has_explicit_seq_dim else out.squeeze(1)


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


class SlotConvEncoder(nn.Module):
    """Encode packed slot latents into model width.

    Input:  x -> [B, S, G, D]
    Output: y -> [B, S, d_model]
    """

    def __init__(
        self,
        groups: int,
        d_chunk: int,
        d_model: int = 32,
        kernel_size: int = 3,
        hidden_mult: int = 1,
        dropout: float = 0.0,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        if groups <= 0 or d_chunk <= 0 or d_model <= 0:
            raise ValueError("groups, d_chunk, and d_model must be > 0")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be > 0")
        if hidden_mult <= 0:
            raise ValueError("hidden_mult must be > 0")
        self.groups = groups
        self.d_chunk = d_chunk
        self.d_model = d_model
        self.use_residual = use_residual

        padding = kernel_size // 2
        hidden_dim = d_chunk * hidden_mult

        self.slot_gate = nn.Parameter(torch.zeros(1, 1, groups, d_chunk))
        self.slot_bias = nn.Parameter(torch.zeros(1, 1, groups, d_chunk))

        self.pre_norm = nn.LayerNorm(d_chunk)
        self.conv1 = nn.Conv1d(d_chunk, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_dim, d_chunk, kernel_size=kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout)

        packed_dim = groups * d_chunk
        self.pack_norm = nn.LayerNorm(packed_dim)
        self.to_model = nn.Linear(packed_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected rank-4 input [B, S, G, D], got rank {x.ndim}")
        bsz, seq, groups, d_chunk = x.shape
        if groups != self.groups or d_chunk != self.d_chunk:
            raise ValueError(
                f"Expected input shape (B, S, {self.groups}, {self.d_chunk}), got {tuple(x.shape)}"
            )

        gate = 2.0 * torch.sigmoid(self.slot_gate)
        x = x * gate + self.slot_bias

        residual = x
        x = self.pre_norm(x)
        x = x.reshape(bsz * seq, groups, d_chunk).transpose(1, 2)

        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)

        x = x.transpose(1, 2).reshape(bsz, seq, groups, d_chunk)
        if self.use_residual:
            x = x + residual

        x = x.reshape(bsz, seq, groups * d_chunk)
        x = self.pack_norm(x)
        return self.to_model(x)


class SlotConvDecoder(nn.Module):
    """Decode model width back into packed slot latents.

    Input:  x -> [B, S, d_model]
    Output: y -> [B, S, G, D]
    """

    def __init__(
        self,
        groups: int,
        d_chunk: int,
        d_model: int = 32,
        kernel_size: int = 3,
        hidden_mult: int = 1,
        dropout: float = 0.0,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        if groups <= 0 or d_chunk <= 0 or d_model <= 0:
            raise ValueError("groups, d_chunk, and d_model must be > 0")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be > 0")
        if hidden_mult <= 0:
            raise ValueError("hidden_mult must be > 0")
        self.groups = groups
        self.d_chunk = d_chunk
        self.d_model = d_model
        self.use_residual = use_residual

        padding = kernel_size // 2
        hidden_dim = d_chunk * hidden_mult
        packed_dim = groups * d_chunk

        self.from_model = nn.Linear(d_model, packed_dim)
        self.slot_gate = nn.Parameter(torch.zeros(1, 1, groups, d_chunk))
        self.slot_bias = nn.Parameter(torch.zeros(1, 1, groups, d_chunk))

        self.pre_norm = nn.LayerNorm(d_chunk)
        self.conv1 = nn.Conv1d(d_chunk, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_dim, d_chunk, kernel_size=kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected rank-3 input [B, S, d_model], got rank {x.ndim}")
        bsz, seq, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected input last dim {self.d_model}, got {d_model}")

        x = self.from_model(x)
        x = x.reshape(bsz, seq, self.groups, self.d_chunk)

        gate = 2.0 * torch.sigmoid(self.slot_gate)
        x = x * gate + self.slot_bias

        residual = x
        x = self.pre_norm(x)
        x = x.reshape(bsz * seq, self.groups, self.d_chunk).transpose(1, 2)

        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)

        x = x.transpose(1, 2).reshape(bsz, seq, self.groups, self.d_chunk)
        if self.use_residual:
            x = x + residual
        return x


class SlotConvAutoencoder(nn.Module):
    """Autoencoder wrapper for slot-conv rewrite patcher type."""

    def __init__(
        self,
        groups: int,
        d_chunk: int,
        d_model: int,
        kernel_size: int = 3,
        hidden_mult: int = 1,
        dropout: float = 0.0,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = SlotConvEncoder(
            groups=groups,
            d_chunk=d_chunk,
            d_model=d_model,
            kernel_size=kernel_size,
            hidden_mult=hidden_mult,
            dropout=dropout,
            use_residual=use_residual,
        )
        self.decoder = SlotConvDecoder(
            groups=groups,
            d_chunk=d_chunk,
            d_model=d_model,
            kernel_size=kernel_size,
            hidden_mult=hidden_mult,
            dropout=dropout,
            use_residual=use_residual,
        )

    def forward(self, packed_slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lat = self.encoder(packed_slots)
        recon = self.decoder(lat)
        return recon, lat
