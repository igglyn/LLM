from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((q_len, k_len), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


class PatchEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int, n_heads: int, n_layers: int, dropout: float, patch_size: int, seq_len: int):
        super().__init__()
        self.patch_size = patch_size
        self.max_patch_len = math.ceil(seq_len / patch_size)
        self.patch_proj = nn.Linear(in_dim * patch_size, d_model)
        self.patch_pos_emb = nn.Embedding(self.max_patch_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=max(1, n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, token_t, d_model = x.shape
        padded_t = math.ceil(token_t / self.patch_size) * self.patch_size
        if padded_t != token_t:
            pad = torch.zeros((bsz, padded_t - token_t, d_model), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        patch_t = padded_t // self.patch_size
        grouped = x.view(bsz, patch_t, self.patch_size, d_model)
        patch_hidden = self.patch_proj(grouped.reshape(bsz, patch_t, self.patch_size * d_model))

        pos = torch.arange(0, patch_t, device=x.device).unsqueeze(0)
        patch_hidden = patch_hidden + self.patch_pos_emb(pos)

        mask = _causal_mask(patch_t, patch_t, x.device)
        return self.blocks(patch_hidden, mask=mask)


class PatchDecoder(nn.Module):
    def __init__(self, token_dim: int, d_model: int, out_dim: int, n_heads: int, n_layers: int, dropout: float, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.query_proj = nn.Linear(token_dim, d_model)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=max(1, n_layers))
        self.out_proj = nn.Linear(d_model, out_dim)

    def _token_to_patch_mask(self, token_len: int, patch_len: int, device: torch.device) -> torch.Tensor:
        token_positions = torch.arange(token_len, device=device)
        patch_positions = torch.arange(patch_len, device=device)
        allowed = patch_positions.unsqueeze(0) <= torch.div(token_positions.unsqueeze(1), self.patch_size, rounding_mode="floor")
        mask = torch.full((token_len, patch_len), float("-inf"), device=device)
        return mask.masked_fill(allowed, 0.0)

    def forward(self, token_hidden: torch.Tensor, patch_hidden: torch.Tensor) -> torch.Tensor:
        token_t = token_hidden.size(1)
        patch_t = patch_hidden.size(1)
        token_patch_mask = self._token_to_patch_mask(token_t, patch_t, token_hidden.device)
        query_hidden = self.query_proj(token_hidden)
        cross_hidden, _ = self.cross_attn(
            query=query_hidden,
            key=patch_hidden,
            value=patch_hidden,
            attn_mask=token_patch_mask,
            need_weights=False,
        )
        dec_hidden = self.blocks(query_hidden + cross_hidden, mask=_causal_mask(token_t, token_t, token_hidden.device))
        return self.out_proj(dec_hidden)


class PatcherAutoencoder(nn.Module):
    """Standalone patcher/unpatcher that compresses token states and reconstructs them."""

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        out_dim: int,
        patch_size: int,
        seq_len: int,
        encoder_layers: int,
        decoder_layers: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = PatchEncoder(
            in_dim=in_dim,
            d_model=latent_dim,
            n_heads=n_heads,
            n_layers=encoder_layers,
            dropout=dropout,
            patch_size=patch_size,
            seq_len=seq_len,
        )
        self.decoder = PatchDecoder(
            token_dim=in_dim,
            d_model=latent_dim,
            out_dim=out_dim,
            n_heads=n_heads,
            n_layers=decoder_layers,
            dropout=dropout,
            patch_size=patch_size,
        )

    def forward(self, token_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patch_hidden = self.encoder(token_hidden)
        recon_hidden = self.decoder(token_hidden, patch_hidden)
        return recon_hidden, patch_hidden


class TinyPatchLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        patch_size: int = 1,
        d_model: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        dropout: float = 0.1,
        patcher_latent_dim: int = 384,
        patcher_encoder_layers: int = 2,
        patcher_decoder_layers: int = 2,
        patcher_heads: int | None = None,
        patcher_dropout: float | None = None,
    ):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        self.seq_len = seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.token_pos_emb = nn.Embedding(seq_len, d_model)

        self.patcher = PatcherAutoencoder(
            in_dim=d_model,
            latent_dim=patcher_latent_dim,
            out_dim=d_model,
            patch_size=patch_size,
            seq_len=seq_len,
            encoder_layers=patcher_encoder_layers,
            decoder_layers=patcher_decoder_layers,
            n_heads=int(patcher_heads or n_heads),
            dropout=float(dropout if patcher_dropout is None else patcher_dropout),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def load_patcher_checkpoint(self, path: str | Path, map_location: torch.device | str | None = None) -> None:
        ckpt = torch.load(Path(path), map_location=map_location)
        state = ckpt["patcher"] if isinstance(ckpt, dict) and "patcher" in ckpt else ckpt
        self.patcher.load_state_dict(state)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        _, token_t = x.shape
        if token_t > self.seq_len:
            raise ValueError(f"sequence length {token_t} exceeds model limit {self.seq_len}")

        token_hidden = self.token_emb(x)
        pos = torch.arange(0, token_t, device=x.device).unsqueeze(0)
        token_hidden = token_hidden + self.token_pos_emb(pos)

        patch_fused, _ = self.patcher(token_hidden)
        token_hidden = self.blocks(patch_fused, mask=_causal_mask(token_t, token_t, x.device))

        token_hidden = self.ln_f(token_hidden)
        logits = self.lm_head(token_hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(1e-5, temperature)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
