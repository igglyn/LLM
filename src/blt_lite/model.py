from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((q_len, k_len), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


class PatchEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float, patch_size: int, seq_len: int):
        super().__init__()
        self.patch_size = patch_size
        self.max_patch_len = math.ceil(seq_len / patch_size)
        self.patch_proj = nn.Linear(d_model * patch_size, d_model)
        self.patch_pos_emb = nn.Embedding(self.max_patch_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=max(1, n_layers))

    def forward(self, token_hidden: torch.Tensor) -> torch.Tensor:
        bsz, token_t, d_model = token_hidden.shape
        padded_t = math.ceil(token_t / self.patch_size) * self.patch_size
        if padded_t != token_t:
            pad = torch.zeros(
                (bsz, padded_t - token_t, d_model),
                device=token_hidden.device,
                dtype=token_hidden.dtype,
            )
            token_hidden = torch.cat([token_hidden, pad], dim=1)

        patch_t = padded_t // self.patch_size
        grouped = token_hidden.view(bsz, patch_t, self.patch_size, d_model)
        patch_hidden = self.patch_proj(grouped.reshape(bsz, patch_t, self.patch_size * d_model))

        patch_pos = torch.arange(0, patch_t, device=token_hidden.device).unsqueeze(0)
        patch_hidden = patch_hidden + self.patch_pos_emb(patch_pos)

        patch_mask = _causal_mask(patch_t, patch_t, token_hidden.device)
        return self.blocks(patch_hidden, mask=patch_mask)


class PatchDecoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def _token_to_patch_mask(self, token_len: int, patch_len: int, device: torch.device) -> torch.Tensor:
        token_positions = torch.arange(token_len, device=device)
        patch_positions = torch.arange(patch_len, device=device)
        allowed = patch_positions.unsqueeze(0) <= torch.div(
            token_positions.unsqueeze(1), self.patch_size, rounding_mode="floor"
        )
        mask = torch.full((token_len, patch_len), float("-inf"), device=device)
        return mask.masked_fill(allowed, 0.0)

    def forward(self, token_hidden: torch.Tensor, patch_hidden: torch.Tensor) -> torch.Tensor:
        token_t = token_hidden.size(1)
        patch_t = patch_hidden.size(1)
        token_patch_mask = self._token_to_patch_mask(token_t, patch_t, token_hidden.device)
        cross_hidden, _ = self.cross_attn(
            query=token_hidden,
            key=patch_hidden,
            value=patch_hidden,
            attn_mask=token_patch_mask,
            need_weights=False,
        )
        return token_hidden + cross_hidden


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
    ):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        self.seq_len = seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.token_pos_emb = nn.Embedding(seq_len, d_model)

        self.patch_encoder = PatchEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=max(1, n_layers // 2),
            dropout=dropout,
            patch_size=patch_size,
            seq_len=seq_len,
        )
        self.patch_decoder = PatchDecoder(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            patch_size=patch_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        _, token_t = x.shape
        if token_t > self.seq_len:
            raise ValueError(f"sequence length {token_t} exceeds model limit {self.seq_len}")

        token_hidden = self.token_emb(x)
        token_pos = torch.arange(0, token_t, device=x.device).unsqueeze(0)
        token_hidden = token_hidden + self.token_pos_emb(token_pos)

        patch_hidden = self.patch_encoder(token_hidden)
        fused_hidden = self.patch_decoder(token_hidden, patch_hidden)

        token_mask = _causal_mask(token_t, token_t, x.device)
        token_hidden = self.blocks(fused_hidden, mask=token_mask)
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
