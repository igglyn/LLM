from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.patch_size = patch_size
        self.max_patch_len = math.ceil(seq_len / patch_size)

        self.token_emb = nn.Embedding(vocab_size, d_model)
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
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def _patchify(self, token_hidden: torch.Tensor) -> tuple[torch.Tensor, int]:
        bsz, t, d_model = token_hidden.shape
        if t > self.seq_len:
            raise ValueError(f"sequence length {t} exceeds model limit {self.seq_len}")

        padded_t = math.ceil(t / self.patch_size) * self.patch_size
        if padded_t != t:
            pad = torch.zeros((bsz, padded_t - t, d_model), device=token_hidden.device, dtype=token_hidden.dtype)
            token_hidden = torch.cat([token_hidden, pad], dim=1)

        n_patches = padded_t // self.patch_size
        patch_hidden = token_hidden.view(bsz, n_patches, self.patch_size, d_model).mean(dim=2)
        return patch_hidden, t

    def _unpatchify(self, patch_hidden: torch.Tensor, original_t: int) -> torch.Tensor:
        token_hidden = patch_hidden.repeat_interleave(self.patch_size, dim=1)
        return token_hidden[:, :original_t, :]

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        token_hidden = self.token_emb(x)
        patch_hidden, original_t = self._patchify(token_hidden)

        _, patch_t, _ = patch_hidden.shape
        pos = torch.arange(0, patch_t, device=x.device).unsqueeze(0)
        patch_hidden = patch_hidden + self.patch_pos_emb(pos)

        patch_mask = torch.full((patch_t, patch_t), float("-inf"), device=x.device)
        patch_mask = torch.triu(patch_mask, diagonal=1)
        patch_hidden = self.blocks(patch_hidden, mask=patch_mask)

        token_hidden = self._unpatchify(patch_hidden, original_t)
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
