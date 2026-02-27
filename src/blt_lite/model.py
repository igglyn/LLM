from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyPatchLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
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
        bsz, t = x.shape
        if t > self.seq_len:
            raise ValueError(f"sequence length {t} exceeds model limit {self.seq_len}")
        pos = torch.arange(0, t, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(pos)
        mask = torch.full((t, t), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        h = self.blocks(h, mask=mask)
        h = self.ln_f(h)
        logits = self.lm_head(h)

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
