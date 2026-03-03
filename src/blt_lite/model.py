from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((q_len, k_len), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.size(-1) // 2]
    x2 = x[..., x.size(-1) // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


def _build_rope_cache(max_seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires even head dimension")
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    return cos, sin


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor | None = None, rope_sin: torch.Tensor | None = None) -> torch.Tensor:
        bsz, t, d = x.shape
        qkv = self.qkv(x).view(bsz, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if rope_cos is not None and rope_sin is not None:
            cos = rope_cos[:, :, :t, :].to(device=x.device, dtype=q.dtype)
            sin = rope_sin[:, :, :t, :].to(device=x.device, dtype=q.dtype)
            q, k = _apply_rope(q, k, cos, sin)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(bsz, t, d)
        return self.resid_dropout(self.out_proj(attn))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor | None = None, rope_sin: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), rope_cos=rope_cos, rope_sin=rope_sin)
        x = x + self.mlp(self.ln_2(x))
        return x


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
        use_amp: bool = True,
        amp_dtype: str = "float16",
        pos_encoding: str = "learned",
    ):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        self.seq_len = seq_len
        self.use_amp = use_amp
        self.amp_dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
        if pos_encoding not in {"learned", "rope"}:
            raise ValueError("pos_encoding must be one of: learned, rope")
        self.pos_encoding = pos_encoding

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.token_pos_emb = nn.Embedding(seq_len, d_model) if pos_encoding == "learned" else None

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

        self.blocks = nn.ModuleList([TransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        if self.pos_encoding == "rope":
            head_dim = d_model // n_heads
            cos, sin = _build_rope_cache(seq_len, head_dim)
            self.register_buffer("rope_cos_cached", cos, persistent=False)
            self.register_buffer("rope_sin_cached", sin, persistent=False)
        else:
            self.register_buffer("rope_cos_cached", None, persistent=False)
            self.register_buffer("rope_sin_cached", None, persistent=False)

    def load_patcher_checkpoint(self, path: str | Path, map_location: torch.device | str | None = None) -> None:
        ckpt = torch.load(Path(path), map_location=map_location)
        state = ckpt["patcher"] if isinstance(ckpt, dict) and "patcher" in ckpt else ckpt
        self.patcher.load_state_dict(state)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        _, token_t = x.shape
        if token_t > self.seq_len:
            raise ValueError(f"sequence length {token_t} exceeds model limit {self.seq_len}")

        amp_enabled = self.use_amp and (x.device.type == "cuda")
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=self.amp_dtype):
            token_hidden = self.token_emb(x)
            if self.token_pos_emb is not None:
                pos = torch.arange(0, token_t, device=x.device).unsqueeze(0)
                token_hidden = token_hidden + self.token_pos_emb(pos)

            patch_fused, _ = self.patcher(token_hidden)

            rope_cos = self.rope_cos_cached if self.pos_encoding == "rope" else None
            rope_sin = self.rope_sin_cached if self.pos_encoding == "rope" else None
            token_hidden = patch_fused
            for block in self.blocks:
                token_hidden = block(token_hidden, rope_cos=rope_cos, rope_sin=rope_sin)

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
