from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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


def _run_block_with_optional_checkpoint(
    block: nn.Module,
    x: torch.Tensor,
    rope_cos: torch.Tensor | None,
    rope_sin: torch.Tensor | None,
    grad_checkpointing: bool,
) -> torch.Tensor:
    if grad_checkpointing and x.requires_grad:
        def _forward(inp: torch.Tensor) -> torch.Tensor:
            return block(inp, rope_cos=rope_cos, rope_sin=rope_sin)

        return checkpoint(_forward, x, use_reentrant=False)
    return block(x, rope_cos=rope_cos, rope_sin=rope_sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, use_flash_attention: bool = True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        dropout_p = self.dropout if self.training else 0.0
        if self.use_flash_attention and q.device.type == "cuda":
            attention_ns = getattr(torch.nn, "attention", None)
            sdpa_kernel = getattr(attention_ns, "sdpa_kernel", None) if attention_ns is not None else None
            sdp_backend = getattr(attention_ns, "SDPBackend", None) if attention_ns is not None else None
            if sdpa_kernel is not None and sdp_backend is not None:
                backends = [
                    sdp_backend.FLASH_ATTENTION,
                    sdp_backend.EFFICIENT_ATTENTION,
                    sdp_backend.MATH,
                ]
                with sdpa_kernel(backends=backends):
                    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor | None = None, rope_sin: torch.Tensor | None = None) -> torch.Tensor:
        bsz, t, d_model = x.shape
        qkv = self.qkv(x).view(bsz, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if rope_cos is not None and rope_sin is not None:
            cos = rope_cos[:, :, :t, :].to(device=x.device, dtype=q.dtype)
            sin = rope_sin[:, :, :t, :].to(device=x.device, dtype=q.dtype)
            q, k = _apply_rope(q, k, cos, sin)

        out = self._attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(bsz, t, d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, use_flash_attention: bool = True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, use_flash_attention=use_flash_attention)
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
    def __init__(
        self,
        in_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        patch_size: int,
        seq_len: int,
        pos_encoding: str = "learned",
        grad_checkpointing: bool = False,
        flash_attention: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.max_patch_len = math.ceil(seq_len / patch_size)
        self.grad_checkpointing = grad_checkpointing
        self.patch_proj = nn.Linear(in_dim * patch_size, d_model)
        if pos_encoding not in {"learned", "rope"}:
            raise ValueError("patcher pos_encoding must be one of: learned, rope")
        self.pos_encoding = pos_encoding
        self.patch_pos_emb = nn.Embedding(self.max_patch_len, d_model) if pos_encoding == "learned" else None
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout, use_flash_attention=flash_attention)
                for _ in range(max(1, n_layers))
            ]
        )

        if self.pos_encoding == "rope":
            head_dim = d_model // n_heads
            cos, sin = _build_rope_cache(self.max_patch_len, head_dim)
            self.register_buffer("rope_cos_cached", cos, persistent=False)
            self.register_buffer("rope_sin_cached", sin, persistent=False)
        else:
            self.register_buffer("rope_cos_cached", None, persistent=False)
            self.register_buffer("rope_sin_cached", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, token_t, d_model = x.shape
        padded_t = math.ceil(token_t / self.patch_size) * self.patch_size
        if padded_t != token_t:
            pad = torch.zeros((bsz, padded_t - token_t, d_model), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        patch_t = padded_t // self.patch_size
        grouped = x.view(bsz, patch_t, self.patch_size, d_model)
        out = self.patch_proj(grouped.reshape(bsz, patch_t, self.patch_size * d_model))

        if self.patch_pos_emb is not None:
            pos = torch.arange(0, patch_t, device=x.device).unsqueeze(0)
            out = out + self.patch_pos_emb(pos)

        rope_cos = self.rope_cos_cached if self.pos_encoding == "rope" else None
        rope_sin = self.rope_sin_cached if self.pos_encoding == "rope" else None
        for block in self.blocks:
            out = _run_block_with_optional_checkpoint(block, out, rope_cos, rope_sin, self.training and self.grad_checkpointing)
        return out


class PatchDecoder(nn.Module):
    def __init__(
        self,
        token_dim: int,
        d_model: int,
        out_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        patch_size: int,
        seq_len: int,
        pos_encoding: str = "learned",
        grad_checkpointing: bool = False,
        flash_attention: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.token_seq_len = seq_len
        self.grad_checkpointing = grad_checkpointing
        self.query_proj = nn.Linear(token_dim, d_model)
        if pos_encoding not in {"learned", "rope"}:
            raise ValueError("patcher pos_encoding must be one of: learned, rope")
        self.pos_encoding = pos_encoding
        self.token_pos_emb = nn.Embedding(self.token_seq_len, d_model) if pos_encoding == "learned" else None

        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout, use_flash_attention=flash_attention)
                for _ in range(max(1, n_layers))
            ]
        )
        self.out_proj = nn.Linear(d_model, out_dim)

        if self.pos_encoding == "rope":
            head_dim = d_model // n_heads
            cos, sin = _build_rope_cache(self.token_seq_len, head_dim)
            self.register_buffer("rope_cos_cached", cos, persistent=False)
            self.register_buffer("rope_sin_cached", sin, persistent=False)
        else:
            self.register_buffer("rope_cos_cached", None, persistent=False)
            self.register_buffer("rope_sin_cached", None, persistent=False)

    def _token_to_patch_mask(self, token_len: int, patch_len: int, device: torch.device) -> torch.Tensor:
        token_positions = torch.arange(token_len, device=device)
        patch_positions = torch.arange(patch_len, device=device)
        allowed = patch_positions.unsqueeze(0) <= torch.div(token_positions.unsqueeze(1), self.patch_size, rounding_mode="floor")
        mask = torch.full((token_len, patch_len), float("-inf"), device=device)
        return mask.masked_fill(allowed, 0.0)

    def forward(self, token_hidden: torch.Tensor, patch_hidden: torch.Tensor) -> torch.Tensor:
        token_t = token_hidden.size(1)
        patch_t = patch_hidden.size(1)
        query = self.query_proj(token_hidden)
        token_patch_mask = self._token_to_patch_mask(token_t, patch_t, token_hidden.device)
        cross_hidden, _ = self.cross_attn(query=query, key=patch_hidden, value=patch_hidden, attn_mask=token_patch_mask, need_weights=False)
        out = query + cross_hidden

        if self.token_pos_emb is not None:
            pos = torch.arange(0, token_t, device=token_hidden.device).unsqueeze(0)
            out = out + self.token_pos_emb(pos)

        rope_cos = self.rope_cos_cached if self.pos_encoding == "rope" else None
        rope_sin = self.rope_sin_cached if self.pos_encoding == "rope" else None
        for block in self.blocks:
            out = _run_block_with_optional_checkpoint(block, out, rope_cos, rope_sin, self.training and self.grad_checkpointing)
        return self.out_proj(out)


class PatcherAutoencoder(nn.Module):
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
        pos_encoding: str = "learned",
        grad_checkpointing: bool = False,
        flash_attention: bool = True,
    ):
        super().__init__()
        self.token_seq_len = seq_len
        self.encoder = PatchEncoder(
            in_dim=in_dim,
            d_model=latent_dim,
            n_heads=n_heads,
            n_layers=encoder_layers,
            dropout=dropout,
            patch_size=patch_size,
            seq_len=self.token_seq_len,
            pos_encoding=pos_encoding,
            grad_checkpointing=grad_checkpointing,
            flash_attention=flash_attention,
        )
        self.decoder = PatchDecoder(
            token_dim=in_dim,
            d_model=latent_dim,
            out_dim=out_dim,
            n_heads=n_heads,
            n_layers=decoder_layers,
            dropout=dropout,
            patch_size=patch_size,
            seq_len=self.token_seq_len,
            pos_encoding=pos_encoding,
            grad_checkpointing=grad_checkpointing,
            flash_attention=flash_attention,
        )

    def forward(self, token_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lat = self.encoder(token_hidden)
        recon = self.decoder(token_hidden, lat)
        return recon, lat


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
        patcher_pos_encoding: str = "learned",
        patcher2_patch_size: int = 2,
        patcher2_latent_dim: int = 384,
        patcher2_encoder_layers: int = 2,
        patcher2_decoder_layers: int = 2,
        patcher2_heads: int | None = None,
        patcher2_dropout: float | None = None,
        patcher2_pos_encoding: str = "learned",
        use_amp: bool = True,
        amp_dtype: str = "float16",
        pos_encoding: str = "learned",
        grad_checkpointing: bool = False,
        flash_attention: bool = True,
        patcher_grad_checkpointing: bool = False,
        patcher2_grad_checkpointing: bool = False,
        patcher_flash_attention: bool = True,
        patcher2_flash_attention: bool = True,
    ):
        super().__init__()
        if patch_size <= 0 or patcher2_patch_size <= 0:
            raise ValueError("patch sizes must be > 0")
        self.seq_len = seq_len  # large-patch context length
        self.large_patch_size = patch_size * patcher2_patch_size
        self.token_seq_len = seq_len * self.large_patch_size
        self.use_amp = use_amp
        self.amp_dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
        self.grad_checkpointing = grad_checkpointing
        if pos_encoding not in {"learned", "rope"}:
            raise ValueError("pos_encoding must be one of: learned, rope")
        self.pos_encoding = pos_encoding

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.token_pos_emb = nn.Embedding(self.token_seq_len, d_model) if pos_encoding == "learned" else None

        self.patcher = PatcherAutoencoder(
            in_dim=d_model,
            latent_dim=patcher_latent_dim,
            out_dim=d_model,
            patch_size=patch_size,
            seq_len=self.token_seq_len,
            encoder_layers=patcher_encoder_layers,
            decoder_layers=patcher_decoder_layers,
            n_heads=int(patcher_heads or n_heads),
            dropout=float(dropout if patcher_dropout is None else patcher_dropout),
            pos_encoding=patcher_pos_encoding,
            grad_checkpointing=patcher_grad_checkpointing,
            flash_attention=patcher_flash_attention,
        )
        self.patcher2 = PatcherAutoencoder(
            in_dim=d_model,
            latent_dim=patcher2_latent_dim,
            out_dim=d_model,
            patch_size=patcher2_patch_size,
            seq_len=self.token_seq_len,
            encoder_layers=patcher2_encoder_layers,
            decoder_layers=patcher2_decoder_layers,
            n_heads=int(patcher2_heads or n_heads),
            dropout=float(dropout if patcher2_dropout is None else patcher2_dropout),
            pos_encoding=patcher2_pos_encoding,
            grad_checkpointing=patcher2_grad_checkpointing,
            flash_attention=patcher2_flash_attention,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout, use_flash_attention=flash_attention)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        if self.pos_encoding == "rope":
            head_dim = d_model // n_heads
            cos, sin = _build_rope_cache(self.token_seq_len, head_dim)
            self.register_buffer("rope_cos_cached", cos, persistent=False)
            self.register_buffer("rope_sin_cached", sin, persistent=False)
        else:
            self.register_buffer("rope_cos_cached", None, persistent=False)
            self.register_buffer("rope_sin_cached", None, persistent=False)

    def load_patcher_checkpoint(self, path: str | Path, map_location: torch.device | str | None = None) -> None:
        ckpt = torch.load(Path(path), map_location=map_location)
        state = ckpt["patcher"] if isinstance(ckpt, dict) and "patcher" in ckpt else ckpt
        self.patcher.load_state_dict(state)

    def load_patcher2_checkpoint(self, path: str | Path, map_location: torch.device | str | None = None) -> None:
        ckpt = torch.load(Path(path), map_location=map_location)
        state = ckpt["patcher2"] if isinstance(ckpt, dict) and "patcher2" in ckpt else ckpt
        self.patcher2.load_state_dict(state)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        _, token_t = x.shape
        if token_t > self.token_seq_len:
            raise ValueError(f"sequence length {token_t} exceeds model token limit {self.token_seq_len} (derived from model.seq_len large patches)")

        amp_enabled = self.use_amp and (x.device.type == "cuda")
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=self.amp_dtype):
            h = self.token_emb(x)
            if self.token_pos_emb is not None:
                pos = torch.arange(0, token_t, device=x.device).unsqueeze(0)
                h = h + self.token_pos_emb(pos)

            h, _ = self.patcher(h)
            h, _ = self.patcher2(h)

            rope_cos = self.rope_cos_cached if self.pos_encoding == "rope" else None
            rope_sin = self.rope_sin_cached if self.pos_encoding == "rope" else None
            for block in self.blocks:
                h = _run_block_with_optional_checkpoint(block, h, rope_cos, rope_sin, self.training and self.grad_checkpointing)

            h = self.ln_f(h)
            logits = self.lm_head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.token_seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(1e-5, temperature)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
