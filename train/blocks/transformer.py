from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.specs import RuntimeState


class TransformerBlockModule(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff1 = nn.Linear(d_model, d_model * 4, bias=False)
        self.ff2 = nn.Linear(d_model * 4, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        # self attention with pre-norm
        residual = x
        x = self.norm1(x)
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale

        if mask is None:
            # causal mask
            causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(causal, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = self.out_proj(x)
        x = residual + self.dropout(x)

        # FFN with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.ff2(self.dropout(F.gelu(self.ff1(x))))
        x = residual + self.dropout(x)

        return x


@dataclass(frozen=True)
class TransformerBlock:
    d_model: int
    n_heads: int
    dropout: float = 0.1

    @property
    def block_name(self) -> str:
        return "Transformer"

    def build(self) -> TransformerBlockModule:
        return TransformerBlockModule(self.d_model, self.n_heads, self.dropout)

    def run(self, state: RuntimeState) -> RuntimeState:
        # runtime smoke/trace path, no training
        tensor = state.tensor
        return RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, f"Transformer(self_attention,d_model={self.d_model},n_heads={self.n_heads})"],
            moe_metrics=dict(state.moe_metrics),
            tensor_shape=tuple(tensor.shape) if tensor is not None else state.tensor_shape,
            tensor=tensor,
        )
