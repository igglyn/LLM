"""Byte-level LM head."""

from __future__ import annotations

import torch
from torch import nn

from llm_lab.interfaces.head import Head
from llm_lab.registry import register_head


@register_head("byte")
class ByteLMHead(nn.Module, Head):
    """Linear projection from hidden states to byte logits."""

    def __init__(self, d_model: int = 16) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        self.proj = nn.Linear(d_model, 256)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)
