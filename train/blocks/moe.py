from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.specs import ExpertRuntime, RuntimeState


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.ff1 = nn.Linear(d_model, d_model * 4, bias=False)
        self.ff2 = nn.Linear(d_model * 4, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff2(self.dropout(F.gelu(self.ff1(x))))


class MoEModule(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(d_model, dropout) for _ in range(num_experts)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_flat = x.view(B * T, C)

        # router logits and top-k selection
        logits = self.router(x_flat)
        weights, indices = torch.topk(F.softmax(logits, dim=-1), self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # dispatch to experts
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = indices[:, k]
            expert_weights = weights[:, k].unsqueeze(-1)
            for i, expert in enumerate(self.experts):
                mask = (expert_idx == i)
                if mask.any():
                    output[mask] += expert_weights[mask] * expert(x_flat[mask])

        # load balancing loss stored for later use
        self.aux_loss = self._aux_loss(logits)

        return self.dropout(output.view(B, T, C))

    def _aux_loss(self, logits: torch.Tensor) -> torch.Tensor:
        # encourage uniform expert utilization
        probs = F.softmax(logits, dim=-1)
        avg_probs = probs.mean(dim=0)
        return (avg_probs * torch.log(avg_probs + 1e-9)).sum() * -1.0


@dataclass(frozen=True)
class MixOfExpertsBlock:
    name: str
    experts: list[ExpertRuntime]
    d_model: int
    top_k: int = 2
    dropout: float = 0.1

    @property
    def block_name(self) -> str:
        return "MixOfExperts"

    def build(self) -> MoEModule:
        return MoEModule(
            d_model=self.d_model,
            num_experts=len(self.experts),
            top_k=self.top_k,
            dropout=self.dropout,
        )

    def run(self, state: RuntimeState) -> RuntimeState:
        # runtime smoke/trace path
        metrics = dict(state.moe_metrics)
        usage = dict(metrics.get(self.name, {}))
        usage["num_experts"] = len(self.experts)
        usage["top_k"] = self.top_k
        metrics[self.name] = usage

        return RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, f"MoE({self.name},experts={len(self.experts)},top_k={self.top_k})"],
            moe_metrics=metrics,
            tensor_shape=state.tensor_shape,
            tensor=state.tensor,
        )
