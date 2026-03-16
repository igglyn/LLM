from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from train.specs import RuntimeState


class VocabEmbeddingModule(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


@dataclass(frozen=True)
class VocabEmbeddingBlock:
    vocab_size: int
    d_model: int

    @property
    def block_name(self) -> str:
        return "VocabEmbedding"

    def build(self) -> VocabEmbeddingModule:
        return VocabEmbeddingModule(self.vocab_size, self.d_model)

    def run(self, state: RuntimeState) -> RuntimeState:
        # runtime smoke/trace path, no training
        tensor = state.tensor
        return RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, f"VocabEmbedding(vocab_size={self.vocab_size},d_model={self.d_model})"],
            moe_metrics=dict(state.moe_metrics),
            tensor_shape=tuple(tensor.shape) if tensor is not None else state.tensor_shape,
            tensor=tensor,
        )
