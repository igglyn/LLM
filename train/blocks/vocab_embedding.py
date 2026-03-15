from __future__ import annotations

from dataclasses import dataclass

import torch

from train.specs import RuntimeState


@dataclass(frozen=True)
class VocabEmbeddingBlock:
    vocab_size: int

    @property
    def block_name(self) -> str:
        return "VocabEmbedding"

    def run(self, state: RuntimeState) -> RuntimeState:
        tensor = state.tensor
        if tensor is not None:
            embedding = torch.arange(tensor.shape[-1], dtype=tensor.dtype, device=tensor.device)
            embedding = embedding.view(1, 1, tensor.shape[-1])
            token_ids = torch.remainder(tensor[..., 0], float(self.vocab_size)).to(torch.long)
            token_embed = embedding.expand(tensor.shape[0], tensor.shape[1], -1)
            tensor = token_embed + token_ids.unsqueeze(-1).to(tensor.dtype)

        return RuntimeState(
            text=state.text,
            execution_trace=[*state.execution_trace, f"VocabEmbedding(vocab_size={self.vocab_size})"],
            moe_metrics=dict(state.moe_metrics),
            tensor_shape=tuple(tensor.shape) if tensor is not None else state.tensor_shape,
            tensor=tensor,
        )
