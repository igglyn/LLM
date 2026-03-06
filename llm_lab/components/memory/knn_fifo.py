"""Simple FIFO kNN memory module."""

from __future__ import annotations

import torch

from llm_lab.interfaces.memory import Memory
from llm_lab.registry import register_memory


@register_memory("knn_fifo")
class KNNFIFOMemory(Memory):
    """FIFO memory with optional kNN retrieval.

    Stores hidden states in a bounded FIFO queue over time dimension.
    """

    def __init__(self, d_model: int = 16, max_tokens: int = 1024, k: int = 8) -> None:
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if k <= 0:
            raise ValueError("k must be > 0")

        self.d_model = d_model
        self.max_tokens = max_tokens
        self.k = k
        self._store: torch.Tensor | None = None

    def read(self) -> torch.Tensor:
        """Return full memory store with shape ``[B, Tm, D]``."""
        if self._store is None:
            return torch.zeros(1, 0, self.d_model)
        return self._store

    def write(self, h: torch.Tensor) -> None:
        """Append hidden states and keep only latest ``max_tokens`` tokens."""
        if h.ndim != 3:
            raise ValueError("KNNFIFOMemory.write expects [B, T, D]")
        if h.shape[-1] != self.d_model:
            raise ValueError("Last dimension mismatch with d_model")

        if self._store is None:
            self._store = h.detach()
        else:
            if self._store.shape[0] != h.shape[0]:
                raise ValueError("Batch size mismatch between memory and write input")
            self._store = torch.cat([self._store, h.detach()], dim=1)

        if self._store.shape[1] > self.max_tokens:
            self._store = self._store[:, -self.max_tokens :, :]

    def retrieve(self, query: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """Return kNN values from FIFO store for each query token.

        Args:
            query: Tensor with shape ``[B, Tq, D]``.
            k: Optional number of neighbors (defaults to constructor ``k``).

        Returns:
            Tensor with shape ``[B, Tq, K, D]`` where ``K=min(k, Tm)``.
        """
        if query.ndim != 3:
            raise ValueError("query must have shape [B, Tq, D]")
        if query.shape[-1] != self.d_model:
            raise ValueError("query last dimension must match d_model")

        memory = self.read().to(query.device)
        if memory.shape[1] == 0:
            return query.new_zeros(query.shape[0], query.shape[1], 0, query.shape[2])

        if memory.shape[0] != query.shape[0]:
            raise ValueError("Batch size mismatch between query and memory")

        k_eff = min(k or self.k, memory.shape[1])
        # L2 distance over D: [B, Tq, Tm]
        dist = torch.cdist(query, memory)
        idx = torch.topk(dist, k=k_eff, dim=-1, largest=False).indices

        # gather neighbors -> [B, Tq, K, D]
        memory_expand = memory[:, None, :, :].expand(-1, query.shape[1], -1, -1)
        idx_expand = idx[..., None].expand(-1, -1, -1, memory.shape[-1])
        return torch.gather(memory_expand, dim=2, index=idx_expand)
