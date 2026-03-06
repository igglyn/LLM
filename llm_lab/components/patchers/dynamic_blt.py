"""Dynamic-boundary raw-byte patcher scaffold."""

from __future__ import annotations

import torch
from torch import nn

from llm_lab.interfaces.patcher import Patcher
from llm_lab.registry import register_patcher


@register_patcher("dynamic_blt")
class DynamicBLTPatcher(nn.Module, Patcher):
    """Scaffold patcher with heuristic variable-length boundaries.

    This module is intentionally structured so the segmentation routine can be
    replaced by a learned boundary detector later while preserving interface.
    """

    _WHITESPACE_OR_PUNCT = {
        9,   # \t
        10,  # \n
        13,  # \r
        32,  # space
        33, 44, 46, 58, 59, 63,  # ! , . : ; ?
    }

    def __init__(
        self,
        max_patch_size: int = 8,
        min_patch_size: int = 2,
        d_model: int = 16,
        heuristic: str = "fixed",
    ) -> None:
        super().__init__()
        if max_patch_size <= 0:
            raise ValueError("max_patch_size must be > 0")
        if min_patch_size <= 0:
            raise ValueError("min_patch_size must be > 0")
        if min_patch_size > max_patch_size:
            raise ValueError("min_patch_size must be <= max_patch_size")
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if heuristic not in {"fixed", "whitespace"}:
            raise ValueError("heuristic must be 'fixed' or 'whitespace'")

        self.max_patch_size = max_patch_size
        self.min_patch_size = min_patch_size
        self.d_model = d_model
        self.heuristic = heuristic

        self.byte_embed = nn.Embedding(256, d_model)

    def _segment_fixed(self, seq: torch.Tensor) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        start = 0
        t = int(seq.numel())
        while start < t:
            end = min(start + self.max_patch_size, t)
            spans.append((start, end))
            start = end
        return spans

    def _segment_whitespace(self, seq: torch.Tensor) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        t = int(seq.numel())
        start = 0
        while start < t:
            max_end = min(start + self.max_patch_size, t)
            min_end = min(start + self.min_patch_size, t)

            chosen = max_end
            if min_end < max_end:
                window = seq[min_end:max_end]
                rel = None
                for i in range(window.numel() - 1, -1, -1):
                    if int(window[i].item()) in self._WHITESPACE_OR_PUNCT:
                        rel = i
                        break
                if rel is not None:
                    chosen = min_end + rel + 1

            spans.append((start, chosen))
            start = chosen
        return spans

    def _segment(self, seq: torch.Tensor) -> list[tuple[int, int]]:
        if self.heuristic == "fixed":
            return self._segment_fixed(seq)
        return self._segment_whitespace(seq)

    def _encode_patches(self, seq: torch.Tensor, spans: list[tuple[int, int]]) -> torch.Tensor:
        # Replaceable by learned boundary/aggregation logic later.
        patch_vecs: list[torch.Tensor] = []
        for s, e in spans:
            token_ids = seq[s:e].to(torch.long)
            emb = self.byte_embed(token_ids)
            patch_vecs.append(emb.mean(dim=0))
        return torch.stack(patch_vecs, dim=0)

    def forward(self, x_u8: torch.Tensor) -> torch.Tensor:
        if x_u8.ndim != 2:
            raise ValueError("DynamicBLTPatcher.forward expects rank-2 tensor [B, T].")
        if x_u8.dtype != torch.uint8:
            raise ValueError("DynamicBLTPatcher.forward expects torch.uint8 input.")

        encoded: list[torch.Tensor] = []
        max_tp = 0
        for b in range(x_u8.shape[0]):
            seq = x_u8[b]
            spans = self._segment(seq)
            h = self._encode_patches(seq, spans)
            encoded.append(h)
            max_tp = max(max_tp, h.shape[0])

        out = x_u8.new_zeros((x_u8.shape[0], max_tp, self.d_model), dtype=torch.float32)
        for b, h in enumerate(encoded):
            out[b, : h.shape[0], :] = h
        return out
