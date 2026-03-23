"""nnv2.py — NNv2Block (U1XToU1X on emit space) and NNv2Adapter (U1XToInc, float output).

Pipeline:
    NNv5Block   — U1XToU1X on token space   → choices, emit bitsets
    NNv2Block   — U1XToU1X on emit space    → compressed choices
    NNv2Adapter — U1XToInc, float emit      → float vector → transformer
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nnv5 import (NNv5Block, NNv5CapacityError,
                   _forward_all_shards, _pack_choices_torch,
                   _unpack_choices_torch, _tc)


# ---------------------------------------------------------------------------
# NNv2Block — pure U1XToU1X on emit space
# ---------------------------------------------------------------------------

class NNv2Block:
    """
    U1XToU1X operating on NNv5's emit (group membership) bitsets.

    compress_range reads directly from NNv5's on-device tensors — no transfer.
    State layout: (2, n_shards, shard_size, emit_words) int64 on device.

    Tracks per-case activation counts for waterfall split ratio computation.
    When a match split occurs, notifies the adapter via adapter.match_split().
    """

    def __init__(self, group_capacity: int, case_capacity: int,
                 shard_size: int = 32, device: torch.device = None):
        assert case_capacity & (case_capacity - 1) == 0, "case_capacity must be power of 2"
        assert shard_size    & (shard_size    - 1) == 0, "shard_size must be power of 2"
        assert case_capacity >= shard_size

        self.group_capacity = group_capacity
        self.case_capacity  = case_capacity
        self.shard_size     = shard_size
        self.device         = device or torch.device('cpu')
        self.shard_bits     = shard_size.bit_length() - 1
        self.array_used     = 0
        self.emit_words     = (group_capacity + 63) // 64

        W   = self.emit_words
        ss  = shard_size
        n_s = case_capacity // shard_size

        self.match = torch.zeros(2, n_s, ss, W, dtype=torch.int64, device=self.device)
        self.match[1] = -1
        self.emit  = torch.zeros(2, n_s, ss, W, dtype=torch.int64, device=self.device)

        # Per-case activation counts for split ratio computation
        self.activation_counts = torch.zeros(case_capacity, dtype=torch.int64,
                                             device=self.device)

    def _shard_slot(self, flat_idx: int):
        return flat_idx >> self.shard_bits, flat_idx & (self.shard_size - 1)

    @property
    def n_shards(self) -> int:
        return self.case_capacity // self.shard_size

    @property
    def n_shards_used(self) -> int:
        if self.array_used == 0:
            return 0
        return (self.array_used + self.shard_size - 1) >> self.shard_bits

    def _has_pos_and_neg(self, pos: torch.Tensor, neg: torch.Tensor) -> bool:
        return pos.any().item() and neg.any().item()

    # ------------------------------------------------------------------
    # Compress — reads directly from NNv5 on-device tensors
    # ------------------------------------------------------------------

    def compress(self, nnv5: NNv5Block, adapter: "NNv2Adapter" = None):
        self.compress_range(nnv5, 0, nnv5.array_used, adapter)

    def compress_range(self, nnv5: NNv5Block, start: int, end: int,
                       adapter: "NNv2Adapter" = None):
        """
        Run compression on NNv5 cases [start, end).
        Reads directly from nnv5.match / nnv5.emit on device.
        Notifies adapter of match splits for waterfall float_emit update.
        """
        if nnv5.emit_used == 0 or start >= end:
            return

        n_new = end - start
        E     = self.array_used
        EW    = nnv5.emit_words
        C5    = nnv5.array_used

        nnv5_emit  = nnv5.emit.reshape(2, -1, EW)[:, :C5]

        src_ep = nnv5_emit[0, start:end]
        src_en = nnv5_emit[1, start:end]

        if E == 0:
            n = min(n_new, self.case_capacity)
            for i in range(n):
                s, slot = self._shard_slot(i)
                self.match[0, s, slot] = src_ep[i]
                self.match[1, s, slot] = src_en[i]
                self.emit[0, s, slot]  = src_ep[i]
                self.emit[1, s, slot]  = src_en[i]
            self.array_used = n
            if adapter is not None:
                adapter.sync(self.array_used)
            return

        my_ep = self.emit[0].reshape(-1, self.emit_words)[:E]
        my_en = self.emit[1].reshape(-1, self.emit_words)[:E]

        reduce_match = (
            (src_ep[:, None, :] == my_ep[None]).all(dim=2) &
            (src_en[:, None, :] == my_en[None]).all(dim=2)
        )
        superset = (
            (my_ep[None] & src_ep[:, None, :]) == src_ep[:, None, :]
        ).all(dim=2) & ~reduce_match

        for i in range(n_new):
            ep_i = src_ep[i]
            en_i = src_en[i]

            # REDUCE
            reduce_cols = reduce_match[i].nonzero(as_tuple=True)[0]
            if reduce_cols.numel() > 0:
                ei = reduce_cols[0].item()
                s, slot = self._shard_slot(ei)
                self.match[0, s, slot] &= ep_i
                self.match[1, s, slot] &= en_i
                continue

            # SPLIT — on match patterns
            split_done = False
            for ei_t in superset[i].nonzero(as_tuple=True)[0]:
                ei = ei_t.item()
                s, slot = self._shard_slot(ei)
                ex_pos = self.emit[0, s, slot].clone()
                ex_neg = self.emit[1, s, slot].clone()

                sub_a_pos = ex_pos & ep_i
                sub_a_neg = ex_neg | en_i
                sub_b_pos = ex_pos & ~ep_i
                sub_b_neg = ex_neg

                if (self._has_pos_and_neg(sub_a_pos, sub_a_neg) and
                        self._has_pos_and_neg(sub_b_pos, sub_b_neg)):
                    if self.array_used >= self.case_capacity:
                        raise NNv5CapacityError(
                            f"NNv2Block capacity exceeded on split: "
                            f"used={self.array_used} >= {self.case_capacity}")

                    # Compute split ratio from activation counts
                    parent_count = self.activation_counts[ei].item()
                    # Child is new — estimate its share as half
                    # Will be refined by actual activations over time
                    ratio = 0.5 if parent_count == 0 else 0.5

                    self.emit[0, s, slot]  = sub_a_pos
                    self.emit[1, s, slot]  = sub_a_neg
                    self.match[0, s, slot] &= ep_i
                    self.match[1, s, slot] &= en_i

                    bi = self.array_used
                    sb, sslot = self._shard_slot(bi)
                    self.match[0, sb, sslot] = self.match[0, s, slot].clone()
                    self.match[1, sb, sslot] = self.match[1, s, slot].clone()
                    self.emit[0, sb, sslot]  = sub_b_pos
                    self.emit[1, sb, sslot]  = sub_b_neg
                    self.array_used += 1

                    # Notify adapter of match split for float_emit waterfall
                    if adapter is not None:
                        adapter.match_split(ei, bi, ratio)
                        adapter.sync(self.array_used)

                    split_done = True
                    break

            if split_done:
                continue

            # APPEND
            if self.array_used >= self.case_capacity:
                raise NNv5CapacityError(
                    f"NNv2Block capacity exceeded on append: "
                    f"used={self.array_used} >= {self.case_capacity}")
            idx = self.array_used
            s, slot = self._shard_slot(idx)
            self.match[0, s, slot] = ep_i
            self.match[1, s, slot] = en_i
            self.emit[0, s, slot]  = ep_i
            self.emit[1, s, slot]  = en_i
            self.array_used += 1

        if adapter is not None:
            adapter.sync(self.array_used)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, emit_in: torch.Tensor):
        """
        Subset match on emit bitsets.
        emit_in: (N, emit_words) int64 — NNv5's per-input emit output
        Updates activation_counts for matched cases.
        Returns: choices_packed (N, ceil(C/64)) int64
                 choices        (N, C) bool
        """
        N = emit_in.shape[0]
        C = self.array_used

        if C == 0:
            cp = torch.zeros(N, 0, dtype=torch.int64, device=self.device)
            return cp, torch.zeros(N, 0, dtype=torch.bool, device=self.device)

        emit_inv = ~emit_in

        choices_t, _, _, _, _, _, _ = _forward_all_shards(
            emit_in, emit_inv,
            self.match,
            self.emit,
            self.n_shards, self.shard_size)

        choices_t = choices_t[:, :C]

        # Update activation counts
        with torch.no_grad():
            self.activation_counts[:C] += choices_t.sum(dim=0)

        choices_packed = _pack_choices_torch(choices_t)
        return choices_packed, choices_t

    def stats(self) -> dict:
        return {
            "nnv2_cases_used":    self.array_used,
            "nnv2_case_capacity": self.case_capacity,
        }


# ---------------------------------------------------------------------------
# NNv2Adapter — U1XToInc with float emit, transformer boundary
# ---------------------------------------------------------------------------

class NNv2Adapter(nn.Module):
    """
    U1XToInc with float emit table and positional embedding.

    float_emit: (case_capacity, chunk_dim) nn.Parameter
        Zero rows are uninitialised — first MSE residual is frontloaded.
        Waterfall split conserves values across NNv2Block match splits.

    pos_embed:  (n_chunks, chunk_dim) nn.Parameter
        Learned positional offset per chunk, added after the matmul.

    Two update paths — structural (waterfall at split time) and
    gradient (MSE backprop every step) — operate at different timescales
    and do not conflict.
    """

    def __init__(self, chunk_dim: int, case_capacity: int, n_chunks: int = 1,
                 device: torch.device = None):
        super().__init__()
        self.chunk_dim      = chunk_dim
        self.case_capacity  = case_capacity
        self.n_chunks       = n_chunks
        self.device         = device or torch.device('cpu')
        self.array_used     = 0

        # float_emit: (case_capacity, chunk_dim) — zero = uninitialised
        self.float_emit = nn.Parameter(
            torch.zeros(case_capacity, chunk_dim, device=self.device))

        # pos_embed: (n_chunks, chunk_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(n_chunks, chunk_dim, device=self.device))

    def sync(self, new_count: int):
        """Register new cases — new rows stay zero for frontloading."""
        self.array_used = new_count

    def match_split(self, parent_idx: int, child_idx: int, ratio: float):
        """
        Waterfall split of float_emit when NNv2Block splits a match case.
        Parent keeps ratio of its value, child gets (1-ratio) of original.
        Conservation: parent_new + child = parent_original.
        Only sets child if it's uninitialised (zero).
        """
        with torch.no_grad():
            parent_val = self.float_emit[parent_idx].clone()
            self.float_emit[parent_idx] *= ratio
            if not self.float_emit[child_idx].any():
                self.float_emit[child_idx] = parent_val * (1.0 - ratio)

    def frontload(self, choices: torch.Tensor, residual: torch.Tensor):
        """
        Absorb first MSE residual into any uninitialised cases that fired.
        choices:  (N, C) bool
        residual: (N, chunk_dim) float32 — target - current prediction
        Called before computing loss so zero cases get a useful starting point.
        """
        C = self.array_used
        if C == 0:
            return
        with torch.no_grad():
            fired = choices.any(dim=0)[:C]           # (C,) which cases fired
            for ci in fired.nonzero(as_tuple=True)[0]:
                ci = ci.item()
                if not self.float_emit[ci].any():
                    # Average residual over inputs that activated this case
                    # Clamp to prevent runaway initialisation
                    mask = choices[:, ci]
                    init = residual[mask].mean(dim=0).detach()
                    init = init.clamp(-1.0, 1.0)  # conservative initialisation
                    self.float_emit[ci] = init

    def forward(self, choices: torch.Tensor, chunk_idx: int = 0) -> torch.Tensor:
        """
        choices:   (N, C) bool — from NNv2Block.forward()
        chunk_idx: which positional chunk this is
        returns:   (N, chunk_dim) float32
        """
        N = choices.shape[0]
        C = self.array_used

        if C == 0 or choices.shape[1] == 0:
            return self.pos_embed[chunk_idx].unsqueeze(0).expand(N, -1)

        out = choices.float() @ self.float_emit[:C]
        out = out + self.pos_embed[chunk_idx]
        return out

    def mse_loss(self, choices: torch.Tensor, target: torch.Tensor,
                 chunk_idx: int = 0) -> torch.Tensor:
        """
        Compute MSE loss against transformer chunk output.
        Frontloads uninitialised cases before computing loss.

        choices: (N, C) bool
        target:  (N, chunk_dim) float32
        returns: scalar MSE loss
        """
        C = self.array_used
        if C == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Frontload uninitialised cases that fired this batch
        with torch.no_grad():
            pred = choices.float() @ self.float_emit[:C]
            pred = pred + self.pos_embed[chunk_idx]
        self.frontload(choices, target - pred)

        out = self.forward(choices, chunk_idx)
        return F.mse_loss(out, target)

    def stats(self) -> dict:
        C = self.array_used
        initialised = int((self.float_emit[:C] != 0).any(dim=1).sum().item()) if C > 0 else 0
        return {
            "adapter_cases_used":    C,
            "adapter_initialised":   initialised,
            "adapter_uninitialised": C - initialised,
            "adapter_case_capacity": self.case_capacity,
            "adapter_n_chunks":      self.n_chunks,
        }
