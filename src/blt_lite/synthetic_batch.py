"""
synthetic_batch.py

Synthetic batch generation via NNv5 case structure.

Pipeline per real batch:
    1. Capture hidden states from trunk via forward hook
    2. Project float hidden states → bool vectors (learned projection)
    3. Run NNv5 forward → case matches, group affinities, diffs
    4. NNv5 assign → update persistent case table (new cases / groups)
    5. Reverse: sample group affinity vectors → OR matching case patterns
       → bool vectors → project back to float (learned inverse projection)
    6. Synthetic hidden states fed back into trunk as additional training signal

The NNv5 case table persists across batches and grows over training.
The bool projections are learned alongside the trunk.

Config keys (under synthetic_batch in tiny.yaml):
    enabled:          bool   — master switch
    bool_dim:         int    — bool vector dimensionality (default: d_model)
    n_synthetic:      int    — synthetic examples per real batch
    match_threshold:  float  — min affinity to count as matched (0.0-1.0)
    case_capacity:    int    — max NNv5 cases to allocate
    group_capacity:   int    — max NNv5 groups to allocate
    synth_loss_weight: float — weight of synthetic batch loss vs real loss
"""
from __future__ import annotations

import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Bool Projection (float → bool, bool → float)
# ---------------------------------------------------------------------------

class BoolProjection(nn.Module):
    """
    Learned projection between float hidden states and bool vectors.

    Forward:  (B, T, d_model) → (B, T, bool_dim)  bool tensor
    Inverse:  (B, T, bool_dim) → (B, T, d_model)  float tensor

    The forward uses straight-through estimator so gradients flow
    through the threshold during backprop.
    """
    def __init__(self, d_model: int, bool_dim: int):
        super().__init__()
        self.d_model  = d_model
        self.bool_dim = bool_dim
        self.to_bool  = nn.Linear(d_model, bool_dim)
        self.to_float = nn.Linear(bool_dim, d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model) float
        returns: (B, T, bool_dim) bool (straight-through in training)
        """
        logits = self.to_bool(x)
        soft   = torch.sigmoid(logits)
        # Straight-through: hard in forward, soft gradient in backward
        hard   = (soft > 0.5).float()
        return hard + (soft - soft.detach())

    def decode(self, b: torch.Tensor) -> torch.Tensor:
        """
        b: (B, T, bool_dim) float (0/1 or soft)
        returns: (B, T, d_model) float
        """
        return self.to_float(b)

    def encode_hard(self, x: torch.Tensor) -> np.ndarray:
        """
        Hard bool encoding for NNv5 — no gradients.
        returns: uint8 numpy array (B*T, bool_dim)
        """
        with torch.no_grad():
            logits = self.to_bool(x)
            hard   = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
        return hard.reshape(-1, self.bool_dim)


# ---------------------------------------------------------------------------
# NNv5 wrapper
# ---------------------------------------------------------------------------

class NNv5Block:
    """
    Thin wrapper around the NNv5 U1XToU1X logic operating on bool arrays.

    Accepts and returns numpy uint8 arrays where each element is 0 or 1,
    handling the bitpacking internally.

    The case table persists across calls — call assign() after forward()
    to update the table with new structure.
    """
    def __init__(self, bool_dim: int, case_capacity: int, group_capacity: int):
        self.bool_dim       = bool_dim
        self.case_capacity  = case_capacity
        self.group_capacity = group_capacity

        self.array_used = 0
        self.emit_used  = 0
        self.emit_words = (group_capacity + 63) // 64

        # match: (2, bool_dim, case_capacity)  uint8
        # plane 0 = positive (must be present)
        # plane 1 = negative (must be absent)
        self.match = np.zeros((2, bool_dim, case_capacity), dtype=np.uint8)

        # emit: (2, case_capacity, emit_words)  uint64
        self.emit = np.zeros((2, case_capacity, self.emit_words), dtype=np.uint64)

    def _set_emit_group_bit(self, case_idx, group_idx: int, *, is_member: bool):
        assert 0 <= group_idx < self.group_capacity
        word, bit = divmod(group_idx, 64)
        flag = np.uint64(1) << np.uint64(bit)
        plane = 0 if is_member else 1
        self.emit[plane, case_idx, word] |= flag

    def forward(self, tokens: np.ndarray):
        """
        tokens: (N, bool_dim) uint8
        returns: (diff, emit, was_matched, choices, activation_counts)
        """
        if self.array_used == 0:
            n = tokens.shape[0]
            tokens_inv = 1 - tokens
            inp = np.stack((tokens, tokens_inv), axis=1)       # (N, 2, bool_dim)
            # Deduplicate
            inp2, idxs = np.unique(inp, axis=0, return_index=True)
            diff        = inp2
            emit        = np.zeros((inp2.shape[0], 2, self.emit_words), dtype=np.uint64)
            was_matched = np.zeros(inp2.shape[0], dtype=bool)  # nothing matched
            choices     = np.zeros((n, 0), dtype=bool)
            counts      = np.zeros(0, dtype=np.uint64)
            return diff, emit, was_matched, choices, counts

        tokens_inv = 1 - tokens                                     # (N, bool_dim)
        inp = np.stack((tokens, tokens_inv), axis=1)                # (N, 2, bool_dim)

        match_view = self.match[None, ..., :self.array_used]        # (1, 2, D, C)
        inp_exp    = inp[..., None]                                  # (N, 2, D, 1)

        selection = inp_exp & match_view                            # (N, 2, D, C)
        choices   = (selection == match_view).all((1, 2))           # (N, C)

        was_matched = choices.any(axis=1)                           # (N,)

        emit_view = self.emit[None, :, :self.array_used]            # (1, 2, C, W)
        emit_broad = np.broadcast_to(
            emit_view,
            (tokens.shape[0], 2, self.array_used, self.emit_words),
        )
        emit_mask = self.emit[:, :self.array_used].any(axis=(0, 2)) # (C,)

        emit_out = np.bitwise_and.reduce(
            emit_broad,
            axis=2,
            where=choices[:, None, :, None] & emit_mask[None, None, :, None],
        )
        emit_out[~was_matched] = np.uint64(0)

        # Reverse: reconstruct input from matched cases
        match_broad = np.broadcast_to(
            match_view,
            (tokens.shape[0], 2, self.bool_dim, self.array_used),
        )
        reverse = np.bitwise_or.reduce(
            match_broad, axis=3,
            where=choices[:, None, None, :],
        )

        diff = reverse & inp ^ inp

        # Filter trivial diffs
        non_zero = diff[:, 0].any(axis=1)
        diff_cul = diff[non_zero]
        emit_cul = emit_out[non_zero]
        was_matched_cul = was_matched[non_zero]

        # Deduplicate
        if diff_cul.shape[0] > 0:
            diff_cul2, idxs = np.unique(diff_cul, axis=0, return_index=True)
            emit_cul2       = emit_cul[idxs]
            was_matched_cul2 = was_matched_cul[idxs]
        else:
            diff_cul2, emit_cul2, was_matched_cul2 = diff_cul, emit_cul, was_matched_cul

        counts = np.count_nonzero(choices, axis=0).astype(np.uint64)

        return diff_cul2, emit_cul2, was_matched_cul2, choices, counts

    def forward_torch(
        self,
        tokens: np.ndarray,
        device: torch.device,
        emit_chunk_size: int = 64,
    ):
        """
        Torch forward pass — same logic as forward() but runs on GPU.
        Uses int64 bitwise ops (reinterpreting uint64 emit arrays).
        Returns numpy arrays for assign compatibility.

        tokens: (N, bool_dim) uint8 numpy array
        """
        if self.array_used == 0:
            n = tokens.shape[0]
            tokens_inv = 1 - tokens
            inp = np.stack((tokens, tokens_inv), axis=1)
            inp2, idxs = np.unique(inp, axis=0, return_index=True)
            diff        = inp2
            emit        = np.zeros((inp2.shape[0], 2, self.emit_words), dtype=np.uint64)
            was_matched = np.zeros(inp2.shape[0], dtype=bool)
            choices     = np.zeros((n, 0), dtype=bool)
            counts      = np.zeros(0, dtype=np.uint64)
            return diff, emit, was_matched, choices, counts

        C = self.array_used
        N = tokens.shape[0]

        # Zero-copy view as torch — match stays uint8, emit reinterpreted as int64
        t_match = torch.from_numpy(self.match[:, :, :C]).to(device)          # (2, D, C) uint8
        t_emit  = torch.from_numpy(
            self.emit[:, :C].view(np.int64)
        ).to(device)                                                           # (2, C, W) int64

        t_inp = torch.from_numpy(
            np.stack((tokens, 1 - tokens), axis=1).copy()
        ).to(device)                                                           # (N, 2, D) uint8

        # --- FORWARD: subset match ---
        inp_exp    = t_inp.unsqueeze(-1)                                       # (N, 2, D, 1)
        match_view = t_match.unsqueeze(0)                                      # (1, 2, D, C)
        selection  = inp_exp & match_view                                      # (N, 2, D, C)
        choices_t  = (selection == match_view).all(dim=(1, 2))                # (N, C)
        was_matched_t = choices_t.any(dim=1)                                  # (N,)

        # --- EMIT AGGREGATION: vectorized int64 AND reduction ---
        emit_mask = t_emit.any(dim=(0, 2))                                    # (C,) bool

        # active: cases that fired AND have non-empty emit — (N, C)
        active = choices_t & emit_mask.unsqueeze(0)

        # Expand emit for broadcasting: (1, 2, C, W)
        emit_exp = t_emit.unsqueeze(0)                                         # (1, 2, C, W)

        # Replace inactive cases with AND identity (-1 in int64 = all 1s)
        active_exp = active.unsqueeze(1).unsqueeze(3)                         # (N, 1, C, 1)
        masked_emit = torch.where(
            active_exp.expand(N, 2, C, self.emit_words),
            emit_exp.expand(N, 2, C, self.emit_words),
            torch.full((1,), -1, dtype=torch.int64, device=device).expand(N, 2, C, self.emit_words),
        )

        # AND reduce over case dimension — functools.reduce over slices
        emit_out_t = functools.reduce(
            torch.bitwise_and,
            [masked_emit[:, :, ci] for ci in range(C)]
        )

        emit_out_t[~was_matched_t] = 0

        # --- REVERSE: OR match patterns of matched cases ---
        # choices_t: (N, C) → (N, 1, 1, C), match: (1, 2, D, C)
        cho_exp  = choices_t.unsqueeze(1).unsqueeze(1)                        # (N, 1, 1, C)
        mat_exp  = t_match.unsqueeze(0)                                        # (1, 2, D, C)
        contrib  = torch.where(
            cho_exp.expand(N, 2, self.bool_dim, C),
            mat_exp.expand(N, 2, self.bool_dim, C),
            torch.zeros(1, dtype=torch.uint8, device=device).expand(N, 2, self.bool_dim, C),
        )
        reverse_t = contrib.any(dim=3).to(torch.uint8)                        # (N, 2, D)

        diff_t = reverse_t & t_inp ^ t_inp

        # Back to numpy
        choices_np     = choices_t.cpu().numpy()
        diff_np        = diff_t.cpu().numpy()
        emit_out_np    = emit_out_t.cpu().numpy().view(np.uint64)
        was_matched_np = was_matched_t.cpu().numpy()

        # Filter trivial diffs
        non_zero = diff_np[:, 0].any(axis=1)
        diff_cul = diff_np[non_zero]
        emit_cul = emit_out_np[non_zero]
        was_matched_cul = was_matched_np[non_zero]

        if diff_cul.shape[0] > 0:
            diff_cul2, idxs = np.unique(diff_cul, axis=0, return_index=True)
            emit_cul2        = emit_cul[idxs]
            was_matched_cul2 = was_matched_cul[idxs]
        else:
            diff_cul2, emit_cul2, was_matched_cul2 = diff_cul, emit_cul, was_matched_cul

        counts = np.count_nonzero(choices_np, axis=0).astype(np.uint64)

        return diff_cul2, emit_cul2, was_matched_cul2, choices_np, counts

    def assign_candidates(self, diff: np.ndarray, emit: np.ndarray, was_matched: np.ndarray) -> tuple:
        """
        Process diffs into candidate cases without committing to the table.
        Returns (neg_case, new_cases, new_cases_emit, new_group_cases) for
        downstream NNv2 processing or direct commit.
        """
        if diff.shape[0] == 0:
            empty = np.zeros((0, 2, self.bool_dim), dtype=np.uint8)
            empty_emit = np.zeros((0, 2, self.emit_words), dtype=np.uint64)
            neg_case = np.ones(self.bool_dim, dtype=np.uint8)
            return neg_case, empty, empty_emit, empty

        neg_case = np.bitwise_and.reduce(diff[:, 1], axis=0)

        if self.array_used > 0:
            diff2 = diff[..., None] ^ self.match[None, ..., :self.array_used]
            diffed_neg_mask = diff2[:, 0].any(axis=1).all(axis=0)
            new_case_mask   = diff2[:, 0].any(axis=1).all(axis=1)
            neg_cols = np.where(~diffed_neg_mask)[0]
            if neg_cols.size > 0:
                self.match[1, np.arange(self.bool_dim)[:, None], neg_cols[None, :]] &= neg_case[:, None]
        else:
            new_case_mask = np.ones(diff.shape[0], dtype=bool)

        new_cases      = diff[new_case_mask & was_matched]
        new_cases_emit = emit[new_case_mask & was_matched]
        new_group_cases = diff[new_case_mask & ~was_matched]

        return neg_case, new_cases, new_cases_emit, new_group_cases

    def commit_candidates(self, neg_case: np.ndarray, new_cases: np.ndarray,
                          new_cases_emit: np.ndarray, new_group_cases: np.ndarray):
        """Directly commit candidates to the table — NNv5 path."""
        if new_cases.shape[0] > 0:
            n = new_cases.shape[0]
            assert self.array_used + n <= self.case_capacity, "Case capacity exceeded"
            sl = slice(self.array_used, self.array_used + n)
            self.match[:, :, sl] = np.permute_dims(new_cases, (1, 2, 0))
            self.match[1, :, sl] &= neg_case[..., None]
            emit_shift = np.permute_dims(new_cases_emit, (1, 0, 2))
            self.emit[0, sl] |= emit_shift[0]
            self.emit[1, sl] &= emit_shift[1]
            self.array_used += n

        if new_group_cases.shape[0] > 0:
            assert self.array_used + 1 <= self.case_capacity, "Case capacity exceeded"
            assert self.emit_used < self.group_capacity, "Group capacity exceeded"

            gidx = self.emit_used
            self.emit_used += 1

            self.match[:, :, self.array_used] = np.permute_dims(new_group_cases[0], (0, 1))
            self.match[1, :, self.array_used] &= neg_case
            self._set_emit_group_bit(self.array_used, gidx, is_member=True)

            if gidx > 0:
                full_words, rem_bits = divmod(gidx, 64)
                self.emit[1, self.array_used, :full_words] = np.iinfo(np.uint64).max
                if rem_bits:
                    self.emit[1, self.array_used, full_words] |= (np.uint64(1) << np.uint64(rem_bits)) - np.uint64(1)

            self._set_emit_group_bit(slice(None, self.array_used), gidx, is_member=False)
            self.array_used += 1

    def assign(self, diff: np.ndarray, emit: np.ndarray, was_matched: np.ndarray):
        """Full assign — candidates + direct commit. Used when NNv2 is not active."""
        neg_case, new_cases, new_cases_emit, new_group_cases = self.assign_candidates(
            diff, emit, was_matched
        )
        self.commit_candidates(neg_case, new_cases, new_cases_emit, new_group_cases)

    def activation_density(self, choices: np.ndarray) -> np.ndarray:
        """
        Sum of case activations per input — the Inc aggregation.
        choices: (N, C) bool
        returns: (N,) float — activation density per input
        """
        return choices.sum(axis=1).astype(np.float32)

    def group_affinities(self, choices: np.ndarray) -> np.ndarray:
        """
        Unpack group membership bits into per-input group affinity scores.
        choices: (N, C) bool
        returns: (N, emit_used) float
        """
        if self.emit_used == 0 or self.array_used == 0:
            return np.zeros((choices.shape[0], 0), dtype=np.float32)

        emit_pos = self.emit[0, :self.array_used]               # (C, W)
        affinities = np.zeros((choices.shape[0], self.emit_used), dtype=np.float32)

        for g in range(self.emit_used):
            word, bit = divmod(g, 64)
            flag = np.uint64(1) << np.uint64(bit)
            member_cases = ((emit_pos[:, word] & flag) != 0)    # (C,) bool
            # affinity = fraction of member cases that fired per input
            n_member = member_cases.sum()
            if n_member > 0:
                affinities[:, g] = choices[:, member_cases].sum(axis=1) / n_member

        return affinities

    def synthesize(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Generate n synthetic bool vectors by combining case patterns.

        For each synthetic example:
            - Sample a group affinity target (random weighted combination)
            - Find cases belonging to sampled groups
            - OR their positive match patterns together
            - Apply negative constraints (zero out negative bits)

        returns: (n, bool_dim) uint8
        """
        if self.array_used == 0 or self.emit_used == 0:
            return rng.integers(0, 2, size=(n, self.bool_dim), dtype=np.uint8)

        results = np.zeros((n, self.bool_dim), dtype=np.uint8)

        for i in range(n):
            # Sample a random subset of groups to target
            n_groups = rng.integers(1, max(2, self.emit_used))
            target_groups = rng.choice(self.emit_used, size=n_groups, replace=False)

            # Find cases that are members of any target group
            emit_pos = self.emit[0, :self.array_used]
            emit_neg = self.emit[1, :self.array_used]
            member_mask = np.zeros(self.array_used, dtype=bool)

            for g in target_groups:
                word, bit = divmod(g, 64)
                flag = np.uint64(1) << np.uint64(bit)
                member_mask |= ((emit_pos[:, word] & flag) != 0)

            if not member_mask.any():
                # Fall back to random if no cases found
                results[i] = rng.integers(0, 2, size=self.bool_dim, dtype=np.uint8)
                continue

            # OR positive match patterns of member cases
            pos_patterns = self.match[0, :, :self.array_used][:, member_mask]  # (D, K)
            synth = np.bitwise_or.reduce(pos_patterns, axis=1)                 # (D,)

            # Apply negative constraints: zero bits required to be absent
            neg_patterns = self.match[1, :, :self.array_used][:, member_mask]
            neg_union = np.bitwise_or.reduce(neg_patterns, axis=1)
            synth &= ~neg_union

            results[i] = synth

        return results

    def synthesize_from(self, bool_vecs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Generate synthetic bool vectors anchored to real input positions.
        Vectorized — computes group memberships and case masks in batch.

        bool_vecs: (N, bool_dim) uint8
        returns:   (N, bool_dim) uint8 synthetic vectors
        """
        if self.array_used == 0 or self.emit_used == 0:
            return bool_vecs.copy()

        N = bool_vecs.shape[0]
        C = self.array_used
        G = self.emit_used
        emit_pos = self.emit[0, :C]   # (C, W)

        # Run forward to get choices for all inputs at once
        _, _, _, choices, _ = self.forward(bool_vecs)  # (N, C)

        # Unpack group membership bits for all cases — (C, G) bool
        case_group_pos = np.zeros((C, G), dtype=bool)
        for g in range(G):
            word, bit = divmod(g, 64)
            flag = np.uint64(1) << np.uint64(bit)
            case_group_pos[:, g] = (emit_pos[:, word] & flag) != 0

        # For each input: which groups does it belong to via matched cases?
        # choices: (N, C), case_group_pos: (C, G) → input_groups: (N, G)
        input_groups = choices @ case_group_pos  # (N, G) — count of matched cases per group

        # For each input: which cases are reachable via its groups?
        # input_groups > 0: (N, G), case_group_pos: (C, G) → reachable: (N, C)
        reachable = (input_groups > 0) @ case_group_pos.T  # (N, C)

        # Build synthetic vectors
        results = bool_vecs.copy()
        pos_match = self.match[0, :, :C]  # (D, C)
        neg_match = self.match[1, :, :C]  # (D, C)

        for i in range(N):
            if not choices[i].any():
                continue  # unmatched — keep original

            member_idxs = np.where(reachable[i])[0]
            if len(member_idxs) == 0:
                continue

            # Random subset of reachable cases
            n_sample = max(1, rng.integers(1, len(member_idxs) + 1))
            sampled  = rng.choice(member_idxs, size=n_sample, replace=False)

            synth = bool_vecs[i].copy()
            synth |= np.bitwise_or.reduce(pos_match[:, sampled], axis=1)
            synth &= ~np.bitwise_or.reduce(neg_match[:, sampled], axis=1)
            results[i] = synth

        return results

    def stats(self) -> dict:
        return {
            "cases_used":  self.array_used,
            "groups_used": self.emit_used,
            "case_capacity":  self.case_capacity,
            "group_capacity": self.group_capacity,
        }


# ---------------------------------------------------------------------------
# NNv2 Block — emit-first compression layer
# ---------------------------------------------------------------------------

class NNv2Block:
    """
    Compression layer operating on NNv5's case structure.

    NNv2 operates emit-first, match-second — it takes NNv5's group membership
    bitsets as inputs and produces a compressed case table bounded by N*(N-1)
    where N is the group count.

    Three operations:
        Reduce  — merge cases sharing identical group membership
        Split   — divide a case covering multiple groups into subcases
                  (only when each subcase can satisfy the lower bound:
                   at least one positive AND one negative group membership)
        Append  — add new cases not covered by existing structure

    NNv2 activates only once NNv5 has formed at least one group (emit_used > 0).
    After activation, NNv5's assign_candidates output is routed through NNv2
    instead of being committed directly.
    """
    def __init__(self, bool_dim: int, group_capacity: int, case_capacity: int):
        self.bool_dim       = bool_dim
        self.group_capacity = group_capacity
        self.case_capacity  = case_capacity
        self.emit_words     = (group_capacity + 63) // 64

        self.array_used = 0

        # match: (2, bool_dim, case_capacity) uint8
        self.match = np.zeros((2, bool_dim, case_capacity), dtype=np.uint8)
        # emit: (2, case_capacity, emit_words) uint64 — group membership bitsets
        self.emit  = np.zeros((2, case_capacity, self.emit_words), dtype=np.uint64)
        # float_emit: (case_capacity, group_capacity) float32
        # Per-case affinity weight per group — used for weighted synthesis.
        # Initialised to 1.0 for positive member groups, 0.0 otherwise.
        # On Reduce: average the affinities of merged cases.
        # On Split: older subcase keeps its affinities, newer gets remainder
        #           (waterfall-with-half, respecting case age via insertion order).
        self.float_emit = np.zeros((case_capacity, group_capacity), dtype=np.float32)

    def _emit_pos(self, idx) -> np.ndarray:
        return self.emit[0, idx]

    def _emit_neg(self, idx) -> np.ndarray:
        return self.emit[1, idx]

    def _has_pos_and_neg(self, emit_pos_row: np.ndarray, emit_neg_row: np.ndarray) -> bool:
        """Check that a case has at least one positive and one negative group bit."""
        return emit_pos_row.any() and emit_neg_row.any()

    def _emit_rows_equal(self, a_pos, a_neg, b_pos, b_neg) -> bool:
        return np.array_equal(a_pos, b_pos) and np.array_equal(a_neg, b_neg)

    def _init_float_emit(self, case_idx: int, emit_pos: np.ndarray):
        """Initialise float affinities for a new case — 1.0 for positive member groups."""
        for g in range(self.group_capacity):
            word, bit = divmod(g, 64)
            flag = np.uint64(1) << np.uint64(bit)
            if (emit_pos[word] & flag) != 0:
                self.float_emit[case_idx, g] = 1.0

    def _init_float_emit_to(self, target: np.ndarray, emit_pos: np.ndarray):
        """Write float affinities into a pre-allocated array."""
        target[:] = 0.0
        for g in range(self.group_capacity):
            word, bit = divmod(g, 64)
            flag = np.uint64(1) << np.uint64(bit)
            if (emit_pos[word] & flag) != 0:
                target[g] = 1.0

    def _group_affinity(self, case_idx: int) -> np.ndarray:
        """Return float affinity vector for a case. (group_capacity,)"""
        return self.float_emit[case_idx]

    def _waterfall_split(self, parent_affinity: np.ndarray,
                         sub_a_pos: np.ndarray, sub_b_pos: np.ndarray) -> tuple:
        """
        Split parent float affinities between two subcases using waterfall logic.

        Older subcase (A, kept in place) preserves its share.
        Newer subcase (B, appended) receives the remainder.

        Each subcase only inherits affinity for its own positive groups.
        If a group belongs to both subcases it's split half+1 to A, rest to B.
        """
        a_aff = np.zeros(self.group_capacity, dtype=np.float32)
        b_aff = np.zeros(self.group_capacity, dtype=np.float32)

        for g in range(self.group_capacity):
            word, bit = divmod(g, 64)
            flag = np.uint64(1) << np.uint64(bit)
            in_a = bool((sub_a_pos[word] & flag) != 0)
            in_b = bool((sub_b_pos[word] & flag) != 0)
            v = parent_affinity[g]

            if in_a and in_b:
                # Shared group — half+1 to older (A), rest to newer (B)
                a_share = v / 2.0 + (v % 1.0 > 0) * 0.5  # approximate half+1
                a_aff[g] = min(v, a_share)
                b_aff[g] = max(0.0, v - a_aff[g])
            elif in_a:
                a_aff[g] = v
            elif in_b:
                b_aff[g] = v

        return a_aff, b_aff

    def compress(self, nnv5: "NNv5Block"):
        """Compress full NNv5 table — use compress_range for incremental updates."""
        self.compress_range(nnv5, 0, nnv5.array_used)

    def compress_range(self, nnv5: "NNv5Block", start: int, end: int):
        """
        Run NNv2 compression on a range of NNv5 cases [start, end).
        Called once per NNv5 chunk with only the newly added cases.
        """
        if nnv5.emit_used == 0 or start >= end:
            return

        for ci in range(start, end):
            src_match_pos = nnv5.match[0, :, ci]   # (bool_dim,)
            src_match_neg = nnv5.match[1, :, ci]
            src_emit_pos  = nnv5.emit[0, ci]        # (emit_words,)
            src_emit_neg  = nnv5.emit[1, ci]

            # --- REDUCE: find existing case with identical emit ---
            reduced = False
            for ei in range(self.array_used):
                if self._emit_rows_equal(
                    self._emit_pos(ei), self._emit_neg(ei),
                    src_emit_pos, src_emit_neg,
                ):
                    # Merge match patterns via AND (tighten constraints)
                    self.match[0, :, ei] &= src_match_pos
                    self.match[1, :, ei] &= src_match_neg
                    # Average float affinities of merged cases
                    src_aff = np.zeros(self.group_capacity, dtype=np.float32)
                    self._init_float_emit_to(src_aff, src_emit_pos)
                    self.float_emit[ei] = (self.float_emit[ei] + src_aff) * 0.5
                    reduced = True
                    break

            if reduced:
                continue

            # --- SPLIT: find existing case whose positive emit is superset ---
            split = False
            for ei in range(self.array_used):
                existing_pos = self._emit_pos(ei)
                existing_neg = self._emit_neg(ei)
                if not np.array_equal(existing_pos & src_emit_pos, src_emit_pos):
                    continue
                if np.array_equal(existing_pos, src_emit_pos):
                    continue  # identical — handled by reduce

                sub_a_pos = existing_pos & src_emit_pos
                sub_a_neg = existing_neg | src_emit_neg
                sub_b_pos = existing_pos & ~src_emit_pos
                sub_b_neg = existing_neg

                if (self._has_pos_and_neg(sub_a_pos, sub_a_neg) and
                        self._has_pos_and_neg(sub_b_pos, sub_b_neg)):
                    # Waterfall split of float affinities
                    parent_aff = self.float_emit[ei].copy()
                    a_aff, b_aff = self._waterfall_split(parent_aff, sub_a_pos, sub_b_pos)

                    # Replace existing with subcase A
                    self.emit[0, ei] = sub_a_pos
                    self.emit[1, ei] = sub_a_neg
                    self.match[0, :, ei] &= src_match_pos
                    self.match[1, :, ei] &= src_match_neg
                    self.float_emit[ei] = a_aff

                    # Append subcase B (newer — gets remainder)
                    if self.array_used < self.case_capacity:
                        bi = self.array_used
                        self.match[0, :, bi] = self.match[0, :, ei]
                        self.match[1, :, bi] = self.match[1, :, ei]
                        self.emit[0, bi] = sub_b_pos
                        self.emit[1, bi] = sub_b_neg
                        self.float_emit[bi] = b_aff
                        self.array_used += 1

                    split = True
                    break

            if split:
                continue

            # --- APPEND: new case ---
            if self.array_used < self.case_capacity:
                idx = self.array_used
                self.match[0, :, idx] = src_match_pos
                self.match[1, :, idx] = src_match_neg
                self.emit[0, idx] = src_emit_pos
                self.emit[1, idx] = src_emit_neg
                self._init_float_emit(idx, src_emit_pos)
                self.array_used += 1

    def forward(self, tokens: np.ndarray):
        """
        Same forward pass as NNv5 — subset match on bool arrays.
        Used when NNv2 takes over from NNv5 as primary case handler.

        tokens: (N, bool_dim) uint8
        returns: (diff, emit, was_matched, choices)
        """
        if self.array_used == 0:
            n = tokens.shape[0]
            tokens_inv = 1 - tokens
            inp = np.stack((tokens, tokens_inv), axis=1)
            inp2, idxs = np.unique(inp, axis=0, return_index=True)
            return (inp2,
                    np.zeros((inp2.shape[0], 2, self.emit_words), dtype=np.uint64),
                    np.zeros(inp2.shape[0], dtype=bool),
                    np.zeros((n, 0), dtype=bool))

        tokens_inv = 1 - tokens
        inp = np.stack((tokens, tokens_inv), axis=1)         # (N, 2, D)
        match_view = self.match[None, ..., :self.array_used] # (1, 2, D, C)
        selection  = inp[..., None] & match_view
        choices    = (selection == match_view).all((1, 2))   # (N, C)
        was_matched = choices.any(axis=1)

        emit_view  = self.emit[None, :, :self.array_used]
        emit_mask  = self.emit[:, :self.array_used].any(axis=(0, 2))
        emit_broad = np.broadcast_to(
            emit_view, (tokens.shape[0], 2, self.array_used, self.emit_words)
        )
        emit_out = np.bitwise_and.reduce(
            emit_broad, axis=2,
            where=choices[:, None, :, None] & emit_mask[None, None, :, None],
        )
        emit_out[~was_matched] = np.uint64(0)

        match_broad = np.broadcast_to(
            match_view, (tokens.shape[0], 2, self.bool_dim, self.array_used)
        )
        reverse = np.bitwise_or.reduce(
            match_broad, axis=3, where=choices[:, None, None, :]
        )
        diff = reverse & inp ^ inp
        non_zero = diff[:, 0].any(axis=1)
        diff_cul = diff[non_zero]
        emit_cul = emit_out[non_zero]
        was_matched_cul = was_matched[non_zero]

        if diff_cul.shape[0] > 0:
            diff_cul2, idxs = np.unique(diff_cul, axis=0, return_index=True)
            emit_cul2 = emit_cul[idxs]
            was_matched_cul2 = was_matched_cul[idxs]
        else:
            diff_cul2, emit_cul2, was_matched_cul2 = diff_cul, emit_cul, was_matched_cul

        return diff_cul2, emit_cul2, was_matched_cul2, choices

    def stats(self) -> dict:
        return {
            "nnv2_cases_used": self.array_used,
            "nnv2_case_capacity": self.case_capacity,
        }

    def sync_to_nnv5(self, nnv5: "NNv5Block"):
        """
        Replace NNv5's active case table with NNv2's compressed table.
        NNv5's forward pass then matches against NNv2's smaller case set,
        making subsequent forward passes cheaper.
        Groups and emit_used are preserved — only case arrays are replaced.
        """
        n = self.array_used
        if n == 0:
            return
        nnv5.match[:, :, :n] = self.match[:, :, :n]
        nnv5.emit[:, :n]     = self.emit[:, :n]
        nnv5.array_used      = n

    def synthesize_from(self, bool_vecs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Generate synthetic bool vectors anchored to real input positions.
        Uses float affinities for weighted case sampling during synthesis.
        """
        if self.array_used == 0:
            return bool_vecs.copy()

        N = bool_vecs.shape[0]
        results = np.zeros((N, self.bool_dim), dtype=np.uint8)
        emit_pos = self.emit[0, :self.array_used]

        _, _, _, choices = self.forward(bool_vecs)

        for i in range(N):
            if choices.shape[1] == 0 or not choices[i].any():
                results[i] = bool_vecs[i]
                continue

            matched_emit = emit_pos[choices[i]]
            member_groups = []
            for g in range(self.group_capacity):
                word, bit = divmod(g, 64)
                flag = np.uint64(1) << np.uint64(bit)
                if (matched_emit[:, word] & flag).any():
                    member_groups.append(g)

            if not member_groups:
                results[i] = bool_vecs[i]
                continue

            # Find cases in member groups — weighted by float affinity
            member_mask = np.zeros(self.array_used, dtype=bool)
            for g in member_groups:
                word, bit = divmod(g, 64)
                flag = np.uint64(1) << np.uint64(bit)
                member_mask |= ((emit_pos[:, word] & flag) != 0)

            member_idxs = np.where(member_mask)[0]

            # Weight by sum of float affinities across member groups
            weights = np.array([
                self.float_emit[idx, member_groups].sum()
                for idx in member_idxs
            ], dtype=np.float32)
            total = weights.sum()
            if total > 0:
                weights /= total
            else:
                weights = np.ones(len(member_idxs), dtype=np.float32) / len(member_idxs)

            n_sample = max(1, rng.integers(1, len(member_idxs) + 1))
            sampled = rng.choice(member_idxs, size=n_sample, replace=False, p=weights)

            synth = bool_vecs[i].copy()
            for idx in sampled:
                synth |= self.match[0, :, idx]

            neg_union = np.zeros(self.bool_dim, dtype=np.uint8)
            for idx in sampled:
                neg_union |= self.match[1, :, idx]
            synth &= ~neg_union

            results[i] = synth

        return results




from dataclasses import dataclass

@dataclass
class NNv5LayerConfig:
    """Per-layer configuration for the two-layer NNv5 pipeline."""
    chunk_dim:          int = 32    # bits per NNv5 chunk — must satisfy chunk_dim1 * chunk_dim2 == d_model
    case_capacity:      int = 2048
    group_capacity:     int = 128
    nnv5_chunk_size:    int = 16
    nnv2_case_capacity: int = 1024


class SyntheticBatchHarness:
    """
    Wraps a TinyPatchLM trunk to provide synthetic batch augmentation.

    Two-layer NNv5 pipeline (no learned projection):
        Layer 1: raw hidden states → threshold to bool → chunk by chunk_dim1
                 → NNv5+NNv2 per chunk → concatenated emit bitsets
        Layer 2: layer1 emit bits → chunk by chunk_dim2
                 → NNv5+NNv2 → final group affinities for synthesis

    Constraint: layer1.chunk_dim * layer2.chunk_dim == d_model

    Usage in training loop:
        harness = SyntheticBatchHarness(model, cfg, device)
        harness.attach()
        logits, loss = model(x, y)
        synth_loss = harness.synthetic_pass(harness.captured_hidden, y)
        total_loss = loss + synth_weight * synth_loss
        loss.backward()
        harness.detach()
    """
    def __init__(
        self,
        model,
        d_model: int,
        layer1: NNv5LayerConfig,
        layer2: NNv5LayerConfig,
        n_synthetic: int,
        match_threshold: float,
        device: torch.device,
        seed: int = 42,
        nnv5_update_steps: int = 100,
        emit_chunk_size: int = 64,
        synth_loss_weight: float = 0.5,
    ):
        # Validate chunk_dim constraint
        assert layer1.chunk_dim * layer2.chunk_dim == d_model, (
            f"layer1.chunk_dim ({layer1.chunk_dim}) * layer2.chunk_dim ({layer2.chunk_dim}) "
            f"must equal d_model ({d_model})"
        )

        self.model             = model
        self.d_model           = d_model
        self.n_synthetic       = n_synthetic
        self.match_threshold   = match_threshold
        self.device            = device
        self.nnv5_update_steps = nnv5_update_steps
        self.emit_chunk_size   = emit_chunk_size
        self.synth_loss_weight = synth_loss_weight
        self.rng               = np.random.default_rng(seed)

        self.layer1_cfg = layer1
        self.layer2_cfg = layer2

        # Layer 1 — operates on d_model // chunk_dim1 chunks of chunk_dim1 bits
        n_chunks1 = d_model // layer1.chunk_dim
        assert (layer1.group_capacity * n_chunks1) % layer2.chunk_dim == 0, (
            f"layer2.chunk_dim ({layer2.chunk_dim}) must divide "
            f"layer1 total emit bits ({layer1.group_capacity * n_chunks1})"
        )
        self.nnv5_l1 = [
            NNv5Block(layer1.chunk_dim, layer1.case_capacity, layer1.group_capacity)
            for _ in range(n_chunks1)
        ]
        self.nnv2_l1 = [
            NNv2Block(layer1.chunk_dim, layer1.group_capacity, layer1.nnv2_case_capacity)
            for _ in range(n_chunks1)
        ]

        # Layer 2 — flat chunks over layer1 total emit bits
        layer2_input_width = layer1.group_capacity * n_chunks1
        n_chunks2 = layer2_input_width // layer2.chunk_dim
        self.nnv5_l2 = [
            NNv5Block(layer2.chunk_dim, layer2.case_capacity, layer2.group_capacity)
            for _ in range(n_chunks2)
        ]
        self.nnv2_l2 = [
            NNv2Block(layer2.chunk_dim, layer2.group_capacity, layer2.nnv2_case_capacity)
            for _ in range(n_chunks2)
        ]

        self.n_chunks1         = n_chunks1
        self.n_chunks2         = n_chunks2
        self.layer2_input_width = layer2_input_width

        self.captured_hidden: torch.Tensor | None = None
        self._hook = None

    def attach(self):
        """Register forward hook to capture trunk hidden states."""
        def _hook_fn(module, inp, out):
            self.captured_hidden = inp[0].detach().float()
        self._hook = self.model.ln_f.register_forward_hook(_hook_fn)

    def detach(self):
        """Remove forward hook."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def _hidden_to_bool(self, hidden: torch.Tensor) -> np.ndarray:
        """
        Convert float hidden states to bool arrays by thresholding at 0.
        No learned projection — direct quantization.

        hidden: (N, d_model) float
        returns: (N, d_model) uint8
        """
        return (hidden > 0).cpu().numpy().astype(np.uint8)

    def _run_layer(
        self,
        bool_vecs: np.ndarray,
        nnv5_list: list,
        nnv2_list: list,
        chunk_dim: int,
        nnv5_chunk_size: int,
    ) -> np.ndarray:
        """
        Run one NNv5+NNv2 layer on chunked bool inputs.

        bool_vecs: (N, input_dim) uint8
        Splits input_dim into chunks of chunk_dim, runs NNv5+NNv2 on each,
        concatenates the emit bitsets from all chunks.

        returns: (N, n_chunks * emit_words * 64) uint8 — unpacked emit bits
        """
        N = bool_vecs.shape[0]
        input_dim = bool_vecs.shape[1]
        n_chunks  = input_dim // chunk_dim
        use_torch = self.device.type == "cuda"

        emit_parts = []

        for ci, (nnv5, nnv2) in enumerate(zip(nnv5_list, nnv2_list)):
            chunk_vecs = bool_vecs[:, ci * chunk_dim:(ci + 1) * chunk_dim]

            # Process in NNv5 input chunks
            for start in range(0, N, nnv5_chunk_size):
                cv = chunk_vecs[start:start + nnv5_chunk_size]
                cases_before = nnv5.array_used
                nnv2_active  = nnv5.emit_used > 0

                if use_torch:
                    diff, emit, was_matched, choices, _ = nnv5.forward_torch(cv, self.device, self.emit_chunk_size)
                else:
                    diff, emit, was_matched, choices, _ = nnv5.forward(cv)

                if nnv2_active:
                    neg_case, new_cases, new_cases_emit, new_group_cases = \
                        nnv5.assign_candidates(diff, emit, was_matched)
                    nnv5.commit_candidates(neg_case, new_cases, new_cases_emit, new_group_cases)
                else:
                    nnv5.assign(diff, emit, was_matched)

                cases_after = nnv5.array_used
                if nnv5.emit_used > 0 and cases_after > cases_before:
                    nnv2.compress_range(nnv5, cases_before, cases_after)
                    nnv2.sync_to_nnv5(nnv5)

            # Extract emit bits for this chunk — unpack uint64 to uint8 bits
            # Use final NNv5 forward pass to get per-input emit
            if use_torch:
                _, emit_out, _, _, _ = nnv5.forward_torch(chunk_vecs, self.device, self.emit_chunk_size)
            else:
                _, emit_out, _, _, _ = nnv5.forward(chunk_vecs)

            # Unpack emit bits: (N, 2, emit_words) → (N, group_capacity) uint8
            emit_bits = np.unpackbits(
                emit_out[:, 0].view(np.uint8),
                axis=1, bitorder='little'
            )[:, :nnv5.emit_used]                        # (N, emit_used)
            emit_parts.append(emit_bits)

        # Concatenate emit bits from all chunks
        if emit_parts:
            return np.concatenate(emit_parts, axis=1)    # (N, total_emit_bits)
        return np.zeros((N, 0), dtype=np.uint8)

    def update_case_table(self, hidden: torch.Tensor):
        """
        Run two-layer NNv5+NNv2 pipeline on the current batch's hidden states.

        Layer 1: raw bool chunks of d_model → per-position emit bitsets
        Layer 2: flat chunks of layer1 emit bits → higher-order group structure

        NNv5 naturally encodes positional information into group structure
        through subset matching — no explicit windowing needed.

        hidden: (B, T, D) float
        """
        flat = hidden.reshape(-1, self.d_model)
        bool_vecs = self._hidden_to_bool(flat)           # (N, d_model) uint8

        # Layer 1
        emit1 = self._run_layer(
            bool_vecs,
            self.nnv5_l1, self.nnv2_l1,
            self.layer1_cfg.chunk_dim,
            self.layer1_cfg.nnv5_chunk_size,
        )                                                # (N, total_emit_bits) uint8

        if emit1.shape[1] == 0:
            return

        # Pad to chunk_dim2 boundary
        pad = (-emit1.shape[1]) % self.layer2_cfg.chunk_dim
        if pad > 0:
            emit1 = np.pad(emit1, ((0, 0), (0, pad)))

        # Layer 2
        self._run_layer(
            emit1,
            self.nnv5_l2, self.nnv2_l2,
            self.layer2_cfg.chunk_dim,
            self.layer2_cfg.nnv5_chunk_size,
        )

    def synthetic_pass(
        self,
        real_hidden: torch.Tensor,
        targets: torch.Tensor,
        step: int = 0,
    ) -> torch.Tensor:
        """
        Generate synthetic hidden states and compute loss on them.

        real_hidden: (B, T, D) float
        targets:     (B, T)    int64
        step:        int       — current training step
        """
        # Only update case tables for first nnv5_update_steps steps
        if step <= self.nnv5_update_steps:
            self.update_case_table(real_hidden)
            if step == self.nnv5_update_steps:
                l1_cases  = sum(n.array_used for n in self.nnv5_l1)
                l1_groups = sum(n.emit_used  for n in self.nnv5_l1)
                l2_cases  = sum(n.array_used for n in self.nnv5_l2)
                l2_groups = sum(n.emit_used  for n in self.nnv5_l2)
                print(f"  NNv5 tables frozen at step {step} — "
                      f"L1 cases={l1_cases} groups={l1_groups} | "
                      f"L2 cases={l2_cases} groups={l2_groups}")

        # Check layer2 has any structure to synthesize from
        if not any(n.array_used > 0 for n in self.nnv5_l2):
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        B, T, D = real_hidden.shape
        flat      = real_hidden.reshape(-1, D)
        bool_vecs = self._hidden_to_bool(flat)           # (N, d_model)

        # Get layer1 emit for real inputs
        emit1 = self._run_layer(
            bool_vecs,
            self.nnv5_l1, self.nnv2_l1,
            self.layer1_cfg.chunk_dim,
            self.layer1_cfg.nnv5_chunk_size,
        )

        if emit1.shape[1] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        pad = (-emit1.shape[1]) % self.layer2_cfg.chunk_dim
        if pad > 0:
            emit1 = np.pad(emit1, ((0, 0), (0, pad)))

        # Synthesize from layer2 — flat chunks over layer1 emit bits
        # TODO: future — respect positional ordering implicit in NNv5 group structure
        N          = bool_vecs.shape[0]
        chunk_dim2 = self.layer2_cfg.chunk_dim
        synth_emit1 = emit1.copy()

        for ci, (nnv5, nnv2) in enumerate(zip(self.nnv5_l2, self.nnv2_l2)):
            chunk_vecs = emit1[:, ci * chunk_dim2:(ci + 1) * chunk_dim2]
            if nnv2.array_used > 0:
                synth_chunk = nnv2.synthesize_from(chunk_vecs, self.rng)
            elif nnv5.array_used > 0:
                synth_chunk = nnv5.synthesize_from(chunk_vecs, self.rng)
            else:
                synth_chunk = chunk_vecs.copy()
            synth_emit1[:, ci * chunk_dim2:(ci + 1) * chunk_dim2] = synth_chunk

        # Project synthetic emit1 back to hidden states via layer1 inverse
        # Use synthesize_from on layer1 chunks
        synth_bool = np.zeros((N, self.d_model), dtype=np.uint8)
        chunk_dim1 = self.layer1_cfg.chunk_dim

        for ci, (nnv5, nnv2) in enumerate(zip(self.nnv5_l1, self.nnv2_l1)):
            # Use the emit bits from synthetic emit1 to select cases
            emit_chunk = synth_emit1[:, ci * self.layer1_cfg.group_capacity:
                                        (ci + 1) * self.layer1_cfg.group_capacity]
            real_chunk = bool_vecs[:, ci * chunk_dim1:(ci + 1) * chunk_dim1]

            if nnv2.array_used > 0:
                synth_chunk = nnv2.synthesize_from(real_chunk, self.rng)
            elif nnv5.array_used > 0:
                synth_chunk = nnv5.synthesize_from(real_chunk, self.rng)
            else:
                synth_chunk = real_chunk.copy()

            synth_bool[:, ci * chunk_dim1:(ci + 1) * chunk_dim1] = synth_chunk

        # Convert synthetic bool back to float hidden states
        # Scale to match real hidden state range
        synth_hidden = torch.from_numpy(synth_bool.astype(np.float32)).to(self.device)
        # Rescale: real hidden states have non-unit variance, bool is 0/1
        # Match mean and std of real hidden states
        real_flat   = flat.float()
        real_mean   = real_flat.mean(dim=0, keepdim=True)
        real_std    = real_flat.std(dim=0, keepdim=True) + 1e-8
        synth_hidden = (synth_hidden - synth_hidden.mean(dim=0, keepdim=True)) / \
                       (synth_hidden.std(dim=0, keepdim=True) + 1e-8)
        synth_hidden = synth_hidden * real_std + real_mean
        synth_hidden = synth_hidden.reshape(B, T, D)

        _, synth_loss = self.model.forward_from_hidden(synth_hidden, targets)
        return synth_loss

    def stats(self) -> dict:
        l1_cases  = sum(n.array_used for n in self.nnv5_l1)
        l1_groups = sum(n.emit_used  for n in self.nnv5_l1)
        l2_cases  = sum(n.array_used for n in self.nnv5_l2)
        l2_groups = sum(n.emit_used  for n in self.nnv5_l2)
        nnv2_l1   = sum(n.array_used for n in self.nnv2_l1)
        nnv2_l2   = sum(n.array_used for n in self.nnv2_l2)
        return {
            "l1_cases": l1_cases, "l1_groups": l1_groups,
            "l2_cases": l2_cases, "l2_groups": l2_groups,
            "nnv2_l1_cases": nnv2_l1, "nnv2_l2_cases": nnv2_l2,
        }
