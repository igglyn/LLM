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
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# NNv5 wrapper
# ---------------------------------------------------------------------------

class NNv5Block:
    """
    NNv5 U1XToU1X logic operating on packed uint64 arrays.

    bool_dim specifies the number of bits. Internally bits are packed into
    ceil(bool_dim / 64) uint64 words per case. Inversion uses bitwise NOT
    which correctly flips all bits simultaneously.

    The case table persists across calls.
    """
    def __init__(self, bool_dim: int, case_capacity: int, group_capacity: int):
        self.bool_dim       = bool_dim
        self.case_capacity  = case_capacity
        self.group_capacity = group_capacity

        self.array_used = 0
        self.emit_used  = 0
        self.match_words = (bool_dim + 63) // 64
        self.emit_words  = (group_capacity + 63) // 64

        # match: (2, match_words, case_capacity) uint64
        # plane 0 = positive (required bits)
        # plane 1 = negative (forbidden bits)
        self.match = np.zeros((2, self.match_words, case_capacity), dtype=np.uint64)

        # emit: (2, case_capacity, emit_words) uint64
        self.emit = np.zeros((2, case_capacity, self.emit_words), dtype=np.uint64)

    def _set_emit_group_bit(self, case_idx, group_idx: int, *, is_member: bool):
        assert 0 <= group_idx < self.group_capacity
        word, bit = divmod(group_idx, 64)
        flag = np.uint64(1) << np.uint64(bit)
        plane = 0 if is_member else 1
        self.emit[plane, case_idx, word] |= flag

    def forward(self, tokens: np.ndarray):
        """
        tokens: (N, match_words) uint64 — packed bit representation
        returns: (diff, emit, was_matched, choices, activation_counts)
        """
        if self.array_used == 0:
            n = tokens.shape[0]
            tokens_inv = ~tokens.copy()
            inp = np.stack((tokens, tokens_inv), axis=1)       # (N, 2, match_words)
            inp2, idxs = np.unique(inp, axis=0, return_index=True)
            diff        = inp2
            emit        = np.zeros((inp2.shape[0], 2, self.emit_words), dtype=np.uint64)
            was_matched = np.zeros(inp2.shape[0], dtype=bool)
            choices     = np.zeros((n, 0), dtype=bool)
            counts      = np.zeros(0, dtype=np.uint64)
            return diff, emit, was_matched, choices, counts

        tokens_inv = ~tokens.copy()                                 # (N, match_words)
        inp = np.stack((tokens, tokens_inv), axis=1)                # (N, 2, match_words)

        match_view = self.match[None, ..., :self.array_used]        # (1, 2, W, C)
        inp_exp    = inp[..., None]                                  # (N, 2, W, 1)

        selection = inp_exp & match_view                            # (N, 2, W, C)
        choices   = (selection == match_view).all((1, 2))           # (N, C)

        was_matched = choices.any(axis=1)                           # (N,)

        emit_view = self.emit[None, :, :self.array_used]            # (1, 2, C, EW)
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

        match_broad = np.broadcast_to(
            match_view,
            (tokens.shape[0], 2, self.match_words, self.array_used),
        )
        reverse = np.bitwise_or.reduce(
            match_broad, axis=3,
            where=choices[:, None, None, :],
        )

        diff = reverse & inp ^ inp

        non_zero = diff[:, 0].any(axis=1)
        diff_cul = diff[non_zero]
        emit_cul = emit_out[non_zero]
        was_matched_cul = was_matched[non_zero]

        if diff_cul.shape[0] > 0:
            diff_cul2, idxs = np.unique(diff_cul, axis=0, return_index=True)
            emit_cul2        = emit_cul[idxs]
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

        tokens: (N, match_words) uint64 numpy array
        """
        if self.array_used == 0:
            n = tokens.shape[0]
            tokens_inv = ~tokens.copy()
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

        # Reinterpret uint64 as int64 for torch — bitwise ops work identically
        t_match = torch.from_numpy(
            self.match[:, :, :C].view(np.int64)
        ).to(device)                                                           # (2, W, C) int64
        t_emit  = torch.from_numpy(
            self.emit[:, :C].view(np.int64)
        ).to(device)                                                           # (2, C, EW) int64

        # Bitwise NOT for inversion — same as ~tokens on uint64
        tokens_np  = tokens.view(np.int64)
        t_inp_pos  = torch.from_numpy(tokens_np.copy()).to(device)            # (N, W)
        t_inp_neg  = ~t_inp_pos                                                # (N, W)
        t_inp      = torch.stack((t_inp_pos, t_inp_neg), dim=1)               # (N, 2, W)

        # --- FORWARD: subset match ---
        inp_exp    = t_inp.unsqueeze(-1)                                       # (N, 2, W, 1)
        match_view = t_match.unsqueeze(0)                                      # (1, 2, W, C)
        selection  = inp_exp & match_view                                      # (N, 2, W, C)
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
        cho_exp  = choices_t.unsqueeze(1).unsqueeze(1)                        # (N, 1, 1, C)
        mat_exp  = t_match.unsqueeze(0)                                        # (1, 2, W, C)
        contrib  = torch.where(
            cho_exp.expand(N, 2, self.match_words, C),
            mat_exp.expand(N, 2, self.match_words, C),
            torch.zeros(1, dtype=torch.int64, device=device).expand(N, 2, self.match_words, C),
        )
        reverse_t = contrib.any(dim=3)                                         # (N, 2, W) — wrong, need OR not any
        # Use bitwise OR reduction over cases dimension
        reverse_t = functools.reduce(
            torch.bitwise_or,
            [torch.where(choices_t[:, ci:ci+1].unsqueeze(2),
                         t_match[:, :, ci:ci+1].permute(2, 0, 1).expand(N, 2, self.match_words),
                         torch.zeros(N, 2, self.match_words, dtype=torch.int64, device=device))
             for ci in range(C)]
        ) if C > 0 else torch.zeros(N, 2, self.match_words, dtype=torch.int64, device=device)

        diff_t = reverse_t & t_inp ^ t_inp

        # Back to numpy — reinterpret int64 as uint64
        choices_np     = choices_t.cpu().numpy()
        diff_np        = diff_t.cpu().numpy().view(np.uint64)
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
        diff: (K, 2, match_words) uint64
        """
        if diff.shape[0] == 0:
            empty      = np.zeros((0, 2, self.match_words), dtype=np.uint64)
            empty_emit = np.zeros((0, 2, self.emit_words), dtype=np.uint64)
            neg_case   = np.full(self.match_words, np.iinfo(np.uint64).max, dtype=np.uint64)
            return neg_case, empty, empty_emit, empty

        neg_case = np.bitwise_and.reduce(diff[:, 1], axis=0)

        if self.array_used > 0:
            diff2 = diff[..., None] ^ self.match[None, ..., :self.array_used]
            diffed_neg_mask = diff2[:, 0].any(axis=1).all(axis=0)
            new_case_mask   = diff2[:, 0].any(axis=1).all(axis=1)
            neg_cols = np.where(~diffed_neg_mask)[0]
            if neg_cols.size > 0:
                self.match[1, np.arange(self.match_words)[:, None], neg_cols[None, :]] &= neg_case[:, None]
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

    def reverse(self, emit: np.ndarray) -> np.ndarray:
        """
        Reconstruct inputs from emit group membership vectors.

        Given per-input emit bitsets, find all cases belonging to those groups
        and OR their positive match patterns to produce reconstructed inputs.

        emit:    (N, 2, emit_words) uint64
        returns: (N, match_words) uint64 — reconstructed positive pattern
        """
        if self.array_used == 0:
            return np.zeros((emit.shape[0], self.match_words), dtype=np.uint64)

        N = emit.shape[0]
        C = self.array_used

        emit_pos_input = emit[:, 0]              # (N, emit_words)
        emit_pos_cases = self.emit[0, :C]        # (C, emit_words)

        # member[n, c] = any group bit overlap between input n and case c
        overlap = (emit_pos_input[:, None, :] & emit_pos_cases[None, :, :]).any(axis=2)  # (N, C)

        match_pos = self.match[0, :, :C]         # (match_words, C)
        # Vectorized OR: mask match cols by overlap, reduce over C
        # overlap: (N, C), match_pos: (match_words, C) → need (N, C, match_words)
        match_pos_t = match_pos.T                # (C, match_words)
        masked = np.where(
            overlap[:, :, None],                 # (N, C, 1)
            match_pos_t[None, :, :],             # (1, C, match_words)
            np.uint64(0)
        )                                        # (N, C, match_words)
        result = np.bitwise_or.reduce(masked, axis=1)  # (N, match_words)

        return result

    def synthesize_from(self, packed_vecs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Generate synthetic packed uint64 vectors anchored to real input positions.

        packed_vecs: (N, match_words) uint64
        returns:     (N, match_words) uint64
        """
        if self.array_used == 0 or self.emit_used == 0:
            return packed_vecs.copy()

        N = packed_vecs.shape[0]
        C = self.array_used
        G = self.emit_used
        emit_pos = self.emit[0, :C]   # (C, emit_words)

        _, _, _, choices, _ = self.forward(packed_vecs)  # (N, C)

        # Unpack group membership bits for all cases — (C, G) bool
        case_group_pos = np.zeros((C, G), dtype=bool)
        for g in range(G):
            word, bit = divmod(g, 64)
            flag = np.uint64(1) << np.uint64(bit)
            case_group_pos[:, g] = (emit_pos[:, word] & flag) != 0

        input_groups = choices @ case_group_pos   # (N, G)
        reachable    = (input_groups > 0) @ case_group_pos.T  # (N, C)

        results   = packed_vecs.copy()
        pos_match = self.match[0, :, :C]  # (match_words, C)
        neg_match = self.match[1, :, :C]

        for i in range(N):
            if not choices[i].any():
                continue

            member_idxs = np.where(reachable[i])[0]
            if len(member_idxs) == 0:
                continue

            n_sample = max(1, rng.integers(1, len(member_idxs) + 1))
            sampled  = rng.choice(member_idxs, size=n_sample, replace=False)

            synth = packed_vecs[i].copy()
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
        self.bool_dim       = bool_dim   # bits
        self.match_words    = (bool_dim + 63) // 64
        self.group_capacity = group_capacity
        self.case_capacity  = case_capacity
        self.emit_words     = (group_capacity + 63) // 64

        self.array_used = 0

        # match: (2, match_words, case_capacity) uint64
        self.match = np.zeros((2, self.match_words, case_capacity), dtype=np.uint64)
        # emit: (2, case_capacity, emit_words) uint64
        self.emit  = np.zeros((2, case_capacity, self.emit_words), dtype=np.uint64)
        # float_emit: (case_capacity, group_capacity) float32
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

        tokens: (N, match_words) uint64
        returns: (diff, emit, was_matched, choices)
        """
        if self.array_used == 0:
            n = tokens.shape[0]
            tokens_inv = ~tokens.copy()
            inp = np.stack((tokens, tokens_inv), axis=1)
            inp2, idxs = np.unique(inp, axis=0, return_index=True)
            return (inp2,
                    np.zeros((inp2.shape[0], 2, self.emit_words), dtype=np.uint64),
                    np.zeros(inp2.shape[0], dtype=bool),
                    np.zeros((n, 0), dtype=bool))

        tokens_inv = ~tokens.copy()
        inp = np.stack((tokens, tokens_inv), axis=1)         # (N, 2, W)
        match_view = self.match[None, ..., :self.array_used] # (1, 2, W, C)
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
            match_view, (tokens.shape[0], 2, self.match_words, self.array_used)
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
        # Sync emit_used so NNv5 doesn't allocate new groups beyond NNv2's compressed count
        max_group = 0
        for ci in range(n):
            for g in range(self.group_capacity):
                word, bit = divmod(g, 64)
                flag = np.uint64(1) << np.uint64(bit)
                if (self.emit[0, ci, word] & flag) != 0:
                    max_group = max(max_group, g + 1)
        nnv5.emit_used = max_group
        # Derive actual group count from NNv2's emit positive bits
        if n > 0:
            emit_pos = self.emit[0, :n]  # (n, emit_words)
            max_group = 0
            for g in range(nnv5.group_capacity):
                word, bit = divmod(g, 64)
                flag = np.uint64(1) << np.uint64(bit)
                if (emit_pos[:, word] & flag).any():
                    max_group = g + 1
            nnv5.emit_used = max_group

    def synthesize_from(self, packed_vecs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Generate synthetic packed uint64 vectors.
        Uses float affinities for weighted case sampling.

        packed_vecs: (N, match_words) uint64
        returns:     (N, match_words) uint64
        """
        if self.array_used == 0:
            return packed_vecs.copy()

        N        = packed_vecs.shape[0]
        results  = packed_vecs.copy()
        emit_pos = self.emit[0, :self.array_used]

        _, _, _, choices = self.forward(packed_vecs)

        for i in range(N):
            if choices.shape[1] == 0 or not choices[i].any():
                continue

            matched_emit = emit_pos[choices[i]]
            member_groups = []
            for g in range(self.group_capacity):
                word, bit = divmod(g, 64)
                flag = np.uint64(1) << np.uint64(bit)
                if (matched_emit[:, word] & flag).any():
                    member_groups.append(g)

            if not member_groups:
                continue

            member_mask = np.zeros(self.array_used, dtype=bool)
            for g in member_groups:
                word, bit = divmod(g, 64)
                flag = np.uint64(1) << np.uint64(bit)
                member_mask |= ((emit_pos[:, word] & flag) != 0)

            member_idxs = np.where(member_mask)[0]

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
            sampled  = rng.choice(member_idxs, size=n_sample, replace=False, p=weights)

            synth = packed_vecs[i].copy()
            for idx in sampled:
                synth |= self.match[0, :, idx]
            neg_union = np.zeros(self.match_words, dtype=np.uint64)
            for idx in sampled:
                neg_union |= self.match[1, :, idx]
            synth &= ~neg_union
            results[i] = synth

        return results




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
        self.synth_loss_weight = synth_loss_weight
        self.rng               = np.random.default_rng(seed)

        self.layer1_cfg = layer1
        self.layer2_cfg = layer2

        # Derived dimensions
        n_chunks1 = d_model // layer1.chunk_dim
        assert (layer1.group_capacity * n_chunks1) % layer2.chunk_dim == 0, (
            f"layer2.chunk_dim ({layer2.chunk_dim}) must divide "
            f"layer1 total emit bits ({layer1.group_capacity * n_chunks1})"
        )
        layer2_input_width = layer1.group_capacity * n_chunks1
        n_chunks2          = layer2_input_width // layer2.chunk_dim

        self.n_chunks1          = n_chunks1
        self.n_chunks2          = n_chunks2
        self.layer2_input_width = layer2_input_width

        # Single NNv5+NNv2 instance per layer — all chunks batched together (for assign)
        self.nnv5_l1 = NNv5Block(layer1.chunk_dim, layer1.case_capacity, layer1.group_capacity)
        self.nnv2_l1 = NNv2Block(layer1.chunk_dim, layer1.group_capacity, layer1.nnv2_case_capacity)
        self.nnv5_l2 = NNv5Block(layer2.chunk_dim, layer2.case_capacity, layer2.group_capacity)
        self.nnv2_l2 = NNv2Block(layer2.chunk_dim, layer2.group_capacity, layer2.nnv2_case_capacity)


        self.captured_hidden: torch.Tensor | None = None
        self._hook = None

        # Persistent input buffers — accumulate across LLM batches
        # flushed to NNv5 when they reach nnv5_chunk_size
        l1_match_words = (layer1.chunk_dim + 63) // 64
        l2_match_words = (layer2.chunk_dim + 63) // 64
        self._buf_l1 = np.empty((0, l1_match_words), dtype=np.uint64)
        self._buf_l2 = np.empty((0, l2_match_words), dtype=np.uint64)

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
        Convert float hidden states to packed uint64 bool arrays.
        Threshold at 0 — positive activations become 1 bits.

        hidden: (N, d_model) float
        returns: (N, match_words) uint64  where match_words = ceil(d_model / 64)
        """
        N      = hidden.shape[0]
        bits   = (hidden > 0).cpu().numpy().astype(bool)   # (N, d_model)
        n_bits = bits.shape[1]
        words  = (n_bits + 63) // 64
        # Pad to multiple of 64
        if n_bits % 64 != 0:
            pad  = 64 - (n_bits % 64)
            bits = np.pad(bits, ((0, 0), (0, pad)))
        # Pack 64 bits per uint64 word — LSB first
        packed = np.packbits(bits, axis=1, bitorder='little')  # (N, words*8) uint8
        return packed.view(np.uint64)                           # (N, words) uint64

    def _run_layer(
        self,
        packed_vecs: np.ndarray,
        nnv5: NNv5Block,
        nnv2: NNv2Block,
        chunk_dim: int,
        nnv5_chunk_size: int,
        buf: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run one NNv5+NNv2 layer on packed uint64 inputs with a persistent buffer.

        packed_vecs:   (N, chunk_words) uint64  — chunked inputs (N = positions * n_chunks)
        """
        use_torch = self.device.type == "cuda"
        M         = packed_vecs.shape[0]

        # Append to buffer
        buf = np.concatenate([buf, packed_vecs], axis=0) if buf.shape[0] > 0 else packed_vecs.copy()

        # Flush complete chunks from buffer
        while buf.shape[0] >= nnv5_chunk_size:
            cv           = buf[:nnv5_chunk_size]
            buf          = buf[nnv5_chunk_size:]
            cases_before = nnv5.array_used
            nnv2_active  = nnv5.emit_used > 0

            if use_torch:
                diff, emit, was_matched, choices, _ = nnv5.forward_torch(cv, self.device)
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

        # Emit extraction pass on current inputs only
        if use_torch:
            _, _, _, choices_full, _ = nnv5.forward_torch(packed_vecs, self.device)
        else:
            _, _, _, choices_full, _ = nnv5.forward(packed_vecs)

        C        = nnv5.array_used
        emit_pos = nnv5.emit[0, :C]
        per_row_emit = np.zeros((M, nnv5.emit_words), dtype=np.uint64)
        if C > 0 and choices_full.shape[1] > 0:
            matched = choices_full[:, :C]                              # (M, C)
            # Vectorized OR: mask emit rows by choices, reduce over C
            masked = np.where(
                matched[:, :, None],                                   # (M, C, 1)
                emit_pos[None, :, :],                                  # (1, C, EW)
                np.uint64(0)
            )                                                          # (M, C, EW)
            per_row_emit = np.bitwise_or.reduce(masked, axis=1)       # (M, EW)


        return per_row_emit, buf

    def update_case_table(self, hidden: torch.Tensor):
        """
        Run two-layer NNv5+NNv2 pipeline on the current batch's hidden states.
        Uses persistent buffers — NNv5 flushes independently of LLM batch size.

        hidden: (B, T, D) float
        """
        flat   = hidden.reshape(-1, self.d_model)
        packed = self._hidden_to_bool(flat)          # (N, d_model_words) uint64

        N             = packed.shape[0]
        d_model_words = packed.shape[1]
        chunk_words1  = (self.layer1_cfg.chunk_dim + 63) // 64
        n_chunks1     = d_model_words // chunk_words1

        # Reshape to (N * n_chunks1, chunk_words1)
        chunked1 = packed.reshape(N * n_chunks1, chunk_words1)

        emit1_rows, self._buf_l1 = self._run_layer(
            chunked1,
            self.nnv5_l1, self.nnv2_l1,
            self.layer1_cfg.chunk_dim,
            self.layer1_cfg.nnv5_chunk_size,
            self._buf_l1,
        )
        # emit1_rows: (N * n_chunks1, emit_words_l1)
        # Reshape to (N, n_chunks1 * emit_words_l1) then to (N * n_chunks2, chunk_words2)
        if emit1_rows.shape[0] == 0 or self.nnv5_l1.emit_used == 0:
            return

        n_chunks2    = self.n_chunks2
        chunk_words2 = (self.layer2_cfg.chunk_dim + 63) // 64
        emit1_flat   = emit1_rows.reshape(N, n_chunks1 * self.nnv5_l1.emit_words)

        # Pad to chunk_words2 boundary
        target_words = n_chunks2 * chunk_words2
        if emit1_flat.shape[1] < target_words:
            emit1_flat = np.pad(emit1_flat, ((0,0),(0, target_words - emit1_flat.shape[1])))

        chunked2 = emit1_flat.reshape(N * n_chunks2, chunk_words2)

        _, self._buf_l2 = self._run_layer(
            chunked2,
            self.nnv5_l2, self.nnv2_l2,
            self.layer2_cfg.chunk_dim,
            self.layer2_cfg.nnv5_chunk_size,
            self._buf_l2,
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
                print(f"  NNv5 tables frozen at step {step} — "
                      f"L1 cases={self.nnv5_l1.array_used} groups={self.nnv5_l1.emit_used} | "
                      f"L2 cases={self.nnv5_l2.array_used} groups={self.nnv5_l2.emit_used}")

        # Check layer2 has any structure to synthesize from
        if self.nnv5_l2.array_used == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        B, T, D = real_hidden.shape
        flat   = real_hidden.reshape(-1, D)
        packed = self._hidden_to_bool(flat)              # (N, d_model_words) uint64
        N             = packed.shape[0]
        d_model_words = packed.shape[1]
        chunk_words1  = (self.layer1_cfg.chunk_dim + 63) // 64
        n_chunks1     = d_model_words // chunk_words1
        chunked1      = packed.reshape(N * n_chunks1, chunk_words1)

        if self.nnv5_l1.array_used == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        use_torch = self.device.type == "cuda"
        if use_torch:
            _, _, _, choices1, _ = self.nnv5_l1.forward_torch(chunked1, self.device)
        else:
            _, _, _, choices1, _ = self.nnv5_l1.forward(chunked1)

        C1        = self.nnv5_l1.array_used
        emit_pos1 = self.nnv5_l1.emit[0, :C1]
        emit1_rows = np.zeros((N * n_chunks1, self.nnv5_l1.emit_words), dtype=np.uint64)
        if C1 > 0 and choices1.shape[1] > 0:
            m = choices1[:, :C1]
            masked1 = np.where(m[:, :, None], emit_pos1[None, :, :], np.uint64(0))
            emit1_rows = np.bitwise_or.reduce(masked1, axis=1)

        # --- Synthesis via reverse pipeline ---
        if self.nnv5_l2.array_used == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Synthesize from layer2
        n_chunks2    = self.n_chunks2
        chunk_words2 = (self.layer2_cfg.chunk_dim + 63) // 64
        emit1_flat   = emit1_rows.reshape(N, n_chunks1 * self.nnv5_l1.emit_words)
        target_words = n_chunks2 * chunk_words2
        if emit1_flat.shape[1] < target_words:
            emit1_flat = np.pad(emit1_flat, ((0,0),(0, target_words - emit1_flat.shape[1])))

        emit1_chunked = emit1_flat.reshape(N * n_chunks2, chunk_words2)
        if use_torch:
            _, _, _, choices2, _ = self.nnv5_l2.forward_torch(emit1_chunked, self.device)
        else:
            _, _, _, choices2, _ = self.nnv5_l2.forward(emit1_chunked)

        # Extract per-row L2 emit from choices
        C2        = self.nnv5_l2.array_used
        emit_pos2 = self.nnv5_l2.emit[0, :C2]
        l2_emit_rows_flat = np.zeros((N * n_chunks2, self.nnv5_l2.emit_words), dtype=np.uint64)
        if C2 > 0 and choices2.shape[1] > 0:
            m2 = choices2[:, :C2]
            masked2 = np.where(m2[:, :, None], emit_pos2[None, :, :], np.uint64(0))
            l2_emit_rows_flat = np.bitwise_or.reduce(masked2, axis=1)

        # Aggregate L2 emit to per-position — OR over chunks
        l2_emit_per_pos = np.zeros((N, self.nnv5_l2.emit_words), dtype=np.uint64)
        l2_emit_rows = l2_emit_rows_flat.reshape(N, n_chunks2, self.nnv5_l2.emit_words)
        for i in range(n_chunks2):
            l2_emit_per_pos |= l2_emit_rows[:, i, :]

        # Step 2: L2 reverse → synthetic L1 emit (per position)
        # Stack with neg plane (all zeros — no exclusions for synthesis)
        l2_emit_stacked = np.stack(
            [l2_emit_per_pos, np.zeros_like(l2_emit_per_pos)], axis=1
        )  # (N, 2, emit_words)
        synth_emit1_flat = self.nnv5_l2.reverse(l2_emit_stacked)  # (N, l2_match_words)

        # Apply NNv2 float affinity weighting on L2 reverse output if available
        if self.nnv2_l2.array_used > 0:
            _, _, _, choices_nnv2 = self.nnv2_l2.forward(synth_emit1_flat)
            synth_emit1_flat = self.nnv2_l2.reverse(
                np.stack([
                    np.array([
                        np.bitwise_or.reduce(
                            self.nnv2_l2.emit[0, choices_nnv2[i]], axis=0
                        ) if choices_nnv2[i].any() else np.zeros(self.nnv2_l2.emit_words, dtype=np.uint64)
                        for i in range(N)
                    ], dtype=np.uint64),
                    np.zeros((N, self.nnv2_l2.emit_words), dtype=np.uint64)
                ], axis=1)
            ) if hasattr(self.nnv2_l2, 'reverse') else synth_emit1_flat

        # Step 3: L1 reverse → synthetic packed bool
        # Use synth_emit1_flat as the emit for L1 reverse
        # Re-chunk to L1 emit shape
        l1_emit_words = self.nnv5_l1.emit_words
        synth_emit1_rechunked = synth_emit1_flat.reshape(N * n_chunks1, l1_emit_words) \
            if synth_emit1_flat.shape[1] == n_chunks1 * l1_emit_words \
            else np.zeros((N * n_chunks1, l1_emit_words), dtype=np.uint64)

        l1_emit_stacked = np.stack(
            [synth_emit1_rechunked, np.zeros_like(synth_emit1_rechunked)], axis=1
        )
        synth_chunked1 = self.nnv5_l1.reverse(l1_emit_stacked)   # (N*n_chunks1, chunk_words1)
        synth_packed   = synth_chunked1.reshape(N, n_chunks1 * chunk_words1)
        # Unpack uint64 → bits → float
        synth_bits = np.unpackbits(
            synth_packed.view(np.uint8), axis=1, bitorder='little'
        )[:, :self.d_model].astype(np.float32)               # (N, d_model)

        # Rescale to match real hidden state distribution
        synth_hidden = torch.from_numpy(synth_bits).to(self.device)
        real_flat    = flat.float()
        real_mean    = real_flat.mean(dim=0, keepdim=True)
        real_std     = real_flat.std(dim=0, keepdim=True) + 1e-8
        synth_hidden = (synth_hidden - synth_hidden.mean(dim=0, keepdim=True)) / \
                       (synth_hidden.std(dim=0, keepdim=True) + 1e-8)
        synth_hidden = synth_hidden * real_std + real_mean
        synth_hidden = synth_hidden.reshape(B, T, D)

        _, synth_loss = self.model.forward_from_hidden(synth_hidden, targets)
        return synth_loss

    def stats(self) -> dict:
        return {
            "l1_cases":       self.nnv5_l1.array_used,
            "l1_groups":      self.nnv5_l1.emit_used,
            "l2_cases":       self.nnv5_l2.array_used,
            "l2_groups":      self.nnv5_l2.emit_used,
            "nnv2_l1_cases":  self.nnv2_l1.array_used,
            "nnv2_l2_cases":  self.nnv2_l2.array_used,
        }
