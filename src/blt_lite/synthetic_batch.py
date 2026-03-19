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

    def assign(self, diff: np.ndarray, emit: np.ndarray, was_matched: np.ndarray):
        """Update case table from forward pass residuals."""
        if diff.shape[0] == 0:
            return

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

        new_cases = diff[new_case_mask & was_matched]
        if new_cases.shape[0] > 0:
            n = new_cases.shape[0]
            assert self.array_used + n <= self.case_capacity, "Case capacity exceeded"
            sl = slice(self.array_used, self.array_used + n)
            self.match[:, :, sl] = np.permute_dims(new_cases, (1, 2, 0))
            self.match[1, :, sl] &= neg_case[..., None]
            emit_shift = np.permute_dims(emit[new_case_mask & was_matched], (1, 0, 2))
            self.emit[0, sl] |= emit_shift[0]
            self.emit[1, sl] &= emit_shift[1]
            self.array_used += n

        new_group_cases = diff[new_case_mask & ~was_matched]
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

    def stats(self) -> dict:
        return {
            "cases_used":  self.array_used,
            "groups_used": self.emit_used,
            "case_capacity":  self.case_capacity,
            "group_capacity": self.group_capacity,
        }


# ---------------------------------------------------------------------------
# Synthetic Batch Harness
# ---------------------------------------------------------------------------

class SyntheticBatchHarness:
    """
    Wraps a TinyPatchLM trunk to provide synthetic batch augmentation.

    Usage in training loop:
        harness = SyntheticBatchHarness(model, cfg, device)
        harness.attach()                        # register forward hook

        # Real batch forward
        logits, loss = model(x, y)
        real_hidden = harness.captured_hidden   # (B, T, D)

        # Synthetic pass
        synth_loss = harness.synthetic_pass(real_hidden, y)
        total_loss = loss + synth_weight * synth_loss

        loss.backward()
        harness.detach()                        # call when done training
    """
    def __init__(
        self,
        model,
        d_model: int,
        bool_dim: int,
        case_capacity: int,
        group_capacity: int,
        n_synthetic: int,
        match_threshold: float,
        device: torch.device,
        seed: int = 42,
    ):
        self.model          = model
        self.d_model        = d_model
        self.bool_dim       = bool_dim
        self.n_synthetic    = n_synthetic
        self.match_threshold = match_threshold
        self.device         = device
        self.rng            = np.random.default_rng(seed)

        self.projection = BoolProjection(d_model, bool_dim).to(device)
        self.nnv5       = NNv5Block(bool_dim, case_capacity, group_capacity)

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

    def update_case_table(self, hidden: torch.Tensor):
        """
        Run NNv5 forward + assign on the current batch's hidden states.
        Updates the persistent case table.

        hidden: (B, T, D) float
        """
        # Project to bool
        bool_vecs = self.projection.encode_hard(hidden.reshape(-1, self.d_model))

        # NNv5 forward
        diff, emit, was_matched, choices, _ = self.nnv5.forward(bool_vecs)

        # Update case table
        self.nnv5.assign(diff, emit, was_matched)

        return choices

    def synthetic_pass(
        self,
        real_hidden: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate synthetic hidden states and compute loss on them.

        real_hidden: (B, T, D) float — captured from real forward pass
        targets:     (B, T)    int64 — token targets

        returns: scalar loss tensor
        """
        # Update case table from this batch
        self.update_case_table(real_hidden)

        if self.nnv5.array_used == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Generate synthetic bool vectors
        B, T, D = real_hidden.shape
        synth_bools = self.nnv5.synthesize(self.n_synthetic * T, self.rng)  # (N*T, bool_dim)
        synth_bools_t = torch.from_numpy(synth_bools).float().to(self.device)  # (N*T, bool_dim)

        # Project back to float hidden states
        synth_hidden = self.projection.decode(synth_bools_t)  # (N*T, d_model)
        synth_hidden = synth_hidden.reshape(self.n_synthetic, T, D)

        # Repeat targets for synthetic batch
        # Use the first n_synthetic examples' targets (or tile if batch is smaller)
        if B >= self.n_synthetic:
            synth_targets = targets[:self.n_synthetic]
        else:
            reps = (self.n_synthetic + B - 1) // B
            synth_targets = targets.repeat(reps, 1)[:self.n_synthetic]

        # Forward through trunk only (hidden states bypass patcher)
        _, synth_loss = self.model.forward_from_hidden(synth_hidden, synth_targets)

        return synth_loss

    def projection_parameters(self):
        """Return projection parameters for optimizer."""
        return self.projection.parameters()

    def stats(self) -> dict:
        return self.nnv5.stats()
