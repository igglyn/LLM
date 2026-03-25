
"""nnv5.py — NNv5Block and NNv5CapacityError.

All state is stored as torch.int64 tensors on device.
Numpy is used only at the assign boundary (diff/emit arrays).
All tensor ops use torch.compile for multi-core CPU and GPU acceleration.
"""
from __future__ import annotations
import numpy as np
import torch

try:
    import torch._dynamo
    torch._dynamo.config.capture_scalar_outputs = True
except Exception:
    pass

def _tc(fn):
    """Torch compile disabled — eager mode is sufficient for small tensors."""
    return fn


# ---------------------------------------------------------------------------
# Compiled kernels — all int64 (bitwise identical to uint64)
# ---------------------------------------------------------------------------

def _merge_diffs_torch(diff, emit, was_matched):
    """
    Dedup diffs via tril grid, then merge by positive plane.
    diff:        (K, 2, W) int64
    emit:        (K, 2, EW) int64
    was_matched: (K,) bool
    """
    K  = diff.shape[0]
    W  = diff.shape[2]
    EW = emit.shape[2]
    if K == 0:
        return diff, emit, was_matched
    AND_ID = torch.tensor(-1, dtype=torch.int64, device=diff.device)

    # Full dedup
    flat = diff.reshape(K, -1)
    eq   = (flat[:, None, :] == flat[None, :, :]).all(dim=2)
    keep = ~torch.tril(eq, diagonal=-1).any(dim=0)
    diff2, emit2, wm2 = diff[keep], emit[keep], was_matched[keep]
    K2 = diff2.shape[0]
    if K2 <= 1:
        return diff2, emit2, wm2

    # Merge by positive plane
    pos    = diff2[:, 0, :]
    pos_eq = (pos[:, None, :] == pos[None, :, :]).all(dim=2)
    pos_keep = ~torch.tril(pos_eq, diagonal=-1).any(dim=0)
    n_g    = int(pos_keep.sum().item())
    if n_g == K2:
        return diff2, emit2, wm2

    unique_pos = pos[pos_keep]
    group_id   = (pos[:, None, :] == unique_pos[None]).all(dim=2).float().argmax(dim=1)  # (K2,)

    md = torch.zeros(n_g, 2, W,  dtype=torch.int64, device=diff.device)
    me = torch.zeros(n_g, 2, EW, dtype=torch.int64, device=diff.device)
    mm = torch.ones(n_g,         dtype=torch.bool,  device=diff.device)
    md[:, 1] = AND_ID
    me[:, 1] = AND_ID

    # Scatter pos plane — all rows in same group share pos, just assign
    md[:, 0].scatter_(0, group_id.unsqueeze(1).expand(-1, W), diff2[:, 0])

    # AND-reduce neg plane per group via scatter — use float trick:
    # initialize neg to AND_ID, then AND each row in
    gid_w  = group_id.unsqueeze(1).expand(-1, W)
    gid_ew = group_id.unsqueeze(1).expand(-1, EW)
    # neg plane: need per-group AND — do it by iterating K2 (small in practice)
    for i in range(K2):
        g = int(group_id[i].item())
        md[g, 1] &= diff2[i, 1]
        me[g, 0] |= emit2[i, 0]
        me[g, 1] &= emit2[i, 1]
        mm[g]    &= wm2[i]

    return md, me, mm


@_tc
def _pack_choices_torch(choices: torch.Tensor) -> torch.Tensor:
    """Pack (N, C) bool → (N, ceil(C/64)) int64. Vectorized, compiles once."""
    N, C = choices.shape
    if C == 0:
        return torch.zeros(N, 0, dtype=torch.int64, device=choices.device)
    words   = (C + 63) // 64
    bits    = torch.arange(C, dtype=torch.int64, device=choices.device)
    word    = bits >> 6
    bit     = bits & 63
    shifted = choices.to(torch.int64) << bit
    packed  = torch.zeros(N, words, dtype=torch.int64, device=choices.device)
    packed.scatter_add_(1, word.unsqueeze(0).expand(N, -1), shifted)
    return packed


@_tc
def _unpack_choices_torch(packed: torch.Tensor, C: int) -> torch.Tensor:
    """Unpack (N, ceil(C/64)) int64 → (N, C) bool. Vectorized, compiles once."""
    N = packed.shape[0]
    if C == 0:
        return torch.zeros(N, 0, dtype=torch.bool, device=packed.device)
    bits = torch.arange(C, dtype=torch.int64, device=packed.device)
    word = bits >> 6
    bit  = bits & 63
    return ((packed[:, word] >> bit) & 1).bool()


def _forward_all_shards(tokens, tokens_inv, match, emit, n_shards, shard_size):
    """
    Fully batched forward — all shards in one kernel dispatch, no Python loop.
    Uses NOT-OR-NOT trick for bitwise AND reduce over int64.

    tokens/tokens_inv: (N, W) int64
    match:  (2, n_shards, shard_size, W) int64
    emit:   (2, n_shards, shard_size, EW) int64
    Returns: choices (N, cap) bool, emit_p (N, EW), emit_n (N, EW),
             reverse (N, 2, W), counts (cap,) int64
    """
    N   = tokens.shape[0]
    W   = tokens.shape[1]
    EW  = emit.shape[3]
    dev = tokens.device
    cap = n_shards * shard_size
    z   = torch.zeros(1, dtype=torch.int64, device=dev)

    mp = match[0]   # (ns, ss, W)
    mn = match[1]
    ep = emit[0]    # (ns, ss, EW)
    en = emit[1]

    # Broadcast match — (N, 1, 1, W) & (1, ns, ss, W) → (N, ns, ss)
    t  = tokens[:, None, None, :]
    ti = tokens_inv[:, None, None, :]
    mp_ = mp[None]
    mn_ = mn[None]

    sc = ((t & mp_) == mp_).all(dim=3) & \
         ((ti & mn_) == mn_).all(dim=3)         # (N, ns, ss)

    choices = sc.reshape(N, cap)
    counts  = choices.sum(dim=0)

    # Emit AND — mask inactive slots with all-ones (AND identity)
    em  = ep.any(dim=2) | en.any(dim=2)          # (ns, ss) has_emit
    act = sc & em[None]                           # (N, ns, ss)

    # (N, ns, ss, EW) — inactive slots stay at all-ones
    mep = torch.where(act[..., None], ep[None], torch.tensor(-1, dtype=torch.int64, device=dev))
    men = torch.where(act[..., None], en[None], torch.tensor(-1, dtype=torch.int64, device=dev))

    # AND reduce — used for structural learning (assign/commit)
    mep_flat = mep.reshape(N, cap, EW)
    men_flat = men.reshape(N, cap, EW)
    ep_and = mep_flat[:, 0].clone()
    en_and = men_flat[:, 0].clone()
    for i in range(1, cap):
        ep_and = ep_and & mep_flat[:, i]
        en_and = en_and & men_flat[:, i]

    # OR reduce — used for adapter output (union of matched group memberships)
    ep_or_flat = torch.where(act[..., None], ep[None],
                             torch.zeros(1, dtype=torch.int64, device=dev))
    en_or_flat = torch.where(act[..., None], en[None],
                             torch.zeros(1, dtype=torch.int64, device=dev))
    ep_or_flat = ep_or_flat.reshape(N, cap, EW)
    en_or_flat = en_or_flat.reshape(N, cap, EW)
    ep_or = ep_or_flat[:, 0].clone()
    en_or = en_or_flat[:, 0].clone()
    for i in range(1, cap):
        ep_or = ep_or | ep_or_flat[:, i]
        en_or = en_or | en_or_flat[:, i]

    # Zero unmatched
    wm = choices.any(dim=1)
    zeros_ew = torch.zeros(N, EW, dtype=torch.int64, device=dev)
    ep_and = torch.where(wm[:, None], ep_and, zeros_ew)
    en_and = torch.where(wm[:, None], en_and, zeros_ew)
    ep_or  = torch.where(wm[:, None], ep_or,  zeros_ew)
    en_or  = torch.where(wm[:, None], en_or,  zeros_ew)

    # Reverse OR
    rp = torch.where(sc[..., None], mp_, z)
    rn = torch.where(sc[..., None], mn_, z)
    rp_flat = rp.reshape(N, cap, W)
    rn_flat = rn.reshape(N, cap, W)
    rp_out = rp_flat[:, 0].clone()
    for i in range(1, cap):
        rp_out = rp_out | rp_flat[:, i]
    rn_out = rn_flat[:, 0].clone()
    for i in range(1, cap):
        rn_out = rn_out | rn_flat[:, i]

    reverse = torch.stack([rp_out, rn_out], dim=1)   # (N, 2, W)

    # Returns: AND emit for learning, OR emit for adapter output
    return choices, ep_and, en_and, ep_or, en_or, reverse, counts


@_tc
def _new_case_mask_torch(diff_pos, match_pos):
    """
    diff_pos:  (K, W) int64
    match_pos: (C, W) int64
    Returns new_case_mask (K,) bool, diffed_neg_mask (C,) bool
    """
    differs = (diff_pos[:, None, :] ^ match_pos[None]).any(dim=2)
    return differs.all(dim=1), differs.any(dim=0)


@_tc
def _tighten_neg_torch(match_neg_flat, neg_cols, neg_case):
    """
    match_neg_flat: (C, W) int64 — cloned and tightened
    neg_cols: (M,) int64
    neg_case: (W,) int64
    """
    result = match_neg_flat.clone()
    result[neg_cols] = result[neg_cols] & neg_case[None, :]
    return result


# ---------------------------------------------------------------------------
# NNv5CapacityError
# ---------------------------------------------------------------------------

class NNv5CapacityError(RuntimeError):
    """Raised when NNv5Block exceeds allocated capacity."""
    pass


# ---------------------------------------------------------------------------
# NNv5Block
# ---------------------------------------------------------------------------

class NNv5Block:
    """
    NNv5 U1XToU1X — bitwise subset match with residual-driven case discovery.

    State layout: (2, n_shards, shard_size, W) int64 on device.
    Planes: 0=positive (required bits), 1=negative (forbidden bits).
    Unused slots have match[1]=MAX so they never match any input.

    case_capacity and shard_size must be powers of 2.
    case_capacity must be >= shard_size.
    """
    def __init__(self, bool_dim: int, case_capacity: int, group_capacity: int,
                 shard_size: int = 32, device: torch.device = None):
        assert case_capacity & (case_capacity - 1) == 0, "case_capacity must be power of 2"
        assert shard_size    & (shard_size    - 1) == 0, "shard_size must be power of 2"
        assert case_capacity >= shard_size,               "case_capacity must be >= shard_size"

        self.bool_dim       = bool_dim
        self.case_capacity  = case_capacity
        self.group_capacity = group_capacity
        self.shard_size     = shard_size
        self.device         = device or torch.device('cpu')
        self.shard_bits     = shard_size.bit_length() - 1
        self.array_used     = 0
        self.emit_used      = 0
        self.match_words    = (bool_dim + 63) // 64
        self.emit_words     = (group_capacity + 63) // 64

        W   = self.match_words
        EW  = self.emit_words
        ss  = shard_size
        n_s = case_capacity // shard_size

        self.match = torch.zeros(2, n_s, ss, W,  dtype=torch.int64, device=self.device)
        self.match[1] = -1

        self.emit  = torch.zeros(2, n_s, ss, EW, dtype=torch.int64, device=self.device)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Pack / unpack choices
    # ------------------------------------------------------------------

    @staticmethod
    def pack_choices(choices) -> torch.Tensor:
        """Pack (N, C) bool → (N, ceil(C/64)) int64 torch tensor."""
        if not isinstance(choices, torch.Tensor):
            choices = torch.from_numpy(choices.astype(bool))
        return _pack_choices_torch(choices)

    @staticmethod
    def unpack_choices(packed, C: int) -> torch.Tensor:
        """Unpack (N, ceil(C/64)) int64 → (N, C) bool torch tensor."""
        if not isinstance(packed, torch.Tensor):
            packed = torch.from_numpy(packed.view(np.int64))
        return _unpack_choices_torch(packed, C)

    # ------------------------------------------------------------------
    # Merge diffs
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_diffs(diff, emit, was_matched) -> tuple:
        """Dispatch to torch merge — diff/emit may be numpy or torch."""
        if isinstance(diff, np.ndarray):
            diff        = torch.from_numpy(diff.view(np.int64))
            emit        = torch.from_numpy(emit.view(np.int64))
            was_matched = torch.from_numpy(was_matched)
            d, e, wm = _merge_diffs_torch(diff, emit, was_matched)
            return d.numpy().view(np.uint64), e.numpy().view(np.uint64), wm.numpy()
        return _merge_diffs_torch(diff, emit, was_matched)

    # ------------------------------------------------------------------
    # Emit group bit helpers
    # ------------------------------------------------------------------

    def _set_emit_group_bit(self, flat_idx, group_idx: int, *, is_member: bool):
        if not (0 <= group_idx < self.group_capacity):
            raise NNv5CapacityError(
                f"emit group index out of bounds: {group_idx} >= {self.group_capacity}")
        word, bit = divmod(group_idx, 64)
        flag_val = np.array([np.uint64(1) << np.uint64(bit)], dtype=np.uint64).view(np.int64)[0]
        flag     = torch.tensor(int(flag_val), dtype=torch.int64, device=self.device)
        plane = 0 if is_member else 1
        if isinstance(flat_idx, int):
            s, c = self._shard_slot(flat_idx)
            self.emit[plane, s, c, word] |= flag
        else:
            stop = self.array_used if flat_idx == slice(None, self.array_used) \
                   else (flat_idx.stop or self.array_used)
            for fi in range(stop):
                s, c = self._shard_slot(fi)
                self.emit[plane, s, c, word] |= flag

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, tokens: np.ndarray):
        """
        tokens: (N, match_words) uint64 numpy
        Returns 6-tuple:
            diff, emit_and, was_matched_diff  — numpy, diff-aligned, for assign
            choices_packed                    — (N, ceil(C/64)) int64 torch tensor
            counts                            — (C,) uint64 numpy
            emit_out                          — (N, 2, EW) uint64 numpy, per-input
        """
        N  = tokens.shape[0]
        EW = self.emit_words

        if self.array_used == 0:
            ti  = ~tokens
            inp = np.stack((tokens, ti), axis=1)
            inp2, _ = np.unique(inp, axis=0, return_index=True)
            cp_empty = torch.zeros(N, 0, dtype=torch.int64, device=self.device)
            return (inp2,
                    np.zeros((inp2.shape[0], 2, EW), dtype=np.uint64),
                    np.zeros(inp2.shape[0], dtype=bool),
                    cp_empty,
                    np.zeros(0, dtype=np.uint64),
                    np.zeros((N, 2, EW), dtype=np.uint64))

        t_tok  = torch.from_numpy(tokens.view(np.int64)).to(self.device)
        t_tinv = torch.bitwise_not(t_tok)

        choices_t, ep_t, en_t, ep_or_t, en_or_t, rev_t, cnt_t = _forward_all_shards(
            t_tok, t_tinv,
            self.match,
            self.emit,
            self.n_shards, self.shard_size)

        C = self.array_used
        choices_t = choices_t[:, :C]
        cnt_t     = cnt_t[:C]

        wm_t = choices_t.any(dim=1)

        # AND emit — used for diff computation and learning
        ep_t[~wm_t] = 0
        en_t[~wm_t] = 0

        # Post-process diff on GPU — uses AND emit
        t_inp  = torch.stack([t_tok, t_tinv], dim=1)
        diff_t = rev_t & t_inp ^ t_inp
        nz_t   = diff_t[:, 0].any(dim=1)

        dc, ec, wc = diff_t[nz_t], torch.stack([ep_t, en_t], dim=1)[nz_t], wm_t[nz_t]
        dc2, ec2, wc2 = NNv5Block._merge_diffs(dc, ec, wc)

        choices_packed = _pack_choices_torch(choices_t)

        # emit_out uses OR emit — union of group memberships across matched cases
        emit_out = torch.stack([ep_or_t, en_or_t], dim=1).cpu().numpy().view(np.uint64)

        return (dc2.numpy().view(np.uint64),
                ec2.numpy().view(np.uint64),
                wc2.numpy(),
                choices_packed,
                cnt_t.cpu().numpy().astype(np.uint64),
                emit_out)

    def forward_torch(self, tokens: np.ndarray, device: torch.device):
        """Same as forward() but uses the given device explicitly."""
        N  = tokens.shape[0]
        EW = self.emit_words

        if self.array_used == 0:
            ti   = ~tokens
            inp  = np.stack((tokens, ti), axis=1)
            inp2, _ = np.unique(inp, axis=0, return_index=True)
            cp_empty = torch.zeros(N, 0, dtype=torch.int64, device=device)
            return (inp2,
                    np.zeros((inp2.shape[0], 2, EW), dtype=np.uint64),
                    np.zeros(inp2.shape[0], dtype=bool),
                    cp_empty,
                    np.zeros(0, dtype=np.uint64),
                    np.zeros((N, 2, EW), dtype=np.uint64))

        t_tok  = torch.from_numpy(tokens.view(np.int64)).to(device)
        t_tinv = torch.bitwise_not(t_tok)

        choices_t, ep_t, en_t, ep_or_t, en_or_t, rev_t, cnt_t = _forward_all_shards(
            t_tok, t_tinv,
            self.match.to(device),
            self.emit.to(device),
            self.n_shards, self.shard_size)

        C = self.array_used
        choices_t = choices_t[:, :C]
        cnt_t     = cnt_t[:C]

        wm_t = choices_t.any(dim=1)
        ep_t[~wm_t] = 0
        en_t[~wm_t] = 0

        t_inp  = torch.stack([t_tok, t_tinv], dim=1)
        diff_t = rev_t & t_inp ^ t_inp
        nz_t   = diff_t[:, 0].any(dim=1)

        dc, ec, wc = diff_t[nz_t], torch.stack([ep_t, en_t], dim=1)[nz_t], wm_t[nz_t]
        dc2, ec2, wc2 = _merge_diffs_torch(dc, ec, wc)

        choices_packed = _pack_choices_torch(choices_t)

        # OR emit for output signal
        emit_out = torch.stack([ep_or_t, en_or_t], dim=1).cpu().numpy().view(np.uint64)

        return (dc2.cpu().numpy().view(np.uint64),
                ec2.cpu().numpy().view(np.uint64),
                wc2.cpu().numpy(),
                choices_packed,
                cnt_t.cpu().numpy().astype(np.uint64),
                emit_out)

    # ------------------------------------------------------------------
    # Assign
    # ------------------------------------------------------------------

    def assign_candidates(self, diff: np.ndarray, emit: np.ndarray,
                           was_matched: np.ndarray) -> tuple:
        """Process diffs into candidate cases without committing."""
        if diff.shape[0] == 0:
            empty    = np.zeros((0, 2, self.match_words), dtype=np.uint64)
            em_empty = np.zeros((0, 2, self.emit_words),  dtype=np.uint64)
            neg_case = np.full(self.match_words, np.iinfo(np.uint64).max, dtype=np.uint64)
            return neg_case, empty, em_empty, empty

        neg_case = np.bitwise_and.reduce(diff[:, 1], axis=0)

        if self.array_used > 0:
            C = self.array_used
            W = self.match_words

            t_diff_pos  = torch.from_numpy(diff[:, 0].view(np.int64)).to(self.device)
            t_match_pos = self.match[0].reshape(-1, W)[:C]

            new_mask_t, diffed_t = _new_case_mask_torch(t_diff_pos, t_match_pos)
            new_case_mask   = new_mask_t.cpu().numpy()
            diffed_neg_mask = diffed_t.cpu().numpy()

            neg_cols = np.where(~diffed_neg_mask)[0]
            if len(neg_cols) > 0:
                t_neg  = self.match[1].reshape(-1, W)[:C]
                t_nc   = torch.from_numpy(neg_cols.astype(np.int64)).to(self.device)
                t_neg_case = torch.from_numpy(neg_case.view(np.int64)).to(self.device)
                updated = _tighten_neg_torch(t_neg, t_nc, t_neg_case)
                for ci in neg_cols:
                    s, slot = self._shard_slot(int(ci))
                    self.match[1, s, slot] = updated[ci].to(self.device)
        else:
            new_case_mask = np.ones(diff.shape[0], dtype=bool)

        new_cases       = diff[new_case_mask & was_matched]
        new_cases_emit  = emit[new_case_mask & was_matched]
        new_group_cases = diff[new_case_mask & ~was_matched]

        return neg_case, new_cases, new_cases_emit, new_group_cases

    def commit_candidates(self, neg_case: np.ndarray, new_cases: np.ndarray,
                          new_cases_emit: np.ndarray, new_group_cases: np.ndarray):
        """Commit candidate cases to the table."""
        def _t(a):
            return torch.from_numpy(a.view(np.int64)).to(self.device)

        if new_cases.shape[0] > 0:
            n = new_cases.shape[0]
            if self.array_used + n > self.case_capacity:
                raise NNv5CapacityError(
                    f"NNv5Block case capacity exceeded: used={self.array_used}+{n} "
                    f"> capacity={self.case_capacity}")
            for i in range(n):
                s, slot = self._shard_slot(self.array_used + i)
                self.match[0, s, slot] = _t(new_cases[i, 0])
                self.match[1, s, slot] = _t(new_cases[i, 1] & neg_case)
                self.emit[0, s, slot] |= _t(new_cases_emit[i, 0])
                self.emit[1, s, slot] &= _t(new_cases_emit[i, 1])
            self.array_used += n

        if new_group_cases.shape[0] > 0:
            if self.array_used + 1 > self.case_capacity:
                raise NNv5CapacityError(
                    f"NNv5Block case capacity exceeded on group creation: "
                    f"used={self.array_used} >= capacity={self.case_capacity}")
            if self.emit_used >= self.group_capacity:
                raise NNv5CapacityError(
                    f"NNv5Block group capacity exceeded: "
                    f"emit_used={self.emit_used} >= group_capacity={self.group_capacity}")

            gidx    = self.emit_used
            self.emit_used += 1
            s, slot = self._shard_slot(self.array_used)

            self.match[0, s, slot] = _t(new_group_cases[0, 0])
            self.match[1, s, slot] = _t(new_group_cases[0, 1] & neg_case)
            self._set_emit_group_bit(self.array_used, gidx, is_member=True)

            if gidx > 0:
                full_words, rem_bits = divmod(gidx, 64)
                self.emit[1, s, slot, :full_words] = -1
                if rem_bits:
                    mask_val = np.array([(np.uint64(1) << np.uint64(rem_bits)) - np.uint64(1)], dtype=np.uint64).view(np.int64)[0]
                    self.emit[1, s, slot, full_words] |= \
                        torch.tensor(int(mask_val), dtype=torch.int64, device=self.device)

            self._set_emit_group_bit(slice(None, self.array_used), gidx, is_member=False)
            self.array_used += 1

    def assign(self, diff: np.ndarray, emit: np.ndarray, was_matched: np.ndarray):
        """Full assign — candidates + commit."""
        neg, nc, nce, ngc = self.assign_candidates(diff, emit, was_matched)
        self.commit_candidates(neg, nc, nce, ngc)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def reverse(self, emit: np.ndarray) -> np.ndarray:
        """Reconstruct inputs from emit group membership vectors."""
        if self.array_used == 0:
            return np.zeros((emit.shape[0], self.match_words), dtype=np.uint64)
        C = self.array_used
        emit_pos_cases = self.emit[0].reshape(-1, self.emit_words)[:C].cpu().numpy().view(np.uint64)
        input_bits = np.unpackbits(emit[:, 0].view(np.uint8), axis=1, bitorder='little').astype(np.float32)
        case_bits  = np.unpackbits(emit_pos_cases.view(np.uint8), axis=1, bitorder='little').astype(np.float32)
        overlap    = (input_bits @ case_bits.T) > 0
        match_pos  = self.match[0].reshape(-1, self.match_words)[:C].cpu().numpy().view(np.uint64)
        match_bits = np.unpackbits(np.ascontiguousarray(match_pos).view(np.uint8), axis=1, bitorder='little').astype(np.float32)
        result_bits = (overlap.astype(np.float32) @ match_bits) > 0
        return np.packbits(result_bits.astype(np.uint8), axis=1, bitorder='little').view(np.uint64)[:, :self.match_words]

    def synthesize_from(self, packed_vecs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Generate vocab-masked random noise."""
        N = packed_vecs.shape[0]
        if self.array_used == 0 or self.emit_used == 0:
            return packed_vecs.copy()
        C = self.array_used
        match_pos = self.match[0].reshape(-1, self.match_words)[:C].cpu().numpy().view(np.uint64)
        vocab = np.bitwise_or.reduce(match_pos, axis=0)
        noise = rng.integers(np.iinfo(np.uint64).min, np.iinfo(np.uint64).max,
                             size=(N, self.match_words), dtype=np.uint64)
        return noise & vocab[None, :]

    def stats(self) -> dict:
        return {
            "cases_used":     self.array_used,
            "groups_used":    self.emit_used,
            "case_capacity":  self.case_capacity,
            "group_capacity": self.group_capacity,
        }
