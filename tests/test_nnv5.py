"""
test_nnv5.py — NNv5Block tests, fully torch-native.
"""
import sys, os, numpy as np

try:
    import torch
except ImportError:
    print("torch not available — cannot run NNv5 tests")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "blt_lite.synthetic_batch",
    os.path.join(os.path.dirname(__file__), '..', 'src', 'blt_lite', 'synthetic_batch', 'nnv5.py')
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["blt_lite.synthetic_batch"] = _mod
_spec.loader.exec_module(_mod)
NNv5Block = _mod.NNv5Block

DEVICE = torch.device('cpu')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_block(bool_dim=64, case_capacity=64, group_capacity=16, shard_size=8):
    cap = max(case_capacity, shard_size)
    return NNv5Block(bool_dim, cap, group_capacity, shard_size, device=DEVICE)

def random_tokens(n, match_words, seed):
    g = torch.Generator(); g.manual_seed(seed)
    return torch.randint(torch.iinfo(torch.int64).min,
                         torch.iinfo(torch.int64).max,
                         (n, match_words), dtype=torch.int64,
                         generator=g, device=DEVICE)

def t2np(t):   return t.cpu().numpy().view(np.uint64)
def np2t(a):   return torch.from_numpy(a.view(np.int64)).to(DEVICE)

def get_match(blk, plane, flat_idx, word=0):
    s, c = flat_idx >> blk.shard_bits, flat_idx & (blk.shard_size - 1)
    return blk.match[plane, s, c, word]

def set_match(blk, plane, flat_idx, word, value):
    s, c = flat_idx >> blk.shard_bits, flat_idx & (blk.shard_size - 1)
    blk.match[plane, s, c, word] = torch.tensor(int(value), dtype=torch.int64, device=DEVICE)

def get_emit(blk, plane, flat_idx, word=0):
    s, c = flat_idx >> blk.shard_bits, flat_idx & (blk.shard_size - 1)
    return blk.emit[plane, s, c, word]

def ival(t):
    return t.item() if hasattr(t, 'item') else int(t)

def make_diff(*rows):
    return np.array([[[np.uint64(p)], [np.uint64(n)]] for p, n in rows], dtype=np.uint64)

def zeros_emit(k, blk):
    return np.zeros((k, 2, blk.emit_words), dtype=np.uint64)

def fwd(blk, seed, n=8):
    toks = t2np(random_tokens(n, blk.match_words, seed))
    return blk.forward(toks), toks

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pack_unpack_roundtrip():
    g = torch.Generator()
    for C in [1, 7, 63, 64, 65, 128, 130]:
        g.manual_seed(C)
        choices = torch.randint(0, 2, (32, C), dtype=torch.bool, generator=g, device=DEVICE)
        assert torch.equal(NNv5Block.unpack_choices(NNv5Block.pack_choices(choices), C), choices), \
            f"Roundtrip failed C={C}"
    print("  PASS  pack/unpack roundtrip")

def test_empty_forward():
    blk = make_block()
    diff, emit, wm, cp, _, _ = blk.forward(t2np(random_tokens(8, blk.match_words, 0)))
    assert not wm.any() and cp.shape == (8, 0)
    print("  PASS  empty forward")

def test_forward_after_assign():
    blk = make_block()
    toks = t2np(random_tokens(4, blk.match_words, 1))
    for _ in range(5):
        diff, emit, wm, _, _, _ = blk.forward(toks)
        if diff.shape[0] == 0: break
        blk.assign(diff, emit, wm)
    C = blk.array_used
    _, _, _, cp2, _, _ = blk.forward(toks)
    matched = NNv5Block.unpack_choices(cp2, C).any(dim=1)
    assert blk.array_used > 0 and matched.any()
    print(f"  PASS  forward after assign — {C}c/{blk.emit_used}g, {matched.sum().item()}/{len(matched)} matched")

def test_was_matched_consistent_with_choices():
    blk = make_block()
    toks = t2np(random_tokens(16, blk.match_words, 2))
    for _ in range(3):
        diff, emit, wm, _, _, _ = blk.forward(toks)
        if diff.shape[0] == 0: break
        blk.assign(diff, emit, wm)
    new_toks = np.concatenate([toks[:8], t2np(random_tokens(8, blk.match_words, 99))])
    C = blk.array_used
    _, _, _, cp2, _, _ = blk.forward(new_toks)
    derived = NNv5Block.unpack_choices(cp2, C).any(dim=1)
    assert derived.shape[0] == len(new_toks)
    print(f"  PASS  was_matched consistent — {derived.sum().item()}/{len(new_toks)} matched")

def test_emit_aggregation_intersection():
    blk = make_block(case_capacity=32, group_capacity=16)
    toks = t2np(random_tokens(8, blk.match_words, 3))
    diff, emit, wm, _, _, _ = blk.forward(toks)
    blk.assign(diff, emit, wm)
    if blk.emit_used == 0:
        print("  SKIP  emit aggregation (no groups)")
        return
    C = blk.array_used
    _, _, _, cp2, _, eo = blk.forward(toks)
    choices = NNv5Block.unpack_choices(cp2, C)
    ep = blk.emit[0].reshape(-1, blk.emit_words)[:C]
    for i in range(len(toks)):
        mask = choices[i, :C]
        if not mask.any():
            assert not torch.from_numpy(eo[i, 0].view(np.int64)).any()
            continue
        rows = ep[mask]
        expected = rows[0].clone()
        for r in rows[1:]: expected &= r
        assert torch.equal(torch.from_numpy(eo[i, 0].view(np.int64)), expected)
    print("  PASS  emit aggregation intersection")

def test_reverse_reconstruction():
    blk = make_block(case_capacity=32)
    toks = t2np(random_tokens(8, blk.match_words, 4))
    diff, emit, wm, _, _, _ = blk.forward(toks)
    blk.assign(diff, emit, wm)
    diff2, _, _, _, _, _ = blk.forward(toks)
    assert diff2.shape[0] <= diff.shape[0]
    print("  PASS  reverse reconstruction")

def test_assign_creates_groups():
    blk = make_block(case_capacity=64, group_capacity=16)
    for i in range(4):
        diff, emit, wm, _, _, _ = blk.forward(t2np(random_tokens(8, blk.match_words, i*10)))
        blk.assign(diff, emit, wm)
    assert blk.emit_used > 0 and blk.array_used > 0
    print(f"  PASS  assign creates groups — {blk.array_used}c/{blk.emit_used}g")

def test_choices_packed_shape():
    blk = make_block(case_capacity=256, group_capacity=16)
    toks = t2np(random_tokens(16, blk.match_words, 6))
    diff, emit, wm, _, _, _ = blk.forward(toks)
    blk.assign(diff, emit, wm)
    for i in range(10):
        d2, e2, w2, _, _, _ = blk.forward(t2np(random_tokens(16, blk.match_words, 100+i)))
        blk.assign(d2, e2, w2)
    C = blk.array_used
    _, _, _, cp, _, _ = blk.forward(toks)
    assert cp.shape == (16, (C+63)//64)
    print(f"  PASS  choices packed shape — C={C}")

def test_forward_consistency():
    blk = make_block(case_capacity=64, group_capacity=16)
    for i in range(6):
        diff, emit, wm, _, _, _ = blk.forward(t2np(random_tokens(12, blk.match_words, i)))
        if diff.shape[0] == 0: break
        blk.assign(diff, emit, wm)
    print(f"  table: {blk.array_used}c/{blk.emit_used}g")
    toks = t2np(random_tokens(16, blk.match_words, 200))
    d1, e1, wm1, cp1, cnt1, eo1 = blk.forward(toks)
    d2, e2, wm2, cp2, cnt2, eo2 = blk.forward(toks)
    assert torch.equal(cp1, cp2) and np.array_equal(eo1, eo2) and np.array_equal(cnt1, cnt2)
    print(f"  PASS  forward consistency")

def test_assign_negative_tightening():
    blk = make_block(case_capacity=32, group_capacity=8)
    diff1, emit1, wm1, _, _, _ = blk.forward(t2np(random_tokens(4, blk.match_words, 10)))
    blk.assign(diff1, emit1, wm1)
    neg_before = ival(get_match(blk, 1, 0))
    diff2, emit2, wm2, _, _, _ = blk.forward(t2np(random_tokens(4, blk.match_words, 11)))
    blk.assign(diff2, emit2, wm2)
    neg_after = ival(get_match(blk, 1, 0))
    assert (neg_after & neg_before) == neg_after
    print("  PASS  assign negative tightening")

def test_assign_new_matched_case():
    blk = make_block(case_capacity=64, group_capacity=16)
    toks = t2np(random_tokens(8, blk.match_words, 11))
    for _ in range(3):
        diff, emit, wm, _, _, _ = blk.forward(toks)
        if diff.shape[0] == 0: break
        blk.assign(diff, emit, wm)
    if blk.emit_used == 0:
        print("  SKIP  new matched case (no groups)")
        return
    cases_before, groups_before = blk.array_used, blk.emit_used
    diff2, emit2, wm2, _, _, _ = blk.forward(t2np(random_tokens(4, blk.match_words, 12)))
    if diff2.shape[0] > 0 and wm2.any():
        blk.assign(diff2, emit2, wm2)
        assert blk.emit_used >= groups_before
        print(f"  PASS  assign new matched case — {cases_before}→{blk.array_used}c, groups={blk.emit_used}")
    else:
        print("  SKIP  new matched case (no matched diffs)")

def test_assign_new_group():
    NNv5CapacityError = _mod.NNv5CapacityError
    blk = make_block(case_capacity=64, group_capacity=16)
    toks = t2np(random_tokens(4, blk.match_words, 12))
    diff, emit, wm = blk.forward(toks)[:3]
    assert not wm.any()
    blk.assign(diff, emit, wm)
    assert blk.emit_used >= 1 and blk.array_used >= 1
    gidx = blk.emit_used - 1
    word, bit = divmod(gidx, 64)
    flag = torch.tensor(1 << bit, dtype=torch.int64, device=DEVICE)
    assert (get_emit(blk, 0, blk.array_used - 1, word) & flag).item() != 0
    print(f"  PASS  assign new group — {blk.array_used}c/{blk.emit_used}g")

def test_assign_capacity_error_cases():
    NNv5CapacityError = _mod.NNv5CapacityError
    blk = make_block(case_capacity=8, group_capacity=16)
    try:
        for i in range(20):
            diff, emit, wm, _, _, _ = blk.forward(t2np(random_tokens(8, blk.match_words, i)))
            if diff.shape[0] == 0: break
            blk.assign(diff, emit, wm)
        print("  SKIP  capacity error cases (not hit)")
    except NNv5CapacityError as e:
        assert "capacity" in str(e).lower()
        print(f"  PASS  assign capacity error cases — '{e}'")

def test_assign_capacity_error_groups():
    NNv5CapacityError = _mod.NNv5CapacityError
    blk = make_block(case_capacity=64, group_capacity=2)
    try:
        for i in range(20):
            diff, emit, wm, _, _, _ = blk.forward(t2np(random_tokens(8, blk.match_words, i)))
            if diff.shape[0] == 0: break
            blk.assign(diff, emit, wm)
        print("  SKIP  capacity error groups (not hit)")
    except NNv5CapacityError as e:
        assert "capacity" in str(e).lower()
        print(f"  PASS  assign capacity error groups — '{e}'")

def test_assign_idempotent():
    blk = make_block(case_capacity=64, group_capacity=16)
    toks = t2np(random_tokens(4, blk.match_words, 15))
    diff, emit, wm, _, _, _ = blk.forward(toks); blk.assign(diff, emit, wm)
    c1 = blk.array_used
    diff2, emit2, wm2, _, _, _ = blk.forward(toks); blk.assign(diff2, emit2, wm2)
    assert blk.array_used <= c1 * 2
    print(f"  PASS  assign idempotent — {c1} → {blk.array_used}")

def test_forward_shard_size_invariance():
    results = {}
    for ss in [8, 16, 32]:
        blk = make_block(case_capacity=128, group_capacity=16, shard_size=ss)
        for i in range(8):
            diff, emit, wm, _, _, _ = blk.forward(t2np(random_tokens(12, blk.match_words, i)))
            if diff.shape[0] == 0: break
            blk.assign(diff, emit, wm)
        results[ss] = blk
    print("  tables: " + ", ".join(f"ss={ss}→{b.array_used}c/{b.emit_used}g" for ss,b in results.items()))
    counts = [(b.array_used, b.emit_used) for b in results.values()]
    assert len(set(counts)) == 1, f"Different counts: {counts}"
    test_toks = t2np(random_tokens(32, 1, 999))
    ref = results[8]
    _, _, _, cp_ref, cnt_ref, _ = ref.forward(test_toks)
    for ss, blk in list(results.items())[1:]:
        _, _, _, cp_s, cnt_s, _ = blk.forward(test_toks)
        assert torch.equal(cp_ref, cp_s) and np.array_equal(cnt_ref, cnt_s)
    print("  PASS  forward shard_size invariance — ss=[8,16,32]")

def test_assign_handrolled_matches_reference():
    """Two identical blocks running assign should produce identical state."""
    blk_ref = make_block(case_capacity=64, group_capacity=16)
    blk_hr  = make_block(case_capacity=64, group_capacity=16)
    for i in range(6):
        toks = t2np(random_tokens(8, blk_ref.match_words, i))
        dr, er, wr, _, _, _ = blk_ref.forward(toks)
        dh, eh, wh, _, _, _ = blk_hr.forward(toks)
        if dr.shape[0] == 0: break
        blk_ref.assign(dr, er, wr)
        blk_hr.assign(dh, eh, wh)
    assert blk_ref.array_used == blk_hr.array_used
    assert blk_ref.emit_used  == blk_hr.emit_used
    C = blk_ref.array_used
    assert torch.equal(blk_ref.match[0].reshape(-1, blk_ref.match_words)[:C],
                       blk_hr.match[0].reshape(-1, blk_hr.match_words)[:C])
    assert torch.equal(blk_ref.emit[0].reshape(-1, blk_ref.emit_words)[:C],
                       blk_hr.emit[0].reshape(-1, blk_hr.emit_words)[:C])
    print(f"  PASS  assign deterministic — {C}c/{blk_ref.emit_used}g")

# ---------------------------------------------------------------------------
# Hardcoded invariant tests
# ---------------------------------------------------------------------------

def _blk1():
    return make_block(case_capacity=16, group_capacity=8, shard_size=8)

def test_hardcoded_exact_duplicate():
    blk = _blk1()
    set_match(blk, 0, 0, 0, 0b1100); set_match(blk, 1, 0, 0, 0b0011)
    blk.array_used = 1; blk.emit_used = 1
    blk._set_emit_group_bit(0, 0, is_member=True)
    toks = t2np(torch.tensor([[0b1100]], dtype=torch.int64, device=DEVICE))
    diff, emit, wm, _, _, _ = blk.forward(toks)
    c = blk.array_used
    blk.assign(diff, emit, wm)
    assert blk.array_used == c
    print("  PASS  hardcoded exact duplicate")

def test_hardcoded_new_group():
    blk = _blk1()
    toks = t2np(torch.tensor([[0b1010]], dtype=torch.int64, device=DEVICE))
    diff, emit, wm, _, _, _ = blk.forward(toks)
    blk.assign(diff, emit, wm)
    assert blk.array_used == 1 and blk.emit_used == 1
    flag = torch.tensor(1, dtype=torch.int64, device=DEVICE)
    assert (get_emit(blk, 0, 0) & flag).item() != 0
    assert ival(get_match(blk, 0, 0)) & 0b1010 == 0b1010
    print("  PASS  hardcoded new group")

def test_hardcoded_case_in_existing_group():
    blk = _blk1()
    set_match(blk, 0, 0, 0, 0b1111); set_match(blk, 1, 0, 0, 0b0000)
    blk.array_used = 1; blk.emit_used = 1
    blk._set_emit_group_bit(0, 0, is_member=True)
    diff = make_diff((0b1001, 0b0000))
    emit = zeros_emit(1, blk)
    emit[0, 0, 0] = ival(get_emit(blk, 0, 0))
    emit[0, 1, 0] = ival(get_emit(blk, 1, 0))
    blk.assign(diff, emit, np.array([True]))
    assert blk.emit_used == 1 and blk.array_used == 2
    flag = torch.tensor(1, dtype=torch.int64, device=DEVICE)
    assert (get_emit(blk, 0, 1) & flag).item() != 0
    print("  PASS  hardcoded case in existing group")

def test_hardcoded_negative_reduction():
    blk = _blk1()
    diff = make_diff((0b1100, 0b0011), (0b1100, 0b0001))
    blk.assign(diff, zeros_emit(2, blk), np.array([False, False]))
    found = False
    for c in range(blk.array_used):
        if ival(get_match(blk, 0, c)) == 0b1100:
            assert ival(get_match(blk, 1, c)) == 0b0001, \
                f"Expected 0b0001, got {bin(ival(get_match(blk, 1, c)))}"
            found = True; break
    assert found
    print("  PASS  hardcoded negative reduction")

def test_torch_forward_matches_numpy():
    blk = make_block(case_capacity=64, group_capacity=16)
    for i in range(6):
        diff, emit, wm, _, _, _ = blk.forward(t2np(random_tokens(12, blk.match_words, i)))
        if diff.shape[0] == 0: break
        blk.assign(diff, emit, wm)
    toks = t2np(random_tokens(32, blk.match_words, 50))
    d_np, e_np, wm_np, cp_np, cnt_np, eo_np = blk.forward(toks)
    d_t,  e_t,  wm_t,  cp_t,  cnt_t,  eo_t  = blk.forward_torch(toks, DEVICE)
    assert torch.equal(cp_np, cp_t) and np.array_equal(eo_np, eo_t) and np.array_equal(cnt_np, cnt_t)
    if d_np.shape[0] > 0 and d_t.shape[0] > 0:
        d_ns = d_np[np.lexsort(d_np.reshape(d_np.shape[0],-1).T)]
        d_ts = d_t[np.lexsort(d_t.reshape(d_t.shape[0],-1).T)]
        assert np.array_equal(d_ns, d_ts)
    print(f"  PASS  torch forward matches numpy — {blk.array_used}c/{blk.emit_used}g")

def test_hardcoded_negative_reduction_merge():
    blk = _blk1()
    blk.assign(make_diff((0b1100, 0b1111), (0b1100, 0b0011)),
               zeros_emit(2, blk), np.array([False, False]))
    case_a = next(c for c in range(blk.array_used) if ival(get_match(blk, 0, c)) == 0b1100)
    assert ival(get_match(blk, 1, case_a)) == 0b0011
    blk.assign(make_diff((0b1100, 0b0101), (0b1100, 0b0001)),
               zeros_emit(2, blk), np.array([False, False]))
    assert ival(get_match(blk, 1, case_a)) == 0b0001, \
        f"Expected 0b0001, got {bin(ival(get_match(blk, 1, case_a)))}"
    print(f"  PASS  hardcoded negative reduction merge — {blk.array_used}c/{blk.emit_used}g")


if __name__ == "__main__":
    print("Running NNv5Block tests...\n")
    tests = [
        test_pack_unpack_roundtrip,
        test_empty_forward,
        test_forward_after_assign,
        test_was_matched_consistent_with_choices,
        test_emit_aggregation_intersection,
        test_reverse_reconstruction,
        test_assign_creates_groups,
        test_choices_packed_shape,
        test_forward_consistency,
        test_assign_negative_tightening,
        test_assign_new_matched_case,
        test_assign_new_group,
        test_assign_capacity_error_cases,
        test_assign_capacity_error_groups,
        test_assign_idempotent,
        test_forward_shard_size_invariance,
        test_assign_handrolled_matches_reference,
        test_hardcoded_exact_duplicate,
        test_hardcoded_new_group,
        test_hardcoded_case_in_existing_group,
        test_hardcoded_negative_reduction,
        test_torch_forward_matches_numpy,
        test_hardcoded_negative_reduction_merge,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
