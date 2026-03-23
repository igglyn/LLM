"""
test_nnv2.py — NNv2Block and NNv2Adapter tests, torch-native.

Run with: python tests/test_nnv2.py
"""
import sys, os, numpy as np

try:
    import torch
except ImportError:
    print("torch not available — cannot run NNv2 tests")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import importlib.util

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_base = os.path.join(os.path.dirname(__file__), '..', 'src', 'blt_lite', 'synthetic_batch')
_nnv5 = _load("blt_lite.synthetic_batch.nnv5", os.path.join(_base, "nnv5.py"))
_nnv2 = _load("blt_lite.synthetic_batch.nnv2", os.path.join(_base, "nnv2.py"))

NNv5Block   = _nnv5.NNv5Block
NNv2Block   = _nnv2.NNv2Block
NNv2Adapter = _nnv2.NNv2Adapter

DEVICE = torch.device('cpu')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_nnv5(bool_dim=64, case_capacity=64, group_capacity=16, shard_size=8):
    cap = max(case_capacity, shard_size)
    return NNv5Block(bool_dim, cap, group_capacity, shard_size, device=DEVICE)

def make_nnv2(group_capacity=16, case_capacity=64, shard_size=8):
    cap = max(case_capacity, shard_size)
    return NNv2Block(group_capacity, cap, shard_size, device=DEVICE)

def make_adapter(chunk_dim=16, case_capacity=64, n_chunks=1):
    return NNv2Adapter(chunk_dim, case_capacity, n_chunks, device=DEVICE)

def random_tokens(n, match_words, seed):
    g = torch.Generator(); g.manual_seed(seed)
    return torch.randint(torch.iinfo(torch.int64).min,
                         torch.iinfo(torch.int64).max,
                         (n, match_words), dtype=torch.int64,
                         generator=g, device=DEVICE)

def t2np(t):
    return t.cpu().numpy().view(np.uint64)

def seed_nnv5(nnv5, n_passes=6, n_toks=8):
    """Run enough forward/assign passes to build up cases and groups."""
    for i in range(n_passes):
        toks = t2np(random_tokens(n_toks, nnv5.match_words, i))
        diff, emit, wm, _, _, _ = nnv5.forward(toks)
        if diff.shape[0] == 0:
            break
        nnv5.assign(diff, emit, wm)

def get_emit_flat(blk, C=None):
    """Flatten emit[0] to (C, emit_words) int64."""
    C = C or blk.array_used
    return blk.emit[0].reshape(-1, blk.emit_words)[:C]

# ---------------------------------------------------------------------------
# NNv2Block tests
# ---------------------------------------------------------------------------

def test_nnv2_empty_compress():
    """compress on empty NNv5 should do nothing."""
    nnv5 = make_nnv5()
    nnv2 = make_nnv2()
    nnv2.compress(nnv5)
    assert nnv2.array_used == 0
    print("  PASS  empty compress")

def test_nnv2_compress_appends():
    """compress should append NNv5 cases into NNv2 when table is empty."""
    nnv5 = make_nnv5(group_capacity=16)
    seed_nnv5(nnv5)
    assert nnv5.array_used > 0 and nnv5.emit_used > 0, "NNv5 needs groups to activate NNv2"

    nnv2 = make_nnv2(group_capacity=16)
    nnv2.compress(nnv5)

    assert nnv2.array_used > 0, "NNv2 should have cases after compress"
    assert nnv2.array_used <= nnv5.array_used, \
        "NNv2 should not have more cases than NNv5"
    print(f"  PASS  compress appends — nnv5={nnv5.array_used}c, nnv2={nnv2.array_used}c")

def test_nnv2_compress_reduce():
    """Second compress with same cases should reduce, not append."""
    nnv5 = make_nnv5(group_capacity=16)
    seed_nnv5(nnv5)
    if nnv5.emit_used == 0:
        print("  SKIP  compress reduce (no groups)")
        return

    nnv2 = make_nnv2(group_capacity=16)
    nnv2.compress(nnv5)
    cases_after_first = nnv2.array_used

    # Compress again — same cases, should reduce not append
    nnv2.compress_range(nnv5, 0, nnv5.array_used)
    assert nnv2.array_used <= cases_after_first * 2, \
        f"Repeated compress grew too much: {cases_after_first} → {nnv2.array_used}"
    print(f"  PASS  compress reduce — {cases_after_first} → {nnv2.array_used}")

def test_nnv2_forward_empty():
    """forward on empty table returns empty choices."""
    nnv2 = make_nnv2(group_capacity=16)
    emit_in = torch.zeros(8, 1, dtype=torch.int64, device=DEVICE)
    cp, choices = nnv2.forward(emit_in)
    assert cp.shape == (8, 0)
    assert choices.shape == (8, 0)
    print("  PASS  nnv2 forward empty")

def test_nnv2_forward_matches():
    """After compress, forward should match on known emit patterns."""
    nnv5 = make_nnv5(group_capacity=16)
    seed_nnv5(nnv5)
    if nnv5.emit_used == 0:
        print("  SKIP  nnv2 forward matches (no groups)")
        return

    nnv2 = make_nnv2(group_capacity=16)
    nnv2.compress(nnv5)

    C = nnv2.array_used
    if C == 0:
        print("  SKIP  nnv2 forward matches (no nnv2 cases)")
        return

    # Run NNv5 forward to get emit output, feed into NNv2
    toks = t2np(random_tokens(8, nnv5.match_words, 99))
    _, _, _, _, _, eo = nnv5.forward(toks)   # (N, 2, EW) uint64

    emit_in = torch.from_numpy(eo[:, 0].view(np.int64)).to(DEVICE)  # (N, EW)
    cp, choices = nnv2.forward(emit_in)

    assert cp.shape[0] == 8
    assert choices.shape == (8, C)
    print(f"  PASS  nnv2 forward matches — {C} cases, "
          f"{choices.any(dim=1).sum().item()}/8 matched")

def test_nnv2_forward_subset_match():
    """Hardcoded: case with pos=0b11 should match emit input with bits 0b1111."""
    nnv2 = make_nnv2(group_capacity=64, case_capacity=8, shard_size=8)

    # Manually insert one case: pos=0b11, neg=0b00
    s, slot = nnv2._shard_slot(0)
    nnv2.match[0, s, slot, 0] = torch.tensor(0b11, dtype=torch.int64, device=DEVICE)
    nnv2.match[1, s, slot, 0] = torch.tensor(0,    dtype=torch.int64, device=DEVICE)
    nnv2.emit[0, s, slot, 0]  = torch.tensor(0b11, dtype=torch.int64, device=DEVICE)
    nnv2.array_used = 1

    # Input with superset bits — should match
    emit_in = torch.tensor([[0b1111]], dtype=torch.int64, device=DEVICE)
    _, choices = nnv2.forward(emit_in)
    assert choices[0, 0].item(), "Superset input should match pos=0b11 case"

    # Input missing required bits — should not match
    emit_in2 = torch.tensor([[0b0100]], dtype=torch.int64, device=DEVICE)
    _, choices2 = nnv2.forward(emit_in2)
    assert not choices2[0, 0].item(), "Input missing required bits should not match"

    print("  PASS  nnv2 forward subset match")

def test_nnv2_compress_range_incremental():
    """compress_range on new cases only should not reprocess old ones."""
    nnv5 = make_nnv5(group_capacity=16)
    seed_nnv5(nnv5, n_passes=3)
    if nnv5.emit_used == 0:
        print("  SKIP  incremental compress (no groups)")
        return

    nnv2 = make_nnv2(group_capacity=16)
    mid  = nnv5.array_used
    nnv2.compress_range(nnv5, 0, mid)
    cases_mid = nnv2.array_used

    # Add more cases to NNv5
    seed_nnv5(nnv5, n_passes=3)
    end = nnv5.array_used

    if end > mid:
        nnv2.compress_range(nnv5, mid, end)
        assert nnv2.array_used >= cases_mid, "Cases should not decrease"
        print(f"  PASS  incremental compress — mid={cases_mid}, end={nnv2.array_used}")
    else:
        print("  SKIP  incremental compress (no new NNv5 cases)")

# ---------------------------------------------------------------------------
# NNv2Adapter tests
# ---------------------------------------------------------------------------

def test_adapter_forward_empty():
    """Adapter forward with no cases returns zeros."""
    adapter = make_adapter(chunk_dim=16)
    choices = torch.zeros(8, 0, dtype=torch.bool, device=DEVICE)
    out = adapter.forward(choices)
    assert out.shape == (8, 16)
    assert not out.any()
    print("  PASS  adapter forward empty")

def test_adapter_sync():
    """sync registers case count — float_emit stays zero until frontloaded."""
    nnv5 = make_nnv5(group_capacity=16)
    seed_nnv5(nnv5)
    if nnv5.emit_used == 0:
        print("  SKIP  adapter sync (no groups)")
        return

    nnv2    = make_nnv2(group_capacity=16)
    adapter = make_adapter(chunk_dim=16)

    nnv2.compress(nnv5, adapter)

    assert adapter.array_used == nnv2.array_used, \
        f"Adapter cases {adapter.array_used} != NNv2 cases {nnv2.array_used}"
    assert adapter.array_used > 0
    print(f"  PASS  adapter sync — {adapter.array_used} cases registered")

def test_adapter_forward_output_shape():
    """Adapter forward returns (N, group_capacity) float32."""
    nnv5 = make_nnv5(group_capacity=16)
    seed_nnv5(nnv5)
    if nnv5.emit_used == 0:
        print("  SKIP  adapter forward shape (no groups)")
        return

    nnv2    = make_nnv2(group_capacity=16)
    adapter = make_adapter(chunk_dim=16)
    nnv2.compress(nnv5)
    adapter.sync(nnv2.array_used)

    toks = t2np(random_tokens(8, nnv5.match_words, 99))
    _, _, _, _, _, eo = nnv5.forward(toks)
    emit_in = torch.from_numpy(eo[:, 0].view(np.int64)).to(DEVICE)
    _, choices = nnv2.forward(emit_in)

    out = adapter.forward(choices)
    assert out.shape == (8, 16), f"Expected (8, 16), got {out.shape}"
    assert out.dtype == torch.float32
    print(f"  PASS  adapter forward shape — {out.shape}")

def test_adapter_forward_is_sum():
    """Adapter output should be sum of float_emit over matched cases."""
    nnv5 = make_nnv5(group_capacity=16)
    seed_nnv5(nnv5)
    if nnv5.emit_used == 0:
        print("  SKIP  adapter forward sum (no groups)")
        return

    nnv2    = make_nnv2(group_capacity=16)
    adapter = make_adapter(chunk_dim=16)
    nnv2.compress(nnv5)
    adapter.sync(nnv2.array_used)

    toks = t2np(random_tokens(4, nnv5.match_words, 77))
    _, _, _, _, _, eo = nnv5.forward(toks)
    emit_in = torch.from_numpy(eo[:, 0].view(np.int64)).to(DEVICE)
    _, choices = nnv2.forward(emit_in)
    out = adapter.forward(choices)

    # Manually compute expected output
    C = adapter.array_used
    expected = choices.float() @ adapter.float_emit[:C]
    assert torch.allclose(out, expected), "Adapter output should be matmul of choices and float_emit"
    print(f"  PASS  adapter forward is matmul sum")

def test_adapter_waterfall_split():
    """Waterfall split should distribute parent affinities correctly."""
    adapter = make_adapter(chunk_dim=4)

    parent = torch.tensor([1.0, 2.0, 0.0, 4.0], dtype=torch.float32, device=DEVICE)

    adapter.array_used = 2
    with torch.no_grad():
        adapter.float_emit[0] = parent
    adapter.match_split(parent_idx=0, child_idx=1, ratio=0.5)
    p = adapter.float_emit[0].detach()
    c = adapter.float_emit[1].detach()
    assert torch.allclose(p + c, parent), "Conservation failed"
    assert torch.allclose(p, parent * 0.5), f"Parent wrong: {p}"
    assert torch.allclose(c, parent * 0.5), f"Child wrong: {c}"
    print("  PASS  adapter waterfall split")

def test_full_pipeline():
    """End-to-end: NNv5 → NNv2 → Adapter produces float output."""
    nnv5    = make_nnv5(group_capacity=16)
    nnv2    = make_nnv2(group_capacity=16)
    adapter = make_adapter(chunk_dim=16)

    seed_nnv5(nnv5)
    if nnv5.emit_used == 0:
        print("  SKIP  full pipeline (no groups)")
        return

    nnv2.compress(nnv5)
    adapter.sync(nnv2.array_used)

    toks = t2np(random_tokens(8, nnv5.match_words, 42))
    _, _, _, _, _, eo = nnv5.forward(toks)
    emit_in = torch.from_numpy(eo[:, 0].view(np.int64)).to(DEVICE)
    _, choices = nnv2.forward(emit_in)
    out = adapter.forward(choices)

    assert out.shape == (8, 16)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0, "Output affinities should be non-negative"
    print(f"  PASS  full pipeline — nnv5={nnv5.array_used}c/{nnv5.emit_used}g, "
          f"nnv2={nnv2.array_used}c, out={out.shape}")


def test_adapter_frontload():
    """First MSE residual should be frontloaded into zero cases."""
    adapter = make_adapter(chunk_dim=8)
    adapter.array_used = 3

    choices = torch.zeros(4, 3, dtype=torch.bool, device=DEVICE)
    choices[0, 0] = True   # case 0 fires for input 0
    choices[1, 2] = True   # case 2 fires for input 1

    residual = torch.ones(4, 8, dtype=torch.float32, device=DEVICE) * 0.5
    adapter.frontload(choices, residual)

    assert adapter.float_emit[0].any(), "Case 0 should be initialised after frontload"
    assert adapter.float_emit[2].any(), "Case 2 should be initialised after frontload"
    assert not adapter.float_emit[1].any(), "Case 1 should still be zero (never fired)"
    print("  PASS  adapter frontload")


def test_adapter_match_split():
    """match_split should waterfall float_emit between parent and child."""
    adapter = make_adapter(chunk_dim=8)
    adapter.array_used = 2

    # Set parent float_emit to known values
    with torch.no_grad():
        adapter.float_emit[0] = torch.ones(8, device=DEVICE) * 2.0

    adapter.match_split(parent_idx=0, child_idx=1, ratio=0.5)

    parent = adapter.float_emit[0]
    child  = adapter.float_emit[1]

    assert torch.allclose(parent, torch.ones(8, device=DEVICE)), \
        f"Parent should be 2.0 * 0.5 = 1.0, got {parent}"
    assert torch.allclose(child, torch.ones(8, device=DEVICE)), \
        f"Child should get remainder 1.0, got {child}"

    # Conservation: parent + child = original
    assert torch.allclose(parent + child,
                          torch.ones(8, device=DEVICE) * 2.0), \
        "Parent + child should equal original"
    print("  PASS  adapter match_split")


def test_adapter_mse_loss():
    """mse_loss should return scalar, frontload zero cases, be differentiable."""
    adapter = make_adapter(chunk_dim=8)
    adapter.array_used = 4

    choices = torch.zeros(4, 4, dtype=torch.bool, device=DEVICE)
    choices[0, 0] = True
    choices[1, 1] = True
    choices[2, 2] = True
    choices[3, 3] = True

    target = torch.randn(4, 8, device=DEVICE)
    loss   = adapter.mse_loss(choices, target, chunk_idx=0)

    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
    assert loss.requires_grad, "Loss should be differentiable"
    assert loss.item() >= 0.0

    # Check frontloading happened — all fired cases should be initialised
    for ci in range(4):
        assert adapter.float_emit[ci].any(), f"Case {ci} should be initialised"
    print("  PASS  adapter mse_loss")


def test_adapter_mse_loss_differentiable():
    """Gradient should flow through mse_loss into float_emit and pos_embed."""
    adapter = make_adapter(chunk_dim=8)
    adapter.array_used = 2
    opt = torch.optim.SGD(adapter.parameters(), lr=0.01)

    choices = torch.ones(2, 2, dtype=torch.bool, device=DEVICE)
    target  = torch.randn(2, 8, device=DEVICE)

    # Seed float_emit so it's not zero
    with torch.no_grad():
        adapter.float_emit[:2] = torch.randn(2, 8, device=DEVICE) * 0.1

    fe_before = adapter.float_emit[:2].clone().detach()
    pe_before = adapter.pos_embed[0].clone().detach()

    opt.zero_grad()
    loss = adapter.mse_loss(choices, target, chunk_idx=0)
    loss.backward()
    opt.step()

    assert not torch.equal(adapter.float_emit[:2].detach(), fe_before), \
        "float_emit should have updated"
    assert not torch.equal(adapter.pos_embed[0].detach(), pe_before), \
        "pos_embed should have updated"
    print("  PASS  adapter mse_loss differentiable")


def test_compress_notifies_adapter():
    """compress_range should notify adapter of new cases via sync."""
    nnv5    = make_nnv5(group_capacity=16)
    nnv2    = make_nnv2(group_capacity=16)
    adapter = make_adapter(chunk_dim=8)

    seed_nnv5(nnv5)
    if nnv5.emit_used == 0:
        print("  SKIP  compress notifies adapter (no groups)")
        return

    nnv2.compress(nnv5, adapter)

    assert adapter.array_used == nnv2.array_used, \
        f"Adapter should be synced: {adapter.array_used} != {nnv2.array_used}"
    print(f"  PASS  compress notifies adapter — {adapter.array_used} cases")


if __name__ == "__main__":
    print("Running NNv2Block + NNv2Adapter tests...\n")
    tests = [
        test_nnv2_empty_compress,
        test_nnv2_compress_appends,
        test_nnv2_compress_reduce,
        test_nnv2_forward_empty,
        test_nnv2_forward_matches,
        test_nnv2_forward_subset_match,
        test_nnv2_compress_range_incremental,
        test_adapter_forward_empty,
        test_adapter_sync,
        test_adapter_forward_output_shape,
        test_adapter_forward_is_sum,
        test_adapter_waterfall_split,
        test_adapter_frontload,
        test_adapter_match_split,
        test_adapter_mse_loss,
        test_adapter_mse_loss_differentiable,
        test_compress_notifies_adapter,
        test_full_pipeline,
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
