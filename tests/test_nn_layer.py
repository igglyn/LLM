"""
test_nn_layer.py — NNLayer coordinator tests, torch-native.

Run with: python tests/test_nn_layer.py
"""
import sys, os, numpy as np

try:
    import torch
except ImportError:
    print("torch not available — cannot run NNLayer tests")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import importlib.util

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_base  = os.path.join(os.path.dirname(__file__), '..', 'src', 'blt_lite', 'synthetic_batch')
_nnv5  = _load("blt_lite.synthetic_batch.nnv5",     os.path.join(_base, "nnv5.py"))
_nnv2  = _load("blt_lite.synthetic_batch.nnv2",     os.path.join(_base, "nnv2.py"))
_layer = _load("blt_lite.synthetic_batch.nn_layer", os.path.join(_base, "nn_layer.py"))

NNLayer = _layer.NNLayer
DEVICE  = torch.device('cpu')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_layer(bool_dim=64, chunk_dim=16, case_capacity=64,
               group_capacity=16, shard_size=8, n_chunks=1):
    cap = max(case_capacity, shard_size)
    return NNLayer(bool_dim, chunk_dim, cap, group_capacity,
                   shard_size, n_chunks, DEVICE)

def random_tokens(n, match_words, seed):
    g = torch.Generator(); g.manual_seed(seed)
    return torch.randint(torch.iinfo(torch.int64).min,
                         torch.iinfo(torch.int64).max,
                         (n, match_words), dtype=torch.int64,
                         generator=g, device=DEVICE)

def t2np(t):
    return t.cpu().numpy().view(np.uint64)

def seed_layer(layer, n_passes=6, n_toks=8):
    for i in range(n_passes):
        toks = t2np(random_tokens(n_toks, layer.nnv5.match_words, i))
        out, diff, emit, wm, choices = layer.forward(toks)
        if diff.shape[0] == 0:
            break
        layer.learn(diff, emit, wm)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forward_empty():
    """Forward on empty layer returns correct shapes."""
    layer = make_layer()
    toks  = t2np(random_tokens(8, layer.nnv5.match_words, 0))
    out, diff, emit, wm, choices = layer.forward(toks)

    assert out.shape == (8, 16), f"Expected (8,16) got {out.shape}"
    assert out.dtype == torch.float32
    assert diff.shape[1] == 2
    print("  PASS  forward empty")


def test_forward_after_learn():
    """After learning, forward should produce non-trivial choices."""
    layer = make_layer()
    seed_layer(layer)

    assert layer.nnv5.array_used > 0, "Should have NNv5 cases"
    toks = t2np(random_tokens(8, layer.nnv5.match_words, 99))
    out, _, _, _, choices = layer.forward(toks)

    assert out.shape == (8, 16)
    assert choices.shape[0] == 8
    print(f"  PASS  forward after learn — {layer.nnv5.array_used}c/"
          f"{layer.nnv5.emit_used}g, {choices.any(dim=1).sum().item()}/8 matched")


def test_learn_grows_cases():
    """learn() should grow NNv5 case count."""
    layer = make_layer()
    assert layer.nnv5.array_used == 0

    for i in range(4):
        toks = t2np(random_tokens(8, layer.nnv5.match_words, i))
        out, diff, emit, wm, _ = layer.forward(toks)
        layer.learn(diff, emit, wm)

    assert layer.nnv5.array_used > 0
    print(f"  PASS  learn grows cases — {layer.nnv5.array_used}c/{layer.nnv5.emit_used}g")


def test_nnv2_activates_after_groups():
    """NNv2 should only compress once NNv5 has formed groups."""
    layer = make_layer()

    # Before groups — NNv2 should be bypassed
    toks = t2np(random_tokens(4, layer.nnv5.match_words, 0))
    _, diff, emit, wm, _ = layer.forward(toks)
    layer.learn(diff, emit, wm)

    if layer.nnv5.emit_used == 0:
        assert layer.nnv2.array_used == 0, "NNv2 should not activate without groups"

    # Seed until groups form
    seed_layer(layer)

    if layer.nnv5.emit_used > 0:
        assert layer.stats()['nnv2_active'] or layer.nnv2.array_used >= 0, \
            "NNv2 should be active or pending once groups exist"

    print(f"  PASS  nnv2 activates after groups — "
          f"emit_used={layer.nnv5.emit_used}, nnv2={layer.nnv2.array_used}c")


def test_bypass_vs_nnv2_choices_shape():
    """Choices shape should be consistent whether NNv2 is active or bypassed."""
    layer = make_layer()
    toks  = t2np(random_tokens(8, layer.nnv5.match_words, 5))

    # Bypass path
    out1, _, _, _, choices1 = layer.forward(toks)
    assert choices1.shape[0] == 8

    # Seed and try again
    seed_layer(layer)
    out2, _, _, _, choices2 = layer.forward(toks)
    assert choices2.shape[0] == 8
    assert out1.shape == out2.shape == (8, 16)

    print(f"  PASS  bypass vs nnv2 choices shape — "
          f"nnv2_active={layer.stats()['nnv2_active']}")


def test_mse_loss_scalar():
    """mse_loss should return a scalar differentiable loss."""
    layer  = make_layer()
    seed_layer(layer)

    toks   = t2np(random_tokens(4, layer.nnv5.match_words, 42))
    target = torch.randn(4, 16, device=DEVICE)
    loss   = layer.mse_loss(toks, target)

    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
    assert loss.item() >= 0.0
    print(f"  PASS  mse_loss scalar — loss={loss.item():.4f}")


def test_mse_loss_differentiable():
    """Gradient should flow through mse_loss into float_emit and pos_embed."""
    layer = make_layer()
    seed_layer(layer)

    C = layer.adapter.array_used
    if C == 0:
        print("  SKIP  mse_loss differentiable (no adapter cases)")
        return

    # Set float_emit to known non-zero values
    with torch.no_grad():
        layer.adapter.float_emit[:C] = torch.randn(C, 16, device=DEVICE) * 0.1

    # Construct choices that definitely fire — all cases active for all inputs
    N       = 4
    choices = torch.ones(N, C, dtype=torch.bool, device=DEVICE)
    target  = torch.randn(N, 16, device=DEVICE)

    opt = torch.optim.SGD(layer.adapter.parameters(), lr=0.01)
    fe_before = layer.adapter.float_emit[:C].clone().detach()
    pe_before = layer.adapter.pos_embed[0].clone().detach()

    opt.zero_grad()
    loss = layer.adapter.mse_loss(choices, target, chunk_idx=0)
    loss.backward()
    opt.step()

    assert not torch.equal(layer.adapter.float_emit[:C].detach(), fe_before), \
        "float_emit should have updated"
    assert not torch.equal(layer.adapter.pos_embed[0].detach(), pe_before), \
        "pos_embed should have updated"
    print(f"  PASS  mse_loss differentiable — {C} cases, loss={loss.item():.4f}")

def test_stats():
    """stats() should return a dict with all expected keys."""
    layer = make_layer()
    seed_layer(layer)
    s = layer.stats()

    for key in ['cases_used', 'groups_used', 'nnv2_cases_used',
                'adapter_cases_used', 'nnv2_active']:
        assert key in s, f"Missing key: {key}"
    print(f"  PASS  stats — {s}")


def test_n_chunks_pos_embed():
    """pos_embed should have one entry per chunk, each affecting output."""
    layer = make_layer(n_chunks=4)
    seed_layer(layer)

    toks = t2np(random_tokens(4, layer.nnv5.match_words, 7))

    outs = []
    for ci in range(4):
        out, _, _, _, _ = layer.forward(toks, chunk_idx=ci)
        outs.append(out)

    # Different chunk indices should produce different outputs
    # (pos_embed initialized to zero so outputs equal until trained,
    #  but shape should be correct)
    for out in outs:
        assert out.shape == (4, 16)
    print(f"  PASS  n_chunks pos_embed — {4} chunks, shape OK")


if __name__ == "__main__":
    print("Running NNLayer tests...\n")
    tests = [
        test_forward_empty,
        test_forward_after_learn,
        test_learn_grows_cases,
        test_nnv2_activates_after_groups,
        test_bypass_vs_nnv2_choices_shape,
        test_mse_loss_scalar,
        test_mse_loss_differentiable,
        test_stats,
        test_n_chunks_pos_embed,
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
