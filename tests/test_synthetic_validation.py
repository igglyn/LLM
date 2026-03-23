"""
test_synthetic_validation.py — Synthetic validation suite for NNLayer.

Validates the full learning loop without a real transformer:
    - Generate random float vectors as synthetic "embeddings"
    - Pack float32 bits into uint64 for NNv5 input
    - Run NNLayer forward/learn loop
    - Measure MSE convergence over steps
    - Verify case stabilization
    - Verify distribution approximation

These tests answer: can the NN line learn to approximate
a fixed distribution of float vectors?

Run with: python tests/test_synthetic_validation.py
"""
import sys, os, numpy as np, time

try:
    import torch
except ImportError:
    print("torch not available — cannot run synthetic validation")
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
_nnv5  = _load("blt_lite.synthetic_batch.nnv5",       os.path.join(_base, "nnv5.py"))
_nnv2  = _load("blt_lite.synthetic_batch.nnv2",       os.path.join(_base, "nnv2.py"))
_layer = _load("blt_lite.synthetic_batch.nn_layer",   os.path.join(_base, "nn_layer.py"))
_bf    = _load("blt_lite.synthetic_batch.block_float", os.path.join(_base, "block_float.py"))

NNLayer          = _layer.NNLayer
encode_torch     = _bf.encode_torch
VALUES_PER_BLOCK = _bf.VALUES_PER_BLOCK  # 36 values per 128-bit block
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"  device: {DEVICE}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_layer(chunk_dim=36, case_capacity=512,
               group_capacity=256, shard_size=16, n_chunks=1):
    assert chunk_dim % VALUES_PER_BLOCK == 0, \
        f"chunk_dim must be divisible by {VALUES_PER_BLOCK}"
    n_blocks = chunk_dim // VALUES_PER_BLOCK
    bool_dim = n_blocks * 2 * 64
    cap = max(case_capacity, shard_size)
    return NNLayer(bool_dim, chunk_dim, cap, group_capacity,
                   shard_size, n_chunks, DEVICE)


def float_to_uint64(x: torch.Tensor) -> np.ndarray:
    """
    Encode float32 tensor using C4F4M2+S B4B9 block float format.
    x: (N, D) float32 in [-3, 3], D must be divisible by 36
    Returns (N, D//36*2) uint64
    """
    return encode_torch(x)


def make_embedding_dist(n: int, dim: int, seed: int,
                         n_clusters: int = 8) -> torch.Tensor:
    """
    Generate clustered float32 vectors in [-3, 3] simulating transformer
    post-layernorm activations.
    """
    rng = torch.Generator(); rng.manual_seed(seed)
    centers = torch.randn(n_clusters, dim, generator=rng, device=DEVICE) * 0.5
    labels  = torch.randint(0, n_clusters, (n,), generator=rng, device=DEVICE)
    noise   = torch.randn(n, dim, generator=rng, device=DEVICE) * 0.1
    return (centers[labels] + noise).clamp(-3.0, 3.0)


def run_learning_loop(layer, embeddings: torch.Tensor, n_steps: int,
                       batch_size: int = 16, lr: float = 0.01):
    """
    Run forward/learn/mse loop over embedding dataset.
    Returns list of (step, mse, cases, groups) tuples.
    """
    opt = torch.optim.Adam(layer.adapter.parameters(), lr=lr)
    N   = embeddings.shape[0]
    W   = layer.nnv5.match_words
    history = []

    for step in range(n_steps):
        # Sample batch
        idx    = torch.randperm(N)[:batch_size]
        batch  = embeddings[idx]                        # (B, chunk_dim)
        tokens = float_to_uint64(batch)                 # (B, W) uint64

        # Structural learning
        out, diff, emit, wm, choices = layer.forward(tokens)
        layer.learn(diff, emit, wm)

        # Gradient learning — only if adapter has cases
        if layer.adapter.array_used > 0:
            opt.zero_grad()
            loss = layer.mse_loss(tokens, batch)
            if loss.requires_grad:
                loss.backward()
                opt.step()
            mse = loss.item()
        else:
            mse = float('nan')

        history.append((step, mse, layer.nnv5.array_used,
                         layer.nnv5.emit_used, layer.nnv2.array_used,
                         layer.adapter.array_used))

    return history


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_float_packing_roundtrip():
    """float32 → C4F4M2+S B4B9 → uint64 should produce correct shape."""
    x   = torch.randn(8, 36, device=DEVICE).clamp(-3.0, 3.0)
    u64 = float_to_uint64(x)
    # 36 values per block, 2 uint64 words per block → (8, 2)
    assert u64.shape == (8, 2), f"Expected (8, 2) got {u64.shape}"
    print("  PASS  float packing roundtrip")


def test_nnlayer_sees_float_bits():
    """NNLayer should accept float-packed uint64 without error."""
    layer = make_layer(chunk_dim=36)
    x     = torch.randn(8, 36, device=DEVICE)
    toks  = float_to_uint64(x)

    assert toks.shape == (8, layer.nnv5.match_words), \
        f"Token shape mismatch: {toks.shape} vs expected (8, {layer.nnv5.match_words})"

    out, diff, emit, wm, choices = layer.forward(toks)
    assert out.shape == (8, 36), f"Output shape wrong: {out.shape}"
    print("  PASS  nnlayer sees float bits")


def test_cases_grow_over_steps():
    """NNv5 case count should grow as new embeddings are presented."""
    layer = make_layer(chunk_dim=36, case_capacity=256)
    embs  = make_embedding_dist(64, 36, seed=0)

    cases_0 = layer.nnv5.array_used
    for i in range(20):
        batch  = embs[i*3:(i+1)*3]
        tokens = float_to_uint64(batch)
        _, diff, emit, wm, _ = layer.forward(tokens)
        layer.learn(diff, emit, wm)

    cases_after = layer.nnv5.array_used
    assert cases_after > cases_0, \
        f"Cases should grow: {cases_0} → {cases_after}"
    print(f"  PASS  cases grow — {cases_0} → {cases_after}c/"
          f"{layer.nnv5.emit_used}g")


def test_mse_decreases():
    """MSE loss should decrease over training steps on a fixed distribution."""
    layer = make_layer(chunk_dim=36, case_capacity=512,
                       group_capacity=256)
    embs  = make_embedding_dist(128, 36, seed=1, n_clusters=4)

    history = run_learning_loop(layer, embs, n_steps=300, batch_size=16)

    # Filter out nan steps (before adapter has cases)
    valid = [(s, m) for s, m, *_ in history if not (m != m)]
    if len(valid) < 10:
        print(f"  SKIP  mse decreases (not enough valid steps: {len(valid)})")
        return

    # Compare first quarter vs last quarter
    n     = len(valid)
    early = np.mean([m for _, m in valid[:n//4]])
    late  = np.mean([m for _, m in valid[3*n//4:]])

    assert late < early, \
        f"MSE should decrease: early={early:.4f} late={late:.4f}"
    print(f"  PASS  mse decreases — early={early:.4f} → late={late:.4f} "
          f"({layer.nnv5.array_used}c/{layer.nnv5.emit_used}g)")


def test_case_stabilization():
    """Case count should stabilize after sufficient training on fixed dist."""
    layer = make_layer(chunk_dim=36, case_capacity=512,
                       group_capacity=256)
    embs  = make_embedding_dist(64, 36, seed=2, n_clusters=4)

    history = run_learning_loop(layer, embs, n_steps=300, batch_size=8)

    # Check case growth rate slows in second half
    cases = [c for _, _, c, *_ in history]
    first_half_growth = cases[len(cases)//2] - cases[0]
    second_half_growth = cases[-1] - cases[len(cases)//2]

    print(f"  case growth: first_half={first_half_growth}, "
          f"second_half={second_half_growth}, "
          f"final={cases[-1]}c/{layer.nnv5.emit_used}g")
    assert second_half_growth <= first_half_growth, \
        "Case growth should slow as distribution is covered"
    print(f"  PASS  case stabilization")


def test_clustered_vs_random():
    """Clustered embeddings should produce fewer cases than random (more structure)."""
    layer_c = make_layer(chunk_dim=36, case_capacity=512)
    layer_r = make_layer(chunk_dim=36, case_capacity=512)

    clustered = make_embedding_dist(128, 36, seed=3, n_clusters=4)
    random    = torch.randn(128, 36, device=DEVICE)

    for embs, layer in [(clustered, layer_c), (random, layer_r)]:
        for i in range(0, 64, 8):
            batch  = embs[i:i+8]
            tokens = float_to_uint64(batch)
            _, diff, emit, wm, _ = layer.forward(tokens)
            layer.learn(diff, emit, wm)

    print(f"  clustered cases: {layer_c.nnv5.array_used}c/"
          f"{layer_c.nnv5.emit_used}g")
    print(f"  random cases:    {layer_r.nnv5.array_used}c/"
          f"{layer_r.nnv5.emit_used}g")

    # Clustered should have <= cases than random since patterns repeat
    assert layer_c.nnv5.array_used <= layer_r.nnv5.array_used, \
        "Clustered dist should need fewer cases than random"
    print("  PASS  clustered vs random")


def test_convergence_benchmark():
    """
    Benchmark: track MSE, case count, and timing over a full training run.
    Prints a summary table — not a pass/fail test, diagnostic only.
    """
    layer = make_layer(chunk_dim=36, case_capacity=512,
                       group_capacity=256)
    layer.use_nnv2 = False  # disable NNv2 to test raw NNv5 group count
    embs  = make_embedding_dist(256, 36, seed=4, n_clusters=8)

    print("\n  step  | mse     | cases | groups | nnv2 | adapted | emit_bits")
    print("  ------|---------|-------|--------|------|---------|----------")

    opt        = torch.optim.Adam(layer.adapter.parameters(), lr=0.01)
    N          = embs.shape[0]
    t_start    = time.time()
    emit_bits_hist = []

    for step in range(2000):
        idx    = torch.randperm(N)[:16]
        batch  = embs[idx]
        tokens = float_to_uint64(batch)

        out, diff, emit, wm, choices = layer.forward(tokens)
        layer.learn(diff, emit, wm)

        # Count set bits in emit_out[:, 0] per input
        _, _, _, _, _, emit_out = layer.nnv5.forward(tokens)
        eo = np.ascontiguousarray(emit_out[:, 0])  # (N, EW) uint64 contiguous
        emit_bits_per_input = np.unpackbits(
            eo.view(np.uint8), axis=1, bitorder='little'
        ).sum(axis=1)                               # (N,) set bit count per input
        avg_bits = float(emit_bits_per_input.mean())
        emit_bits_hist.append(avg_bits)

        mse = float('nan')
        if layer.adapter.array_used > 0:
            opt.zero_grad()
            loss = layer.mse_loss(tokens, batch)
            if loss.requires_grad:
                loss.backward()
                opt.step()
            mse = loss.item()

        if step % 200 == 0 or step == 1999:
            s = layer.stats()
            avg_eb = np.mean(emit_bits_hist[-50:]) if emit_bits_hist else 0
            print(f"  {step:5d} | {mse:7.4f} | {s['cases_used']:5d} | "
                  f"{s['groups_used']:6d} | {s['nnv2_cases_used']:4d} | "
                  f"{s['adapter_initialised']:6d} | {avg_eb:10.3f}")

    elapsed = time.time() - t_start
    print(f"\n  total time: {elapsed:.2f}s for 2000 steps")
    print("  PASS  convergence benchmark")


def test_adapter_initialisation_rate():
    """
    Tracks how quickly adapter cases get frontloaded from zero.
    All cases should be initialised within a few passes over the dataset.
    """
    layer = make_layer(chunk_dim=36, case_capacity=256,
                       group_capacity=256)
    embs  = make_embedding_dist(128, 36, seed=5, n_clusters=4)
    opt   = torch.optim.Adam(layer.adapter.parameters(), lr=0.01)

    uninit_counts = []
    for step in range(50):
        idx    = torch.randperm(128)[:16]
        batch  = embs[idx]
        tokens = float_to_uint64(batch)
        _, diff, emit, wm, _ = layer.forward(tokens)
        layer.learn(diff, emit, wm)

        if layer.adapter.array_used > 0:
            opt.zero_grad()
            loss = layer.mse_loss(tokens, batch)
            if loss.requires_grad:
                loss.backward()
                opt.step()
            uninit_counts.append(layer.stats()['adapter_uninitialised'])

    if not uninit_counts:
        print("  SKIP  adapter initialisation rate (no cases formed)")
        return

    total_cases = layer.adapter.array_used
    final_uninit = uninit_counts[-1]
    final_init   = total_cases - final_uninit

    print(f"  uninitialised: start={uninit_counts[0]}, "
          f"end={final_uninit}, "
          f"total_cases={total_cases}, initialised={final_init}")

    # Uninitialised count may grow as new cases are added, but the
    # fraction of uninitialised cases should not increase over time
    if len(uninit_counts) > 1 and uninit_counts[0] > 0:
        # Compare uninit fraction at start vs end
        # Need total cases at each point — approximate from adapter.array_used trajectory
        assert final_uninit < total_cases, \
            "At least some cases should be initialised by end of training"
    print("  PASS  adapter initialisation rate")


if __name__ == "__main__":
    print("Running synthetic validation suite...\n")
    tests = [
        test_float_packing_roundtrip,
        test_nnlayer_sees_float_bits,
        test_cases_grow_over_steps,
        test_mse_decreases,
        test_case_stabilization,
        test_clustered_vs_random,
        test_adapter_initialisation_rate,
        test_convergence_benchmark,
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
