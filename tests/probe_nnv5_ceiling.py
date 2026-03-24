#!/usr/bin/env python
"""
probe_nnv5_ceiling.py — Probe NNv5 group count ceiling on synthetic data.

Tune hyperparameters via command line args and see where groups stabilize.
Not a test — a diagnostic tool.

Usage:
    python scripts/probe_nnv5_ceiling.py
    python scripts/probe_nnv5_ceiling.py --cases 4096 --groups 2048 --batch 64 --steps 10000
    python scripts/probe_nnv5_ceiling.py --blocks 3 --samples 512 --clusters 8
"""
from __future__ import annotations

import sys
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
import importlib.util

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_base    = SRC / "blt_lite" / "synthetic_batch"
_nnv5    = _load("blt_lite.synthetic_batch.nnv5",       str(_base / "nnv5.py"))
_nnv2    = _load("blt_lite.synthetic_batch.nnv2",       str(_base / "nnv2.py"))
_layer   = _load("blt_lite.synthetic_batch.nn_layer",   str(_base / "nn_layer.py"))
_bf      = _load("blt_lite.synthetic_batch.block_float", str(_base / "block_float.py"))

NNLayer          = _layer.NNLayer
encode_torch     = _bf.encode_torch
VALUES_PER_BLOCK = _bf.VALUES_PER_BLOCK

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_layer(chunk_dim, case_capacity, group_capacity, shard_size):
    assert chunk_dim % VALUES_PER_BLOCK == 0
    n_blocks = chunk_dim // VALUES_PER_BLOCK
    bool_dim = n_blocks * 2 * 64
    cap      = max(case_capacity, shard_size)
    layer    = NNLayer(bool_dim, chunk_dim, cap, group_capacity,
                       shard_size, 1, DEVICE)
    layer.use_nnv2 = False
    return layer


def make_embedding_dist(n, dim, seed, n_clusters):
    rng     = torch.Generator(); rng.manual_seed(seed)
    centers = torch.randn(n_clusters, dim, generator=rng, device=DEVICE) * 0.5
    labels  = torch.randint(0, n_clusters, (n,), generator=rng, device=DEVICE)
    noise   = torch.randn(n, dim, generator=rng, device=DEVICE) * 0.1
    return (centers[labels] + noise).clamp(-3.0, 3.0)


def main():
    parser = argparse.ArgumentParser(description="Probe NNv5 group count ceiling.")
    parser.add_argument("--blocks",   type=int,   default=3,    help="Number of 128-bit blocks (chunk_dim = blocks * 36)")
    parser.add_argument("--cases",    type=int,   default=4096, help="Case capacity")
    parser.add_argument("--groups",   type=int,   default=2048, help="Group capacity")
    parser.add_argument("--shard",    type=int,   default=16,   help="Shard size")
    parser.add_argument("--samples",  type=int,   default=512,  help="Number of synthetic samples")
    parser.add_argument("--clusters", type=int,   default=8,    help="Number of clusters in synthetic data")
    parser.add_argument("--batch",    type=int,   default=64,   help="Batch size per step")
    parser.add_argument("--steps",    type=int,   default=10000,help="Number of training steps")
    parser.add_argument("--print_every", type=int, default=1000, help="Print interval")
    parser.add_argument("--seed",     type=int,   default=42,   help="Random seed")
    args = parser.parse_args()

    chunk_dim = args.blocks * VALUES_PER_BLOCK

    print(f"\n{'='*70}")
    print(f"  NNv5 Ceiling Probe")
    print(f"{'='*70}")
    print(f"  blocks={args.blocks}  chunk_dim={chunk_dim}  bool_dim={args.blocks*128}")
    print(f"  cases={args.cases}  groups={args.groups}  shard={args.shard}")
    print(f"  samples={args.samples}  clusters={args.clusters}")
    print(f"  batch={args.batch}  steps={args.steps}  seed={args.seed}")
    print(f"  device={DEVICE}")
    print(f"{'='*70}\n")

    layer = make_layer(chunk_dim, args.cases, args.groups, args.shard)
    embs  = make_embedding_dist(args.samples, chunk_dim, args.seed, args.clusters)

    header = (f"  {'step':>6} | {'cases':>6} | {'groups':>6} | "
              f"{'c/g ratio':>9} | {'emit_bits':>10} | {'new_c':>6} | {'new_g':>6}")
    sep    = "  " + "-"*6 + "+" + "-"*8 + "+" + "-"*8 + "+" + "-"*11 + "+" + "-"*12 + "+" + "-"*8 + "+" + "-"*8
    print(header)
    print(sep)

    N         = embs.shape[0]
    prev_c    = 0
    prev_g    = 0
    t_start   = time.time()
    emit_hist = []

    for step in range(args.steps):
        idx    = torch.randperm(N)[:args.batch]
        batch  = embs[idx]
        tokens = encode_torch(batch)

        out, diff, emit, wm, choices = layer.forward(tokens)
        layer.learn(diff, emit, wm)

        # Emit sparsity
        _, _, _, _, _, emit_out = layer.nnv5.forward(tokens)
        eo   = np.ascontiguousarray(emit_out[:, 0])
        bits = np.unpackbits(eo.view(np.uint8), axis=1, bitorder='little').sum(axis=1)
        emit_hist.append(float(bits.mean()))

        if step % args.print_every == 0 or step == args.steps - 1:
            c      = layer.nnv5.array_used
            g      = layer.nnv5.emit_used
            ratio  = c / max(1, g)
            avg_eb = np.mean(emit_hist[-min(100, len(emit_hist)):])
            new_c  = c - prev_c
            new_g  = g - prev_g
            prev_c = c
            prev_g = g
            print(f"  {step:>6} | {c:>6} | {g:>6} | "
                  f"{ratio:>9.2f} | {avg_eb:>10.3f} | {new_c:>6} | {new_g:>6}")

    elapsed = time.time() - t_start
    c = layer.nnv5.array_used
    g = layer.nnv5.emit_used
    print(f"\n  Finished {args.steps} steps in {elapsed:.1f}s")
    print(f"  Final: {c} cases / {g} groups  ratio={c/max(1,g):.2f}")
    print(f"  Case capacity used: {c/args.cases*100:.1f}%")
    print(f"  Group capacity used: {g/args.groups*100:.1f}%")


if __name__ == "__main__":
    main()
