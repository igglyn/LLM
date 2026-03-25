"""Benchmark AdEMAMix backends for throughput and optimizer-state memory."""

from __future__ import annotations

import argparse
import time

import torch

from llm_lab.train.ademamix_fused import AdEMAMixFused


def _state_bytes(opt: AdEMAMixFused) -> int:
    total = 0
    for state in opt.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def run_bench(device: str, backend: str, state_backend: str, size: int, steps: int) -> dict[str, float]:
    w = torch.nn.Parameter(torch.randn(size, device=device, dtype=torch.float32))
    opt = AdEMAMixFused([w], lr=1e-3, backend=backend, state_backend=state_backend)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        w.grad = torch.randn_like(w)
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    toks_per_sec = (size * steps) / max(dt, 1e-9)

    return {
        "seconds": dt,
        "elements_per_sec": toks_per_sec,
        "state_megabytes": _state_bytes(opt) / (1024**2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--min-speedup", type=float, default=0.95)
    parser.add_argument("--max-state-mb", type=float, default=1000.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    eager = run_bench(device=device, backend="eager", state_backend="fp32", size=args.size, steps=args.steps)
    fused = run_bench(device=device, backend="fused", state_backend="fp32", size=args.size, steps=args.steps)
    qint8 = run_bench(device=device, backend="eager", state_backend="qint8", size=args.size, steps=args.steps)

    speedup = fused["elements_per_sec"] / max(eager["elements_per_sec"], 1e-9)
    state_ok = qint8["state_megabytes"] <= args.max_state_mb
    speed_ok = speedup >= args.min_speedup

    print(f"device={device}")
    print(f"eager_fp32: {eager}")
    print(f"fused_fp32: {fused}")
    print(f"eager_qint8: {qint8}")
    print(f"speedup_fused_over_eager={speedup:.3f} threshold={args.min_speedup}")
    print(f"qint8_state_mb={qint8['state_megabytes']:.3f} threshold={args.max_state_mb}")

    if not speed_ok:
        raise SystemExit("fused backend did not meet speed acceptance threshold")
    if not state_ok:
        raise SystemExit("qint8 state backend did not meet memory acceptance threshold")


if __name__ == "__main__":
    main()
