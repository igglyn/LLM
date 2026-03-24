# AdEMAMix Fused Implementation Notes

This file is the working checklist for implementing the combined optimizer end-to-end.

## Current objective
Deliver a production-ready fused AdEMAMix optimizer path that preserves local semantics (including slow_ema reset) while supporting a quantized-state backend and efficient grouped execution.

## Progress tracker

### Phase 1 — Baseline semantics and plumbing
- [x] Add in-repo `AdEMAMixFused` optimizer implementation.
- [x] Preserve AdEMAMix math shape (`m1`, `m2`, `nu`, `alpha_t`, `beta3_t`).
- [x] Keep decoupled AdamW-style weight decay.
- [x] Add explicit slow_ema reset hook (`slow_ema_reset_steps`).
- [x] Wire `optimizer="ademamix_fused"` into config + train entrypoint.
- [x] Add baseline tests for step updates and optimizer builder selection.

### Phase 2 — Execution model parity and state layout
- [x] Add grouped `torch._foreach_*` path plus scalar fallback.
- [x] Align state layout to stacked `m1_m2` + `nu`.
- [x] Add foreach-vs-scalar parity test.
- [x] Add slow_ema reset behavior test.
- [x] Add backward-compatibility migration in `__setstate__` for legacy `m1`/`m2` state dicts.

### Phase 3 — Remaining implementation work
- [x] Add a true fused kernel backend (Triton/CUDA) behind the same optimizer API.
- [x] Define and implement one quantization backend for optimizer state:
  - [x] bnb-style blockwise int8 (`qmap` + `absmax`) **or**
  - [ ] FlashOptim grouped quantization + scales.
- [x] Add backend switch and compatibility layer between eager and fused implementations.
- [x] Add checkpoint compatibility matrix tests:
  - [x] old eager state -> new layout,
  - [x] eager <-> fused,
  - [x] fp32 <-> quantized state (if enabled).
- [x] Add performance benchmarks and acceptance thresholds.

## Near-term next steps
1. Implement fused backend scaffold with no quantization first (correctness-first).
2. Add parity harness comparing eager vs fused over many randomized steps.
3. Introduce quantized state backend once fused correctness is stable.
4. Add benchmark script and establish throughput/memory gates.

## Risks to watch
- Kernel/eager numerical drift in low precision.
- State-dict incompatibility across backend changes.
- Slow_ema reset parity regressions when switching backends.
