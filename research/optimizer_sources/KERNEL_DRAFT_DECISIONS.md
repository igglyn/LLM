# Fused AdEMAMix Kernel: Decisions to Lock Before Writing Code

This checklist captures architectural choices that should be finalized before implementing the fused kernel.

## 1) Algorithmic semantics (must match base behavior)
- **AdEMAMix update equation** to preserve exactly:
  - `m1 <- beta1*m1 + (1-beta1)*g`
  - `m2 <- beta3_t*m2 + (1-beta3_t)*g` (slow EMA)
  - `nu <- beta2*nu + (1-beta2)*g^2`
  - `update <- (m1 / (1-beta1^t) + alpha_t * m2) / (sqrt(nu)/(sqrt(1-beta2^t)) + eps)`
  - AdamW-style decoupled weight decay on parameters.
- **Slow EMA reset behavior** from local base is non-negotiable and must be reproduced exactly.
- **Scheduler behavior**:
  - lock exact definitions for `alpha_t` and `beta3_t` (warmup vs closed-form interpolation).

## 2) State layout contract
- Parameter state tensors:
  - `state1`: stacked `(m1, m2)` layout (same shape contract as bnb AdEMAMix path).
  - `state2`: `nu` tensor.
  - optional metadata/state for quantization + error correction.
- Decide if step counter is per-parameter tensor, per-group scalar, or both (checkpoint compatibility implications).

## 3) Quantization path selection
- **Primary quantization scheme** to lock:
  - bnb-style blockwise int8 with `qmap` + `absmax` metadata, or
  - FlashOptim grouped quantization/scales path.
- Decide whether v1 supports:
  - fp32-only states first,
  - or both fp32 + quantized states in one release.
- If quantized states are enabled, lock dequant->fp32 update->requant invariants and rounding behavior.

## 4) Fused execution model
- **Kernel granularity**:
  - one fused kernel per dtype bucket (fp16/bf16/fp32), or
  - separate kernels for precondition/update/quantization.
- **Launch orchestration**:
  - DeepSpeed-style multi-tensor batching should be used to reduce kernel launch overhead.
- **Contiguity/shape constraints**:
  - decide if non-contiguous params/gradients are hard-error or copied to contiguous buffers.

## 5) Numerical policy
- Bias-correction implementation (compute on host vs in-kernel).
- Denominator stabilization and epsilon placement (exactly one formula everywhere).
- Mixed precision casting rules (compute in fp32; writeback dtype policy).
- Optional safety features (max update norm, numerics checks) in v1 or deferred.

## 6) Weight decay policy
- Lock default mode to **decoupled AdamW** for fused path.
- Decide if coupled L2 mode is supported as a compatibility toggle.
- If both supported, lock exact flag names and checkpoint serialization behavior.

## 7) API and checkpoint compatibility
- Python optimizer API surface:
  - required constructor args: `betas(3)`, `alpha`, `t_alpha`, `t_beta3`, `eps`, `weight_decay`, quantization flags.
- `state_dict` compatibility:
  - define backward/forward compatibility across fp32 and quantized states.
- Migration rules for loading old checkpoints with missing new fields.

## 8) Integration with local training plumbing
- Preserve current train mode semantics (`full`, `patcher_only`) and trainable filtering behavior.
- Ensure optimizer construction still respects local parameter-group expectations.

## 9) Scope for first implementation
- Recommended v1 scope lock:
  1. Correctness-first fused fp32-state AdEMAMix + AdamW decay + slow EMA reset parity.
  2. Add quantized state path after parity tests pass.
  3. Add advanced extras (ECC-like paths / gradient-release style hooks) only after core parity + perf validation.

## 10) Test gates to define before coding
- Step-level parity tests vs reference Python AdEMAMix over many seeds/steps.
- Dedicated tests for slow EMA reset trigger behavior and checkpoint round-trip.
- Numerical drift bounds for fp16/bf16 with fused path.
- Throughput benchmark targets (tokens/s or step/s) and memory targets.

---

## Suggested lock-in order
1. Mathematical parity + reset semantics.
2. State layout + checkpoint contract.
3. Fused launch model + dtype bucketing.
4. Quantization scheme decision.
5. v1 scope and acceptance tests.
