# Optimizer Source Fetch Bundle

This directory captures source material needed to design a fused custom optimizer kernel that combines AdEMAMix + fused AdamW behavior.

## 1) Existing local optimizer implementation
- `local/optim.py`
- Origin: this repository (`llm_lab/train/optim.py`).
- Notes: currently only wires AdamW and includes local trainability filtering behavior.

## 2) bitsandbytes AdEMAMix implementation
- `bitsandbytes/ademamix.py`
  - Origin repo: `bitsandbytes-foundation/bitsandbytes`
  - Commit: `88e802cace45918e2ee9dac6e1881a5e5e0a712e`
  - Upstream path: `bitsandbytes/optim/ademamix.py`
- `bitsandbytes/kernels_optim.py`
  - Origin repo: `bitsandbytes-foundation/bitsandbytes`
  - Commit: `88e802cace45918e2ee9dac6e1881a5e5e0a712e`
  - Upstream path: `bitsandbytes/backends/triton/kernels_optim.py`
  - Notes: includes triton optimizer update paths for `optimizer_name == "ademamix"`.

## 3) DeepSpeed fused AdamW implementation
- `deepspeed/fused_adam.py`
  - Origin repo: `microsoft/DeepSpeed`
  - Commit: `26c954ffaaef94ef4d0b55ecd04e5058479fea6b`
  - Upstream path: `deepspeed/ops/adam/fused_adam.py`
- `deepspeed/multi_tensor_adam.cu`
  - Origin repo: `microsoft/DeepSpeed`
  - Commit: `26c954ffaaef94ef4d0b55ecd04e5058479fea6b`
  - Upstream path: `csrc/adam/multi_tensor_adam.cu`

## 4) FlashAdam kernel discovery status
A broad repository scan did **not** find a canonical, actively maintained project that exposes an optimizer specifically named `FlashAdam` with a dedicated kernel implementation.

What was checked:
- `Dao-AILab/flash-attention` (no flash-adam optimizer kernel found).
- GitHub repository searches for `flashadam`, `flash adam optimizer`, and `flash-adam`.

## 5) FlashAdam fallback research item (optimizer-only)
FlashAdam source is now taken from Databricks FlashOptim:

- `flashadam_candidate/adam.py`
  - Origin repo: `databricks/flashoptim`
  - Commit: `8b2d966e0329ed2b0c22c3782cae3989c0ccb03b`
  - Upstream path: `flashoptim/optimizers.py`
  - Notes: includes production `FlashAdam`/`FlashAdamW`, quantized state handling, and fused Triton kernels in one source file.

## Additional cloned references (not copied into this folder)
- `pytorch/pytorch` @ `b12a208aaf7daa19779eabfa9c7f79ee58aeba19` (no `AdEMAMix` implementation found by text search).

- **Combination summary:** keep local AdEMAMix semantics (including slow_ema reset); take bitsandbytes AdEMAMix math/state layout and blockwise 8-bit dequant→update→requant flow (including stacked m1/m2 + nu handling); align FlashOptim Adam fused Triton kernel structure and quantized-state interfaces where compatible; and reuse DeepSpeed multi-tensor launch/group stepping mechanics for fused execution orchestration.
