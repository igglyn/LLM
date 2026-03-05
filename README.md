# BLT Lite

A small local playground for byte-identity tokenization and a tiny byte-level language model.

## Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Add one or more `.txt` files to `data/raw/`.

3. Prepare explicit stage-1 patcher data (raw text -> token IDs):

```bash
python scripts/prepare_data.py --config configs/tiny.yaml
```

4. Pretrain first patcher/unpatcher (tokens -> patches):

```bash
python scripts/train_patcher.py --config configs/tiny.yaml
```

5. Prepare explicit stage-2 patcher data artifacts:

```bash
python scripts/prepare_data_patcher2.py --config configs/tiny.yaml
```

6. Pretrain second stacked patcher (patches -> larger patches, optional when `patcher2.enabled: false`):

```bash
python scripts/train_patcher2.py --config configs/tiny.yaml
```

7. Prepare explicit TinyPatchLM data artifacts:

```bash
python scripts/prepare_data_tiny.py --config configs/tiny.yaml
```

8. Train tiny LM (requires pretrained patcher checkpoint; patcher2 checkpoint required only when `patcher2.enabled: true`; loaded patchers are always frozen):

```bash
python scripts/train_tiny.py --config configs/tiny.yaml
```

Resume training from a checkpoint named like `step_<N>.pt` (also accepts `best.pt` / `last.pt`):

```bash
python scripts/train_tiny.py --config configs/tiny.yaml --checkpoint outputs/step_200.pt
```

9. Sample (`sample.max_new_patches` controls generation horizon in large-patch units):

```bash
python scripts/sample_tiny.py --config configs/tiny.yaml --prompt "Hello"
```

## Notes

- Tokenization is byte-identity (raw UTF-8 bytes map to token IDs 0..255) with BOS/EOS special tokens. Data preparation is patch-agnostic; patching happens inside model/patcher modules.
- Training scripts consume only staged prepared directories: `data.processed_dir_patcher`, `data.processed_dir_patcher2`, and `data.processed_dir_tiny` (no fallback to base processed dir).
- `prepare_data_patcher2.py` now precomputes and stores stage-1 hidden streams (`*_stage1_hidden.npy`), and `prepare_data_tiny.py` precomputes stage-2 hidden streams (`*_stage2_hidden.npy`) so later training stages avoid re-running upstream patchers every step.
- Patcher and unpatcher are fully isolated in `PatcherAutoencoder` (`PatchEncoder` + `PatchDecoder`). They can be pretrained first with reconstruction loss and then plugged into `TinyPatchLM`.
- Patcher architecture and training knobs are configurable under `patcher` and `patcher_train` in the YAML config. Both LM training and patcher pretraining use AdamW; patcher pretraining can reduce LR automatically at up to two validation-loss milestones (`patcher_train.lr_reduce_threshold`, `patcher_train.lr_reduce_threshold_2`).
- Patcher stage sequence lengths are independently configurable with `patcher_train.seq_len_tokens` and `patcher2_train.seq_len_tokens`, so each patcher can train on its own context budget instead of inheriting full LM token context.
- Patcher2 pretraining supports the same dual loss-threshold LR drops via `patcher2_train.lr_reduce_threshold` and `patcher2_train.lr_reduce_threshold_2`.
- For patcher debugging during dataset generation, `prepare_data_patcher2.py --debug-token-match` reports matched vs unmatched token ratios while producing stage-1 hidden caches.
- `patcher2.enabled` can disable the second patcher stage entirely; when disabled, stage-2 training can be skipped and effective large-patch size becomes just `patcher.patch_size`.
- Main LM training supports the same two-threshold LR drop behavior (`train.lr_reduce_threshold`, `train.lr_reduce_threshold_2`) in addition to warmup-cosine scheduling.
- Main model AMP behavior is configurable under `train` (`amp_enabled`, `amp_dtype`) and is applied inside `TinyPatchLM` forward for CUDA runs.
- Patcher AMP is independently configurable via `patcher_train.amp_enabled` and `patcher_train.amp_dtype`.
- Gradient checkpointing and flash attention toggles are configurable per module: `model.grad_checkpointing`/`model.flash_attention`, `patcher.grad_checkpointing`/`patcher.flash_attention`, and `patcher2.grad_checkpointing`/`patcher2.flash_attention`.
- Block attention is configurable per patcher with `patcher.block_attention`/`patcher.block_size` and `patcher2.block_attention`/`patcher2.block_size` (default block size `8`); enabling it allows full attention within each block while remaining causal across blocks, trading memory/throughput against local parallelism.
- Positional encoding mode is configurable with `model.pos_encoding: `learned | rope`; RoPE rotates attention Q/K only and uses precomputed cos/sin caches up to max sequence length.
- Patcher positional encoding is also configurable via `patcher.pos_encoding: learned | rope`, applying the same Q/K-only RoPE option in patch encoder/decoder self-attention.
- Sequence length is configured under `model.seq_len` as large-patch context length; effective token context is derived as `model.seq_len * patcher.patch_size * (patcher2.patch_size if patcher2.enabled else 1)` for training and generation.
- Eval is capped by `train.eval_batches` so validation cost stays bounded as datasets grow.
- Prioritizes tokenizer coherence and minimal code over model quality.
