# BLT Lite

A small local playground for byte-identity tokenization and a tiny byte-level language model.

## Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Add one or more `.txt` files to `data/raw/`.

3. Prepare tokenizer + encoded dataset:

```bash
python scripts/prepare_data.py --config configs/tiny.yaml
```

4. Pretrain first patcher/unpatcher (tokens -> patches):

```bash
python scripts/train_patcher.py --config configs/tiny.yaml
```

5. Pretrain second stacked patcher (patches -> larger patches):

```bash
python scripts/train_patcher2.py --config configs/tiny.yaml
```

6. Train tiny LM (optionally loading/freezeing pretrained patcher from config):

```bash
python scripts/train_tiny.py --config configs/tiny.yaml
```

Resume training from a checkpoint named like `step_<N>.pt` (also accepts `best.pt` / `last.pt`):

```bash
python scripts/train_tiny.py --config configs/tiny.yaml --checkpoint outputs/step_200.pt
```

7. Sample (`sample.max_new_patches` controls generation horizon in large-patch units):

```bash
python scripts/sample_tiny.py --config configs/tiny.yaml --prompt "Hello"
```

## Notes

- Tokenization is byte-identity (raw UTF-8 bytes map to token IDs 0..255) with BOS/EOS special tokens. Data preparation is patch-agnostic; patching happens inside model/patcher modules.
- Patcher and unpatcher are fully isolated in `PatcherAutoencoder` (`PatchEncoder` + `PatchDecoder`). They can be pretrained first with reconstruction loss and then plugged into `TinyPatchLM`.
- Patcher architecture and training knobs are configurable under `patcher` and `patcher_train` in the YAML config. Both LM training and patcher pretraining use AdamW; patcher pretraining can reduce LR automatically once `patcher_train.lr_reduce_threshold` is reached.
- Main model AMP behavior is configurable under `train` (`amp_enabled`, `amp_dtype`) and is applied inside `TinyPatchLM` forward for CUDA runs.
- Patcher AMP is independently configurable via `patcher_train.amp_enabled` and `patcher_train.amp_dtype`.
- Positional encoding mode is configurable with `model.pos_encoding: `learned | rope`; RoPE rotates attention Q/K only and uses precomputed cos/sin caches up to max sequence length.
- Patcher positional encoding is also configurable via `patcher.pos_encoding: learned | rope`, applying the same Q/K-only RoPE option in patch encoder/decoder self-attention.
- Sequence length is configured under `model.seq_len` as large-patch context length; effective token context is derived as `model.seq_len * patcher.patch_size * patcher2.patch_size` for training and generation.
- Eval is capped by `train.eval_batches` so validation cost stays bounded as datasets grow.
- Prioritizes tokenizer coherence and minimal code over model quality.
