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

4. Train tiny model:

```bash
python scripts/train_tiny.py --config configs/tiny.yaml
```

Resume training from a checkpoint named like `step_<N>.pt`:

```bash
python scripts/train_tiny.py --config configs/tiny.yaml --checkpoint outputs/step_200.pt
```

5. Sample:

```bash
python scripts/sample_tiny.py --config configs/tiny.yaml --prompt "Hello"
```

## Notes

- Tokenization is byte-identity (raw UTF-8 bytes map to token IDs 0..255) with BOS/EOS special tokens, and configurable `tokenizer.patch_size` chunking.
- Model uses isolated patch encoder/decoder modules: the patch encoder projects grouped multi-byte token windows, processes them causally, then the patch decoder cross-attends those states into token states before the main causal transformer.
- Eval is capped by `train.eval_batches` so validation cost stays bounded as datasets grow.
- Prioritizes tokenizer coherence and minimal code over model quality.
