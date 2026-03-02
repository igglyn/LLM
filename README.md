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

4. Pretrain patcher/unpatcher (compression stage):

```bash
python scripts/train_patcher.py --config configs/tiny.yaml
```

5. Train tiny LM (optionally loading/freezeing pretrained patcher from config):

```bash
python scripts/train_tiny.py --config configs/tiny.yaml
```

Resume training from a checkpoint named like `step_<N>.pt` (also accepts `best.pt` / `last.pt`):

```bash
python scripts/train_tiny.py --config configs/tiny.yaml --checkpoint outputs/step_200.pt
```

6. Sample:

```bash
python scripts/sample_tiny.py --config configs/tiny.yaml --prompt "Hello"
```

## Notes

- Tokenization is byte-identity (raw UTF-8 bytes map to token IDs 0..255) with BOS/EOS special tokens, and configurable `tokenizer.patch_size` chunking.
- Patcher and unpatcher are fully isolated in `PatcherAutoencoder` (`PatchEncoder` + `PatchDecoder`). They can be pretrained first with reconstruction loss and then plugged into `TinyPatchLM`.
- Patcher architecture and training knobs are configurable under `patcher` and `patcher_train` in the YAML config.
- Eval is capped by `train.eval_batches` so validation cost stays bounded as datasets grow.
- Prioritizes tokenizer coherence and minimal code over model quality.
