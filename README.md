# BLT Lite

A small local playground for fixed-byte patch tokenization and a tiny patch-level language model.

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

5. Sample:

```bash
python scripts/sample_tiny.py --config configs/tiny.yaml --prompt "Hello"
```

## Notes

- Tokenization is fixed-length patches over UTF-8 bytes.
- Prioritizes tokenizer coherence and minimal code over model quality.
