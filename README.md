# blt_lite

Byte-level patch language model. Two-stage patcher pipeline, frozen before trunk training.

## Structure

```
src/blt_lite/
  model.py       — PatcherAutoencoder, TinyPatchLM
  dataset.py     — TokenSequenceDataset, TextFolderProvider
  tokenizer.py   — FixedPatchTokenizer (byte identity)
  train.py       — build_dataloaders, evaluate
  utils.py       — load_config, set_seed, get_device

scripts/
  prepare_data.py           — tokenise raw .txt → train/val .npy
  prepare_data_patcher2.py  — run frozen patcher1 → cache hidden states
  train_patcher.py          — train patcher1 (MSE on hidden reconstruction)
  train_patcher2.py         — train patcher2 (MSE on patcher1 hidden states)
  train_tiny.py             — train trunk with both patchers frozen

configs/
  tiny.yaml                 — all hyperparameters
```

## Pipeline

### 1. Put raw text files in `data/raw/`

Any `.txt` files. Public domain books from Gutenberg work well.
~15MB is enough to see learning; overfitting starts around there.

### 2. Prepare patcher1 data

```bash
python scripts/prepare_data.py --config configs/tiny.yaml
```

Outputs: `data/processed/patcher/train_tokens.npy`, `val_tokens.npy`, `tokenizer.json`

### 3. Train patcher1

```bash
python scripts/train_patcher.py --config configs/tiny.yaml
```

Loss is MSE on reconstructed hidden states (not cross-entropy on bytes).
Outputs checkpoints to `outputs/patcher/`.

### 4. Cache patcher1 hidden states for patcher2

```bash
python scripts/prepare_data_patcher2.py \
  --config configs/tiny.yaml \
  --patcher-checkpoint outputs/patcher/best.pt
```

Or set `patcher.pretrained_path` in `tiny.yaml` and omit the flag.

Outputs: `data/processed/patcher2/train_stage1_hidden.npy`, `val_stage1_hidden.npy`

This is the step that was missing. Without it, patcher2 has no training data.

### 5. Train patcher2

```bash
python scripts/train_patcher2.py --config configs/tiny.yaml
```

Trains on the cached hidden states from step 4.
Outputs checkpoints to `outputs/patcher2/`.

### 6. Train trunk

Set checkpoint paths in `tiny.yaml`:
```yaml
patcher:
  pretrained_path: outputs/patcher/best.pt
patcher2:
  pretrained_path: outputs/patcher2/best.pt
```

Then:
```bash
python scripts/train_tiny.py --config configs/tiny.yaml
```

Both patchers are frozen. Trunk trains on cross-entropy.

## Why the two-stage caching matters

Patcher2 trains on the output of patcher1, not raw tokens.
If patcher2 were trained jointly with patcher1 (or from raw tokens directly),
gradient leakage would cause the most optimal loss to collapse toward noise —
the same problem that motivated separating patcher training from trunk training.

The cache step makes the isolation explicit and enforced.
