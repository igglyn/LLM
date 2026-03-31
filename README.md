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

Fetch directly: `https://www.gutenberg.org/cache/epub/{ID}/pg{ID}.txt`

Recommended IDs — varied in style, era, and prose density:

**Novels / Fiction**
- 1342  Pride and Prejudice — Austen
- 84    Frankenstein — Shelley
- 2701  Moby Dick — Melville
- 98    A Tale of Two Cities — Dickens
- 1400  Great Expectations — Dickens
- 74    Tom Sawyer — Twain
- 76    Huckleberry Finn — Twain
- 1661  Sherlock Holmes (Adventures) — Doyle
- 2852  Sherlock Holmes (Hound) — Doyle
- 345   Dracula — Stoker
- 1260  Jane Eyre — Brontë
- 158   Emma — Austen
- 768   Wuthering Heights — Brontë
- 4300  Ulysses — Joyce
- 2600  War and Peace — Tolstoy
- 2554  Crime and Punishment — Dostoevsky
- 5200  Metamorphosis — Kafka
- 174   Picture of Dorian Gray — Wilde
- 844   The Importance of Being Earnest — Wilde
- 1952  The Yellow Wallpaper — Gilman
- 219   Heart of Darkness — Conrad
- 526   The War of the Worlds — Wells
- 35    The Time Machine — Wells
- 36    The Island of Doctor Moreau — Wells
- 55    The Wizard of Oz — Baum
- 16    Peter Pan — Barrie
- 1184  The Count of Monte Cristo — Dumas
- 1257  The Three Musketeers — Dumas

**Essays / Philosophy / Non-fiction**
- 2680  Meditations — Marcus Aurelius
- 996   Don Quixote — Cervantes
- 1080  A Modest Proposal — Swift
- 910   Two Treatises of Government — Locke
- 7370  Leviathan — Hobbes
- 4705  On Liberty — Mill
- 1635  The Republic — Plato
- 1713  Discourse on Method — Descartes
- 2500  Relativity — Einstein
- 1228  The Origin of Species — Darwin
- 3207  Autobiography of Benjamin Franklin

**Drama / Poetry**
- 100   Complete Works of Shakespeare
- 844   The Importance of Being Earnest — Wilde

**Short stories (good for diversity at small dataset sizes)**
- 1064  Poe stories
- 932   Fall of the House of Usher — Poe
- 23    Narrative of Frederick Douglass

**Epic / Ancient**
- 16328 Beowulf
- 1727  The Odyssey — Homer
- 22    The Iliad — Homer

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

#### Slot-conv patcher2 (rewrite path)

If `patcher2.type: slot_conv`, use:

```bash
python rewrite/train_slot_conv_patcher.py --config configs/tiny.yaml --print-effective-config
```

Optional wiring guard:

```bash
python rewrite/train_slot_conv_patcher.py --config configs/tiny.yaml --prepare-data-if-missing
```

`rewrite/train_slot_conv_patcher.py` validates unknown config keys (with typo hints),
accepts `groups`/`d_chunk` (or aliases `group_count`/`chunk_dim`), and can auto-run
`scripts/prepare_data_patcher2.py` when stage1 cache files are missing.
For slot-conv rewrite training, `seq_len_tokens` is deprecated/ignored; sequence
length is derived from `model.seq_len * patcher.patch_size * patcher2.patch_size`.

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
