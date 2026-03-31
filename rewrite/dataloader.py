"""Rewrite dataloader focused on first patcher training.

Key behavior:
- preprocessing tokenizes each source text file independently,
  writing one `.npy` token file per source;
- training windows are fixed to first-patcher patch length (`patcher.patch_size`),
  so no separate seq_len argument is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from blt_lite.tokenizer import FixedPatchTokenizer


class SourcePatchDataset(Dataset):
    """Samples fixed windows by patch count from per-source token files."""

    def __init__(self, token_files: list[Path], patch_size: int, patch_count: int):
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if patch_count <= 0:
            raise ValueError("patch_count must be > 0")
        self.patch_size = patch_size
        self.patch_count = patch_count
        self.token_window = patch_size * patch_count
        self.tokens_per_file = [np.load(path, mmap_mode="r") for path in token_files]
        self.index: list[tuple[int, int]] = []

        for file_idx, arr in enumerate(self.tokens_per_file):
            if len(arr) < self.token_window:
                continue
            for start in range(0, len(arr) - self.token_window + 1, self.token_window):
                self.index.append((file_idx, start))

        if not self.index:
            raise ValueError("No valid source windows found for the requested patch window")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx, start = self.index[idx]
        arr = self.tokens_per_file[file_idx]
        x = arr[start : start + self.token_window]
        return torch.from_numpy(np.asarray(x, dtype=np.int64))


def preprocess_sources(cfg: dict, output_root: Path) -> tuple[Path, Path, Path]:
    """Tokenize each raw source into its own file for train/val splits."""

    data_cfg = cfg["data"]
    tok_cfg = cfg.get("tokenizer", {})
    raw_root = Path(data_cfg["raw_path"])
    pattern = data_cfg.get("pattern", "*.txt")
    split = float(data_cfg.get("train_split", 0.95))

    source_paths = sorted(raw_root.glob(pattern))
    if not source_paths:
        raise RuntimeError(f"No source files found under {raw_root} matching {pattern}")

    texts = [path.read_text(encoding="utf-8") for path in source_paths]
    tokenizer = FixedPatchTokenizer(patch_size=1)
    tokenizer.fit(texts)

    add_bos = bool(tok_cfg.get("add_bos", True))
    add_eos = bool(tok_cfg.get("add_eos", True))

    train_dir = output_root / "train_sources"
    val_dir = output_root / "val_sources"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict[str, int | str]] = {}
    for source_path, text in zip(source_paths, texts):
        ids = np.asarray(tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos), dtype=np.int32)
        cut = int(len(ids) * split)
        train_ids = ids[:cut]
        val_ids = ids[cut:]

        stem = source_path.stem
        train_file = train_dir / f"{stem}.npy"
        val_file = val_dir / f"{stem}.npy"
        np.save(train_file, train_ids)
        np.save(val_file, val_ids)

        manifest[source_path.name] = {
            "train_file": str(train_file),
            "val_file": str(val_file),
            "train_tokens": int(len(train_ids)),
            "val_tokens": int(len(val_ids)),
        }

    tokenizer_path = output_root / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    with open(output_root / "source_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return train_dir, val_dir, tokenizer_path


def build_first_patcher_dataloaders(cfg: dict, output_root: Path, batch_size: int) -> tuple[DataLoader, DataLoader, Path]:
    """Build dataloaders for first patcher only (window by patch count)."""

    patch_size = int(cfg.get("patcher", {}).get("patch_size", 1))
    patch_count = int(cfg.get("patcher_train", {}).get("window_patches", cfg.get("model", {}).get("seq_len", 1)))
    train_dir, val_dir, tokenizer_path = preprocess_sources(cfg, output_root=output_root)

    train_files = sorted(train_dir.glob("*.npy"))
    val_files = sorted(val_dir.glob("*.npy"))
    if not train_files or not val_files:
        raise RuntimeError("Preprocess step produced no train/val source files")

    train_ds = SourcePatchDataset(train_files, patch_size=patch_size, patch_count=patch_count)
    val_ds = SourcePatchDataset(val_files, patch_size=patch_size, patch_count=patch_count)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dl, val_dl, tokenizer_path
