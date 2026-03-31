"""Rewrite dataloader for patcher + main-model rewrite layers.

Key behavior:
- preprocessing tokenizes each source text file independently,
  writing one `.npy` token file per source;
- training windows are fixed by patch geometry (`patch_size * patch_count`),
  so no separate seq_len argument is required.
- supports both patcher reconstruction windows and causal token windows
  from the same preprocessed source files.

This module can be called from project root as a script:
    python rewrite/dataloader.py --config configs/tiny.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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


class SourceCausalDataset(Dataset):
    """Samples fixed token windows for main-model causal next-token training."""

    def __init__(self, token_files: list[Path], seq_len: int):
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        self.seq_len = seq_len
        self.tokens_per_file = [np.load(path, mmap_mode="r") for path in token_files]
        self.index: list[tuple[int, int]] = []

        required = self.seq_len + 1
        for file_idx, arr in enumerate(self.tokens_per_file):
            if len(arr) < required:
                continue
            for start in range(0, len(arr) - required + 1, self.seq_len):
                self.index.append((file_idx, start))

        if not self.index:
            raise ValueError("No valid source windows found for the requested main-model sequence length")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_idx, start = self.index[idx]
        arr = self.tokens_per_file[file_idx]
        chunk = np.asarray(arr[start : start + self.seq_len + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


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


def _resolve_seq_len_tokens(cfg: dict) -> int:
    model_cfg = cfg.get("model", {})
    p1 = int(cfg.get("patcher", {}).get("patch_size", 1))
    p2_enabled = bool(cfg.get("patcher2", {}).get("enabled", True))
    p2 = int(cfg.get("patcher2", {}).get("patch_size", 2)) if p2_enabled else 1
    default_seq_len = int(model_cfg.get("seq_len", 1)) * p1 * p2
    return int(cfg.get("train", {}).get("seq_len_tokens", default_seq_len))


def build_rewrite_dataloaders(cfg: dict, output_root: Path, batch_size: int) -> tuple[dict[str, DataLoader], Path]:
    """Build combined rewrite dataloaders for patcher + main-model layers."""

    patch_size = int(cfg.get("patcher", {}).get("patch_size", 1))
    patch_count = int(cfg.get("patcher_train", {}).get("window_patches", cfg.get("model", {}).get("seq_len", 1)))
    seq_len_tokens = _resolve_seq_len_tokens(cfg)
    train_dir, val_dir, tokenizer_path = preprocess_sources(cfg, output_root=output_root)

    train_files = sorted(train_dir.glob("*.npy"))
    val_files = sorted(val_dir.glob("*.npy"))
    if not train_files or not val_files:
        raise RuntimeError("Preprocess step produced no train/val source files")

    train_ds = SourcePatchDataset(train_files, patch_size=patch_size, patch_count=patch_count)
    val_ds = SourcePatchDataset(val_files, patch_size=patch_size, patch_count=patch_count)
    train_main_ds = SourceCausalDataset(train_files, seq_len=seq_len_tokens)
    val_main_ds = SourceCausalDataset(val_files, seq_len=seq_len_tokens)

    loaders = {
        "patcher_train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        "patcher_val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False),
        "main_train": DataLoader(train_main_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        "main_val": DataLoader(val_main_ds, batch_size=batch_size, shuffle=False, drop_last=False),
    }
    return loaders, tokenizer_path


def build_first_patcher_dataloaders(cfg: dict, output_root: Path, batch_size: int) -> tuple[DataLoader, DataLoader, Path]:
    """Backward-compatible patcher-only dataloader builder."""
    loaders, tokenizer_path = build_rewrite_dataloaders(cfg, output_root=output_root, batch_size=batch_size)
    return loaders["patcher_train"], loaders["patcher_val"], tokenizer_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite first-patcher preprocessing + dataloader smoke entrypoint")
    parser.add_argument("--config", required=True, help="Path to config YAML from project root")
    args = parser.parse_args()

    from blt_lite.utils import load_config
    cfg = load_config(args.config)
    output_root = Path(cfg["data"].get("processed_dir_patcher", cfg["data"]["processed_dir"])) / "rewrite_sources"
    batch_size = int(cfg.get("patcher_train", {}).get("batch_size", cfg.get("train", {}).get("batch_size", 8)))

    loaders, tokenizer_path = build_rewrite_dataloaders(cfg, output_root=output_root, batch_size=batch_size)
    print(f"Prepared rewrite dataloader assets under: {output_root}")
    print(f"Tokenizer: {tokenizer_path}")
    print(
        "Patcher batches: "
        f"train={len(loaders['patcher_train'])} val={len(loaders['patcher_val'])} | "
        "Main batches: "
        f"train={len(loaders['main_train'])} val={len(loaders['main_val'])}"
    )


if __name__ == "__main__":
    main()
