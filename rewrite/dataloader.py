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
    """Causal next-token windows from per-source token files."""

    def __init__(self, token_files: list[Path], seq_len: int):
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if not token_files:
            raise ValueError("token_files must not be empty")
        self.seq_len = seq_len
        self.tokens_per_file = [np.load(path, mmap_mode="r") for path in token_files]
        self.index: list[tuple[int, int]] = []

        for file_idx, arr in enumerate(self.tokens_per_file):
            # Need x[idx:idx+seq_len] and y[idx+1:idx+seq_len+1].
            if len(arr) <= seq_len:
                continue
            for start in range(0, len(arr) - seq_len - 1):
                self.index.append((file_idx, start))

        if not self.index:
            raise ValueError("No valid causal windows found for the requested sequence length")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_idx, start = self.index[idx]
        arr = self.tokens_per_file[file_idx]
        x = np.asarray(arr[start : start + self.seq_len], dtype=np.int64)
        y = np.asarray(arr[start + 1 : start + self.seq_len + 1], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


class SourceMetaPatchDataset(Dataset):
    """Samples windows of explicit first-layer patches for second patcher training.

    Each item is shaped `(patches_per_window, patch_size)`, where each row is one
    concrete first-layer patch extracted from source tokens. This avoids deriving
    stage-2 targets from a raw token-length product and keeps patch extraction as an
    explicit step (important for future whitespace-aware padding rules).
    """

    def __init__(
        self,
        token_files: list[Path],
        patch_size: int,
        meta_patch_size: int,
        window_meta_patches: int,
    ):
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if meta_patch_size <= 0:
            raise ValueError("meta_patch_size must be > 0")
        if window_meta_patches <= 0:
            raise ValueError("window_meta_patches must be > 0")

        self.patch_size = patch_size
        self.meta_patch_size = meta_patch_size
        self.window_meta_patches = window_meta_patches
        self.patches_per_window = meta_patch_size * window_meta_patches
        self.tokens_per_file = [np.load(path, mmap_mode="r") for path in token_files]
        self.index: list[tuple[int, int]] = []

        for file_idx, arr in enumerate(self.tokens_per_file):
            available_patch_count = len(arr) // self.patch_size
            if available_patch_count < self.patches_per_window:
                continue
            for patch_start in range(0, available_patch_count - self.patches_per_window + 1, self.patches_per_window):
                self.index.append((file_idx, patch_start))

        if not self.index:
            raise ValueError("No valid patch windows found for second patcher training")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx, patch_start = self.index[idx]
        arr = self.tokens_per_file[file_idx]
        token_start = patch_start * self.patch_size
        token_end = token_start + (self.patches_per_window * self.patch_size)
        tokens = np.asarray(arr[token_start:token_end], dtype=np.int64)
        patches = tokens.reshape(self.patches_per_window, self.patch_size)
        return torch.from_numpy(patches)


class ExportedMetaPatchDataset(Dataset):
    """Reads exported patcher2 windows from `.npy` files."""

    def __init__(self, window_files: list[Path]):
        self.window_files = window_files
        self.windows_per_file = [np.load(path, mmap_mode="r") for path in window_files]
        self.index: list[tuple[int, int]] = []

        for file_idx, arr in enumerate(self.windows_per_file):
            if arr.ndim != 3:
                raise ValueError(
                    f"Expected 3D exported patcher2 windows in {window_files[file_idx]}, got shape={arr.shape}"
                )
            for row_idx in range(arr.shape[0]):
                self.index.append((file_idx, row_idx))

        if not self.index:
            raise ValueError("No exported patcher2 windows found")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx, row_idx = self.index[idx]
        sample = np.asarray(self.windows_per_file[file_idx][row_idx], dtype=np.int64)
        return torch.from_numpy(sample)


def preprocess_sources(cfg: dict, output_root: Path) -> tuple[Path, Path, Path]:
    """Tokenize each raw source into its own file for train/val splits."""

    data_cfg = cfg["data"]
    tok_cfg = cfg.get("tokenizer", {})
    raw_root = Path(data_cfg["raw_path"])
    pattern = data_cfg.get("pattern", "*.txt")
    split = float(data_cfg.get("train_split", 0.95))
    if not (0.0 < split < 1.0):
        raise ValueError(f"data.train_split must be in (0, 1), got {split}")

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


def export_second_patcher_windows(
    source_dir: Path,
    export_dir: Path,
    patch_size: int,
    meta_patch_size: int,
    window_meta_patches: int,
) -> list[Path]:
    """Export second-patcher train/val files from tokenized per-source files.

    Each exported file is shaped:
    `(num_windows, meta_patch_size * window_meta_patches, patch_size)`.
    """

    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if meta_patch_size <= 0:
        raise ValueError("meta_patch_size must be > 0")
    if window_meta_patches <= 0:
        raise ValueError("window_meta_patches must be > 0")

    patches_per_window = meta_patch_size * window_meta_patches
    token_files = sorted(source_dir.glob("*.npy"))
    export_dir.mkdir(parents=True, exist_ok=True)
    exported_files: list[Path] = []

    for token_file in token_files:
        arr = np.load(token_file, mmap_mode="r")
        available_patch_count = len(arr) // patch_size
        if available_patch_count < patches_per_window:
            continue

        window_count = (available_patch_count - patches_per_window) // patches_per_window + 1
        windows = np.empty((window_count, patches_per_window, patch_size), dtype=np.int32)
        for i, patch_start in enumerate(range(0, available_patch_count - patches_per_window + 1, patches_per_window)):
            token_start = patch_start * patch_size
            token_end = token_start + (patches_per_window * patch_size)
            windows[i] = np.asarray(arr[token_start:token_end], dtype=np.int32).reshape(patches_per_window, patch_size)

        out_file = export_dir / token_file.name
        np.save(out_file, windows)
        exported_files.append(out_file)

    return exported_files


def _resolve_seq_len_tokens(cfg: dict) -> int:
    patcher_cfg = cfg.get("patcher", {})
    model_cfg = cfg.get("model", {})

    seq_len_tokens = int(patcher_cfg.get("seq_len_tokens", model_cfg.get("seq_len", 1)))
    if seq_len_tokens <= 0:
        raise ValueError(f"Resolved seq_len_tokens must be > 0, got {seq_len_tokens}")
    return seq_len_tokens


def build_rewrite_dataloaders(cfg: dict, output_root: Path, batch_size: int) -> tuple[dict[str, DataLoader], Path]:
    """Build rewrite patcher + main-model dataloaders with config validation."""

    patch_size = int(cfg.get("patcher", {}).get("patch_size", 1))
    patch_count = int(cfg.get("patcher_train", {}).get("window_patches", cfg.get("model", {}).get("seq_len", 1)))
    seq_len_tokens = _resolve_seq_len_tokens(cfg)
    if seq_len_tokens % patch_size != 0:
        raise ValueError(
            "patcher.seq_len_tokens must be divisible by patcher.patch_size "
            f"(got seq_len_tokens={seq_len_tokens}, patch_size={patch_size})"
        )

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


def build_second_patcher_dataloaders(cfg: dict, output_root: Path, batch_size: int) -> tuple[DataLoader, DataLoader, Path]:
    """Build dataloaders for second patcher windows expressed in first-layer patches.

    The second patcher combines first-layer patches into meta-patches. This loader
    explicitly materializes first-layer patch windows per sample:
    `(meta_patch_size * window_meta_patches, patch_size)`.
    """

    patcher_cfg = cfg.get("patcher", {})
    patcher2_cfg = cfg.get("patcher2", {})
    patcher_train = cfg.get("patcher_train", {})
    patcher2_train = cfg.get("patcher2_train", {})
    model_cfg = cfg.get("model", {})

    patch_size = int(patcher_cfg.get("patch_size", 1))
    meta_patch_size = int(patcher2_cfg.get("patch_size", 2))
    window_meta_patches = int(
        patcher2_train.get(
            "window_meta_patches",
            patcher_train.get("window_patches", model_cfg.get("seq_len", 1)),
        )
    )
    train_dir, val_dir, tokenizer_path = preprocess_sources(cfg, output_root=output_root)
    patcher2_train_dir = output_root / "patcher2_train"
    patcher2_val_dir = output_root / "patcher2_val"
    train_files = export_second_patcher_windows(
        source_dir=train_dir,
        export_dir=patcher2_train_dir,
        patch_size=patch_size,
        meta_patch_size=meta_patch_size,
        window_meta_patches=window_meta_patches,
    )
    val_files = export_second_patcher_windows(
        source_dir=val_dir,
        export_dir=patcher2_val_dir,
        patch_size=patch_size,
        meta_patch_size=meta_patch_size,
        window_meta_patches=window_meta_patches,
    )
    if not train_files or not val_files:
        raise RuntimeError("Second patcher export step produced no train/val files")

    train_ds = ExportedMetaPatchDataset(train_files)
    val_ds = ExportedMetaPatchDataset(val_files)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dl, val_dl, tokenizer_path


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
