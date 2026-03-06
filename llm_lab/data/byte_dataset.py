"""Byte-level text dataset."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


class ByteDataset(Dataset[Tensor]):
    """Dataset that yields fixed-length byte sequences from `.txt` files.

    The dataset:
    1. Reads all `.txt` files in ``data_dir``.
    2. Concatenates their UTF-8 text bytes.
    3. Splits into non-overlapping chunks of length ``seq_len``.
    4. Optionally shuffles chunk order deterministically using ``seed``.

    Each item is a ``torch.uint8`` tensor with shape ``[T]`` where
    ``T == seq_len``.
    """

    def __init__(self, data_dir: str, seq_len: int, seed: int = 0) -> None:
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.seed = seed

        if self.seq_len <= 0:
            raise ValueError("seq_len must be > 0")

        txt_files = sorted(self.data_dir.glob("*.txt"))
        raw = b"".join(path.read_bytes() for path in txt_files)

        n_chunks = len(raw) // self.seq_len
        self._chunks = [
            raw[i * self.seq_len : (i + 1) * self.seq_len] for i in range(n_chunks)
        ]

        rng = random.Random(self.seed)
        rng.shuffle(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, index: int) -> Tensor:
        chunk = self._chunks[index]
        return torch.tensor(list(chunk), dtype=torch.uint8)
