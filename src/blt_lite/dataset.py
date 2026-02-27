from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class TextFolderProvider:
    root: str
    pattern: str = "*.txt"

    def iter_texts(self):
        for path in sorted(Path(self.root).glob(self.pattern)):
            yield path.read_text(encoding="utf-8")


class TokenSequenceDataset(Dataset):
    def __init__(self, tokens: np.ndarray, seq_len: int):
        if len(tokens) <= seq_len:
            raise ValueError("Not enough tokens for sequence length")
        self.tokens = torch.from_numpy(tokens.astype(np.int64))
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.tokens) - self.seq_len - 1

    def __getitem__(self, idx: int):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return x, y
