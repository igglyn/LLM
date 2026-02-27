from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import TokenSequenceDataset
from .model import TinyPatchLM


def build_dataloaders(train_path: Path, val_path: Path, seq_len: int, batch_size: int):
    train_tokens = np.load(train_path)
    val_tokens = np.load(val_path)
    train_ds = TokenSequenceDataset(train_tokens, seq_len)
    val_ds = TokenSequenceDataset(val_tokens, seq_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dl, val_dl


def evaluate(model: TinyPatchLM, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(1, len(losses)))
