"""Loss helpers for assembled model training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_reconstruction_loss(
    recon_logits: torch.Tensor,
    x_u8: torch.Tensor,
    *,
    patch_size: int,
) -> torch.Tensor:
    """Cross-entropy reconstruction loss against raw bytes.

    Args:
        recon_logits: ``[B, T1, patch_size, 256]``.
        x_u8: raw bytes ``[B, T]``.
        patch_size: patch size used by patcher1.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if recon_logits.ndim != 4:
        raise ValueError("recon_logits must be rank-4 [B, T1, patch_size, 256].")

    bsz, t1, p, vocab = recon_logits.shape
    if p != patch_size:
        raise ValueError(f"recon_logits patch axis ({p}) must equal patch_size ({patch_size}).")
    if vocab != 256:
        raise ValueError(f"recon_logits vocab axis must be 256, got {vocab}.")

    target_len = t1 * patch_size
    if x_u8.size(1) < target_len:
        pad = target_len - x_u8.size(1)
        x_target = torch.nn.functional.pad(x_u8, (0, pad), value=0)
    else:
        x_target = x_u8[:, :target_len]

    target = x_target.view(bsz, t1, patch_size).to(torch.long)
    return F.cross_entropy(recon_logits.reshape(-1, 256), target.reshape(-1))
