"""Rewrite-oriented main model that owns projection glue around patchers.

The rewrite patchers intentionally omit a few adapter blocks (linear projection glue).
This module keeps those blocks in the main model so patchers stay isolated and reusable.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rewrite.patcher_models import EmbeddedPatcherConfig, RewritePatcherAutoencoder


@dataclass(frozen=True)
class RewriteMainModelConfig:
    """Config for the rewrite main model wrapper."""

    vocab_size: int = 256
    d_model: int = 384
    patch_size: int = 4
    n_layers: int = 4
    n_heads: int = 6
    dropout: float = 0.1
    use_patcher2: bool = True
    patcher2_patch_size: int = 2


class RewriteMainModel(nn.Module):
    """Minimal main model that keeps adapter/projection blocks outside patchers.

    Transferred blocks now owned by the main model:
    - token embedding and optional positional embedding
    - token->patcher linear adapter
    - patcher->trunk residual gate
    - final LM head
    """

    def __init__(
        self,
        cfg: RewriteMainModelConfig = RewriteMainModelConfig(),
        patcher_cfg: EmbeddedPatcherConfig = EmbeddedPatcherConfig(),
        patcher2_cfg: EmbeddedPatcherConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg

        if patcher_cfg.d_model <= 0:
            raise ValueError("patcher d_model must be > 0")

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.token_to_patcher = nn.Linear(cfg.d_model, patcher_cfg.d_model)
        self.patcher_to_token = nn.Linear(patcher_cfg.d_model, cfg.d_model)
        self.patcher = RewritePatcherAutoencoder(patch_size=cfg.patch_size, cfg=patcher_cfg)
        self.use_patcher2 = bool(cfg.use_patcher2 and patcher2_cfg is not None)
        if self.use_patcher2:
            if patcher2_cfg is None or patcher2_cfg.d_model <= 0:
                raise ValueError("patcher2 d_model must be > 0 when patcher2 is enabled")
            self.token_to_patcher2 = nn.Linear(cfg.d_model, patcher2_cfg.d_model)
            self.patcher2_to_token = nn.Linear(patcher2_cfg.d_model, cfg.d_model)
            self.patcher2 = RewritePatcherAutoencoder(patch_size=cfg.patcher2_patch_size, cfg=patcher2_cfg)
        else:
            self.token_to_patcher2 = None
            self.patcher2_to_token = None
            self.patcher2 = None

        self.trunk = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.n_heads,
                    dim_feedforward=cfg.d_model * 4,
                    dropout=cfg.dropout,
                    batch_first=True,
                )
                for _ in range(max(1, cfg.n_layers))
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def load_patcher_checkpoint(self, path: str | Path, map_location: torch.device | str | None = None) -> None:
        ckpt = torch.load(path, map_location=map_location or "cpu")
        state = ckpt.get("patcher", ckpt) if isinstance(ckpt, dict) else ckpt
        self.patcher.load_state_dict(state)

    def load_patcher2_checkpoint(self, path: str | Path, map_location: torch.device | str | None = None) -> None:
        if self.patcher2 is None:
            raise ValueError("patcher2 is disabled for this model; cannot load checkpoint")
        ckpt = torch.load(path, map_location=map_location or "cpu")
        state = ckpt.get("patcher2", ckpt) if isinstance(ckpt, dict) else ckpt
        self.patcher2.load_state_dict(state)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        h = self.token_emb(token_ids)

        patcher_in = self.token_to_patcher(h)
        patcher_recon, patch1_latent = self.patcher(patcher_in)
        h = h + self.patcher_to_token(patcher_recon)
        patch2_latent: torch.Tensor | None = None

        if self.patcher2 is not None and self.token_to_patcher2 is not None and self.patcher2_to_token is not None:
            patcher2_in = self.token_to_patcher2(h)
            patcher2_recon, patch2_latent = self.patcher2(patcher2_in)
            h = h + self.patcher2_to_token(patcher2_recon)

        for block in self.trunk:
            h = block(h)
        logits = self.lm_head(self.ln_f(h))
        return logits, {"patch1": patch1_latent, "patch2": patch2_latent}
