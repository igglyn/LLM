"""nn_layer.py — NNLayer: NNv5 → NNv2 → NNv2Adapter."""
from __future__ import annotations
import numpy as np
import torch
from .nnv5 import NNv5Block
from .nnv2 import NNv2Block, NNv2Adapter


class NNLayer:
    def __init__(self, bool_dim, chunk_dim, case_capacity, group_capacity,
                 shard_size=32, n_chunks=1, device=None, use_nnv2=True):
        self.bool_dim  = bool_dim
        self.chunk_dim = chunk_dim
        self.device    = device or torch.device('cpu')
        self.use_nnv2  = use_nnv2

        self.nnv5 = NNv5Block(bool_dim, case_capacity, group_capacity,
                               shard_size, device=self.device)
        self.nnv2 = NNv2Block(group_capacity, case_capacity,
                               shard_size, device=self.device)
        self.adapter = NNv2Adapter(chunk_dim, case_capacity,
                                   n_chunks, device=self.device)

    def forward(self, tokens, chunk_idx=0):
        diff, emit, wm, cp, counts, emit_out = self.nnv5.forward(tokens)
        choices = self._route_choices(cp, emit_out)
        out = self.adapter.forward(choices, chunk_idx)
        return out, diff, emit, wm, choices

    def _route_choices(self, cp, emit_out):
        C = self.nnv5.array_used
        if self.use_nnv2 and self.nnv2.array_used > 0 and self.nnv5.emit_used > 0:
            emit_in = torch.from_numpy(
                emit_out[:, 0].view(np.int64)).to(self.device)
            _, choices = self.nnv2.forward(emit_in)
        else:
            choices = NNv5Block.unpack_choices(cp, C)
        return choices

    def preprocess_diffs(self, diff, emit, was_matched):
        K = diff.shape[0]
        W = diff.shape[2]
        C = self.nnv5.array_used
        if K == 0 or C == 0:
            return diff, emit, was_matched
        match_pos = self.nnv5.match[0].reshape(-1, W)[:C].cpu().numpy().view(np.uint64)
        pos = diff[:, 0]
        covered = ((pos[:, :, None] & match_pos[None, :, :].transpose(0, 2, 1))
                   == match_pos[None, :, :].transpose(0, 2, 1))
        covered_any = covered.any(axis=2)
        diff_out = diff.copy()
        diff_out[:, 0] = np.where(covered_any, np.uint64(0), pos)
        has_pos = diff_out[:, 0].any(axis=1)
        return diff_out[has_pos], emit[has_pos], was_matched[has_pos]

    def learn(self, diff, emit, was_matched):
        if diff.shape[0] == 0:
            return
        diff, emit, was_matched = self.preprocess_diffs(diff, emit, was_matched)
        if diff.shape[0] == 0:
            return
        prev_used = self.nnv5.array_used
        self.nnv5.assign(diff, emit, was_matched)
        new_used = self.nnv5.array_used
        if new_used > prev_used:
            if self.use_nnv2 and self.nnv5.emit_used > 0:
                self.nnv2.compress_range(self.nnv5, prev_used, new_used, self.adapter)
            else:
                self.adapter.sync(new_used)

    def mse_loss(self, tokens, target, chunk_idx=0):
        out, diff, emit, wm, choices = self.forward(tokens, chunk_idx)
        if self.adapter.array_used == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return self.adapter.mse_loss(choices, target, chunk_idx)

    def stats(self):
        return {
            **self.nnv5.stats(),
            **self.nnv2.stats(),
            **self.adapter.stats(),
            "nnv2_active": self.nnv2.array_used > 0 and self.nnv5.emit_used > 0,
        }
