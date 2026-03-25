"""pipeline.py — NNv5LayerConfig and SyntheticBatchHarness."""
from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional
from .nnv5 import NNv5Block, NNv5CapacityError
from .nnv2 import NNv2Block

@dataclass
class NNv5LayerConfig:
    """Per-layer configuration for the two-layer NNv5 pipeline."""
    chunk_dim:          int = 32    # bits per NNv5 chunk — must satisfy chunk_dim1 * chunk_dim2 == d_model
    case_capacity:      int = 2048
    group_capacity:     int = 128
    nnv5_chunk_size:    int = 16
    nnv2_case_capacity: int = 1024


class SyntheticBatchHarness:
    """
    Wraps a TinyPatchLM trunk to provide synthetic batch augmentation.

    Two-layer NNv5 pipeline (no learned projection):
        Layer 1: raw hidden states → threshold to bool → chunk by chunk_dim1
                 → NNv5+NNv2 per chunk → concatenated emit bitsets
        Layer 2: layer1 emit bits → chunk by chunk_dim2
                 → NNv5+NNv2 → final group affinities for synthesis

    Constraint: layer1.chunk_dim * layer2.chunk_dim == d_model

    Usage in training loop:
        harness = SyntheticBatchHarness(model, cfg, device)
        harness.attach()
        logits, loss = model(x, y)
        synth_loss = harness.synthetic_pass(harness.captured_hidden, y)
        total_loss = loss + synth_weight * synth_loss
        loss.backward()
        harness.detach()
    """
    def __init__(
        self,
        model,
        d_model: int,
        layer1: NNv5LayerConfig,
        layer2: NNv5LayerConfig,
        n_synthetic: int,
        match_threshold: float,
        device: torch.device,
        seed: int = 42,
        nnv5_update_steps: int = 100,
        synth_loss_weight: float = 0.5,
        forward_batch_size: int = 4096,
    ):
        self.model              = model
        self.d_model            = d_model
        self.n_synthetic        = n_synthetic
        self.match_threshold    = match_threshold
        self.device             = device
        self.nnv5_update_steps  = nnv5_update_steps
        self.synth_loss_weight  = synth_loss_weight
        self.forward_batch_size = forward_batch_size
        self.rng                = np.random.default_rng(seed)

        self.layer1_cfg = layer1
        self.layer2_cfg = layer2
        self.layer2_enabled = layer2.case_capacity > 0

        # Derived dimensions
        n_chunks1 = d_model // layer1.chunk_dim
        self.n_chunks1 = n_chunks1

        # Single NNv5+NNv2 instance for layer 1
        self.nnv5_l1 = NNv5Block(layer1.chunk_dim, layer1.case_capacity, layer1.group_capacity)
        self.nnv2_l1 = NNv2Block(layer1.chunk_dim, layer1.group_capacity, layer1.nnv2_case_capacity)

        # Layer 2 — only if enabled
        if self.layer2_enabled:
            assert (layer1.group_capacity * n_chunks1) % layer2.chunk_dim == 0, (
                f"layer2.chunk_dim ({layer2.chunk_dim}) must divide "
                f"layer1 total emit bits ({layer1.group_capacity * n_chunks1})"
            )
            layer2_input_width = layer1.group_capacity * n_chunks1
            n_chunks2          = layer2_input_width // layer2.chunk_dim
            self.n_chunks2          = n_chunks2
            self.layer2_input_width = layer2_input_width
            self.nnv5_l2 = NNv5Block(layer2.chunk_dim, layer2.case_capacity, layer2.group_capacity)
            self.nnv2_l2 = NNv2Block(layer2.chunk_dim, layer2.group_capacity, layer2.nnv2_case_capacity)
        else:
            self.n_chunks2          = 0
            self.layer2_input_width = 0
            self.nnv5_l2 = None
            self.nnv2_l2 = None


        self.captured_hidden: torch.Tensor | None = None
        self._hook = None
        self._last_update_step = -1

        # Persistent input buffers
        l1_match_words = (layer1.chunk_dim + 63) // 64
        self._buf_l1 = np.empty((0, l1_match_words), dtype=np.uint64)
        if self.layer2_enabled:
            l2_match_words = (layer2.chunk_dim + 63) // 64
            self._buf_l2 = np.empty((0, l2_match_words), dtype=np.uint64)
        else:
            self._buf_l2 = None

    def attach(self):
        """Register forward hook to capture trunk hidden states."""
        def _hook_fn(module, inp, out):
            self.captured_hidden = inp[0].detach().float()
        self._hook = self.model.ln_f.register_forward_hook(_hook_fn)

    def detach(self):
        """Remove forward hook."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def _hidden_to_bool(self, hidden: torch.Tensor) -> np.ndarray:
        """
        Convert float hidden states to packed uint64 bool arrays.
        Threshold at 0 — positive activations become 1 bits.

        hidden: (N, d_model) float
        returns: (N, match_words) uint64  where match_words = ceil(d_model / 64)
        """
        N      = hidden.shape[0]
        bits   = (hidden > 0).cpu().numpy().astype(bool)   # (N, d_model)
        n_bits = bits.shape[1]
        words  = (n_bits + 63) // 64
        # Pad to multiple of 64
        if n_bits % 64 != 0:
            pad  = 64 - (n_bits % 64)
            bits = np.pad(bits, ((0, 0), (0, pad)))
        # Pack 64 bits per uint64 word — LSB first
        packed = np.packbits(bits, axis=1, bitorder='little')  # (N, words*8) uint8
        return packed.view(np.uint64)                           # (N, words) uint64

    def _run_layer(
        self,
        packed_vecs: np.ndarray,
        nnv5: NNv5Block,
        nnv2: NNv2Block,
        chunk_dim: int,
        nnv5_chunk_size: int,
        buf: np.ndarray,
        forward_batch_size: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run one NNv5+NNv2 layer on packed uint64 inputs with a persistent buffer.

        forward_batch_size controls how many rows forward processes at once.
        nnv5_chunk_size controls how many diffs assign processes at once —
        kept small to prevent case explosion.

        The two are separated because forward can handle large batches efficiently
        while assign needs small chunks to control case growth.
        """
        use_torch = self.device.type == "cuda"
        M         = packed_vecs.shape[0]

        # Append to buffer
        buf = np.concatenate([buf, packed_vecs], axis=0) if buf.shape[0] > 0 else packed_vecs.copy()

        # Process buffer in forward_batch_size chunks.
        # On CUDA, cap staged rows to at most 3 forward batches to limit peak VRAM.
        max_gpu_rows = forward_batch_size * 3
        while buf.shape[0] >= forward_batch_size:
            if use_torch:
                staged_rows = min(buf.shape[0], max_gpu_rows)
                buf_t = torch.from_numpy(buf[:staged_rows].view(np.int64)).to(
                    self.device, non_blocking=True
                )
                consumed_rows = 0
                while consumed_rows + forward_batch_size <= staged_rows:
                    fwd_batch_t = buf_t[consumed_rows:consumed_rows + forward_batch_size]
                    consumed_rows += forward_batch_size
                    nnv2_active = nnv5.emit_used > 0

                    diff, emit, was_matched, _, _, _ = nnv5.forward_torch(fwd_batch_t, self.device)

                    # Assign diffs in nnv5_chunk_size slices to control case growth
                    n_diffs = diff.shape[0]
                    for start in range(0, max(1, n_diffs), nnv5_chunk_size):
                        end          = min(n_diffs, start + nnv5_chunk_size)
                        d_chunk      = diff[start:end]
                        e_chunk      = emit[start:end]
                        m_chunk      = was_matched[start:end]
                        cases_before = nnv5.array_used

                        if nnv2_active:
                            neg_case, new_cases, new_cases_emit, new_group_cases = \
                                nnv5.assign_candidates(d_chunk, e_chunk, m_chunk)
                            nnv5.commit_candidates(neg_case, new_cases, new_cases_emit, new_group_cases)
                        else:
                            nnv5.assign(d_chunk, e_chunk, m_chunk)
                            if nnv5.emit_used > 0:
                                nnv2_active = True

                        cases_after = nnv5.array_used
                        if nnv5.emit_used > 0 and cases_after > cases_before:
                            nnv2.compress_range(nnv5, cases_before, cases_after)
                            nnv2.sync_to_nnv5(nnv5)

                del buf_t
                if consumed_rows == 0:
                    break
                buf = buf[consumed_rows:]
            else:
                fwd_batch = buf[:forward_batch_size]
                buf       = buf[forward_batch_size:]
                nnv2_active = nnv5.emit_used > 0

                diff, emit, was_matched, _, _, _ = nnv5.forward(fwd_batch)

                # Assign diffs in nnv5_chunk_size slices to control case growth
                n_diffs = diff.shape[0]
                for start in range(0, max(1, n_diffs), nnv5_chunk_size):
                    end          = min(n_diffs, start + nnv5_chunk_size)
                    d_chunk      = diff[start:end]
                    e_chunk      = emit[start:end]
                    m_chunk      = was_matched[start:end]
                    cases_before = nnv5.array_used

                    if nnv2_active:
                        neg_case, new_cases, new_cases_emit, new_group_cases = \
                            nnv5.assign_candidates(d_chunk, e_chunk, m_chunk)
                        nnv5.commit_candidates(neg_case, new_cases, new_cases_emit, new_group_cases)
                    else:
                        nnv5.assign(d_chunk, e_chunk, m_chunk)
                        if nnv5.emit_used > 0:
                            nnv2_active = True

                    cases_after = nnv5.array_used
                    if nnv5.emit_used > 0 and cases_after > cases_before:
                        nnv2.compress_range(nnv5, cases_before, cases_after)
                        nnv2.sync_to_nnv5(nnv5)

        # Emit extraction pass on current inputs only
        if use_torch:
            packed_vecs_t = torch.from_numpy(packed_vecs.view(np.int64)).to(self.device, non_blocking=True)
            _, _, _, choices_full, _, _ = nnv5.forward_torch(packed_vecs_t, self.device)
        else:
            _, _, _, choices_full, _, _ = nnv5.forward(packed_vecs)

        C        = nnv5.array_used
        emit_pos = nnv5.emit[0, :C]
        per_row_emit = np.zeros((M, nnv5.emit_words), dtype=np.uint64)
        if C > 0 and choices_full.shape[1] > 0:
            matched = NNv5Block.unpack_choices(choices_full, C)    # (M, C) bool
            masked  = np.where(matched[:, :, None], emit_pos[None, :, :], np.uint64(0))
            per_row_emit = np.bitwise_or.reduce(masked, axis=1)    # (M, EW)

        return per_row_emit, buf

    def update_case_table(self, hidden: torch.Tensor):
        """
        Run two-layer NNv5+NNv2 pipeline on the current batch's hidden states.
        Uses persistent buffers — NNv5 flushes independently of LLM batch size.

        hidden: (B, T, D) float
        """
        flat   = hidden.reshape(-1, self.d_model)
        packed = self._hidden_to_bool(flat)          # (N, d_model_words) uint64

        N             = packed.shape[0]
        d_model_words = packed.shape[1]
        chunk_words1  = (self.layer1_cfg.chunk_dim + 63) // 64
        n_chunks1     = d_model_words // chunk_words1

        # Reshape to (N * n_chunks1, chunk_words1)
        chunked1 = packed.reshape(N * n_chunks1, chunk_words1)

        emit1_rows, self._buf_l1 = self._run_layer(
            chunked1,
            self.nnv5_l1, self.nnv2_l1,
            self.layer1_cfg.chunk_dim,
            self.layer1_cfg.nnv5_chunk_size,
            self._buf_l1,
            self.forward_batch_size,
        )
        # emit1_rows: (N * n_chunks1, emit_words_l1)
        # Reshape to (N, n_chunks1 * emit_words_l1) then to (N * n_chunks2, chunk_words2)
        if emit1_rows.shape[0] == 0 or self.nnv5_l1.emit_used == 0 or not self.layer2_enabled:
            return

        n_chunks2    = self.n_chunks2
        chunk_words2 = (self.layer2_cfg.chunk_dim + 63) // 64
        emit1_flat   = emit1_rows.reshape(N, n_chunks1 * self.nnv5_l1.emit_words)

        target_words = n_chunks2 * chunk_words2
        if emit1_flat.shape[1] < target_words:
            emit1_flat = np.pad(emit1_flat, ((0,0),(0, target_words - emit1_flat.shape[1])))

        chunked2 = emit1_flat.reshape(N * n_chunks2, chunk_words2)

        _, self._buf_l2 = self._run_layer(
            chunked2,
            self.nnv5_l2, self.nnv2_l2,
            self.layer2_cfg.chunk_dim,
            self.layer2_cfg.nnv5_chunk_size,
            self._buf_l2,
            self.forward_batch_size,
        )

    def synthetic_pass(
        self,
        real_hidden: torch.Tensor,
        targets: torch.Tensor,
        step: int = 0,
    ) -> torch.Tensor:
        """
        Generate synthetic hidden states and compute loss on them.

        real_hidden: (B, T, D) float
        targets:     (B, T)    int64
        step:        int       — current training step
        """
        # Only update case tables for first nnv5_update_steps steps — once per model step
        if step <= self.nnv5_update_steps and step != self._last_update_step:
            self._last_update_step = step
            self.update_case_table(real_hidden)
            if step == self.nnv5_update_steps:
                l2_str = f" | L2 cases={self.nnv5_l2.array_used} groups={self.nnv5_l2.emit_used}" if self.layer2_enabled and self.nnv5_l2 is not None else ""
                print(f"  NNv5 tables frozen at step {step} — "
                      f"L1 cases={self.nnv5_l1.array_used} groups={self.nnv5_l1.emit_used}{l2_str}")

        # Check layer1 has any structure to synthesize from
        if self.nnv5_l1.array_used == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        B, T, D = real_hidden.shape
        flat   = real_hidden.reshape(-1, D)
        packed = self._hidden_to_bool(flat)              # (N, d_model_words) uint64
        N             = packed.shape[0]
        d_model_words = packed.shape[1]
        chunk_words1  = (self.layer1_cfg.chunk_dim + 63) // 64
        n_chunks1     = d_model_words // chunk_words1
        chunked1      = packed.reshape(N * n_chunks1, chunk_words1)

        if self.nnv5_l1.array_used == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        use_torch = self.device.type == "cuda"
        if use_torch:
            _, _, _, choices1, _, _ = self.nnv5_l1.forward_torch(chunked1, self.device)
        else:
            _, _, _, choices1, _, _ = self.nnv5_l1.forward(chunked1)

        C1        = self.nnv5_l1.array_used
        emit_pos1 = self.nnv5_l1.emit[0, :C1]
        emit1_rows = np.zeros((N * n_chunks1, self.nnv5_l1.emit_words), dtype=np.uint64)
        if C1 > 0 and choices1.shape[1] > 0:
            m       = NNv5Block.unpack_choices(choices1, C1)
            masked1 = np.where(m[:, :, None], emit_pos1[None, :, :], np.uint64(0))
            emit1_rows = np.bitwise_or.reduce(masked1, axis=1)

        # --- Synthesis ---
        if self.layer2_enabled and self.nnv5_l2 is not None and self.nnv5_l2.array_used > 0:            # Full two-layer reverse pipeline
            n_chunks2    = self.n_chunks2
            chunk_words2 = (self.layer2_cfg.chunk_dim + 63) // 64
            emit1_flat   = emit1_rows.reshape(N, n_chunks1 * self.nnv5_l1.emit_words)
            target_words = n_chunks2 * chunk_words2
            if emit1_flat.shape[1] < target_words:
                emit1_flat = np.pad(emit1_flat, ((0,0),(0, target_words - emit1_flat.shape[1])))

            emit1_chunked = emit1_flat.reshape(N * n_chunks2, chunk_words2)
            if use_torch:
                _, _, _, choices2, _, _ = self.nnv5_l2.forward_torch(emit1_chunked, self.device)
            else:
                _, _, _, choices2, _, _ = self.nnv5_l2.forward(emit1_chunked)

            C2        = self.nnv5_l2.array_used
            emit_pos2 = self.nnv5_l2.emit[0, :C2]
            l2_emit_rows_flat = np.zeros((N * n_chunks2, self.nnv5_l2.emit_words), dtype=np.uint64)
            if C2 > 0 and choices2.shape[1] > 0:
                m2      = NNv5Block.unpack_choices(choices2, C2)
                masked2 = np.where(m2[:, :, None], emit_pos2[None, :, :], np.uint64(0))
                l2_emit_rows_flat = np.bitwise_or.reduce(masked2, axis=1)

            l2_emit_per_pos = np.zeros((N, self.nnv5_l2.emit_words), dtype=np.uint64)
            l2_emit_rows = l2_emit_rows_flat.reshape(N, n_chunks2, self.nnv5_l2.emit_words)
            for i in range(n_chunks2):
                l2_emit_per_pos |= l2_emit_rows[:, i, :]

            l2_emit_stacked  = np.stack([l2_emit_per_pos, np.zeros_like(l2_emit_per_pos)], axis=1)
            synth_emit1_flat = self.nnv5_l2.reverse(l2_emit_stacked)

            l1_emit_words = self.nnv5_l1.emit_words
            synth_emit1_rechunked = synth_emit1_flat.reshape(N * n_chunks1, l1_emit_words) \
                if synth_emit1_flat.shape[1] == n_chunks1 * l1_emit_words \
                else np.zeros((N * n_chunks1, l1_emit_words), dtype=np.uint64)

            l1_emit_stacked = np.stack([synth_emit1_rechunked, np.zeros_like(synth_emit1_rechunked)], axis=1)
            synth_chunked1  = self.nnv5_l1.reverse(l1_emit_stacked)
            synth_packed    = synth_chunked1.reshape(N, n_chunks1 * chunk_words1)

        else:
            # Layer 1 only — use reverse pass from L1 emit
            if self.nnv5_l1.array_used > 0 and self.nnv5_l1.emit_used > 0:
                # Get L1 emit per chunk row
                l1_emit_stacked = np.stack(
                    [emit1_rows, np.zeros_like(emit1_rows)], axis=1
                )  # (N*n_chunks1, 2, emit_words)
                synth_chunked1 = self.nnv5_l1.reverse(l1_emit_stacked)  # (N*n_chunks1, chunk_words1)
                synth_packed   = synth_chunked1.reshape(N, n_chunks1 * chunk_words1)
            else:
                # No structure yet — vocab noise fallback
                synth_packed = self.nnv5_l1.synthesize_from(
                    packed.reshape(N * n_chunks1, chunk_words1), self.rng
                ).reshape(N, n_chunks1 * chunk_words1)

        # Unpack uint64 → bits → float
        synth_bits = np.unpackbits(
            synth_packed.view(np.uint8), axis=1, bitorder='little'
        )[:, :self.d_model].astype(np.float32)

        # Rescale to match real hidden state distribution
        synth_hidden = torch.from_numpy(synth_bits).to(self.device)
        real_flat    = flat.float()
        real_mean    = real_flat.mean(dim=0, keepdim=True)
        real_std     = real_flat.std(dim=0, keepdim=True) + 1e-8
        synth_hidden = (synth_hidden - synth_hidden.mean(dim=0, keepdim=True)) / \
                       (synth_hidden.std(dim=0, keepdim=True) + 1e-8)
        synth_hidden = synth_hidden * real_std + real_mean
        synth_hidden = synth_hidden.reshape(B, T, D)

        _, synth_loss = self.model.forward_from_hidden(synth_hidden, targets)
        return synth_loss

    def stats(self) -> dict:
        s = {
            "l1_cases":      self.nnv5_l1.array_used,
            "l1_groups":     self.nnv5_l1.emit_used,
            "nnv2_l1_cases": self.nnv2_l1.array_used,
        }
        if self.layer2_enabled and self.nnv5_l2 is not None:
            s["l2_cases"]      = self.nnv5_l2.array_used
            s["l2_groups"]     = self.nnv5_l2.emit_used
            s["nnv2_l2_cases"] = self.nnv2_l2.array_used
        return s
