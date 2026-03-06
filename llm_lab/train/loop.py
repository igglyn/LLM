"""Training loop utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from llm_lab.models.loss import compute_reconstruction_loss


def _next_batch(loader: DataLoader, iterator: Any):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _reset_memory_state(model: nn.Module) -> None:
    """Reset model memory state when available."""
    memory = getattr(model, "memory", None)
    if memory is None:
        return
    reset_fn = getattr(memory, "reset_state", None)
    if callable(reset_fn):
        reset_fn()
        return
    raise ValueError(
        "Model has memory but does not expose reset_state(); "
        "cannot apply explicit memory reset controls."
    )


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, bool]:
    """Unpack dataloader batch into bytes tensor and optional doc-boundary signal."""
    if isinstance(batch, tuple) and len(batch) == 2:
        x_u8, is_doc_boundary = batch
        return x_u8, bool(is_doc_boundary)
    if isinstance(batch, dict):
        x_u8 = batch.get("x")
        if x_u8 is None:
            x_u8 = batch.get("bytes")
        if x_u8 is None:
            raise ValueError("dict batch must contain 'x' or 'bytes'.")
        return x_u8, bool(batch.get("document_boundary", False))
    return batch, False


def _split_bptt_segments(batch_u8: torch.Tensor, segment_len: int | None) -> list[torch.Tensor]:
    """Split batch into truncation segments along sequence dimension."""
    if segment_len is None:
        return [batch_u8]
    if segment_len <= 0:
        raise ValueError("truncate_bptt_segments must be > 0 when provided.")
    segments: list[torch.Tensor] = []
    t = int(batch_u8.shape[1])
    for start in range(0, t, segment_len):
        seg = batch_u8[:, start : start + segment_len]
        if seg.shape[1] > 0:
            segments.append(seg)
    return segments


def _validate_byte_loss_alignment(*, segment_u8: torch.Tensor, logits: torch.Tensor) -> None:
    """Validate that logits can be aligned to next-byte targets.

    Main loss in this loop is byte-level next-token cross entropy:
    - target bytes are ``segment_u8[:, 1:]`` (predict next byte at each position)
    - therefore logits must provide one position per input byte, i.e. ``T_logits == T_in``

    If a model emits compressed patch-space logits (``T_logits < T_in``), there is no
    explicit mapping to byte-level targets in this objective and training would be
    semantically ambiguous.
    """
    t_in = int(segment_u8.shape[1])
    t_out = int(logits.shape[1])
    if t_out != t_in:
        raise ValueError(
            "Main loss expects byte-aligned logits with one output position per input byte "
            f"(got logits_len={t_out}, input_len={t_in}). "
            "Current objective uses next-byte targets segment_u8[:, 1:], so compressed "
            "patch-space outputs require an explicit mapping rule that is not implemented."
        )


def save_checkpoint(
    path: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    scaler: torch.cuda.amp.GradScaler | None,
) -> None:
    """Save model/optimizer/scaler state."""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    map_location: str | torch.device = "cpu",
) -> int:
    """Load model/optimizer/scaler state and return resume step."""
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("step", 0))


def train_loop(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    steps: int,
    device: str | torch.device = "cpu",
    grad_accum_steps: int = 1,
    grad_clip_norm: float = 1.0,
    amp: bool = False,
    checkpoint_path: str | None = None,
    resume: bool = False,
    checkpoint_every: int = 0,
    enable_aux_reconstruction: bool = False,
    aux_reconstruction_weight: float = 0.0,
    preserve_memory_across_batches: bool = True,
    reset_memory_on_document_boundary: bool = False,
    truncate_bptt_segments: int | None = None,
    mode: str = "full",
) -> dict[str, float | int]:
    """Run minimal next-byte prediction training.

    Objective: next-byte prediction using cross entropy.
    """
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be > 0")

    if mode not in {"full", "patcher_only"}:
        raise ValueError("mode must be 'full' or 'patcher_only'.")
    if mode == "patcher_only":
        if not hasattr(model, "component_param_groups") or not callable(model.component_param_groups):
            raise ValueError("patcher_only mode requires model.component_param_groups().")
        groups = model.component_param_groups()
        patcher_param_ids = {id(p) for p in groups.get("patcher1", []) + groups.get("patcher2", [])}
        if not patcher_param_ids:
            raise ValueError("patcher_only mode requires at least one patcher parameter.")
        opt_param_ids = {
            id(p)
            for group in optimizer.param_groups
            for p in group.get("params", [])
        }
        if not opt_param_ids.issubset(patcher_param_ids):
            raise ValueError("patcher_only mode requires optimizer to include only patcher params.")

    model = model.to(device)
    model.train()

    use_amp = amp and torch.cuda.is_available() and str(device).startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_step = 0
    if resume and checkpoint_path and Path(checkpoint_path).exists():
        start_step = load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler if use_amp else None,
            map_location=device,
        )

    iterator = iter(dataloader)
    last_loss = 0.0

    for step in range(start_step, steps):
        optimizer.zero_grad(set_to_none=True)

        for _ in range(grad_accum_steps):
            raw_batch, iterator = _next_batch(dataloader, iterator)
            batch_u8, is_doc_boundary = _unpack_batch(raw_batch)
            batch_u8 = batch_u8.to(device)

            # Explicit memory controls.
            if not preserve_memory_across_batches:
                _reset_memory_state(model)
            elif reset_memory_on_document_boundary and is_doc_boundary:
                _reset_memory_state(model)

            segments = _split_bptt_segments(batch_u8, truncate_bptt_segments)
            per_batch_loss = 0.0

            for seg_idx, segment_u8 in enumerate(segments):
                if seg_idx > 0 and not preserve_memory_across_batches:
                    _reset_memory_state(model)

                with torch.autocast(device_type="cuda", enabled=use_amp):
                    if enable_aux_reconstruction and hasattr(model, "forward_with_aux"):
                        logits, recon_logits = model.forward_with_aux(segment_u8)
                    else:
                        logits = model(segment_u8)
                        recon_logits = None

                    _validate_byte_loss_alignment(segment_u8=segment_u8, logits=logits)

                    # Byte-level next-token objective:
                    # logit at position t predicts byte at t+1.
                    target = segment_u8[:, 1:].to(torch.long)
                    logits_bt = logits[:, :-1, :]
                    main_loss = F.cross_entropy(
                        logits_bt.reshape(-1, logits_bt.size(-1)),
                        target.reshape(-1),
                    )

                    aux_loss = torch.tensor(0.0, device=logits.device)
                    if enable_aux_reconstruction and recon_logits is not None:
                        patch_size = int(getattr(getattr(model, "patcher1", None), "patch_size", 1))
                        aux_loss = compute_reconstruction_loss(
                            recon_logits,
                            segment_u8,
                            patch_size=patch_size,
                        )

                    segment_loss = main_loss + (aux_reconstruction_weight * aux_loss)

                per_batch_loss = per_batch_loss + float(segment_loss.detach().item())
                scaled_segment_loss = segment_loss / (grad_accum_steps * max(len(segments), 1))
                if use_amp:
                    scaler.scale(scaled_segment_loss).backward()
                else:
                    scaled_segment_loss.backward()

            last_loss = per_batch_loss / max(len(segments), 1)

        if grad_clip_norm > 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if checkpoint_path and checkpoint_every > 0 and ((step + 1) % checkpoint_every == 0):
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=step + 1,
                scaler=scaler if use_amp else None,
            )

    if checkpoint_path:
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            step=steps,
            scaler=scaler if use_amp else None,
        )

    return {"step": steps, "loss": last_loss}
