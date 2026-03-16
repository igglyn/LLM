from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from shared.config import parse_config, resolve_config
from train.builder import build_model_runtime
from train.blocks import CrossAttentionBlock, DRopeBlock, LayerNormBlock, MixOfExpertsBlock, PosEmbeddingBlock, RoPEBlock, TransformerBlock
from train.specs import ModelRuntime, RuntimeState, RuntimeSchedulerConfig


def load_model_runtime(config_path: str) -> ModelRuntime:
    return build_model_runtime(resolve_config(parse_config(config_path)))


def run_smoke(model_runtime: ModelRuntime, text: str) -> RuntimeState:
    return model_runtime.smoke(text)


def run_dummy_forward(model_runtime: ModelRuntime, batch_size: int, seq_len: int, d_model: int) -> RuntimeState:
    return model_runtime.forward_dummy(batch_size=batch_size, seq_len=seq_len, d_model=d_model)


def train_model(
    model_runtime: ModelRuntime,
    distill_dir: str,
    output_dir: str,
    max_steps: int | None = None,
    resume_from: str | None = None,
    log_every_steps: int = 0,
) -> dict[str, Any]:
    _validate_trunk_blocks(model_runtime)

    distill_root = Path(distill_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    texts, data_files = _load_distill_texts(distill_root)
    token_to_id, token_mapping_file = _load_token_mapping(distill_root, texts)

    configured_steps = max([model_runtime.trunk.train_config.steps, *[p.train_config.steps for p in model_runtime.patchers], 1])
    steps = configured_steps if max_steps is None else min(configured_steps, max_steps)
    configured_batch_size = model_runtime.trunk.train_config.batch_size
    save_every = model_runtime.trunk.train_config.save_every
    optimizer_type = model_runtime.trunk.train_config.optimizer_type
    configured_device = model_runtime.trunk.train_config.device
    has_pos_embedding = _has_positional_embedding(model_runtime)
    has_vocab_embedding = _has_vocab_embedding(model_runtime)

    patchers = model_runtime.patchers if model_runtime.patchers else [model_runtime.trunk]
    _validate_stage_batch_sizes(patchers=patchers, trunk=model_runtime.trunk)

    combined_sequence_len = max(2, model_runtime.trunk.context + sum(max(1, getattr(patcher, "patch_size", 1)) for patcher in patchers))
    context_limit = combined_sequence_len
    sequences = _encode_texts(texts, token_to_id, max_tokens=context_limit)
    vocab_size = max(token_to_id.values()) + 1
    config_d_model = _infer_d_model(model_runtime)
    d_model = config_d_model
    config_n_heads = _infer_n_heads(model_runtime)
    n_heads = _safe_n_heads(d_model, config_n_heads)
    max_seq_len = context_limit
    transformer_layer_count = _count_transformer_layers(model_runtime)
    moe_expert_count = _count_moe_experts(model_runtime)

    trunk_model = _TrainTrunkModel(
        d_model=d_model,
        use_positional_embedding=has_pos_embedding,
        blocks=model_runtime.trunk.blocks,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        dropout=model_runtime.trunk.train_config.dropout,
    )
    trunk_model = trunk_model.to(configured_device)

    trunk_optimizer = _build_optimizer(
        optimizer_type=optimizer_type,
        parameters=trunk_model.parameters(),
        weight_decay=model_runtime.trunk.train_config.weight_decay,
    )
    trunk_schedule_fn = _build_schedule_fn(model_runtime.trunk.train_config.schedulers, total_steps=steps)
    trunk_loss_thresholds = _loss_threshold_schedulers(model_runtime.trunk.train_config.schedulers, total_steps=steps)
    trunk_lr_multiplier = 1.0

    patcher_train_config = patchers[0].train_config
    patcher_encoders: list[_PatcherEncoder] = []
    patcher_decoders: list[_PatcherDecoder] = []
    patch_sizes: list[int] = []
    for idx, patcher in enumerate(patchers):
        patcher_encoders.append(
            _PatcherEncoder(
                d_model=d_model,
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
                use_vocab_embedding=has_vocab_embedding and idx == 0,
                use_positional_embedding=has_pos_embedding,
                blocks=patcher.blocks,
                n_heads=n_heads,
                dropout=patcher.train_config.dropout,
            ).to(configured_device)
        )
        patcher_decoders.append(
            _PatcherDecoder(
                d_model=d_model,
                output_dim=vocab_size if idx == 0 else d_model,
                blocks=patcher.blocks,
                n_heads=n_heads,
                dropout=patcher.train_config.dropout,
            ).to(configured_device)
        )
        patch_sizes.append(max(1, getattr(patcher, 'patch_size', 1)))

    encoder_param_ids = {id(param) for encoder in patcher_encoders for param in encoder.parameters()}
    decoder_param_ids = {id(param) for decoder in patcher_decoders for param in decoder.parameters()}
    if encoder_param_ids & decoder_param_ids:
        raise ValueError("patcher encoder and decoder must not share parameters")

    patcher_optimizer = _build_optimizer(
        optimizer_type=patcher_train_config.optimizer_type,
        parameters=[
            {"params": [param for encoder in patcher_encoders for param in encoder.parameters()], "name": "patcher_encoder"},
            {"params": [param for decoder in patcher_decoders for param in decoder.parameters()], "name": "patcher_decoder"},
        ],
        weight_decay=patcher_train_config.weight_decay,
    )
    patcher_schedule_fn = _build_schedule_fn(patcher_train_config.schedulers, total_steps=steps)
    patcher_loss_thresholds = _loss_threshold_schedulers(patcher_train_config.schedulers, total_steps=steps)
    patcher_lr_multiplier = 1.0

    start_step = 0
    if resume_from:
        start_step = _resume_training_state(
            trunk_model=trunk_model,
            trunk_optimizer=trunk_optimizer,
            patcher_encoders=patcher_encoders,
            patcher_decoders=patcher_decoders,
            patcher_optimizer=patcher_optimizer,
            checkpoint_file=resume_from,
            training_device=configured_device,
        )

    checkpoints_dir = output_path / "checkpoints"
    checkpoint_files: list[str] = []

    losses: list[float] = []
    trunk_losses: list[float] = []
    patcher_losses: list[float] = []
    for step in range(start_step, steps):
        trunk_batch_sequences = [
            sequences[(step * configured_batch_size + batch_index) % len(sequences)]
            for batch_index in range(configured_batch_size)
        ]
        patcher_batch_size = max(1, patcher_train_config.batch_size)
        patcher_batch_sequences = [
            sequences[(step * patcher_batch_size + batch_index) % len(sequences)]
            for batch_index in range(patcher_batch_size)
        ]
        trunk_input_ids, _ = _next_token_batch(trunk_batch_sequences, pad_id=token_to_id["<pad>"])
        patcher_input_ids, _ = _next_token_batch(patcher_batch_sequences, pad_id=token_to_id["<pad>"])
        trunk_input_ids = trunk_input_ids.to(configured_device)
        patcher_input_ids = patcher_input_ids.to(configured_device)

        scheduled_trunk_lr = trunk_schedule_fn(step)
        trunk_lr = scheduled_trunk_lr * trunk_lr_multiplier
        for param_group in trunk_optimizer.param_groups:
            param_group["lr"] = trunk_lr

        scheduled_patcher_lr = patcher_schedule_fn(step)
        patcher_lr = scheduled_patcher_lr * patcher_lr_multiplier
        for param_group in patcher_optimizer.param_groups:
            param_group["lr"] = patcher_lr

        patcher_skip_step = _is_offset_step(patcher_train_config.schedulers, step)
        trunk_skip_step = _is_offset_step(model_runtime.trunk.train_config.schedulers, step)
        patcher_offset_flags = [_is_offset_step(patcher.train_config.schedulers, step) for patcher in patchers]

        pair_losses: list[torch.Tensor] = []
        patcher_input: torch.Tensor = patcher_input_ids
        for idx, (encoder, decoder, patch_size) in enumerate(zip(patcher_encoders, patcher_decoders, patch_sizes)):
            stage_offset = patcher_offset_flags[idx] if idx < len(patcher_offset_flags) else patcher_skip_step
            next_stage_offset = (idx + 1) < len(patcher_offset_flags) and patcher_offset_flags[idx + 1]
            stage_mode = _patcher_stage_execution_mode(stage_offset=stage_offset, next_stage_offset=next_stage_offset)
            if stage_mode == "skip":
                continue

            stage_grad_context = torch.no_grad if stage_mode == "no_grad" else torch.enable_grad
            with stage_grad_context():
                if stage_mode == "grad":
                    patcher_optimizer.zero_grad()
                token_latents = encoder(patcher_input)
                patch_latents = _pool_patch_latents(token_latents, patch_size=patch_size, include_incomplete_tail=True)
                expanded_patch_latents = _expand_patch_latents(patch_latents, patch_size=patch_size, target_len=token_latents.shape[1])
                decoded = decoder(expanded_patch_latents)
                if idx == 0:
                    pair_loss = _masked_token_cross_entropy(
                        logits=decoded,
                        target_ids=patcher_input_ids,
                        pad_id=token_to_id["<pad>"],
                    )
                else:
                    pair_loss = _masked_latent_mse(decoded, patcher_input)

                if stage_mode == "grad":
                    pair_loss.backward()
                    if patcher_train_config.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_([
                            *encoder.parameters(),
                            *decoder.parameters(),
                        ], patcher_train_config.grad_clip)
                    patcher_optimizer.step()
                pair_losses.append(pair_loss.detach())
                patcher_input = patch_latents.detach()

        if pair_losses:
            patcher_loss = torch.stack(pair_losses).mean()
        else:
            patcher_loss = torch.tensor(0.0, device=configured_device)

        if not trunk_skip_step:
            trunk_optimizer.zero_grad()
        with torch.no_grad():
            patcher_stage_sequences = _encode_patcher_stage_sequences(
                encoders=patcher_encoders,
                patch_sizes=patch_sizes,
                model_input=trunk_input_ids,
                include_incomplete_tail=True,
            )

        context_window = max(1, model_runtime.trunk.context)
        final_stage_sequence = patcher_stage_sequences[-1] if patcher_stage_sequences else torch.zeros(
            trunk_input_ids.shape[0],
            0,
            d_model,
            device=configured_device,
        )
        buffer_steps = max(context_window, final_stage_sequence.shape[1] - 1)
        stage_buffers, stage_filled_counts = _init_stage_buffers(
            batch_size=trunk_input_ids.shape[0],
            context=context_window,
            d_model=d_model,
            stage_count=len(patcher_stage_sequences),
            device=configured_device,
        )

        trunk_iteration_losses: list[torch.Tensor] = []
        trunk_grad_context = torch.no_grad if trunk_skip_step else torch.enable_grad
        with trunk_grad_context():
            for iteration in range(max(0, buffer_steps)):
                _update_stage_buffers(
                    stage_buffers=stage_buffers,
                    stage_filled_counts=stage_filled_counts,
                    stage_sequences=patcher_stage_sequences,
                    iteration=iteration,
                )

                final_stage_buffer = stage_buffers[-1] if stage_buffers else torch.zeros(
                    trunk_input_ids.shape[0],
                    context_window,
                    d_model,
                    device=configured_device,
                )
                predicted_patch_latents = trunk_model(final_stage_buffer)[:, -1, :]
                target_latents, target_mask = _next_iteration_target(
                    sequence=final_stage_sequence,
                    iteration=iteration,
                )
                iteration_loss = _masked_latent_mse(
                    predicted_latents=predicted_patch_latents,
                    target_latents=target_latents,
                    mask=target_mask,
                )
                if torch.any(target_mask):
                    trunk_iteration_losses.append(iteration_loss)

        if trunk_iteration_losses:
            next_patch_loss = torch.stack(trunk_iteration_losses).mean()
            if not trunk_skip_step:
                next_patch_loss.backward()
                trunk_optimizer.step()
        else:
            next_patch_loss = torch.tensor(0.0, device=configured_device)

        loss = patcher_loss + next_patch_loss

        detached_loss = float(loss.detach().cpu())
        detached_trunk_loss = float(next_patch_loss.detach().cpu())
        detached_patcher_loss = float(patcher_loss.detach().cpu())

        trunk_lr_multiplier = _apply_loss_threshold_decay(
            step=step,
            loss_value=detached_trunk_loss,
            configured_thresholds=trunk_loss_thresholds,
            lr_multiplier=trunk_lr_multiplier,
        )
        patcher_lr_multiplier = _apply_loss_threshold_decay(
            step=step,
            loss_value=detached_patcher_loss,
            configured_thresholds=patcher_loss_thresholds,
            lr_multiplier=patcher_lr_multiplier,
        )

        updated_trunk_lr = scheduled_trunk_lr * trunk_lr_multiplier
        for param_group in trunk_optimizer.param_groups:
            param_group["lr"] = updated_trunk_lr
        updated_patcher_lr = scheduled_patcher_lr * patcher_lr_multiplier
        for param_group in patcher_optimizer.param_groups:
            param_group["lr"] = updated_patcher_lr
        losses.append(detached_loss)
        trunk_losses.append(detached_trunk_loss)
        patcher_losses.append(detached_patcher_loss)

        if log_every_steps > 0 and (step + 1) % log_every_steps == 0:
            actual_trunk_lr = float(trunk_optimizer.param_groups[0]["lr"])
            actual_patcher_lr = float(patcher_optimizer.param_groups[0]["lr"])
            print(
                f"step={step + 1} loss={detached_loss:.6f} trunk_loss={detached_trunk_loss:.6f} "
                f"patcher_loss={detached_patcher_loss:.6f} trunk_lr={actual_trunk_lr:.8f} patcher_lr={actual_patcher_lr:.8f}"
            )

        if save_every > 0 and (step + 1) % save_every == 0:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoints_dir / f"checkpoint_step_{step + 1}.pt"
            torch.save(
                {
                    "model_state": trunk_model.state_dict(),
                    "optimizer_state": trunk_optimizer.state_dict(),
                    "patcher_encoder_state": patcher_encoders[0].state_dict(),
                    "patcher_decoder_state": patcher_decoders[0].state_dict(),
                    "patcher_encoder_stack_state": [encoder.state_dict() for encoder in patcher_encoders],
                    "patcher_decoder_stack_state": [decoder.state_dict() for decoder in patcher_decoders],
                    "patcher_optimizer_state": patcher_optimizer.state_dict(),
                    "step": step + 1,
                    "loss": detached_loss,
                    "trunk_loss": detached_trunk_loss,
                    "patcher_loss": detached_patcher_loss,
                },
                checkpoint_path,
            )
            checkpoint_files.append(str(checkpoint_path))

    weights_file = output_path / "weights.pt"
    torch.save(
        {
            "model_state": trunk_model.state_dict(),
            "patcher_encoder_state": patcher_encoders[0].state_dict(),
            "patcher_decoder_state": patcher_decoders[0].state_dict(),
            "patcher_encoder_stack_state": [encoder.state_dict() for encoder in patcher_encoders],
            "patcher_decoder_stack_state": [decoder.state_dict() for decoder in patcher_decoders],
            "meta": {
                "vocab_size": vocab_size,
                "d_model": d_model,
                "config_d_model": config_d_model,
                "config_n_heads": config_n_heads,
                "n_heads": n_heads,
                "max_seq_len": max_seq_len,
                "context_limit": context_limit,
                "use_vocab_embedding": has_vocab_embedding,
                "use_positional_embedding": has_pos_embedding,
                "transformer_layers": transformer_layer_count,
                "moe_expert_count": moe_expert_count,
                "pad_id": token_to_id["<pad>"],
                "unk_id": token_to_id["<unk>"],
                "token_to_id": token_to_id,
                "batch_size": configured_batch_size,
                "save_every": save_every,
                "optimizer_type": optimizer_type,
                "training_device": configured_device,
                "grad_clip": model_runtime.trunk.train_config.grad_clip,
                "schedulers": [
                    {"type": scheduler.scheduler_type, "attributes": dict(scheduler.attributes)}
                    for scheduler in model_runtime.trunk.train_config.schedulers
                ],
                "prediction_target": "patch_latent_next_state_mse_plus_patcher_reconstruction",
                "latent_dim": trunk_model.latent_dim,
                "decoder_head": "patcher_decoder_for_reconstruction_only",
                "patch_size": patch_sizes[0],
                "patch_sizes": patch_sizes,
            },
        },
        weights_file,
    )

    initial_loss = losses[0] if losses else 0.0
    final_loss = losses[-1] if losses else 0.0
    initial_trunk_loss = trunk_losses[0] if trunk_losses else 0.0
    final_trunk_loss = trunk_losses[-1] if trunk_losses else 0.0
    initial_patcher_loss = patcher_losses[0] if patcher_losses else 0.0
    final_patcher_loss = patcher_losses[-1] if patcher_losses else 0.0

    return {
        "configured_steps": configured_steps,
        "steps": steps,
        "start_step": start_step,
        "configured_batch_size": configured_batch_size,
        "save_every": save_every,
        "optimizer_type": optimizer_type,
        "training_device": configured_device,
        "grad_clip": model_runtime.trunk.train_config.grad_clip,
        "checkpoint_files": checkpoint_files,
        "dataset_examples": len(texts),
        "data_files": data_files,
        "token_mapping_file": token_mapping_file,
        "used_vocab_embedding": has_vocab_embedding,
        "used_positional_embedding": has_pos_embedding,
        "transformer_layers": transformer_layer_count,
        "moe_expert_count": moe_expert_count,
        "config_d_model": config_d_model,
        "config_n_heads": config_n_heads,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_trunk_loss": initial_trunk_loss,
        "final_trunk_loss": final_trunk_loss,
        "initial_patcher_loss": initial_patcher_loss,
        "final_patcher_loss": final_patcher_loss,
        "weights_file": str(weights_file),
        "final_trunk_lr": float(trunk_optimizer.param_groups[0]["lr"]),
        "final_patcher_lr": float(patcher_optimizer.param_groups[0]["lr"]),
    }


def run_trained_model(
    model_runtime: ModelRuntime,
    weights_file: str,
    text: str,
    max_new_tokens: int = 1,
) -> dict[str, Any]:
    _validate_trunk_blocks(model_runtime)

    payload = torch.load(weights_file, map_location="cpu")
    state_dict = payload.get("model_state")
    meta = payload.get("meta", {})
    if not isinstance(state_dict, dict):
        raise ValueError(f"weights file is missing model_state: {weights_file}")
    if not isinstance(meta, dict):
        raise ValueError(f"weights file is missing meta dictionary: {weights_file}")

    token_to_id = meta.get("token_to_id")
    if not isinstance(token_to_id, dict):
        raise ValueError(f"weights file is missing token_to_id: {weights_file}")

    d_model = int(meta.get("d_model", _infer_d_model(model_runtime)))
    config_d_model = int(meta.get("config_d_model", d_model))
    config_n_heads = int(meta.get("config_n_heads", _infer_n_heads(model_runtime)))

    vocab_size = int(meta.get("vocab_size", max(int(v) for v in token_to_id.values()) + 1))
    max_seq_len = int(meta.get("max_seq_len", 8))
    use_vocab_embedding = bool(meta.get("use_vocab_embedding", _has_vocab_embedding(model_runtime)))
    use_positional_embedding = bool(meta.get("use_positional_embedding", _has_positional_embedding(model_runtime)))
    transformer_layers = int(meta.get("transformer_layers", _count_transformer_layers(model_runtime)))
    moe_expert_count = int(meta.get("moe_expert_count", _count_moe_experts(model_runtime)))
    n_heads = int(meta.get("n_heads", config_n_heads))
    if n_heads != config_n_heads:
        raise ValueError(
            f"weights n_heads mismatch for config: expected compatible n_heads={config_n_heads} for d_model={d_model}, got {n_heads}"
        )

    model = _TrainTrunkModel(
        d_model=d_model,
        use_positional_embedding=use_positional_embedding,
        blocks=model_runtime.trunk.blocks,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        dropout=float(meta.get("dropout", model_runtime.trunk.train_config.dropout)),
    )
    model.load_state_dict(state_dict)
    model.eval()

    patcher_encoder_state = payload.get("patcher_encoder_state")
    patcher_decoder_state = payload.get("patcher_decoder_state")
    patcher_encoder_stack_state = payload.get("patcher_encoder_stack_state")
    if not isinstance(patcher_encoder_state, dict):
        raise ValueError(f"weights file is missing patcher_encoder_state: {weights_file}")
    if not isinstance(patcher_decoder_state, dict):
        raise ValueError(f"weights file is missing patcher_decoder_state: {weights_file}")
    patcher_encoder = _PatcherEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        use_vocab_embedding=use_vocab_embedding,
        use_positional_embedding=use_positional_embedding,
        blocks=model_runtime.patchers[0].blocks if model_runtime.patchers else model_runtime.trunk.blocks,
        n_heads=n_heads,
        dropout=float(meta.get("dropout", model_runtime.trunk.train_config.dropout)),
    )
    patcher_encoder.load_state_dict(patcher_encoder_state)
    patcher_encoder.eval()

    patcher_decoder = _PatcherDecoder(
        output_dim=vocab_size,
        d_model=d_model,
        blocks=model_runtime.patchers[0].blocks if model_runtime.patchers else model_runtime.trunk.blocks,
        n_heads=n_heads,
        dropout=float(meta.get("dropout", model_runtime.trunk.train_config.dropout)),
    )
    patcher_decoder.load_state_dict(patcher_decoder_state)
    patcher_decoder.eval()

    patcher_encoders = [patcher_encoder]
    patch_sizes = [int(meta.get("patch_size", 1))]
    if isinstance(patcher_encoder_stack_state, list) and len(patcher_encoder_stack_state) == len(model_runtime.patchers):
        patcher_encoders = []
        patch_sizes = []
        for idx, patcher in enumerate(model_runtime.patchers):
            encoder = _PatcherEncoder(
                vocab_size=vocab_size,
                d_model=d_model,
                max_seq_len=max_seq_len,
                use_vocab_embedding=use_vocab_embedding and idx == 0,
                use_positional_embedding=use_positional_embedding,
                blocks=patcher.blocks,
                n_heads=n_heads,
                dropout=float(meta.get("dropout", model_runtime.trunk.train_config.dropout)),
            )
            encoder.load_state_dict(patcher_encoder_stack_state[idx])
            encoder.eval()
            patcher_encoders.append(encoder)
            patch_sizes.append(max(1, patcher.patch_size))

    encoded = _encode_texts([text], {str(k): int(v) for k, v in token_to_id.items()}, max_tokens=max_seq_len)[0]
    pad_id = int(meta.get("pad_id", 0))

    generated_ids: list[int] = []
    generated_scores: list[float] = []
    for _ in range(max(0, max_new_tokens)):
        window = encoded[-max_seq_len:]
        input_ids, _ = _next_token_batch([window], pad_id=pad_id)
        patch_sequence = _encode_patcher_stack(
            encoders=patcher_encoders,
            patch_sizes=patch_sizes,
            model_input=input_ids,
            include_incomplete_tail=True,
        )
        if patch_sequence.shape[1] == 0:
            break
        context = max(1, int(meta.get("context_limit", model_runtime.trunk.context)))
        patch_input = patch_sequence[:, -context:, :]
        with torch.no_grad():
            predicted_patch = model(patch_input)[:, -1:, :]
            logits = patcher_decoder(predicted_patch)

        next_token_logits = logits[0, -1]
        next_token_id = int(torch.argmax(next_token_logits).item())
        encoded.append(next_token_id)
        generated_ids.append(next_token_id)
        generated_scores.append(float(torch.max(next_token_logits).item()))

    id_to_token = {int(v): str(k) for k, v in token_to_id.items()}
    generated_tokens = [id_to_token.get(token_id, "<unk>") for token_id in generated_ids]

    state = model_runtime.smoke(text)
    return {
        "text": text,
        "predicted_tokens": generated_tokens,
        "predicted_token_ids": generated_ids,
        "scores": generated_scores,
        "max_new_tokens": max(0, max_new_tokens),
        "d_model": d_model,
        "used_vocab_embedding": use_vocab_embedding,
        "transformer_layers": transformer_layers,
        "n_heads": n_heads,
        "moe_expert_count": moe_expert_count,
        "config_d_model": config_d_model,
        "config_n_heads": config_n_heads,
        "trace": state.execution_trace,
        "moe_metrics": state.moe_metrics,
    }


class _PatcherEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        max_seq_len: int,
        use_vocab_embedding: bool,
        use_positional_embedding: bool,
        blocks: list[Any],
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_embedding = nn.Embedding(vocab_size, d_model) if use_vocab_embedding else None
        self.token_projection = nn.Linear(1, d_model) if not use_vocab_embedding else None
        self.positional_embedding = nn.Embedding(max_seq_len, d_model) if use_positional_embedding else None
        self.layers = _compile_decoder_layers(blocks=blocks, fallback_d_model=d_model, fallback_n_heads=n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.latent_head = nn.Linear(d_model, d_model)

    def _encode_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.vocab_embedding is not None:
            x = self.vocab_embedding(input_ids)
        elif self.token_projection is not None:
            x = self.token_projection(input_ids.to(torch.float32).unsqueeze(-1))
        else:
            raise ValueError("model is missing both vocab_embedding and token_projection")

        if self.positional_embedding is not None:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            x = x + self.positional_embedding(positions)
        return x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = input_ids if input_ids.dtype.is_floating_point else self._encode_tokens(input_ids)
        seq_len = x.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            if isinstance(layer, _TransformerDecoderBlock):
                x = layer(x, causal_mask)
            else:
                x = layer(x)
        x = self.norm(x)
        return self.latent_head(x)


class _PatcherDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        output_dim: int,
        blocks: list[Any],
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = _compile_decoder_layers(blocks=blocks, fallback_d_model=d_model, fallback_n_heads=n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        x = latents
        seq_len = x.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            if isinstance(layer, _TransformerDecoderBlock):
                x = layer(x, causal_mask)
            else:
                x = layer(x)
        x = self.norm(x)
        return self.head(x)


class _TrainTrunkModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        use_positional_embedding: bool,
        blocks: list[Any],
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.positional_embedding = nn.Embedding(max_seq_len, d_model) if use_positional_embedding else None
        self.layers = _compile_decoder_layers(blocks=blocks, fallback_d_model=d_model, fallback_n_heads=n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.latent_dim = d_model
        self.latent_head = nn.Linear(d_model, self.latent_dim)

    def forward(self, patch_latents: torch.Tensor) -> torch.Tensor:
        x = patch_latents
        if self.positional_embedding is not None:
            positions = torch.arange(patch_latents.shape[1], device=patch_latents.device).unsqueeze(0)
            x = x + self.positional_embedding(positions)

        seq_len = x.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            if isinstance(layer, _TransformerDecoderBlock):
                x = layer(x, causal_mask)
            else:
                x = layer(x)
        x = self.norm(x)
        return self.latent_head(x)


class _TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, moe_expert_count: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.moe_gate = nn.Linear(d_model, moe_expert_count) if moe_expert_count > 0 else None
        self.moe_experts = (
            nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model), nn.GELU()) for _ in range(moe_expert_count)])
            if moe_expert_count > 0
            else nn.ModuleList()
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, attn_mask=causal_mask, need_weights=False)
        x = x + self.attn_dropout(attn_out)

        ffn_input = self.ffn_norm(x)
        if self.moe_gate is not None and self.moe_experts:
            gate = torch.softmax(self.moe_gate(ffn_input), dim=-1)
            expert_outputs = torch.stack([expert(ffn_input) for expert in self.moe_experts], dim=-2)
            ffn_out = torch.sum(expert_outputs * gate.unsqueeze(-1), dim=-2)
        else:
            ffn_out = self.ffn(ffn_input)
        return x + self.ffn_dropout(ffn_out)


class _TrainRoPE(nn.Module):
    def __init__(self, base: float, scale: float) -> None:
        super().__init__()
        self.base = base
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < 2:
            return x
        even = x[..., 0::2]
        odd = x[..., 1::2]
        angle = (1.0 / self.base) * self.scale
        cos = torch.cos(torch.tensor(angle, dtype=x.dtype, device=x.device))
        sin = torch.sin(torch.tensor(angle, dtype=x.dtype, device=x.device))
        out = x.clone()
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return out


class _TrainDRope(nn.Module):
    def __init__(self, base: float, scale: float) -> None:
        super().__init__()
        self.base = base
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decay = 1.0 / (1.0 + (self.scale / self.base))
        positions = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
        mask = torch.exp(-positions * (1.0 - decay)).view(1, -1, 1)
        return x * mask


def _compile_decoder_layers(blocks: list[Any], fallback_d_model: int, fallback_n_heads: int, dropout: float) -> nn.ModuleList:
    layers: list[nn.Module] = []
    for block in blocks:
        if isinstance(block, (TransformerBlock, CrossAttentionBlock)):
            layers.append(
                _TransformerDecoderBlock(
                    d_model=getattr(block, "d_model", fallback_d_model),
                    n_heads=getattr(block, "n_heads", fallback_n_heads),
                    moe_expert_count=0,
                    dropout=dropout,
                )
            )
        elif isinstance(block, MixOfExpertsBlock):
            expert_count = max(0, len(block.experts))
            layers.append(
                _TransformerDecoderBlock(
                    d_model=fallback_d_model,
                    n_heads=fallback_n_heads,
                    moe_expert_count=expert_count,
                    dropout=dropout,
                )
            )
        elif isinstance(block, LayerNormBlock):
            layers.append(nn.LayerNorm(fallback_d_model, eps=block.eps))
        elif isinstance(block, RoPEBlock):
            layers.append(_TrainRoPE(base=block.base, scale=block.scale))
        elif isinstance(block, DRopeBlock):
            layers.append(_TrainDRope(base=block.base, scale=block.scale))
        elif isinstance(block, PosEmbeddingBlock):
            continue

    return nn.ModuleList(layers)


def _masked_token_cross_entropy(logits: torch.Tensor, target_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        ignore_index=pad_id,
    )


def _masked_latent_mse(predicted_latents: torch.Tensor, target_latents: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if predicted_latents.numel() == 0 or target_latents.numel() == 0:
        return torch.tensor(0.0, device=predicted_latents.device if predicted_latents.numel() > 0 else target_latents.device)
    if mask is None:
        return torch.mean((predicted_latents - target_latents) ** 2)

    broadcast_mask = mask.to(dtype=predicted_latents.dtype).view(-1, 1)
    valid = torch.sum(broadcast_mask)
    if valid <= 0:
        return torch.tensor(0.0, device=predicted_latents.device)
    squared_error = (predicted_latents - target_latents) ** 2
    return torch.sum(squared_error * broadcast_mask) / (valid * predicted_latents.shape[-1])


def _build_optimizer(optimizer_type: str, parameters: Any, weight_decay: float) -> torch.optim.Optimizer:
    if optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(parameters, lr=3e-3, weight_decay=weight_decay)
    if optimizer_type.lower() == "sgd":
        return torch.optim.SGD(parameters, lr=3e-3, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer type in config: {optimizer_type}")


def _build_schedule_fn(schedulers: list[RuntimeSchedulerConfig], total_steps: int) -> Callable[[int], float]:
    active_schedulers = [scheduler for scheduler in schedulers if scheduler.scheduler_type != "loss_threshold"]
    if not active_schedulers:
        return lambda _step: 3e-3

    intervals: list[tuple[int, int, str, dict[str, str]]] = []
    for scheduler in active_schedulers:
        start = int(scheduler.attributes.get("start_step", "0"))
        end = int(scheduler.attributes.get("end_step", str(total_steps)))
        intervals.append((start, end, scheduler.scheduler_type, dict(scheduler.attributes)))

    def _lr(step: int) -> float:
        for start, end, scheduler_type, attrs in intervals:
            if start <= step < end:
                min_lr = float(attrs.get("min_lr", "1e-5"))
                max_lr = float(attrs.get("max_lr", "3e-3"))
                if scheduler_type == "warmup":
                    span = max(1, end - start)
                    pct = (step - start) / span
                    return min_lr + (max_lr - min_lr) * pct
                if scheduler_type == "cosine":
                    span = max(1, end - start)
                    pct = (step - start) / span
                    return min_lr + 0.5 * (max_lr - min_lr) * (1 + torch.cos(torch.tensor(pct * 3.1415926535)).item())
                return max_lr
        return 1e-5

    return _lr


def _loss_threshold_schedulers(schedulers: list[RuntimeSchedulerConfig], total_steps: int) -> list[dict[str, Any]]:
    configured: list[dict[str, Any]] = []
    for scheduler in schedulers:
        if scheduler.scheduler_type != "loss_threshold":
            continue
        attrs = dict(scheduler.attributes)
        configured.append(
            {
                "start": int(attrs.get("start_step", "0")),
                "end": int(attrs.get("end_step", str(total_steps))),
                "threshold": float(attrs.get("threshold", "0.0")),
                "decay_factor": float(attrs.get("decay_factor", "1.0")),
                "monitor": str(attrs.get("monitor", "train_loss")),
                "triggered": False,
            }
        )
    return configured


def _apply_loss_threshold_decay(
    step: int,
    loss_value: float,
    configured_thresholds: list[dict[str, Any]],
    lr_multiplier: float,
) -> float:
    updated_multiplier = lr_multiplier
    for threshold in configured_thresholds:
        if threshold["triggered"]:
            continue
        if not threshold["start"] <= step < threshold["end"]:
            continue
        if loss_value <= threshold["threshold"]:
            updated_multiplier *= threshold["decay_factor"]
            threshold["triggered"] = True
    return updated_multiplier



def _patcher_stage_execution_mode(stage_offset: bool, next_stage_offset: bool) -> str:
    if next_stage_offset:
        return "skip"
    if stage_offset:
        return "no_grad"
    return "grad"


def _is_offset_step(schedulers: list[RuntimeSchedulerConfig], step: int) -> bool:
    for scheduler in schedulers:
        if scheduler.scheduler_type != "offset":
            continue
        start = int(scheduler.attributes.get("start_step", "0"))
        end = int(scheduler.attributes.get("end_step", "0"))
        if start <= step < end:
            return True
    return False


def _resume_training_state(
    trunk_model: nn.Module,
    trunk_optimizer: torch.optim.Optimizer,
    patcher_encoders: list[nn.Module],
    patcher_decoders: list[nn.Module],
    patcher_optimizer: torch.optim.Optimizer,
    checkpoint_file: str,
    training_device: str,
) -> int:
    payload = torch.load(checkpoint_file, map_location=training_device)
    model_state = payload.get("model_state")
    optimizer_state = payload.get("optimizer_state")
    if not isinstance(model_state, dict) or not isinstance(optimizer_state, dict):
        raise ValueError(f"checkpoint file is missing model/optimizer states: {checkpoint_file}")
    trunk_model.load_state_dict(model_state)
    trunk_optimizer.load_state_dict(optimizer_state)

    patcher_encoder_state = payload.get("patcher_encoder_state")
    patcher_decoder_state = payload.get("patcher_decoder_state")
    patcher_encoder_stack_state = payload.get("patcher_encoder_stack_state")
    patcher_decoder_stack_state = payload.get("patcher_decoder_stack_state")
    patcher_optimizer_state = payload.get("patcher_optimizer_state")
    if (
        isinstance(patcher_encoder_stack_state, list)
        and isinstance(patcher_decoder_stack_state, list)
        and len(patcher_encoder_stack_state) == len(patcher_encoders)
        and len(patcher_decoder_stack_state) == len(patcher_decoders)
        and isinstance(patcher_optimizer_state, dict)
    ):
        for encoder, state in zip(patcher_encoders, patcher_encoder_stack_state):
            encoder.load_state_dict(state)
        for decoder, state in zip(patcher_decoders, patcher_decoder_stack_state):
            decoder.load_state_dict(state)
        patcher_optimizer.load_state_dict(patcher_optimizer_state)
    elif isinstance(patcher_encoder_state, dict) and isinstance(patcher_decoder_state, dict) and isinstance(patcher_optimizer_state, dict):
        patcher_encoders[0].load_state_dict(patcher_encoder_state)
        patcher_decoders[0].load_state_dict(patcher_decoder_state)
        patcher_optimizer.load_state_dict(patcher_optimizer_state)

    return int(payload.get("step", 0))


def _count_transformer_layers(model_runtime: ModelRuntime) -> int:
    count = 0
    for patcher in model_runtime.patchers:
        count += sum(1 for block in patcher.blocks if isinstance(block, TransformerBlock))
    count += sum(1 for block in model_runtime.trunk.blocks if isinstance(block, TransformerBlock))
    for block in model_runtime.trunk.blocks:
        if isinstance(block, MixOfExpertsBlock):
            for expert in block.experts:
                count += sum(1 for inner in expert.blocks if isinstance(inner, TransformerBlock))
    return max(1, count)


def _count_patcher_transformer_layers(model_runtime: ModelRuntime) -> int:
    count = 0
    for patcher in model_runtime.patchers:
        count += sum(1 for block in patcher.blocks if isinstance(block, TransformerBlock))
    return max(1, count)


def _count_moe_experts(model_runtime: ModelRuntime) -> int:
    return sum(len(block.experts) for block in model_runtime.trunk.blocks if isinstance(block, MixOfExpertsBlock))


def _load_distill_texts(distill_root: Path) -> tuple[list[str], list[str]]:
    if not distill_root.exists() or not distill_root.is_dir():
        raise ValueError(f"distill_dir does not exist or is not a directory: {distill_root}")

    preferred = ["stage_a.jsonl", "mixed.jsonl"]
    jsonl_paths: list[Path] = []
    for name in preferred:
        candidate = distill_root / name
        if candidate.exists():
            jsonl_paths.append(candidate)
    if not jsonl_paths:
        jsonl_paths = sorted(distill_root.glob("*.jsonl"))
    if not jsonl_paths:
        raise ValueError(f"distill_dir must contain stage_a.jsonl, mixed.jsonl, or other .jsonl files: {distill_root}")

    texts: list[str] = []
    for path in jsonl_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            text = _extract_text_from_distill_row(row)
            if text:
                texts.append(text)
    if not texts:
        raise ValueError(f"no trainable text found in distill data under: {distill_root}")

    return texts, [str(path) for path in jsonl_paths]


def _extract_text_from_distill_row(row: dict[str, Any]) -> str:
    if isinstance(row.get("text"), str):
        return row["text"]
    prompt = row.get("prompt_text")
    target = row.get("target_text")
    if isinstance(prompt, str) and isinstance(target, str):
        return f"{prompt} {target}".strip()
    if isinstance(prompt, str):
        return prompt
    return ""


def _load_token_mapping(distill_root: Path, texts: list[str]) -> tuple[dict[str, int], str | None]:
    for name in ["token_mappings.json", "token_mapping.json", "vocab.json"]:
        path = distill_root / name
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "token_to_id" in raw and isinstance(raw["token_to_id"], dict):
            mapping = {str(k): int(v) for k, v in raw["token_to_id"].items()}
            return _ensure_special_tokens(mapping), str(path)
        if isinstance(raw, dict):
            maybe = {str(k): int(v) for k, v in raw.items() if isinstance(v, int)}
            if maybe:
                return _ensure_special_tokens(maybe), str(path)
        if isinstance(raw, list) and all(isinstance(token, str) for token in raw):
            mapping = {token: idx for idx, token in enumerate(raw)}
            return _ensure_special_tokens(mapping), str(path)

    for name in ["token_mapping.jsonl", "token_mappings.jsonl", "token_definitions.jsonl"]:
        path = distill_root / name
        if not path.exists():
            continue

        mapping: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                continue

            token = row.get("token")
            mapped_id = row.get("mapped_id")
            if isinstance(token, str) and isinstance(mapped_id, int):
                mapping[token] = mapped_id
                continue

            token_id = row.get("id")
            if isinstance(token, str) and isinstance(token_id, int):
                mapping[token] = token_id

        if mapping:
            return _ensure_special_tokens(mapping), str(path)

    built = {"<pad>": 0, "<unk>": 1}
    for text in texts:
        for token in text.split():
            if token not in built:
                built[token] = len(built)
    return built, None


def _ensure_special_tokens(mapping: dict[str, int]) -> dict[str, int]:
    normalized = {str(token): int(index) for token, index in mapping.items()}
    if "<pad>" not in normalized:
        normalized["<pad>"] = max(normalized.values(), default=-1) + 1
    if "<unk>" not in normalized:
        normalized["<unk>"] = max(normalized.values(), default=-1) + 1
    return normalized


def _encode_texts(texts: list[str], token_to_id: dict[str, int], max_tokens: int) -> list[list[int]]:
    unk_id = token_to_id["<unk>"]
    pad_id = token_to_id["<pad>"]
    sequences: list[list[int]] = []
    for text in texts:
        ids = [token_to_id.get(token, unk_id) for token in text.split()[:max_tokens]]
        if len(ids) < 2:
            ids = [*ids, pad_id]
        sequences.append(ids)
    return sequences


def _next_token_batch(sequences: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(sequence) for sequence in sequences)
    input_rows: list[list[int]] = []
    target_rows: list[list[int]] = []

    for sequence in sequences:
        padded = [*sequence, *([pad_id] * (max_len - len(sequence)))]
        input_rows.append(padded[:-1])
        target_rows.append(padded[1:])

    input_ids = torch.tensor(input_rows, dtype=torch.long)
    target_ids = torch.tensor(target_rows, dtype=torch.long)
    return input_ids, target_ids


def _pool_patch_latents(
    patcher_latents: torch.Tensor,
    patch_size: int,
    include_incomplete_tail: bool = False,
) -> torch.Tensor:
    step = max(1, patch_size)
    pooled = patcher_latents[:, step - 1 :: step, :]
    if include_incomplete_tail and patcher_latents.shape[1] > 0 and patcher_latents.shape[1] % step != 0:
        pooled = torch.cat([pooled, patcher_latents[:, -1:, :]], dim=1)
    return pooled




def _expand_patch_latents(patch_latents: torch.Tensor, patch_size: int, target_len: int) -> torch.Tensor:
    expanded = patch_latents.repeat_interleave(max(1, patch_size), dim=1)
    if expanded.shape[1] < target_len:
        pad = expanded[:, -1:, :].expand(-1, target_len - expanded.shape[1], -1) if expanded.shape[1] > 0 else torch.zeros(patch_latents.shape[0], target_len, patch_latents.shape[2], device=patch_latents.device, dtype=patch_latents.dtype)
        expanded = torch.cat([expanded, pad], dim=1)
    return expanded[:, :target_len, :]


def _context_patch_prediction_batch(patch_sequence: torch.Tensor, context: int) -> tuple[torch.Tensor, torch.Tensor]:
    if patch_sequence.shape[1] <= context:
        empty = patch_sequence[:, :0, :]
        return empty, empty
    inputs = []
    targets = []
    for b in range(patch_sequence.shape[0]):
        for t in range(context, patch_sequence.shape[1]):
            inputs.append(patch_sequence[b, t - context : t, :])
            targets.append(patch_sequence[b, t, :])
    if not inputs:
        empty = patch_sequence[:, :0, :]
        return empty, empty
    return torch.stack(inputs, dim=0), torch.stack(targets, dim=0)


def _encode_patcher_stack(encoders: list[_PatcherEncoder], patch_sizes: list[int], model_input: torch.Tensor, include_incomplete_tail: bool = False) -> torch.Tensor:
    encoded: torch.Tensor = model_input
    for encoder, patch_size in zip(encoders, patch_sizes):
        token_latents = encoder(encoded)
        encoded = _pool_patch_latents(token_latents, patch_size=patch_size, include_incomplete_tail=include_incomplete_tail)
    return encoded

def _next_patch_latent_batch(patcher_latents: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    patch_sequence = _pool_patch_latents(patcher_latents, patch_size)
    if patch_sequence.shape[1] < 2:
        empty = patch_sequence[:, :0, :]
        return empty, empty
    return patch_sequence[:, :-1, :], patch_sequence[:, 1:, :]


def _validate_stage_batch_sizes(patchers: list[Any], trunk: Any) -> None:
    stage_batch_sizes = [max(1, int(getattr(patcher.train_config, "batch_size", 1))) for patcher in patchers]
    stage_batch_sizes.append(max(1, int(trunk.train_config.batch_size)))
    for idx in range(len(stage_batch_sizes) - 1):
        previous = stage_batch_sizes[idx]
        current = stage_batch_sizes[idx + 1]
        if previous % current != 0:
            raise ValueError(
                "Batch size constraint violation between stages: "
                f"stage {idx} batch_size={previous} must be equal to or a multiple of stage {idx + 1} batch_size={current}."
            )


def _encode_patcher_stage_sequences(
    encoders: list[_PatcherEncoder],
    patch_sizes: list[int],
    model_input: torch.Tensor,
    include_incomplete_tail: bool = False,
) -> list[torch.Tensor]:
    encoded: torch.Tensor = model_input
    stage_outputs: list[torch.Tensor] = []
    for encoder, patch_size in zip(encoders, patch_sizes):
        token_latents = encoder(encoded)
        encoded = _pool_patch_latents(token_latents, patch_size=patch_size, include_incomplete_tail=include_incomplete_tail)
        stage_outputs.append(encoded)
    return stage_outputs


def _init_stage_buffers(
    batch_size: int,
    context: int,
    d_model: int,
    stage_count: int,
    device: torch.device | str,
) -> tuple[list[torch.Tensor], list[int]]:
    buffers = [torch.zeros(batch_size, context, d_model, device=device) for _ in range(stage_count)]
    filled_counts = [0 for _ in range(stage_count)]
    return buffers, filled_counts


def _update_stage_buffers(
    stage_buffers: list[torch.Tensor],
    stage_filled_counts: list[int],
    stage_sequences: list[torch.Tensor],
    iteration: int,
) -> None:
    for stage_idx, (buffer, sequence) in enumerate(zip(stage_buffers, stage_sequences)):
        if iteration >= sequence.shape[1]:
            continue
        filled = stage_filled_counts[stage_idx]
        next_latents = sequence[:, iteration, :]
        if filled < buffer.shape[1]:
            buffer[:, filled, :] = next_latents
            stage_filled_counts[stage_idx] = filled + 1
        else:
            buffer[:, :-1, :] = buffer[:, 1:, :].clone()
            buffer[:, -1, :] = next_latents


def _next_iteration_target(sequence: torch.Tensor, iteration: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = sequence.shape[0]
    d_model = sequence.shape[2] if sequence.dim() == 3 else 0
    if iteration + 1 >= sequence.shape[1]:
        return (
            torch.zeros(batch_size, d_model, device=sequence.device, dtype=sequence.dtype),
            torch.zeros(batch_size, device=sequence.device, dtype=torch.bool),
        )
    return sequence[:, iteration + 1, :], torch.ones(batch_size, device=sequence.device, dtype=torch.bool)


def _validate_trunk_blocks(model_runtime: ModelRuntime) -> None:
    if any(block.block_name == "VocabEmbedding" for block in model_runtime.trunk.blocks):
        raise ValueError("Trunk must not contain VocabEmbedding blocks for patch-latent training.")


def _has_positional_embedding(model_runtime: ModelRuntime) -> bool:
    patcher_has_pos = any(block.block_name == "PosEmbedding" for patcher in model_runtime.patchers for block in patcher.blocks)
    trunk_has_pos = any(block.block_name == "PosEmbedding" for block in model_runtime.trunk.blocks)
    return patcher_has_pos or trunk_has_pos


def _has_vocab_embedding(model_runtime: ModelRuntime) -> bool:
    patcher_has_vocab = any(block.block_name == "VocabEmbedding" for patcher in model_runtime.patchers for block in patcher.blocks)
    return patcher_has_vocab


def _infer_d_model(model_runtime: ModelRuntime) -> int:
    for patcher in model_runtime.patchers:
        for block in patcher.blocks:
            d_model = getattr(block, "d_model", None)
            if isinstance(d_model, int) and d_model > 0:
                return d_model
    for block in model_runtime.trunk.blocks:
        d_model = getattr(block, "d_model", None)
        if isinstance(d_model, int) and d_model > 0:
            return d_model
    return 128


def _infer_n_heads(model_runtime: ModelRuntime) -> int:
    for patcher in model_runtime.patchers:
        for block in patcher.blocks:
            n_heads = getattr(block, "n_heads", None)
            if isinstance(n_heads, int) and n_heads > 0:
                return n_heads
    for block in model_runtime.trunk.blocks:
        n_heads = getattr(block, "n_heads", None)
        if isinstance(n_heads, int) and n_heads > 0:
            return n_heads
    return 8



def _safe_n_heads(d_model: int, configured_heads: int) -> int:
    candidate = max(1, min(configured_heads, d_model))
    while d_model % candidate != 0 and candidate > 1:
        candidate -= 1
    return max(1, candidate)


def generate_tokens(
    model_runtime: ModelRuntime,
    weights_file: str,
    text: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    result = run_trained_model(
        model_runtime=model_runtime,
        weights_file=weights_file,
        text=text,
        max_new_tokens=max_new_tokens,
    )
    return {
        "text": text,
        "generated_tokens": result["predicted_tokens"],
        "generated_token_ids": result["predicted_token_ids"],
        "max_new_tokens": max_new_tokens,
    }


__all__ = [
    "ModelRuntime",
    "RuntimeState",
    "build_model_runtime",
    "load_model_runtime",
    "run_smoke",
    "run_dummy_forward",
    "train_model",
    "run_trained_model",
    "generate_tokens",
]
