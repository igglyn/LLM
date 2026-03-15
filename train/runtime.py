from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from shared.config import parse_config, resolve_config
from train.builder import build_model_runtime
from train.blocks import MixOfExpertsBlock, TransformerBlock
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
    has_pos_embedding = _has_positional_embedding(model_runtime)
    has_vocab_embedding = _has_vocab_embedding(model_runtime)

    context_limit = max(2, model_runtime.trunk.context)
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
        transformer_layers=transformer_layer_count,
        n_heads=n_heads,
        moe_expert_count=moe_expert_count,
        dropout=model_runtime.trunk.train_config.dropout,
    )
    trunk_optimizer = _build_optimizer(
        optimizer_type=optimizer_type,
        parameters=trunk_model.parameters(),
        weight_decay=model_runtime.trunk.train_config.weight_decay,
    )
    trunk_schedule_fn = _build_schedule_fn(model_runtime.trunk.train_config.schedulers, total_steps=steps)

    patcher_layer_count = _count_patcher_transformer_layers(model_runtime)
    patcher_train_config = model_runtime.patchers[0].train_config if model_runtime.patchers else model_runtime.trunk.train_config
    patcher_model = _TrainPatcherModel(
        d_model=d_model,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        use_vocab_embedding=has_vocab_embedding,
        use_positional_embedding=has_pos_embedding,
        transformer_layers=max(1, patcher_layer_count),
        n_heads=n_heads,
        moe_expert_count=0,
        dropout=patcher_train_config.dropout,
    )
    patcher_optimizer = _build_optimizer(
        optimizer_type=patcher_train_config.optimizer_type,
        parameters=patcher_model.parameters(),
        weight_decay=patcher_train_config.weight_decay,
    )
    patcher_schedule_fn = _build_schedule_fn(patcher_train_config.schedulers, total_steps=steps)

    start_step = 0
    if resume_from:
        start_step = _resume_training_state(
            trunk_model=trunk_model,
            trunk_optimizer=trunk_optimizer,
            patcher_model=patcher_model,
            patcher_optimizer=patcher_optimizer,
            checkpoint_file=resume_from,
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

        trunk_lr = trunk_schedule_fn(step)
        for param_group in trunk_optimizer.param_groups:
            param_group["lr"] = trunk_lr

        patcher_lr = patcher_schedule_fn(step)
        for param_group in patcher_optimizer.param_groups:
            param_group["lr"] = patcher_lr

        patcher_skip_step = _is_offset_step(patcher_train_config.schedulers, step)
        trunk_skip_step = _is_offset_step(model_runtime.trunk.train_config.schedulers, step)

        if patcher_skip_step:
            patcher_loss = torch.tensor(0.0)
        else:
            patcher_logits = patcher_model.reconstruct_logits(patcher_input_ids)
            patcher_loss = _masked_token_cross_entropy(
                logits=patcher_logits,
                target_ids=patcher_input_ids,
                pad_id=token_to_id["<pad>"],
            )
            patcher_optimizer.zero_grad()
            patcher_loss.backward()
            patcher_optimizer.step()

        if trunk_skip_step:
            next_patch_loss = torch.tensor(0.0)
        else:
            trunk_optimizer.zero_grad()
            with torch.no_grad():
                patcher_latents = patcher_model.encode_patch_latents(trunk_input_ids)
            patch_input_latents, patch_target_latents = _next_patch_latent_batch(
                patcher_latents=patcher_latents,
                patch_size=max(1, model_runtime.patchers[0].patch_size) if model_runtime.patchers else 1,
            )
            if patch_input_latents.shape[1] == 0:
                next_patch_loss = torch.tensor(0.0)
            else:
                predicted_patch_latents = trunk_model(patch_input_latents)
                next_patch_loss = _masked_latent_mse(
                    predicted_latents=predicted_patch_latents,
                    target_latents=patch_target_latents,
                )
                next_patch_loss.backward()
                trunk_optimizer.step()

        loss = patcher_loss + next_patch_loss

        detached_loss = float(loss.detach().cpu())
        detached_trunk_loss = float(next_patch_loss.detach().cpu())
        detached_patcher_loss = float(patcher_loss.detach().cpu())
        losses.append(detached_loss)
        trunk_losses.append(detached_trunk_loss)
        patcher_losses.append(detached_patcher_loss)

        if log_every_steps > 0 and (step + 1) % log_every_steps == 0:
            print(
                f"step={step + 1} loss={detached_loss:.6f} trunk_loss={detached_trunk_loss:.6f} "
                f"patcher_loss={detached_patcher_loss:.6f} trunk_lr={trunk_lr:.8f} patcher_lr={patcher_lr:.8f}"
            )

        if save_every > 0 and (step + 1) % save_every == 0:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoints_dir / f"checkpoint_step_{step + 1}.pt"
            torch.save(
                {
                    "model_state": trunk_model.state_dict(),
                    "optimizer_state": trunk_optimizer.state_dict(),
                    "patcher_model_state": patcher_model.state_dict(),
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
            "patcher_model_state": patcher_model.state_dict(),
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
                "schedulers": [
                    {"type": scheduler.scheduler_type, "attributes": dict(scheduler.attributes)}
                    for scheduler in model_runtime.trunk.train_config.schedulers
                ],
                "prediction_target": "patch_latent_next_state_mse_plus_patcher_reconstruction",
                "latent_dim": trunk_model.latent_dim,
                "decoder_head": "patcher_decoder_for_reconstruction_only",
                "patch_size": max(1, model_runtime.patchers[0].patch_size) if model_runtime.patchers else 1,
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
        transformer_layers=transformer_layers,
        n_heads=n_heads,
        moe_expert_count=moe_expert_count,
        dropout=float(meta.get("dropout", model_runtime.trunk.train_config.dropout)),
    )
    model.load_state_dict(state_dict)
    model.eval()

    patcher_state = payload.get("patcher_model_state")
    if not isinstance(patcher_state, dict):
        raise ValueError(f"weights file is missing patcher_model_state: {weights_file}")
    patcher_model = _TrainPatcherModel(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        use_vocab_embedding=use_vocab_embedding,
        use_positional_embedding=use_positional_embedding,
        transformer_layers=max(1, _count_patcher_transformer_layers(model_runtime)),
        n_heads=n_heads,
        moe_expert_count=0,
        dropout=float(meta.get("dropout", model_runtime.trunk.train_config.dropout)),
    )
    patcher_model.load_state_dict(patcher_state)
    patcher_model.eval()

    encoded = _encode_texts([text], {str(k): int(v) for k, v in token_to_id.items()}, max_tokens=max_seq_len)[0]
    pad_id = int(meta.get("pad_id", 0))

    patch_size = int(meta.get("patch_size", 1))
    generated_ids: list[int] = []
    generated_scores: list[float] = []
    for _ in range(max(0, max_new_tokens)):
        window = encoded[-max_seq_len:]
        input_ids, _ = _next_token_batch([window], pad_id=pad_id)
        patch_latents = patcher_model.encode_patch_latents(input_ids)
        patch_sequence = _pool_patch_latents(patch_latents, patch_size)
        if patch_sequence.shape[1] == 0:
            break
        patch_input = patch_sequence[:, -1:, :]
        with torch.no_grad():
            predicted_patch = model(patch_input)
            logits = patcher_model.decode_latents(predicted_patch)

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


class _TrainPatcherModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        max_seq_len: int,
        use_vocab_embedding: bool,
        use_positional_embedding: bool,
        transformer_layers: int,
        n_heads: int,
        moe_expert_count: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_embedding = nn.Embedding(vocab_size, d_model) if use_vocab_embedding else None
        self.token_projection = nn.Linear(1, d_model) if not use_vocab_embedding else None
        self.positional_embedding = nn.Embedding(max_seq_len, d_model) if use_positional_embedding else None
        self.layers = nn.ModuleList(
            [
                _TransformerDecoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    moe_expert_count=moe_expert_count,
                    dropout=dropout,
                )
                for _ in range(transformer_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.latent_dim = d_model
        self.latent_head = nn.Linear(d_model, self.latent_dim)
        self.latent_to_model = nn.Linear(self.latent_dim, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def encode_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
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

    def encode_patch_latents(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.forward_latents_from_encoded(self.encode_tokens(input_ids))

    def forward_latents_from_encoded(self, encoded_inputs: torch.Tensor) -> torch.Tensor:
        x = encoded_inputs
        seq_len = encoded_inputs.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=encoded_inputs.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, causal_mask)
        x = self.norm(x)
        return self.latent_head(x)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.head(self.latent_to_model(latents))

    def reconstruct_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.reconstruct_logits_from_encoded(self.encode_tokens(input_ids))

    def reconstruct_logits_from_encoded(self, encoded_inputs: torch.Tensor) -> torch.Tensor:
        return self.decode_latents(self.latent_head(encoded_inputs))


class _TrainTrunkModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        use_positional_embedding: bool,
        transformer_layers: int,
        n_heads: int,
        moe_expert_count: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.positional_embedding = nn.Embedding(4096, d_model) if use_positional_embedding else None
        self.layers = nn.ModuleList(
            [
                _TransformerDecoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    moe_expert_count=moe_expert_count,
                    dropout=dropout,
                )
                for _ in range(transformer_layers)
            ]
        )
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
            x = layer(x, causal_mask)
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


def _masked_token_cross_entropy(logits: torch.Tensor, target_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        ignore_index=pad_id,
    )


def _masked_latent_mse(predicted_latents: torch.Tensor, target_latents: torch.Tensor) -> torch.Tensor:
    if predicted_latents.numel() == 0 or target_latents.numel() == 0:
        return torch.tensor(0.0)
    return torch.mean((predicted_latents - target_latents) ** 2)


def _build_optimizer(optimizer_type: str, parameters: Any, weight_decay: float) -> torch.optim.Optimizer:
    if optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(parameters, lr=3e-3, weight_decay=weight_decay)
    if optimizer_type.lower() == "sgd":
        return torch.optim.SGD(parameters, lr=3e-3, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer type in config: {optimizer_type}")


def _build_schedule_fn(schedulers: list[RuntimeSchedulerConfig], total_steps: int) -> Callable[[int], float]:
    if not schedulers:
        return lambda _step: 3e-3

    intervals: list[tuple[int, int, str, dict[str, str]]] = []
    for scheduler in schedulers:
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
                if scheduler_type == "loss_threshold":
                    return max(min_lr, max_lr * float(attrs.get("decay_factor", "1.0")))
                return max_lr
        return 1e-5

    return _lr



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
    patcher_model: nn.Module,
    patcher_optimizer: torch.optim.Optimizer,
    checkpoint_file: str,
) -> int:
    payload = torch.load(checkpoint_file, map_location="cpu")
    model_state = payload.get("model_state")
    optimizer_state = payload.get("optimizer_state")
    if not isinstance(model_state, dict) or not isinstance(optimizer_state, dict):
        raise ValueError(f"checkpoint file is missing model/optimizer states: {checkpoint_file}")
    trunk_model.load_state_dict(model_state)
    trunk_optimizer.load_state_dict(optimizer_state)

    patcher_model_state = payload.get("patcher_model_state")
    patcher_optimizer_state = payload.get("patcher_optimizer_state")
    if isinstance(patcher_model_state, dict) and isinstance(patcher_optimizer_state, dict):
        patcher_model.load_state_dict(patcher_model_state)
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


def _pool_patch_latents(patcher_latents: torch.Tensor, patch_size: int) -> torch.Tensor:
    step = max(1, patch_size)
    chunks: list[torch.Tensor] = []
    for start in range(0, patcher_latents.shape[1], step):
        chunk = patcher_latents[:, start : start + step, :]
        chunks.append(chunk.mean(dim=1, keepdim=True))
    if not chunks:
        return torch.zeros((patcher_latents.shape[0], 0, patcher_latents.shape[2]), dtype=patcher_latents.dtype)
    return torch.cat(chunks, dim=1)


def _next_patch_latent_batch(patcher_latents: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    patch_sequence = _pool_patch_latents(patcher_latents, patch_size)
    if patch_sequence.shape[1] < 2:
        empty = patch_sequence[:, :0, :]
        return empty, empty
    return patch_sequence[:, :-1, :], patch_sequence[:, 1:, :]


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
