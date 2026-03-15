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

    model = _TrainLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        use_vocab_embedding=has_vocab_embedding,
        use_positional_embedding=has_pos_embedding,
        transformer_layers=transformer_layer_count,
        n_heads=n_heads,
        moe_expert_count=moe_expert_count,
    )
    optimizer = _build_optimizer(
        optimizer_type=optimizer_type,
        parameters=model.parameters(),
        weight_decay=model_runtime.trunk.train_config.weight_decay,
    )
    schedule_fn = _build_schedule_fn(model_runtime.trunk.train_config.schedulers, total_steps=steps)

    start_step = 0
    if resume_from:
        start_step = _resume_training_state(model=model, optimizer=optimizer, checkpoint_file=resume_from)

    checkpoints_dir = output_path / "checkpoints"
    checkpoint_files: list[str] = []

    losses: list[float] = []
    for step in range(start_step, steps):
        batch_sequences = [
            sequences[(step * configured_batch_size + batch_index) % len(sequences)]
            for batch_index in range(configured_batch_size)
        ]
        input_ids, target_ids = _next_token_batch(batch_sequences, pad_id=token_to_id["<pad>"])

        current_lr = schedule_fn(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        predicted_latents = model.forward_latents(input_ids)
        with torch.no_grad():
            target_latents = model.encode_target_latents(target_ids)
        loss = _masked_latent_mse(
            predicted_latents=predicted_latents,
            target_latents=target_latents,
            target_ids=target_ids,
            pad_id=token_to_id["<pad>"],
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        detached_loss = float(loss.detach().cpu())
        losses.append(detached_loss)

        if log_every_steps > 0 and (step + 1) % log_every_steps == 0:
            print(f"step={step + 1} loss={detached_loss:.6f} lr={current_lr:.8f}")

        if save_every > 0 and (step + 1) % save_every == 0:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoints_dir / f"checkpoint_step_{step + 1}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": step + 1,
                    "loss": detached_loss,
                },
                checkpoint_path,
            )
            checkpoint_files.append(str(checkpoint_path))

    weights_file = output_path / "weights.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
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
                "prediction_target": "latent_state",
                "latent_dim": model.latent_dim,
                "decoder_head": "token_logits_for_inference",
            },
        },
        weights_file,
    )

    initial_loss = losses[0] if losses else 0.0
    final_loss = losses[-1] if losses else 0.0

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
        "weights_file": str(weights_file),
    }


def run_trained_model(model_runtime: ModelRuntime, weights_file: str, text: str) -> dict[str, Any]:
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

    model = _TrainLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        use_vocab_embedding=use_vocab_embedding,
        use_positional_embedding=use_positional_embedding,
        transformer_layers=transformer_layers,
        n_heads=n_heads,
        moe_expert_count=moe_expert_count,
    )
    model.load_state_dict(state_dict)
    model.eval()

    encoded = _encode_texts([text], {str(k): int(v) for k, v in token_to_id.items()}, max_tokens=max_seq_len)[0]
    input_ids, _ = _next_token_batch([encoded], pad_id=int(meta.get("pad_id", 0)))

    with torch.no_grad():
        logits = model(input_ids)
    next_token_id = int(torch.argmax(logits[0, -1]).item())
    id_to_token = {int(v): str(k) for k, v in token_to_id.items()}

    state = model_runtime.smoke(text)
    return {
        "text": text,
        "predicted_token": id_to_token.get(next_token_id, "<unk>"),
        "predicted_token_id": next_token_id,
        "score": float(torch.max(logits[0, -1]).item()),
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


class _TrainLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        use_vocab_embedding: bool,
        use_positional_embedding: bool,
        transformer_layers: int,
        n_heads: int,
        moe_expert_count: int,
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

    def encode_target_latents(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.latent_head(self.encode_tokens(input_ids))

    def forward_latents(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.encode_tokens(input_ids)
        seq_len = input_ids.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, causal_mask)
        x = self.norm(x)
        return self.latent_head(x)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.head(self.latent_to_model(latents))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decode_latents(self.forward_latents(input_ids))


class _TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, moe_expert_count: int) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.moe_gate = nn.Linear(d_model, moe_expert_count) if moe_expert_count > 0 else None
        self.moe_experts = (
            nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model), nn.GELU()) for _ in range(moe_expert_count)])
            if moe_expert_count > 0
            else nn.ModuleList()
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out

        ffn_input = self.ffn_norm(x)
        if self.moe_gate is not None and self.moe_experts:
            gate = torch.softmax(self.moe_gate(ffn_input), dim=-1)
            expert_outputs = torch.stack([expert(ffn_input) for expert in self.moe_experts], dim=-2)
            ffn_out = torch.sum(expert_outputs * gate.unsqueeze(-1), dim=-2)
        else:
            ffn_out = self.ffn(ffn_input)
        return x + ffn_out


def _masked_latent_mse(
    predicted_latents: torch.Tensor,
    target_latents: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    mask = (target_ids != pad_id).unsqueeze(-1).to(predicted_latents.dtype)
    squared_error = (predicted_latents - target_latents) ** 2
    masked_error = squared_error * mask
    denom = torch.clamp(mask.sum() * predicted_latents.shape[-1], min=1.0)
    return masked_error.sum() / denom


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


def _resume_training_state(model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_file: str) -> int:
    payload = torch.load(checkpoint_file, map_location="cpu")
    model_state = payload.get("model_state")
    optimizer_state = payload.get("optimizer_state")
    if not isinstance(model_state, dict) or not isinstance(optimizer_state, dict):
        raise ValueError(f"checkpoint file is missing model/optimizer states: {checkpoint_file}")
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
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


def _has_positional_embedding(model_runtime: ModelRuntime) -> bool:
    patcher_has_pos = any(block.block_name == "PosEmbedding" for patcher in model_runtime.patchers for block in patcher.blocks)
    trunk_has_pos = any(block.block_name == "PosEmbedding" for block in model_runtime.trunk.blocks)
    return patcher_has_pos or trunk_has_pos


def _has_vocab_embedding(model_runtime: ModelRuntime) -> bool:
    patcher_has_vocab = any(block.block_name == "VocabEmbedding" for patcher in model_runtime.patchers for block in patcher.blocks)
    trunk_has_vocab = any(block.block_name == "VocabEmbedding" for block in model_runtime.trunk.blocks)
    return patcher_has_vocab or trunk_has_vocab


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
    payload = torch.load(weights_file, map_location="cpu")
    state_dict = payload.get("model_state")
    meta = payload.get("meta", {})
    if not isinstance(state_dict, dict):
        raise ValueError(f"weights file is missing model_state: {weights_file}")
    if not isinstance(meta, dict):
        raise ValueError(f"weights file is missing meta dictionary: {weights_file}")

    token_to_id_raw = meta.get("token_to_id")
    if not isinstance(token_to_id_raw, dict):
        raise ValueError(f"weights file is missing token_to_id: {weights_file}")
    token_to_id = {str(k): int(v) for k, v in token_to_id_raw.items()}
    id_to_token = {int(v): str(k) for k, v in token_to_id.items()}

    d_model = int(meta.get("d_model", _infer_d_model(model_runtime)))
    config_n_heads = int(meta.get("config_n_heads", _infer_n_heads(model_runtime)))
    n_heads = int(meta.get("n_heads", config_n_heads))
    if n_heads != config_n_heads:
        raise ValueError(
            f"weights n_heads mismatch for config: expected compatible n_heads={config_n_heads} for d_model={d_model}, got {n_heads}"
        )


    model = _TrainLanguageModel(
        vocab_size=int(meta.get("vocab_size", max(token_to_id.values()) + 1)),
        d_model=d_model,
        max_seq_len=int(meta.get("max_seq_len", 8)),
        use_vocab_embedding=bool(meta.get("use_vocab_embedding", _has_vocab_embedding(model_runtime))),
        use_positional_embedding=bool(meta.get("use_positional_embedding", _has_positional_embedding(model_runtime))),
        transformer_layers=int(meta.get("transformer_layers", _count_transformer_layers(model_runtime))),
        n_heads=n_heads,
        moe_expert_count=int(meta.get("moe_expert_count", _count_moe_experts(model_runtime))),
    )
    model.load_state_dict(state_dict)
    model.eval()

    encoded = _encode_texts([text], token_to_id, max_tokens=int(meta.get("max_seq_len", 8)))[0]
    max_seq_len = int(meta.get("max_seq_len", 8))
    pad_id = int(meta.get("pad_id", token_to_id.get("<pad>", 0)))

    for _ in range(max(0, max_new_tokens)):
        window = encoded[-max_seq_len:]
        input_ids, _ = _next_token_batch([window], pad_id=pad_id)
        with torch.no_grad():
            logits = model(input_ids)
        next_id = int(torch.argmax(logits[0, -1]).item())
        encoded.append(next_id)

    generated_ids = encoded[-max(0, max_new_tokens):] if max_new_tokens > 0 else []
    generated_tokens = [id_to_token.get(token_id, "<unk>") for token_id in generated_ids]
    return {
        "text": text,
        "generated_tokens": generated_tokens,
        "generated_token_ids": generated_ids,
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
