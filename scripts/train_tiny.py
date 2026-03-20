#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import math
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from blt_lite.model import TinyPatchLM
from blt_lite.tokenizer import FixedPatchTokenizer
from blt_lite.train import build_dataloaders, evaluate
from blt_lite.utils import ensure_dir, get_device, load_config, set_seed
from blt_lite.ademamix import AdEMAMix
from blt_lite.synthetic_batch import SyntheticBatchHarness
from torch.optim import AdamW


_STEP_RE = re.compile(r"step_(\d+)\.pt$")


class HiddenCausalDataset(Dataset):
    def __init__(self, hidden_path: Path, tokens_path: Path, seq_len: int):
        hidden = np.load(hidden_path, mmap_mode="r")
        tokens = np.load(tokens_path, mmap_mode="r")
        if hidden.ndim != 2:
            raise ValueError(f"Expected 2D hidden cache at {hidden_path}, got shape={hidden.shape}")
        if hidden.shape[0] != tokens.shape[0]:
            raise ValueError("Hidden cache and token stream length mismatch")
        if hidden.shape[0] <= seq_len:
            raise ValueError("Not enough cached hidden states for sequence length")
        self.hidden = torch.from_numpy(np.asarray(hidden, dtype=hidden.dtype))
        self.tokens = torch.from_numpy(np.asarray(tokens, dtype=np.int64))
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.hidden.shape[0] - self.seq_len - 1

    def __getitem__(self, idx: int):
        xh = self.hidden[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return xh, y


def warmup_cosine_lr(step: int, max_steps: int, warmup_steps: int, lr_max: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)

    decay_total = max(1, max_steps - warmup_steps)
    decay_step = min(step - warmup_steps, decay_total)
    cosine = 0.5 * (1 + math.cos(math.pi * decay_step / decay_total))
    return lr_min + (lr_max - lr_min) * cosine


def parse_step_from_checkpoint_name(path: Path) -> int:
    match = _STEP_RE.search(path.name)
    if match:
        return int(match.group(1))
    if path.name in {"best.pt", "last.pt"}:
        return 0
    raise ValueError(f"Checkpoint filename must match step_<N>.pt, best.pt, or last.pt; got: {path.name}")




def _has_required_token_artifacts(processed_dir: Path) -> bool:
    return (processed_dir / "train_tokens.npy").exists() and (processed_dir / "val_tokens.npy").exists() and (processed_dir / "tokenizer.json").exists()


def _resolve_processed_dir(cfg: dict) -> Path:
    data_cfg = cfg["data"]
    patcher2_enabled = bool(cfg.get("patcher2", {}).get("enabled", True))
    tiny_dir = Path(data_cfg["processed_dir_tiny"])
    if patcher2_enabled:
        return tiny_dir

    if _has_required_token_artifacts(tiny_dir):
        return tiny_dir

    patcher_dir = Path(data_cfg["processed_dir_patcher"])
    if _has_required_token_artifacts(patcher_dir):
        return patcher_dir

    raise FileNotFoundError(
        "No valid token dataset for tiny training with patcher2 disabled. "
        f"Checked: {tiny_dir} and {patcher_dir}."
    )

def _resolve_checkpoint_path(raw_path: str, out_dir: Path) -> Path:
    ckpt_path = Path(raw_path)
    if not ckpt_path.is_absolute() and not ckpt_path.exists():
        ckpt_path = out_dir / ckpt_path
    return ckpt_path


def _build_model_from_cfg(cfg: dict, tokenizer: FixedPatchTokenizer, device: torch.device) -> TinyPatchLM:
    model_cfg = cfg["model"]
    patcher_cfg = cfg.get("patcher", {})
    patcher2_cfg = cfg.get("patcher2", {})
    use_patcher2 = bool(patcher2_cfg.get("enabled", True))
    return TinyPatchLM(
        vocab_size=tokenizer.vocab_len,
        seq_len=int(model_cfg["seq_len"]),
        patch_size=int(cfg.get("patcher", {}).get("patch_size", getattr(tokenizer, "patch_size", 1))),
        d_model=int(model_cfg["d_model"]),
        n_layers=int(model_cfg["n_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        dropout=float(model_cfg["dropout"]),
        patcher_latent_dim=int(patcher_cfg.get("latent_dim", model_cfg["d_model"])),
        patcher_encoder_layers=int(patcher_cfg.get("encoder_layers", 2)),
        patcher_decoder_layers=int(patcher_cfg.get("decoder_layers", 2)),
        patcher_heads=int(patcher_cfg.get("n_heads", model_cfg["n_heads"])),
        patcher_dropout=float(patcher_cfg.get("dropout", model_cfg["dropout"])),
        patcher_pos_encoding=str(patcher_cfg.get("pos_encoding", "learned")),
        use_patcher2=use_patcher2,
        patcher2_patch_size=int(cfg.get("patcher2", {}).get("patch_size", 2)),
        patcher2_latent_dim=int(cfg.get("patcher2", {}).get("latent_dim", model_cfg["d_model"])),
        patcher2_encoder_layers=int(cfg.get("patcher2", {}).get("encoder_layers", 2)),
        patcher2_decoder_layers=int(cfg.get("patcher2", {}).get("decoder_layers", 2)),
        patcher2_heads=int(cfg.get("patcher2", {}).get("n_heads", model_cfg["n_heads"])),
        patcher2_dropout=float(cfg.get("patcher2", {}).get("dropout", model_cfg["dropout"])),
        patcher2_pos_encoding=str(cfg.get("patcher2", {}).get("pos_encoding", "learned")),
        use_amp=bool(cfg.get("train", {}).get("amp_enabled", True)),
        amp_dtype=str(cfg.get("train", {}).get("amp_dtype", "float16")),
        pos_encoding=str(model_cfg.get("pos_encoding", "learned")),
        grad_checkpointing=bool(model_cfg.get("grad_checkpointing", False)),
        flash_attention=bool(model_cfg.get("flash_attention", True)),
        patcher_grad_checkpointing=bool(patcher_cfg.get("grad_checkpointing", False)),
        patcher2_grad_checkpointing=bool(cfg.get("patcher2", {}).get("grad_checkpointing", False)),
        patcher_flash_attention=bool(patcher_cfg.get("flash_attention", True)),
        patcher2_flash_attention=bool(cfg.get("patcher2", {}).get("flash_attention", True)),
        patcher_block_attention=bool(patcher_cfg.get("block_attention", False)),
        patcher_block_size=int(patcher_cfg.get("block_size", 8)),
        patcher2_block_attention=bool(cfg.get("patcher2", {}).get("block_attention", False)),
        patcher2_block_size=int(cfg.get("patcher2", {}).get("block_size", 8)),
        trunk_block_attention=bool(model_cfg.get("block_attention", False)),
        trunk_block_size=int(model_cfg.get("block_size", 8)),
    ).to(device=device, dtype={"float32": torch.float32, "float64": torch.float64, "float16": torch.float16}.get(
        str(model_cfg.get("model_dtype", "float32")), torch.float32
    ))


def _maybe_load_pretrained_patcher(model: TinyPatchLM, cfg: dict, device: torch.device) -> None:
    patcher_cfg = cfg.get("patcher", {})
    path = patcher_cfg.get("pretrained_path", "")
    if not path:
        raise ValueError("patcher.pretrained_path must be set for tiny LM training (patcher is always frozen).")
    model.load_patcher_checkpoint(path, map_location=device)
    print(f"Loaded pretrained patcher from {path}")
    for p in model.patcher.parameters():
        p.requires_grad = False
    print("Froze patcher parameters")

    patcher2_cfg = cfg.get("patcher2", {})
    if bool(patcher2_cfg.get("enabled", True)):
        path2 = patcher2_cfg.get("pretrained_path", "")
        if not path2:
            raise ValueError("patcher2.pretrained_path must be set for tiny LM training when patcher2 is enabled.")
        model.load_patcher2_checkpoint(path2, map_location=device)
        print(f"Loaded pretrained second patcher from {path2}")
        if model.patcher2 is not None:
            for p in model.patcher2.parameters():
                p.requires_grad = False
        print("Froze second patcher parameters")


def _token_seq_len_from_cfg(cfg: dict) -> int:
    model_cfg = cfg["model"]
    p1 = int(cfg.get("patcher", {}).get("patch_size", 1))
    patcher2_enabled = bool(cfg.get("patcher2", {}).get("enabled", True))
    p2 = int(cfg.get("patcher2", {}).get("patch_size", 2)) if patcher2_enabled else 1
    return int(model_cfg["seq_len"]) * p1 * p2


def _evaluate_from_hidden(model: TinyPatchLM, val_loader: DataLoader, device: torch.device, max_batches: int | None = None) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, (xh, y) in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            xh = xh.to(device)
            y = y.to(device)
            _, loss = model.forward_from_hidden(xh, y)
            losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(1, len(losses)))



def maybe_reduce_main_lr_by_thresholds(optimizer, val_loss: float, train_cfg: dict, reduction_state: dict) -> dict:
    thresholds = [
        ("lr_reduce_threshold", "lr_reduce_factor"),
        ("lr_reduce_threshold_2", "lr_reduce_factor_2"),
    ]
    lr_min = float(train_cfg.get("lr_min", 3e-5))
    scale = float(reduction_state.get("scale", 1.0))
    for idx, (thr_key, fac_key) in enumerate(thresholds):
        if reduction_state.get(idx, False):
            continue
        threshold = train_cfg.get(thr_key)
        if threshold is None or val_loss > float(threshold):
            continue
        factor = float(train_cfg.get(fac_key, train_cfg.get("lr_reduce_factor", 0.5)))
        scale *= factor
        for group in optimizer.param_groups:
            old_lr = float(group["lr"])
            group["lr"] = max(lr_min, old_lr * factor)
            print(f"Reduced main LR via {thr_key}: {old_lr:.8f} -> {group['lr']:.8f}")
        reduction_state[idx] = True
    reduction_state["scale"] = scale
    return reduction_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional checkpoint filename or path to resume from (expects step_<N>.pt naming).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    set_seed(int(tcfg.get("seed", 42)))

    device = get_device()
    out_dir = ensure_dir(tcfg.get("out_dir", "outputs"))

    resume_ckpt = None
    step = 0
    if args.checkpoint:
        ckpt_path = _resolve_checkpoint_path(args.checkpoint, out_dir)
        resume_ckpt = torch.load(ckpt_path, map_location=device)
        if "config" in resume_ckpt and isinstance(resume_ckpt["config"], dict):
            cfg = resume_ckpt["config"]
            tcfg = cfg["train"]
        step = parse_step_from_checkpoint_name(ckpt_path)

    processed_dir = _resolve_processed_dir(cfg)
    if not bool(cfg.get("patcher2", {}).get("enabled", True)):
        print(f"patcher2 disabled; using token dataset from {processed_dir}")

    tokenizer_path = processed_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = Path(cfg["data"]["processed_dir_patcher"]) / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError("tokenizer.json not found in tiny or patcher processed dirs")
        print(f"tokenizer.json not in tiny dir, loading from patcher dir")
    tokenizer = FixedPatchTokenizer.load(tokenizer_path)

    seq_len = _token_seq_len_from_cfg(cfg)
    patcher2_enabled = bool(cfg.get("patcher2", {}).get("enabled", True))
    if patcher2_enabled:
        cached_train = processed_dir / "train_stage2_hidden.npy"
        cached_val   = processed_dir / "val_stage2_hidden.npy"
    else:
        cached_train = processed_dir / "train_stage1_hidden.npy"
        cached_val   = processed_dir / "val_stage1_hidden.npy"
    use_cached_hidden = cached_train.exists() and cached_val.exists()

    if use_cached_hidden:
        print("Using cached stage2 hidden states for tiny LM training")
        train_loader = DataLoader(
            HiddenCausalDataset(cached_train, processed_dir / "train_tokens.npy", seq_len),
            batch_size=int(tcfg["batch_size"]),
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            HiddenCausalDataset(cached_val, processed_dir / "val_tokens.npy", seq_len),
            batch_size=int(tcfg["batch_size"]),
            shuffle=False,
            drop_last=False,
        )
    else:
        token_dir = processed_dir
        if not (token_dir / "train_tokens.npy").exists():
            token_dir = Path(cfg["data"]["processed_dir_patcher"])
            print(f"train_tokens.npy not in tiny dir, loading from patcher dir")
        train_loader, val_loader = build_dataloaders(
            token_dir / "train_tokens.npy",
            token_dir / "val_tokens.npy",
            seq_len=seq_len,
            batch_size=int(tcfg["batch_size"]),
        )

    model = _build_model_from_cfg(cfg, tokenizer, device)
    if not use_cached_hidden:
        _maybe_load_pretrained_patcher(model, cfg, device)

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        print(f"Resumed from checkpoint {ckpt_path} at step={step}")

    optimizer_type = str(tcfg.get("optimizer", "adamw")).lower()
    if optimizer_type == "ademamix":
        betas = (
            float(tcfg.get("beta1", 0.9)),
            float(tcfg.get("beta2", 0.999)),
            float(tcfg.get("beta3", 0.9999)),
        )
        optimizer = AdEMAMix(
            model.parameters(),
            lr=float(tcfg.get("lr_max", 3e-4)),
            betas=betas,
            alpha=float(tcfg.get("alpha", 2.0)),
            beta3_warmup=tcfg.get("beta3_warmup", None),
            alpha_warmup=tcfg.get("alpha_warmup", None),
            weight_decay=float(tcfg["weight_decay"]),
        )
        print(f"Using AdEMAMix optimizer")
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=float(tcfg.get("lr_max", 3e-4)),
            weight_decay=float(tcfg["weight_decay"]),
            fused=(torch.cuda.is_available()),
        )
        print("Using AdamW optimizer")

    # --- Synthetic batch harness ---
    synth_cfg = cfg.get("synthetic_batch", {})
    synth_enabled = bool(synth_cfg.get("enabled", False))
    synth_harness = None
    if synth_enabled:
        from blt_lite.synthetic_batch import NNv5LayerConfig
        model_cfg = cfg["model"]
        d_model   = int(model_cfg["d_model"])

        l1cfg = synth_cfg.get("layer1", {})
        l2cfg = synth_cfg.get("layer2", {})

        layer1 = NNv5LayerConfig(
            chunk_dim          = int(l1cfg.get("chunk_dim", 32)),
            case_capacity      = int(l1cfg.get("case_capacity", 2048)),
            group_capacity     = int(l1cfg.get("group_capacity", 128)),
            nnv5_chunk_size    = int(l1cfg.get("nnv5_chunk_size", 16)),
            nnv2_case_capacity = int(l1cfg.get("nnv2_case_capacity", 1024)),
        )
        layer2 = NNv5LayerConfig(
            chunk_dim          = int(l2cfg.get("chunk_dim", d_model // layer1.chunk_dim)),
            case_capacity      = int(l2cfg.get("case_capacity", 1024)),
            group_capacity     = int(l2cfg.get("group_capacity", 64)),
            nnv5_chunk_size    = int(l2cfg.get("nnv5_chunk_size", 16)),
            nnv2_case_capacity = int(l2cfg.get("nnv2_case_capacity", 512)),
        )

        synth_harness = SyntheticBatchHarness(
            model=model,
            d_model=d_model,
            layer1=layer1,
            layer2=layer2,
            n_synthetic=int(synth_cfg.get("n_synthetic", 4)),
            match_threshold=float(synth_cfg.get("match_threshold", 0.1)),
            device=device,
            seed=int(tcfg.get("seed", 42)),
            nnv5_update_steps=int(synth_cfg.get("nnv5_update_steps", 100)),
            synth_loss_weight=float(synth_cfg.get("synth_loss_weight", 0.5)),
        )
        synth_harness.attach()
        synth_loss_weight = float(synth_cfg.get("synth_loss_weight", 0.5))
        print(f"Synthetic batch enabled — L1 chunk_dim={layer1.chunk_dim} "
              f"L2 chunk_dim={layer2.chunk_dim} "
              f"n_synthetic={synth_cfg.get('n_synthetic', 4)}")
    amp_enabled = bool(tcfg.get("amp_enabled", True))
    amp_dtype = torch.float16 if str(tcfg.get("amp_dtype", "float16")) == "float16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and amp_enabled))

    max_steps = int(tcfg["max_steps"])
    eval_every = int(tcfg.get("eval_every", 100))
    save_every = int(tcfg.get("save_every", 200))
    grad_accum_steps = int(tcfg.get("grad_accum_steps", 1))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    lr_max = float(tcfg.get("lr_max", 3e-4))
    lr_min = float(tcfg.get("lr_min", 3e-5))
    warmup_steps = int(tcfg.get("warmup_steps", 1000))
    eval_batches = int(tcfg.get("eval_batches", 50))

    best_val = float("inf")
    lr_reduction_state: dict[int, bool] = {}
    model.train()

    # Dtype probe — log what the trunk actually receives on first batch
    _dtype_probed = False
    while step < max_steps:
        for batch in train_loader:
            lr = warmup_cosine_lr(step, max_steps, warmup_steps, lr_max, lr_min) * float(lr_reduction_state.get("scale", 1.0))
            for group in optimizer.param_groups:
                group["lr"] = lr

            if use_cached_hidden:
                xh, y = batch
                xh, y = xh.to(device), y.to(device)
                if not _dtype_probed:
                    print(f"  dtype probe — cached hidden: {xh.dtype}  targets: {y.dtype}")
                    _dtype_probed = True
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and amp_enabled), dtype=amp_dtype):
                    _, loss = model.forward_from_hidden(xh, y)
                    loss = loss / grad_accum_steps
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                if not _dtype_probed:
                    print(f"  dtype probe — token input: {x.dtype}  targets: {y.dtype}")
                    _dtype_probed = True
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and amp_enabled), dtype=amp_dtype):
                    _, loss = model(x, y)
                    loss = loss / grad_accum_steps

            # Synthetic batch pass — between forward and backward
            if synth_harness is not None and synth_harness.captured_hidden is not None:
                synth_loss = synth_harness.synthetic_pass(
                    synth_harness.captured_hidden,
                    y if use_cached_hidden else y,
                    step=step,
                )
                loss = loss + synth_loss_weight * synth_loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if step % 20 == 0:
                print(f"step={step} train_loss={loss.item() * grad_accum_steps:.4f} lr={lr:.6f}")
                if synth_harness is not None:
                    s = synth_harness.stats()
                    print(f"  synth L1 cases={s['l1_cases']} groups={s['l1_groups']} nnv2={s['nnv2_l1_cases']} | "
                          f"L2 cases={s['l2_cases']} groups={s['l2_groups']} nnv2={s['nnv2_l2_cases']}")

            if step % eval_every == 0 and step > 0:
                val_loss = _evaluate_from_hidden(model, val_loader, device, max_batches=eval_batches) if use_cached_hidden else evaluate(model, val_loader, device, max_batches=eval_batches)
                print(f"step={step} val_loss={val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": cfg,
                            "val_loss": val_loss,
                        },
                        out_dir / "best.pt",
                    )
                lr_reduction_state = maybe_reduce_main_lr_by_thresholds(optimizer, val_loss, tcfg, lr_reduction_state)

            if step % save_every == 0 and step > 0:
                torch.save({"model": model.state_dict(), "config": cfg}, out_dir / f"step_{step}.pt")

            step += 1
            if step >= max_steps:
                break

    torch.save({"model": model.state_dict(), "config": cfg}, out_dir / "last.pt")
    if synth_harness is not None:
        synth_harness.detach()
    print(f"Training complete. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
