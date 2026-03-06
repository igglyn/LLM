"""Training entrypoint for minimal baseline runs."""

from __future__ import annotations

import argparse
import random

from llm_lab.config.defaults import load_config


def _train_loop_precomputed(*, model, dataloader, optimizer, steps: int, device: str):
    """Lightweight training loop for precomputed hidden inputs.

    This path is intentionally isolated from the default raw-byte training loop.
    """
    import torch

    model = model.to(device)
    model.train()

    iterator = iter(dataloader)
    last_loss = 0.0

    for _ in range(steps):
        try:
            h = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            h = next(iterator)

        h = h.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(h)

        # Simple optional objective for experimentation path: keep logits bounded.
        loss = (logits ** 2).mean()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().item())

    return {"step": steps, "loss": last_loss}


def main() -> None:
    """Run a minimal training loop for next-byte prediction."""
    parser = argparse.ArgumentParser(description="llm_lab train")
    parser.add_argument("--config", default="configs/examples/tiny_baseline.toml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        print(f"missing dependency: {exc}. install project dependencies to train.")
        return

    from llm_lab.data.byte_dataset import ByteDataset
    from llm_lab.data.collate import collate_batch
    from llm_lab.data.precomputed_patch_dataset import PrecomputedPatchDataset
    from llm_lab.models.assemble import assemble_model
    from llm_lab.train.loop import train_loop
    from llm_lab.train.optim import build_optimizer

    cfg = load_config(args.config)

    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    model = assemble_model(cfg)
    steps = args.steps if args.steps is not None else cfg.train.steps
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_precomputed = bool(getattr(cfg.data, "use_precomputed_patches", False))
    precomputed_path = str(getattr(cfg.data, "precomputed_patches_path", "") or "")
    train_mode = str(getattr(cfg.train, "mode", "full"))

    if use_precomputed:
        if train_mode == "patcher_only":
            raise ValueError(
                "train.mode='patcher_only' is not supported with precomputed patches: "
                "patcher1 is not part of this training path."
            )
        if not precomputed_path:
            print("precomputed path is enabled but data.precomputed_patches_path is empty")
            return

        dataset = PrecomputedPatchDataset(precomputed_path)
        if len(dataset) == 0:
            print("precomputed dataset is empty")
            return

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            drop_last=True,
        )

        class _PrecomputedPathModel(torch.nn.Module):
            def __init__(self, assembled):
                super().__init__()
                self.patcher2 = assembled.patcher2
                self.mixers = assembled.mixers
                self.head = assembled.head
                self.memory = assembled.memory
                self.patcher2_stop_gradient = bool(getattr(assembled, "patcher2_stop_gradient", False))

            def forward(self, h):
                x = h
                h2 = None
                if self.patcher2 is not None:
                    h2 = self.patcher2(x)
                    if self.patcher2_stop_gradient:
                        h2 = h2.detach()

                uses_context_pyramid = self.memory is not None and hasattr(self.memory, "combine")
                if uses_context_pyramid:
                    x, recent_len = self.memory.combine(x, h2)
                else:
                    x = h2 if h2 is not None else x
                    recent_len = x.shape[1]

                for mixer in self.mixers:
                    x = mixer(x)

                x_head = x[:, -recent_len:, :] if uses_context_pyramid else x
                return self.head(x_head)

        pre_model = _PrecomputedPathModel(model)
        optimizer = build_optimizer(pre_model, lr=cfg.train.lr, mode=train_mode)
        metrics = _train_loop_precomputed(
            model=pre_model,
            dataloader=dataloader,
            optimizer=optimizer,
            steps=steps,
            device=device,
        )
    else:
        dataset = ByteDataset(
            data_dir=cfg.data.path,
            seq_len=cfg.data.seq_len,
            seed=cfg.train.seed,
        )
        if len(dataset) == 0:
            print("dataset is empty; provide .txt files in data.path")
            return

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            drop_last=True,
        )

        optimizer = build_optimizer(model, lr=cfg.train.lr, mode=train_mode)

        metrics = train_loop(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            steps=steps,
            device=device,
            grad_accum_steps=args.grad_accum,
            grad_clip_norm=args.grad_clip,
            amp=args.amp,
            checkpoint_path=args.checkpoint or None,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            enable_aux_reconstruction=cfg.train.enable_aux_reconstruction,
            aux_reconstruction_weight=cfg.train.aux_reconstruction_weight,
            preserve_memory_across_batches=cfg.train.preserve_memory_across_batches,
            reset_memory_on_document_boundary=cfg.train.reset_memory_on_document_boundary,
            truncate_bptt_segments=cfg.train.truncate_bptt_segments,
            mode=train_mode,
        )

    print(f"completed step={metrics['step']} loss={metrics['loss']:.6f}")


if __name__ == "__main__":
    main()
