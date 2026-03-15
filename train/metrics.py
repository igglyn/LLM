from __future__ import annotations

from typing import Any

from train.blocks import MixOfExpertsBlock
from train.specs import ModelRuntime, RuntimeTrainConfig


def _train_summary(config: RuntimeTrainConfig) -> dict[str, Any]:
    return {
        "steps": config.steps,
        "batch_size": config.batch_size,
        "save_every": config.save_every,
        "optimizer": {
            "type": config.optimizer_type,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "grad_clip": config.grad_clip,
        },
        "schedulers": [
            {"type": scheduler.scheduler_type, "attributes": dict(scheduler.attributes)}
            for scheduler in config.schedulers
        ],
    }


def summarize_model_runtime(model_runtime: ModelRuntime) -> dict[str, Any]:
    patcher_blocks = sum(len(p.blocks) for p in model_runtime.patchers)
    trunk_moes = [b for b in model_runtime.trunk.blocks if isinstance(b, MixOfExpertsBlock)]

    return {
        "patcher_count": len(model_runtime.patchers),
        "trunk_block_count": len(model_runtime.trunk.blocks),
        "total_block_count": patcher_blocks + len(model_runtime.trunk.blocks),
        "has_moe": bool(trunk_moes),
        "patchers": [
            {
                "name": p.name,
                "patch_size": p.patch_size,
                "train": _train_summary(p.train_config),
                "block_order": [b.block_name for b in p.blocks],
                "block_count": len(p.blocks),
            }
            for p in model_runtime.patchers
        ],
        "trunk": {
            "name": model_runtime.trunk.name,
            "context": model_runtime.trunk.context,
            "train": _train_summary(model_runtime.trunk.train_config),
            "block_order": [b.block_name for b in model_runtime.trunk.blocks],
            "block_count": len(model_runtime.trunk.blocks),
            "moe_blocks": [
                {
                    "name": moe.name,
                    "expert_count": len(moe.experts),
                    "experts": [
                        {
                            "name": expert.name,
                            "block_order": [block.block_name for block in expert.blocks],
                            "block_count": len(expert.blocks),
                        }
                        for expert in moe.experts
                    ],
                }
                for moe in trunk_moes
            ],
        },
    }
