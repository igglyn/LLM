from __future__ import annotations

from typing import Any

from train.blocks import MixOfExpertsBlock
from train.specs import ModelRuntime


def summarize_model_runtime(model_runtime: ModelRuntime) -> dict[str, Any]:
    return {
        "patchers": [
            {
                "name": p.name,
                "patch_size": p.patch_size,
                "train_mode": p.train_config.mode,
                "optimizer": p.train_config.optimizer_type,
                "scheduler_count": len(p.train_config.schedulers),
                "block_order": [b.block_name for b in p.blocks],
            }
            for p in model_runtime.patchers
        ],
        "trunk": {
            "name": model_runtime.trunk.name,
            "context": model_runtime.trunk.context,
            "train_mode": model_runtime.trunk.train_config.mode,
            "optimizer": model_runtime.trunk.train_config.optimizer_type,
            "scheduler_count": len(model_runtime.trunk.train_config.schedulers),
            "block_order": [b.block_name for b in model_runtime.trunk.blocks],
            "moe_blocks": [b.name for b in model_runtime.trunk.blocks if isinstance(b, MixOfExpertsBlock)],
        },
    }
