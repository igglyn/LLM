from __future__ import annotations

from shared.config.specs import TrainSpec
from train.specs import RuntimeTrainConfig


def runtime_train_config_from_spec(train_spec: TrainSpec) -> RuntimeTrainConfig:
    return RuntimeTrainConfig(
        mode=train_spec.mode,
        optimizer_type=train_spec.optimizer.optimizer_type,
        lr=train_spec.optimizer.lr,
        weight_decay=train_spec.optimizer.weight_decay,
        schedulers=list(train_spec.optimizer.schedulers),
    )
