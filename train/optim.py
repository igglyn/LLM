from __future__ import annotations

from shared.config.specs import TrainSpec
from train.specs import RuntimeSchedulerConfig, RuntimeTrainConfig


def runtime_train_config_from_spec(train_spec: TrainSpec) -> RuntimeTrainConfig:
    schedulers = [
        RuntimeSchedulerConfig(scheduler_type=s.scheduler_type, attributes=dict(s.attributes))
        for s in train_spec.optimizer.schedulers
    ]
    return RuntimeTrainConfig(
        mode=train_spec.mode,
        optimizer_type=train_spec.optimizer.optimizer_type,
        lr=train_spec.optimizer.lr,
        weight_decay=train_spec.optimizer.weight_decay,
        schedulers=schedulers,
    )
