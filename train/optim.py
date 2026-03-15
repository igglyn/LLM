from __future__ import annotations

from shared.config.specs import TrainSpec
from train.specs import RuntimeSchedulerConfig, RuntimeTrainConfig


def runtime_train_config_from_spec(train_spec: TrainSpec) -> RuntimeTrainConfig:
    schedulers = [
        RuntimeSchedulerConfig(scheduler_type=s.scheduler_type, attributes=dict(s.attributes))
        for s in train_spec.optimizer.schedulers
    ]
    return RuntimeTrainConfig(
        steps=train_spec.steps,
        batch_size=train_spec.batch_size,
        save_every=train_spec.save_every,
        optimizer_type=train_spec.optimizer.optimizer_type,
        weight_decay=train_spec.optimizer.weight_decay,
        dropout=train_spec.optimizer.dropout,
        grad_clip=train_spec.optimizer.grad_clip,
        schedulers=schedulers,
    )
