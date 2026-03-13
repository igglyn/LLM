from __future__ import annotations

from shared.config.specs import ResolvedConfigSpec


class StageBNotImplemented(RuntimeError):
    pass


def run_stage_b(config: ResolvedConfigSpec) -> None:
    _ = config
    raise StageBNotImplemented("StageB runtime is intentionally deferred.")
