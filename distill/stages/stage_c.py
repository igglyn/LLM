from __future__ import annotations

from shared.config.specs import ResolvedConfigSpec


class StageCNotImplemented(RuntimeError):
    pass


def run_stage_c(config: ResolvedConfigSpec) -> None:
    _ = config
    raise StageCNotImplemented("StageC runtime is intentionally deferred.")
