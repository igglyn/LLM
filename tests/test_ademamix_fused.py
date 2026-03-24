from __future__ import annotations

import torch

from llm_lab.train.ademamix_fused import AdEMAMixFused
from llm_lab.train.optim import build_optimizer


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32))


class _Pair(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32))
        self.w2 = torch.nn.Parameter(torch.tensor([0.5, 0.25], dtype=torch.float32))


def test_ademamix_fused_step_updates_params() -> None:
    model = _Tiny()
    opt = AdEMAMixFused(model.parameters(), lr=1e-2, weight_decay=0.0)

    model.w.grad = torch.tensor([0.2, -0.1], dtype=torch.float32)
    before = model.w.detach().clone()
    opt.step()

    assert not torch.allclose(before, model.w.detach())


def test_build_optimizer_supports_ademamix_fused() -> None:
    model = _Tiny()
    opt = build_optimizer(
        model,
        lr=1e-3,
        optimizer_name="ademamix_fused",
        ademamix_slow_ema_reset_steps=10,
    )

    assert isinstance(opt, AdEMAMixFused)


def test_slow_ema_reset_changes_m2_accumulation() -> None:
    model_no_reset = _Tiny()
    model_reset = _Tiny()

    opt_no_reset = AdEMAMixFused(model_no_reset.parameters(), lr=1e-2, slow_ema_reset_steps=None)
    opt_reset = AdEMAMixFused(model_reset.parameters(), lr=1e-2, slow_ema_reset_steps=1)

    for _ in range(2):
        model_no_reset.w.grad = torch.tensor([0.2, -0.1], dtype=torch.float32)
        model_reset.w.grad = torch.tensor([0.2, -0.1], dtype=torch.float32)
        opt_no_reset.step()
        opt_reset.step()

    p_no_reset = next(iter(opt_no_reset.param_groups[0]["params"]))
    p_reset = next(iter(opt_reset.param_groups[0]["params"]))

    m2_no_reset = opt_no_reset.state[p_no_reset]["m1_m2"][1]
    m2_reset = opt_reset.state[p_reset]["m1_m2"][1]

    assert torch.norm(m2_no_reset).item() > torch.norm(m2_reset).item()


def test_foreach_and_scalar_paths_match() -> None:
    model_scalar = _Pair()
    model_foreach = _Pair()

    model_foreach.load_state_dict(model_scalar.state_dict())

    opt_scalar = AdEMAMixFused(model_scalar.parameters(), lr=1e-2, use_foreach=False)
    opt_foreach = AdEMAMixFused(model_foreach.parameters(), lr=1e-2, use_foreach=True)

    for _ in range(4):
        grads = [torch.tensor([0.1, -0.05]), torch.tensor([-0.2, 0.04])]
        model_scalar.w1.grad, model_scalar.w2.grad = [g.clone() for g in grads]
        model_foreach.w1.grad, model_foreach.w2.grad = [g.clone() for g in grads]
        opt_scalar.step()
        opt_foreach.step()

    assert torch.allclose(model_scalar.w1, model_foreach.w1, atol=1e-7, rtol=1e-6)
    assert torch.allclose(model_scalar.w2, model_foreach.w2, atol=1e-7, rtol=1e-6)


def test_state_dict_migrates_legacy_m1_m2_layout() -> None:
    model = _Tiny()
    opt = AdEMAMixFused(model.parameters(), lr=1e-2)

    model.w.grad = torch.tensor([0.2, -0.1], dtype=torch.float32)
    opt.step()

    sd = opt.state_dict()
    for state in sd["state"].values():
        if "m1_m2" in state:
            state["m1"] = state["m1_m2"][0].clone()
            state["m2"] = state["m1_m2"][1].clone()
            del state["m1_m2"]

    model2 = _Tiny()
    opt2 = AdEMAMixFused(model2.parameters(), lr=1e-2)
    opt2.load_state_dict(sd)

    p2 = next(iter(opt2.param_groups[0]["params"]))
    assert "m1_m2" in opt2.state[p2]
    assert "m1" not in opt2.state[p2]
    assert "m2" not in opt2.state[p2]
