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


def _step_pair(opt: AdEMAMixFused, model: _Pair, num_steps: int = 4) -> None:
    for _ in range(num_steps):
        grads = [torch.tensor([0.1, -0.05]), torch.tensor([-0.2, 0.04])]
        model.w1.grad, model.w2.grad = [g.clone() for g in grads]
        opt.step()


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
        ademamix_backend="eager",
        ademamix_state_backend="qint8",
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

    _step_pair(opt_scalar, model_scalar)
    _step_pair(opt_foreach, model_foreach)

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


def test_backend_switch_eager_and_fused_match_via_fallback() -> None:
    model_eager = _Pair()
    model_fused = _Pair()
    model_fused.load_state_dict(model_eager.state_dict())

    opt_eager = AdEMAMixFused(model_eager.parameters(), lr=1e-2, backend="eager")
    opt_fused = AdEMAMixFused(model_fused.parameters(), lr=1e-2, backend="fused")

    _step_pair(opt_eager, model_eager, num_steps=6)
    _step_pair(opt_fused, model_fused, num_steps=6)

    assert torch.allclose(model_eager.w1, model_fused.w1, atol=1e-6, rtol=1e-5)
    assert torch.allclose(model_eager.w2, model_fused.w2, atol=1e-6, rtol=1e-5)


def test_quantized_state_backend_updates_and_stores_int8_state() -> None:
    model = _Tiny()
    opt = AdEMAMixFused(model.parameters(), lr=1e-2, state_backend="qint8", quant_block_size=8)

    model.w.grad = torch.tensor([0.2, -0.1], dtype=torch.float32)
    opt.step()

    p = next(iter(opt.param_groups[0]["params"]))
    st = opt.state[p]
    assert "m1_m2_q" in st and st["m1_m2_q"].dtype == torch.int8
    assert "nu_q" in st and st["nu_q"].dtype == torch.int8


def test_checkpoint_compatibility_matrix_eager_fused_and_qint8() -> None:
    src_model = _Tiny()
    src_opt = AdEMAMixFused(src_model.parameters(), lr=1e-2, backend="eager", state_backend="fp32")
    src_model.w.grad = torch.tensor([0.2, -0.1], dtype=torch.float32)
    src_opt.step()
    ckpt = src_opt.state_dict()

    model_fused = _Tiny()
    opt_fused = AdEMAMixFused(model_fused.parameters(), lr=1e-2, backend="fused", state_backend="fp32")
    opt_fused.load_state_dict(ckpt)
    p_fused = next(iter(opt_fused.param_groups[0]["params"]))
    assert "m1_m2" in opt_fused.state[p_fused]

    model_q = _Tiny()
    opt_q = AdEMAMixFused(model_q.parameters(), lr=1e-2, backend="eager", state_backend="qint8")
    opt_q.load_state_dict(ckpt)
    model_q.w.grad = torch.tensor([0.2, -0.1], dtype=torch.float32)
    opt_q.step()
    p_q = next(iter(opt_q.param_groups[0]["params"]))
    assert "m1_m2_q" in opt_q.state[p_q]

    ckpt_q = opt_q.state_dict()
    model_fp32 = _Tiny()
    opt_fp32 = AdEMAMixFused(model_fp32.parameters(), lr=1e-2, state_backend="fp32")
    opt_fp32.load_state_dict(ckpt_q)
    model_fp32.w.grad = torch.tensor([0.2, -0.1], dtype=torch.float32)
    opt_fp32.step()
    p_fp32 = next(iter(opt_fp32.param_groups[0]["params"]))
    assert "m1_m2" in opt_fp32.state[p_fp32]
