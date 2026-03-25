"""AdEMAMix optimizer with a backend switch for eager/"fused" and quantized state."""

from __future__ import annotations

import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dependency.
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _ademamix_update_kernel(
        p_ptr,
        g_ptr,
        m1_ptr,
        m2_ptr,
        nu_ptr,
        n_elements,
        beta1,
        beta2,
        beta3_t,
        one_minus_beta1,
        one_minus_beta2,
        one_minus_beta3,
        alpha_t,
        bias_correction1,
        bias_correction2,
        eps,
        lr,
        weight_decay,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        p = tl.load(p_ptr + offs, mask=mask)
        g = tl.load(g_ptr + offs, mask=mask)
        m1 = tl.load(m1_ptr + offs, mask=mask)
        m2 = tl.load(m2_ptr + offs, mask=mask)
        nu = tl.load(nu_ptr + offs, mask=mask)

        m1 = m1 * beta1 + g * one_minus_beta1
        m2 = m2 * beta3_t + g * one_minus_beta3
        nu = nu * beta2 + (g * g) * one_minus_beta2

        mixed = (m1 / bias_correction1) + alpha_t * m2
        denom = tl.sqrt(nu) / bias_correction2 + eps

        if weight_decay != 0:
            p = p * (1.0 - lr * weight_decay)
        p = p - lr * (mixed / denom)

        tl.store(p_ptr + offs, p, mask=mask)
        tl.store(m1_ptr + offs, m1, mask=mask)
        tl.store(m2_ptr + offs, m2, mask=mask)
        tl.store(nu_ptr + offs, nu, mask=mask)


class AdEMAMixFused(torch.optim.Optimizer):
    """AdEMAMix variant with pluggable execution and state backends.

    backends:
      - eager: foreach/scalar PyTorch update path
      - fused: Triton kernel when available on CUDA, with eager fallback

    state_backend:
      - fp32: regular dense tensors for m1/m2/nu
      - qint8: blockwise int8 tensors with per-block absmax scales
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        t_alpha: Optional[int] = None,
        t_beta3: Optional[int] = None,
        slow_ema_reset_steps: Optional[int] = None,
        use_foreach: bool = True,
        backend: str = "eager",
        state_backend: str = "fp32",
        quant_block_size: int = 256,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if len(betas) != 3:
            raise ValueError("betas must be a tuple of (beta1, beta2, beta3).")
        beta1, beta2, beta3 = betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0 and 0.0 <= beta3 < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if slow_ema_reset_steps is not None and slow_ema_reset_steps <= 0:
            raise ValueError("slow_ema_reset_steps must be > 0 when provided.")
        if backend not in {"eager", "fused"}:
            raise ValueError("backend must be 'eager' or 'fused'.")
        if state_backend not in {"fp32", "qint8"}:
            raise ValueError("state_backend must be 'fp32' or 'qint8'.")
        if quant_block_size <= 0:
            raise ValueError("quant_block_size must be > 0")

        defaults = dict(
            lr=lr,
            betas=betas,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            t_alpha=t_alpha,
            t_beta3=t_beta3,
            slow_ema_reset_steps=slow_ema_reset_steps,
            use_foreach=use_foreach,
            backend=backend,
            state_backend=state_backend,
            quant_block_size=quant_block_size,
        )
        super().__init__(params, defaults)
        self._warned_fused_fallback = False

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("use_foreach", True)
            group.setdefault("slow_ema_reset_steps", None)
            group.setdefault("backend", "eager")
            group.setdefault("state_backend", "fp32")
            group.setdefault("quant_block_size", 256)
        for _, st in self.state.items():
            if "m1_m2" in st:
                continue
            if "m1" in st and "m2" in st:
                st["m1_m2"] = torch.stack([st.pop("m1"), st.pop("m2")])
            elif "m1" in st or "m2" in st:
                raise ValueError("Invalid AdEMAMixFused state: expected both m1 and m2.")

    def load_state_dict(self, state_dict):
        # Keep runtime backend choice from the receiving optimizer instance.
        backend_overrides = [
            {
                "backend": g.get("backend", "eager"),
                "state_backend": g.get("state_backend", "fp32"),
                "quant_block_size": g.get("quant_block_size", 256),
                "use_foreach": g.get("use_foreach", True),
                "slow_ema_reset_steps": g.get("slow_ema_reset_steps", None),
            }
            for g in self.param_groups
        ]
        out = super().load_state_dict(state_dict)
        for group, override in zip(self.param_groups, backend_overrides):
            group.update(override)
        return out

    @staticmethod
    def _scheduled_alpha(step: int, alpha: float, t_alpha: Optional[int]) -> float:
        if t_alpha is None:
            return alpha
        return min(step * alpha / t_alpha, alpha)

    @staticmethod
    def _scheduled_beta3(step: int, beta1: float, beta3: float, t_beta3: Optional[int]) -> float:
        if t_beta3 is None:
            return beta3
        ln_beta1 = math.log(beta1)
        ln_beta3 = math.log(beta3)
        step_scale = step / t_beta3
        return min(
            math.exp((ln_beta1 * ln_beta3) / (((1 - step_scale) * ln_beta3) + (step_scale * ln_beta1))),
            beta3,
        )

    @staticmethod
    def _can_use_foreach(params: list[torch.Tensor], grads: list[torch.Tensor]) -> bool:
        if len(params) == 0:
            return False
        d0 = params[0].device
        t0 = params[0].dtype
        return all((not g.is_sparse) and p.device == d0 and p.dtype == t0 and g.device == d0 and g.dtype == t0 for p, g in zip(params, grads))

    @staticmethod
    def _can_use_fused(params: list[torch.Tensor], grads: list[torch.Tensor]) -> bool:
        if not _TRITON_AVAILABLE:
            return False
        if len(params) == 0:
            return False
        return all(
            p.is_cuda and g.is_cuda and p.is_contiguous() and g.is_contiguous() and p.dtype == torch.float32 and g.dtype == torch.float32
            for p, g in zip(params, grads)
        )

    @staticmethod
    def _quantize_blockwise(t: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        flat = t.reshape(-1)
        n = flat.numel()
        blocks = (n + block_size - 1) // block_size
        pad = blocks * block_size - n
        if pad > 0:
            flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)], dim=0)
        view = flat.reshape(blocks, block_size)
        absmax = view.abs().amax(dim=1)
        scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
        q = torch.round(view / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        return q.reshape(-1)[:n], absmax

    @staticmethod
    def _dequantize_blockwise(q: torch.Tensor, absmax: torch.Tensor, shape: torch.Size, block_size: int) -> torch.Tensor:
        flat = q.reshape(-1)
        n = flat.numel()
        blocks = absmax.numel()
        pad = blocks * block_size - n
        if pad > 0:
            flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)], dim=0)
        view = flat.reshape(blocks, block_size).to(torch.float32)
        scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
        out = (view * scale.unsqueeze(1)).reshape(-1)[:n]
        return out.reshape(shape)

    def _init_state(self, p: torch.Tensor, state: dict, state_backend: str, block_size: int) -> None:
        if state_backend == "fp32":
            state["m1_m2"] = torch.zeros((2, *p.shape), dtype=p.dtype, device=p.device)
            state["nu"] = torch.zeros_like(p)
            return

        m1_m2 = torch.zeros((2, *p.shape), dtype=torch.float32, device=p.device)
        nu = torch.zeros_like(p, dtype=torch.float32)
        q_m1_m2, s_m1_m2 = self._quantize_blockwise(m1_m2, block_size)
        q_nu, s_nu = self._quantize_blockwise(nu, block_size)
        state["m1_m2_q"] = q_m1_m2
        state["m1_m2_absmax"] = s_m1_m2
        state["nu_q"] = q_nu
        state["nu_absmax"] = s_nu
        state["shape"] = tuple(p.shape)

    def _materialize_state(self, state: dict, p: torch.Tensor, state_backend: str, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if state_backend == "fp32":
            if "m1_m2" in state and "nu" in state:
                return state["m1_m2"], state["nu"]
            shape = torch.Size(state.get("shape", tuple(p.shape)))
            m1_m2 = self._dequantize_blockwise(state["m1_m2_q"], state["m1_m2_absmax"], torch.Size((2, *shape)), block_size).to(p.device)
            nu = self._dequantize_blockwise(state["nu_q"], state["nu_absmax"], shape, block_size).to(p.device)
            return m1_m2.to(dtype=p.dtype), nu.to(dtype=p.dtype)
        if "m1_m2_q" not in state or "nu_q" not in state:
            return state["m1_m2"], state["nu"]

        shape = torch.Size(state.get("shape", tuple(p.shape)))
        m1_m2 = self._dequantize_blockwise(state["m1_m2_q"], state["m1_m2_absmax"], torch.Size((2, *shape)), block_size).to(p.device)
        nu = self._dequantize_blockwise(state["nu_q"], state["nu_absmax"], shape, block_size).to(p.device)
        return m1_m2.to(dtype=p.dtype), nu.to(dtype=p.dtype)

    def _store_state(self, state: dict, m1_m2: torch.Tensor, nu: torch.Tensor, state_backend: str, block_size: int) -> None:
        if state_backend == "fp32":
            state["m1_m2"] = m1_m2
            state["nu"] = nu
            for k in ["m1_m2_q", "m1_m2_absmax", "nu_q", "nu_absmax", "shape"]:
                state.pop(k, None)
            return

        q_m1_m2, s_m1_m2 = self._quantize_blockwise(m1_m2.to(torch.float32), block_size)
        q_nu, s_nu = self._quantize_blockwise(nu.to(torch.float32), block_size)
        state["m1_m2_q"] = q_m1_m2
        state["m1_m2_absmax"] = s_m1_m2
        state["nu_q"] = q_nu
        state["nu_absmax"] = s_nu
        state["shape"] = tuple(nu.shape)
        state.pop("m1_m2", None)
        state.pop("nu", None)

    def _fused_update(
        self,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        m1s: list[torch.Tensor],
        m2s: list[torch.Tensor],
        nus: list[torch.Tensor],
        *,
        beta1: float,
        beta2: float,
        beta3_t: float,
        alpha_t: float,
        bias_correction1: float,
        bias_correction2: float,
        eps: float,
        lr: float,
        weight_decay: float,
    ) -> bool:
        if not self._can_use_fused(params, grads):
            return False
        for p, g, m1, m2, nu in zip(params, grads, m1s, m2s, nus):
            n = p.numel()
            BLOCK = 1024
            grid = (triton.cdiv(n, BLOCK),)
            _ademamix_update_kernel[grid](
                p,
                g,
                m1,
                m2,
                nu,
                n,
                beta1,
                beta2,
                beta3_t,
                1.0 - beta1,
                1.0 - beta2,
                1.0 - beta3_t,
                alpha_t,
                bias_correction1,
                bias_correction2,
                eps,
                lr,
                weight_decay,
                BLOCK=BLOCK,
            )
        return True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            alpha = group["alpha"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            t_alpha = group["t_alpha"]
            t_beta3 = group["t_beta3"]
            reset_every = group["slow_ema_reset_steps"]
            use_foreach = group["use_foreach"]
            backend = group["backend"]
            state_backend = group["state_backend"]
            block_size = group["quant_block_size"]

            if "step" not in group:
                group["step"] = 0
            group["step"] += 1
            step = int(group["step"])

            alpha_t = self._scheduled_alpha(step, alpha, t_alpha)
            beta3_t = self._scheduled_beta3(step, beta1, beta3, t_beta3)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = math.sqrt(1.0 - beta2**step)

            params: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            m1_m2_states: list[torch.Tensor] = []
            nus: list[torch.Tensor] = []
            states: list[dict] = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdEMAMixFused does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, state, state_backend, block_size)

                m1_m2, nu = self._materialize_state(state, p, state_backend, block_size)
                params.append(p)
                grads.append(grad)
                m1_m2_states.append(m1_m2)
                nus.append(nu)
                states.append(state)

            if len(params) == 0:
                continue

            m1s = [x[0] for x in m1_m2_states]
            m2s = [x[1] for x in m1_m2_states]

            if reset_every is not None and step % reset_every == 0:
                if use_foreach and self._can_use_foreach(params, grads):
                    torch._foreach_zero_(m2s)
                else:
                    for m2 in m2s:
                        m2.zero_()

            fused_done = False
            if backend == "fused":
                fused_done = self._fused_update(
                    params,
                    grads,
                    m1s,
                    m2s,
                    nus,
                    beta1=beta1,
                    beta2=beta2,
                    beta3_t=beta3_t,
                    alpha_t=alpha_t,
                    bias_correction1=bias_correction1,
                    bias_correction2=bias_correction2,
                    eps=eps,
                    lr=lr,
                    weight_decay=weight_decay,
                )
                if not fused_done and not self._warned_fused_fallback:
                    self._warned_fused_fallback = True
                    print("AdEMAMixFused: fused backend unavailable, falling back to eager path.")

            if not fused_done and use_foreach and self._can_use_foreach(params, grads):
                torch._foreach_mul_(m1s, beta1)
                torch._foreach_add_(m1s, grads, alpha=1.0 - beta1)

                torch._foreach_mul_(m2s, beta3_t)
                torch._foreach_add_(m2s, grads, alpha=1.0 - beta3_t)

                torch._foreach_mul_(nus, beta2)
                torch._foreach_addcmul_(nus, grads, grads, value=1.0 - beta2)

                mixed_momentum = torch._foreach_div(m1s, bias_correction1)
                torch._foreach_add_(mixed_momentum, m2s, alpha=alpha_t)

                denom = torch._foreach_sqrt(nus)
                torch._foreach_div_(denom, bias_correction2)
                torch._foreach_add_(denom, eps)

                update = torch._foreach_div(mixed_momentum, denom)

                if weight_decay > 0.0:
                    torch._foreach_mul_(params, 1.0 - lr * weight_decay)
                torch._foreach_add_(params, update, alpha=-lr)

            elif not fused_done:
                for p, grad, m1, m2, nu in zip(params, grads, m1s, m2s, nus):
                    m1.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    m2.mul_(beta3_t).add_(grad, alpha=1.0 - beta3_t)
                    nu.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    mixed_momentum = (m1 / bias_correction1) + (alpha_t * m2)
                    denom = (nu.sqrt() / bias_correction2).add_(eps)

                    if weight_decay > 0.0:
                        p.mul_(1.0 - lr * weight_decay)
                    p.addcdiv_(mixed_momentum, denom, value=-lr)

            for st, m1_m2, nu in zip(states, m1_m2_states, nus):
                self._store_state(st, m1_m2, nu, state_backend, block_size)

        return loss
