"""AdEMAMix optimizer with an implementation shape suitable for fused-kernel migration."""

from __future__ import annotations

import math
from typing import Optional

import torch


class AdEMAMixFused(torch.optim.Optimizer):
    """AdEMAMix variant that keeps local slow-EMA semantics and AdamW decay.

    This class is intentionally written to mirror a fused-kernel contract:
    - state1 is a stacked tensor containing fast EMA (m1) and slow EMA (m2)
    - state2 is second-moment accumulator (nu)
    - schedule-controlled ``alpha_t`` and ``beta3_t``
    - optional periodic slow-EMA reset hook to preserve local behavior

    Current implementation prefers a grouped foreach update path (multi-tensor
    style on eager PyTorch), while keeping an elementwise fallback.
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
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        # Backward compatibility: migrate legacy states that stored m1/m2 separately.
        for group in self.param_groups:
            group.setdefault("use_foreach", True)
            group.setdefault("slow_ema_reset_steps", None)
        for param, st in self.state.items():
            if "m1_m2" in st:
                continue
            if "m1" in st and "m2" in st:
                st["m1_m2"] = torch.stack([st.pop("m1"), st.pop("m2")])
            elif "m1" in st or "m2" in st:
                # Partial legacy state should not silently continue.
                raise ValueError("Invalid AdEMAMixFused state: expected both m1 and m2.")

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
            m1s: list[torch.Tensor] = []
            m2s: list[torch.Tensor] = []
            nus: list[torch.Tensor] = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdEMAMixFused does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["m1_m2"] = torch.zeros((2, *p.shape), dtype=p.dtype, device=p.device)
                    state["nu"] = torch.zeros_like(p)

                params.append(p)
                grads.append(grad)
                m1s.append(state["m1_m2"][0])
                m2s.append(state["m1_m2"][1])
                nus.append(state["nu"])

            if len(params) == 0:
                continue

            if reset_every is not None and step % reset_every == 0:
                if use_foreach and self._can_use_foreach(params, grads):
                    torch._foreach_zero_(m2s)
                else:
                    for m2 in m2s:
                        m2.zero_()

            if use_foreach and self._can_use_foreach(params, grads):
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
                continue

            for p, grad, m1, m2, nu in zip(params, grads, m1s, m2s, nus):
                m1.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                m2.mul_(beta3_t).add_(grad, alpha=1.0 - beta3_t)
                nu.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                mixed_momentum = (m1 / bias_correction1) + (alpha_t * m2)
                denom = (nu.sqrt() / bias_correction2).add_(eps)

                if weight_decay > 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                p.addcdiv_(mixed_momentum, denom, value=-lr)

        return loss
