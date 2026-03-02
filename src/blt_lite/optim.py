from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim import Optimizer


class AdEMAMix(Optimizer):
    """AdEMAMix optimizer.

    PyTorch implementation inspired by public AdEMAMix references.
    Blends short/long EMAs of gradients with Adam-style second moment.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        b1, b2, b3 = betas
        if not 0.0 <= b1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {b1}")
        if not 0.0 <= b2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {b2}")
        if not 0.0 <= b3 < 1.0:
            raise ValueError(f"Invalid beta3 parameter: {b3}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            alpha = group["alpha"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(p)
                    state["m2"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m1 = state["m1"]
                m2 = state["m2"]
                v = state["v"]

                state["step"] += 1
                step = state["step"]

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                m1.mul_(beta1).add_(grad, alpha=1 - beta1)
                m2.mul_(beta3).add_(grad, alpha=1 - beta3)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_c1 = 1 - beta1**step
                bias_c2 = 1 - beta2**step
                bias_c3 = 1 - beta3**step

                m1_hat = m1 / bias_c1
                m2_hat = m2 / bias_c3
                v_hat = v / bias_c2

                mixed = m1_hat + alpha * m2_hat
                denom = v_hat.sqrt().add_(eps)
                p.addcdiv_(mixed, denom, value=-lr)

        return loss
