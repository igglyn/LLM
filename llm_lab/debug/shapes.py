"""Runtime tensor shape/dtype validation helpers."""

from __future__ import annotations

from typing import Any


def _fmt_actual(x: Any) -> str:
    dtype = getattr(x, "dtype", "<unknown>")
    shape = tuple(getattr(x, "shape", ()))
    return f"actual dtype={dtype}, shape={shape}"


def assert_bytes(x: Any, name: str = "x") -> None:
    """Assert raw-byte tensor convention: uint8 and rank-2 [B, T]."""
    import torch

    if getattr(x, "dtype", None) != torch.uint8:
        raise ValueError(
            f"{name}: expected dtype torch.uint8 and rank 2 [B, T]; "
            f"{_fmt_actual(x)}"
        )
    if getattr(x, "ndim", None) != 2:
        raise ValueError(
            f"{name}: expected dtype torch.uint8 and rank 2 [B, T]; "
            f"{_fmt_actual(x)}"
        )


def assert_ids(x: Any, name: str = "x") -> None:
    """Assert id tensor convention: int64 and rank-2 [B, T] / [B, Tp]."""
    import torch

    if getattr(x, "dtype", None) != torch.int64:
        raise ValueError(
            f"{name}: expected dtype torch.int64 and rank 2 [B, T|Tp]; "
            f"{_fmt_actual(x)}"
        )
    if getattr(x, "ndim", None) != 2:
        raise ValueError(
            f"{name}: expected dtype torch.int64 and rank 2 [B, T|Tp]; "
            f"{_fmt_actual(x)}"
        )


def assert_hidden(x: Any, name: str = "x") -> None:
    """Assert hidden tensor convention: floating and rank-3 [B, T, D]."""
    import torch

    if not torch.is_tensor(x) or not torch.is_floating_point(x):
        raise ValueError(
            f"{name}: expected floating dtype and rank 3 [B, T, D]; "
            f"{_fmt_actual(x)}"
        )
    if getattr(x, "ndim", None) != 3:
        raise ValueError(
            f"{name}: expected floating dtype and rank 3 [B, T, D]; "
            f"{_fmt_actual(x)}"
        )


def assert_logits(x: Any, vocab_size: int | None = None, name: str = "x") -> None:
    """Assert logits convention: floating and rank-3 [B, T, V]."""
    import torch

    if not torch.is_tensor(x) or not torch.is_floating_point(x):
        raise ValueError(
            f"{name}: expected floating dtype and rank 3 [B, T, V]; "
            f"{_fmt_actual(x)}"
        )
    if getattr(x, "ndim", None) != 3:
        raise ValueError(
            f"{name}: expected floating dtype and rank 3 [B, T, V]; "
            f"{_fmt_actual(x)}"
        )
    if vocab_size is not None and x.shape[-1] != vocab_size:
        raise ValueError(
            f"{name}: expected last dim V={vocab_size} with rank 3 [B, T, V]; "
            f"{_fmt_actual(x)}"
        )
