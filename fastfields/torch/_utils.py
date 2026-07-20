"""Shared helpers for the fastfields_torch wrappers."""

from __future__ import annotations

import torch
from torch import Tensor

# The bindings only implement float32/float64 kernels.
_SUPPORTED_DTYPES = (torch.float32, torch.float64)


def check_dtype(*tensors: Tensor) -> None:
    """Raise if any tensor has an unsupported (non float32/float64) dtype."""
    for t in tensors:
        if t.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(
                f"fastfields_torch only supports float32/float64 tensors, "
                f"got {t.dtype}."
            )


def as_contiguous(t: Tensor) -> Tensor:
    """Return a contiguous view of ``t`` (the bindings require dense memory)."""
    return t if t.is_contiguous() else t.contiguous()
