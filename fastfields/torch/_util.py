"""Shared helpers for the fastfields.torch wrappers.

Stream semantics
----------------
The ``fastfields.dlpack`` bindings take a trailing ``stream`` argument (a CUDA
stream handle as an ``int``; ``0`` means the CPU/default stream). torch queues
its own CUDA work on the *current* stream of each device, so -- to keep the
binding's kernels correctly ordered with respect to the surrounding torch ops
(output allocation, later reads, ...) -- every wrapper forwards the current
stream of its primary CUDA tensor via :func:`stream_ptr`. For CPU tensors this
is ``0``, so CPU behaviour is unchanged. This mirrors how ``fastfields.cupy``
forwards ``cupy.cuda.get_current_stream().ptr``.
"""

from __future__ import annotations

from fastfields.dlpack import Bound, Spline

import torch
from torch import Tensor

# The bindings only implement float32/float64 kernels.
_SUPPORTED_DTYPES = (torch.float32, torch.float64)

# Friendly string aliases for the spline-order and boundary-condition
# arguments, mirroring the numpy wrapper so `order=`/`bound=` accept an int, a
# Spline/Bound enum, or a name on every backend.
_SPLINE_ALIASES = {
    "nearest": Spline.Nearest,
    "constant": Spline.Nearest,
    "linear": Spline.Linear,
    "quadratic": Spline.Quadratic,
    "cubic": Spline.Cubic,
    "fourth": Spline.FourthOrder,
    "fifth": Spline.FifthOrder,
    "sixth": Spline.SixthOrder,
    "seventh": Spline.SeventhOrder,
}

_BOUND_ALIASES = {
    "zero": Bound.Zero,
    "zeros": Bound.Zero,
    "replicate": Bound.Replicate,
    "nearest": Bound.Replicate,
    "dct1": Bound.DCT1,
    "dct2": Bound.DCT2,
    "neumann": Bound.DCT2,
    "reflect": Bound.DCT2,
    "dst1": Bound.DST1,
    "dst2": Bound.DST2,
    "dirichlet": Bound.DST2,
    "dft": Bound.DFT,
    "wrap": Bound.DFT,
    "circular": Bound.DFT,
    "nocheck": Bound.NoCheck,
}


def as_spline(value: int | str | Spline) -> int:
    """Normalise a spline-order argument to an ``int`` in ``0..7``.

    Accepts an integer, a :class:`Spline` enum, or a friendly string alias
    (e.g. ``"cubic"``). Raises ``ValueError`` for an unknown alias or an
    out-of-range integer.
    """
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _SPLINE_ALIASES:
            raise ValueError(
                f"unknown spline order {value!r}; "
                f"expected an int 0..7 or one of {sorted(_SPLINE_ALIASES)}"
            )
        return int(_SPLINE_ALIASES[key])
    ivalue = int(value)
    if not 0 <= ivalue <= 7:
        raise ValueError(f"spline order must be in 0..7, got {ivalue}")
    return ivalue


def as_bound(value: int | str | Bound) -> int:
    """Normalise a boundary-condition argument to an ``int`` in ``0..7``.

    Accepts an integer, a :class:`Bound` enum, or a friendly string alias
    (e.g. ``"dct2"``, ``"wrap"``). Raises ``ValueError`` for an unknown alias
    or an out-of-range integer.
    """
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _BOUND_ALIASES:
            raise ValueError(
                f"unknown boundary condition {value!r}; "
                f"expected an int 0..7 or one of {sorted(_BOUND_ALIASES)}"
            )
        return int(_BOUND_ALIASES[key])
    ivalue = int(value)
    if not 0 <= ivalue <= 7:
        raise ValueError(f"boundary condition must be in 0..7, got {ivalue}")
    return ivalue


def check_dtype(*tensors: Tensor) -> None:
    """Validate that every tensor has a supported (float32/float64) dtype.

    Parameters
    ----------
    *tensors : torch.Tensor
        Tensors to validate.

    Raises
    ------
    TypeError
        If any tensor is not float32 or float64.
    """
    for t in tensors:
        if t.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(
                f"fastfields.torch only supports float32/float64 tensors, "
                f"got {t.dtype}."
            )


def stream_ptr(t: Tensor) -> int:
    """Return the CUDA stream handle to forward to the binding for ``t``.

    Parameters
    ----------
    t : torch.Tensor
        The primary tensor of the operation; its device selects the stream.

    Returns
    -------
    int
        ``torch.cuda.current_stream(t.device).cuda_stream`` for a CUDA tensor,
        or ``0`` (the default/CPU stream) for a CPU tensor.
    """
    if t.is_cuda:
        return torch.cuda.current_stream(t.device).cuda_stream
    return 0
