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

import torch
from torch import Tensor

# The bindings only implement float32/float64 kernels.
_SUPPORTED_DTYPES = (torch.float32, torch.float64)


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


def as_contiguous(t: Tensor) -> Tensor:
    """Return a contiguous version of ``t`` (a no-op if already contiguous).

    Used to materialise the **output** buffers that the bindings write into;
    read-only inputs are passed with their native strides (the stride-aware
    C++/CUDA library consumes them zero-copy) and do not go through this.

    Parameters
    ----------
    t : torch.Tensor
        Tensor to make contiguous.

    Returns
    -------
    torch.Tensor
        ``t`` if already contiguous, otherwise a contiguous copy.
    """
    return t if t.is_contiguous() else t.contiguous()


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
