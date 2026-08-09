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


def check_inplace_leaf(name: str, t: Tensor) -> None:
    """Reject an in-place write to a grad-requiring leaf *before* mutating it.

    ``torch.autograd.Function`` performs this check only when it wraps the
    outputs -- i.e. **after** ``forward`` has already run. Our ``forward``
    bodies hand the caller's buffer straight to the native binding, so
    without this up-front guard the tensor is overwritten and *then* the
    ``RuntimeError`` is raised, leaving the caller with silently corrupted
    data on what looks like a cleanly-failed call. Native torch in-place ops
    (``Tensor.mul_`` & co.) raise without touching the data; this restores
    that contract.

    Mirrors autograd's own rule, including the ``no_grad`` escape hatch that
    makes the usual optimizer pattern (``with torch.no_grad(): p.mul_(...)``)
    keep working.

    Parameters
    ----------
    name : str
        Operation name, used in the error message.
    t : torch.Tensor
        The tensor the caller is asking to mutate in place.

    Raises
    ------
    RuntimeError
        If ``t`` is a grad-requiring leaf and grad mode is enabled.
    """
    if torch.is_grad_enabled() and t.requires_grad and t.is_leaf:
        raise RuntimeError(
            f"a leaf Variable that requires grad is being used in an "
            f"in-place operation ({name})."
        )


def raise_not_differentiable(name: str, reason: str) -> None:
    """Raise the standard "no gradient through this op" ``RuntimeError``.

    Used as the body of a ``torch.autograd.Function.backward`` for ops that
    are exposed on torch (for API parity with numpy/cupy) but have no
    supported gradient. Forward always runs normally -- including when an
    input requires grad -- so a graph can be built through the op; only an
    actual ``.backward()`` call that reaches this node raises. This is
    deliberately different from rejecting a grad-requiring input up front at
    call time: it lets the op sit inside a larger graph (e.g. as a
    stop-gradient boundary) and only fails, loudly, if someone actually tries
    to backpropagate through it.

    Parameters
    ----------
    name : str
        The operation name (as called by the user), used in the message.
    reason : str
        A short clause explaining *why* there is no gradient, e.g. "the
        distance transform has no meaningful gradient".

    Raises
    ------
    RuntimeError
        Always. Naming ``name`` and stating ``reason``.
    """
    raise RuntimeError(
        f"{name} is not differentiable: {reason} Do not call `.backward()` "
        f"through its output; call `.detach()` on the output (or the "
        f"inputs) first if you need to use it inside a larger autograd "
        f"graph."
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
