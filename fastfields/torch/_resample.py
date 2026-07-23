"""Spline resampling, restriction and coefficient prefilter, with autograd.

Wraps ``fastfields.dlpack.resample`` / ``restriction`` (prolongation and its
adjoint) and ``fastfields.dlpack.spline_coeff`` (interpolating-coefficient
prefilter along the last axis).

Adjoint relationship (verified empirically against the bindings, and matching
the spirit of ``jitfields`` ``resize.py`` where the backward of ``resize`` is
``restrict`` and vice-versa)::

    transpose( resample(scale=s, shift=h) ) == restriction(scale=1/s, shift=h)

i.e. the adjoint uses the **reciprocal** scale and the **same** shift, with the
input/output shapes swapped. The backward passes below implement exactly that.
The spline-coefficient prefilter is linear and self-adjoint, so its backward
applies the same prefilter to the gradient.

Notes
-----
* ``scale`` uses the binding convention "input-index per output-index".
* The binding segfaults when ``scale=None`` is passed, so this wrapper always
  computes an explicit scale from the shapes when the caller does not provide
  one (``scale[d] = inshape[d] / outshape[d]``).
* On CUDA tensors the current stream is forwarded to the binding (see
  :mod:`fastfields.torch._util`).
"""

from __future__ import annotations

from typing import Optional, Sequence

import fastfields.dlpack as _fb

import torch
from torch import Tensor

from ._util import check_dtype, stream_ptr

__all__ = ["resample", "restriction", "spline_coeff"]


def _normalize_shape(shape: int | Sequence[int], ndim: int) -> list[int]:
    """Normalise a shape argument to a list of length ``ndim``.

    Parameters
    ----------
    shape : int or sequence of int
        Output spatial shape (an ``int`` is repeated ``ndim`` times).
    ndim : int
        Expected number of spatial dimensions.

    Returns
    -------
    list of int
        The normalised shape.

    Raises
    ------
    ValueError
        If ``shape`` does not have length ``ndim``.
    """
    if isinstance(shape, int):
        shape = [shape] * ndim
    shape = list(shape)
    if len(shape) != ndim:
        raise ValueError(f"Expected shape of length ndim={ndim}, got {shape}.")
    return shape


def _effective_scale(
    scale: Optional[Sequence[float]],
    inshape: Sequence[int],
    outshape: Sequence[int],
    ndim: int,
) -> list[float]:
    """Resolve the per-dim scale, defaulting to ``inshape/outshape``.

    Parameters
    ----------
    scale : sequence of float, optional
        Explicit per-dim scale (input-index per output-index). When ``None``,
        it is computed from the shapes.
    inshape, outshape : sequence of int
        Input and output spatial shapes.
    ndim : int
        Number of spatial dimensions.

    Returns
    -------
    list of float
        The per-dim scale, length ``ndim``.

    Raises
    ------
    ValueError
        If an explicit ``scale`` does not have length ``ndim``.
    """
    if scale is None:
        return [float(inshape[d]) / float(outshape[d]) for d in range(ndim)]
    scale = list(scale)
    if len(scale) != ndim:
        raise ValueError(f"Expected scale of length ndim={ndim}, got {scale}.")
    return [float(s) for s in scale]


def resample(
    inp: Tensor,
    shape: int | Sequence[int],
    spline: int = 2,
    bound: int = 3,
    shift: float = 0.0,
    scale: Optional[Sequence[float]] = None,
    ndim: int = 1,
) -> Tensor:
    """Spline resample (prolongation) of the last ``ndim`` axes.

    Differentiable with respect to ``inp`` (backward is ``restriction``).

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor, shape ``(..., *inshape)``.
    shape : int or sequence of int
        Output spatial shape (the last ``ndim`` axes of the result).
    spline : int, default=2
        Spline order.
    bound : int, default=3
        Boundary condition (default DCT2).
    shift : float, default=0.0
        Sampling shift.
    scale : sequence of float, optional
        Per-dim scale (input-index per output-index), length ``ndim``.
        Defaults to ``inshape/outshape``.
    ndim : int, default=1
        Number of spatial dimensions.

    Returns
    -------
    torch.Tensor
        Resampled tensor, shape ``(..., *shape)``.
    """
    check_dtype(inp)
    shape = _normalize_shape(shape, ndim)
    scale = _effective_scale(scale, inp.shape[-ndim:], shape, ndim)
    return _Resample.apply(inp, shape, spline, bound, shift, scale, ndim)


def restriction(
    inp: Tensor,
    shape: int | Sequence[int],
    spline: int = 2,
    bound: int = 3,
    shift: float = 0.0,
    scale: Optional[Sequence[float]] = None,
    ndim: int = 1,
) -> Tensor:
    """Restriction (adjoint of :func:`resample`) of the last ``ndim`` axes.

    The binding accumulates into a buffer, which this wrapper pre-zeroes.
    Differentiable with respect to ``inp`` (backward is ``resample``).

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor, shape ``(..., *inshape)``.
    shape : int or sequence of int
        Output spatial shape (the last ``ndim`` axes of the result).
    spline : int, default=2
        Spline order.
    bound : int, default=3
        Boundary condition (default DCT2).
    shift : float, default=0.0
        Sampling shift.
    scale : sequence of float, optional
        Per-dim scale (input-index per output-index), length ``ndim``.
        Defaults to ``inshape/outshape``.
    ndim : int, default=1
        Number of spatial dimensions.

    Returns
    -------
    torch.Tensor
        Restricted tensor, shape ``(..., *shape)``.
    """
    check_dtype(inp)
    shape = _normalize_shape(shape, ndim)
    scale = _effective_scale(scale, inp.shape[-ndim:], shape, ndim)
    return _Restriction.apply(inp, shape, spline, bound, shift, scale, ndim)


def spline_coeff(inp: Tensor, spline: int = 3, bound: int = 3) -> Tensor:
    """Compute interpolating spline coefficients along the last axis.

    Returns a new tensor (does not modify ``inp``), so it is safe for autograd.
    Differentiable with respect to ``inp``.

    Parameters
    ----------
    inp : torch.Tensor
        Input samples, shape ``(..., N)``.
    spline : int, default=3
        Spline order (orders 0 and 1 are no-ops).
    bound : int, default=3
        Boundary condition (default DCT2).

    Returns
    -------
    torch.Tensor
        Spline coefficients, shape ``(..., N)``.
    """
    check_dtype(inp)
    return _SplineCoeff.apply(inp, spline, bound)


def _do_resample(
    out: Tensor,
    inp: Tensor,
    spline: int,
    bound: int,
    shift: float,
    scale: Sequence[float],
    ndim: int,
) -> None:
    """Call the resample binding, forwarding ``out``'s CUDA stream."""
    _fb.resample(
        out, inp, spline, bound, shift, scale, ndim, stream=stream_ptr(out)
    )


def _do_restriction(
    out: Tensor,
    inp: Tensor,
    spline: int,
    bound: int,
    shift: float,
    scale: Sequence[float],
    ndim: int,
) -> None:
    """Call the restriction binding (accumulating), forwarding the stream."""
    # `restriction` accumulates into `out`; it must be pre-zeroed.
    out.zero_()
    _fb.restriction(
        out, inp, spline, bound, shift, scale, ndim, stream=stream_ptr(out)
    )


def _reciprocal(scale: Sequence[float]) -> list[float]:
    """Return the element-wise reciprocal of ``scale``."""
    return [1.0 / s for s in scale]


class _Resample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, shape, spline, bound, shift, scale, ndim):
        # inp is a read-only input: the stride-aware binding reads it
        # zero-copy, so we do NOT force contiguity. The output buffer
        # (new_empty) is contiguous regardless of inp's layout.
        out = inp.new_empty([*inp.shape[:-ndim], *shape])
        _do_resample(out, inp, spline, bound, shift, scale, ndim)
        ctx.opt = (list(inp.shape[-ndim:]), spline, bound, shift, scale, ndim)
        return out

    @staticmethod
    def backward(ctx, grad):
        inshape, spline, bound, shift, scale, ndim = ctx.opt
        ginp = None
        if ctx.needs_input_grad[0]:
            ginp = grad.new_empty([*grad.shape[:-ndim], *inshape])
            # adjoint of resample(s) is restriction(1/s, same shift)
            _do_restriction(
                ginp, grad, spline, bound, shift, _reciprocal(scale), ndim
            )
        return (ginp,) + (None,) * 6


class _Restriction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, shape, spline, bound, shift, scale, ndim):
        # See _Resample.forward: read-only input, contiguous output.
        out = inp.new_empty([*inp.shape[:-ndim], *shape])
        _do_restriction(out, inp, spline, bound, shift, scale, ndim)
        ctx.opt = (list(inp.shape[-ndim:]), spline, bound, shift, scale, ndim)
        return out

    @staticmethod
    def backward(ctx, grad):
        inshape, spline, bound, shift, scale, ndim = ctx.opt
        ginp = None
        if ctx.needs_input_grad[0]:
            ginp = grad.new_empty([*grad.shape[:-ndim], *inshape])
            # adjoint of restriction(s) is resample(1/s, same shift)
            _do_resample(
                ginp, grad, spline, bound, shift, _reciprocal(scale), ndim
            )
        return (ginp,) + (None,) * 6


class _SplineCoeff(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, spline, bound):
        # Layout-preserving copy: the binding filters in place through the
        # tensor's strides, so this clone is the output buffer (its layout
        # matches inp -- the stride-aware binding needs no contiguous copy).
        out = inp.clone()
        _fb.spline_coeff(out, spline, bound, stream=stream_ptr(out))
        ctx.spline = spline
        ctx.bound = bound
        return out

    @staticmethod
    def backward(ctx, grad):
        gout = None
        if ctx.needs_input_grad[0]:
            gout = grad.clone()
            _fb.spline_coeff(
                gout, ctx.spline, ctx.bound, stream=stream_ptr(gout)
            )
        return gout, None, None
