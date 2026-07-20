"""Spline resampling (prolongation) and restriction (its adjoint), with autograd.

Wraps ``fastfields.dlpack.resample`` and ``fastfields.dlpack.restriction``.

Adjoint relationship (verified empirically against the bindings, and matching
the spirit of ``jitfields`` ``resize.py`` where the backward of ``resize`` is
``restrict`` and vice-versa):

    transpose( resample(scale=s, shift=h) )  ==  restriction(scale=1/s, shift=h)

i.e. the adjoint uses the **reciprocal** scale and the **same** shift, with the
input/output shapes swapped. The backward passes below implement exactly that.

Notes
-----
* ``scale`` uses the binding convention "input-index per output-index".
* The binding segfaults when ``scale=None`` is passed, so this wrapper always
  computes an explicit scale from the shapes when the caller does not provide
  one (``scale[d] = inshape[d] / outshape[d]``).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

import fastfields.dlpack as _fb

from ._utils import as_contiguous, check_dtype

__all__ = ["resample", "restriction"]


def _normalize_shape(shape, ndim):
    if isinstance(shape, int):
        shape = [shape] * ndim
    shape = list(shape)
    if len(shape) != ndim:
        raise ValueError(f"Expected shape of length ndim={ndim}, got {shape}.")
    return shape


def _effective_scale(scale, inshape, outshape, ndim):
    """Resolve the per-dim scale, defaulting to ``inshape/outshape``."""
    if scale is None:
        return [float(inshape[d]) / float(outshape[d]) for d in range(ndim)]
    scale = list(scale)
    if len(scale) != ndim:
        raise ValueError(f"Expected scale of length ndim={ndim}, got {scale}.")
    return [float(s) for s in scale]


def resample(
    inp: Tensor,
    shape,
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
    inp : `(..., *inshape) tensor`
        Input tensor.
    shape : `int or sequence[int]`
        Output spatial shape (the last ``ndim`` axes of the result).
    spline : `int`, default=2
        Spline order.
    bound : `int`, default=3 (DCT2)
        Boundary condition.
    shift : `float`, default=0.0
        Sampling shift.
    scale : `sequence[float]`, optional
        Per-dim scale (input-index per output-index), length ``ndim``.
        Defaults to ``inshape/outshape``.
    ndim : `int`, default=1
        Number of spatial dimensions.

    Returns
    -------
    out : `(..., *shape) tensor`
        Resampled tensor.
    """
    check_dtype(inp)
    shape = _normalize_shape(shape, ndim)
    scale = _effective_scale(scale, inp.shape[-ndim:], shape, ndim)
    return _Resample.apply(inp, shape, spline, bound, shift, scale, ndim)


def restriction(
    inp: Tensor,
    shape,
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
    inp : `(..., *inshape) tensor`
        Input tensor.
    shape : `int or sequence[int]`
        Output spatial shape (the last ``ndim`` axes of the result).
    spline, bound, shift, ndim
        See :func:`resample`.
    scale : `sequence[float]`, optional
        Per-dim scale (input-index per output-index), length ``ndim``.
        Defaults to ``inshape/outshape``.

    Returns
    -------
    out : `(..., *shape) tensor`
        Restricted tensor.
    """
    check_dtype(inp)
    shape = _normalize_shape(shape, ndim)
    scale = _effective_scale(scale, inp.shape[-ndim:], shape, ndim)
    return _Restriction.apply(inp, shape, spline, bound, shift, scale, ndim)


def _do_resample(out, inp, spline, bound, shift, scale, ndim):
    _fb.resample(out, inp, spline, bound, shift, scale, ndim)


def _do_restriction(out, inp, spline, bound, shift, scale, ndim):
    # `restriction` accumulates into `out`; it must be pre-zeroed.
    out.zero_()
    _fb.restriction(out, inp, spline, bound, shift, scale, ndim)


def _reciprocal(scale):
    return [1.0 / s for s in scale]


class _Resample(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, shape, spline, bound, shift, scale, ndim):
        inp = as_contiguous(inp)
        out = inp.new_empty([*inp.shape[:-ndim], *shape])
        _do_resample(out, inp, spline, bound, shift, scale, ndim)
        ctx.opt = (list(inp.shape[-ndim:]), spline, bound, shift, scale, ndim)
        return out

    @staticmethod
    def backward(ctx, grad):
        inshape, spline, bound, shift, scale, ndim = ctx.opt
        ginp = None
        if ctx.needs_input_grad[0]:
            grad = as_contiguous(grad)
            ginp = grad.new_empty([*grad.shape[:-ndim], *inshape])
            # adjoint of resample(scale=s) is restriction(scale=1/s, same shift)
            _do_restriction(ginp, grad, spline, bound, shift,
                            _reciprocal(scale), ndim)
        return (ginp,) + (None,) * 6


class _Restriction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, shape, spline, bound, shift, scale, ndim):
        inp = as_contiguous(inp)
        out = inp.new_empty([*inp.shape[:-ndim], *shape])
        _do_restriction(out, inp, spline, bound, shift, scale, ndim)
        ctx.opt = (list(inp.shape[-ndim:]), spline, bound, shift, scale, ndim)
        return out

    @staticmethod
    def backward(ctx, grad):
        inshape, spline, bound, shift, scale, ndim = ctx.opt
        ginp = None
        if ctx.needs_input_grad[0]:
            grad = as_contiguous(grad)
            ginp = grad.new_empty([*grad.shape[:-ndim], *inshape])
            # adjoint of restriction(scale=s) is resample(scale=1/s, same shift)
            _do_resample(ginp, grad, spline, bound, shift,
                         _reciprocal(scale), ndim)
        return (ginp,) + (None,) * 6
