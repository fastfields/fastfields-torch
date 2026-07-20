"""Spline coefficient prefilter (interpolating coefficients), with autograd.

Wraps ``fastfields_bind.spline_coeff`` (in-place prefilter along the last axis).
The prefilter is a linear, self-adjoint operator, so its backward applies the
same prefilter to the gradient -- mirroring ``jitfields`` ``SplineCoeff_``.
"""

from __future__ import annotations

import torch
from torch import Tensor

import fastfields_bind as _fb

from ._utils import as_contiguous, check_dtype

__all__ = ["spline_coeff"]


def spline_coeff(inp: Tensor, spline: int = 3, bound: int = 3) -> Tensor:
    """Compute interpolating spline coefficients along the last axis.

    Returns a new tensor (does not modify ``inp``), so it is safe for autograd.
    Differentiable with respect to ``inp``.

    Parameters
    ----------
    inp : `(..., N) tensor`
        Input samples.
    spline : `int`, default=3
        Spline order (orders 0 and 1 are no-ops).
    bound : `int`, default=3 (DCT2)
        Boundary condition.

    Returns
    -------
    coeff : `(..., N) tensor`
        Spline coefficients.
    """
    check_dtype(inp)
    return _SplineCoeff.apply(inp, spline, bound)


class _SplineCoeff(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, spline, bound):
        # Work on a contiguous copy: the binding filters in place.
        out = as_contiguous(inp).clone()
        _fb.spline_coeff(out, spline, bound)
        ctx.spline = spline
        ctx.bound = bound
        return out

    @staticmethod
    def backward(ctx, grad):
        gout = None
        if ctx.needs_input_grad[0]:
            gout = as_contiguous(grad).clone()
            _fb.spline_coeff(gout, ctx.spline, ctx.bound)
        return gout, None, None
