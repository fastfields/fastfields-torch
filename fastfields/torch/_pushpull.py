"""Spline-interpolation gather / scatter (pushpull) — torch.

Channel-last, x-first coordinate convention (spatial rank ``D`` = ``grid``'s
trailing axis; ``B`` leading batch dims):

* ``inp``  : ``(*B, *inshape,  C)``   — the volume (spline coefficients)
* ``grid`` : ``(*B, *outshape, D)``   — sampling coordinates, in voxels

``pull`` and ``push`` are differentiable **with respect to the field** (each is
the other's adjoint); the ``grid`` is treated as a constant (differentiating
through sample positions is not supported yet — pass a grid that does not
require grad).
"""

from __future__ import annotations

from typing import Sequence

import fastfields.dlpack as _fb
from fastfields.dlpack import as_bound, as_spline

import torch
from torch import Tensor

from ._util import check_dtype, stream_ptr

__all__ = ["pull", "push", "count", "grad"]


def _spatial(shape: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,) * ndim
    out = tuple(int(s) for s in shape)
    if len(out) != ndim:
        raise ValueError(f"shape must have length ndim={ndim}, got {shape!r}")
    return out


def _no_grid_grad(grid: Tensor) -> None:
    if grid.requires_grad:
        raise NotImplementedError(
            "differentiating pushpull through `grid` is not supported yet; "
            "pass a grid that does not require grad (grad flows to the field)."
        )


class _Pull(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, grid, spline, bound, extrapolate):
        out = inp.new_zeros((*grid.shape[:-1], inp.shape[-1]))
        _fb.pull(
            out,
            inp,
            grid,
            spline=spline,
            bound=bound,
            extrapolate=extrapolate,
            stream=stream_ptr(out),
        )
        d = grid.shape[-1]
        nbatch = grid.ndim - d - 1
        ctx.save_for_backward(grid)
        ctx.spline, ctx.bound, ctx.extrapolate = spline, bound, extrapolate
        ctx.inshape = tuple(inp.shape[nbatch : nbatch + d])
        ctx.channels = inp.shape[-1]
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (grid,) = ctx.saved_tensors
        ginp = None
        if ctx.needs_input_grad[0]:
            d = grid.shape[-1]
            nbatch = grid.ndim - d - 1
            ginp = grad_out.new_zeros(
                (*grid.shape[:nbatch], *ctx.inshape, ctx.channels)
            )
            _fb.push(
                ginp,
                grad_out.contiguous(),
                grid,
                spline=ctx.spline,
                bound=ctx.bound,
                extrapolate=ctx.extrapolate,
                stream=stream_ptr(ginp),
            )
        return ginp, None, None, None, None


class _Push(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, grid, spline, bound, extrapolate, spatial):
        d = grid.shape[-1]
        nbatch = grid.ndim - d - 1
        out = inp.new_zeros((*grid.shape[:nbatch], *spatial, inp.shape[-1]))
        _fb.push(
            out,
            inp,
            grid,
            spline=spline,
            bound=bound,
            extrapolate=extrapolate,
            stream=stream_ptr(out),
        )
        ctx.save_for_backward(grid)
        ctx.spline, ctx.bound, ctx.extrapolate = spline, bound, extrapolate
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (grid,) = ctx.saved_tensors
        ginp = None
        if ctx.needs_input_grad[0]:
            ginp = grad_out.new_zeros((*grid.shape[:-1], grad_out.shape[-1]))
            _fb.pull(
                ginp,
                grad_out.contiguous(),
                grid,
                spline=ctx.spline,
                bound=ctx.bound,
                extrapolate=ctx.extrapolate,
                stream=stream_ptr(ginp),
            )
        return ginp, None, None, None, None, None


def pull(
    inp: Tensor,
    grid: Tensor,
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
) -> Tensor:
    """Sample (pull) ``inp`` at ``grid``. Differentiable wrt ``inp``."""
    check_dtype(inp, grid)
    _no_grid_grad(grid)
    if grid.ndim != inp.ndim:
        raise ValueError("inp and grid must have the same rank")
    return _Pull.apply(
        inp, grid, as_spline(order), as_bound(bound), int(extrapolate)
    )


def push(
    inp: Tensor,
    grid: Tensor,
    shape: int | Sequence[int],
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
) -> Tensor:
    """Splat (push) ``inp`` into a volume of spatial size ``shape``.

    Adjoint of :func:`pull`; differentiable wrt ``inp``.
    """
    check_dtype(inp, grid)
    _no_grid_grad(grid)
    if grid.ndim != inp.ndim:
        raise ValueError("inp and grid must have the same rank")
    spatial = _spatial(shape, grid.shape[-1])
    return _Push.apply(
        inp, grid, as_spline(order), as_bound(bound), int(extrapolate), spatial
    )


def count(
    grid: Tensor,
    shape: int | Sequence[int],
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
) -> Tensor:
    """Splat ones into a volume of spatial size ``shape`` (not diff'able)."""
    check_dtype(grid)
    d = grid.shape[-1]
    nbatch = grid.ndim - d - 1
    spatial = _spatial(shape, d)
    out = grid.new_zeros((*grid.shape[:nbatch], *spatial, 1))
    _fb.count(
        out,
        grid.detach(),
        spline=as_spline(order),
        bound=as_bound(bound),
        extrapolate=int(extrapolate),
        stream=stream_ptr(out),
    )
    return out


def grad(
    inp: Tensor,
    grid: Tensor,
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
    abs: bool = False,
) -> Tensor:
    """Sample spatial gradients of ``inp`` at ``grid`` -> ``(*B,*out,C,D)``.

    Not differentiable (it already is a derivative operator).
    """
    check_dtype(inp, grid)
    if grid.ndim != inp.ndim:
        raise ValueError("inp and grid must have the same rank")
    d = grid.shape[-1]
    out = inp.new_zeros((*grid.shape[:-1], inp.shape[-1], d))
    _fb.grad(
        out,
        inp.detach(),
        grid.detach(),
        spline=as_spline(order),
        bound=as_bound(bound),
        extrapolate=int(extrapolate),
        abs=bool(abs),
        stream=stream_ptr(out),
    )
    return out
