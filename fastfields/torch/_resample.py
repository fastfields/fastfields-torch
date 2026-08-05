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
* This wrapper always passes the binding an explicit per-dim ``scale`` --
  derived from ``anchor`` and the shapes, or from an explicit ``scale``
  override (see :func:`resample`). (The binding itself now handles
  ``scale=None`` by defaulting to ``inshape / outshape``; the wrapper resolves
  the scale up front so the ``anchor`` conventions are applied consistently.)
* On CUDA tensors the current stream is forwarded to the binding (see
  :mod:`fastfields.torch._util`).
"""

from __future__ import annotations

from typing import Optional, Sequence

import fastfields.dlpack as _fb
from fastfields.dlpack import (
    anchor_scale_shift,
    as_bound,
    as_spline,
    check_ndim,
    infer_ndim,
    resolve_out_spatial,
)

import torch
from torch import Tensor

from ._util import check_dtype, stream_ptr

__all__ = ["resample", "restriction", "spline_coeff", "spline_coeff_"]


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
    factor: float | Sequence[float] | None = None,
    shape: int | Sequence[int] | None = None,
    *,
    order: int | str = 2,
    bound: int | str = "dct2",
    ndim: Optional[int] = None,
    anchor: str = "centers",
    shift: Optional[float] = None,
    scale: Optional[Sequence[float]] = None,
) -> Tensor:
    """Spline resample (prolongation) of the last ``ndim`` axes.

    Differentiable with respect to ``inp`` (backward is ``restriction``). The
    signature matches the numpy/cupy wrappers so ``fastfields.auto.resample``
    dispatches consistently.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor, shape ``(..., *inshape)``.
    factor : float or sequence of float, optional
        Per-axis resize multiplier (scalar or sequence). Mutually exclusive
        with ``shape``; with neither, this is the identity.
    shape : int or sequence of int, optional
        Explicit output spatial size (the last ``ndim`` axes of the result).
    order : int or str, default=2
        Spline order (int ``0..7``, a :class:`Spline` enum, or a name such as
        ``"cubic"``).
    bound : int or str, default="dct2"
        Boundary condition (int, a :class:`Bound` enum, or a name such as
        ``"dct2"``/``"wrap"``).
    ndim : int, optional
        Number of trailing spatial dimensions (inferred from ``shape``/
        ``factor`` when omitted, else 1).
    anchor : {"centers", "edges", "first", "last"}, default="centers"
        Sampling-grid convention, matching ``interpol.resize``. Sets the
        default per-dim ``scale`` and ``shift`` (see
        :func:`fastfields.dlpack.anchor_scale_shift`). Abbreviations
        (``"c"``/``"e"``/``"f"``/
        ``"l"``) are accepted.
    shift : float, optional
        Sampling-shift override (default: the shift implied by ``anchor``).
    scale : sequence of float, optional
        Per-dim scale override (default: derived from ``anchor`` and the
        shapes), length ``ndim``.

    Returns
    -------
    torch.Tensor
        Resampled tensor, shape ``(..., *outshape)``.

    Raises
    ------
    ValueError
        If ``ndim`` is outside ``1..inp.dim()``, ``anchor``/``order``/``bound``
        is unknown, or ``factor``/``shape`` has the wrong length.
    """
    check_dtype(inp)
    ndim = infer_ndim(ndim, factor, shape)
    check_ndim(ndim, inp.dim())
    spatial_in = tuple(inp.shape[-ndim:])
    out_spatial = resolve_out_spatial(spatial_in, ndim, factor, shape)
    a_scale, a_shift = anchor_scale_shift(
        anchor, spatial_in, out_spatial, ndim
    )
    if scale is not None:
        a_scale = _effective_scale(scale, spatial_in, out_spatial, ndim)
    if shift is not None:
        a_shift = float(shift)
    return _Resample.apply(
        inp,
        list(out_spatial),
        as_spline(order),
        as_bound(bound),
        a_shift,
        a_scale,
        ndim,
    )


def restriction(
    inp: Tensor,
    factor: float | Sequence[float] | None = None,
    shape: int | Sequence[int] | None = None,
    *,
    order: int | str = 2,
    bound: int | str = "dct2",
    ndim: Optional[int] = None,
    anchor: str = "centers",
    shift: Optional[float] = None,
    scale: Optional[Sequence[float]] = None,
) -> Tensor:
    """Restriction (adjoint of :func:`resample`) of the last ``ndim`` axes.

    The binding accumulates into a buffer, which this wrapper pre-zeroes.
    Differentiable with respect to ``inp`` (backward is ``resample``). Shares
    :func:`resample`'s ``factor``/``shape``/``order`` signature.

    The ``anchor`` convention matches :func:`resample`; because the scale is
    derived from this call's own (input, output) shapes, a ``resample`` and a
    matching ``restriction`` use reciprocal scales and the same shift -- the
    adjoint relationship the binding expects.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor, shape ``(..., *inshape)``.
    factor : float or sequence of float, optional
        Per-axis resize multiplier (mutually exclusive with ``shape``).
    shape : int or sequence of int, optional
        Explicit output spatial size.
    order : int or str, default=2
        Spline order (see :func:`resample`).
    bound : int or str, default="dct2"
        Boundary condition (see :func:`resample`).
    ndim : int, optional
        Number of trailing spatial dimensions (inferred when omitted).
    anchor : {"centers", "edges", "first", "last"}, default="centers"
        Sampling-grid convention (see :func:`resample`).
    shift : float, optional
        Sampling-shift override (see :func:`resample`).
    scale : sequence of float, optional
        Per-dim scale override (see :func:`resample`).

    Returns
    -------
    torch.Tensor
        Restricted tensor, shape ``(..., *outshape)``.

    Raises
    ------
    ValueError
        If ``ndim`` is outside ``1..inp.dim()``, ``anchor``/``order``/``bound``
        is unknown, or ``factor``/``shape`` has the wrong length.
    """
    check_dtype(inp)
    ndim = infer_ndim(ndim, factor, shape)
    check_ndim(ndim, inp.dim())
    spatial_in = tuple(inp.shape[-ndim:])
    out_spatial = resolve_out_spatial(spatial_in, ndim, factor, shape)
    a_scale, a_shift = anchor_scale_shift(
        anchor, spatial_in, out_spatial, ndim
    )
    if scale is not None:
        a_scale = _effective_scale(scale, spatial_in, out_spatial, ndim)
    if shift is not None:
        a_shift = float(shift)
    return _Restriction.apply(
        inp,
        list(out_spatial),
        as_spline(order),
        as_bound(bound),
        a_shift,
        a_scale,
        ndim,
    )


def spline_coeff(
    inp: Tensor, order: int | str = 3, bound: int | str = "dct2"
) -> Tensor:
    """Compute interpolating spline coefficients along the last axis.

    Returns a new tensor (does not modify ``inp``), so it is safe for autograd.
    Differentiable with respect to ``inp``.

    Parameters
    ----------
    inp : torch.Tensor
        Input samples, shape ``(..., N)``.
    order : int or str, default=3
        Spline order (orders 0 and 1 are no-ops); accepts an int, a
        :class:`Spline` enum, or a name such as ``"cubic"`` (unified with the
        numpy/cupy wrappers).
    bound : int or str, default="dct2"
        Boundary condition (int, a :class:`Bound` enum, or a name).

    Returns
    -------
    torch.Tensor
        Spline coefficients, shape ``(..., N)``.
    """
    check_dtype(inp)
    return _SplineCoeff.apply(inp, as_spline(order), as_bound(bound))


def spline_coeff_(
    inp: Tensor, order: int | str = 3, bound: int | str = "dct2"
) -> Tensor:
    """In-place interpolating spline-coefficient prefilter, last axis.

    Differentiable with respect to ``inp`` (mirrors :func:`spline_coeff`):
    the prefilter is linear and self-adjoint, so its backward applies the
    same prefilter to the gradient and never reads the pre-mutation ``inp``
    -- only the saved ``order``/``bound`` scalars -- so overwriting ``inp``
    in place destroys no information backward needs (see
    ``API_CONTRACT.md``, "In-place policy"). As with any in-place op, a leaf
    tensor with ``requires_grad=True`` cannot be mutated (torch's ordinary
    leaf rule).

    Parameters
    ----------
    inp : torch.Tensor
        Input samples, shape ``(..., N)``. Mutated in place and returned.
    order : int or str, default=3
        Spline order (orders 0 and 1 are no-ops); accepts an int, a
        :class:`Spline` enum, or a name such as ``"cubic"``.
    bound : int or str, default="dct2"
        Boundary condition (int, a :class:`Bound` enum, or a name).

    Returns
    -------
    torch.Tensor
        ``inp`` (the same tensor object), now holding the spline
        coefficients.
    """
    check_dtype(inp)
    return _SplineCoeffInPlace.apply(inp, as_spline(order), as_bound(bound))


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


class _SplineCoeffInPlace(torch.autograd.Function):
    """``inp <- spline_coeff(inp)`` in place.

    Autograd-safe for the same reason as :class:`_SplineCoeff`: the
    prefilter is self-adjoint and its backward only needs the saved
    ``spline``/``bound`` scalars, never the pre-mutation ``inp`` --
    ``ctx.mark_dirty(inp)`` bumps the version counter so a stale save
    elsewhere raises instead of silently returning a wrong gradient.
    """

    @staticmethod
    def forward(ctx, inp, spline, bound):
        _fb.spline_coeff(inp, spline, bound, stream=stream_ptr(inp))
        ctx.mark_dirty(inp)
        ctx.spline = spline
        ctx.bound = bound
        return inp

    @staticmethod
    def backward(ctx, grad):
        gout = None
        if ctx.needs_input_grad[0]:
            gout = grad.clone()
            _fb.spline_coeff(
                gout, ctx.spline, ctx.bound, stream=stream_ptr(gout)
            )
        return gout, None, None
