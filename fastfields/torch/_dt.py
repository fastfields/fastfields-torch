"""Distance-transform wrappers (torch).

Image distance transforms (``dt_euclidean``, ``dt_l1``), point-to-mesh
distance (``dt_mesh``) and point-to-spline distance
(``dt_spline_{table,brent,gaussnewton}``).

These operators are **not differentiable** -- the distance transform / point-
to-mesh distance have no meaningful gradient. They are still exposed on torch
(for API parity with ``fastfields.numpy``/``fastfields.cupy``): forward always
runs, including when an input requires grad, so the output can sit inside a
larger graph; only calling ``.backward()`` through that output raises a clear
``RuntimeError`` (see :func:`fastfields.torch._util.raise_not_differentiable`).
This mirrors how ``jitfields`` guards these ops, but the guard now lives in
``backward`` instead of at call time -- see ``API_CONTRACT.md``, "In-place
policy", and fastfields#4.

The underlying bindings mutate their input in place. The out-of-place
functions (``dt_euclidean``, ``dt_l1``) clone first; the in-place variants
(``dt_euclidean_``, ``dt_l1_``) write through the caller's tensor, exactly
like the numpy/cupy ``_``-suffixed forms. On CUDA tensors the current stream
is forwarded to the binding (see :mod:`fastfields.torch._util`).
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import fastfields.dlpack as _fb
from fastfields.helpers import as_bound, as_spline

import torch
from torch import Tensor

from ._util import (
    check_dtype,
    check_inplace_leaf,
    raise_not_differentiable,
    stream_ptr,
)

__all__ = [
    "dt_euclidean",
    "dt_euclidean_",
    "dt_l1",
    "dt_l1_",
    "dt_mesh",
    "dt_spline_table",
    "dt_spline_brent",
    "dt_spline_gaussnewton",
]


class _DtEuclidean(torch.autograd.Function):
    """``out = dt_euclidean(inp)`` (out-of-place, ``inp`` left untouched)."""

    @staticmethod
    def forward(ctx, inp, voxel_spacing):
        out = inp.clone()
        _fb.dt_euclidean(out, voxel_spacing, stream=stream_ptr(out))
        return out

    @staticmethod
    def backward(ctx, grad):
        raise_not_differentiable(
            "dt_euclidean",
            "the distance transform has no meaningful gradient.",
        )


class _DtEuclidean_(torch.autograd.Function):
    """``inp <- dt_euclidean(inp)`` in place; returns ``inp``."""

    @staticmethod
    def forward(ctx, inp, voxel_spacing):
        _fb.dt_euclidean(inp, voxel_spacing, stream=stream_ptr(inp))
        ctx.mark_dirty(inp)
        return inp

    @staticmethod
    def backward(ctx, grad):
        raise_not_differentiable(
            "dt_euclidean_",
            "the distance transform has no meaningful gradient.",
        )


def dt_euclidean(inp: Tensor, voxel_spacing: float = 1.0) -> Tensor:
    """Squared Euclidean distance transform along the last axis.

    Not differentiable -- calling ``.backward()`` through this raises
    ``RuntimeError``. Forward still runs normally (including when ``inp``
    requires grad); only an actual backward call fails.

    Parameters
    ----------
    inp : torch.Tensor
        Input holding ``0`` at feature locations and ``+inf`` elsewhere.
    voxel_spacing : float, default=1.0
        Physical spacing between samples along the last axis.

    Returns
    -------
    torch.Tensor
        The distance transform (a new tensor with ``inp``'s layout); ``inp``
        is left untouched. See :func:`dt_euclidean_` for the in-place
        variant.

    Raises
    ------
    TypeError
        If ``inp`` is not float32/float64.
    RuntimeError
        If ``.backward()`` is called through the output.
    """
    check_dtype(inp)
    return _DtEuclidean.apply(inp, voxel_spacing)


def dt_euclidean_(inp: Tensor, voxel_spacing: float = 1.0) -> Tensor:
    """In-place squared Euclidean distance transform along the last axis.

    Not differentiable -- calling ``.backward()`` through this raises
    ``RuntimeError``. As with any in-place op, a leaf tensor with
    ``requires_grad=True`` cannot be mutated (torch's ordinary leaf rule).

    Parameters
    ----------
    inp : torch.Tensor
        Input holding ``0`` at feature locations and ``+inf`` elsewhere.
        Mutated in place and returned.
    voxel_spacing : float, default=1.0
        Physical spacing between samples along the last axis.

    Returns
    -------
    torch.Tensor
        ``inp`` (the same tensor object), now holding the distance
        transform.

    Raises
    ------
    TypeError
        If ``inp`` is not float32/float64.
    RuntimeError
        If ``.backward()`` is called through the output.
    """
    check_dtype(inp)
    check_inplace_leaf("dt_euclidean_", inp)
    return _DtEuclidean_.apply(inp, voxel_spacing)


class _DtL1(torch.autograd.Function):
    """``out = dt_l1(inp)`` (out-of-place, ``inp`` left untouched)."""

    @staticmethod
    def forward(ctx, inp, voxel_spacing):
        out = inp.clone()
        _fb.dt_l1(out, voxel_spacing, stream=stream_ptr(out))
        return out

    @staticmethod
    def backward(ctx, grad):
        raise_not_differentiable(
            "dt_l1", "the distance transform has no meaningful gradient."
        )


class _DtL1_(torch.autograd.Function):
    """``inp <- dt_l1(inp)`` in place; returns ``inp``."""

    @staticmethod
    def forward(ctx, inp, voxel_spacing):
        _fb.dt_l1(inp, voxel_spacing, stream=stream_ptr(inp))
        ctx.mark_dirty(inp)
        return inp

    @staticmethod
    def backward(ctx, grad):
        raise_not_differentiable(
            "dt_l1_", "the distance transform has no meaningful gradient."
        )


def dt_l1(inp: Tensor, voxel_spacing: float = 1.0) -> Tensor:
    """L1 distance transform along the last axis.

    Not differentiable -- calling ``.backward()`` through this raises
    ``RuntimeError``. See :func:`dt_euclidean` for the input convention.

    Parameters
    ----------
    inp : torch.Tensor
        Input holding ``0`` at feature locations and ``+inf`` elsewhere.
    voxel_spacing : float, default=1.0
        Physical spacing between samples along the last axis.

    Returns
    -------
    torch.Tensor
        The distance transform (a new tensor with ``inp``'s layout); ``inp``
        is left untouched. See :func:`dt_l1_` for the in-place variant.

    Raises
    ------
    TypeError
        If ``inp`` is not float32/float64.
    RuntimeError
        If ``.backward()`` is called through the output.
    """
    check_dtype(inp)
    return _DtL1.apply(inp, voxel_spacing)


def dt_l1_(inp: Tensor, voxel_spacing: float = 1.0) -> Tensor:
    """In-place L1 distance transform along the last axis.

    Not differentiable -- calling ``.backward()`` through this raises
    ``RuntimeError``. As with any in-place op, a leaf tensor with
    ``requires_grad=True`` cannot be mutated (torch's ordinary leaf rule).

    Parameters
    ----------
    inp : torch.Tensor
        Input holding ``0`` at feature locations and ``+inf`` elsewhere.
        Mutated in place and returned.
    voxel_spacing : float, default=1.0
        Physical spacing between samples along the last axis.

    Returns
    -------
    torch.Tensor
        ``inp`` (the same tensor object), now holding the distance
        transform.

    Raises
    ------
    TypeError
        If ``inp`` is not float32/float64.
    RuntimeError
        If ``.backward()`` is called through the output.
    """
    check_dtype(inp)
    check_inplace_leaf("dt_l1_", inp)
    return _DtL1_.apply(inp, voxel_spacing)


class _DtMesh(torch.autograd.Function):
    """Point-to-mesh (squared) distance; ``loc``/``vertices`` left untouched.

    No in-place variant exists on any backend (the output shape/target
    differs from every input, so there is no natural buffer to mutate).
    """

    @staticmethod
    def forward(ctx, loc, vertices, faces, signed, naive, return_nearest):
        # A single mesh is queried by an arbitrarily batched point set: only
        # `loc` carries batch dims, and `vertices`/`faces` stay unbatched
        # (the binding requires exactly (N, D) / (M, D)). Outputs are
        # contiguous real buffers shaped like loc.shape[:-1].
        batch = loc.shape[:-1]
        dist = loc.new_empty(batch)
        nearest = None
        if return_nearest:
            nearest = torch.empty(batch, dtype=torch.int64, device=loc.device)
        _fb.dt_mesh(
            dist,
            nearest,
            loc,
            vertices,
            faces,
            signed,
            naive,
            stream=stream_ptr(dist),
        )
        if return_nearest:
            return dist, nearest
        return dist

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise_not_differentiable(
            "dt_mesh", "point-to-mesh distance has no meaningful gradient."
        )


def dt_mesh(
    loc: Tensor,
    vertices: Tensor,
    faces: Tensor,
    signed: bool = True,
    naive: bool = False,
    return_nearest: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Point-to-triangular-mesh (squared) distance.

    Not differentiable -- calling ``.backward()`` through ``dist`` raises
    ``RuntimeError``. Forward still runs normally (including when ``loc``/
    ``vertices`` require grad); only an actual backward call fails.

    Parameters
    ----------
    loc : torch.Tensor
        Query points, shape ``(*B, D)``.
    vertices : torch.Tensor
        Mesh vertices, shape ``(V, D)`` (same float dtype as ``loc``).
    faces : torch.Tensor
        Triangle vertex indices, shape ``(F, D)`` (integer tensor; cast
        to int64 before the binding).
    signed : bool, default=True
        Return signed distances.
    naive : bool, default=False
        Use the naive (brute-force) algorithm.
    return_nearest : bool, default=False
        Also return the nearest-vertex index per query point.

    Returns
    -------
    dist : torch.Tensor
        (Squared) distance per query point, shape ``(*B,)``.
    nearest_vertex : torch.Tensor, optional
        Nearest-vertex index per query point, only if ``return_nearest`` is
        ``True``.

    Raises
    ------
    TypeError
        If ``loc``/``vertices`` are not float32/float64.
    ValueError
        If ``vertices`` or ``faces`` is not 2D.
    RuntimeError
        If ``.backward()`` is called through ``dist``.

    Notes
    -----
    A **single** mesh is queried by an arbitrarily batched point set: only
    ``loc`` carries batch dims, and ``dist`` (and ``nearest_vertex``) are
    allocated with ``loc.shape[:-1]``. The mesh operands are *not* broadcast
    against that batch -- ``vertices`` and ``faces`` must be exactly
    ``(V, D)`` / ``(F, D)``, matching jitfields' ``mesh_distance`` and the
    ``ff::dt_mesh`` contract. (An earlier revision broadcast them to
    ``(*B, V, D)``, which the binding rejects outright; see fastfields#32.)
    """
    check_dtype(loc, vertices)
    # faces holds integer vertex indices: the binding reads them at int64
    # width, so an int32 (or other) faces array would be misread. Normalize to
    # int64 before the binding (mirrors the numpy wrapper).
    if faces.dtype != torch.int64:
        faces = faces.to(torch.int64)
    if vertices.ndim != 2:
        raise ValueError(
            "dt_mesh: vertices must be a 2D (V, D) tensor describing a "
            f"single mesh, got shape {tuple(vertices.shape)}"
        )
    if faces.ndim != 2:
        raise ValueError(
            "dt_mesh: faces must be a 2D (F, D) tensor describing a single "
            f"mesh, got shape {tuple(faces.shape)}"
        )
    return _DtMesh.apply(loc, vertices, faces, signed, naive, return_nearest)

# --------------------------------------------------------------------------- #
# Point-to-spline distance                                                    #
# --------------------------------------------------------------------------- #
#
# Three strategies for the same problem -- find the time `t` minimising
# ||loc - spline(t)||^2 -- and the same contract: `loc` core dim (D,), `coeff`
# core dim (N, D), both outputs shaped like the broadcast batch. Like the
# other distance ops they are not differentiable (the result is the *argmin*
# of an inner optimisation, which the library does not differentiate), so
# forward always runs and only `.backward()` through an output raises.
#
# `dt_spline_brent` and `dt_spline_gaussnewton` *refine* the time/distance
# buffers they are handed, so these wrappers seed them: `time` with the
# midpoint of the spline's parameter range and `dist` with +inf, which is the
# cold start. Seed them yourself with a coarse `dt_spline_table` pass (and
# call the binding directly) if you have a better starting point.


def _spline_operands(loc: Tensor, coeff: Tensor, *rest: Tensor):
    """Validate and broadcast a (loc, coeff, *rest) spline-distance call."""
    check_dtype(loc, coeff, *rest)
    if loc.ndim < 1:
        raise ValueError("dt_spline_*: loc must have at least one axis (D,)")
    if coeff.ndim < 2:
        raise ValueError(
            "dt_spline_*: coeff must have at least two axes (N, D)"
        )
    if coeff.shape[-1] != loc.shape[-1]:
        raise ValueError(
            f"dt_spline_*: loc has D={loc.shape[-1]} coordinates but coeff "
            f"has D={coeff.shape[-1]}"
        )
    shapes = [loc.shape[:-1], coeff.shape[:-2]]
    shapes += [t.shape[:-1] for t in rest]
    batch = torch.broadcast_shapes(*shapes)
    out = [loc.expand(*batch, loc.shape[-1]),
           coeff.expand(*batch, *coeff.shape[-2:])]
    out += [t.expand(*batch, t.shape[-1]) for t in rest]
    return batch, out


def _spline_outputs(loc: Tensor, batch: Sequence[int], seed: bool):
    """Allocate the (time, dist) output buffers for a spline-distance call."""
    time = loc.new_empty(tuple(batch))
    dist = loc.new_empty(tuple(batch))
    if seed:
        # Brent / Gauss-Newton refine whatever is in the buffers.
        time.fill_(0.0)
        dist.fill_(float("inf"))
    return time, dist


class _DtSpline(torch.autograd.Function):
    """Any of the three point-to-spline distance ops; not differentiable.

    ``loc`` / ``coeff`` / ``times`` are passed as real ``Function`` inputs,
    not captured in a closure, so that autograd records the edge: if one of
    them requires grad the outputs do too, and a ``.backward()`` through them
    reaches :meth:`backward` and gets the op-specific message rather than
    torch's generic "does not require grad". That is the contract the other
    non-differentiable ops in this module follow.
    """

    @staticmethod
    def forward(ctx, kind, time, dist, loc, coeff, times, params):
        spline, bound, max_iter, tol, step = params
        if kind == "dt_spline_table":
            _fb.dt_spline_table(
                time, dist, loc, coeff, times,
                spline=spline, bound=bound, stream=stream_ptr(dist),
            )
        elif kind == "dt_spline_brent":
            _fb.dt_spline_brent(
                time, dist, loc, coeff, max_iter, tol, step,
                spline=spline, bound=bound, stream=stream_ptr(dist),
            )
        else:
            _fb.dt_spline_gaussnewton(
                time, dist, loc, coeff, max_iter, tol,
                spline=spline, bound=bound, stream=stream_ptr(dist),
            )
        ctx.kind = kind
        return dist, time

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise_not_differentiable(
            ctx.kind,
            "point-to-spline distance is the value at an argmin found by an "
            "inner optimisation, which the library does not differentiate.",
        )


def dt_spline_table(
    loc: Tensor,
    coeff: Tensor,
    times: Tensor,
    *,
    order: int | str = 3,
    bound: int | str = "dct2",
) -> Tuple[Tensor, Tensor]:
    """Point-to-spline (squared) distance via a dictionary of candidate times.

    Not differentiable -- calling ``.backward()`` through either output raises
    ``RuntimeError``. Forward still runs normally.

    Parameters
    ----------
    loc : `(*B, D) tensor`
        Query points.
    coeff : `(*B, N, D) tensor`
        Spline coefficients.
    times : `(*B, K) tensor`
        Candidate times to try.
    order : int or str, default=3
        Spline order.
    bound : int or str, default="dct2"
        Boundary condition.

    Returns
    -------
    dist : `(*B,) tensor`
        Squared distance per query point.
    time : `(*B,) tensor`
        Best time per query point.

    Notes
    -----
    Batch dims of ``loc`` / ``coeff`` / ``times`` are broadcast together with
    ``Tensor.expand`` (0-stride views, no copy), which the stride-aware
    binding consumes directly. The return order matches
    ``fastfields.numpy.dt_spline_table``: ``(dist, time)``.
    """
    batch, (loc, coeff, times) = _spline_operands(loc, coeff, times)
    time, dist = _spline_outputs(loc, batch, seed=False)
    return _DtSpline.apply(
        "dt_spline_table",
        time,
        dist,
        loc,
        coeff,
        times,
        (as_spline(order), as_bound(bound), 0, 0.0, 0.0),
    )


def dt_spline_brent(
    loc: Tensor,
    coeff: Tensor,
    *,
    max_iter: int = 64,
    tol: float = 1e-6,
    step: float = 0.1,
    order: int | str = 3,
    bound: int | str = "dct2",
) -> Tuple[Tensor, Tensor]:
    """Point-to-spline (squared) distance via Brent's method.

    Not differentiable (see :func:`dt_spline_table`).

    Parameters
    ----------
    loc : `(*B, D) tensor`
        Query points.
    coeff : `(*B, N, D) tensor`
        Spline coefficients.
    max_iter : int, default=64
        Maximum number of Brent iterations.
    tol : float, default=1e-6
        Convergence tolerance.
    step : float, default=0.1
        Initial bracket half-width; must be wide enough to bracket the
        optimum.
    order : int or str, default=3
        Spline order.
    bound : int or str, default="dct2"
        Boundary condition.

    Returns
    -------
    dist : `(*B,) tensor`
        Squared distance per query point.
    time : `(*B,) tensor`
        Best time per query point.
    """
    batch, (loc, coeff) = _spline_operands(loc, coeff)
    time, dist = _spline_outputs(loc, batch, seed=True)
    return _DtSpline.apply(
        "dt_spline_brent",
        time,
        dist,
        loc,
        coeff,
        None,
        (
            as_spline(order),
            as_bound(bound),
            int(max_iter),
            float(tol),
            float(step),
        ),
    )


def dt_spline_gaussnewton(
    loc: Tensor,
    coeff: Tensor,
    *,
    max_iter: int = 64,
    tol: float = 1e-6,
    order: int | str = 3,
    bound: int | str = "dct2",
) -> Tuple[Tensor, Tensor]:
    """Point-to-spline (squared) distance via Gauss-Newton optimization.

    Not differentiable (see :func:`dt_spline_table`).

    Parameters
    ----------
    loc : `(*B, D) tensor`
        Query points.
    coeff : `(*B, N, D) tensor`
        Spline coefficients.
    max_iter : int, default=64
        Maximum number of Gauss-Newton iterations.
    tol : float, default=1e-6
        Convergence tolerance.
    order : int or str, default=3
        Spline order.
    bound : int or str, default="dct2"
        Boundary condition.

    Returns
    -------
    dist : `(*B,) tensor`
        Squared distance per query point.
    time : `(*B,) tensor`
        Best time per query point.
    """
    batch, (loc, coeff) = _spline_operands(loc, coeff)
    time, dist = _spline_outputs(loc, batch, seed=True)
    return _DtSpline.apply(
        "dt_spline_gaussnewton",
        time,
        dist,
        loc,
        coeff,
        None,
        (as_spline(order), as_bound(bound), int(max_iter), float(tol), 0.0),
    )
