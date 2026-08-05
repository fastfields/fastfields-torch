"""Distance-transform wrappers (torch).

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

from typing import Tuple, Union

import fastfields.dlpack as _fb

import torch
from torch import Tensor

from ._util import check_dtype, raise_not_differentiable, stream_ptr

__all__ = [
    "dt_euclidean",
    "dt_euclidean_",
    "dt_l1",
    "dt_l1_",
    "dt_mesh",
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
    return _DtL1_.apply(inp, voxel_spacing)


class _DtMesh(torch.autograd.Function):
    """Point-to-mesh (squared) distance; ``loc``/``vertices`` left untouched.

    No in-place variant exists on any backend (the output shape/target
    differs from every input, so there is no natural buffer to mutate).
    """

    @staticmethod
    def forward(ctx, loc, vertices, faces, signed, naive, return_nearest):
        # 0-stride broadcast views (zero-copy); the stride-aware binding
        # reads them directly. Outputs are contiguous real buffers.
        batch = torch.broadcast_shapes(
            loc.shape[:-1], vertices.shape[:-2], faces.shape[:-2]
        )
        loc_b = loc.broadcast_to((*batch, loc.shape[-1]))
        vert_b = vertices.broadcast_to((*batch, *vertices.shape[-2:]))
        faces_b = faces.broadcast_to((*batch, *faces.shape[-2:]))
        dist = loc.new_empty(batch)
        nearest = None
        if return_nearest:
            nearest = torch.empty(batch, dtype=torch.int64, device=loc.device)
        _fb.dt_mesh(
            dist,
            nearest,
            loc_b,
            vert_b,
            faces_b,
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
        Mesh vertices, shape ``(*B, V, D)`` (same float dtype as ``loc``).
    faces : torch.Tensor
        Triangle vertex indices, shape ``(*B, F, D)`` (integer tensor; cast
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
    RuntimeError
        If ``.backward()`` is called through ``dist``.

    Notes
    -----
    The batch (leading) dims of ``loc`` (core ``(D,)``), ``vertices`` (core
    ``(V, D)``) and ``faces`` (core ``(F, D)``) are broadcast together via
    ``Tensor.broadcast_to`` (0-stride views, no copy); ``dist`` (and
    ``nearest_vertex``) are allocated with the broadcast batch shape.
    """
    check_dtype(loc, vertices)
    # faces holds integer vertex indices: the binding reads them at int64
    # width, so an int32 (or other) faces array would be misread. Normalize to
    # int64 before the binding (mirrors the numpy wrapper).
    if faces.dtype != torch.int64:
        faces = faces.to(torch.int64)
    return _DtMesh.apply(loc, vertices, faces, signed, naive, return_nearest)
