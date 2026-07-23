"""Distance-transform wrappers (torch).

These operators are **not differentiable**; the wrappers raise if an input
requires grad (mirroring how ``jitfields`` guards its non-differentiable ops).
The underlying bindings mutate their input in place, so these wrappers operate
on a fresh copy and return a new tensor. On CUDA tensors the current stream is
forwarded to the binding (see :mod:`fastfields.torch._util`).
"""

from __future__ import annotations

from typing import Tuple, Union

import fastfields.dlpack as _fb

import torch
from torch import Tensor

from ._util import check_dtype, stream_ptr

__all__ = ["dt_euclidean", "dt_l1", "dt_mesh"]


def _reject_grad(name: str, *tensors: Tensor | None) -> None:
    """Raise if any tensor requires grad (these ops are not differentiable).

    Parameters
    ----------
    name : str
        Operation name used in the error message.
    *tensors : torch.Tensor or None
        Tensors to check.

    Raises
    ------
    ValueError
        If any tensor requires grad.
    """
    for t in tensors:
        if t is not None and torch.is_tensor(t) and t.requires_grad:
            raise ValueError(
                f"{name} is not differentiable; call `.detach()` on inputs "
                f"that require grad."
            )


def dt_euclidean(inp: Tensor, voxel_spacing: float = 1.0) -> Tensor:
    """Euclidean distance transform along the last axis (returns a new tensor).

    Parameters
    ----------
    inp : torch.Tensor
        Input holding ``0`` at feature locations and ``+inf`` elsewhere. Must
        not require grad (this op is not differentiable).
    voxel_spacing : float, default=1.0
        Physical spacing between samples along the last axis.

    Returns
    -------
    torch.Tensor
        The distance transform (a new tensor with ``inp``'s layout).

    Raises
    ------
    TypeError
        If ``inp`` is not float32/float64.
    ValueError
        If ``inp`` requires grad.
    """
    check_dtype(inp)
    _reject_grad("dt_euclidean", inp)
    out = inp.clone()
    _fb.dt_euclidean(out, voxel_spacing, stream=stream_ptr(out))
    return out


def dt_l1(inp: Tensor, voxel_spacing: float = 1.0) -> Tensor:
    """L1 distance transform along the last axis (returns a new tensor).

    Parameters
    ----------
    inp : torch.Tensor
        Input holding ``0`` at feature locations and ``+inf`` elsewhere. Must
        not require grad (this op is not differentiable).
    voxel_spacing : float, default=1.0
        Physical spacing between samples along the last axis.

    Returns
    -------
    torch.Tensor
        The distance transform (a new tensor with ``inp``'s layout).

    Raises
    ------
    TypeError
        If ``inp`` is not float32/float64.
    ValueError
        If ``inp`` requires grad.
    """
    check_dtype(inp)
    _reject_grad("dt_l1", inp)
    out = inp.clone()
    _fb.dt_l1(out, voxel_spacing, stream=stream_ptr(out))
    return out


def dt_mesh(
    loc: Tensor,
    vertices: Tensor,
    faces: Tensor,
    signed: bool = True,
    naive: bool = False,
    return_nearest: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Point-to-triangular-mesh (squared) distance (not differentiable).

    Parameters
    ----------
    loc : torch.Tensor
        Query points, shape ``(*B, D)``.
    vertices : torch.Tensor
        Mesh vertices, shape ``(*B, V, D)`` (same float dtype as ``loc``).
    faces : torch.Tensor
        Triangle vertex indices, shape ``(*B, F, D)`` (integer tensor).
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
        If ``loc`` or ``vertices`` requires grad.

    Notes
    -----
    The batch (leading) dims of ``loc`` (core ``(D,)``), ``vertices`` (core
    ``(V, D)``) and ``faces`` (core ``(F, D)``) are broadcast together via
    ``Tensor.broadcast_to`` (0-stride views, no copy); ``dist`` (and
    ``nearest_vertex``) are allocated with the broadcast batch shape.
    """
    check_dtype(loc, vertices)
    _reject_grad("dt_mesh", loc, vertices)
    batch = torch.broadcast_shapes(
        loc.shape[:-1], vertices.shape[:-2], faces.shape[:-2]
    )
    # 0-stride broadcast views (zero-copy); the stride-aware binding reads them
    # directly. Outputs are contiguous real buffers.
    loc = loc.broadcast_to((*batch, loc.shape[-1]))
    vertices = vertices.broadcast_to((*batch, *vertices.shape[-2:]))
    faces = faces.broadcast_to((*batch, *faces.shape[-2:]))
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
