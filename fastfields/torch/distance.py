"""Distance-transform wrappers.

These operators are **not differentiable**; the wrappers raise if an input
requires grad (mirroring how ``jitfields`` guards its non-differentiable ops).
The underlying bindings mutate their input in place, so these wrappers operate
on a fresh copy and return a new tensor.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import Tensor

import fastfields.dlpack as _fb

from ._utils import as_contiguous, check_dtype

__all__ = ["dt_euclidean", "dt_l1", "dt_mesh"]


def _reject_grad(name, *tensors):
    for t in tensors:
        if t is not None and torch.is_tensor(t) and t.requires_grad:
            raise ValueError(
                f"{name} is not differentiable; call `.detach()` on inputs "
                f"that require grad."
            )


def dt_euclidean(inp: Tensor, voxel_spacing: float = 1.0) -> Tensor:
    """Euclidean distance transform along the last axis (returns a new tensor).

    Input convention: 0 at features, +inf elsewhere.
    """
    check_dtype(inp)
    _reject_grad("dt_euclidean", inp)
    out = as_contiguous(inp).clone()
    _fb.dt_euclidean(out, voxel_spacing)
    return out


def dt_l1(inp: Tensor, voxel_spacing: float = 1.0) -> Tensor:
    """L1 distance transform along the last axis (returns a new tensor)."""
    check_dtype(inp)
    _reject_grad("dt_l1", inp)
    out = as_contiguous(inp).clone()
    _fb.dt_l1(out, voxel_spacing)
    return out


def dt_mesh(
    loc: Tensor,
    vertices: Tensor,
    faces: Tensor,
    signed: bool = True,
    naive: bool = False,
    return_nearest: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Point-to-triangular-mesh (squared) distance.

    Not differentiable.

    Parameters
    ----------
    loc : `(*B, D) tensor`
        Query points.
    vertices : `(*B, V, D) tensor`
        Mesh vertices (same float dtype as ``loc``).
    faces : `(*B, F, D) int tensor`
        Triangle vertex indices.

    The batch (leading) dims of ``loc`` (core ``(D,)``), ``vertices`` (core
    ``(V, D)``) and ``faces`` (core ``(F, D)``) are broadcast together via
    ``Tensor.broadcast_to`` (0-stride views, no copy); ``dist`` (and
    ``nearest_vertex``) are allocated with the broadcast batch shape.
    signed : `bool`, default=True
        Return signed distances.
    naive : `bool`, default=False
        Use the naive (brute-force) algorithm.
    return_nearest : `bool`, default=False
        Also return the nearest-vertex index per query point.

    Returns
    -------
    dist : `(...) tensor`
        (Squared) distance per query point.
    nearest_vertex : `(...) int tensor`, optional
        Only if ``return_nearest`` is True.
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
    _fb.dt_mesh(dist, nearest, loc, vertices, faces, signed, naive)
    if return_nearest:
        return dist, nearest
    return dist
