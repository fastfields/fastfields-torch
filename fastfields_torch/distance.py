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

import fastfields_bind as _fb

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
    loc : `(..., D) tensor`
        Query points.
    vertices : `(V, D) tensor`
        Mesh vertices (same float dtype as ``loc``).
    faces : `(F, D) int tensor`
        Triangle vertex indices.
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
    loc = as_contiguous(loc)
    vertices = as_contiguous(vertices)
    faces = as_contiguous(faces)
    dist = loc.new_empty(loc.shape[:-1])
    nearest = None
    if return_nearest:
        nearest = torch.empty(loc.shape[:-1], dtype=torch.int64, device=loc.device)
    _fb.dt_mesh(dist, nearest, loc, vertices, faces, signed, naive)
    if return_nearest:
        return dist, nearest
    return dist
