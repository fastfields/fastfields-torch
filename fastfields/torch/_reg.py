"""Spatial regularisers — torch.

* **field** — multi-channel field ``(*batch, *spatial, C)``; per-channel
  ``absolute`` / ``membrane`` / ``bending`` (a scalar broadcasts to ``C``).
* **flow** — vector flow field; scalar penalties.

The ``*_matvec`` operators are symmetric, so they are differentiable wrt the
input field via the same (self-adjoint) operator. The ``*_diag``
preconditioners have no differentiable input.
"""

from __future__ import annotations

from typing import Optional, Sequence

import fastfields.dlpack as _fb
from fastfields.dlpack import as_bound

import torch
from torch import Tensor

from ._util import check_dtype, stream_ptr

__all__ = ["field_matvec", "field_diag", "flow_matvec", "flow_diag"]


def _per_channel(value, channels: int, name: str) -> Optional[list]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [float(value)] * channels
    out = [float(v) for v in value]
    if len(out) != channels:
        raise ValueError(
            f"{name} must be a scalar or a length-C={channels} sequence"
        )
    return out


def _voxel(value, ndim: int) -> Optional[list]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [float(value)] * ndim
    out = [float(v) for v in value]
    if len(out) != ndim:
        raise ValueError(
            f"voxel_size must be a scalar or a length-ndim={ndim} sequence"
        )
    return out


class _FieldMatvec(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, inp, voxel_size, absolute, membrane, bending, bound, ndim
    ):
        out = inp.new_zeros(inp.shape)
        _fb.field_matvec(
            out,
            inp,
            voxel_size=voxel_size,
            absolute=absolute,
            membrane=membrane,
            bending=bending,
            bound=bound,
            ndim=ndim,
            stream=stream_ptr(out),
        )
        ctx.args = (voxel_size, absolute, membrane, bending, bound, ndim)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ginp = None
        if ctx.needs_input_grad[0]:
            vs, ab, mem, ben, bnd, ndim = ctx.args
            ginp = grad_out.new_zeros(grad_out.shape)
            _fb.field_matvec(
                ginp,
                grad_out.contiguous(),
                voxel_size=vs,
                absolute=ab,
                membrane=mem,
                bending=ben,
                bound=bnd,
                ndim=ndim,
                stream=stream_ptr(ginp),
            )
        return ginp, None, None, None, None, None, None


class _FlowMatvec(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, inp, voxel_size, absolute, membrane, bending, bound, ndim
    ):
        out = inp.new_zeros(inp.shape)
        _fb.flow_matvec(
            out,
            inp,
            voxel_size=voxel_size,
            absolute=absolute,
            membrane=membrane,
            bending=bending,
            bound=bound,
            ndim=ndim,
            stream=stream_ptr(out),
        )
        ctx.args = (voxel_size, absolute, membrane, bending, bound, ndim)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ginp = None
        if ctx.needs_input_grad[0]:
            vs, ab, mem, ben, bnd, ndim = ctx.args
            ginp = grad_out.new_zeros(grad_out.shape)
            _fb.flow_matvec(
                ginp,
                grad_out.contiguous(),
                voxel_size=vs,
                absolute=ab,
                membrane=mem,
                bending=ben,
                bound=bnd,
                ndim=ndim,
                stream=stream_ptr(ginp),
            )
        return ginp, None, None, None, None, None, None


def field_matvec(
    inp: Tensor,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Apply the field regulariser (differentiable wrt ``inp``)."""
    check_dtype(inp)
    channels = inp.shape[-1]
    return _FieldMatvec.apply(
        inp,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
    )


def flow_matvec(
    inp: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Apply the flow regulariser (scalar penalties; diff'able wrt inp)."""
    check_dtype(inp)
    return _FlowMatvec.apply(
        inp,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        as_bound(bound),
        ndim,
    )


def field_diag(
    shape: Sequence[int],
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> Tensor:
    """Diagonal (preconditioner) of the field regulariser, shaped ``shape``."""
    out = torch.zeros(tuple(int(s) for s in shape), dtype=dtype, device=device)
    channels = out.shape[-1]
    _fb.field_diag(
        out,
        voxel_size=_voxel(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=as_bound(bound),
        ndim=ndim,
        stream=stream_ptr(out),
    )
    return out


def flow_diag(
    shape: Sequence[int],
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> Tensor:
    """Diagonal (preconditioner) of the flow regulariser, shaped ``shape``."""
    out = torch.zeros(tuple(int(s) for s in shape), dtype=dtype, device=device)
    _fb.flow_diag(
        out,
        voxel_size=_voxel(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        bound=as_bound(bound),
        ndim=ndim,
        stream=stream_ptr(out),
    )
    return out
