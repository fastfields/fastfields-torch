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

from ._sym import sym_matvec, sym_solve
from ._util import check_dtype, stream_ptr

__all__ = [
    "field_matvec",
    "field_diag",
    "flow_matvec",
    "flow_matvec_add",
    "flow_matvec_add_",
    "flow_matvec_sub",
    "flow_matvec_sub_",
    "flow_diag",
    "flow_diag_add",
    "flow_diag_add_",
    "flow_diag_sub",
    "flow_diag_sub_",
    "flow_kernel",
    "flow_relax",
    "flow_precond",
    "flow_forward",
]


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
        ctx, inp, voxel_size, absolute, membrane, bending, shears, div,
        bound, ndim
    ):
        out = inp.new_zeros(inp.shape)
        _fb.flow_matvec(
            out,
            inp,
            voxel_size=voxel_size,
            absolute=absolute,
            membrane=membrane,
            bending=bending,
            shears=shears,
            div=div,
            bound=bound,
            ndim=ndim,
            stream=stream_ptr(out),
        )
        ctx.args = (voxel_size, absolute, membrane, bending, shears, div,
                    bound, ndim)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # The flow regulariser operator is self-adjoint, so the adjoint of the
        # forward matvec is the same matvec applied to grad_out.
        ginp = None
        if ctx.needs_input_grad[0]:
            vs, ab, mem, ben, sh, dv, bnd, ndim = ctx.args
            ginp = grad_out.new_zeros(grad_out.shape)
            _fb.flow_matvec(
                ginp,
                grad_out.contiguous(),
                voxel_size=vs,
                absolute=ab,
                membrane=mem,
                bending=ben,
                shears=sh,
                div=dv,
                bound=bnd,
                ndim=ndim,
                stream=stream_ptr(ginp),
            )
        return ginp, None, None, None, None, None, None, None, None


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
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Apply the flow regulariser (scalar penalties; diff'able wrt inp).

    ``shears`` (Lamé mu) and ``div`` (Lamé lambda) add the linear-elastic
    penalty coupling the flow channels.
    """
    check_dtype(inp)
    return _FlowMatvec.apply(
        inp,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
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
    shears: float = 0.0,
    div: float = 0.0,
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
        shears=float(shears),
        div=float(div),
        bound=as_bound(bound),
        ndim=ndim,
        stream=stream_ptr(out),
    )
    return out


def flow_kernel(
    ndim: int,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    dtype: torch.dtype = torch.float64,
    device=None,
) -> Tensor:
    """Materialise the flow regulariser's Toeplitz convolution stencil.

    Returns the small centred kernel that, convolved with a flow field,
    reproduces :func:`flow_matvec`. The shape is ``(*k, ndim)`` for the
    per-channel vector stencil, or ``(*k, ndim, ndim)`` when ``shears``/``div``
    select the cross-channel (Lamé) matrix stencil, where ``k`` is the stencil
    width per spatial dim: 1 (absolute only), 3 (membrane/Lamé) or 5 (bending).
    Not differentiable (it builds from a shape descriptor, not an input field).
    """
    ndim = int(ndim)
    is_matrix = shears != 0.0 or div != 0.0
    if shears == div == membrane == bending == 0.0:
        width = 1
    elif bending == 0.0:
        width = 3
    else:
        width = 5
    shape = [width] * ndim + [ndim]
    if is_matrix:
        shape += [ndim]
    out = torch.zeros(tuple(shape), dtype=dtype, device=device)
    _fb.flow_kernel(
        out,
        voxel_size=_voxel(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=as_bound(bound),
        ndim=ndim,
        stream=stream_ptr(out),
    )
    return out


def flow_relax(
    flow: Tensor,
    hes: Tensor,
    grd: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
    nb_iter: int = 1,
) -> Tensor:
    """Refine ``flow`` in place with ``nb_iter`` relaxation sweeps.

    Solves ``(H + L) x = g`` with per-voxel symmetric Hessian ``hes`` and
    gradient ``grd``. Not differentiable (an in-place iterative solver);
    ``flow`` is the warm start, mutated and returned.
    """
    check_dtype(flow)
    _fb.flow_relax(
        flow,
        hes.contiguous(),
        grd.contiguous(),
        voxel_size=_voxel(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=as_bound(bound),
        ndim=ndim,
        nb_iter=int(nb_iter),
        stream=stream_ptr(flow),
    )
    return flow


def flow_precond(
    mat: Tensor,
    vec: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Apply the preconditioner ``(M + diag(R)) \\ vec``.

    ``M`` is the per-voxel compact-symmetric matrix ``mat``; ``diag(R)`` is the
    diagonal of the flow regulariser (same penalties as :func:`flow_matvec`).
    A composition of :func:`flow_diag` and :func:`sym_solve` — no new kernel;
    differentiable wrt ``vec`` (through the self-adjoint solve).
    """
    check_dtype(vec)
    diag = flow_diag(
        vec.shape, absolute, membrane, bending, shears, div,
        voxel_size=voxel_size, bound=bound, ndim=ndim,
        dtype=vec.dtype, device=vec.device,
    )
    return sym_solve(mat, vec, diag)


def flow_forward(
    mat: Tensor,
    vec: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Apply the forward matrix-vector product ``(M + R) @ vec``.

    ``M`` is the per-voxel compact-symmetric matrix ``mat`` and ``R`` the flow
    regulariser operator. A composition of :func:`sym_matvec` and
    :func:`flow_matvec` — no new kernel; differentiable wrt ``mat``/``vec``.
    """
    check_dtype(vec)
    out = sym_matvec(mat, vec)
    out = out + flow_matvec(
        vec, absolute, membrane, bending, shears, div,
        voxel_size=voxel_size, bound=bound, ndim=ndim,
    )
    return out


# --- accumulate variants -------------------------------------------------
#
# jitfields' ``_add`` / ``_sub`` (fresh array) and trailing-underscore in-place
# forms, as thin compositions ``inp ± op(...)``. The fresh forms compose the
# autograd flow_matvec, so they stay differentiable; the in-place forms use
# augmented assignment (``add_``/``sub_``), intended for non-autograd use.


def flow_matvec_add(
    inp: Tensor,
    flow: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Return ``inp + L @ flow`` (fresh); ``L`` is the flow regulariser."""
    return inp + flow_matvec(
        flow, absolute, membrane, bending, shears, div,
        voxel_size=voxel_size, bound=bound, ndim=ndim,
    )


def flow_matvec_sub(
    inp: Tensor,
    flow: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Return ``inp - L @ flow`` (fresh)."""
    return inp - flow_matvec(
        flow, absolute, membrane, bending, shears, div,
        voxel_size=voxel_size, bound=bound, ndim=ndim,
    )


def flow_matvec_add_(
    inp: Tensor,
    flow: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """In place ``inp += L @ flow``; returns ``inp``."""
    inp += flow_matvec(
        flow, absolute, membrane, bending, shears, div,
        voxel_size=voxel_size, bound=bound, ndim=ndim,
    )
    return inp


def flow_matvec_sub_(
    inp: Tensor,
    flow: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """In place ``inp -= L @ flow``; returns ``inp``."""
    inp -= flow_matvec(
        flow, absolute, membrane, bending, shears, div,
        voxel_size=voxel_size, bound=bound, ndim=ndim,
    )
    return inp


def flow_diag_add(
    inp: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Return ``inp + diag(L)`` (fresh), shaped like ``inp``."""
    return inp + flow_diag(
        inp.shape, absolute, membrane, bending, shears, div,
        voxel_size=voxel_size, bound=bound, ndim=ndim,
        dtype=inp.dtype, device=inp.device,
    )


def flow_diag_sub(
    inp: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """Return ``inp - diag(L)`` (fresh), shaped like ``inp``."""
    return inp - flow_diag(
        inp.shape, absolute, membrane, bending, shears, div,
        voxel_size=voxel_size, bound=bound, ndim=ndim,
        dtype=inp.dtype, device=inp.device,
    )


def flow_diag_add_(
    inp: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """In place ``inp += diag(L)``; returns ``inp``."""
    inp += flow_diag(
        inp.shape, absolute, membrane, bending, shears, div,
        voxel_size=voxel_size, bound=bound, ndim=ndim,
        dtype=inp.dtype, device=inp.device,
    )
    return inp


def flow_diag_sub_(
    inp: Tensor,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Tensor:
    """In place ``inp -= diag(L)``; returns ``inp``."""
    inp -= flow_diag(
        inp.shape, absolute, membrane, bending, shears, div,
        voxel_size=voxel_size, bound=bound, ndim=ndim,
        dtype=inp.dtype, device=inp.device,
    )
    return inp
