"""Linear algebra on batches of compact-symmetric matrices, with autograd.

A compact-symmetric matrix of channel size ``C`` is stored along the last axis
as ``C*(C+1)//2`` values: the diagonal first, then the rows of the upper
triangle, e.g. for ``C == 3``::

    [ a d e ]
    [ . b f ]  =>  [a b c d e f]
    [ . . c ]

These wrappers mirror the autograd structure of ``jitfields/jitfields/sym.py``
(classes ``MatVec`` and ``Solve``).
"""

from __future__ import annotations

from typing import Optional

import fastfields.dlpack as _fb

import torch
from torch import Tensor

from ._util import check_dtype, stream_ptr

__all__ = ["sym_matvec", "sym_solve", "sym_invert"]


def sym_matvec(mat: Tensor, vec: Tensor) -> Tensor:
    """Matrix-vector product ``out = mat @ vec`` for compact-symmetric ``mat``.

    Differentiable with respect to both ``mat`` and ``vec``. The batch
    (leading) dims of ``mat`` and ``vec`` are broadcast together; the broadcast
    uses
    ``Tensor.expand`` (0-stride views, no copy) which the stride-aware binding
    consumes directly, and autograd reduces the broadcast gradients back to the
    original operand shapes.

    Parameters
    ----------
    mat : `(..., C*(C+1)//2) tensor`
        Compact-symmetric matrix (diagonal, then upper-triangle rows).
    vec : `(..., C) tensor`
        Vector.

    Returns
    -------
    out : `(..., C) tensor`
        Matrix-vector product (broadcast batch shape + ``(C,)``).
    """
    check_dtype(mat, vec)
    batch = torch.broadcast_shapes(mat.shape[:-1], vec.shape[:-1])
    mat = mat.expand(*batch, mat.shape[-1])
    vec = vec.expand(*batch, vec.shape[-1])
    return _MatVec.apply(mat, vec)


def sym_solve(
    mat: Tensor, vec: Tensor, weight: Optional[Tensor] = None
) -> Tensor:
    """Solve the symmetric system ``out = (mat + diag(weight)) \\ vec``.

    Differentiable with respect to ``vec`` (and ``weight`` is treated as a
    constant). As in ``jitfields``, the solve does **not** backpropagate
    through ``mat``: pass a matrix that does not require grad (detach it).

    Parameters
    ----------
    mat : `(..., C*(C+1)//2) tensor`
        Compact-symmetric matrix.
    vec : `(..., C) tensor`
        Right-hand side.
    weight : `(..., C) tensor`, optional
        Diagonal regularizer added to ``mat`` before solving.

    Returns
    -------
    out : `(..., C) tensor`
        Solution of the linear system.
    """
    check_dtype(mat, vec)
    if weight is not None:
        check_dtype(weight)
    shapes = [mat.shape[:-1], vec.shape[:-1]]
    if weight is not None:
        shapes.append(weight.shape[:-1])
    batch = torch.broadcast_shapes(*shapes)
    mat = mat.expand(*batch, mat.shape[-1])
    vec = vec.expand(*batch, vec.shape[-1])
    if weight is not None:
        weight = weight.expand(*batch, weight.shape[-1])
    return _Solve.apply(mat, vec, weight)


def sym_invert(mat: Tensor) -> Tensor:
    """Invert a compact-symmetric matrix (``out = inv(mat)``).

    Not differentiable: raises if ``mat`` requires grad (mirrors jitfields).
    """
    check_dtype(mat)
    if mat.requires_grad:
        raise ValueError(
            "sym_invert does not backpropagate gradients through the matrix. "
            "Use `mat.detach()`."
        )
    # mat is a read-only input: the stride-aware binding reads it zero-copy, so
    # we do NOT force contiguity. The output must be a real contiguous buffer
    # (new_empty is contiguous even when mat is strided).
    out = mat.new_empty(mat.shape)
    _fb.sym_invert(out, mat, stream=stream_ptr(mat))
    return out


class _MatVec(torch.autograd.Function):
    """Autograd for ``out = mat @ vec`` (mirrors jitfields ``MatVec``)."""

    @staticmethod
    def forward(ctx, mat, vec):
        # mat/vec may be 0-stride broadcast views (from Tensor.expand); the
        # stride-aware binding consumes them zero-copy, so we do NOT force
        # contiguity. Only the output must be a real contiguous buffer.
        out = vec.new_empty(vec.shape)
        _fb.sym_matvec(out, mat, vec, stream=stream_ptr(out))
        ctx.save_for_backward(mat, vec)
        return out

    @staticmethod
    def backward(ctx, grad):
        mat, vec = ctx.saved_tensors
        gmat = gvec = None
        # grad_vec = mat @ grad  (mat is symmetric, hence self-adjoint)
        if ctx.needs_input_grad[1]:
            gvec = grad.new_empty(vec.shape)
            _fb.sym_matvec(gvec, mat, grad, stream=stream_ptr(gvec))
        # grad_mat = outer-product contribution, via the dedicated backward op.
        # gmat carries mat's (possibly expanded) shape; autograd's expand
        # backward then sums it down to the caller's original mat shape.
        if ctx.needs_input_grad[0]:
            gmat = grad.new_empty(mat.shape)
            _fb.sym_matvec_backward(gmat, grad, vec, stream=stream_ptr(gmat))
        return gmat, gvec


class _Solve(torch.autograd.Function):
    """Autograd for ``out = (mat + diag(weight)) \\ vec``.

    Mirrors jitfields ``Solve``: differentiable through ``vec`` only. Since the
    system matrix ``M`` is symmetric, ``M^{-1}`` is self-adjoint, so the
    gradient w.r.t. ``vec`` is another solve with the same matrix/weight.
    """

    @staticmethod
    def forward(ctx, mat, vec, weight):
        if mat.requires_grad:
            raise ValueError(
                "sym_solve does not backpropagate gradients through the "
                "matrix. Use `mat.detach()`."
            )
        # mat/vec/weight may be 0-stride broadcast views (Tensor.expand); the
        # stride-aware binding handles them zero-copy. Output is contiguous.
        out = vec.new_empty(vec.shape)
        s = stream_ptr(out)
        if weight is None:
            _fb.sym_solve(out, mat, vec, stream=s)
        else:
            _fb.sym_solve(out, mat, vec, weight, stream=s)
        ctx.save_for_backward(mat, weight)
        return out

    @staticmethod
    def backward(ctx, grad):
        mat, weight = ctx.saved_tensors
        gvec = None
        if ctx.needs_input_grad[1]:
            gvec = grad.new_empty(grad.shape)
            s = stream_ptr(gvec)
            if weight is None:
                _fb.sym_solve(gvec, mat, grad, stream=s)
            else:
                _fb.sym_solve(gvec, mat, grad, weight, stream=s)
        return None, gvec, None
