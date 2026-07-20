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

import torch
from torch import Tensor

import fastfields_bind as _fb

from ._utils import as_contiguous, check_dtype

__all__ = ["sym_matvec", "sym_solve", "sym_invert"]


def sym_matvec(mat: Tensor, vec: Tensor) -> Tensor:
    """Matrix-vector product ``out = mat @ vec`` for compact-symmetric ``mat``.

    Differentiable with respect to both ``mat`` and ``vec``.

    Parameters
    ----------
    mat : `(..., C*(C+1)//2) tensor`
        Compact-symmetric matrix (diagonal, then upper-triangle rows).
    vec : `(..., C) tensor`
        Vector.

    Returns
    -------
    out : `(..., C) tensor`
        Matrix-vector product.
    """
    check_dtype(mat, vec)
    return _MatVec.apply(mat, vec)


def sym_solve(mat: Tensor, vec: Tensor, weight: Optional[Tensor] = None) -> Tensor:
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
    mat = as_contiguous(mat)
    out = torch.empty_like(mat)
    _fb.sym_invert(out, mat)
    return out


class _MatVec(torch.autograd.Function):
    """Autograd for ``out = mat @ vec`` (mirrors jitfields ``MatVec``)."""

    @staticmethod
    def forward(ctx, mat, vec):
        mat = as_contiguous(mat)
        vec = as_contiguous(vec)
        out = torch.empty_like(vec)
        _fb.sym_matvec(out, mat, vec)
        ctx.save_for_backward(mat, vec)
        return out

    @staticmethod
    def backward(ctx, grad):
        mat, vec = ctx.saved_tensors
        grad = as_contiguous(grad)
        gmat = gvec = None
        # grad_vec = mat @ grad  (mat is symmetric, hence self-adjoint)
        if ctx.needs_input_grad[1]:
            gvec = torch.empty_like(vec)
            _fb.sym_matvec(gvec, mat, grad)
        # grad_mat = outer-product contribution, via the dedicated backward op
        if ctx.needs_input_grad[0]:
            gmat = torch.empty_like(mat)
            _fb.sym_matvec_backward(gmat, grad, vec)
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
        mat = as_contiguous(mat)
        vec = as_contiguous(vec)
        weight_c = None if weight is None else as_contiguous(weight)
        out = torch.empty_like(vec)
        if weight_c is None:
            _fb.sym_solve(out, mat, vec)
        else:
            _fb.sym_solve(out, mat, vec, weight_c)
        ctx.save_for_backward(mat, weight_c)
        return out

    @staticmethod
    def backward(ctx, grad):
        mat, weight = ctx.saved_tensors
        gvec = None
        if ctx.needs_input_grad[1]:
            grad = as_contiguous(grad)
            gvec = torch.empty_like(grad)
            if weight is None:
                _fb.sym_solve(gvec, mat, grad)
            else:
                _fb.sym_solve(gvec, mat, grad, weight)
        return None, gvec, None
