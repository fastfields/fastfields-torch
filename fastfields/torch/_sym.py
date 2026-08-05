"""Linear algebra on batches of compact-symmetric matrices, with autograd.

A compact-symmetric matrix of channel size ``C`` is stored along the last axis
as ``C*(C+1)//2`` values: the diagonal first, then the rows of the upper
triangle, e.g. for ``C == 3``::

    [ a d e ]
    [ . b f ]  =>  [a b c d e f]
    [ . . c ]

These wrappers mirror the autograd structure of ``jitfields/jitfields/sym.py``
(classes ``MatVec`` and ``Solve``).

In-place (``_``-suffixed) forms mirror the numpy/cupy convention
(``sym_solve_(inp_out, mat, weight=None)``, ``sym_invert_(mat)`` -- note the
argument order matches numpy/cupy exactly, which is *not* the same order as
the out-of-place ``sym_solve(mat, vec, weight=None)``). Whether an in-place
form is differentiable follows the per-op rule in ``API_CONTRACT.md``
("In-place policy"):

* :func:`sym_solve_` **is** differentiable (w.r.t. the mutated right-hand
  side): the backward of :func:`sym_solve` never reads the pre-mutation
  right-hand side, only the saved ``mat``/``weight``, so overwriting it in
  place destroys no information backward needs.
* :func:`sym_invert_` is **not** differentiable, for two independent
  reasons: the inverse is nonlinear in ``mat`` (a correct backward would
  need the pre-inversion matrix -- gone once mutated in place) *and* no
  gradient is implemented for :func:`sym_invert` (its out-of-place form) on
  this backend at all, mirroring ``jitfields``. Calling ``.backward()``
  through either form's output raises ``RuntimeError``.
"""

from __future__ import annotations

import math
from typing import Optional

import fastfields.dlpack as _fb

import torch
from torch import Tensor

from ._util import check_dtype, raise_not_differentiable, stream_ptr

__all__ = [
    "sym_matvec",
    "sym_solve",
    "sym_solve_",
    "sym_invert",
    "sym_invert_",
]


def _channels_from_packed(packed_len: int) -> int:
    """Return the channel count ``C`` with ``C*(C+1)//2 == packed_len``.

    Raises
    ------
    ValueError
        If ``packed_len`` is not a triangular number ``C*(C+1)/2``.
    """
    c = int((math.isqrt(8 * packed_len + 1) - 1) // 2)
    if c * (c + 1) // 2 != packed_len:
        raise ValueError(
            f"packed length {packed_len} is not a triangular number "
            "(expected C*(C+1)/2 for some integer C)"
        )
    return c


def _check_sym(mat: Tensor, vec: Tensor) -> int:
    """Validate a packed matrix / vector pair and return the channel count.

    Verifies that ``mat``'s packed trailing dim ``C*(C+1)//2`` matches the
    channel count ``C`` implied by ``vec``'s trailing dim, so a mismatched pair
    cannot reach the raw binding (which would OOB-read / segfault). Batch dims
    are broadcast by the caller, so only the channel relation is enforced here.

    Parameters
    ----------
    mat : `(..., C*(C+1)//2) tensor`
        Packed compact-symmetric matrix.
    vec : `(..., C) tensor`
        Vector.

    Returns
    -------
    int
        The channel count ``C``.

    Raises
    ------
    ValueError
        If ``vec``'s channel count does not match the matrix packing.
    """
    c = _channels_from_packed(mat.shape[-1])
    if vec.shape[-1] != c:
        raise ValueError(
            f"vec has {vec.shape[-1]} channels but the packed matrix "
            f"encodes {c} channels (packed length {mat.shape[-1]})"
        )
    return c


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
    _check_sym(mat, vec)
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
    See :func:`sym_solve_` for the in-place variant.

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
    c = _check_sym(mat, vec)
    if weight is not None and weight.shape[-1] != c:
        raise ValueError(
            f"weight has {weight.shape[-1]} channels but the packed matrix "
            f"encodes {c} channels"
        )
    shapes = [mat.shape[:-1], vec.shape[:-1]]
    if weight is not None:
        shapes.append(weight.shape[:-1])
    batch = torch.broadcast_shapes(*shapes)
    mat = mat.expand(*batch, mat.shape[-1])
    vec = vec.expand(*batch, vec.shape[-1])
    if weight is not None:
        weight = weight.expand(*batch, weight.shape[-1])
    return _Solve.apply(mat, vec, weight)


def sym_solve_(
    inp_out: Tensor, mat: Tensor, weight: Optional[Tensor] = None
) -> Tensor:
    """In-place solve: ``inp_out <- (mat + diag(weight)) \\ inp_out``.

    Note the argument order -- ``inp_out`` first, then ``mat`` -- matches the
    numpy/cupy ``sym_solve_`` convention, not :func:`sym_solve`'s
    ``(mat, vec, ...)`` order.

    Differentiable with respect to ``inp_out`` (mirrors :func:`sym_solve`);
    ``weight`` is a constant and ``mat`` must not require grad (as in
    :func:`sym_solve`). Safe under autograd: the backward of
    :func:`sym_solve` never reads the pre-mutation right-hand side -- only
    the saved ``mat``/``weight`` -- so overwriting ``inp_out`` in place
    destroys no information backward needs (see ``API_CONTRACT.md``,
    "In-place policy"). As with any in-place op, a leaf tensor with
    ``requires_grad=True`` cannot be mutated (torch's ordinary leaf rule).

    Parameters
    ----------
    inp_out : `(..., C) tensor`
        Right-hand side on input, solution on output. Mutated in place and
        returned; fixes the batch (leading) shape.
    mat : `(..., C*(C+1)//2) tensor`
        Compact-symmetric matrix, broadcast to ``inp_out``'s batch shape.
        Must not require grad.
    weight : `(..., C) tensor`, optional
        Diagonal regularizer added to ``mat``, broadcast to ``inp_out``'s
        batch shape.

    Returns
    -------
    out : `(..., C) tensor`
        ``inp_out`` (the same tensor object), now holding the solution.

    Raises
    ------
    ValueError
        If ``mat`` requires grad, or channel counts disagree.
    """
    check_dtype(mat, inp_out)
    if weight is not None:
        check_dtype(weight)
    c = _check_sym(mat, inp_out)
    if weight is not None and weight.shape[-1] != c:
        raise ValueError(
            f"weight has {weight.shape[-1]} channels but the packed matrix "
            f"encodes {c} channels"
        )
    batch = inp_out.shape[:-1]
    mat = mat.expand(*batch, mat.shape[-1])
    if weight is not None:
        weight = weight.expand(*batch, weight.shape[-1])
    return _SolveInPlace.apply(inp_out, mat, weight)


def sym_invert(mat: Tensor) -> Tensor:
    """Invert a compact-symmetric matrix (``out = inv(mat)``).

    Not differentiable -- calling ``.backward()`` through this raises
    ``RuntimeError``. Forward still runs normally (including when ``mat``
    requires grad); only an actual backward call fails. See
    :func:`sym_invert_` for the in-place variant.

    Parameters
    ----------
    mat : `(..., C*(C+1)//2) tensor`
        Compact-symmetric matrix.

    Returns
    -------
    out : `(..., C*(C+1)//2) tensor`
        The packed inverse, same shape as ``mat``.

    Raises
    ------
    RuntimeError
        If ``.backward()`` is called through the output.
    """
    check_dtype(mat)
    return _Invert.apply(mat)


def sym_invert_(mat: Tensor) -> Tensor:
    """In-place invert a compact-symmetric matrix (``mat <- inv(mat)``).

    Not differentiable -- calling ``.backward()`` through this raises
    ``RuntimeError`` (mirrors :func:`sym_invert`; the inverse is nonlinear
    in ``mat``, so a correct backward would additionally need the
    pre-inversion matrix, which an in-place write has already destroyed). As
    with any in-place op, a leaf tensor with ``requires_grad=True`` cannot
    be mutated (torch's ordinary leaf rule).

    Parameters
    ----------
    mat : `(..., C*(C+1)//2) tensor`
        Compact-symmetric matrix. Mutated in place and returned.

    Returns
    -------
    out : `(..., C*(C+1)//2) tensor`
        ``mat`` (the same tensor object), now holding its packed inverse.

    Raises
    ------
    RuntimeError
        If ``.backward()`` is called through the output.
    """
    check_dtype(mat)
    return _InvertInPlace.apply(mat)


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


class _SolveInPlace(torch.autograd.Function):
    """``inp_out <- (mat + diag(weight)) \\ inp_out`` in place.

    Autograd-safe: the backward below is identical to :class:`_Solve`'s and
    never reads the pre-mutation ``inp_out`` -- only the saved ``mat``/
    ``weight`` -- so mutating ``inp_out`` in place loses nothing backward
    needs. ``ctx.mark_dirty(inp_out)`` bumps its version counter so that if
    some other op had saved ``inp_out`` for its own backward, autograd
    raises instead of silently computing a wrong gradient.
    """

    @staticmethod
    def forward(ctx, inp_out, mat, weight):
        if mat.requires_grad:
            raise ValueError(
                "sym_solve_ does not backpropagate gradients through the "
                "matrix. Use `mat.detach()`."
            )
        s = stream_ptr(inp_out)
        _fb.sym_solve_(inp_out, mat, weight, stream=s)
        ctx.mark_dirty(inp_out)
        ctx.save_for_backward(mat, weight)
        return inp_out

    @staticmethod
    def backward(ctx, grad):
        mat, weight = ctx.saved_tensors
        ginp = None
        if ctx.needs_input_grad[0]:
            ginp = grad.new_empty(grad.shape)
            s = stream_ptr(ginp)
            if weight is None:
                _fb.sym_solve(ginp, mat, grad, stream=s)
            else:
                _fb.sym_solve(ginp, mat, grad, weight, stream=s)
        return ginp, None, None


class _Invert(torch.autograd.Function):
    """``out = inv(mat)`` (out-of-place, ``mat`` left untouched)."""

    @staticmethod
    def forward(ctx, mat):
        # mat is a read-only input: the stride-aware binding reads it
        # zero-copy, so we do NOT force contiguity. The output must be a
        # real contiguous buffer (new_empty is contiguous even when mat is
        # strided).
        out = mat.new_empty(mat.shape)
        _fb.sym_invert(out, mat, stream=stream_ptr(mat))
        return out

    @staticmethod
    def backward(ctx, grad):
        raise_not_differentiable(
            "sym_invert",
            "the packed-matrix inverse has no gradient implemented on this "
            "backend (mirrors jitfields); it is also nonlinear in `mat`, so "
            "a correct backward would need the pre-inversion matrix.",
        )


class _InvertInPlace(torch.autograd.Function):
    """``mat <- inv(mat)`` in place."""

    @staticmethod
    def forward(ctx, mat):
        _fb.sym_invert_(mat, stream=stream_ptr(mat))
        ctx.mark_dirty(mat)
        return mat

    @staticmethod
    def backward(ctx, grad):
        raise_not_differentiable(
            "sym_invert_",
            "the packed-matrix inverse is nonlinear in `mat`, so a correct "
            "backward would need the pre-inversion matrix -- already "
            "overwritten by this in-place op -- and no gradient is "
            "implemented for sym_invert on this backend anyway (mirrors "
            "jitfields).",
        )
