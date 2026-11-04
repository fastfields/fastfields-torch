"""fastfields.torch: an autograd-enabled torch interface over the bindings.

A user-friendly, autograd-enabled torch interface over the
``fastfields.dlpack`` bindings.

Public API
----------
Differentiable (``torch.autograd.Function``-backed):

- :func:`sym_matvec` -- ``mat @ vec`` (grad wrt ``mat`` and ``vec``)
- :func:`sym_solve`  -- ``(mat + diag(weight)) \\ vec`` (grad wrt ``vec``)
- :func:`sym_solve_` -- in-place :func:`sym_solve` (grad wrt the mutated
  right-hand side; safe under autograd, see ``API_CONTRACT.md``)
- :func:`resample`   -- spline prolongation (backward: ``restriction``)
- :func:`restriction`-- adjoint of resample (backward: ``resample``)
- :func:`spline_coeff` -- interpolating-coefficient prefilter
- :func:`spline_coeff_` -- in-place :func:`spline_coeff` (self-adjoint, safe
  under autograd)

- :func:`pull`, :func:`push` -- spline gather/scatter (grad wrt the field
  *and* the sampling ``grid``)
- :func:`count` -- splat ones (grad wrt ``grid``)
- :func:`grad` -- sample spatial gradients (grad wrt the field and ``grid``;
  this is the second-order term a learned deformation needs)
- the regulariser operators: :func:`field_matvec` / :func:`flow_matvec`,
  their ``_add``/``_sub`` accumulate forms over the operator, the diagonal
  and the stencil, :func:`field_forward` / :func:`field_precond` and their
  flow twins

Not differentiable, but still exposed for parity with
``fastfields.numpy``/``fastfields.cupy``: forward always runs (even when an
input requires grad), and only calling ``.backward()`` through the output
raises a clear ``RuntimeError`` naming the op -- see each function's
docstring, and ``API_CONTRACT.md`` ("In-place policy") for why this differs
from earlier revisions that omitted these ops or rejected grad-requiring
inputs at call time:

- :func:`dt_euclidean`, :func:`dt_euclidean_`
- :func:`dt_l1`, :func:`dt_l1_`
- :func:`dt_mesh` (no in-place form on any backend)
- :func:`dt_spline_table`, :func:`dt_spline_brent`,
  :func:`dt_spline_gaussnewton` -- the value at an argmin found by an inner
  optimisation, which the library does not differentiate
- :func:`sym_invert`, :func:`sym_invert_`
- :func:`sym_addmatvec_`, :func:`sym_submatvec_` -- differentiable wrt the
  accumulator, but not wrt ``mat``/``vec``; the out-of-place
  ``out0 + sym_matvec(mat, vec)`` is differentiable wrt all three

Re-exported enums: :class:`Spline`, :class:`Bound`.

What this wrapper deliberately does not expose
----------------------------------------------
Of the 58 public operations of the C++ hub, this package wraps all but
nine. ``tools/api_inventory.py`` in ``fastfields-dlpack`` prints the full
matrix; the omissions are:

``pull_backward``, ``push_backward``, ``count_backward``, ``grad_backward``,
``sym_matvec_backward``
    Used, not re-exported. These are the adjoints behind the
    ``autograd.Function`` of the corresponding forward op, which is the
    supported way to reach them: call :func:`pull` / :func:`push` /
    :func:`count` / :func:`grad` / :func:`sym_matvec` and
    ``.backward()``. Re-exporting them as functions would mean a second,
    unchecked spelling of a gradient computation, with five pre-zeroed
    buffers and an accumulate-vs-overwrite contract for the caller to get
    right. ``fastfields.dlpack`` has them raw for anyone who wants that.

``field_precond_``, ``flow_precond_``, ``field_cg``
    In-place preconditioner solves and the CG driver. The in-place forms
    would have to resolve against autograd's rules for a mutated buffer
    (``API_CONTRACT.md``, "In-place policy") and ``field_cg`` runs an
    iterative solver whose gradient is a solve with the same operator --
    both worth doing, neither a mechanical wrapper, and both separate from
    exposing the ops. :func:`field_relax` is the differentiable-free
    in-place solver available today; ``fastfields.numpy`` has all three.

The implementation is split by category into :mod:`._util` (shared helpers
and CUDA-stream resolution), :mod:`._dt` (distance transforms), :mod:`._sym`
(compact-symmetric linear algebra), :mod:`._resample` (spline coefficients
and resampling/restriction), :mod:`._pushpull` (spline gather/scatter) and
:mod:`._reg` (spatial regularisers).
"""

from __future__ import annotations

from fastfields.helpers import Bound, Spline

from ._dt import (
    dt_euclidean,
    dt_euclidean_,
    dt_l1,
    dt_l1_,
    dt_mesh,
    dt_spline_brent,
    dt_spline_gaussnewton,
    dt_spline_table,
)
from ._pushpull import count, grad, pull, push
from ._reg import (
    field_adddiag,
    field_adddiag_,
    field_addkernel,
    field_addkernel_,
    field_addmatvec,
    field_addmatvec_,
    field_diag,
    field_diag_rls,
    field_forward,
    field_kernel,
    field_matvec,
    field_matvec_rls,
    field_precond,
    field_relax,
    field_relax_rls,
    field_subdiag,
    field_subdiag_,
    field_subkernel,
    field_subkernel_,
    field_submatvec,
    field_submatvec_,
    flow_adddiag,
    flow_adddiag_,
    flow_addkernel,
    flow_addkernel_,
    flow_addmatvec,
    flow_addmatvec_,
    flow_diag,
    flow_diag_rls,
    flow_forward,
    flow_kernel,
    flow_matvec,
    flow_matvec_rls,
    flow_precond,
    flow_relax,
    flow_relax_rls,
    flow_subdiag,
    flow_subdiag_,
    flow_subkernel,
    flow_subkernel_,
    flow_submatvec,
    flow_submatvec_,
)
from ._resample import resample, restriction, spline_coeff, spline_coeff_
from ._sym import (
    sym_addmatvec_,
    sym_invert,
    sym_invert_,
    sym_matvec,
    sym_solve,
    sym_solve_,
    sym_submatvec_,
)

__all__ = [
    "sym_matvec",
    "sym_addmatvec_",
    "sym_submatvec_",
    "sym_solve",
    "sym_solve_",
    "sym_invert",
    "sym_invert_",
    "resample",
    "restriction",
    "spline_coeff",
    "spline_coeff_",
    "dt_euclidean",
    "dt_euclidean_",
    "dt_l1",
    "dt_l1_",
    "dt_mesh",
    "dt_spline_table",
    "dt_spline_brent",
    "dt_spline_gaussnewton",
    "pull",
    "push",
    "count",
    "grad",
    "field_matvec",
    "field_addmatvec",
    "field_addmatvec_",
    "field_submatvec",
    "field_submatvec_",
    "field_diag",
    "field_adddiag",
    "field_adddiag_",
    "field_subdiag",
    "field_subdiag_",
    "field_kernel",
    "field_addkernel",
    "field_addkernel_",
    "field_subkernel",
    "field_subkernel_",
    "field_relax",
    "field_matvec_rls",
    "field_diag_rls",
    "field_relax_rls",
    "field_precond",
    "field_forward",
    "flow_matvec",
    "flow_addmatvec",
    "flow_addmatvec_",
    "flow_submatvec",
    "flow_submatvec_",
    "flow_diag",
    "flow_adddiag",
    "flow_adddiag_",
    "flow_subdiag",
    "flow_subdiag_",
    "flow_kernel",
    "flow_addkernel",
    "flow_addkernel_",
    "flow_subkernel",
    "flow_subkernel_",
    "flow_relax",
    "flow_matvec_rls",
    "flow_diag_rls",
    "flow_relax_rls",
    "flow_precond",
    "flow_forward",
    "Spline",
    "Bound",
]
