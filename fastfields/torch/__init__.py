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
- :func:`sym_invert`, :func:`sym_invert_`

Re-exported enums: :class:`Spline`, :class:`Bound`.

The implementation is split by category into :mod:`._util` (shared helpers
and CUDA-stream resolution), :mod:`._dt` (distance transforms), :mod:`._sym`
(compact-symmetric linear algebra) and :mod:`._resample` (spline coefficients
and resampling/restriction).
"""

from __future__ import annotations

from fastfields.dlpack import Bound, Spline

from ._dt import dt_euclidean, dt_euclidean_, dt_l1, dt_l1_, dt_mesh
from ._pushpull import count, grad, pull, push
from ._reg import (
    field_adddiag,
    field_adddiag_,
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
    field_submatvec,
    field_submatvec_,
    flow_adddiag,
    flow_adddiag_,
    flow_addmatvec,
    flow_addmatvec_,
    flow_diag,
    flow_forward,
    flow_kernel,
    flow_matvec,
    flow_precond,
    flow_relax,
    flow_subdiag,
    flow_subdiag_,
    flow_submatvec,
    flow_submatvec_,
)
from ._resample import resample, restriction, spline_coeff, spline_coeff_
from ._sym import sym_invert, sym_invert_, sym_matvec, sym_solve, sym_solve_

__all__ = [
    "sym_matvec",
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
    "flow_relax",
    "flow_precond",
    "flow_forward",
    "Spline",
    "Bound",
]
