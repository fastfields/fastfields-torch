"""fastfields.torch: an autograd-enabled torch interface over the bindings.

A user-friendly, autograd-enabled torch interface over the
``fastfields.dlpack`` bindings.

Public API
----------
Differentiable (``torch.autograd.Function``-backed):

- :func:`sym_matvec` -- ``mat @ vec`` (grad wrt ``mat`` and ``vec``)
- :func:`sym_solve`  -- ``(mat + diag(weight)) \\ vec`` (grad wrt ``vec``)
- :func:`resample`   -- spline prolongation (backward: ``restriction``)
- :func:`restriction`-- adjoint of resample (backward: ``resample``)
- :func:`spline_coeff` -- interpolating-coefficient prefilter

Non-differentiable (raise if an input requires grad):

- :func:`sym_invert`
- :func:`dt_euclidean`, :func:`dt_l1`, :func:`dt_mesh`

Re-exported enums: :class:`Spline`, :class:`Bound`.

The implementation is split by category into :mod:`._util` (shared helpers
and CUDA-stream resolution), :mod:`._dt` (distance transforms), :mod:`._sym`
(compact-symmetric linear algebra) and :mod:`._resample` (spline coefficients
and resampling/restriction).
"""

from __future__ import annotations

from fastfields.dlpack import Bound, Spline

from ._dt import dt_euclidean, dt_l1, dt_mesh
from ._pushpull import count, grad, pull, push
from ._reg import (
    field_diag,
    field_diag_add,
    field_diag_add_,
    field_diag_sub,
    field_diag_sub_,
    field_forward,
    field_kernel,
    field_matvec,
    field_matvec_add,
    field_matvec_add_,
    field_matvec_sub,
    field_matvec_sub_,
    field_precond,
    flow_diag,
    flow_diag_add,
    flow_diag_add_,
    flow_diag_sub,
    flow_diag_sub_,
    flow_forward,
    flow_kernel,
    flow_matvec,
    flow_matvec_add,
    flow_matvec_add_,
    flow_matvec_sub,
    flow_matvec_sub_,
    flow_precond,
    flow_relax,
)
from ._resample import resample, restriction, spline_coeff
from ._sym import sym_invert, sym_matvec, sym_solve

__all__ = [
    "sym_matvec",
    "sym_solve",
    "sym_invert",
    "resample",
    "restriction",
    "spline_coeff",
    "dt_euclidean",
    "dt_l1",
    "dt_mesh",
    "pull",
    "push",
    "count",
    "grad",
    "field_matvec",
    "field_matvec_add",
    "field_matvec_add_",
    "field_matvec_sub",
    "field_matvec_sub_",
    "field_diag",
    "field_diag_add",
    "field_diag_add_",
    "field_diag_sub",
    "field_diag_sub_",
    "field_kernel",
    "field_precond",
    "field_forward",
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
    "Spline",
    "Bound",
]
