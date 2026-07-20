"""fastfields.torch: a user-friendly, autograd-enabled torch interface over the
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
"""

from __future__ import annotations

from fastfields.dlpack import Bound, Spline

from .distance import dt_euclidean, dt_l1, dt_mesh
from .resize import resample, restriction
from .splinc import spline_coeff
from .sym import sym_invert, sym_matvec, sym_solve

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
    "Spline",
    "Bound",
]
