# fastfields-torch

A user-friendly, **autograd-enabled** PyTorch interface over the
[`fastfields.dlpack`](../fastfields-bind-py) nanobind bindings.

Functions take and return `torch` tensors (CPU float32/float64), allocate their
own outputs, and route through the bindings' DLPack path so the same code will
work for CUDA tensors once a GPU build is available.

## Public API

| Function | Differentiable | Backward |
| --- | --- | --- |
| `sym_matvec(mat, vec)` | yes (`mat`, `vec`) | `sym_matvec` / `sym_matvec_backward` |
| `sym_solve(mat, vec, weight=None)` | yes (`vec` only) | `sym_solve` (self-adjoint) |
| `resample(inp, shape, ...)` | yes (`inp`) | `restriction` (reciprocal scale) |
| `restriction(inp, shape, ...)` | yes (`inp`) | `resample` (reciprocal scale) |
| `spline_coeff(inp, spline, bound)` | yes (`inp`) | `spline_coeff` (self-adjoint) |
| `sym_invert(mat)` | no (raises if grad) | — |
| `dt_euclidean`, `dt_l1`, `dt_mesh` | no (raise if grad) | — |

Matrices use the compact-symmetric packing (diagonal first, then the rows of the
upper triangle). Enums `Spline` and `Bound` are re-exported from `fastfields.dlpack`.

The autograd structure mirrors `jitfields` (`sym.py`, `resize.py`, `splinc.py`).

## Test

```bash
pip install -e . && python -m pytest tests/ -q
```
