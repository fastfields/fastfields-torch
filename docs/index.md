# fastfields-torch

**fastfields-torch** brings the fastfields field operators to **PyTorch**, and
the core operations are **fully differentiable** — drop them into a model and
gradients flow straight through. Functions take and return `torch` tensors and
allocate their own outputs.

## Install

```sh
pip install fastfields-torch \
    --extra-index-url https://fastfields.github.io/whl/cpu/
```

## Use it

```python
import torch
import fastfields.torch as ff

mat = torch.randn(5, 6, dtype=torch.float64, requires_grad=True)  # packed C=3
vec = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)

out = ff.sym_matvec(mat, vec)     # H @ vec, differentiable
out.sum().backward()              # gradients land on mat and vec
```

## What's inside

| Operation | Functions |
|---|---|
| **Positive-definite linear algebra** | `sym_matvec`, `sym_solve` (differentiable); `sym_invert`, `sym_addmatvec_`, `sym_submatvec_` over whole fields of small symmetric matrices |
| **Resampling** | `resample` (spline up/down-sampling), `restriction` (its adjoint), `spline_coeff` (coefficient prefilter) — all differentiable |
| **Spline gather/scatter** | `pull`, `push`, `count`, `grad` — all differentiable with respect to the sampling `grid` |
| **Distance transforms** | `dt_euclidean`, `dt_l1`, `dt_mesh`, and point-to-spline `dt_spline_table` / `dt_spline_brent` / `dt_spline_gaussnewton` |
| **Regularisers** | `field_*` (multi-channel field, per-channel penalties) and `flow_*` (vector flow, scalar penalties + the Lamé `shears`/`div` terms): the operator `*_matvec`, its diagonal `*_diag`, its stencil `*_kernel`, the `*_add…`/`*_sub…` accumulate forms of each, the RLS/JRLS-weighted `*_rls` variants, `*_relax`, and `*_forward` / `*_precond` |

### What is differentiable

Differentiable: `sym_matvec`, `sym_solve`, `resample`, `restriction`,
`spline_coeff`, all four pushpull operators (`pull`, `push`, `count`, `grad` —
with respect to the field *and* the sampling coordinates), and the regulariser
operators including `field_forward` / `field_precond` and their flow twins.

`grad` deserves a note: it is itself a derivative operator, so differentiating
it produces second-order terms — which is the point, since an image gradient
computed inside a loss has to be differentiable for the deformation that
produced `grid` to be learnable. Its `abs=True` mode is a majorisation, and
its backward is the adjoint of the majorised operator rather than a true
Jacobian; use it as the majorisation it is.

Not differentiable, but exposed for parity with `fastfields.numpy` /
`fastfields.cupy`: the distance transforms (`dt_euclidean`, `dt_l1`,
`dt_mesh`, the three `dt_spline_*`), `sym_invert`, and the `mat`/`vec`
gradients of `sym_addmatvec_` / `sym_submatvec_`. Forward always runs, even
when an input requires grad; only calling `.backward()` through the output
raises, with a `RuntimeError` naming the op.

## What this wrapper does not expose

Of the 58 operations in the underlying C++ hub this package wraps all but
nine, and the nine split into two groups.

**Adjoints that are used, not re-exported.** `pull_backward`,
`push_backward`, `count_backward`, `grad_backward` and `sym_matvec_backward`
are the gradients behind the `autograd.Function` of each forward op, which is
the supported way to reach them: call the forward op and `.backward()`.
Re-exporting them would be a second, unchecked spelling of a gradient, with
five pre-zeroed buffers and an accumulate-vs-overwrite contract for the caller
to get right. `fastfields.dlpack` has them raw.

**Three solver entry points that need an autograd decision first.**
`field_precond_` and `flow_precond_` are in-place preconditioner solves, and
`field_cg` runs an iterative solver whose gradient is a solve with the same
operator. Both are worth doing and neither is a mechanical wrapper, so they
are deliberately not here yet; `field_relax` is the in-place solver available
today, and `fastfields.numpy` has all three for the non-autograd case.

`tests/test_api_coverage.py` fails if that list ever changes by accident.

See the [API reference](api/index.md) for full signatures and options.
