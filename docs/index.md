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
| **Positive-definite linear algebra** | `sym_matvec`, `sym_solve` (differentiable); `sym_invert` over whole fields of small symmetric matrices |
| **Resampling** | `resample` (spline up/down-sampling), `restriction` (its adjoint), `spline_coeff` (coefficient prefilter) — all differentiable |
| **Distance transforms** | `dt_euclidean`, `dt_l1`, `dt_mesh` |

`sym_matvec`, `sym_solve`, `resample`, `restriction` and `spline_coeff` support
autograd; the distance transforms and `sym_invert` do not and will raise if an
input requires grad.

See the [API reference](api/index.md) for full signatures and options.
