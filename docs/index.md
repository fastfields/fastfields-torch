# fastfields-torch

`fastfields-torch` is a user-friendly, **autograd-enabled** PyTorch interface over the `fastfields.dlpack` bindings. Functions take and return `torch` tensors (CPU float32/float64), allocate their own outputs, and route through the bindings' DLPack path.

## Installation

```bash
pip install fastfields-torch
```

## Usage

```python
import torch
import fastfields.torch as fft

mat = torch.randn(5, 6, dtype=torch.float64, requires_grad=True)  # packed C=3
vec = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
out = fft.sym_matvec(mat, vec)     # differentiable: H @ vec
out.sum().backward()
```

See the [API reference](api/index.md) for the full list of operations.
