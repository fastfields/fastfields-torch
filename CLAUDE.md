# fastfields-torch  (imports as `fastfields.torch`)

A user-friendly, **autograd-enabled** PyTorch interface over the
`fastfields.dlpack` (`fastfields-bind-py`) bindings. The autograd structure
mirrors `jitfields` (`sym.py`, `resize.py`, `splinc.py`).

```
… ─ lib ─ bind-py ─ torch ← (you are here) … ─ fastfields
```

## Philosophy / role
- Functions take and return `torch` tensors (CPU float32/float64), allocate
  their own outputs, and route through the DLPack path — so the same code will
  work for CUDA tensors once a GPU build is available. CUDA streams are
  forwarded when on device.
- Wraps ops in `torch.autograd.Function`s so gradients flow through the
  differentiable operations.

## Differentiability (high level)
- **Differentiable**: `sym_matvec` (grad w.r.t. `mat`, `vec`; backward via
  `sym_matvec`/`sym_matvec_backward`), `sym_solve` (w.r.t. `vec`, self-adjoint),
  `resample` / `restriction` (each other's adjoint at reciprocal scale),
  `spline_coeff` (self-adjoint).
- **Not differentiable** (raise if grad required): `sym_invert`,
  `dt_euclidean` / `dt_l1` / `dt_mesh`.
- Posdef matrices use the compact-symmetric packing (diagonal first, then upper
  triangle). `Spline`/`Bound` enums re-exported from `fastfields.dlpack`.

## Layout
`fastfields/torch/`: `__init__.py`, `_dt.py`, `_sym.py`, `_resample.py`,
`_util.py` (dtype/contiguity/autograd helpers). `tests/test_torch.py`.

## Build & test
```
pip install .                    # depends on fastfields-dlpack, torch, numpy
python -m pytest tests/ -q       # import from a neutral cwd
```
Prefer a regular install over editable (native-namespace merge).

## Conventions & caveats
- **PEP 420 namespace**: ships only `fastfields/torch/`, no
  `fastfields/__init__.py`.
- CPU is the exercised path; CUDA tensors depend on a real GPU build of the
  underlying library.
- Ruff: line-length 79, select B/E/F/I/W.

## Pointers
- Hierarchy: `/home/user/.github/profile/README.md`.
- Status: `/home/user/fastfields-lib/MIGRATION.md`.
