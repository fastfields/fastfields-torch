# fastfields-torch  (imports as `fastfields.torch`)

A user-friendly, **autograd-enabled** PyTorch interface over the
`fastfields.dlpack` bindings. The autograd structure
mirrors `jitfields` (`sym.py`, `resize.py`, `splinc.py`).

```
… ─ lib ─ dlpack ─ torch ← (you are here) … ─ fastfields
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
  `sym_solve_` (in-place, same gradient as `sym_solve` — backward never reads
  the pre-mutation value), `resample` / `restriction` (each other's adjoint at
  reciprocal scale), `spline_coeff` / `spline_coeff_` (self-adjoint).
- **Not differentiable, but still exposed** (parity with `fastfields.numpy`/
  `fastfields.cupy`): `dt_euclidean` / `dt_euclidean_`, `dt_l1` / `dt_l1_`,
  `dt_mesh` (no in-place form), `sym_invert` / `sym_invert_`. Forward always
  runs, including when an input requires grad; only calling `.backward()`
  through the output raises `RuntimeError` naming the op (a
  `torch.autograd.Function` whose `backward()` raises — see
  `_util.raise_not_differentiable` — rather than rejecting the call up front
  or omitting the op). See `API_CONTRACT.md` ("In-place policy") and
  fastfields#4.
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
