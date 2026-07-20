"""Tests for fastfields_torch: forward correctness + autograd (gradcheck)."""

import pytest
import torch

import fastfields_torch as fft

torch.manual_seed(0)

# Compact-sym packing (diagonal, then upper-triangle rows) index pairs.
_PACK = {
    2: [(0, 0), (1, 1), (0, 1)],
    3: [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)],
}


def dense_from_packed(mat, C):
    """Build a dense (..., C, C) symmetric matrix from compact storage."""
    pairs = _PACK[C]
    M = mat.new_zeros((*mat.shape[:-1], C, C))
    for k, (i, j) in enumerate(pairs):
        M[..., i, j] = mat[..., k]
        M[..., j, i] = mat[..., k]
    return M


def pack_from_dense(M, C):
    pairs = _PACK[C]
    return torch.stack([M[..., i, j] for (i, j) in pairs], dim=-1)


def random_spd(batch, C, dtype=torch.float64):
    A = torch.randn(*batch, C, C, dtype=dtype)
    A = A @ A.transpose(-1, -2) + (C + 1) * torch.eye(C, dtype=dtype)
    return pack_from_dense(A, C)


# --------------------------------------------------------------------------- #
# Forward correctness
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("C", [2, 3])
def test_sym_matvec_forward(dtype, C):
    mat = torch.randn(4, len(_PACK[C]), dtype=dtype)
    vec = torch.randn(4, C, dtype=dtype)
    out = fft.sym_matvec(mat, vec)
    ref = (dense_from_packed(mat, C) @ vec.unsqueeze(-1)).squeeze(-1)
    tol = 1e-5 if dtype == torch.float32 else 1e-10
    assert torch.allclose(out, ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("C", [2, 3])
def test_sym_solve_roundtrip(dtype, C):
    mat = random_spd((5,), C, dtype=dtype)
    x = torch.randn(5, C, dtype=dtype)
    b = fft.sym_matvec(mat, x)
    sol = fft.sym_solve(mat, b)
    # The binding's solve carries an internal float32 computation, so use a
    # loose tolerance (see project notes / final report).
    assert torch.allclose(sol, x, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("C", [2, 3])
def test_sym_invert_matches_dense(C):
    mat = random_spd((3,), C, dtype=torch.float64)
    inv = fft.sym_invert(mat)
    ref = pack_from_dense(torch.linalg.inv(dense_from_packed(mat, C)), C)
    assert torch.allclose(inv, ref, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_resample_factor1_identity(dtype):
    # scale=1, shift=0, integer nodes => identity for interpolating order 1.
    x = torch.randn(3, 8, dtype=dtype)
    y = fft.resample(x, 8, spline=1, bound=3, shift=0.0, scale=[1.0], ndim=1)
    tol = 1e-5 if dtype == torch.float32 else 1e-10
    assert torch.allclose(y, x, atol=tol, rtol=tol)


def test_resample_restriction_adjoint():
    # <R x, y> == <x, R^T y>, with R = resample and R^T = restriction(1/scale).
    x = torch.randn(3, 5, dtype=torch.float64)
    y = torch.randn(3, 10, dtype=torch.float64)
    Rx = fft.resample(x, 10, spline=2)
    Rty = fft.restriction(y, 5, spline=2, scale=[2.0])
    assert torch.allclose((Rx * y).sum(), (x * Rty).sum(), atol=1e-10)


# --------------------------------------------------------------------------- #
# Autograd (gradcheck) — the key deliverable
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("C", [2, 3])
def test_gradcheck_sym_matvec(C):
    mat = torch.randn(3, len(_PACK[C]), dtype=torch.float64, requires_grad=True)
    vec = torch.randn(3, C, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fft.sym_matvec, (mat, vec))


@pytest.mark.parametrize("C", [2, 3])
def test_gradcheck_sym_solve(C):
    # solve backpropagates through `vec` only; matrix is a detached constant.
    mat = random_spd((2,), C, dtype=torch.float64).detach()
    vec = torch.randn(2, C, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: fft.sym_solve(mat, v), (vec,))


def test_gradcheck_sym_solve_weighted():
    C = 3
    mat = random_spd((2,), C, dtype=torch.float64).detach()
    weight = torch.rand(2, C, dtype=torch.float64) + 0.5
    vec = torch.randn(2, C, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda v: fft.sym_solve(mat, v, weight), (vec,))


def test_gradcheck_resample():
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda t: fft.resample(t, 10, spline=2, bound=3), (x,))


def test_gradcheck_restriction():
    x = torch.randn(2, 10, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda t: fft.restriction(t, 5, spline=2, bound=3), (x,))


def test_gradcheck_spline_coeff():
    x = torch.randn(2, 7, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: fft.spline_coeff(t, 3, 3), (x,))


# --------------------------------------------------------------------------- #
# Distance transforms (non-differentiable)
# --------------------------------------------------------------------------- #

def test_dt_euclidean_values():
    x = torch.tensor([[0.0, 1e30, 1e30, 1e30, 0.0]], dtype=torch.float64)
    d = fft.dt_euclidean(x)
    assert torch.allclose(d, torch.tensor([[0.0, 1.0, 4.0, 1.0, 0.0]],
                                           dtype=torch.float64))


def test_dt_l1_values():
    x = torch.tensor([[0.0, 1e30, 1e30, 0.0]], dtype=torch.float64)
    d = fft.dt_l1(x)
    assert torch.allclose(d, torch.tensor([[0.0, 1.0, 1.0, 0.0]],
                                           dtype=torch.float64))


# --------------------------------------------------------------------------- #
# Non-differentiable ops must refuse grad
# --------------------------------------------------------------------------- #

def test_nondiff_ops_reject_grad():
    with pytest.raises(ValueError):
        fft.sym_invert(torch.zeros(3, dtype=torch.float64, requires_grad=True))
    with pytest.raises(ValueError):
        fft.dt_euclidean(torch.zeros(4, dtype=torch.float64, requires_grad=True))
    # sym_solve must not backprop through the matrix.
    mat = torch.randn(3, dtype=torch.float64, requires_grad=True)
    vec = torch.randn(2, dtype=torch.float64)
    with pytest.raises(ValueError):
        fft.sym_solve(mat, vec)
