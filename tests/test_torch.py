"""Tests for fastfields_torch: forward correctness + autograd (gradcheck)."""

import pytest
import torch

import fastfields.torch as fft

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


def test_sym_matvec_broadcasts_batch_dims():
    # mat batch (1,) vs vec batch (5,): wrapper broadcasts and matches the
    # manually-broadcast dense product; broadcast operand is zero-copy.
    C = 3
    mat = torch.randn(1, len(_PACK[C]), dtype=torch.float64)  # batch (1,)
    vec = torch.randn(5, C, dtype=torch.float64)  # batch (5,)
    out = fft.sym_matvec(mat, vec)
    assert out.shape == (5, C)
    dense = dense_from_packed(mat, C).expand(5, C, C)
    ref = (dense @ vec.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(out, ref, atol=1e-10)


def test_gradcheck_sym_matvec_broadcast():
    # Autograd must reduce the broadcast gradients back to the original
    # (batch (1,)) matrix shape.
    C = 2
    mat = torch.randn(
        1, len(_PACK[C]), dtype=torch.float64, requires_grad=True
    )
    vec = torch.randn(4, C, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fft.sym_matvec, (mat, vec))


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
    # Pin anchor="first" (scale = in/out = 0.5) so the explicit reciprocal
    # scale [2.0] below is the matching adjoint.
    x = torch.randn(3, 5, dtype=torch.float64)
    y = torch.randn(3, 10, dtype=torch.float64)
    Rx = fft.resample(x, 10, spline=2, anchor="first")
    Rty = fft.restriction(y, 5, spline=2, scale=[2.0])
    assert torch.allclose((Rx * y).sum(), (x * Rty).sum(), atol=1e-10)


def test_resample_restriction_adjoint_by_anchor():
    # resample(a -> b) and restriction(b -> a) with the SAME anchor are exact
    # adjoints: restriction derives the reciprocal scale from its own shapes.
    for anchor in ("centers", "edges", "first", "last"):
        x = torch.randn(3, 5, dtype=torch.float64)
        y = torch.randn(3, 10, dtype=torch.float64)
        Rx = fft.resample(x, 10, spline=2, anchor=anchor)
        Rty = fft.restriction(y, 5, spline=2, anchor=anchor)
        assert torch.allclose((Rx * y).sum(), (x * Rty).sum(), atol=1e-10), (
            anchor
        )


# --------------------------------------------------------------------------- #
# anchor conventions (match interpol.resize)
# --------------------------------------------------------------------------- #


def test_anchor_scale_shift_mapping():
    from fastfields.torch._resample import _anchor_scale_shift

    for name, abbr, exp_scale, exp_shift in [
        ("centers", "c", 7 / 3, 0.0),
        ("edges", "e", 2.0, 0.5),
        ("first", "f", 2.0, 0.0),
        ("last", "l", 2.0, 1.0),
    ]:
        scale, shift = _anchor_scale_shift(name, (8,), (4,), 1)
        assert shift == exp_shift
        assert scale == pytest.approx([exp_scale])
        assert _anchor_scale_shift(abbr, (8,), (4,), 1) == (scale, shift)


def test_anchor_unknown_raises():
    with pytest.raises(ValueError, match="anchor"):
        fft.resample(torch.arange(8, dtype=torch.float64), 4, anchor="nope")


@pytest.mark.parametrize(
    "anchor,expected",
    [
        ("centers", [0.0, 7 / 3, 14 / 3, 7.0]),
        ("first", [0.0, 2.0, 4.0, 6.0]),
        ("edges", [0.5, 2.5, 4.5, 6.5]),
        ("last", [1.0, 3.0, 5.0, 7.0]),
    ],
)
def test_resample_anchor_matches_grid(anchor, expected):
    # linear interp of the ramp reproduces the sampled coordinate; all coords
    # below stay inside [0, 7].
    x = torch.arange(8, dtype=torch.float64)
    out = fft.resample(x, 4, spline=1, bound=3, anchor=anchor)
    assert out.shape == (4,)
    assert torch.allclose(
        out, torch.tensor(expected, dtype=torch.float64), atol=1e-10
    )


def test_resample_default_anchor_is_centers():
    x = torch.arange(8, dtype=torch.float64)
    default = fft.resample(x, 4, spline=1, bound=3)
    centers = fft.resample(x, 4, spline=1, bound=3, anchor="centers")
    assert torch.equal(default, centers)


# --------------------------------------------------------------------------- #
# Autograd (gradcheck) — the key deliverable
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("C", [2, 3])
def test_gradcheck_sym_matvec(C):
    mat = torch.randn(
        3, len(_PACK[C]), dtype=torch.float64, requires_grad=True
    )
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
        lambda v: fft.sym_solve(mat, v, weight), (vec,)
    )


def test_gradcheck_resample():
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda t: fft.resample(t, 10, spline=2, bound=3), (x,)
    )


def test_gradcheck_restriction():
    x = torch.randn(2, 10, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda t: fft.restriction(t, 5, spline=2, bound=3), (x,)
    )


def test_gradcheck_spline_coeff():
    x = torch.randn(2, 7, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: fft.spline_coeff(t, 3, 3), (x,))


# --------------------------------------------------------------------------- #
# Distance transforms (non-differentiable)
# --------------------------------------------------------------------------- #


def test_dt_euclidean_values():
    x = torch.tensor([[0.0, 1e30, 1e30, 1e30, 0.0]], dtype=torch.float64)
    d = fft.dt_euclidean(x)
    assert torch.allclose(
        d, torch.tensor([[0.0, 1.0, 4.0, 1.0, 0.0]], dtype=torch.float64)
    )


def test_dt_l1_values():
    x = torch.tensor([[0.0, 1e30, 1e30, 0.0]], dtype=torch.float64)
    d = fft.dt_l1(x)
    assert torch.allclose(
        d, torch.tensor([[0.0, 1.0, 1.0, 0.0]], dtype=torch.float64)
    )


# --------------------------------------------------------------------------- #
# Non-differentiable ops must refuse grad
# --------------------------------------------------------------------------- #


def test_nondiff_ops_reject_grad():
    with pytest.raises(ValueError):
        fft.sym_invert(torch.zeros(3, dtype=torch.float64, requires_grad=True))
    with pytest.raises(ValueError):
        fft.dt_euclidean(
            torch.zeros(4, dtype=torch.float64, requires_grad=True)
        )
    # sym_solve must not backprop through the matrix.
    mat = torch.randn(3, dtype=torch.float64, requires_grad=True)
    vec = torch.randn(2, dtype=torch.float64)
    with pytest.raises(ValueError):
        fft.sym_solve(mat, vec)


# --------------------------------------------------------------------------- #
# Cross-backend validation parity (fastfields-lib#17)
# --------------------------------------------------------------------------- #


def test_sym_matvec_channel_mismatch_raises():
    # C4: mat encodes C=2 (packed len 3) but vec has 3 channels -> the wrapper
    # must raise, not let a mismatched pair reach the raw binding (segfault).
    mat = torch.zeros(3, dtype=torch.float64)  # encodes C=2
    vec = torch.zeros(3, dtype=torch.float64)  # C=3
    with pytest.raises(ValueError):
        fft.sym_matvec(mat, vec)


def test_sym_solve_channel_mismatch_raises():
    mat = torch.zeros(3, dtype=torch.float64)  # encodes C=2
    vec = torch.zeros(3, dtype=torch.float64)  # C=3
    with pytest.raises(ValueError):
        fft.sym_solve(mat, vec)


def test_sym_matvec_non_triangular_packed_raises():
    # A packed length that is not a triangular number cannot encode any C.
    mat = torch.zeros(4, dtype=torch.float64)
    vec = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(ValueError):
        fft.sym_matvec(mat, vec)


def test_sym_solve_weight_channel_mismatch_raises():
    mat = torch.zeros(3, dtype=torch.float64)  # encodes C=2
    vec = torch.zeros(2, dtype=torch.float64)  # C=2 (ok)
    weight = torch.zeros(3, dtype=torch.float64)  # C=3 -> mismatch
    with pytest.raises(ValueError):
        fft.sym_solve(mat, vec, weight)


def test_dt_mesh_normalizes_faces_to_int64(monkeypatch):
    # C5: an int32 (or any non-int64) faces array must be cast to int64 before
    # the binding, which reads indices at int64 width. We intercept the binding
    # to capture the dtype that actually reaches it (the mesh shape contract is
    # finicky, so we do not run the real kernel here).
    import fastfields.torch._dt as dtmod

    seen = {}

    def spy(dist, nearest, loc, vertices, faces, signed, naive, stream=0):
        seen["faces_dtype"] = faces.dtype

    monkeypatch.setattr(dtmod._fb, "dt_mesh", spy)

    loc = torch.zeros(1, 3, dtype=torch.float32)
    verts = torch.zeros(1, 3, 3, dtype=torch.float32)
    for face_dtype in (torch.int32, torch.int16, torch.int64):
        faces = torch.tensor([[[0, 1, 2]]], dtype=face_dtype)
        fft.dt_mesh(loc, verts, faces, signed=False, naive=True)
        assert seen["faces_dtype"] == torch.int64
