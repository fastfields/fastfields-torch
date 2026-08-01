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
    y = fft.resample(
        x, shape=8, order=1, bound=3, shift=0.0, scale=[1.0], ndim=1
    )
    tol = 1e-5 if dtype == torch.float32 else 1e-10
    assert torch.allclose(y, x, atol=tol, rtol=tol)


def test_resample_restriction_adjoint():
    # <R x, y> == <x, R^T y>, with R = resample and R^T = restriction(1/scale).
    # Pin anchor="first" (scale = in/out = 0.5) so the explicit reciprocal
    # scale [2.0] below is the matching adjoint.
    x = torch.randn(3, 5, dtype=torch.float64)
    y = torch.randn(3, 10, dtype=torch.float64)
    Rx = fft.resample(x, shape=10, order=2, anchor="first")
    Rty = fft.restriction(y, shape=5, order=2, scale=[2.0])
    assert torch.allclose((Rx * y).sum(), (x * Rty).sum(), atol=1e-10)


def test_resample_restriction_adjoint_by_anchor():
    # resample(a -> b) and restriction(b -> a) with the SAME anchor are exact
    # adjoints: restriction derives the reciprocal scale from its own shapes.
    for anchor in ("centers", "edges", "first", "last"):
        x = torch.randn(3, 5, dtype=torch.float64)
        y = torch.randn(3, 10, dtype=torch.float64)
        Rx = fft.resample(x, shape=10, order=2, anchor=anchor)
        Rty = fft.restriction(y, shape=5, order=2, anchor=anchor)
        assert torch.allclose((Rx * y).sum(), (x * Rty).sum(), atol=1e-10), (
            anchor
        )


# --------------------------------------------------------------------------- #
# anchor conventions (match interpol.resize)
# --------------------------------------------------------------------------- #


def test_anchor_scale_shift_mapping():
    # the anchor->(scale, shift) map is shared via fastfields.dlpack
    from fastfields.dlpack import anchor_scale_shift

    for name, abbr, exp_scale, exp_shift in [
        ("centers", "c", 7 / 3, 0.0),
        ("edges", "e", 2.0, 0.5),
        ("first", "f", 2.0, 0.0),
        ("last", "l", 2.0, 1.0),
    ]:
        scale, shift = anchor_scale_shift(name, (8,), (4,), 1)
        assert shift == exp_shift
        assert scale == pytest.approx([exp_scale])
        assert anchor_scale_shift(abbr, (8,), (4,), 1) == (scale, shift)


def test_anchor_unknown_raises():
    with pytest.raises(ValueError, match="anchor"):
        fft.resample(
            torch.arange(8, dtype=torch.float64), shape=4, anchor="nope"
        )


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
    out = fft.resample(x, shape=4, order=1, bound=3, anchor=anchor)
    assert out.shape == (4,)
    assert torch.allclose(
        out, torch.tensor(expected, dtype=torch.float64), atol=1e-10
    )


@pytest.mark.parametrize("fn", ["resample", "restriction"])
@pytest.mark.parametrize("bad_ndim", [0, -1, 2])
def test_resample_ndim_out_of_range_raises(fn, bad_ndim):
    # ndim must be in 1..inp.dim(); a 1-D input only supports ndim=1.
    x = torch.arange(8, dtype=torch.float64)
    with pytest.raises(ValueError, match="ndim"):
        getattr(fft, fn)(x, shape=4, ndim=bad_ndim)


def test_resample_default_anchor_is_centers():
    x = torch.arange(8, dtype=torch.float64)
    default = fft.resample(x, shape=4, order=1, bound=3)
    centers = fft.resample(x, shape=4, order=1, bound=3, anchor="centers")
    assert torch.equal(default, centers)


def test_resample_factor_arg_matches_shape():
    # factor=2 on a length-5 axis -> length-10 output, same as shape=10.
    x = torch.arange(5, dtype=torch.float64)
    by_factor = fft.resample(x, factor=2, order=1)
    by_shape = fft.resample(x, shape=10, order=1)
    assert by_factor.shape == (10,)
    assert torch.equal(by_factor, by_shape)


def test_resample_order_bound_string_aliases():
    # order/bound accept ints, enums or names (unified with the numpy wrapper).
    x = torch.arange(8, dtype=torch.float64)
    by_name = fft.resample(x, shape=4, order="linear", bound="dct2")
    by_int = fft.resample(x, shape=4, order=1, bound=3)
    assert torch.equal(by_name, by_int)
    with pytest.raises(ValueError, match="spline order"):
        fft.resample(x, shape=4, order="nope")
    with pytest.raises(ValueError, match="boundary"):
        fft.resample(x, shape=4, bound="nope")


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
        lambda t: fft.resample(t, shape=10, order=2, bound=3), (x,)
    )


def test_gradcheck_restriction():
    x = torch.randn(2, 10, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda t: fft.restriction(t, shape=5, order=2, bound=3), (x,)
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


# --------------------------------------------------------------------------- #
# pushpull (spline gather / scatter) + autograd                               #
# --------------------------------------------------------------------------- #


def test_pull_linear_interpolation():
    inp = torch.tensor([[0.0], [10.0], [20.0], [30.0]], dtype=torch.float64)
    grid = torch.tensor([[0.5], [1.5], [2.5]], dtype=torch.float64)
    out = fft.pull(inp, grid, order=1)
    assert torch.allclose(
        out.squeeze(), torch.tensor([5.0, 15.0, 25.0], dtype=torch.float64)
    )


def test_count_identity_is_ones():
    grid = torch.arange(5.0, dtype=torch.float64).reshape(5, 1)
    assert torch.allclose(
        fft.count(grid, shape=5, order=1).squeeze(),
        torch.ones(5, dtype=torch.float64),
    )


def test_push_is_pull_adjoint():
    grid = torch.linspace(0, 5, 4, dtype=torch.float64).reshape(4, 1)
    x = torch.randn(6, 1, dtype=torch.float64)
    y = torch.randn(4, 1, dtype=torch.float64)
    px = fft.pull(x, grid, order=2)
    py = fft.push(y, grid, shape=6, order=2)
    assert torch.allclose((px * y).sum(), (x * py).sum(), atol=1e-8)


def test_pull_gradcheck_wrt_input():
    grid = torch.tensor([[0.5], [1.5], [2.5]], dtype=torch.float64)
    inp = torch.randn(4, 1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: fft.pull(x, grid, order=2), (inp,), eps=1e-6, atol=1e-4
    )


def test_push_gradcheck_wrt_input():
    grid = torch.linspace(0, 5, 4, dtype=torch.float64).reshape(4, 1)
    inp = torch.randn(4, 1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: fft.push(x, grid, shape=6, order=2),
        (inp,),
        eps=1e-6,
        atol=1e-4,
    )


# --- differentiating through `grid` ---------------------------------------- #
#
# gradcheck is the actual proof that the grid adjoint is right: it compares the
# analytic backward against a central finite difference of the forward.
#
# Grid coordinates are kept away from integer/half-integer positions (the
# spline knots). A spline of order k is C^(k-1), so for order 1 the derivative
# wrt position genuinely jumps at a knot and a finite difference straddling one
# is meaningless -- that is a property of the interpolant, not a bug.


def _safe_grid(*shape, d=1, lo=0.3, hi=None, generator=None):
    """Random coordinates inside (lo, hi), off the knots, requiring grad."""
    hi = 3.4 if hi is None else hi
    g = torch.rand(*shape, d, dtype=torch.float64) * (hi - lo) + lo
    # nudge away from integers / half-integers
    g = g + 0.137
    return g.requires_grad_(True)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_pull_gradcheck_wrt_grid_1d(order):
    inp = torch.randn(7, 2, dtype=torch.float64)
    grid = _safe_grid(4)
    assert torch.autograd.gradcheck(
        lambda g: fft.pull(inp, g, order=order), (grid,), eps=1e-6, atol=1e-5
    )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_pull_gradcheck_wrt_both_1d(order):
    inp = torch.randn(7, 2, dtype=torch.float64, requires_grad=True)
    grid = _safe_grid(4)
    assert torch.autograd.gradcheck(
        lambda x, g: fft.pull(x, g, order=order),
        (inp, grid),
        eps=1e-6,
        atol=1e-5,
    )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_pull_gradcheck_wrt_both_2d(order):
    inp = torch.randn(5, 6, 2, dtype=torch.float64, requires_grad=True)
    grid = _safe_grid(3, 3, d=2)
    assert torch.autograd.gradcheck(
        lambda x, g: fft.pull(x, g, order=order),
        (inp, grid),
        eps=1e-6,
        atol=1e-5,
    )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_pull_gradcheck_wrt_both_3d(order):
    inp = torch.randn(5, 5, 5, 1, dtype=torch.float64, requires_grad=True)
    grid = _safe_grid(2, 2, 2, d=3)
    assert torch.autograd.gradcheck(
        lambda x, g: fft.pull(x, g, order=order),
        (inp, grid),
        eps=1e-6,
        atol=1e-5,
    )


@pytest.mark.parametrize("bound", ["dct2", "dst2", "dft", "replicate", "zero"])
def test_pull_gradcheck_wrt_both_bounds(bound):
    inp = torch.randn(6, 5, 2, dtype=torch.float64, requires_grad=True)
    grid = _safe_grid(3, 2, d=2)
    assert torch.autograd.gradcheck(
        lambda x, g: fft.pull(x, g, order=3, bound=bound),
        (inp, grid),
        eps=1e-6,
        atol=1e-5,
    )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_push_gradcheck_wrt_both_1d(order):
    grid = _safe_grid(4)
    inp = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x, g: fft.push(x, g, shape=7, order=order),
        (inp, grid),
        eps=1e-6,
        atol=1e-5,
    )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_push_gradcheck_wrt_both_2d(order):
    grid = _safe_grid(3, 3, d=2)
    inp = torch.randn(3, 3, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x, g: fft.push(x, g, shape=(5, 6), order=order),
        (inp, grid),
        eps=1e-6,
        atol=1e-5,
    )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_push_gradcheck_wrt_both_3d(order):
    grid = _safe_grid(2, 2, 2, d=3)
    inp = torch.randn(2, 2, 2, 1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x, g: fft.push(x, g, shape=(5, 5, 5), order=order),
        (inp, grid),
        eps=1e-6,
        atol=1e-5,
    )


def test_pull_gradcheck_batched():
    inp = torch.randn(2, 6, 5, 2, dtype=torch.float64, requires_grad=True)
    grid = _safe_grid(2, 3, 2, d=2)
    assert torch.autograd.gradcheck(
        lambda x, g: fft.pull(x, g, order=2),
        (inp, grid),
        eps=1e-6,
        atol=1e-5,
    )


@pytest.mark.parametrize("op", ["pull", "push"])
def test_field_grad_matches_whether_or_not_grid_needs_grad(op):
    """The fused (grid-aware) branch and the cheap push/pull-only branch must
    agree on the field gradient -- they are two code paths for one quantity."""
    grid0 = torch.rand(2, 2, 2, dtype=torch.float64) * 3 + 0.437
    x0 = torch.randn(
        (2, 2, 2) if op == "push" else (6, 5, 2), dtype=torch.float64
    )

    def run(grid_requires_grad):
        x = x0.clone().requires_grad_(True)
        g = grid0.clone().requires_grad_(grid_requires_grad)
        if op == "pull":
            out = fft.pull(x, g, order=3)
        else:
            out = fft.push(x, g, shape=(6, 5), order=3)
        out.pow(2).sum().backward()
        return x.grad

    assert torch.allclose(run(False), run(True), atol=1e-12)


def test_grid_grad_zero_outside_fov():
    """extrapolate=0 gates samples past the voxel centres; their grid gradient
    must be exactly zero (the gate is a step, not a smooth factor)."""
    inp = torch.arange(6.0, dtype=torch.float64).reshape(6, 1)
    grid = torch.tensor(
        [[-4.0], [2.437], [11.0]], dtype=torch.float64, requires_grad=True
    )
    out = fft.pull(inp, grid, order=1, extrapolate=0)
    out.sum().backward()
    assert grid.grad[0].item() == 0.0
    assert grid.grad[2].item() == 0.0
    assert grid.grad[1].item() != 0.0


# --------------------------------------------------------------------------- #
# regularisers + autograd                                                     #
# --------------------------------------------------------------------------- #


def test_field_matvec_absolute_and_gradcheck():
    f = torch.randn(8, 2, dtype=torch.float64, requires_grad=True)
    out = fft.field_matvec(f, absolute=[2.0, 3.0], ndim=1)
    assert torch.allclose(out[:, 0], 2.0 * f.detach()[:, 0])
    assert torch.allclose(out[:, 1], 3.0 * f.detach()[:, 1])
    assert torch.autograd.gradcheck(
        lambda z: fft.field_matvec(
            z, absolute=[2.0, 3.0], membrane=[0.5, 0.5], ndim=1
        ),
        (f,),
        eps=1e-6,
        atol=1e-4,
    )


def test_flow_matvec_gradcheck():
    v = torch.randn(8, 1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda z: fft.flow_matvec(z, absolute=2.0, membrane=0.5, ndim=1),
        (v,),
        eps=1e-6,
        atol=1e-4,
    )


@pytest.mark.parametrize("bound", ["dct2", "dft"])
def test_flow_matvec_lame_gradcheck(bound):
    # The linear-elastic (shears/div) operator is self-adjoint, so its
    # autograd backward (which re-applies the same matvec) must pass gradcheck
    # under reflecting boundaries too.
    v = torch.randn(5, 6, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda z: fft.flow_matvec(
            z,
            absolute=0.2,
            membrane=0.5,
            shears=1.3,
            div=0.7,
            ndim=2,
            bound=bound,
        ),
        (v,),
        eps=1e-6,
        atol=1e-4,
    )


def test_field_diag_absolute():
    d = fft.field_diag((8, 2), absolute=2.0, ndim=1)
    assert torch.allclose(d, torch.tensor(2.0, dtype=torch.float64))


@pytest.mark.parametrize(
    "kw,is_matrix,width",
    [
        ({"absolute": 2.5}, False, 1),
        ({"membrane": 1.0}, False, 3),
        ({"bending": 1.0}, False, 5),
        ({"shears": 1.3, "div": 0.7}, True, 3),
        (
            {
                "absolute": 0.3,
                "membrane": 0.5,
                "bending": 0.4,
                "shears": 1.3,
                "div": 0.7,
            },
            True,
            5,
        ),
    ],
)
def test_flow_kernel_is_matvec_impulse_response(kw, is_matrix, width):
    # The materialised stencil equals flow_matvec's impulse response interior.
    C = 2
    K = fft.flow_kernel(2, **kw)
    assert tuple(K.shape) == (
        (width, width, C, C) if is_matrix else (width, width, C)
    )
    kd = width
    N, cc, half = 2 * kd + 1, kd, kd // 2
    for j0 in range(C):
        x = torch.zeros(N, N, C, dtype=torch.float64)
        x[cc, cc, j0] = 1.0
        o = fft.flow_matvec(x, ndim=2, **kw)
        for a in range(kd):
            for b in range(kd):
                for i in range(C):
                    got = o[cc + a - half, cc + b - half, i]
                    kern = (
                        K[a, b, i, j0]
                        if is_matrix
                        else (K[a, b, i] if i == j0 else 0.0)
                    )
                    assert torch.allclose(
                        got,
                        torch.as_tensor(kern, dtype=torch.float64),
                        atol=1e-10,
                    )


def _flow_hessian_2d(H, W, C=2):
    # Per-voxel SPD Hessian, packed compact-symmetric -> (H, W, C*(C+1)//2).
    return random_spd((H, W), C).reshape(H, W, len(_PACK[C]))


def test_flow_forward_is_sym_matvec_plus_flow_matvec():
    # (M + R) v == M v + R v, by construction.
    H, W = 5, 6
    mat = _flow_hessian_2d(H, W)
    vec = torch.randn(H, W, 2, dtype=torch.float64)
    kw = dict(absolute=0.3, membrane=0.7, shears=1.0, div=0.5)
    fwd = fft.flow_forward(mat, vec, ndim=2, **kw)
    expect = fft.sym_matvec(mat, vec) + fft.flow_matvec(vec, ndim=2, **kw)
    assert torch.allclose(fwd, expect, atol=1e-10)


def test_flow_precond_solves_diagonal_system():
    # x = (M + diag(R)) \ v  =>  M x + diag(R) x == v.
    H, W = 5, 6
    mat = _flow_hessian_2d(H, W)
    vec = torch.randn(H, W, 2, dtype=torch.float64)
    kw = dict(absolute=0.3, membrane=0.7, shears=1.0, div=0.5)
    x = fft.flow_precond(mat, vec, ndim=2, **kw)
    diag = fft.flow_diag(vec.shape, ndim=2, **kw)
    residual = fft.sym_matvec(mat, x) + diag * x - vec
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-5)


def test_flow_forward_gradcheck():
    # Composition of autograd ops -> differentiable wrt mat and vec.
    H, W = 3, 4
    mat = _flow_hessian_2d(H, W).requires_grad_(True)
    vec = torch.randn(H, W, 2, dtype=torch.float64, requires_grad=True)
    kw = dict(absolute=0.2, membrane=0.5, shears=1.3, div=0.7)
    assert torch.autograd.gradcheck(
        lambda m, v: fft.flow_forward(m, v, ndim=2, **kw),
        (mat, vec),
        eps=1e-6,
        atol=1e-4,
    )


def test_flow_precond_gradcheck_vec():
    # sym_solve is self-adjoint and differentiable wrt vec (not mat).
    H, W = 3, 4
    mat = _flow_hessian_2d(H, W)
    vec = torch.randn(H, W, 2, dtype=torch.float64, requires_grad=True)
    kw = dict(absolute=0.3, membrane=0.5, shears=1.0, div=0.5)
    assert torch.autograd.gradcheck(
        lambda v: fft.flow_precond(mat, v, ndim=2, **kw),
        (vec,),
        eps=1e-6,
        atol=1e-4,
    )


def test_flow_accumulate_variants():
    torch.manual_seed(3)
    H, W = 5, 6
    flow = torch.randn(H, W, 2, dtype=torch.float64)
    base = torch.randn(H, W, 2, dtype=torch.float64)
    kw = dict(absolute=0.3, membrane=0.7, shears=1.0, div=0.5, ndim=2)
    L = fft.flow_matvec(flow, **kw)
    d = fft.flow_diag(base.shape, **kw)
    assert torch.allclose(fft.flow_matvec_add(base, flow, **kw), base + L)
    assert torch.allclose(fft.flow_matvec_sub(base, flow, **kw), base - L)
    assert torch.allclose(fft.flow_diag_add(base, **kw), base + d)
    assert torch.allclose(fft.flow_diag_sub(base, **kw), base - d)
    # in-place forms mutate and return the same tensor
    a = base.clone()
    assert fft.flow_matvec_add_(a, flow, **kw) is a
    assert torch.allclose(a, base + L)
    s = base.clone()
    assert fft.flow_diag_sub_(s, **kw) is s
    assert torch.allclose(s, base - d)


def test_flow_matvec_add_gradcheck():
    # fresh-array add composes the autograd flow_matvec -> differentiable.
    H, W = 3, 4
    base = torch.randn(H, W, 2, dtype=torch.float64, requires_grad=True)
    flow = torch.randn(H, W, 2, dtype=torch.float64, requires_grad=True)
    kw = dict(absolute=0.2, membrane=0.5, shears=1.3, div=0.7, ndim=2)
    assert torch.autograd.gradcheck(
        lambda p, f: fft.flow_matvec_add(p, f, **kw),
        (base, flow),
        eps=1e-6,
        atol=1e-4,
    )


def _field_hessian(H, W, C, seed):
    A = torch.randn(H * W, C, C, dtype=torch.float64)
    A = A @ A.transpose(-1, -2) + (C + 1) * torch.eye(C, dtype=torch.float64)
    return pack_from_dense(A, C).reshape(H, W, len(_PACK[C]))


def test_field_forward_and_precond():
    H, W, C = 5, 6, 2
    mat = _field_hessian(H, W, C, 5)
    vec = torch.randn(H, W, C, dtype=torch.float64)
    kw = dict(absolute=[0.3, 0.4], membrane=[0.7, 0.5], ndim=2)
    fwd = fft.field_forward(mat, vec, **kw)
    assert torch.allclose(
        fwd, fft.sym_matvec(mat, vec) + fft.field_matvec(vec, **kw), atol=1e-10
    )
    x = fft.field_precond(mat, vec, **kw)
    diag = fft.field_diag(vec.shape, **kw)
    residual = fft.sym_matvec(mat, x) + diag * x - vec
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-5)


def test_field_accumulate_variants():
    H, W, C = 5, 6, 2
    field = torch.randn(H, W, C, dtype=torch.float64)
    base = torch.randn(H, W, C, dtype=torch.float64)
    kw = dict(absolute=[0.3, 0.4], membrane=[0.7, 0.5], ndim=2)
    L = fft.field_matvec(field, **kw)
    d = fft.field_diag(base.shape, **kw)
    assert torch.allclose(fft.field_matvec_add(base, field, **kw), base + L)
    assert torch.allclose(fft.field_matvec_sub(base, field, **kw), base - L)
    assert torch.allclose(fft.field_diag_add(base, **kw), base + d)
    assert torch.allclose(fft.field_diag_sub(base, **kw), base - d)
    a = base.clone()
    assert fft.field_matvec_add_(a, field, **kw) is a
    assert torch.allclose(a, base + L)


@pytest.mark.parametrize(
    "order,width,kw",
    [
        (1, 1, dict(absolute=[2.5, 1.5])),
        (2, 3, dict(absolute=[0.3, 0.4], membrane=[1.0, 0.7])),
        (
            3,
            5,
            dict(absolute=[0.3, 0.4], membrane=[0.5, 0.6], bending=[1.0, 0.8]),
        ),
    ],
)
def test_field_kernel_is_matvec_impulse_response(order, width, kw):
    C = 2
    K = fft.field_kernel(2, **kw)
    assert tuple(K.shape) == (width, width, C)
    kd = width
    N, cc, half = 2 * kd + 1, kd, kd // 2
    for c0 in range(C):
        x = torch.zeros(N, N, C, dtype=torch.float64)
        x[cc, cc, c0] = 1.0
        o = fft.field_matvec(x, ndim=2, **kw)
        for a in range(kd):
            for b in range(kd):
                for c in range(C):
                    got = o[cc + a - half, cc + b - half, c]
                    kern = (
                        K[a, b, c]
                        if c == c0
                        else torch.tensor(0.0, dtype=torch.float64)
                    )
                    assert torch.allclose(
                        got,
                        torch.as_tensor(kern, dtype=torch.float64),
                        atol=1e-10,
                    )


def test_field_kernel_channels_inference():
    k = fft.field_kernel(2, absolute=[1.0, 2.0, 3.0])
    assert tuple(k.shape) == (1, 1, 3)
    assert tuple(fft.field_kernel(1, absolute=2.0, channels=4).shape) == (1, 4)


# --------------------------------------------------------------------------- #
# In-place accumulate ops: autograd safety                                    #
#                                                                             #
# These ops are `acc <- acc (+/-) g(...)`, i.e. additive in the tensor they   #
# mutate, so d/d(acc) = I and no pre-mutation value is needed by backward.    #
# The tests below verify that property rather than assuming it.               #
# See API_CONTRACT.md, "In-place policy".                                     #
# --------------------------------------------------------------------------- #


_ACC_KW_FIELD = dict(absolute=[0.3, 0.4], membrane=[0.7, 0.5], ndim=2)
_ACC_KW_FLOW = dict(absolute=0.3, membrane=0.7, shears=1.0, div=0.5, ndim=2)


def test_inplace_accumulate_saves_nothing_for_backward():
    """The autograd Functions must not stash the mutated tensor.

    This is the structural form of the safety argument: if nothing is saved,
    mutating the buffer cannot invalidate anything backward needs.
    """
    H, W, C = 4, 5, 2
    field = torch.randn(H, W, C, dtype=torch.float64, requires_grad=True)
    base = torch.zeros(H, W, C, dtype=torch.float64, requires_grad=True)
    out = fft.field_matvec_add_(base * 1.0, field, **_ACC_KW_FIELD)
    # grad_fn is our accumulate Function; it should hold no saved tensors.
    assert out.grad_fn is not None
    assert not getattr(out.grad_fn, "saved_tensors", ())


def test_inplace_accumulate_grad_wrt_accumulator_is_identity():
    """d(acc + L@x)/d(acc) == I, so the incoming gradient passes through."""
    H, W, C = 4, 5, 2
    field = torch.randn(H, W, C, dtype=torch.float64)
    base = torch.randn(H, W, C, dtype=torch.float64, requires_grad=True)
    out = fft.field_matvec_add_(base * 1.0, field, **_ACC_KW_FIELD)
    g = torch.randn_like(out)
    (grad,) = torch.autograd.grad(out, base, g)
    assert torch.allclose(grad, g, atol=1e-12)


def test_inplace_accumulate_sub_grad_wrt_accumulator_is_identity():
    H, W, C = 4, 5, 2
    field = torch.randn(H, W, C, dtype=torch.float64)
    base = torch.randn(H, W, C, dtype=torch.float64, requires_grad=True)
    out = fft.field_matvec_sub_(base * 1.0, field, **_ACC_KW_FIELD)
    g = torch.randn_like(out)
    (grad,) = torch.autograd.grad(out, base, g)
    assert torch.allclose(grad, g, atol=1e-12)


@pytest.mark.parametrize("fn_name", ["field_matvec_add_", "field_matvec_sub_"])
def test_field_matvec_inplace_gradcheck(fn_name):
    """gradcheck through the genuinely in-place op (non-leaf accumulator)."""
    H, W, C = 3, 4, 2
    base = torch.randn(H, W, C, dtype=torch.float64, requires_grad=True)
    field = torch.randn(H, W, C, dtype=torch.float64, requires_grad=True)
    fn = getattr(fft, fn_name)

    def run(p, f):
        # `p * 1.0` makes a non-leaf so torch allows the in-place mutation,
        # exactly as one would do with `Tensor.add_`.
        return fn(p * 1.0, f, **_ACC_KW_FIELD)

    assert torch.autograd.gradcheck(run, (base, field), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("fn_name", ["flow_matvec_add_", "flow_matvec_sub_"])
def test_flow_matvec_inplace_gradcheck(fn_name):
    H, W = 3, 4
    base = torch.randn(H, W, 2, dtype=torch.float64, requires_grad=True)
    flow = torch.randn(H, W, 2, dtype=torch.float64, requires_grad=True)
    fn = getattr(fft, fn_name)

    def run(p, f):
        return fn(p * 1.0, f, **_ACC_KW_FLOW)

    assert torch.autograd.gradcheck(run, (base, flow), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("fn_name", ["field_diag_add_", "field_diag_sub_"])
def test_field_diag_inplace_gradcheck(fn_name):
    H, W, C = 3, 4, 2
    base = torch.randn(H, W, C, dtype=torch.float64, requires_grad=True)
    fn = getattr(fft, fn_name)
    assert torch.autograd.gradcheck(
        lambda p: fn(p * 1.0, **_ACC_KW_FIELD), (base,), eps=1e-6, atol=1e-4
    )


def test_inplace_accumulate_rejects_leaf_requiring_grad():
    """Torch's ordinary leaf rule still applies -- same as ``Tensor.add_``."""
    H, W, C = 3, 4, 2
    field = torch.randn(H, W, C, dtype=torch.float64)
    leaf = torch.randn(H, W, C, dtype=torch.float64, requires_grad=True)
    with pytest.raises(RuntimeError, match="leaf Variable"):
        fft.field_matvec_add_(leaf, field, **_ACC_KW_FIELD)
    # ... and the documented workaround (the out-of-place spelling) works.
    out = fft.field_matvec_add(leaf, field, **_ACC_KW_FIELD)
    assert out.requires_grad


def test_inplace_accumulate_bumps_version_counter():
    """``ctx.mark_dirty`` must fire so a stale save raises, not lies."""
    H, W, C = 3, 4, 2
    field = torch.randn(H, W, C, dtype=torch.float64)
    x = torch.randn(H, W, C, dtype=torch.float64, requires_grad=True)
    y = x * 1.0
    v0 = y._version
    fft.field_matvec_add_(y, field, **_ACC_KW_FIELD)
    assert y._version > v0


def test_out_of_place_accumulate_does_not_mutate_input():
    """Out-of-place must clone, never touch the caller's tensor."""
    H, W, C = 4, 5, 2
    field = torch.randn(H, W, C, dtype=torch.float64)
    base = torch.randn(H, W, C, dtype=torch.float64)
    before = base.clone()
    for fn in (fft.field_matvec_add, fft.field_matvec_sub):
        out = fn(base, field, **_ACC_KW_FIELD)
        assert torch.equal(base, before)
        assert out is not base
    for fn in (fft.field_diag_add, fft.field_diag_sub):
        out = fn(base, **_ACC_KW_FIELD)
        assert torch.equal(base, before)
        assert out is not base


def test_inplace_and_out_of_place_agree():
    """One kernel behind both spellings -> bitwise-equal results."""
    H, W, C = 4, 5, 2
    field = torch.randn(H, W, C, dtype=torch.float64)
    base = torch.randn(H, W, C, dtype=torch.float64)
    a = base.clone()
    fft.field_matvec_add_(a, field, **_ACC_KW_FIELD)
    b = fft.field_matvec_add(base, field, **_ACC_KW_FIELD)
    assert torch.equal(a, b)
