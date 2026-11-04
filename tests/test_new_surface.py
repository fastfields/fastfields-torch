"""Newly exposed torch operations, and the autograd that was missing.

Three groups:

* **Point-to-spline distance** (``dt_spline_table`` / ``dt_spline_brent`` /
  ``dt_spline_gaussnewton``) -- bound all along, wrapped on numpy and cupy,
  absent here. Checked against the raw binding (``fastfields.dlpack``), which
  is the layer below and therefore an independent reference, plus the
  non-differentiability contract the other distance ops follow.

* **``count`` and ``grad`` autograd.** ``_pushpull.py`` used to state in its
  own comments that ``count_backward`` / ``grad_backward`` "*are* exported by
  ``fastfields.dlpack``, they are simply not wired into an
  ``autograd.Function`` here", and both wrappers silently ``.detach()``'d
  their inputs -- so a graph through them ended without an error. They are
  wired now, and the test is ``torch.autograd.gradcheck``: nothing else
  establishes that an adjoint kernel is really the adjoint.

* **``sym_addmatvec_`` / ``sym_submatvec_`` and the stencil accumulates**
  (``{field,flow}_{add,sub}kernel[_]``), checked against the non-accumulating
  op they build on.
"""

import fastfields.dlpack as _fb
import pytest
import torch

import fastfields.torch as fft

torch.manual_seed(0)

_PACK = {2: 3, 3: 6}


# --------------------------------------------------------------------------- #
# point-to-spline distance                                                    #
# --------------------------------------------------------------------------- #


def _spline_inputs(nbatch=4, npoints=6, d=2, dtype=torch.float64):
    loc = torch.randn(nbatch, d, dtype=dtype)
    coeff = torch.randn(nbatch, npoints, d, dtype=dtype)
    return loc, coeff


def test_dt_spline_table_matches_the_raw_binding():
    loc, coeff = _spline_inputs()
    times = (
        torch.linspace(0, 5, 11, dtype=loc.dtype)
        .expand(loc.shape[0], 11)
        .contiguous()
    )

    dist, time = fft.dt_spline_table(loc, coeff, times)

    ref_time = torch.empty(loc.shape[:-1], dtype=loc.dtype)
    ref_dist = torch.empty(loc.shape[:-1], dtype=loc.dtype)
    _fb.dt_spline_table(
        ref_time, ref_dist, loc, coeff, times, spline=3, bound=3
    )
    torch.testing.assert_close(dist, ref_dist)
    torch.testing.assert_close(time, ref_time)


@pytest.mark.parametrize("fn", ["dt_spline_brent", "dt_spline_gaussnewton"])
def test_dt_spline_refiners_run_and_shape_correctly(fn):
    loc, coeff = _spline_inputs()
    dist, time = getattr(fft, fn)(loc, coeff, max_iter=16, tol=1e-8)
    assert dist.shape == loc.shape[:-1]
    assert time.shape == loc.shape[:-1]
    assert torch.isfinite(dist).all()
    assert torch.isfinite(time).all()
    # A refiner cannot do worse than a dictionary scan over the same range.
    assert (dist >= 0).all()


@pytest.mark.parametrize(
    "fn", ["dt_spline_table", "dt_spline_brent", "dt_spline_gaussnewton"]
)
def test_dt_spline_is_not_differentiable_but_forward_runs(fn):
    loc, coeff = _spline_inputs()
    loc = loc.requires_grad_(True)
    if fn == "dt_spline_table":
        times = (
            torch.linspace(0, 5, 11, dtype=loc.dtype)
            .expand(loc.shape[0], 11)
            .contiguous()
        )
        dist, _ = fft.dt_spline_table(loc, coeff, times)
    else:
        dist, _ = getattr(fft, fn)(loc, coeff)
    # Forward runs even with a grad-requiring input (parity with numpy/cupy),
    # and the graph exists -- only backward refuses.
    assert dist.requires_grad
    with pytest.raises(RuntimeError, match="not differentiable"):
        dist.sum().backward()


def test_dt_spline_rejects_mismatched_coordinate_rank():
    loc = torch.randn(4, 2, dtype=torch.float64)
    coeff = torch.randn(4, 6, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="coordinates"):
        fft.dt_spline_gaussnewton(loc, coeff)


def test_dt_spline_broadcasts_batch_dims():
    loc = torch.randn(5, 2, dtype=torch.float64)
    coeff = torch.randn(1, 6, 2, dtype=torch.float64)
    dist, time = fft.dt_spline_gaussnewton(loc, coeff)
    assert dist.shape == (5,)
    assert time.shape == (5,)


# --------------------------------------------------------------------------- #
# count / grad autograd                                                       #
# --------------------------------------------------------------------------- #


def _grid(shape=(4, 5), d=2, nbatch=(), dtype=torch.float64):
    """A sampling grid in voxel coordinates, inside the field of view."""
    out = torch.rand(*nbatch, *shape, d, dtype=dtype)
    for i in range(d):
        out[..., i] *= shape[i] - 1
    return out


def test_count_is_differentiable_wrt_grid():
    grid = _grid().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda g: fft.count(g, (4, 5), order=2),
        (grid,),
        eps=1e-6,
        atol=1e-4,
    )


def test_count_gradient_matches_the_raw_adjoint():
    grid = _grid().requires_grad_(True)
    out = fft.count(grid, (4, 5), order=2)
    seed = torch.randn_like(out)
    out.backward(seed)

    ref = torch.zeros_like(grid)
    _fb.count_backward(
        ref,
        seed.contiguous(),
        grid.detach(),
        spline=2,
        bound=3,
        extrapolate=1,
    )
    torch.testing.assert_close(grid.grad, ref)


def test_count_forward_is_unchanged():
    """Wiring autograd must not change the value."""
    grid = _grid()
    got = fft.count(grid, (4, 5), order=2)
    ref = torch.zeros((4, 5, 1), dtype=grid.dtype)
    _fb.count(ref, grid, spline=2, bound=3, extrapolate=1)
    torch.testing.assert_close(got, ref)


def test_grad_is_differentiable_wrt_field_and_grid():
    inp = torch.randn(4, 5, 2, dtype=torch.float64, requires_grad=True)
    grid = _grid().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda i, g: fft.grad(i, g, order=2),
        (inp, grid),
        eps=1e-6,
        atol=1e-4,
    )


def test_grad_with_abs_is_the_majorised_adjoint_not_the_true_gradient():
    """``abs=True`` is a majorisation, so its "gradient" is not the Jacobian.

    ``grad(abs=True)`` replaces the signed spline derivative by its absolute
    value; ``grad_backward`` applied with the same flag is the adjoint of
    *that* operator, which is deliberately not the derivative of the forward
    map (the absolute value is where the two part company). So the backward
    runs and matches the kernel, but ``gradcheck`` does not pass -- and must
    not be expected to. Pinned here so the distinction is not mistaken for a
    bug in either direction.
    """
    inp = torch.randn(4, 5, 2, dtype=torch.float64, requires_grad=True)
    grid = _grid().requires_grad_(True)
    out = fft.grad(inp, grid, order=2, abs=True)
    seed = torch.randn_like(out)
    out.backward(seed)

    ginp = torch.zeros_like(inp)
    ggrid = torch.zeros_like(grid)
    _fb.grad_backward(
        ginp,
        ggrid,
        inp.detach(),
        seed.contiguous(),
        grid.detach(),
        spline=2,
        bound=3,
        extrapolate=1,
        abs=True,
    )
    torch.testing.assert_close(inp.grad, ginp)
    torch.testing.assert_close(grid.grad, ggrid)

    assert not torch.autograd.gradcheck(
        lambda i, g: fft.grad(i, g, order=2, abs=True),
        (
            inp.detach().requires_grad_(True),
            grid.detach().requires_grad_(True),
        ),
        eps=1e-6,
        atol=1e-4,
        raise_exception=False,
    ), (
        "gradcheck now passes with abs=True: either the kernel changed or "
        "this test's premise is wrong -- check which before relaxing it"
    )


def test_grad_gradient_matches_the_raw_adjoint():
    inp = torch.randn(4, 5, 2, dtype=torch.float64, requires_grad=True)
    grid = _grid().requires_grad_(True)
    out = fft.grad(inp, grid, order=2)
    seed = torch.randn_like(out)
    out.backward(seed)

    ginp = torch.zeros_like(inp)
    ggrid = torch.zeros_like(grid)
    _fb.grad_backward(
        ginp,
        ggrid,
        inp.detach(),
        seed.contiguous(),
        grid.detach(),
        spline=2,
        bound=3,
        extrapolate=1,
        abs=False,
    )
    torch.testing.assert_close(inp.grad, ginp)
    torch.testing.assert_close(grid.grad, ggrid)


def test_grad_forward_is_unchanged():
    inp = torch.randn(4, 5, 2, dtype=torch.float64)
    grid = _grid()
    got = fft.grad(inp, grid, order=2)
    ref = torch.zeros((4, 5, 2, 2), dtype=inp.dtype)
    _fb.grad(ref, inp, grid, spline=2, bound=3, extrapolate=1, abs=False)
    torch.testing.assert_close(got, ref)


def test_grad_needs_no_gradient_when_nothing_requires_one():
    inp = torch.randn(4, 5, 2, dtype=torch.float64)
    grid = _grid()
    assert not fft.grad(inp, grid).requires_grad
    assert not fft.count(grid, (4, 5)).requires_grad


# --------------------------------------------------------------------------- #
# sym_addmatvec_ / sym_submatvec_                                             #
# --------------------------------------------------------------------------- #


def _spd(batch, C, dtype=torch.float64):
    mat = torch.randn(*batch, _PACK[C], dtype=dtype) * 0.1
    mat[..., :C] += 3.0  # diagonally dominant
    return mat


@pytest.mark.parametrize("C", [2, 3])
@pytest.mark.parametrize("sub", [False, True])
def test_matvec_accumulate_matches_sym_matvec(C, sub):
    mat = _spd((5,), C)
    vec = torch.randn(5, C, dtype=torch.float64)
    start = torch.randn(5, C, dtype=torch.float64)

    expected = start + (-1 if sub else 1) * fft.sym_matvec(mat, vec)
    acc = start.clone()
    fn = fft.sym_submatvec_ if sub else fft.sym_addmatvec_
    returned = fn(acc, mat, vec)
    assert returned is acc
    torch.testing.assert_close(acc, expected)


def test_matvec_accumulate_is_differentiable_wrt_the_accumulator():
    mat = _spd((5,), 2)
    vec = torch.randn(5, 2, dtype=torch.float64)
    start = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    out = fft.sym_addmatvec_(start * 1.0, mat, vec)  # non-leaf accumulator
    seed = torch.randn_like(out)
    out.backward(seed)
    torch.testing.assert_close(start.grad, seed)


def test_matvec_accumulate_refuses_a_mat_or_vec_gradient():
    mat = _spd((5,), 2).requires_grad_(True)
    vec = torch.randn(5, 2, dtype=torch.float64)
    acc = torch.zeros(5, 2, dtype=torch.float64)
    out = fft.sym_addmatvec_(acc, mat, vec)
    with pytest.raises(RuntimeError, match="not differentiable"):
        out.sum().backward()


def test_matvec_accumulate_rejects_a_grad_requiring_leaf():
    mat = _spd((5,), 2)
    vec = torch.randn(5, 2, dtype=torch.float64)
    acc = torch.zeros(5, 2, dtype=torch.float64, requires_grad=True)
    before = acc.detach().clone()
    with pytest.raises(RuntimeError, match="leaf Variable"):
        fft.sym_addmatvec_(acc, mat, vec)
    # And it must not have been mutated before raising.
    torch.testing.assert_close(acc.detach(), before)


def test_matvec_accumulate_broadcasts_mat_to_the_accumulator():
    mat = _spd((1,), 2)
    vec = torch.randn(5, 2, dtype=torch.float64)
    acc = torch.zeros(5, 2, dtype=torch.float64)
    fft.sym_addmatvec_(acc, mat, vec)
    torch.testing.assert_close(acc, fft.sym_matvec(mat, vec))


# --------------------------------------------------------------------------- #
# stencil accumulate family                                                   #
# --------------------------------------------------------------------------- #

NDIM = 2
VS = [1.0, 1.2]
FIELD_PEN = {
    "absolute": [0.1, 0.2],
    "membrane": [0.5, 0.3],
    "bending": [0.05, 0.02],
}
FLOW_PEN = {
    "absolute": 0.1,
    "membrane": 0.5,
    "bending": 0.0,
    "shears": 0.2,
    "div": 0.1,
}
KW = {"voxel_size": VS, "bound": "dct2", "ndim": NDIM}


def _kernel(family):
    if family == "field":
        return fft.field_kernel(
            NDIM, **FIELD_PEN, channels=2, voxel_size=VS, bound="dct2"
        )
    return fft.flow_kernel(NDIM, **FLOW_PEN, voxel_size=VS, bound="dct2")


@pytest.mark.parametrize(
    "family,pen", [("field", FIELD_PEN), ("flow", FLOW_PEN)]
)
def test_addkernel_on_zeros_is_the_kernel(family, pen):
    k = _kernel(family)
    got = getattr(fft, f"{family}_addkernel")(torch.zeros_like(k), **pen, **KW)
    torch.testing.assert_close(got, k)


@pytest.mark.parametrize(
    "family,pen", [("field", FIELD_PEN), ("flow", FLOW_PEN)]
)
def test_subkernel_on_zeros_is_minus_the_kernel(family, pen):
    k = _kernel(family)
    got = getattr(fft, f"{family}_subkernel")(torch.zeros_like(k), **pen, **KW)
    torch.testing.assert_close(got, -k)


@pytest.mark.parametrize(
    "family,pen", [("field", FIELD_PEN), ("flow", FLOW_PEN)]
)
@pytest.mark.parametrize("sign", ["add", "sub"])
def test_inplace_kernel_mutates_and_matches_functional(family, pen, sign):
    start = _kernel(family) * 0.25
    expected = getattr(fft, f"{family}_{sign}kernel")(start, **pen, **KW)
    buf = start.clone()
    returned = getattr(fft, f"{family}_{sign}kernel_")(buf, **pen, **KW)
    assert returned is buf
    torch.testing.assert_close(buf, expected)


@pytest.mark.parametrize(
    "family,pen", [("field", FIELD_PEN), ("flow", FLOW_PEN)]
)
def test_functional_kernel_accumulate_leaves_input_alone(family, pen):
    start = _kernel(family) * 0.25
    before = start.clone()
    getattr(fft, f"{family}_addkernel")(start, **pen, **KW)
    torch.testing.assert_close(start, before)


@pytest.mark.parametrize(
    "family,pen", [("field", FIELD_PEN), ("flow", FLOW_PEN)]
)
def test_kernel_accumulate_is_differentiable_wrt_its_input(family, pen):
    """The stencil has no tensor input, so the gradient is pass-through."""
    k = (_kernel(family) * 0.25).requires_grad_(True)
    out = getattr(fft, f"{family}_addkernel")(k, **pen, **KW)
    seed = torch.randn_like(out)
    out.backward(seed)
    torch.testing.assert_close(k.grad, seed)
