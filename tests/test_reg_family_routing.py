"""Guard: every reg wrapper must reach its *own* family's C symbol.

The ``field_*`` and ``flow_*`` regularisers are different operators.
``field_*`` applies a per-channel scalar penalty (channels are independent);
``flow_*`` treats the last axis as a vector displacement in *voxel* units and
additionally offers the cross-channel Lame terms (``shears``/``div``).

Upstream ``jitfields`` shipped a copy-paste bug of exactly this shape:
``jitfields.field_kernel_add`` / ``field_kernel_add_`` (and the ``_sub``
wrappers that delegate to them) call the low-level ``flow_kernel`` binding
instead of ``field_kernel``.  fastfields does not reproduce it today, but the
kernel accumulate wrappers are still being ported, so these tests pin the
routing down mechanically -- including through the autograd backward passes,
where a mis-wired adjoint would otherwise only show up as a bad gradient.

Why this needs to be a *routing* test and not only a numeric one: with an
isotropic ``voxel_size`` and no Lame terms the two families produce identical
output (see ``test_field_and_flow_coincide_when_isotropic``), so a swap is
completely silent under default arguments.
"""

import fastfields.dlpack as _fb
import pytest
import torch

import fastfields.torch as ff

# Anisotropic on purpose: this is what makes field != flow observable.
VS = [1.0, 2.0, 3.0]
NDIM = 3
SHAPE = (6, 6, 6, NDIM)


def _x():
    n = SHAPE[0] * SHAPE[1] * SHAPE[2] * SHAPE[3]
    return torch.arange(n, dtype=torch.float64).reshape(SHAPE) / n


def _hes():
    return torch.ones((6, 6, 6, NDIM * (NDIM + 1) // 2), dtype=torch.float64)


# Every public field_*/flow_* wrapper, with a call that reaches the C layer.
_CASES = {
    "field_matvec": lambda: ff.field_matvec(
        _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "field_kernel": lambda: ff.field_kernel(
        NDIM, membrane=1.0, channels=NDIM, voxel_size=VS
    ),
    "field_diag": lambda: ff.field_diag(
        SHAPE, membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "field_precond": lambda: ff.field_precond(
        _hes(), _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "field_forward": lambda: ff.field_forward(
        _hes(), _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "field_addmatvec": lambda: ff.field_addmatvec(
        _x(), _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "field_submatvec": lambda: ff.field_submatvec(
        _x(), _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "field_addmatvec_": lambda: ff.field_addmatvec_(
        _x(), _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "field_submatvec_": lambda: ff.field_submatvec_(
        _x(), _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "field_adddiag": lambda: ff.field_adddiag(
        _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "field_subdiag": lambda: ff.field_subdiag(
        _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "field_adddiag_": lambda: ff.field_adddiag_(
        _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "field_subdiag_": lambda: ff.field_subdiag_(
        _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "flow_matvec": lambda: ff.flow_matvec(
        _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "flow_kernel": lambda: ff.flow_kernel(NDIM, membrane=1.0, voxel_size=VS),
    "flow_diag": lambda: ff.flow_diag(
        SHAPE, membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "flow_relax": lambda: ff.flow_relax(
        _x(),
        _hes(),
        torch.ones(SHAPE, dtype=torch.float64),
        membrane=1.0,
        voxel_size=VS,
        ndim=NDIM,
    ),
    "flow_precond": lambda: ff.flow_precond(
        _hes(), _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "flow_forward": lambda: ff.flow_forward(
        _hes(), _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "flow_addmatvec": lambda: ff.flow_addmatvec(
        _x(), _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "flow_submatvec": lambda: ff.flow_submatvec(
        _x(), _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "flow_addmatvec_": lambda: ff.flow_addmatvec_(
        _x(), _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "flow_submatvec_": lambda: ff.flow_submatvec_(
        _x(), _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "flow_adddiag": lambda: ff.flow_adddiag(
        _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "flow_subdiag": lambda: ff.flow_subdiag(
        _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "flow_adddiag_": lambda: ff.flow_adddiag_(
        _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
    "flow_subdiag_": lambda: ff.flow_subdiag_(
        _x(), membrane=1.0, voxel_size=VS, ndim=NDIM
    ),
}


def _public_reg_names():
    return {n for n in ff.__all__ if n.startswith(("field_", "flow_"))}


def test_every_public_reg_wrapper_has_a_routing_case():
    """A newly exposed field_*/flow_* wrapper must be added to _CASES.

    This is the forcing function: it fails when someone exposes e.g.
    ``field_kernel_add`` without pinning down which C symbol it routes to.
    """
    assert _public_reg_names() == set(_CASES)


def _spy(monkeypatch):
    """Instrument the dlpack layer; return the list that records calls."""
    seen = []
    names = [n for n in dir(_fb) if n.startswith(("field_", "flow_"))]
    for name in names:
        orig = getattr(_fb, name)

        def spy(*args, _n=name, _o=orig, **kwargs):
            seen.append(_n)
            return _o(*args, **kwargs)

        monkeypatch.setattr(_fb, name, spy)
    return seen


@pytest.mark.parametrize("name", sorted(_CASES))
def test_wrapper_only_calls_its_own_family(monkeypatch, name):
    family = name.split("_")[0]
    seen = _spy(monkeypatch)
    _CASES[name]()
    assert seen, f"{name} reached no fastfields.dlpack symbol"
    wrong = [s for s in seen if not s.startswith(family + "_")]
    assert not wrong, (
        f"{name} is wired to the wrong regulariser family: it called "
        f"{wrong} but must only call {family}_* symbols"
    )


@pytest.mark.parametrize(
    "name", ["field_matvec", "field_forward", "flow_matvec", "flow_forward"]
)
def test_autograd_backward_stays_in_its_family(monkeypatch, name):
    """The adjoint must use the same family as the forward.

    These operators are self-adjoint, so backward re-applies the *same*
    matvec; a copy-pasted backward pointing at the other family would still
    produce plausible-looking numbers.
    """
    family = name.split("_")[0]
    seen = _spy(monkeypatch)
    x = torch.rand(4, 4, 4, NDIM, dtype=torch.float64, requires_grad=True)
    if name.endswith("_forward"):
        mat = torch.ones(4, 4, 4, 6, dtype=torch.float64)
        out = getattr(ff, name)(mat, x, membrane=1.0, voxel_size=VS, ndim=NDIM)
    else:
        out = getattr(ff, name)(x, membrane=1.0, voxel_size=VS, ndim=NDIM)
    seen.clear()
    out.sum().backward()
    assert seen, f"{name} backward reached no fastfields.dlpack symbol"
    wrong = [s for s in seen if not s.startswith(family + "_")]
    assert not wrong, (
        f"{name} backward is wired to the wrong regulariser family: "
        f"it called {wrong} but must only call {family}_* symbols"
    )


@pytest.mark.parametrize(
    "kw", [{"absolute": 1.0}, {"membrane": 1.0}, {"bending": 1.0}]
)
def test_field_and_flow_differ_when_anisotropic(kw):
    """The two families are genuinely different operators.

    Without this, the routing test above would be guarding a distinction
    that does not exist.
    """
    a = ff.field_matvec(_x(), voxel_size=VS, ndim=NDIM, **kw)
    b = ff.flow_matvec(_x(), voxel_size=VS, ndim=NDIM, **kw)
    assert not torch.allclose(a, b)

    ka = ff.field_kernel(NDIM, channels=NDIM, voxel_size=VS, **kw)
    kb = ff.flow_kernel(NDIM, voxel_size=VS, **kw)
    assert ka.shape == kb.shape  # a swap is not caught by shape alone
    assert not torch.allclose(ka, kb)


@pytest.mark.parametrize(
    "kw", [{"absolute": 1.0}, {"membrane": 1.0}, {"bending": 1.0}]
)
def test_field_and_flow_coincide_when_isotropic(kw):
    """Documents *why* the guards above must use an anisotropic voxel size.

    At ``voxel_size=1`` and without Lame terms the flow regulariser reduces to
    the per-channel field regulariser, so a field/flow swap produces identical
    numbers and is undetectable. If this ever starts failing, the two families
    have diverged further and the note above can be relaxed.
    """
    a = ff.field_matvec(_x(), ndim=NDIM, **kw)
    b = ff.flow_matvec(_x(), ndim=NDIM, **kw)
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_flow_lame_has_no_field_equivalent():
    """The Lame terms are flow-only: they change the operator and the shape."""
    plain = ff.flow_kernel(NDIM, membrane=1.0, voxel_size=VS)
    lame = ff.flow_kernel(NDIM, membrane=1.0, shears=1.0, voxel_size=VS)
    # cross-channel matrix stencil, not the per-channel vector stencil
    assert lame.dim() == plain.dim() + 1
    a = ff.field_matvec(_x(), membrane=1.0, voxel_size=VS, ndim=NDIM)
    b = ff.flow_matvec(
        _x(), membrane=1.0, shears=1.0, voxel_size=VS, ndim=NDIM
    )
    assert not torch.allclose(a, b)
