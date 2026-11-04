"""This wrapper must cover the binding, or say why not.

``fastfields.torch`` wraps ``fastfields.dlpack``, which in turn binds every
public entry point of the C++ hub (enforced by
``fastfields-dlpack/tests/test_api_coverage.py``). So "which hub operations
can a torch user reach?" reduces to "which ``fastfields.dlpack`` names does
this package wrap?" -- a question answerable here, without the C++ tree, by
comparing the two ``__all__`` lists.

An operation may be absent on purpose, and on this backend several are: the
adjoints are *used* rather than re-exported, because the supported way to
reach a gradient in torch is ``.backward()``. Those live in
:data:`DELIBERATE` with the reason, and the package docstring repeats the
list for users. The point of the test is the other direction: an op that goes
missing by accident -- a new binding nobody wrapped, or one dropped in a
refactor -- fails here instead of being found by a user.
"""

from __future__ import annotations

import fastfields.dlpack as ffb
import pytest

import fastfields.torch as fft

#: Bindings this wrapper deliberately does not expose -> why.
DELIBERATE = {
    **{
        op: (
            "the adjoint behind the autograd.Function of the corresponding "
            "forward op; reach it with .backward(), or raw via "
            "fastfields.dlpack.%s" % op
        )
        for op in (
            "pull_backward",
            "push_backward",
            "count_backward",
            "grad_backward",
            "sym_matvec_backward",
        )
    },
    **{
        op: (
            "in-place preconditioner solve; needs a decision about autograd "
            "and a mutated buffer (API_CONTRACT.md, 'In-place policy') "
            "before it is wrapped. The out-of-place %s is exposed."
            % op.rstrip("_")
        )
        for op in ("field_precond_", "flow_precond_")
    },
    "field_cg": (
        "iterative solver; its gradient is a solve with the same operator, "
        "which is a real autograd.Function rather than a mechanical wrapper. "
        "field_relax is the in-place solver exposed today, and "
        "fastfields.numpy.field_cg exists for the non-autograd case."
    ),
}

#: Names in ``fastfields.dlpack.__all__`` that are not operations.
NON_OPS = {"__version__", "backend"}


def _binding_ops() -> set[str]:
    return {n for n in ffb.__all__ if n not in NON_OPS}


def _covered(op: str) -> bool:
    """Does this wrapper expose ``op``, under either spelling?

    The hub's accumulate primitives exist only in place
    (``field_adddiag_``) and this wrapper adds an out-of-place spelling that
    calls the same binding; ``dt_euclidean`` exists only out of place and the
    wrapper adds ``dt_euclidean_``. Either counts -- except where the binding
    itself declares both spellings (``sym_solve``/``sym_solve_``,
    ``{field,flow}_precond``/``_precond_``), which are different operations
    and are matched exactly. Without that exception ``field_precond`` would
    make ``field_precond_`` look covered.
    """
    names = set(fft.__all__)
    if op in names:
        return True
    alias = op[:-1] if op.endswith("_") else op + "_"
    if alias in _binding_ops():
        return False
    return alias in names


def test_no_accidental_gap():
    missing = sorted(
        op
        for op in _binding_ops()
        if not _covered(op) and op not in DELIBERATE
    )
    assert not missing, (
        "fastfields.dlpack operations this wrapper neither exposes nor "
        f"documents as deliberately absent: {missing}. Wrap them, or add "
        "them to DELIBERATE in this file (and to the 'What this wrapper "
        "deliberately does not expose' section of the package docstring) "
        "with the reason."
    )


def test_deliberate_gaps_are_really_absent():
    stale = sorted(op for op in DELIBERATE if _covered(op))
    assert not stale, (
        f"listed as deliberately absent but now exposed: {stale}; drop them "
        "from DELIBERATE and from the package docstring."
    )


def test_deliberate_gaps_still_exist_in_the_binding():
    unknown = sorted(op for op in DELIBERATE if op not in _binding_ops())
    assert not unknown, (
        f"DELIBERATE names no longer in fastfields.dlpack.__all__: {unknown}"
    )


def test_docstring_lists_every_deliberate_gap():
    doc = fft.__doc__ or ""
    for op in DELIBERATE:
        assert op in doc, (
            f"{op} is deliberately absent but the package docstring does not "
            "mention it; a gap the user cannot read about is just a gap"
        )


@pytest.mark.parametrize("name", sorted(fft.__all__))
def test_dunder_all_resolves(name):
    assert hasattr(fft, name), f"__all__ names {name!r}, which does not exist"
