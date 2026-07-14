# -*- coding: utf-8 -*-
"""Shared batch-calculator predicates for the dispatcher layer.

Single home for the duck-typed tests that were copy-pasted across the batched
optimizer / saddle / IRC modules (``_is_batch_calc``, the
``_COUPLED_BATCH_CALC_NAMES`` class-name set, ad-hoc ``coupling_mode`` sniffing).
Import from here instead of re-rolling the predicate. Kept dependency-free (no
torch) so importing it is cheap.

Contract mirrors ``BatchCalcABC`` (calculator/batch_calculator_base.py):
  * a batch calc exposes ``prepare()`` + ``get_ef_gpu()`` (the batched EF path);
  * a *coupled* batch calc declares ``SUPPORTS_COUPLING = True`` (MACE-POL,
    charge-equilibration AIMNet2) or carries a ``coupling_mode`` knob -> perturbing
    one molecule leaks into the others, so a block-diagonal cross-reaction batch
    is numerically unsafe (gate B>1 / cross-batch reuse on ``cross_batch_safe``).
"""
from __future__ import annotations


def is_batch_calc(calc) -> bool:
    """Duck-typed: a batched calculator exposes prepare() + get_ef_gpu()."""
    return (calc is not None
            and callable(getattr(calc, "prepare", None))
            and callable(getattr(calc, "get_ef_gpu", None)))


def is_coupled(calc) -> bool:
    """True if perturbing one molecule leaks into the others within a batch.

    Declarative capability flag ``SUPPORTS_COUPLING`` (BatchCalcABC subclasses)
    OR the presence of a ``coupling_mode`` knob (MACE-POL). Replaces the
    hard-coded ``_COUPLED_BATCH_CALC_NAMES`` class-name set.
    """
    return bool(getattr(calc, "SUPPORTS_COUPLING", False)) or hasattr(calc, "coupling_mode")


def cross_batch_safe(calc) -> bool:
    """True if a B>1 block-diagonal batch of DIFFERENT structures is safe.

    i.e. not coupled -> perturbing one structure cannot leak into the others.
    """
    return not is_coupled(calc)


# ---------------------------------------------------------------------------
# Implicit-solvent capability gate (mirrors the PBC fail-fast in BatchCalcABC)
# ---------------------------------------------------------------------------
_NONE_TOKENS = ("none", "", "false", "0", "no", "off", "null", "vacuum", "gas")


def implicit_solvent_requested(params) -> bool:
    """True when the job asked for implicit solvation: ``#solv(method=gbsa, implicit=<s>)``.

    Reads the SAME two keys the single-structure path reads (engine._mlp_initiator:
    ``commandcontrol['solv']['method']`` + ``['solv']['implicit']``). Explicit
    solvation (``#solv(explicit=...)``) is a different feature -- it adds real solvent
    atoms to the structure, so the batched path handles it correctly and it is NOT
    gated here.
    """
    if params is None or not hasattr(params, "get"):
        return False
    solv = params.get("solv")
    if not isinstance(solv, dict):
        return False
    method = str(solv.get("method") or "none").strip().lower()
    implicit = str(solv.get("implicit") or "none").strip().lower()
    return method not in _NONE_TOKENS and implicit not in _NONE_TOKENS


def supports_implicit_solvent(calc) -> bool:
    """True if this batched calculator actually APPLIES an implicit-solvent correction.

    Declarative capability flag ``SUPPORTS_IMPLICIT_SOLVENT`` (BatchCalcABC). Every
    batch backend shipped today is gas-phase-only -> False.
    """
    return bool(getattr(calc, "SUPPORTS_IMPLICIT_SOLVENT", False))


def reject_batched_implicit_solvent(params, calc, *, context="batched job"):
    """Fail fast when a job requests implicit solvent but the batched calc ignores it.

    THE BUG THIS CLOSES: the batched calculators apply no solvent correction at all,
    yet the batched acquisition path never looked at ``#solv``. A multi-structure
    OPT / TS / IRC / SCAN job with ``#solv(method=gbsa, implicit=water)`` was therefore
    silently evaluated in the GAS PHASE -- measured 614.34 Ha away from the solvated
    single-structure oracle and bit-equal (8.76e-8 Ha) to the gas-phase oracle. Worse,
    it BYPASSED the single-structure safety gate
    (calculator_base.reject_implicit_solvent_derivatives), which raises
    NotImplementedError for "implicit solvent + derivatives" because the legacy GB-polar
    correction is energy-only. Silently returning gas-phase numbers for a solvated job
    is the exact failure mode this library's batched==serial contract forbids.

    Capability-gated, not hard-coded: a batch backend that genuinely applies the solvent
    correction declares ``SUPPORTS_IMPLICIT_SOLVENT = True`` and passes through.
    Returns ``calc`` so callers can write ``return reject_batched_implicit_solvent(...)``.
    """
    if not implicit_solvent_requested(params):
        return calc
    if supports_implicit_solvent(calc):
        return calc
    solv = params.get("solv") or {}
    raise NotImplementedError(
        f"Implicit solvation (#solv(method={solv.get('method')!r}, "
        f"implicit={solv.get('implicit')!r})) is not supported on the GPU-batched path: "
        f"{type(calc).__name__} applies NO solvent correction, so this {context} would "
        "silently return GAS-PHASE energies/forces. Run the structures one at a time "
        "(the single-structure path applies the correction, and rejects solvent+forces "
        "explicitly), or drop #solv to run the batch in the gas phase on purpose."
    )
