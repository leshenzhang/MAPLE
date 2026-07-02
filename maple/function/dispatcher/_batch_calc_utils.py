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
