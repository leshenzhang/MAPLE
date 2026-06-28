"""
GROMACS-grompp-style physical preflight + runtime box-size guard for ML-MD.

Why this exists
---------------
An MLIP has a *finite receptive field* (the graph edge cutoff ``r_max``; multi-
layer message passing makes the effective range a small multiple of it).  Under
periodic boundary conditions the calculator builds a minimum-image neighbour
list, so if the shortest periodic box width drops below ``2 * r_max`` an atom
starts to "see" its own periodic image.  The forces / virial then become
*silently wrong* for the condensed phase, with no error from the engine.

GROMACS ``mdrun`` aborts fatally in exactly this regime (``rlist`` vs. the
minimum box dimension).  MAPLE previously did not check at all: the engine only
wrapped coordinates and the NPT barostat shrank the cell without ever testing
the result.  This module restores the GROMACS behaviour:

* ``check_box_size``      -- minimum-image box-width guard (setup + NPT runtime).
* ``composition_sanity``  -- cheap grompp-style sanity warnings (overlapping
                             atoms, unsupported elements).

Decision metric
---------------
We use the **perpendicular width** of each periodic lattice direction
(``V / |a_j x a_k|``), i.e. the true inter-plane distance, exactly like GROMACS.
For an orthorhombic cell this equals the lattice-vector norm; for a skewed
(triclinic) cell it is the rigorous and stricter quantity.  Only directions with
``atoms.pbc[i] == True`` are tested; non-periodic directions (vacuum of a slab
or a gas-phase molecule) are ignored.

Modes (mdp key ``box_check``)
-----------------------------
* ``strict`` (default) -- fatal abort on violation (GROMACS behaviour).
* ``warn``             -- print a warning and continue.
* ``off``              -- skip the check entirely.

The guard self-skips (returns silently) when:
* the calculator does not use PBC (``SUPPORTS_PBC`` is False / absent), since a
  non-periodic neighbour list cannot create image self-interaction; or
* no ``r_max`` can be read from the calculator; or
* the system has no periodic direction.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


_VALID_MODES = ("strict", "warn", "off")
_AXIS_NAMES = ("a", "b", "c")


def _normalize_mode(mode: Optional[str]) -> str:
    """Coerce a user-supplied box_check value to one of strict|warn|off."""
    if mode is None:
        return "strict"
    m = str(mode).strip().lower()
    if m in ("", "true", "1", "on", "yes"):
        return "strict"
    if m in ("false", "0", "no"):
        return "off"
    if m not in _VALID_MODES:
        # Unknown value -> fail safe to strict (never silently disable a guard).
        return "strict"
    return m


def get_calculator_r_max(calc) -> Optional[float]:
    """Return the MLIP receptive-field radius (Angstrom) *iff* the calculator
    actually uses periodic boundaries, else None.

    A non-PBC calculator builds a no-PBC neighbour list (no minimum image), so
    box-image self-interaction cannot occur and the guard must skip it.
    """
    if calc is None:
        return None
    if not bool(getattr(calc, "SUPPORTS_PBC", False)):
        return None
    r_max = getattr(calc, "r_max", None)
    if r_max is None:
        return None
    try:
        r_max = float(r_max)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(r_max) or r_max <= 0.0:
        return None
    return r_max


def box_perpendicular_widths(cell) -> np.ndarray:
    """Perpendicular width of each lattice direction: width_i = V / |a_j x a_k|.

    This is the inter-plane distance that bounds the minimum-image convention
    (GROMACS uses the same quantity).  Returns a length-3 array (a, b, c order).
    Degenerate / zero-area directions yield ``inf`` (treated as "no constraint").
    """
    h = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    volume = abs(float(np.linalg.det(h)))
    widths = np.full(3, np.inf, dtype=np.float64)
    # width along i is V / area of the parallelogram spanned by the other two.
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        cross = np.cross(h[j], h[k])
        area = float(np.linalg.norm(cross))
        if area > 1e-12 and volume > 1e-12:
            widths[i] = volume / area
    return widths


def evaluate_box(atoms, r_max: float) -> Tuple[bool, dict]:
    """Compute the minimum-image box check for ``atoms`` given ``r_max``.

    Returns ``(ok, info)`` where ``ok`` is True when every *periodic* direction
    has perpendicular width >= 2*r_max.  ``info`` carries the named offender for
    a precise message.
    """
    cell = atoms.get_cell()
    pbc = np.asarray(atoms.pbc, dtype=bool).reshape(3)
    widths = box_perpendicular_widths(np.asarray(cell))
    vec_norms = np.linalg.norm(np.asarray(cell, dtype=np.float64).reshape(3, 3), axis=1)
    required = 2.0 * r_max

    periodic_idx = [i for i in range(3) if pbc[i]]
    info = {
        "r_max": r_max,
        "required": required,
        "widths": widths,
        "vec_norms": vec_norms,
        "pbc": pbc,
        "periodic_idx": periodic_idx,
    }
    if not periodic_idx:
        info["ok"] = True
        return True, info

    # Worst (smallest perpendicular width) periodic direction.
    worst = min(periodic_idx, key=lambda i: widths[i])
    info["worst_axis"] = worst
    info["worst_width"] = float(widths[worst])
    info["worst_vec_norm"] = float(vec_norms[worst])
    ok = bool(widths[worst] >= required)
    info["ok"] = ok
    return ok, info


def _format_violation(info: dict, calc_name: str, context: str, severity: str = "FATAL") -> str:
    a = info["worst_axis"]
    widths = info["widths"]
    pbc = info["pbc"]
    width_str = ", ".join(
        f"{_AXIS_NAMES[i]}={widths[i]:.3f}" if np.isfinite(widths[i]) else f"{_AXIS_NAMES[i]}=inf"
        for i in range(3)
    )
    pbc_str = ", ".join(f"{_AXIS_NAMES[i]}={bool(pbc[i])}" for i in range(3))
    return (
        f"MAPLE box preflight {severity} [{context}]: periodic box too small for the "
        f"MLIP receptive field.\n"
        f"  Calculator '{calc_name}' has r_max = {info['r_max']:.3f} Angstrom "
        f"(graph edge cutoff).\n"
        f"  Minimum-image safety requires every periodic box width >= "
        f"2*r_max = {info['required']:.3f} Angstrom.\n"
        f"  Box vector '{_AXIS_NAMES[a]}': perpendicular width = "
        f"{info['worst_width']:.3f} Angstrom (vector norm "
        f"{info['worst_vec_norm']:.3f}) < required {info['required']:.3f} "
        f"Angstrom  ->  VIOLATION.\n"
        f"  Perpendicular widths [{width_str}] Angstrom; pbc [{pbc_str}].\n"
        f"  A box shorter than 2*r_max lets an atom interact with its own periodic "
        f"image, giving SILENTLY WRONG condensed-phase forces/pressure. "
        f"Enlarge the cell (build a supercell) so every periodic width >= "
        f"{info['required']:.3f} Angstrom, or set the mdp key "
        f"'box_check = warn' to override (NOT recommended for production)."
    )


def check_box_size(
    atoms,
    calc=None,
    mode: Optional[str] = "strict",
    *,
    context: str = "MD preflight",
    warn: Optional[Callable[[str], None]] = None,
) -> bool:
    """GROMACS-style minimum-image box guard.

    Returns True when the box is acceptable (or the check is skipped/disabled).
    In ``strict`` mode a violation raises ``RuntimeError`` (fatal abort, mirroring
    GROMACS mdrun).  In ``warn`` mode it prints a warning and returns False.

    Parameters
    ----------
    atoms : ase.Atoms
        System under test (must expose ``cell`` and ``pbc``).
    calc : ASE calculator, optional
        Defaults to ``atoms.calc``.  Used to read ``SUPPORTS_PBC`` and ``r_max``.
    mode : {'strict', 'warn', 'off'}
        Guard severity.  Defaults to 'strict'.
    context : str
        Free-text label woven into the message ("NPT setup preflight",
        "NPT runtime step 1234 (after barostat rescale)", ...).
    warn : callable(str), optional
        Sink for the warning text in 'warn' mode (defaults to ``print``).
    """
    mode = _normalize_mode(mode)
    if mode == "off":
        return True
    if calc is None:
        calc = getattr(atoms, "calc", None)

    r_max = get_calculator_r_max(calc)
    if r_max is None:
        # Non-PBC calculator or no r_max -> minimum-image self-interaction is not
        # possible / not determinable; skip silently.
        return True

    ok, info = evaluate_box(atoms, r_max)
    if ok:
        return True

    calc_name = type(calc).__name__
    if mode == "warn":
        (warn or print)("*** WARNING: " + _format_violation(info, calc_name, context, "WARNING"))
        return False
    # strict
    raise RuntimeError(_format_violation(info, calc_name, context, "FATAL"))


def composition_sanity(
    atoms,
    calc=None,
    mode: Optional[str] = "strict",
    *,
    context: str = "MD preflight",
    warn: Optional[Callable[[str], None]] = None,
    min_distance: float = 0.5,
) -> None:
    """Cheap grompp-style composition / geometry sanity warnings.

    These are *warnings only* (never fatal, even in strict mode): they flag the
    obviously-broken setups GROMACS would also complain about, without trying to
    be a full force-field type checker.

    * Overlapping atoms: shortest interatomic distance < ``min_distance`` Angstrom
      (PBC-aware when the cell is periodic) -> energy/force will blow up.
    * Unsupported element: an atomic number not in the calculator's declared
      ``atomic_numbers`` set (only checked when the calculator exposes a finite
      supported set; foundation models that support the whole periodic table are
      skipped).
    """
    mode = _normalize_mode(mode)
    if mode == "off":
        return
    if calc is None:
        calc = getattr(atoms, "calc", None)
    emit = warn or print

    n = len(atoms)
    if n == 0:
        emit(f"*** WARNING [{context}]: system has 0 atoms.")
        return

    # --- unsupported elements (only if the calc declares a finite element set) ---
    supported = getattr(calc, "atomic_numbers", None)
    if supported is not None:
        try:
            supported_set = {int(z) for z in supported}
        except (TypeError, ValueError):
            supported_set = None
        if supported_set:
            present = {int(z) for z in atoms.get_atomic_numbers()}
            missing = sorted(present - supported_set)
            if missing:
                emit(
                    f"*** WARNING [{context}]: atomic numbers {missing} are not in "
                    f"the calculator '{type(calc).__name__}' supported set "
                    f"(sorted) {sorted(supported_set)[:12]}{'...' if len(supported_set) > 12 else ''}. "
                    f"Predictions for these elements are extrapolative / undefined."
                )

    # --- overlapping atoms (shortest interatomic distance) ---
    if n >= 2:
        try:
            pbc = bool(np.any(atoms.pbc))
            # ase returns an (N, N) distance matrix; mic=True respects the cell.
            d = atoms.get_all_distances(mic=pbc)
            iu = np.triu_indices(n, k=1)
            dmin = float(np.min(d[iu]))
            if dmin < float(min_distance):
                ia, ib = (int(iu[0][np.argmin(d[iu])]), int(iu[1][np.argmin(d[iu])]))
                emit(
                    f"*** WARNING [{context}]: shortest interatomic distance is "
                    f"{dmin:.3f} Angstrom (atoms {ia}-{ib}, < {min_distance} Angstrom). "
                    f"Overlapping atoms typically blow up the MLIP energy/force on "
                    f"step 1; check the initial geometry."
                )
        except Exception:
            # Distance computation is best-effort; never let a sanity warning
            # break the run.
            pass
