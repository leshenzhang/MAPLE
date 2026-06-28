"""Enhanced-sampling bias layer for MAPLE MD.

One injection point for every ensemble: ``maybe_wrap_bias`` replaces
``atoms.calc`` with a bias-aware wrapper when an MD input requests PLUMED or
Colvars, so NVE/NVT/NPT pick up metadynamics / OPES / umbrella / ABF / eABF
with a single line per ensemble and zero integrator changes.

Enable from an mdp/input by setting one of::

    plumed  = plumed.dat        # primary, validated backend (any PLUMED CV/bias)
    colvars = colvars.in        # Colvars backend (eABF / multiple-walker ABF)

The value is a path to the bias-definition file (or an inline script). The two
are mutually exclusive; PLUMED wins if both are set. Umbrella-sampling window
generation + WHAM/MBAR post-processing live in :mod:`.umbrella`.
"""

from .plumed_calc import PlumedCalculator
from .colvars_calc import ColvarsCalculator
from .posres_calc import PosresCalculator, posres_enabled
from . import umbrella

__all__ = ["maybe_wrap_bias", "PlumedCalculator", "ColvarsCalculator",
           "PosresCalculator", "posres_enabled", "umbrella"]


def maybe_wrap_bias(atoms, params, output):
    """Wrap ``atoms.calc`` with a bias calculator if the input asks for one.

    Reads ``params.plumed`` / ``params.colvars`` (empty ⇒ no bias) plus
    ``params.timestep`` (fs) and ``params.temperature`` (K). Mutates
    ``atoms.calc`` in place and returns it (unchanged if no bias requested).
    Safe no-op on params dataclasses without the fields.
    """
    plumed_in = getattr(params, "plumed", "") or ""
    colvars_in = getattr(params, "colvars", "") or ""
    posres_in = getattr(params, "posres", "")
    want_posres = posres_enabled(posres_in)
    if not plumed_in and not colvars_in and not want_posres:
        return atoms.calc

    inner = atoms.calc
    if inner is None:
        raise ValueError("maybe_wrap_bias: atoms.calc is None — set the MLIP "
                         "calculator before applying a bias/restraint.")
    timestep_fs = float(getattr(params, "timestep", 0.5))
    temperature = float(getattr(params, "temperature", 300.0))
    restart_step = int(getattr(params, "_bias_restart_step", 0) or 0)

    wrapped = inner
    if plumed_in:
        wrapped = PlumedCalculator(wrapped, plumed_in, timestep_fs, temperature,
                                   atoms=atoms, output=output,
                                   restart_step=restart_step)
    elif colvars_in:
        wrapped = ColvarsCalculator(wrapped, colvars_in, timestep_fs, temperature,
                                    atoms=atoms, output=output,
                                    restart_step=restart_step)

    # Position restraints (GROMACS -DPOSRES) compose on top of any active bias
    # (additive forces) and work standalone for every ensemble — they wrap the
    # shared force point exactly like a bias, so the integrator is untouched.
    if want_posres:
        wrapped = PosresCalculator(
            wrapped, posres_in,
            getattr(params, "posres_fc", 0.0),
            getattr(params, "posres_group", "heavy"),
            getattr(params, "posres_ramp", ""),
            getattr(params, "steps", 0),
            atoms=atoms, output=output, restart_step=restart_step)

    atoms.calc = wrapped
    return wrapped
