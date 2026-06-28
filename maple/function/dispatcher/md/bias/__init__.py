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
from . import umbrella

__all__ = ["maybe_wrap_bias", "PlumedCalculator", "ColvarsCalculator",
           "umbrella"]


def maybe_wrap_bias(atoms, params, output):
    """Wrap ``atoms.calc`` with a bias calculator if the input asks for one.

    Reads ``params.plumed`` / ``params.colvars`` (empty ⇒ no bias) plus
    ``params.timestep`` (fs) and ``params.temperature`` (K). Mutates
    ``atoms.calc`` in place and returns it (unchanged if no bias requested).
    Safe no-op on params dataclasses without the fields.
    """
    plumed_in = getattr(params, "plumed", "") or ""
    colvars_in = getattr(params, "colvars", "") or ""
    if not plumed_in and not colvars_in:
        return atoms.calc

    inner = atoms.calc
    if inner is None:
        raise ValueError("maybe_wrap_bias: atoms.calc is None — set the MLIP "
                         "calculator before applying a bias.")
    timestep_fs = float(getattr(params, "timestep", 0.5))
    temperature = float(getattr(params, "temperature", 300.0))
    restart_step = int(getattr(params, "_bias_restart_step", 0) or 0)

    if plumed_in:
        wrapped = PlumedCalculator(inner, plumed_in, timestep_fs, temperature,
                                   atoms=atoms, output=output,
                                   restart_step=restart_step)
    else:
        wrapped = ColvarsCalculator(inner, colvars_in, timestep_fs, temperature,
                                    atoms=atoms, output=output,
                                    restart_step=restart_step)
    atoms.calc = wrapped
    return wrapped
