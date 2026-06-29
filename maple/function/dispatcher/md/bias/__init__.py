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
from .gamd import (GamdCalculator, gamd_enabled, gamd_params,
                   gamd_reweight_1d)
from .steered import (SteeredMDCalculator, smd_enabled, jarzynski_1d,
                      read_smd_work)
from . import umbrella
from . import wham2d

__all__ = ["maybe_wrap_bias", "PlumedCalculator", "ColvarsCalculator",
           "PosresCalculator", "posres_enabled", "GamdCalculator",
           "gamd_enabled", "gamd_params", "gamd_reweight_1d",
           "SteeredMDCalculator", "smd_enabled", "jarzynski_1d",
           "read_smd_work", "umbrella", "wham2d"]


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
    gamd_in = getattr(params, "gamd", "")
    smd_in = getattr(params, "smd", "")
    want_posres = posres_enabled(posres_in)
    want_gamd = gamd_enabled(gamd_in)
    want_smd = smd_enabled(smd_in)
    if (not plumed_in and not colvars_in and not want_posres
            and not want_gamd and not want_smd):
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

    # Steered MD (constant-velocity pull on a COM-COM distance CV) is an
    # additive moving restraint: it wraps on top of any active bias/posres,
    # inside the optional outermost GaMD boost. total_steps = params.steps.
    if want_smd:
        _l0 = getattr(params, "smd_lam0", "")
        _lam0 = (None if (_l0 is None or str(_l0).strip().lower()
                          in ("", "auto", "none"))
                 else float(_l0))
        wrapped = SteeredMDCalculator(
            wrapped,
            getattr(params, "smd_group1", ""),
            getattr(params, "smd_group2", ""),
            float(getattr(params, "smd_k", 0.0)),
            _lam0,
            float(getattr(params, "smd_lam1", 0.0)),
            int(getattr(params, "steps", 0)),
            atoms=atoms, output=output, restart_step=restart_step,
            log_every=int(getattr(params, "smd_log_every", 10)))

    # GaMD total-potential boost wraps outermost: ΔV is defined on the full
    # potential the dynamics sees (MLIP + any active bias/restraint). It is a
    # CV-free accelerator, normally used standalone; composing on top is valid.
    if want_gamd:
        wrapped = GamdCalculator(
            wrapped,
            mode=(gamd_in if isinstance(gamd_in, str) else "lower"),
            sigma0_kcal=float(getattr(params, "gamd_sigma0", 6.0)),
            prep_steps=int(getattr(params, "gamd_prep_steps", 2000)),
            temperature=temperature,
            params=getattr(params, "gamd_params", None),
            atoms=atoms, output=output, restart_step=restart_step)

    atoms.calc = wrapped
    return wrapped
