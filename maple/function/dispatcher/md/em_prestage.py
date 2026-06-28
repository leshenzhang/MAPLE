# -*- coding: utf-8 -*-
"""
Energy-minimization (EM) pre-stage for MD runs.

Optional geometry minimization performed BEFORE the MD integrator loop,
reusing the opt-task minimizers (sd / cg / lbfgs).  Triggered by the
GROMACS-style mdp key ``em`` (off / steep / cg / lbfgs).

Motivation
----------
Freshly built / solvated boxes routinely contain bad atomic contacts.
An MLIP evaluates a huge first-step force on such contacts, which blows
up the velocity-Verlet integrator.  Relaxing the geometry first
(max |F| < ``emtol``) gives the MD integrator a stable starting point.

Design
------
* Pure MLIP forces: the minimizer calls ``atoms.get_forces()`` on the same
  calculator the MD run uses (no extra physics, in MAPLE MLIP scope).
* No new minimizer is written: this delegates to
  ``optimization.Optimization`` (the same sd/cg/lbfgs code path used by the
  ``opt`` task).
* Hand-off is geometry-only.  The relaxed positions are left on ``atoms``;
  the MD ensemble re-generates Maxwell-Boltzmann velocities @ T on the
  relaxed geometry (any stale input velocities are dropped here).

Units
-----
``emtol`` is in MAPLE-native force units = Hartree / Angstrom (Eh/Ang), the
same units as the opt-task convergence thresholds (atoms.f_max_th).  This is
NOT GROMACS kJ/mol/nm; keep that in mind when porting an mdp file.
"""

import os

import numpy as np
from ase import Atoms


# GROMACS integrator keyword  ->  MAPLE opt-task method
_EM_METHOD_MAP = {
    "steep": "sd",     # steepest descent
    "cg": "cg",        # conjugate gradient
    "lbfgs": "lbfgs",  # limited-memory BFGS
}

# Values that mean "no EM".
_EM_OFF = {"", "off", "none", "no", "false"}

# Displacement convergence is disabled during EM (force-only criterion).
_LARGE_THRESHOLD = 1.0e30


def _max_force_component(atoms: Atoms) -> float:
    """Max |F| component (Eh/Ang) -- the metric the minimizer converges on."""
    return float(np.abs(atoms.get_forces()).max())


def _count_traj_frames(traj_path: str) -> int:
    """Count frames written by the opt minimizer (one per iteration + initial)."""
    if not os.path.exists(traj_path):
        return 0
    n = 0
    with open(traj_path) as fh:
        for line in fh:
            if line.startswith("Image "):
                n += 1
    return n


def run_em_prestage(atoms: Atoms, params: dict, output: str) -> bool:
    """
    Run the optional EM pre-stage in place on ``atoms``.

    Returns
    -------
    bool
        True if an energy minimization was performed, False if EM was off.

    Notes
    -----
    The caller is responsible for skipping this on restart / load_state
    (the geometry then comes from a checkpoint, not a fresh build).
    """
    em = str(params.get("em", "off")).strip().lower()
    if em in _EM_OFF:
        return False

    if em not in _EM_METHOD_MAP:
        raise ValueError(
            f"Unknown em='{em}'. Choose from: off, steep, cg, lbfgs."
        )
    opt_method = _EM_METHOD_MAP[em]

    emtol = float(params.get("emtol", 0.02))
    emstep = float(params.get("emstep", 0.1))
    em_maxsteps = int(params.get("em_maxsteps", 200))

    if emtol <= 0.0:
        raise ValueError(f"emtol must be > 0 (Eh/Ang), got {emtol}.")
    if emstep <= 0.0:
        raise ValueError(f"emstep must be > 0 (Ang), got {emstep}.")
    if em_maxsteps <= 0:
        raise ValueError(f"em_maxsteps must be > 0, got {em_maxsteps}.")

    # Pre-EM max force (same metric the minimizer converges against).
    f_before = _max_force_component(atoms)

    # Override geometry-convergence thresholds: EM converges on force only.
    #   max_f <= emtol  AND  rms_f <= emtol  (rms_f <= max_f, so it never binds)
    #   displacement criteria disabled (huge thresholds)
    saved_thresholds = {
        key: getattr(atoms, key, None)
        for key in ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th")
    }
    atoms.f_max_th = emtol
    atoms.f_rms_th = emtol
    atoms.dp_max_th = _LARGE_THRESHOLD
    atoms.dp_rms_th = _LARGE_THRESHOLD

    # Reuse the opt-task minimizer (sd / cg / lbfgs).  verbose=0 keeps the
    # MD log clean; the minimizer still emits a short summary + _opt artifacts.
    opt_paras = {
        "method": opt_method,
        "max_step": emstep,
        "max_iter": em_maxsteps,
        "verbose": 0,
    }

    header = [
        "\n" + "=" * 80 + "\n",
        f"{'ENERGY MINIMIZATION (EM) PRE-STAGE':^80}\n",
        "=" * 80 + "\n",
        f"EM method:          {em}  (opt minimizer: {opt_method})\n",
        f"emtol (max |F|):    {emtol:.6g} Eh/Ang\n",
        f"emstep (max disp):  {emstep:.6g} Ang\n",
        f"em_maxsteps:        {em_maxsteps}\n",
        f"Max |F| before EM:  {f_before:.6f} Eh/Ang\n",
        "=" * 80 + "\n",
    ]
    _log(output, header)

    from ..optimization.optimization import Optimization

    Optimization(params=opt_paras, output=output, atoms=atoms).run()

    # Restore the MD-stage thresholds (MD itself does not use them, but we
    # leave atoms in the state the dispatcher prepared).
    for key, val in saved_thresholds.items():
        if val is not None:
            setattr(atoms, key, val)

    f_after = _max_force_component(atoms)
    converged = f_after <= emtol

    base, _ = os.path.splitext(output)
    n_iter = max(_count_traj_frames(base + "_opt_traj.xyz") - 1, 0)

    summary = [
        "\n" + "-" * 80 + "\n",
        f"{'EM PRE-STAGE COMPLETE':^80}\n",
        "-" * 80 + "\n",
        f"EM steps taken:     {n_iter} (cap {em_maxsteps})\n",
        f"Max |F| before EM:  {f_before:.6f} Eh/Ang\n",
        f"Max |F| after  EM:  {f_after:.6f} Eh/Ang\n",
        f"emtol:              {emtol:.6g} Eh/Ang\n",
        f"Converged:          {'Yes' if converged else 'No'}"
        f"{'' if converged else '  (hit em_maxsteps; MD will start from best geometry)'}\n",
        "-" * 80 + "\n",
    ]
    _log(output, summary)

    # Drop any stale input velocities: geometry changed, so MD must
    # re-generate Maxwell-Boltzmann velocities @ T on the relaxed geometry.
    if params.get("init_velocities", True) and "velocities" in atoms.arrays:
        del atoms.arrays["velocities"]
        _log(output, [
            "EM relaxed the geometry; dropping stale input velocities so MD "
            "re-initialises Maxwell-Boltzmann velocities on the relaxed "
            "structure.\n"
        ])

    return True


def _log(output: str, lines) -> None:
    with open(output, "a") as fh:
        for line in lines:
            fh.write(line)
