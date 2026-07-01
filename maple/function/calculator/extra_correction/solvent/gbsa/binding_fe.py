# -*- coding: utf-8 -*-
"""ML-GBSA binding free-energy decomposition + per-frame batched rescoring.

Ports the algorithm of the USER's ai-pbsa ``mlgbsa_endpoint.py`` into MAPLE, but
replaces the OpenMM ``GBSAOBCForce`` external with the native, differentiable
OBC-II core (``solvent.gbsa.obc``) so the whole thing rides MAPLE's GPU batch
axis (frames = batch dimension) instead of a per-frame Python loop.

Scheme (ML-GBSA, single-trajectory ASA, dH-based, no entropy):

    G_state(frame) = E_MLIP(state, frame) + G_polar^GB(state, frame) + G_SA(state, frame)
    dG_bind        = <G_complex> - <G_receptor> - <G_ligand>

where the gas-phase energy ``E_MLIP`` comes from a *pure MLIP* (UMA / MACE-OFF --
there is NO MM force field), and the GB-polar + ACE-SA continuum terms come from
the native OBC-II at FIXED point charges. The three states (complex / receptor /
ligand) are sliced from one complex trajectory by a ligand atom mask.

PB option: ``G_polar`` can instead come from an external Amber ``pbsa`` single
point (see :func:`pb_rescore_external`); PB is endpoint-only (no PB dynamics).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch

from . import obc as _obc

HARTREE_TO_KCAL = 627.5094740631

__all__ = [
    "gb_sa_solv_hartree_batched",
    "StateSpec",
    "binding_fe",
    "pb_rescore_external",
]


def gb_sa_solv_hartree_batched(
    coords_ang,
    charges,
    radii_nm,
    screen,
    *,
    solvent_dielectric: float = 78.5,
    solute_dielectric: float = 1.0,
    include_sa: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    chunk: Optional[int] = None,
) -> np.ndarray:
    """Native OBC-II G_polar+G_SA for a batch of frames -> ``[B]`` array (Hartree).

    ``coords_ang`` is ``[B, N, 3]`` (Angstrom); ``charges``/``radii_nm``/``screen``
    are ``[N]`` (fixed across frames -- the ML-GBSA fixed-charge assumption). The
    GB term vectorises to a single ``[B, N, N]`` op; ``chunk`` caps the batch
    slice to bound memory on large N.
    """
    coords = torch.as_tensor(np.asarray(coords_ang), dtype=dtype, device=device)
    if coords.ndim == 2:
        coords = coords.unsqueeze(0)
    q = torch.as_tensor(np.asarray(charges), dtype=dtype, device=device)
    rn = torch.as_tensor(np.asarray(radii_nm), dtype=dtype, device=device)
    sc = torch.as_tensor(np.asarray(screen), dtype=dtype, device=device)
    B = coords.shape[0]
    cs = chunk or B
    outs = []
    with torch.no_grad():
        for s in range(0, B, cs):
            e = _obc.gbsa_energy_hartree(
                coords[s:s + cs], q, rn, sc,
                solute_dielectric=solute_dielectric,
                solvent_dielectric=solvent_dielectric,
                include_sa=include_sa,
            )
            outs.append(e.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


@dataclass
class StateSpec:
    """One endpoint state (complex / receptor / ligand)."""
    name: str
    idx: np.ndarray                 # atom indices into the complex
    charges: np.ndarray             # [n] fixed point charges (e)
    radii_nm: np.ndarray            # [n] OBC radii (nm)
    screen: np.ndarray              # [n] HCT screening factors
    coords_ang: np.ndarray          # [F, n, 3] per-frame coordinates (Angstrom)
    e_mlip_hartree: Optional[np.ndarray] = None   # [F] MLIP gas-phase energy


def _state_energies(
    spec: StateSpec,
    mlip_energy_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    *,
    solvent_dielectric: float,
    include_sa: bool,
    device: str,
    chunk: Optional[int],
):
    g_solv = gb_sa_solv_hartree_batched(
        spec.coords_ang, spec.charges, spec.radii_nm, spec.screen,
        solvent_dielectric=solvent_dielectric, include_sa=include_sa,
        device=device, chunk=chunk,
    )
    e_mlip = spec.e_mlip_hartree
    if e_mlip is None:
        if mlip_energy_fn is None:
            raise ValueError(
                f"state {spec.name!r} has no e_mlip_hartree and no mlip_energy_fn")
        e_mlip = np.asarray(mlip_energy_fn(spec.coords_ang), dtype=float)
    e_mlip = np.asarray(e_mlip, dtype=float).reshape(-1)
    e_tot = e_mlip + g_solv
    return e_mlip, g_solv, e_tot


def binding_fe(
    states: Dict[str, StateSpec],
    mlip_energy_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    *,
    solvent_dielectric: float = 78.5,
    include_sa: bool = True,
    device: str = "cpu",
    chunk: Optional[int] = None,
    units: str = "kcal",
) -> dict:
    """ML-GBSA binding-FE decomposition over a trajectory.

    ``states`` maps ``'complex'/'receptor'/'ligand'`` -> :class:`StateSpec`. Each
    state's gas-phase energy is either supplied (``e_mlip_hartree``) or computed
    by ``mlip_energy_fn(coords_ang[F,n,3]) -> E_hartree[F]`` (the batched MLIP
    forward). Returns frame-averaged ``dG_bind`` with SEM plus a per-state /
    per-frame breakdown. Output energies in kcal/mol (``units='kcal'``) or
    Hartree (``units='hartree'``).
    """
    for req in ("complex", "receptor", "ligand"):
        if req not in states:
            raise KeyError(f"states missing required key {req!r}")

    scale = HARTREE_TO_KCAL if units == "kcal" else 1.0
    breakdown = {}
    e_tot = {}
    nframes = None
    for name, spec in states.items():
        e_mlip, g_solv, et = _state_energies(
            spec, mlip_energy_fn, solvent_dielectric=solvent_dielectric,
            include_sa=include_sa, device=device, chunk=chunk)
        e_tot[name] = et
        if nframes is None:
            nframes = et.shape[0]
        breakdown[name] = {
            "e_mlip": (e_mlip * scale),
            "g_solv": (g_solv * scale),
            "e_tot": (et * scale),
            "e_tot_mean": float(et.mean() * scale),
            "g_solv_mean": float(g_solv.mean() * scale),
            "e_mlip_mean": float(e_mlip.mean() * scale),
        }

    dG_frame = (e_tot["complex"] - e_tot["receptor"] - e_tot["ligand"]) * scale
    n = max(len(dG_frame), 1)
    return {
        "dG_bind": float(dG_frame.mean()),
        "dG_sem": float(dG_frame.std() / np.sqrt(n)),
        "dG_std": float(dG_frame.std()),
        "n_frames": int(n),
        "units": units,
        "solvent_dielectric": solvent_dielectric,
        "include_sa": include_sa,
        "dG_frame": dG_frame,
        "per_state": {k: {kk: vv for kk, vv in v.items()
                          if not isinstance(vv, np.ndarray)}
                      for k, v in breakdown.items()},
        "breakdown": breakdown,
    }


def pb_rescore_external(npz_dump, ligmask, prmtop, *, pbsa_exe="pbsa",
                        epsin=1.0, epsout=80.0, ipb=2, inp=1, workdir=None):
    """ML-PBSA PB option: thin wrapper around the external Amber ``pbsa`` solver.

    This is a *single-point rescoring backend only* (no PB dynamics, no native PB
    solver, no analytic PB forces). It mirrors ai-pbsa ``mlpbsa_phaseB.py``:
    consume a Phase-A dump of per-state coords + ML charges + E_MLIP, inject the
    ML charges into the prmtop, run ``pbsa`` (ipb=2 inp=1 epsin=1 epsout=80) as a
    CPU single point per state, and return ``dG_bind = dE_MLIP + dG_PB(+SA)``.

    Requires an external AmberTools ``pbsa`` binary at runtime; raises if absent.
    Implemented as a deferred stub here -- the proven driver lives in
    ai-pbsa ``mlpbsa_phaseB.py`` and should be vendored when PB rescoring is wired
    into a MAPLE job. Kept as an explicit, documented entry point so the PB path
    is discoverable and clearly separated from the native GB dynamics path.
    """
    raise NotImplementedError(
        "PB rescoring is an external-`pbsa` single-point backend (no PB dynamics). "
        "Port ai-pbsa mlpbsa_phaseB.py here when wiring PB into a MAPLE job; the "
        "native dynamics/forces path is GB/OBC-II only."
    )
