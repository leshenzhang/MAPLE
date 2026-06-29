# -*- coding: utf-8 -*-
"""
Single-trajectory endpoint binding free energy with an MLIP energy engine.

This is the MAPLE-native half of an MM/GBSA-style binding analysis: the
gas-phase interaction energy is supplied by the **MLIP** (e.g. MACE-OFF via the
``mace-off-generic`` backend) instead of a classical force field, evaluated
through MAPLE's existing single-point dispatcher.  The continuum-solvent polar
term is an optional GB-OBC add-on (igb=2 form); the nonpolar surface term is
intentionally omitted (it nearly cancels in the per-ligand difference for a
congeneric series -- see ``mode`` doc below).

Single-trajectory (1-traj) approximation, the standard cheap MM/PBSA protocol:
receptor and ligand coordinates are sliced from the SAME complex trajectory, so

    dE_int(frame) = E_complex - E_receptor - E_ligand            (gas, MLIP)
    dG_bind       = < dE_int (+ ddG_solv) >_frames               (ensemble avg)

No intramolecular reorganization is double counted because every term uses the
identical complex geometry.

Scope: the MD force engine and all single-point energies are pure MLIP (MAPLE
calls ``atoms.get_potential_energy()`` on the supplied calculator only).  Full
Poisson-Boltzmann ddG and entropy (-T dS) are out of scope here -- those route
to the external sander ML-MMPBSA pipeline; this module is the narrow MAPLE
bridge that turns an MLIP into an endpoint binding score.
"""

import os
import numpy as np
from ase import Atoms

from .amber_io import Prmtop, read_nc_coords, KCAL_PER_HARTREE
from ..function.dispatcher.sp.sp import SinglePoint

# GB-OBC (igb=2) constants
_OBC_ALPHA, _OBC_BETA, _OBC_GAMMA = 0.8, 0.0, 2.909125
_OBC_OFFSET = 0.09                  # Angstrom
_COULOMB = 332.0522                 # kcal*Angstrom / (mol*e^2)
# mbondi intrinsic radii (Angstrom) and HCT screening factors, by element Z
_MBONDI = {1: 1.2, 6: 1.7, 7: 1.55, 8: 1.5, 9: 1.5,
           15: 1.85, 16: 1.8, 17: 1.7, 35: 1.85, 53: 1.98}
_SCREEN = {1: 0.85, 6: 0.72, 7: 0.79, 8: 0.85, 9: 0.88,
           15: 0.86, 16: 0.96, 17: 0.8, 35: 0.8, 53: 0.8}


# --------------------------------------------------------------------------- #
# GB-OBC polar solvation (one frame)                                          #
# --------------------------------------------------------------------------- #
def gb_obc_polar(coords, charges, znum, eps_solute=1.0, eps_solvent=78.5,
                 row_chunk=512):
    """
    GB-OBC (igb=2) polar solvation free energy of ONE structure (kcal/mol).

    coords (n,3) Angstrom, charges (n,) e, znum (n,) atomic numbers.  Vectorized
    with row chunking so an N~5000-atom complex stays well under ~100 MB.

    Validated against the analytic single-ion Born limit (see __main__).
    """
    coords = np.asarray(coords, float)
    q = np.asarray(charges, float)
    n = coords.shape[0]
    radii = np.array([_MBONDI.get(int(z), 1.5) for z in znum], float)
    screen = np.array([_SCREEN.get(int(z), 0.8) for z in znum], float)
    rho = radii - _OBC_OFFSET            # offset radius (rho_i')
    sr = screen * rho                    # scaled radius S_j * rho_j'

    # ---- HCT descreening integral I_i (row-chunked) ----
    I = np.zeros(n)
    for s in range(0, n, row_chunk):
        e = min(s + row_chunk, n)
        d = coords[s:e, None, :] - coords[None, :, :]
        r = np.sqrt((d * d).sum(-1))                 # (chunk, n)
        rho_i = rho[s:e, None]                        # (chunk,1)
        srj = sr[None, :]                             # (1,n)
        # mask out self and pairs where i fully engulfs j's descreen sphere
        active = (rho_i < (r + srj))
        ii = np.arange(s, e)
        active[np.arange(e - s), ii] = False          # exclude diagonal
        rsafe = np.where(active, r, 1.0)
        lower = np.maximum(rho_i, np.abs(rsafe - srj))
        l_ij = 1.0 / lower
        u_ij = 1.0 / (rsafe + srj)
        l2, u2 = l_ij * l_ij, u_ij * u_ij
        rinv = 1.0 / rsafe
        term = (l_ij - u_ij + 0.25 * rsafe * (u2 - l2)
                + 0.5 * rinv * np.log(u_ij / l_ij)
                + 0.25 * srj * srj * rinv * (l2 - u2))
        # extra term when atom i lies fully inside j's scaled sphere
        engulf = active & (rho_i < (srj - rsafe))
        term = term + np.where(engulf, 2.0 * (1.0 / rho_i - l_ij), 0.0)
        I[s:e] = 0.5 * np.where(active, term, 0.0).sum(axis=1)

    # ---- effective Born radii (OBC rescaling) ----
    psi = I * rho
    tanh_arg = _OBC_ALPHA * psi - _OBC_BETA * psi**2 + _OBC_GAMMA * psi**3
    Rinv = 1.0 / rho - np.tanh(tanh_arg) / radii
    R = 1.0 / Rinv                                    # effective Born radius

    # ---- GB pair energy (row-chunked); includes i=j self term ----
    factor = -0.5 * _COULOMB * (1.0 / eps_solute - 1.0 / eps_solvent)
    e_gb = 0.0
    for s in range(0, n, row_chunk):
        en = min(s + row_chunk, n)
        d = coords[s:en, None, :] - coords[None, :, :]
        r2 = (d * d).sum(-1)                          # (chunk,n)
        RiRj = R[s:en, None] * R[None, :]
        f = np.sqrt(r2 + RiRj * np.exp(-r2 / (4.0 * RiRj)))
        qq = q[s:en, None] * q[None, :]
        e_gb += (qq / f).sum()
    return float(factor * e_gb)


# --------------------------------------------------------------------------- #
# Endpoint binding driver                                                      #
# --------------------------------------------------------------------------- #
class EndpointBinding:
    """
    MLIP single-trajectory endpoint binding score for one protein-ligand system.

    Parameters
    ----------
    prmtop, trajectory : AMBER topology + NetCDF trajectory of the *complex*.
    calc               : an instantiated MAPLE/ASE calculator (Hartree-native,
                         e.g. MACEOFFGenericCalculator).  Shared across the
                         complex / receptor / ligand single points.
    ligand_resname     : residue label of the ligand (default 'LIG').
    frames             : None|int|list -- which trajectory frames to average.
    mode               : 'gas'  -> dE_int only (MLIP interaction energy)
                         'gb'   -> dE_int + GB-OBC polar ddG (no nonpolar SA)
    output             : directory for the per-segment SP logs.
    """

    def __init__(self, prmtop, trajectory, calc, ligand_resname="LIG",
                 frames=20, mode="gas", output="mmpbsa_out"):
        self.top = Prmtop(prmtop)
        self.lig_idx, self.rec_idx = self.top.ligand_receptor_masks(ligand_resname)
        self.coords = read_nc_coords(trajectory, frames=frames)  # (F,natom,3)
        if self.coords.shape[1] != self.top.natom:
            raise ValueError(
                f"trajectory atom count {self.coords.shape[1]} != prmtop "
                f"natom {self.top.natom}")
        self.calc = calc
        self.mode = mode
        self.output = output
        os.makedirs(output, exist_ok=True)
        self.symbols = self.top.symbols
        self.z = self.top.atomic_numbers

    def _seg_atoms(self, idx):
        """Build the list of per-frame Atoms for an atom-index selection."""
        syms = [self.symbols[i] for i in idx]
        return [Atoms(symbols=syms, positions=self.coords[f][idx])
                for f in range(self.coords.shape[0])]

    def _seg_energies_kcal(self, idx, tag):
        """Per-frame MLIP energy (kcal/mol) for a segment, via SinglePoint."""
        frames = self._seg_atoms(idx)
        for a in frames:
            a.calc = self.calc
        sp = SinglePoint(os.path.join(self.output, f"sp_{tag}.out"),
                         frames, paras={"verbose": 0})
        sp.run()
        return np.array(sp.energies_hartree) * KCAL_PER_HARTREE

    def run(self):
        all_idx = np.arange(self.top.natom)
        e_cplx = self._seg_energies_kcal(all_idx, "complex")
        e_rec = self._seg_energies_kcal(self.rec_idx, "receptor")
        e_lig = self._seg_energies_kcal(self.lig_idx, "ligand")
        dE_int = e_cplx - e_rec - e_lig                       # (F,) kcal/mol

        result = {
            "mode": self.mode,
            "n_frames": int(self.coords.shape[0]),
            "n_atoms_complex": int(self.top.natom),
            "n_atoms_ligand": int(self.lig_idx.size),
            "dE_int_mean": float(dE_int.mean()),
            "dE_int_sem": float(dE_int.std(ddof=1) / np.sqrt(len(dE_int)))
                          if len(dE_int) > 1 else 0.0,
        }

        if self.mode == "gb":
            if self.top.charges is None:
                raise ValueError("mode='gb' needs CHARGE in prmtop (absent).")
            q = self.top.charges
            ddG = np.empty(self.coords.shape[0])
            for f in range(self.coords.shape[0]):
                c = self.coords[f]
                gc = gb_obc_polar(c, q, self.z)
                gr = gb_obc_polar(c[self.rec_idx], q[self.rec_idx], self.z[self.rec_idx])
                gl = gb_obc_polar(c[self.lig_idx], q[self.lig_idx], self.z[self.lig_idx])
                ddG[f] = gc - gr - gl
            dG = dE_int + ddG
            result.update({
                "ddG_solv_mean": float(ddG.mean()),
                "dG_bind_mean": float(dG.mean()),
                "dG_bind_sem": float(dG.std(ddof=1) / np.sqrt(len(dG)))
                               if len(dG) > 1 else 0.0,
                "score": float(dG.mean()),
            })
        else:
            result["score"] = result["dE_int_mean"]
        return result


if __name__ == "__main__":
    # GB-OBC self-test: single ion -> analytic Born  dG = -166(1-1/eps) q^2 / R,
    # with R = radius - offset (isolated atom has no descreening).
    born_k = 0.5 * _COULOMB                       # = 166.026, the Born prefactor
    # Z=8 -> radius 1.5; isolated atom has no descreening so R = radius - offset.
    R8 = _MBONDI[8] - _OBC_OFFSET
    born8 = -born_k * (1.0 - 1.0 / 78.5) * 1.0 / R8
    got8 = gb_obc_polar(np.zeros((1, 3)), np.array([1.0]), np.array([8]))
    print(f"single-ion GB={got8:.3f}  analytic Born={born8:.3f}")
    assert abs(got8 - born8) < 1e-6, (got8, born8)
    # two opposite charges far apart ~ sum of two Born ions (cross term small)
    two = gb_obc_polar(np.array([[0, 0, 0], [20.0, 0, 0]]),
                       np.array([1.0, -1.0]), np.array([8, 8]))
    print(f"two ions @20A GB={two:.3f}  (~2x Born {2*born8:.3f})")
    assert two < 0
    print("GB-OBC self-test OK")
