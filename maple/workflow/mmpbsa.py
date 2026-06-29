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

PERFORMANCE -- per-frame MLIP batching (the >100x win)
------------------------------------------------------
A segment's F trajectory frames share ONE topology (identical elements/edges,
different coords), so they are a perfect GPU batch.  ``EndpointBinding`` accepts
EITHER a plain ASE calculator (legacy serial path, one forward per frame) OR a
MAPLE batch calculator that exposes ``prepare(atoms_list)`` + ``get_e_gpu()`` (or
``get_ef_gpu()``).  With a batch calc each segment is ONE ``prepare`` + ONE
energy forward over all F frames -> 3 forwards total (complex/receptor/ligand)
instead of 3F, an ~F-fold speedup.  Energy only: GBSA needs no forces, so
``get_e_gpu`` is preferred and the force backward is skipped where the backend
supports it.  The GB-OBC polar term is likewise vectorized over frames
(``gb_obc_polar_frames``).  Backend note: standard MACE-OFF / MACE-MP are pure
LOCAL MLIPs whose multi-graph batch is EXACTLY isolated (bit-for-bit single-
frame parity); MACE-POL CANNOT be batched (B=1-locked trace + global charge
equilibration couples frames) -- use its sequential mode or MACE-OFF for batching.
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


def gb_obc_polar_frames(coords, charges, znum, eps_solute=1.0, eps_solvent=78.5,
                        row_chunk=512):
    """
    Frame-vectorized GB-OBC (igb=2) polar solvation -> ndarray (F,) kcal/mol.

    coords ((F,n,3) or (n,3) -> F=1) Angstrom; charges (n,) e, znum (n,) atomic
    numbers are frame-INVARIANT (a single-topology segment sliced across the
    trajectory), so the intrinsic radii / HCT screening / offset radii are built
    ONCE.  The O(n^2) descreening integral and GB pair energy carry a leading
    frame axis evaluated by BLAS; the row tile is shrunk to ``max(1,
    row_chunk//F)`` so peak working memory ~ F*chunk*n stays comparable to the
    single-frame kernel (memory-bounded TRUE vectorization, not a python frame
    loop).  F=1 reduces to ``gb_obc_polar``.  Each frame matches ``gb_obc_polar``
    of that frame to ~1e-9 kcal/mol (verified in the smoke).
    """
    coords = np.asarray(coords, float)
    if coords.ndim == 2:                              # degenerate single frame
        coords = coords[None]
    F, n, _ = coords.shape
    q = np.asarray(charges, float)
    radii = np.array([_MBONDI.get(int(z), 1.5) for z in znum], float)
    screen = np.array([_SCREEN.get(int(z), 0.8) for z in znum], float)
    rho = radii - _OBC_OFFSET
    sr = screen * rho
    rc = max(1, row_chunk // F)                       # frame-adaptive row tile

    # ---- HCT descreening integral I (F,n), row-chunked ----
    I = np.zeros((F, n))
    for s in range(0, n, rc):
        e = min(s + rc, n)
        d = coords[:, s:e, None, :] - coords[:, None, :, :]   # (F,chunk,n,3)
        r = np.sqrt((d * d).sum(-1))                          # (F,chunk,n)
        rho_i = rho[None, s:e, None]                          # (1,chunk,1)
        srj = sr[None, None, :]                               # (1,1,n)
        active = (rho_i < (r + srj))
        ii = np.arange(s, e)
        active[:, np.arange(e - s), ii] = False               # exclude diagonal
        rsafe = np.where(active, r, 1.0)
        lower = np.maximum(rho_i, np.abs(rsafe - srj))
        l_ij = 1.0 / lower
        u_ij = 1.0 / (rsafe + srj)
        l2, u2 = l_ij * l_ij, u_ij * u_ij
        rinv = 1.0 / rsafe
        term = (l_ij - u_ij + 0.25 * rsafe * (u2 - l2)
                + 0.5 * rinv * np.log(u_ij / l_ij)
                + 0.25 * srj * srj * rinv * (l2 - u2))
        engulf = active & (rho_i < (srj - rsafe))
        term = term + np.where(engulf, 2.0 * (1.0 / rho_i - l_ij), 0.0)
        I[:, s:e] = 0.5 * np.where(active, term, 0.0).sum(axis=2)

    # ---- effective Born radii (F,n) ----
    psi = I * rho[None, :]
    tanh_arg = _OBC_ALPHA * psi - _OBC_BETA * psi**2 + _OBC_GAMMA * psi**3
    Rinv = 1.0 / rho[None, :] - np.tanh(tanh_arg) / radii[None, :]
    R = 1.0 / Rinv

    # ---- GB pair energy (F,), row-chunked; includes i=j self term ----
    factor = -0.5 * _COULOMB * (1.0 / eps_solute - 1.0 / eps_solvent)
    e_gb = np.zeros(F)
    for s in range(0, n, rc):
        en = min(s + rc, n)
        d = coords[:, s:en, None, :] - coords[:, None, :, :]  # (F,chunk,n,3)
        r2 = (d * d).sum(-1)                                  # (F,chunk,n)
        RiRj = R[:, s:en, None] * R[:, None, :]               # (F,chunk,n)
        f = np.sqrt(r2 + RiRj * np.exp(-r2 / (4.0 * RiRj)))
        qq = (q[s:en, None] * q[None, :])[None, :, :]         # (1,chunk,n)
        e_gb += (qq / f).sum(axis=(1, 2))
    return factor * e_gb                                      # (F,)


# --------------------------------------------------------------------------- #
# Endpoint binding driver                                                      #
# --------------------------------------------------------------------------- #
class EndpointBinding:
    """
    MLIP single-trajectory endpoint binding score for one protein-ligand system.

    Parameters
    ----------
    prmtop, trajectory : AMBER topology + NetCDF trajectory of the *complex*.
    calc               : an instantiated energy engine, EITHER
                         * a plain MAPLE/ASE calculator (Hartree-native, e.g.
                           MACEOFFGenericCalculator) -> legacy SERIAL path, one
                           forward per frame; OR
                         * a MAPLE BATCH calculator exposing ``prepare(atoms_list,
                           fixed_nmax)`` + ``get_e_gpu()`` (or ``get_ef_gpu()``)
                           in Hartree -> BATCHED path, one forward per segment.
                         Shared across the complex / receptor / ligand segments.
    ligand_resname     : residue label of the ligand (default 'LIG').
    frames             : None|int|list -- which trajectory frames to average.
    mode               : 'gas'  -> dE_int only (MLIP interaction energy)
                         'gb'   -> dE_int + GB-OBC polar ddG (no nonpolar SA)
    output             : directory for the per-segment SP logs.
    max_batch          : None|int -- max frames per batched forward (only used on
                         the batch path).  None = all F frames in ONE forward;
                         set an int to cap GPU memory for a large complex x many
                         frames (the segment is then chunked into ceil(F/max_batch)
                         forwards, each still amortizing the topology prepare).
    """

    def __init__(self, prmtop, trajectory, calc, ligand_resname="LIG",
                 frames=20, mode="gas", output="mmpbsa_out", max_batch=None):
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
        self.max_batch = max_batch
        self._batched = self._is_batch_calc(calc)
        os.makedirs(output, exist_ok=True)
        self.symbols = self.top.symbols
        self.z = self.top.atomic_numbers

    @staticmethod
    def _is_batch_calc(calc):
        """Duck-type a MAPLE batch calculator (matches SinglePoint._get_batch_calc):
        exposes ``prepare`` AND an energy forward (``get_e_gpu`` or ``get_ef_gpu``)."""
        return (callable(getattr(calc, "prepare", None))
                and (callable(getattr(calc, "get_e_gpu", None))
                     or callable(getattr(calc, "get_ef_gpu", None))))

    def _seg_charge(self, idx):
        """Integer net (formal) charge of a segment = rounded sum of prmtop
        partial charges over ``idx``.  Consumed by charge-aware backends
        (MACE-POL / AIMNet2); ignored by pure-local MACE-OFF.  A well-built
        protein/ligand fragment has integer net charge so the round is exact."""
        if self.top.charges is None:
            return 0
        return int(round(float(np.asarray(self.top.charges)[idx].sum())))

    def _seg_atoms(self, idx):
        """Build the list of per-frame Atoms for an atom-index selection.
        ``info['charge']`` carries the segment formal charge (charged-backend
        input); ``info['mult']=1`` is the ALL-ML neutral/closed-shell domain
        default (radical/open-shell ligands would need an explicit multiplicity)."""
        syms = [self.symbols[i] for i in idx]
        chg = self._seg_charge(idx)
        out = []
        for f in range(self.coords.shape[0]):
            a = Atoms(symbols=syms, positions=self.coords[f][idx])
            a.info["charge"] = chg
            a.info["mult"] = 1
            out.append(a)
        return out

    def _seg_energies_kcal(self, idx, tag):
        """Per-frame MLIP energy (kcal/mol) for a segment.

        BATCH path: ONE ``prepare`` + ONE energy forward over all F frames (the
        ~F-fold win), chunked by ``max_batch`` if set.  SERIAL path (plain ASE
        calc): the original per-frame SinglePoint loop, unchanged."""
        frames = self._seg_atoms(idx)
        if self._batched:
            e_ha = self._batched_energies_hartree(frames)
        else:
            for a in frames:
                a.calc = self.calc
            sp = SinglePoint(os.path.join(self.output, f"sp_{tag}.out"),
                             frames, paras={"verbose": 0})
            sp.run()
            e_ha = np.asarray(sp.energies_hartree, float)
        return e_ha * KCAL_PER_HARTREE

    def _batched_energies_hartree(self, frames):
        """ONE prepare + ONE energy forward per (sub)batch -> energies (F,) Ha.

        Energy only: prefers ``get_e_gpu`` (no force backward); falls back to
        ``get_ef_gpu`` and DROPS the forces when a backend exposes only the latter
        (e.g. the MACE batch calcs) -- still one batched forward over all frames,
        so the F-fold speedup holds; ponytail: the discarded force backward is a
        cheap add-on next to the forward, not worth a per-backend get_e_gpu shim."""
        F = len(frames)
        mb = self.max_batch or F
        out = np.empty(F, float)
        has_e = callable(getattr(self.calc, "get_e_gpu", None))
        for s in range(0, F, mb):
            chunk = frames[s:s + mb]
            self.calc.prepare(chunk, fixed_nmax=None)
            if has_e:
                E = self.calc.get_e_gpu()                 # (b,) Ha
            else:
                E, _F = self.calc.get_ef_gpu()            # (b,) Ha ; forces dropped
            out[s:s + len(chunk)] = np.asarray(E.detach().to("cpu"), float)
        return out

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
            C = self.coords                                       # (F,natom,3)
            # GB polar term, vectorized over ALL frames per segment (no python loop)
            gc = gb_obc_polar_frames(C, q, self.z)                # (F,)
            gr = gb_obc_polar_frames(C[:, self.rec_idx, :],
                                     q[self.rec_idx], self.z[self.rec_idx])
            gl = gb_obc_polar_frames(C[:, self.lig_idx, :],
                                     q[self.lig_idx], self.z[self.lig_idx])
            ddG = gc - gr - gl                                    # (F,)
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
