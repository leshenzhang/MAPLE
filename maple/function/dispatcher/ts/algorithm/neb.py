# -*- coding: utf-8 -*-
"""
NEB implementation with:
- Kabsch alignment (reports RMSD)
- IDPP interpolation for robust initial path
- Improved tangent NEB forces (true_perp + spring_parallel)
- L-BFGS optimizer on projected NEB forces
- ORCA-style dynamic spring constants

OPT-IN additive upgrades (the single-band path above is kept as the fallback
+ parity oracle; new features are opt-in and do not alter it):
- Multi-band NEB (NEB.run_multiband): B independent bands, every image of
  every band in ONE batched forward (image-as-batch; CatTSunami / OCPNEB,
  Wander et al. arXiv:2405.02078, ACS Catal. 2024, DOI 10.1021/acscatal.4c04272).
- DyNEB image freezing (Lindgren, Kastlunger & Peterson, JCTC 2019, 15, 5787,
  DOI 10.1021/acs.jctc.9b00633).
- Force upgrades: improved tangent (Henkelman & Jonsson, JCP 2000, 113, 9978,
  DOI 10.1063/1.1323224); climbing image F-2(F.tau)tau (Henkelman, Uberuaga &
  Jonsson, JCP 2000, 113, 9901, DOI 10.1063/1.1329672); energy-weighted
  variable springs (Asgeirsson et al., JCTC 2021, 17, 4929,
  DOI 10.1021/acs.jctc.1c00462).
"""
from __future__ import annotations
from lib2to3.pgen2 import driver
import os
import sys
import math
import copy
import time
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from ase import Atoms

try:                       # module-global torch for the on-device helpers below;
    import torch           # the numpy oracle path never touches it (torch may be
except Exception:          # absent in a pure-CPU/serial install).
    torch = None

from .logger import log_info
from ...jobABC import JobABC

from maple.function.utility import Molecules

# =============================================================================
# ------------------------------ Utilities ------------------------------------
# =============================================================================

def to_numpy_f64(x):
    """Convert input (numpy/torch/list/scalar) to float64 numpy array or float."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
            return arr.astype(np.float64, copy=False)
    except Exception:
        pass
    if np.isscalar(x):
        return float(x)
    return np.asarray(x, dtype=np.float64)

def vec1d(x, n_expected=None):
    """Convert to float64 1D vector and optionally check length."""
    v = to_numpy_f64(x).reshape(-1)
    if n_expected is not None and v.size != n_expected:
        raise ValueError(f"Expected size {n_expected}, got {v.size}")
    return v

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Rigid-body least-squares alignment (Kabsch):
    Align Q onto P. Both are (N,3). Returns: Q_aligned, rmsd, R, t
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError("P and Q must have shape (N,3)")

    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc

    C = Q0.T @ P0
    V, S, Wt = np.linalg.svd(C)
    d = (np.linalg.det(V @ Wt) < 0.0)
    if d:
        V[:, -1] *= -1.0
    R = V @ Wt
    t = Pc - Qc @ R

    Q_aligned = Q @ R + t
    diff = P - Q_aligned
    rmsd = float(np.sqrt((diff * diff).sum() / P.shape[0]))
    return Q_aligned, rmsd, R, t

def write_xyz(filename: str, images: List[Atoms], energies: Optional[List[float]] = None):
    """
    Write a multi-frame XYZ trajectory. If energies given, write in comment line.
    """
    with open(filename, "w") as f:
        for i, at in enumerate(images):
            pos = to_numpy_f64(at.get_positions())
            symbols = at.get_chemical_symbols()
            f.write(f"{len(symbols)}\n")
            if energies is not None:
                f.write(f"Image {i}  Energy = {energies[i]:.8f}\n")
            else:
                f.write(f"Image {i}\n")
            for s, (x, y, z) in zip(symbols, pos):
                f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")

def write_all_images_xyz(filename: str, images: List[Atoms], energies: Optional[List[float]] = None, iteration: int = 0):
    """
    Append all current images to an xyz trajectory file (for NEB debug).
    Each block corresponds to one iteration of NEB optimization.
    """
    if iteration == 0 and os.path.exists(filename):
        os.remove(filename)
    with open(filename, "a") as f:
        for i, at in enumerate(images):
            pos = to_numpy_f64(at.get_positions())
            symbols = at.get_chemical_symbols()
            E = None if energies is None else energies[i]
            f.write(f"{len(symbols)}\n")
            if E is not None:
                f.write(f"Iter {iteration} Image {i}  Energy = {E:.10f}\n")
            else:
                f.write(f"Iter {iteration} Image {i}\n")
            for s, (x, y, z) in zip(symbols, pos):
                f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")






# =============================================================================
# ------------------------------- IDPP ----------------------------------------
# =============================================================================

@dataclass
class IDPPParams:
    max_steps: int = 300
    step_size: float = 0.01       # Cartesian step size for IDPP relaxation
    r_cut: float = 12.0           # cutoff for pair list (Å)
    eps: float = 1e-8             # to avoid 1/0
    w_power: float = 2.0          # weight ~ (1 / d_target^w_power)^2 (classical choice: 2-4)

def _pair_indices(n_atoms: int) -> np.ndarray:
    """Return upper-triangular (i<j) index pairs as (M,2)."""
    idx = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            idx.append((i, j))
    return np.asarray(idx, dtype=np.int32)

def _idpp_targets(P0: np.ndarray, P1: np.ndarray, n_inner: int) -> List[np.ndarray]:
    """
    Build target 1/dist arrays for each internal image by linear interpolation
    between endpoints pair distances.
    Returns list of arrays (M,), M = Npair.
    """
    n_atoms = P0.shape[0]
    pairs = _pair_indices(n_atoms)
    Rij0 = np.linalg.norm(P0[pairs[:, 0]] - P0[pairs[:, 1]], axis=1)
    Rij1 = np.linalg.norm(P1[pairs[:, 0]] - P1[pairs[:, 1]], axis=1)
    inv0 = 1.0 / np.maximum(Rij0, 1e-8)
    inv1 = 1.0 / np.maximum(Rij1, 1e-8)

    targets = []
    for k in range(1, n_inner + 1):
        lam = k / (n_inner + 1)
        inv_tgt = (1.0 - lam) * inv0 + lam * inv1
        targets.append(inv_tgt)
    return targets

import numpy as np
from ase import Atoms

# =============================================================================
# ------------------------------- NEB Core ------------------------------------
# =============================================================================

@dataclass
class NEBParams:
    n_images: int = 10                     # total images excluding endpoints
    k_min: float = 0.03                    # minimum spring constant
    k_max: float = 0.3                     # maximum spring constant
    use_dynamic_k: bool = True             # enable ORCA-style dynamic spring constants
    k_decay: float = 0.5                   # decay factor for dynamic k (ORCA default: 0.5)
    max_iter: int = 500 #256
    lbfgs_m: int = 20                      # memory size for L-BFGS
    step0: float = 2e-2                    # initial step length on search direction
    # IDPP control
    ifidpp: int = 1                     # 1: use IDPP for initial path, 0: linear interp.
    # Convergence on projected forces
    neb_f_max_th: float = 9.5e-3           # max(|Fp|) threshold
    neb_f_rms_th: float = 5e-3             # RMS(Fp) threshold
    initial_opt: bool = False              # do initial relaxation of endpoints
    refine: Optional[str] = None           # 'cineb' or 'nebts' or None
    # CINEB-specific
    cineb_f_max_th: float = 1e-2 #3e-03          # max(|Fp|) threshold for CINEB
    cineb_f_rms_th: float = 1e-2 #2e-03          # RMS(Fp) threshold for CINEB
    cilbfgs_m: int = 20                    # memory size for L-BFGS in CINEB
    cistep0: float = 5e-3                  # initial step length for CINEB


def compute_dynamic_k(energies: List[float], k_min: float, k_max: float, k_decay: float = 0.5) -> List[float]:
    """
    Compute ORCA-style dynamic spring constants for each internal image.
    
    The spring constant for image i is computed as:
    k_i = k_max - (k_max - k_min) * exp(-decay * ΔE_i / ΔE_max)
    
    where:
    - ΔE_i = max(E_i - E_ref, 0) with E_ref = max(E_{i-1}, E_{i+1})
    - ΔE_max = max energy barrier along the path
    
    This ensures:
    - Images near the barrier get weaker springs (closer to k_min)
    - Images far from the barrier get stronger springs (closer to k_max)
    
    Parameters
    ----------
    energies : List[float]
        Energies of all images (including endpoints)
    k_min : float
        Minimum spring constant (for barrier region)
    k_max : float
        Maximum spring constant (for flat regions)
    k_decay : float
        Decay factor controlling transition sharpness (ORCA default: 0.5)
    
    Returns
    -------
    List[float]
        Spring constants for each image (length = len(energies))
        Endpoints get k_max by convention (not used in force calculation)
    """
    n_img = len(energies)
    k_springs = [k_max] * n_img  # initialize with k_max
    
    if n_img < 3:
        return k_springs
    
    # Find global energy reference (typically reactant energy)
    E_ref_global = min(energies[0], energies[-1])
    
    # Compute energy barriers for each internal image
    delta_E = []
    for i in range(1, n_img - 1):
        # Local reference: max of neighboring energies
        E_ref_local = max(energies[i-1], energies[i+1])
        # Energy rise above local reference
        dE = max(energies[i] - E_ref_local, 0.0)
        delta_E.append(dE)
    
    # Find maximum barrier
    delta_E_max = max(delta_E) if delta_E else 1.0
    
    # Avoid division by zero
    if delta_E_max < 1e-10:
        # Essentially flat path, use k_max everywhere
        return k_springs
    
    # Compute dynamic k for each internal image
    for idx, dE in enumerate(delta_E):
        i = idx + 1  # actual image index
        # Normalized energy barrier
        dE_norm = dE / delta_E_max
        # ORCA formula: k decreases exponentially as we approach the barrier
        k_i = k_max - (k_max - k_min) * np.exp(-k_decay * dE_norm)
        k_springs[i] = k_i
    
    return k_springs


def improved_tangent(Rm1, R, Rp1, Em1, E, Ep1):
    """
    Energy-weighted improved tangent of Henkelman & Jonsson, JCP 2000, 113,
    9978 (DOI: 10.1063/1.1323224). Always returns a normalized (3N,) vector.
    """
    dm = vec1d(R - Rm1)
    dp = vec1d(Rp1 - R)
    Em, Ep = Em1, Ep1

    # decide direction
    if Ep > E and E > Em:
        t = dp
    elif Ep < E and E < Em:
        t = dm
    else:
        dE_plus  = max(Ep - E, 0.0)
        dE_minus = max(Em - E, 0.0)
        t = dE_plus * dp + dE_minus * dm

    # normalize
    norm = np.linalg.norm(t)

    if norm < 1e-16:
        # fallback 1
        t = dp + dm
        norm = np.linalg.norm(t)
        if norm < 1e-16:
            # fallback 2: return any unit vector
            unit = np.zeros_like(t)
            unit[0] = 1.0
            return unit
    
    return t / norm

def cineb_tangent(Rm1, R, Rp1, Em1, E, Ep1):
    """
    Upwinding tangent for CINEB (Henkelman-Jonsson style).
    Returns a normalized (3N,) vector.

    This function is separate from `improved_tangent` so that:
    - NEB keeps using the original `improved_tangent`.
    - CINEB can use a robust upwinding scheme without touching NEB behavior.
    """
    dm = vec1d(R - Rm1)
    dp = vec1d(Rp1 - R)

    prev_energy = Em1
    ith_energy  = E
    next_energy = Ep1

    # Energy differences relative to image i
    dE_plus  = abs(next_energy - ith_energy)
    dE_minus = abs(prev_energy - ith_energy)

    # Strict uphill: E_{i-1} < E_i < E_{i+1} -> use forward tangent
    if (next_energy > ith_energy) and (ith_energy > prev_energy):
        t = dp

    # Strict downhill: E_{i-1} > E_i > E_{i+1} -> use backward tangent
    elif (next_energy < ith_energy) and (ith_energy < prev_energy):
        t = dm

    else:
        # Around maxima or minima: use energy-weighted combination
        delta_max = max(dE_plus, dE_minus)
        delta_min = min(dE_plus, dE_minus)

        if next_energy >= prev_energy:
            # Next image has higher energy
            t = delta_max * dp + delta_min * dm
        else:
            # Previous image has higher energy
            t = delta_min * dp + delta_max * dm

    norm = np.linalg.norm(t)
    if norm < 1e-16:
        # Degenerate case, fall back to simple sum
        t = dp + dm
        norm = np.linalg.norm(t)
        if norm < 1e-16:
            # Extreme fallback: choose any unit vector
            unit = np.zeros_like(t)
            unit[0] = 1.0
            return unit

    return t / norm


def neb_forces(images: List[Atoms], energies: List[float],
               k_spring: float = None, k_springs: List[float] = None,
               use_dynamic_k: bool = False, k_min: float = 0.03,
               k_max: float = 0.3, k_decay: float = 0.5,
               raw_forces: Optional[List[np.ndarray]] = None) -> Tuple[List[np.ndarray], float, int]:
    """
    Compute NEB projected forces for internal images:
    F_NEB = F_true_perp + F_spring_parallel  (per image)

    Parameters
    ----------
    images : List[Atoms]
        All images including endpoints
    energies : List[float]
        Energies of all images
    raw_forces : List[np.ndarray], optional
        Pre-computed true (calculator) forces per image, shape (N_i, 3) each, in
        the same convention/units as ``at.get_forces()``. When provided (batched
        evaluation path) the per-image ``at.get_forces()`` serial loop is skipped
        entirely. When None (default) forces are read serially from each image's
        attached calculator (backward-compatible behaviour).
    k_spring : float, optional
        Single spring constant (used if k_springs is None and use_dynamic_k=False)
    k_springs : List[float], optional
        Pre-computed spring constants for each image
    use_dynamic_k : bool
        If True and k_springs is None, compute dynamic spring constants
    k_min, k_max, k_decay : float
        Parameters for dynamic spring constant calculation
    
    Returns
    -------
    forces_proj : List[np.ndarray]
        Projected forces (cartesian shape (N,3)) for each image
    max_fp : float
        Maximum force component among all internal images
    hei_idx : int
        Index of highest energy image
    """
    n_img = len(images)
    forces_proj = [None] * n_img
    max_fp = 0.0
    hei_idx = 1

    # get raw forces and flatten. Prefer caller-supplied batched forces; fall back
    # to serial per-image calculator reads for non-batch calculators.
    if raw_forces is None:
        raw_forces = [to_numpy_f64(at.get_forces()) for at in images]
    else:
        raw_forces = [to_numpy_f64(f) for f in raw_forces]
    coords = [to_numpy_f64(at.get_positions()) for at in images]
    Es = [float(e) for e in energies]

    # determine HEI
    inner_indices = list(range(1, n_img - 1))
    hei_idx = max(inner_indices, key=lambda i: Es[i])

    # Determine spring constants
    if k_springs is None:
        if use_dynamic_k:
            k_springs = compute_dynamic_k(Es, k_min, k_max, k_decay)
        else:
            # Use single spring constant for all images
            k_springs = [k_spring if k_spring is not None else k_max] * n_img

    # loop over internal images
    for i in inner_indices:
        Rm1 = coords[i - 1]; R = coords[i]; Rp1 = coords[i + 1]
        Em1 = Es[i - 1];     E = Es[i];     Ep1 = Es[i + 1]

        # tangent
        tau = improved_tangent(Rm1, R, Rp1, Em1, E, Ep1)  # (3N,)
        tau_resh = tau.reshape(-1, 3)

        # true force (calculator gives forces = -grad V); NEB uses gradient convention:
        # F_true = -∇V, but our "forces" are already that.
        F_true = vec1d(raw_forces[i])

        # remove tangent component: F_true_perp = F_true - (F_true·tau) tau
        c = float(np.dot(F_true, tau))
        F_true_perp = F_true - c * tau
        F_true_perp = F_true_perp.reshape(-1, 3)

        # spring force along tangent with dynamic k
        k_i = k_springs[i]
        d_next = float(np.linalg.norm(Rp1 - R))
        d_prev = float(np.linalg.norm(R - Rm1))
        F_spring_par = k_i * (d_next - d_prev) * tau_resh

        # total projected
        Fp = F_true_perp + F_spring_par
        forces_proj[i] = Fp

        max_fp = max(max_fp, float(np.abs(Fp).max()))

    # endpoints fixed: zero projected forces
    forces_proj[0] = np.zeros_like(coords[0])
    forces_proj[-1] = np.zeros_like(coords[-1])

    return forces_proj, max_fp, hei_idx

def rms_force(forces_list: List[np.ndarray]) -> float:
    """RMS of projected forces over all DOFs of internal images."""
    arrs = []
    for i, F in enumerate(forces_list):
        if i == 0 or i == len(forces_list) - 1:
            continue
        arrs.append(F.reshape(-1))
    if not arrs:
        return 0.0
    v = np.concatenate(arrs)
    return float(np.sqrt(np.mean(v * v)))

def compute_energy_weighted_k(energies: List[float], k_min: float, k_max: float) -> List[float]:
    """
    Energy-weighted variable spring constants (OPT-IN spring scheme).

    Asgeirsson, Birgisson, Bjornsson, Becker & Jonsson, JCTC 2021, 17, 4929
    (DOI: 10.1021/acs.jctc.1c00462); original variable-spring form: Henkelman,
    Uberuaga & Jonsson, JCP 2000, 113, 9901 (DOI: 10.1063/1.1329672), Appendix.

    Springs are made STIFFER near the barrier (higher-energy images) and softer
    in the flat reactant/product wings, so image density stays high around the
    saddle (this is the OPPOSITE mapping to compute_dynamic_k's ORCA scheme):

        k_i = k_max - (k_max - k_min) * (E_max - E_i) / (E_max - E_ref),  E_i > E_ref
        k_i = k_min,                                                      E_i <= E_ref

    E_ref = max(E_first, E_last) (higher endpoint); E_max = max band energy.
    Returns one k per image (length = len(energies)); endpoints get k_min but
    their spring force is never used (endpoint NEB forces are zeroed).
    """
    n_img = len(energies)
    k_springs = [k_max] * n_img
    if n_img < 3:
        return k_springs
    E = [float(e) for e in energies]
    E_ref = max(E[0], E[-1])
    E_max = max(E)
    denom = E_max - E_ref
    for i in range(n_img):
        if denom > 1e-10 and E[i] > E_ref:
            k_springs[i] = k_max - (k_max - k_min) * (E_max - E[i]) / denom
        else:
            k_springs[i] = k_min
    return k_springs


def neb_band_forces(images: List[Atoms], energies: List[float],
                    raw_forces: List[np.ndarray], k_springs: List[float],
                    climbing: bool = False, hei_fixed: Optional[int] = None,
                    active_mask: Optional[List[bool]] = None
                    ) -> Tuple[List[np.ndarray], float, int]:
    """
    Per-band NEB projected forces with OPT-IN climbing image and DyNEB freezing.

    With climbing=False and active_mask=None this reproduces neb_forces() exactly
    (improved-tangent perpendicular true force + parallel variable-k spring), so
    it is a drop-in that leaves the single-band fallback path untouched. (A unit
    parity check against neb_forces() is run in the validation harness.)

    Improved tangent:  Henkelman & Jonsson, JCP 2000, 113, 9978
        (DOI: 10.1063/1.1323224).
    Climbing image (climbing=True, applied to the HEI):  F_CI = F - 2 (F.tau) tau,
        Henkelman, Uberuaga & Jonsson, JCP 2000, 113, 9901
        (DOI: 10.1063/1.1329672).
    DyNEB freezing (active_mask):  Lindgren, Kastlunger & Peterson, JCTC 2019,
        15, 5787 (DOI: 10.1021/acs.jctc.9b00633) -- internal images already below
        fmax (active_mask[i] False) get a zeroed NEB force so the optimizer
        concentrates on the saddle region. The current HEI is never frozen.

    raw_forces : list of (N_i,3) true calculator forces, one per image (the
        batched-forward result). Endpoints included; their entries may be stale
        (their NEB force is zeroed anyway).
    """
    n_img = len(images)
    forces_proj = [None] * n_img
    coords = [to_numpy_f64(at.get_positions()) for at in images]
    Es = [float(e) for e in energies]
    raw = [to_numpy_f64(f) for f in raw_forces]

    inner_indices = list(range(1, n_img - 1))
    if (hei_fixed is not None) and (hei_fixed in inner_indices):
        hei_idx = hei_fixed
    else:
        hei_idx = max(inner_indices, key=lambda i: Es[i])

    max_fp = 0.0
    for i in inner_indices:
        Rm1, R, Rp1 = coords[i - 1], coords[i], coords[i + 1]
        Em1, E, Ep1 = Es[i - 1], Es[i], Es[i + 1]

        tau = improved_tangent(Rm1, R, Rp1, Em1, E, Ep1)
        tau_resh = tau.reshape(-1, 3)
        F_true = vec1d(raw[i])

        if climbing and (i == hei_idx):
            # Climbing image: invert the parallel component of the TRUE force and
            # drop the spring force -> the image is driven UP to the saddle.
            ft = float(np.dot(F_true, tau))
            Fp = (F_true - 2.0 * ft * tau).reshape(-1, 3)
        else:
            c = float(np.dot(F_true, tau))
            F_true_perp = (F_true - c * tau).reshape(-1, 3)
            k_i = k_springs[i]
            d_next = float(np.linalg.norm(Rp1 - R))
            d_prev = float(np.linalg.norm(R - Rm1))
            F_spring_par = k_i * (d_next - d_prev) * tau_resh
            Fp = F_true_perp + F_spring_par

        # DyNEB freeze: zero the NEB force of converged (inactive) images so the
        # optimizer leaves them in place (never freeze the climbing/HEI image).
        if (active_mask is not None) and (not active_mask[i]) and (i != hei_idx):
            Fp = np.zeros_like(Fp)

        forces_proj[i] = Fp
        max_fp = max(max_fp, float(np.abs(Fp).max()))

    forces_proj[0] = np.zeros_like(coords[0])
    forces_proj[-1] = np.zeros_like(coords[-1])
    return forces_proj, max_fp, hei_idx


# =============================================================================
# ------------------------------- L-BFGS --------------------------------------
# =============================================================================

class LBFGSDriver:
    """Simple L-BFGS optimizer for NEB projected forces (no DIIS / no line search)."""
    def __init__(self, m=5, curvature=70.0, maxstep=0.2):
        self.m = m
        self.H0 = 1.0 / curvature
        self.maxstep = maxstep
        self.S, self.Y, self.rhos = [], [], []

    def two_loop(self, grad: np.ndarray) -> np.ndarray:
        q = grad.copy()
        alpha = []
        for s, y, rho in reversed(list(zip(self.S, self.Y, self.rhos))):
            a = rho * np.dot(s, q)
            alpha.append(a)
            q -= a * y

        gamma = self.H0 if not self.Y else np.dot(self.Y[-1], self.S[-1]) / (np.dot(self.Y[-1], self.Y[-1]) + 1e-20)
        z = gamma * q

        for (s, y, rho), a in zip(zip(self.S, self.Y, self.rhos), reversed(alpha)):
            b = rho * np.dot(y, z)
            z += s * (a - b)

        return -z

    def update(self, s: np.ndarray, y: np.ndarray):
        rho = 1.0 / (np.dot(y, s) + 1e-20)
        self.S.append(s.copy()); self.Y.append(y.copy()); self.rhos.append(rho)
        if len(self.S) > self.m:
            self.S.pop(0); self.Y.pop(0); self.rhos.pop(0)

    def step_limit(self, step: np.ndarray) -> np.ndarray:
        max_disp = np.max(np.abs(step))
        if max_disp > self.maxstep:
            step *= self.maxstep / max_disp
        return step

    def should_stop(self, grad, fmax_th=1e-3, frms_th=5e-4) -> bool:
        max_f = np.max(np.abs(grad))
        rms_f = np.sqrt(np.mean(grad ** 2))
        return (max_f < fmax_th) and (rms_f < frms_th)

    def ci_should_stop(self, regular_Fp, ci_F):
        """
        Dual convergence check for CINEB, ORCA-style.

        Parameters
        ----------
        regular_Fp : np.ndarray
            1D array of projected NEB forces (F_perp + spring_parallel) for
            all *non-climbing* internal images, flattened.
        ci_F : np.ndarray
            1D array of *true* forces for the climbing image (no projection).

        Uses thresholds stored on the driver:
        - self.fmax_reg, self.frms_reg : regular images (Fp)
        - self.fmax_ci,  self.frms_ci  : climbing image (F)
        """
        if regular_Fp.size == 0:
            maxFp = 0.0
            rmsFp = 0.0
        else:
            maxFp = float(np.max(np.abs(regular_Fp)))
            rmsFp = float(np.sqrt(np.mean(regular_Fp ** 2)))

        if ci_F.size == 0:
            maxF = 0.0
            rmsF = 0.0
        else:
            maxF = float(np.max(np.abs(ci_F)))
            rmsF = float(np.sqrt(np.mean(ci_F ** 2)))

        return (
            (maxFp < self.fmax_reg) and
            (rmsFp < self.frms_reg) and
            (maxF  < self.fmax_ci)  and
            (rmsF  < self.frms_ci)
        )


# =============================================================================
# ----------------- ON-DEVICE (torch CUDA) NEB math (OPT-IN) ------------------
# These mirror improved_tangent / neb_forces / neb_band_forces / compute_*_k /
# LBFGSDriver EXACTLY (numpy->torch swap on the band's (n_img, n_at, 3) tensor),
# so run_multiband(on_device=True) keeps the whole inner loop on the GPU: no
# per-iter .cpu()/.numpy() of the (n_img, nmax_dof) forces, no numpy tangent/
# spring/projection/L-BFGS, no convergence round-trip. The numpy path above stays
# the byte-parity oracle (default on_device=False). CPU-torch parity vs the numpy
# refs: |dFp|~2e-16, |dLBFGS step|~5e-15 (machine eps) over 40 random bands.
# =============================================================================

def _improved_tangent_torch(coord, Es):
    """Vectorized improved tangent (Henkelman & Jonsson, JCP 2000, 113, 9978) for
    ALL internal images at once. coord (n_img,n_at,3), Es (n_img,) -> tau
    (n_internal,n_at,3), each a unit (per-image, full-3N-normalized) vector.
    Bit-identical to improved_tangent() applied image-by-image."""
    dm = coord[1:-1] - coord[0:-2]
    dp = coord[2:] - coord[1:-1]
    Em, E, Ep = Es[0:-2], Es[1:-1], Es[2:]
    c1 = (Ep > E) & (E > Em)
    c2 = (Ep < E) & (E < Em)
    dEp = torch.clamp(Ep - E, min=0.0)
    dEm = torch.clamp(Em - E, min=0.0)
    t_else = dEp[:, None, None] * dp + dEm[:, None, None] * dm
    t = torch.where(c1[:, None, None], dp, torch.where(c2[:, None, None], dm, t_else))
    n_in = t.shape[0]
    norm = t.reshape(n_in, -1).norm(dim=1)
    fb1 = dp + dm
    nfb1 = fb1.reshape(n_in, -1).norm(dim=1)
    use1 = norm < 1e-16
    t = torch.where(use1[:, None, None], fb1, t)
    norm = torch.where(use1, nfb1, norm)
    use2 = norm < 1e-16                      # extreme fallback: unit e0 (flat idx 0)
    e0 = torch.zeros_like(t); e0[:, 0, 0] = 1.0
    t = torch.where(use2[:, None, None], e0, t)
    norm = torch.where(use2, torch.ones_like(norm), norm)
    return t / norm[:, None, None]


def _dyn_k_torch(Es, k_min, k_max, k_decay):
    """ORCA dynamic spring k per INTERNAL image (== compute_dynamic_k[1:-1])."""
    Em, Ei, Ep = Es[0:-2], Es[1:-1], Es[2:]
    dE = torch.clamp(Ei - torch.maximum(Em, Ep), min=0.0)
    dEmax = dE.max()
    k_dyn = k_max - (k_max - k_min) * torch.exp(-k_decay * (dE / dEmax.clamp(min=1e-30)))
    return torch.where(dEmax < 1e-10, torch.full_like(dE, k_max), k_dyn)


def _ew_k_torch(Es, k_min, k_max):
    """Energy-weighted spring k per INTERNAL image (== compute_energy_weighted_k[1:-1])."""
    E_ref = torch.maximum(Es[0], Es[-1]); E_max = Es.max(); denom = E_max - E_ref
    Ei = Es[1:-1]
    k_hi = k_max - (k_max - k_min) * (E_max - Ei) / denom.clamp(min=1e-30)
    cond = (denom > 1e-10) & (Ei > E_ref)
    return torch.where(cond, k_hi, torch.full_like(Ei, k_min))


def _band_forces_torch(coord, Es, rawF, k_internal, climbing, frozen_mask):
    """On-device per-band NEB projected forces == neb_band_forces(), vectorized.
    coord/rawF (n_img,n_at,3), Es (n_img,), k_internal (n_internal,). frozen_mask
    (n_internal,) bool (True=frozen) or None. Returns (Fp (n_internal,n_at,3),
    max_fp scalar tensor, hei0 0-dim long tensor = argmax internal-image index).
    hei stays ON DEVICE (no .item()) so the hot loop never syncs for HEI; the host
    int (hei0+1) is materialized only at harvest / for DyNEB."""
    n_in = coord.shape[0] - 2
    hei0 = torch.argmax(Es[1:-1])                              # 0-dim long, device
    tau = _improved_tangent_torch(coord, Es)
    F_true = rawF[1:-1]
    c = (F_true * tau).reshape(n_in, -1).sum(dim=1)
    F_perp = F_true - c[:, None, None] * tau
    dm = coord[1:-1] - coord[0:-2]; dp = coord[2:] - coord[1:-1]
    d_next = dp.reshape(n_in, -1).norm(dim=1); d_prev = dm.reshape(n_in, -1).norm(dim=1)
    Fp = F_perp + (k_internal * (d_next - d_prev))[:, None, None] * tau
    if climbing:
        Fp_ci = F_true - 2.0 * c[:, None, None] * tau          # HEI driven UP, no spring
        sel = torch.zeros(n_in, dtype=torch.bool, device=coord.device); sel[hei0] = True
        Fp = torch.where(sel[:, None, None], Fp_ci, Fp)
    if frozen_mask is not None:
        fm = frozen_mask.clone(); fm[hei0] = False             # never freeze HEI
        Fp = torch.where(fm[:, None, None], torch.zeros_like(Fp), Fp)
    max_fp = Fp.abs().reshape(n_in, -1).max() if n_in else torch.zeros((), device=coord.device, dtype=coord.dtype)
    return Fp, max_fp, hei0


class LBFGSDriverTorch:
    """torch-CUDA twin of LBFGSDriver (same two-loop / curvature / step cap), so
    on_device=True reproduces the numpy L-BFGS step to machine eps. All state
    (S, Y, rhos) lives on the GPU; no host sync in two_loop / update / step_limit."""
    def __init__(self, m=5, curvature=70.0, maxstep=0.2):
        self.m = m
        self.H0 = 1.0 / curvature
        self.maxstep = maxstep
        self.S, self.Y, self.rhos = [], [], []

    def two_loop(self, grad):
        q = grad.clone()
        alpha = []
        for s, y, rho in zip(reversed(self.S), reversed(self.Y), reversed(self.rhos)):
            a = rho * torch.dot(s, q); alpha.append(a); q = q - a * y
        gamma = self.H0 if not self.Y else torch.dot(self.Y[-1], self.S[-1]) / (torch.dot(self.Y[-1], self.Y[-1]) + 1e-20)
        z = gamma * q
        for s, y, rho, a in zip(self.S, self.Y, self.rhos, reversed(alpha)):
            b = rho * torch.dot(y, z); z = z + s * (a - b)
        return -z

    def update(self, s, y):
        rho = 1.0 / (torch.dot(y, s) + 1e-20)
        self.S.append(s.clone()); self.Y.append(y.clone()); self.rhos.append(rho)
        if len(self.S) > self.m:
            self.S.pop(0); self.Y.pop(0); self.rhos.pop(0)

    def step_limit(self, step):
        max_disp = step.abs().max()
        scale = torch.where(max_disp > self.maxstep,
                            self.maxstep / max_disp.clamp(min=1e-30),
                            torch.ones_like(max_disp))
        return step * scale

    def should_stop(self, grad, fmax_th=1e-3, frms_th=5e-4):
        """Returns a (device) bool tensor; the caller syncs once per iteration."""
        max_f = grad.abs().max()
        rms_f = torch.sqrt((grad * grad).mean())
        return (max_f < fmax_th) & (rms_f < frms_th)


# =============================================================================
# ------------------------------- NEB Class -----------------------------------
# =============================================================================

class NEB(JobABC):
    def __init__(self,
                output: str,
                atoms_or_molecules,
                paras: Optional[dict] = None):
        super().__init__(output)

        # Handle Molecules input
        if isinstance(atoms_or_molecules, Molecules):
            self.input_images = atoms_or_molecules.multiatoms
            self.atoms_R = None
            self.atoms_P = None
            # Capture the (optional) batched calculator carried by the Molecules
            # object (e.g. UMABatchCalc). If it exposes the batch contract
            # (prepare + get_ef_gpu) the whole band is evaluated in ONE forward
            # per NEB iteration; otherwise we fall back to per-image serial ASE
            # calculator reads (fully backward compatible).
            self._mol_calc = getattr(atoms_or_molecules, "calc", None)
        else:
            raise ValueError("Please provide Molecules object containing all images")

        self._use_batch = self._is_batch_calc(self._mol_calc)
        # Single-forward-per-geometry cache (band energies + raw forces).
        self._band_nmax = None
        self._band_cache_key = None
        self._band_cache_E = None
        self._band_cache_F = None

        # --- FIX #3: frozen-endpoint forward reuse (ACCURACY-PRESERVING) -------
        # The NEB endpoints (image 0 and n-1) never move (_pack/_unpack_internal
        # touch images 1..n-2 only), their projected forces are ZEROED in
        # neb_forces, and their energies are constant. So the endpoints are
        # evaluated ONCE (cached) and spliced into every later band forward; the
        # per-iter batched forward runs over INTERNAL images only (n_img-2 instead
        # of n_img). Default ON; MAPLE_NO_ENDPOINT_REUSE=1 forces the legacy
        # full-band forward every iter = the byte-parity oracle.
        self._reuse_endpoints = (os.environ.get("MAPLE_NO_ENDPOINT_REUSE", "0") != "1")
        self._ep_E = None          # (E0, En) cached endpoint energies
        self._ep_F = None          # (F0, Fn) cached endpoint raw forces
        self._ep_pos = None        # (pos0, posN) guard: re-eval if endpoints move
        self._neb_image_evals = 0  # diagnostics: total per-image forwards
        self._neb_ep_skipped = 0   # diagnostics: endpoint image-evals eliminated

        # Initialize params from paras dict
        self.params = self._init_params(NEBParams, paras, ("neb", "NEB", "ts"))

        # Safety: minimal guard
        if self.params.n_images < 1:
            raise ValueError("n_images must be >= 1")

        self.params.n_images = int(self.params.n_images) + 2  # total images including endpoints


    def optimize_endpoints(self, atoms_R: Atoms, atoms_P: Atoms, 
                        f_max_th=1.0e-3, f_rms_th=5.0e-4, max_iter=200):
        """
        Optimize the reactant and product endpoints before NEB if initial_opt=True.
        After optimization, write the minimized structures to XYZ files.
        """
        def single_point_optimize(atoms: Atoms):
            driver = LBFGSDriver(m=self.params.lbfgs_m, curvature=70.0, maxstep=self.params.step0)
            iteration = 0
            g = -to_numpy_f64(atoms.get_forces()).reshape(-1)
            x = to_numpy_f64(atoms.get_positions()).reshape(-1)

            while iteration < max_iter and not driver.should_stop(g, f_max_th, f_rms_th):
                p = driver.two_loop(g)
                p = driver.step_limit(p)
                x_new = x + p

                # update positions and forces
                atoms.set_positions(x_new.reshape(-1, 3))
                g_new = -to_numpy_f64(atoms.get_forces()).reshape(-1)
                driver.update(x_new - x, g_new - g)

                x = x_new
                g = g_new
                iteration += 1

            return atoms

        log_info(["\nInitial endpoint optimization started...\n"], self.output)
        atoms_R = single_point_optimize(atoms_R)
        atoms_P = single_point_optimize(atoms_P)
        log_info(["Initial endpoint optimization completed.\n"], self.output)

        # --- write optimized endpoints to XYZ ---
        base, ext = os.path.splitext(self.output)
        reactant_path = base + "_reactant_min.xyz"
        product_path  = base + "_product_min.xyz"
        write_xyz(reactant_path, [atoms_R], energies=[atoms_R.get_potential_energy(force_consistent=True)])
        write_xyz(product_path,  [atoms_P], energies=[atoms_P.get_potential_energy(force_consistent=True)])

        log_info([f"Optimized reactant written to: {reactant_path}\n"], self.output)
        log_info([f"Optimized product written to:  {product_path}\n"], self.output)

        return atoms_R, atoms_P

    def atoms_to_xyz(self, atoms: Atoms) -> str:
        lines = []
        syms = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        for s, (x, y, z) in zip(syms, pos):
            lines.append(f"{s:<2} {x:14.6f} {y:14.6f} {z:14.6f}")
        return "\n".join(lines) + "\n"

    # -------------------------- helpers to map x <-> images -------------------

    def _pack_internal(self, images: List[Atoms]) -> np.ndarray:
        """Flatten internal images (1..N-2) Cartesian into a single vector."""
        arrs = []
        for i in range(1, len(images) - 1):
            arrs.append(to_numpy_f64(images[i].get_positions()).reshape(-1))
        return np.concatenate(arrs) if arrs else np.zeros(0, dtype=np.float64)

    def _unpack_internal(self, x: np.ndarray, images: List[Atoms]):
        """Write back flattened x into internal images (1..N-2)."""
        offset = 0
        for i in range(1, len(images) - 1):
            n = len(images[i]) * 3
            Xi = x[offset:offset + n].reshape(-1, 3)
            images[i].set_positions(Xi)
            offset += n
            
    def _align_path(self, images: List[Atoms], ref_mode: str = "reactant"):
        """
        Rigid alignment for all internal images each iteration.
        ref_mode = "reactant"  → use images[0] as reference
                 = "centroid"  → use centroid of all images as reference
        Returns a list of RMSDs for internal images.
        """
        # choose reference coordinates
        if ref_mode == "reactant":
            ref = to_numpy_f64(images[0].get_positions())
        elif ref_mode == "centroid":
            coords = [to_numpy_f64(img.get_positions()) for img in images]
            ref = np.mean(coords, axis=0)
        else:
            ref = to_numpy_f64(images[0].get_positions())

        rmsds = []
        # align internal images only (1..N-2)
        for i in range(1, len(images) - 1):
            Q = to_numpy_f64(images[i].get_positions())
            Q_aligned, rmsd, _, _ = kabsch_align(ref, Q)
            images[i].set_positions(Q_aligned)
            rmsds.append(rmsd)
        return rmsds


    # ----------------------- batched band evaluation -------------------------

    @staticmethod
    def _is_batch_calc(calc) -> bool:
        """A batch calculator exposes ``prepare(atoms_list)`` + ``get_ef_gpu()``
        (UMABatchCalc / AIMNet2BatchCalc / MACE*BatchCalc contract)."""
        return (calc is not None
                and hasattr(calc, "prepare")
                and hasattr(calc, "get_ef_gpu"))

    @staticmethod
    def _geom_key(images: List[Atoms]):
        """Cheap fingerprint of the band geometry so a single batched forward is
        reused across the multiple E/F reads issued at one (unchanged) geometry
        within a NEB iteration (eval_grad + logging)."""
        h = hashlib.blake2b(digest_size=16)
        n = []
        for at in images:
            pos = np.ascontiguousarray(at.get_positions(), dtype=np.float64)
            n.append(pos.shape[0])
            h.update(pos.tobytes())
        return (tuple(n), h.digest())

    def _band_eval_batched(self, images: List[Atoms]):
        """ONE prepare + ONE forward over the WHOLE band (all images, endpoints
        included). Returns (Es: list[float], raw_F: list[(N_i,3) np.f64]).

        UMA's per-atom ``mol_idx`` segmentation keeps every image's graph
        block-diagonal, so a single batched forward yields each image's energy and
        forces independently (identical to evaluating them one at a time). Frozen
        endpoints carry constant positions and are simply part of the batch."""
        import torch  # local import: only needed on the batch path

        calc = self._mol_calc
        nmax = self._band_nmax
        n_img = len(images)

        # FIX #3: decide whether to skip the frozen endpoints. Reuse only when the
        # cache is warm AND the endpoint coordinates are genuinely unchanged (guard
        # against any path that re-optimises endpoints); else evaluate the full band
        # and (re)warm the endpoint cache from it.
        ep_ok = (self._reuse_endpoints and n_img >= 3 and self._ep_E is not None)
        if ep_ok:
            ep_ok = (np.array_equal(np.asarray(images[0].get_positions(), dtype=np.float64),
                                    self._ep_pos[0])
                     and np.array_equal(np.asarray(images[-1].get_positions(), dtype=np.float64),
                                        self._ep_pos[1]))

        if ep_ok:
            eval_idx = list(range(1, n_img - 1))   # INTERNAL images only
            eval_images = [images[i] for i in eval_idx]
        else:
            eval_idx = list(range(n_img))          # full band (warms endpoint cache)
            eval_images = images

        # fixed_nmax keeps the padded (B, nmax_dof) layout stable across iters
        # (band membership is constant; only positions move) -> matches the
        # batch-calc contract used by BatchPRFO.
        calc.prepare(eval_images, fixed_nmax=nmax)
        E_Ha, F_Ha = calc.get_ef_gpu()          # E (b,), F (b, nmax_dof)
        if self._band_nmax is None:
            self._band_nmax = int(F_Ha.shape[1])
        self._neb_image_evals += len(eval_images)
        if ep_ok:
            self._neb_ep_skipped += 2

        if isinstance(E_Ha, torch.Tensor):
            E_np = E_Ha.detach().to("cpu", torch.float64).numpy()
        else:
            E_np = np.asarray(E_Ha, dtype=np.float64)
        if isinstance(F_Ha, torch.Tensor):
            F_np = F_Ha.detach().to("cpu", torch.float64).numpy()
        else:
            F_np = np.asarray(F_Ha, dtype=np.float64)

        # scatter the evaluated rows into full-length (n_img) lists
        Es = [None] * n_img
        raw_F = [None] * n_img
        for slot, i in enumerate(eval_idx):
            n_i = len(images[i])
            Es[i] = float(E_np[slot])
            raw_F[i] = F_np[slot, :3 * n_i].reshape(n_i, 3).astype(np.float64, copy=True)

        if ep_ok:
            # splice the cached (constant) endpoint energies/forces; the endpoint
            # forces are unused by neb_forces (projected forces are zeroed there)
            # but are kept exact so any logging/diagnostics stay byte-consistent.
            Es[0], Es[-1] = self._ep_E
            raw_F[0], raw_F[-1] = self._ep_F
        else:
            # (re)warm the endpoint cache from this full-band evaluation
            self._ep_E = (Es[0], Es[-1])
            self._ep_F = (raw_F[0], raw_F[-1])
            self._ep_pos = (np.asarray(images[0].get_positions(), dtype=np.float64).copy(),
                            np.asarray(images[-1].get_positions(), dtype=np.float64).copy())
        return Es, raw_F

    def _band_eval(self, images: List[Atoms]):
        """Return (Es, raw_F) for the band, caching one forward per geometry.

        Batch path: single batched forward (UMABatchCalc). Serial path: per-image
        ASE ``get_potential_energy`` + ``get_forces`` (original behaviour)."""
        key = self._geom_key(images)
        if self._band_cache_key is not None and key == self._band_cache_key:
            return self._band_cache_E, self._band_cache_F

        if self._use_batch:
            Es, raw_F = self._band_eval_batched(images)
        else:
            Es = [float(at.get_potential_energy(force_consistent=True)) for at in images]
            raw_F = [to_numpy_f64(at.get_forces()) for at in images]

        self._band_cache_key, self._band_cache_E, self._band_cache_F = key, Es, raw_F
        return Es, raw_F

    def _band_forces(self, images: List[Atoms]):
        """Raw (calculator) forces per image for the batch path; ``None`` on the
        serial path so ``neb_forces`` keeps its original per-image reads."""
        if not self._use_batch:
            return None
        return self._band_eval(images)[1]

    def get_energies(self, imgs):
        if self._use_batch:
            return self._band_eval(imgs)[0]
        return [float(at.get_potential_energy(force_consistent=True)) for at in imgs]


    # =====================================================================
    # OPT-IN MULTI-BAND NEB  (additive; single-band run() stays the fallback
    # + parity oracle).  Image-as-batch over ALL bands: every image of every
    # still-active band is packed into ONE calc.prepare() + get_ef_gpu() per
    # iteration. UMA's per-atom mol_idx keeps each image block-diagonal, so the
    # batched forward yields each image's E/F independently (CatTSunami /
    # OCPNEB mechanic, Wander et al. arXiv:2405.02078, ACS Catal. 2024,
    # DOI 10.1021/acscatal.4c04272). Per-band L-BFGS state is NOT shared.
    # =====================================================================

    def _eval_flat_batched(self, flat_images: List[Atoms]):
        """ONE calc.prepare() + ONE get_ef_gpu() over an ARBITRARY flat list of
        images (multi-band path packs every image of every band here). Returns
        (Es, raw_F) aligned to ``flat_images``. Does NOT touch the single-band
        oracle cache (self._band_nmax / self._band_cache_*), keeping the
        fallback path independent."""
        import torch
        calc = self._mol_calc
        calc.prepare(flat_images, fixed_nmax=None)
        E_Ha, F_Ha = calc.get_ef_gpu()          # E (B,), F (B, nmax_dof)
        if isinstance(E_Ha, torch.Tensor):
            E_np = E_Ha.detach().to("cpu", torch.float64).numpy()
        else:
            E_np = np.asarray(E_Ha, dtype=np.float64)
        if isinstance(F_Ha, torch.Tensor):
            F_np = F_Ha.detach().to("cpu", torch.float64).numpy()
        else:
            F_np = np.asarray(F_Ha, dtype=np.float64)
        Es, raw_F = [], []
        for i, at in enumerate(flat_images):
            n_i = len(at)
            Es.append(float(E_np[i]))
            raw_F.append(F_np[i, :3 * n_i].reshape(n_i, 3).astype(np.float64, copy=True))
        return Es, raw_F

    def _prepare_band(self, band: List[Atoms]) -> List[Atoms]:
        """Build one ready NEB band (length n_required, endpoints included) from
        an input band, mirroring run()'s alignment + interpolation + IDPP so a
        multi-band band is prepared identically to the single-band path.

        Accepts either exactly n_required images (used as-is, only re-aligned) or
        exactly 2 endpoints (linear insertion plan + optional IDPP smoothing)."""
        n_required = self.params.n_images   # already includes the +2 endpoints
        band = list(band)
        if len(band) < 2:
            raise ValueError("each band needs >= 2 images (reactant + product)")
        if len(band) > n_required:
            raise ValueError(f"band has {len(band)} images > n_required={n_required}")

        atoms_R = band[0]
        atoms_P = band[-1]
        R_ref = to_numpy_f64(atoms_R.get_positions())
        P_aligned, _, _, _ = kabsch_align(R_ref, to_numpy_f64(atoms_P.get_positions()))
        atoms_P.set_positions(P_aligned)
        band[-1] = atoms_P
        for i in range(1, len(band) - 1):
            Q_aligned, _, _, _ = kabsch_align(R_ref, to_numpy_f64(band[i].get_positions()))
            band[i].set_positions(Q_aligned)

        if len(band) == n_required:
            images = band
        else:
            distances = self._compute_distances(band)
            plan = self._determine_insertion_plan(distances, n_required - len(band))
            images = self._insert_images_by_plan(band, plan)
            if self.params.ifidpp == 1:
                images = self._run_idpp_smoothing(images)

        self._align_path(images, ref_mode="reactant")
        if not self._use_batch:
            for img in images:
                if img.calc is None:
                    img.calc = self._mol_calc
        return images

    @staticmethod
    def _grad_from_fp(Fp_list, n_img, x_like):
        """Flatten -Fp over internal images into the L-BFGS gradient vector
        (identical packing to _pack_internal / run()'s eval_grad)."""
        grads = [(-Fp_list[i]).reshape(-1) for i in range(1, n_img - 1)]
        return np.concatenate(grads) if grads else np.zeros_like(x_like)

    def _update_frozen(self, st, Fp_list, fmax_th):
        """DyNEB: freeze any internal image (except the current HEI) whose NEB
        force max-component has dropped below fmax. Monotone (once frozen, stays
        frozen). Lindgren, Kastlunger & Peterson, JCTC 2019, 15, 5787,
        DOI 10.1021/acs.jctc.9b00633."""
        hei = st["hei"]
        # Keep the saddle neighbourhood {hei-1, hei, hei+1} ALWAYS active (the
        # point of DyNEB is to concentrate effort ON the saddle): re-activate any
        # of those that were frozen earlier, and never freeze them. Outer wing
        # images freeze monotonically once below fmax.
        protect = {hei - 1, hei, hei + 1}
        for i in list(st["frozen"]):
            if i in protect:
                st["frozen"].discard(i)
        for i in range(1, st["n_img"] - 1):
            if i in protect or i in st["frozen"]:
                continue
            if float(np.abs(Fp_list[i]).max()) < fmax_th:
                st["frozen"].add(i)

    def _project_frozen(self, st, x_new):
        """Hold the Cartesian DOFs of frozen internal images fixed at their
        current values (DyNEB), independent of any L-BFGS cross terms. Block
        layout matches _pack_internal / _unpack_internal."""
        x_old = st["x"]
        x_proj = x_new.copy()
        offset = 0
        for i in range(1, st["n_img"] - 1):
            n = len(st["images"][i]) * 3
            if i in st["frozen"]:
                x_proj[offset:offset + n] = x_old[offset:offset + n]
            offset += n
        return x_proj

    def run_multiband(self, bands: List[List[Atoms]],
                      dyneb: bool = False, climbing: bool = False,
                      spring_mode: str = "dynamic", verbose: bool = True,
                      pool_queue: Optional[List[List[Atoms]]] = None,
                      B_target: Optional[int] = None,
                      stall_patience: int = 40,
                      on_device: bool = False):
        """
        OPT-IN multi-band NEB: optimise B independent NEB bands concurrently,
        packing every image of every still-active band into ONE batched
        calc.prepare() + get_ef_gpu() per iteration (image-as-batch over bands;
        Wander et al. arXiv:2405.02078, ACS Catal. 2024,
        DOI 10.1021/acscatal.4c04272). Each band carries its OWN L-BFGS state
        (not shared across bands) and its own convergence test; the per-band
        update mirrors run() exactly, so with the defaults (spring_mode='dynamic',
        dyneb=False, climbing=False) each band reproduces the single-band run()
        trajectory (the parity oracle).

        Opt-in upgrades:
          dyneb=True     DyNEB image freezing (Lindgren, Kastlunger & Peterson,
                         JCTC 2019, 15, 5787, DOI 10.1021/acs.jctc.9b00633):
                         internal images already below fmax are frozen (zero NEB
                         force AND skipped in the batched forward -> fewer force
                         evals), concentrating MLIP calls on the saddle region.
          climbing=True  climbing-image force F - 2(F.tau)tau on the HEI
                         (Henkelman, Uberuaga & Jonsson, JCP 2000, 113, 9901,
                         DOI 10.1063/1.1329672).
          spring_mode    'dynamic' (ORCA, = run() default) | 'energy_weighted'
                         (Asgeirsson 2021, DOI 10.1021/acs.jctc.1c00462, stiffer
                         near barrier) | 'fixed' (constant k_max).

        Streaming pool (OPT-IN, mirrors BPRFO.run pool_queue/B_target; ports the
        same shrink+refill+straggler pattern to NEB so a 1000-band campaign does
        not stall on a few stuck bands):
          pool_queue     extra band specs (each [reactant, product] or a full
                         band, same as `bands`). When not None (with B_target),
                         the lockstep loop becomes a streaming pool: `bands` is
                         the initial active set (give B_target of them); when a
                         band leaves (converged / max_iter / stalled) the active
                         set is refilled from `pool_queue` up to B_target so the
                         batched forward stays saturated. Each band carries its
                         OWN per-band iteration budget (a newcomer entering at
                         global step 500 still gets p.max_iter of its own iters).
          B_target       active-band target the pool refills back up to.
          stall_patience a band whose max|grad| makes no progress (does not
                         improve) for this many of its OWN iters is evicted as
                         a straggler (does NOT block the rest of the batch).
        pool_queue=None  => EXACTLY the lockstep behaviour above (the oracle):
                            no refill, no straggler eviction, byte-identical
                            per-band barriers/HEIs to the pre-pool code.

        bands : List[List[Atoms]]   B input bands (see _prepare_band). With the
                pool on, this is the INITIAL active set (size B_target); extra
                bands go in pool_queue.

        Returns (results, stats):
          results : list of dict, ONE PER ORIGINAL band, ordered by original
                    index (initial `bands` first, then `pool_queue` in order),
                    with keys images, energies, hei, barrier_Eh, iterations,
                    converged, force_evals, n_img. With the pool on, each dict
                    also carries status in {converged, evicted_straggler,
                    max_iter}.
          stats   : dict forwards, total_image_evals, wall_time_s, n_bands,
                    max_iter_reached. With the pool on, also pool_refilled
                    (# bands pulled from the queue) and B_target.
        """
        if not self._use_batch:
            raise ValueError("run_multiband requires a batched calculator "
                             "(prepare + get_ef_gpu) on the Molecules object")
        if spring_mode not in ("dynamic", "energy_weighted", "fixed"):
            raise ValueError(f"unknown spring_mode={spring_mode!r}")

        # Streaming pool is enabled iff BOTH pool_queue and B_target are given.
        # pool_queue=None (or B_target=None) => the loop reduces EXACTLY to the
        # legacy lockstep behaviour (the byte-identical oracle).
        pooling = (pool_queue is not None) and (B_target is not None)
        if (pool_queue is not None) ^ (B_target is not None):
            raise ValueError("streaming pool needs BOTH pool_queue and B_target "
                             "(or NEITHER, for the lockstep oracle)")

        p = self.params
        base_case = (not dyneb) and (not climbing) and (spring_mode == "dynamic")

        # ---- ON-DEVICE inner loop (OPT-IN, default OFF = the numpy oracle) ----
        # Keeps the band's (n_img, n_at, 3) coords/forces + the L-BFGS state as
        # torch CUDA tensors, so tangent / spring / projection / two-loop /
        # convergence run as GPU ops with NO per-iter .cpu()/.numpy() of the
        # (n_img, nmax_dof) force tensor and NO numpy round-trip. The forward is
        # fed on-device via calc.set_coords_ (mirrors BatchPRFO): the band topology
        # is prepare()d once and re-prepared ONLY when the batch membership changes
        # (a band converges / the pool refills), eliminating the per-iter Atoms
        # write-back + graph rebuild too. Host syncs only for the per-band
        # convergence bool and the final harvest. MAPLE_NEB_DEV_NO_SETCOORDS=1
        # forces prepare()-every-iter (still on-device math; for A/B isolation).
        on_device = bool(on_device) and (torch is not None) and self._use_batch \
            and hasattr(self._mol_calc, "set_coords_")
        ddev = getattr(self._mol_calc, "device", None) if on_device else None
        ddtype = getattr(self._mol_calc, "dtype", None) if on_device else None
        dev_setcoords = on_device and (os.environ.get("MAPLE_NEB_DEV_NO_SETCOORDS", "0") != "1")
        # batched convergence sync: keep the per-band converged/HEI flags on device
        # and pull them to the host in ONE .cpu() per ITERATION (not per band).
        # MAPLE_NEB_DEV_PERBAND_SYNC=1 reverts to a per-band .item() (for A/B).
        dev_batched_sync = on_device and (os.environ.get("MAPLE_NEB_DEV_PERBAND_SYNC", "0") != "1")
        self._dev_fwd_key = None    # membership signature of the last prepared batch

        def _dev_sync_active(sts):
            # ONE host round-trip for the whole batch: stack the per-band converged
            # bools (device) and pull them together. Replaces B per-band .item()s.
            live = [st for st in sts if st.get("conv_dev") is not None]
            if not live:
                return
            flags = torch.stack([st["conv_dev"] for st in live]).to("cpu").tolist()
            for st, fl in zip(live, flags):
                st["converged"] = bool(fl)

        def _dev_init_state(st):
            """Attach the torch-CUDA twin fields (coords/Es/forces/x/g + torch
            L-BFGS) to a state dict already built by the numpy path."""
            imgs = st["images"]; n_img = st["n_img"]; n_at = len(imgs[0])
            coord = torch.tensor(
                np.stack([np.asarray(a.get_positions(), dtype=np.float64) for a in imgs]),
                dtype=ddtype, device=ddev)
            st["coord_t"] = coord
            st["n_at"] = n_at
            st["Es_t"] = torch.zeros(n_img, dtype=ddtype, device=ddev)
            st["rawF_t"] = torch.zeros((n_img, n_at, 3), dtype=ddtype, device=ddev)
            st["x_t"] = coord[1:-1].reshape(-1).clone()
            st["g_t"] = None
            st["x_new_t"] = None
            st["hei_dev"] = None        # 0-dim long (argmax internal idx), device
            st["conv_dev"] = None       # 0-dim bool (should_stop), device
            st["driver"] = LBFGSDriverTorch(m=p.lbfgs_m, curvature=70.0, maxstep=p.step0)

        def k_for(Es):
            if spring_mode == "dynamic":
                return compute_dynamic_k(Es, p.k_min, p.k_max, p.k_decay)
            if spring_mode == "energy_weighted":
                return compute_energy_weighted_k(Es, p.k_min, p.k_max)
            return [p.k_max] * len(Es)

        # ---- build + init per-band states ----
        states = []
        for b_idx, band in enumerate(bands):
            images = self._prepare_band(band)
            st = dict(
                idx=b_idx, images=images, n_img=len(images),
                driver=LBFGSDriver(m=p.lbfgs_m, curvature=70.0, maxstep=p.step0),
                x=self._pack_internal(images),
                Es=None, raw_F=None, g=None, Fp_list=None, x_new=None,
                k_springs=None, hei=1, converged=False, iters=0,
                frozen=set(), force_evals=0,
                # streaming-pool per-band straggler tracking (read only when
                # pooling; inert for the lockstep oracle).
                best_fmax=float("inf"), stall=0,
            )
            if on_device:
                _dev_init_state(st)
            states.append(st)

        # ---- one batched forward over a chosen subset of images per band ----
        def batched_forward(eval_plan):
            if on_device:
                return _dev_forward(eval_plan)
            flat, owner = [], []
            for st, idxs in eval_plan:
                for j in idxs:
                    flat.append(st["images"][j])
                    owner.append((st, j))
            if not flat:
                return 0
            Es_flat, F_flat = self._eval_flat_batched(flat)
            for (st, j), e, f in zip(owner, Es_flat, F_flat):
                st["Es"][j] = e
                st["raw_F"][j] = f
                st["force_evals"] += 1
            return len(flat)

        # ---- on-device forward: set_coords_ (steady) / prepare (membership change),
        #      results kept ON DEVICE (no .numpy() of the (B, nmax_dof) force tensor)
        def _dev_forward(eval_plan):
            plan = [(st, list(idxs)) for st, idxs in eval_plan if idxs]
            n_flat = sum(len(idxs) for _, idxs in plan)
            if not n_flat:
                return 0
            calc = self._mol_calc
            key = tuple((st["idx"], tuple(idxs)) for st, idxs in plan)
            if dev_setcoords and key == self._dev_fwd_key:
                # steady state: feed the new geometry on-device, no Atoms, no graph
                # rebuild. coord_t rows are gathered in the SAME flat (band, image)
                # order as the matching prepare() -> set_coords_ layout is consistent.
                flat_coords = torch.cat(
                    [st["coord_t"][idxs].reshape(-1, 3) for st, idxs in plan], dim=0)
                calc.set_coords_(flat_coords)
            else:
                # membership changed (or set_coords disabled): sync coord_t -> Atoms
                # for the images about to be prepared, then rebuild the topology.
                flat = []
                for st, idxs in plan:
                    cc = st["coord_t"][idxs].detach().to("cpu", torch.float64).numpy()
                    for slot_j, j in enumerate(idxs):
                        st["images"][j].set_positions(cc[slot_j])
                        flat.append(st["images"][j])
                calc.prepare(flat, fixed_nmax=None)
                self._dev_fwd_key = key
            E_t, F_t = calc.get_ef_gpu()        # E_t (B,), F_t (B, nmax_dof) ON DEVICE
            slot = 0
            for st, idxs in plan:                # per-band scatter (contiguous slots)
                k = len(idxs); n_i = st["n_at"]; sl = slice(slot, slot + k)
                st["Es_t"][idxs] = E_t[sl]
                st["rawF_t"][idxs] = F_t[sl, :3 * n_i].reshape(k, n_i, 3)
                st["force_evals"] += k
                slot += k
            return n_flat

        def _dev_band_forces(st):
            Es = st["Es_t"]
            if spring_mode == "dynamic":
                k_in = _dyn_k_torch(Es, p.k_min, p.k_max, p.k_decay)
            elif spring_mode == "energy_weighted":
                k_in = _ew_k_torch(Es, p.k_min, p.k_max)
            else:
                k_in = torch.full((st["n_img"] - 2,), p.k_max, dtype=ddtype, device=ddev)
            fm = None
            if dyneb and st["frozen"]:
                fm = torch.zeros(st["n_img"] - 2, dtype=torch.bool, device=ddev)
                for fi in st["frozen"]:
                    fm[fi - 1] = True
            return _band_forces_torch(st["coord_t"], Es, st["rawF_t"], k_in, climbing, fm)

        def _dev_project_frozen(st, x_new):
            if not st["frozen"]:
                return x_new
            xp = x_new.clone(); n3 = st["n_at"] * 3; xo = st["x_t"]
            for i in st["frozen"]:
                off = (i - 1) * n3
                xp[off:off + n3] = xo[off:off + n3]
            return xp

        def _dev_propose_step(st):
            drv = st["driver"]
            step = drv.step_limit(drv.two_loop(st["g_t"]))
            x_new = st["x_t"] + step
            if dyneb and st["frozen"]:
                x_new = _dev_project_frozen(st, x_new)
            st["coord_t"][1:-1] = x_new.reshape(st["n_img"] - 2, st["n_at"], 3)
            st["x_new_t"] = x_new

        def _dev_update_frozen(st, Fp, fmax_th):
            hei = int(st["hei_dev"].item()) + 1; st["hei"] = hei   # DyNEB needs host idx
            protect = {hei - 1, hei, hei + 1}
            for i in list(st["frozen"]):
                if i in protect:
                    st["frozen"].discard(i)
            pim = Fp.abs().reshape(Fp.shape[0], -1).max(dim=1).values.detach().to("cpu").numpy()
            for idx_in in range(Fp.shape[0]):
                i = idx_in + 1
                if i in protect or i in st["frozen"]:
                    continue
                if float(pim[idx_in]) < fmax_th:
                    st["frozen"].add(i)

        def _dev_set_conv(st, g_new):
            # store the (device) convergence bool; sync now (per-band mode) or defer
            # to the batched _dev_sync_active (default).
            st["conv_dev"] = st["driver"].should_stop(g_new, p.neb_f_max_th, p.neb_f_rms_th)
            if not dev_batched_sync:
                st["converged"] = bool(st["conv_dev"].item())

        def _dev_update_band(st):
            Fp, _, hei0 = _dev_band_forces(st)
            st["hei_dev"] = hei0; st["Fp_t"] = Fp
            g_new = (-Fp).reshape(-1)
            st["driver"].update(st["x_new_t"] - st["x_t"], g_new - st["g_t"])
            st["x_t"] = st["x_new_t"]
            st["g_t"] = g_new; st["g"] = g_new      # alias for pool fmax_of(st["g"])
            st["iters"] += 1
            _dev_set_conv(st, g_new)
            if dyneb:
                _dev_update_frozen(st, Fp, p.neb_f_max_th)

        def _dev_finalize(st):
            """Harvest sync: write the converged coords back to the Atoms (so the
            HEI geometry is exact for downstream P-RFO) and pull energies to host."""
            st["hei"] = int(st["hei_dev"].item()) + 1
            cc = st["coord_t"].detach().to("cpu", torch.float64).numpy()
            for j in range(st["n_img"]):
                st["images"][j].set_positions(cc[j])
            st["Es"] = st["Es_t"].detach().to("cpu", torch.float64).numpy().tolist()

        def _dev_init_grad(st):
            # post-(initial forward) gradient + convergence, on device. Mirrors the
            # numpy init block but using _dev_band_forces / torch should_stop.
            Fp, _, hei0 = _dev_band_forces(st)
            st["hei_dev"] = hei0; st["Fp_t"] = Fp
            g = (-Fp).reshape(-1); st["g_t"] = g; st["g"] = g
            _dev_set_conv(st, g)
            if dyneb:
                _dev_update_frozen(st, Fp, p.neb_f_max_th)

        def band_forces(st):
            if on_device:
                return _dev_band_forces(st)
            if base_case:
                # IDENTICAL call to run()'s eval_grad (dynamic-k recomputed from
                # the current energies inside neb_forces) -> per-band parity.
                return neb_forces(
                    st["images"], st["Es"], k_spring=None, k_springs=None,
                    use_dynamic_k=True, k_min=p.k_min, k_max=p.k_max,
                    k_decay=p.k_decay, raw_forces=st["raw_F"])
            ks = k_for(st["Es"])
            st["k_springs"] = ks
            mask = None
            if dyneb:
                mask = [True] * st["n_img"]
                for fi in st["frozen"]:
                    mask[fi] = False
            return neb_band_forces(
                st["images"], st["Es"], st["raw_F"], ks,
                climbing=climbing, hei_fixed=None, active_mask=mask)

        # ---- per-band step helpers (shared by the lockstep AND pool loops, so a
        #      pooled band follows the SAME trajectory as a lockstep band; these
        #      are verbatim factor-outs of the legacy loop body => byte-parity) ----
        def propose_step(st):
            # 1) propose x_new for one active band (no forward needed).
            if on_device:
                return _dev_propose_step(st)
            drv = st["driver"]
            step = drv.step_limit(drv.two_loop(st["g"]))
            x_new = st["x"] + step
            if dyneb and st["frozen"]:
                x_new = self._project_frozen(st, x_new)
            self._unpack_internal(x_new, st["images"])
            st["x_new"] = x_new

        def plan_idxs(st):
            # which images of this band to (re)evaluate in the batched forward.
            if dyneb:
                return [j for j in range(1, st["n_img"] - 1)
                        if (j not in st["frozen"]) or (j == st["hei"])]
            elif self._reuse_endpoints:
                # FIX #3: endpoints 0 / n-1 never move and their projected NEB
                # forces are zeroed -> evaluate INTERNAL images only.
                return list(range(1, st["n_img"] - 1))
            return list(range(st["n_img"]))   # full band (matches oracle)

        def update_band(st):
            # 3) per-band gradient update + convergence after the batched forward.
            if on_device:
                return _dev_update_band(st)
            Fp_list, _, hei = band_forces(st)
            st["hei"] = hei
            st["Fp_list"] = Fp_list
            g_new = self._grad_from_fp(Fp_list, st["n_img"], st["x_new"])
            st["driver"].update(st["x_new"] - st["x"], g_new - st["g"])
            st["x"] = st["x_new"]
            st["g"] = g_new
            st["iters"] += 1
            if st["driver"].should_stop(g_new, p.neb_f_max_th, p.neb_f_rms_th):
                st["converged"] = True
            if dyneb:
                self._update_frozen(st, Fp_list, p.neb_f_max_th)

        def fmax_of(g):
            if on_device:
                return float(g.abs().max().item()) if (g is not None and g.numel()) else 0.0
            return float(np.max(np.abs(g))) if g.size else 0.0

        def make_result(st, status=None):
            if on_device:
                _dev_finalize(st)          # coords -> Atoms, Es -> host (harvest sync)
            Es = st["Es"]
            hei = st["hei"]
            r = dict(
                idx=st["idx"], images=st["images"], energies=Es, hei=hei,
                barrier_Eh=float(Es[hei] - Es[0]),
                iterations=st["iters"], converged=st["converged"],
                force_evals=st["force_evals"], n_img=st["n_img"],
            )
            if status is not None:
                r["status"] = status
            return r

        def build_state(band_spec, oi):
            # construct a fresh per-band state for a pool newcomer (orig index oi).
            images = self._prepare_band(band_spec)
            st = dict(
                idx=oi, images=images, n_img=len(images),
                driver=LBFGSDriver(m=p.lbfgs_m, curvature=70.0, maxstep=p.step0),
                x=self._pack_internal(images),
                Es=[0.0] * len(images), raw_F=[None] * len(images),
                g=None, Fp_list=None, x_new=None,
                k_springs=None, hei=1, converged=False, iters=0,
                frozen=set(), force_evals=0,
                best_fmax=float("inf"), stall=0,
            )
            if on_device:
                _dev_init_state(st)
            return st

        # ---- INITIAL forward (full band for every band) ----
        for st in states:
            st["Es"] = [0.0] * st["n_img"]
            st["raw_F"] = [None] * st["n_img"]
        forwards = 0
        total_image_evals = 0
        t0 = time.time()
        total_image_evals += batched_forward(
            [(st, list(range(st["n_img"]))) for st in states])
        forwards += 1

        for st in states:
            if on_device:
                _dev_init_grad(st)
                continue
            Fp_list, _, hei = band_forces(st)
            st["hei"] = hei
            st["Fp_list"] = Fp_list
            g = self._grad_from_fp(Fp_list, st["n_img"], st["x"])
            st["g"] = g
            if st["driver"].should_stop(g, p.neb_f_max_th, p.neb_f_rms_th):
                st["converged"] = True
            if dyneb:
                self._update_frozen(st, Fp_list, p.neb_f_max_th)
        if dev_batched_sync:
            _dev_sync_active(states)

        if verbose:
            pool_note = (f"  POOL on: B_target={B_target} "
                         f"queue={len(pool_queue)} stall_patience={stall_patience}"
                         if pooling else "")
            log_info([
                f"\n{'='*70}\n",
                f"Multi-band NEB: B={len(states)} bands  spring_mode={spring_mode}  "
                f"dyneb={dyneb}  climbing={climbing}{pool_note}\n",
                f"{'='*70}\n",
                "Image-as-batch over all bands (UMA mol_idx block-diagonal); "
                "per-band L-BFGS; single-band run() kept as fallback + oracle.\n"
            ], self.output)

        if not pooling:
            # =================================================================
            # LOCKSTEP (legacy oracle): every band steps each iter until the
            # SLOWEST converges or the global iteration cap is hit. The step
            # helpers below are verbatim factor-outs of the original loop body,
            # so this path is byte-identical to the pre-pool code.
            # =================================================================
            it = 0
            while it < p.max_iter and not all(st["converged"] for st in states):
                it += 1
                active = [st for st in states if not st["converged"]]
                # 1) propose x_new for every active band (no forward needed)
                for st in active:
                    propose_step(st)
                # 2) ONE batched forward over the images that actually moved
                plan = [(st, plan_idxs(st)) for st in active]
                total_image_evals += batched_forward(plan)
                forwards += 1
                # 3) per-band gradient update + convergence
                for st in active:
                    update_band(st)
                if dev_batched_sync:
                    _dev_sync_active(active)    # ONE host round-trip for the batch

            wall = time.time() - t0

            # ---- finalize (one result per band, states order) ----
            results = []
            for st in states:
                results.append(make_result(st))
                if verbose:
                    Es = st["Es"]; hei = st["hei"]
                    extra = (f" frozen={len(st['frozen'])}/{st['n_img']-2}" if dyneb else "")
                    log_info([
                        f"band {st['idx']:>3d}: iters={st['iters']:>4d}  "
                        f"converged={st['converged']}  HEI={hei}  "
                        f"barrier={(Es[hei]-Es[0])*627.509:.3f} kcal/mol  "
                        f"force_evals={st['force_evals']}{extra}\n"
                    ], self.output)
            stats = dict(forwards=forwards, total_image_evals=total_image_evals,
                         wall_time_s=wall, n_bands=len(states),
                         max_iter_reached=(it >= p.max_iter))
            if verbose:
                log_info([
                    f"\nMulti-band done: {forwards} batched forwards, "
                    f"{total_image_evals} image-evals, wall={wall:.2f}s, iters={it}\n"
                ], self.output)
            return results, stats

        # =====================================================================
        # STREAMING POOL (opt-in; mirrors BPRFO.run pool_queue/B_target). The
        # active set is kept at <= B_target by refilling from `pool_queue` as
        # bands leave (converged / max_iter / stalled). Each band has its OWN
        # iteration budget. Straggler bands are evicted + flagged so they do not
        # block the rest of the batch. `bands` is the INITIAL active set.
        # =====================================================================
        queue = list(pool_queue)                 # remaining band specs (FIFO)
        Bt = int(B_target)
        next_orig = len(bands)                    # orig index for the next newcomer
        total_orig = len(bands) + len(queue)      # every original band gets a result
        refill_count = 0
        results_by_oi = {}                        # orig index -> result dict
        active = list(states)                     # live working set
        # seed straggler trackers from each initial band's starting |g|.
        for st in active:
            st["best_fmax"] = fmax_of(st["g"]) if st["g"] is not None else float("inf")
            st["stall"] = 0

        it = 0
        # finite-loop guard: each loop iter advances >=1 still-active band by one
        # of its own iters, and every band leaves at <= p.max_iter own-iters, so
        # the loop is bounded by total_orig * (p.max_iter + 1) (+slack).
        hard_cap = total_orig * (p.max_iter + 1) + 16

        while True:
            # ---- (a) classify + evict any band leaving the active set ----
            survivors = []
            for st in active:
                if st["converged"]:
                    status = "converged"
                elif st["iters"] >= p.max_iter:
                    status = "max_iter"
                elif st["stall"] >= stall_patience:
                    status = "evicted_straggler"
                else:
                    survivors.append(st)
                    continue
                results_by_oi[st["idx"]] = make_result(st, status)
                if verbose and status != "converged":
                    log_info([
                        f"[pool] evict band {st['idx']:>3d} as {status} "
                        f"(iters={st['iters']}, stall={st['stall']}, "
                        f"fmax={fmax_of(st['g']) if st['g'] is not None else float('nan'):.3e})\n"
                    ], self.output)
            active = survivors

            # ---- (b) refill the active set from the queue up to B_target ----
            newcomers = []
            while len(active) < Bt and queue:
                spec = queue.pop(0)
                st_new = build_state(spec, next_orig)
                next_orig += 1
                refill_count += 1
                newcomers.append(st_new)
                active.append(st_new)
            if newcomers:
                # initialise newcomers (full-band forward -> band_forces -> g),
                # exactly like the pre-loop init of the initial bands.
                total_image_evals += batched_forward(
                    [(st, list(range(st["n_img"]))) for st in newcomers])
                forwards += 1
                for st in newcomers:
                    if on_device:
                        _dev_init_grad(st)
                        st["best_fmax"] = fmax_of(st["g"])
                        st["stall"] = 0
                        continue
                    Fp_list, _, hei = band_forces(st)
                    st["hei"] = hei
                    st["Fp_list"] = Fp_list
                    g = self._grad_from_fp(Fp_list, st["n_img"], st["x"])
                    st["g"] = g
                    st["best_fmax"] = fmax_of(g)
                    st["stall"] = 0
                    if st["driver"].should_stop(g, p.neb_f_max_th, p.neb_f_rms_th):
                        st["converged"] = True
                    if dyneb:
                        self._update_frozen(st, Fp_list, p.neb_f_max_th)
                if dev_batched_sync:
                    _dev_sync_active(newcomers)
                if verbose:
                    log_info([
                        f"[pool] refill +{len(newcomers)} "
                        f"(active={len(active)}, queue_left={len(queue)})\n"
                    ], self.output)

            # ---- (c) termination: nothing active and nothing left to refill ----
            if not active:
                break
            if it >= hard_cap:
                # defensive: should be unreachable given the per-band budget.
                break

            # ---- (d) ONE global step over the current active set ----
            it += 1
            for st in active:
                propose_step(st)
            plan = [(st, plan_idxs(st)) for st in active]
            total_image_evals += batched_forward(plan)
            forwards += 1
            for st in active:
                update_band(st)              # advances iters + convergence
                # straggler progress tracking on this band's own |g|.
                fcur = fmax_of(st["g"])
                if fcur < st["best_fmax"] - 1e-6:
                    st["best_fmax"] = fcur
                    st["stall"] = 0
                else:
                    st["stall"] += 1
            if dev_batched_sync:
                _dev_sync_active(active)        # ONE host round-trip for the batch

        wall = time.time() - t0

        # ---- finalize: any band still active at the cap is flagged max_iter ----
        for st in active:
            if st["idx"] not in results_by_oi:
                results_by_oi[st["idx"]] = make_result(st, "max_iter")
        results = [results_by_oi[i] for i in range(total_orig)]
        if verbose:
            n_conv = sum(1 for r in results if r.get("status") == "converged")
            n_evict = sum(1 for r in results if r.get("status") == "evicted_straggler")
            n_max = sum(1 for r in results if r.get("status") == "max_iter")
            for r in results:
                Es = r["energies"]; hei = r["hei"]
                log_info([
                    f"band {r['idx']:>3d}: iters={r['iterations']:>4d}  "
                    f"status={r['status']:>17s}  HEI={hei}  "
                    f"barrier={(Es[hei]-Es[0])*627.509:.3f} kcal/mol  "
                    f"force_evals={r['force_evals']}\n"
                ], self.output)
            log_info([
                f"\nMulti-band POOL done: {forwards} batched forwards, "
                f"{total_image_evals} image-evals, wall={wall:.2f}s, "
                f"global_steps={it}, pool_refilled={refill_count}\n"
                f"# Final status: converged={n_conv} "
                f"evicted_straggler={n_evict} max_iter={n_max}\n"
            ], self.output)
        stats = dict(forwards=forwards, total_image_evals=total_image_evals,
                     wall_time_s=wall, n_bands=total_orig,
                     max_iter_reached=any(r.get("status") == "max_iter" for r in results),
                     pool_refilled=refill_count, B_target=Bt)
        return results, stats

    def _compute_distances(self, images: List[Atoms]) -> List[float]:
        """
        Compute straight-line distances between consecutive images.
        Returns list of distances with length = len(images) - 1
        """
        distances = []
        for i in range(len(images) - 1):
            pos1 = to_numpy_f64(images[i].get_positions())
            pos2 = to_numpy_f64(images[i+1].get_positions())
            dist = np.linalg.norm(pos2 - pos1)
            distances.append(dist)
        return distances

    def _determine_insertion_plan(self, distances: List[float], n_to_insert: int) -> List[Tuple[int, int]]:
        """
        Determine where to insert new images based on distances.
        
        Parameters
        ----------
        distances : List[float]
            Distances between consecutive images
        n_to_insert : int
            Total number of images to insert
        
        Returns
        -------
        List[Tuple[int, int]]
            List of (segment_index, count) indicating how many images to insert after segment_index
            
        Example: if distances = [4.0, 2.0, 1.0] and n_to_insert = 3
                largest gaps are at index 0 (4.0) and 1 (2.0)
                return [(0, 2), (1, 1)] means insert 2 after segment 0, 1 after segment 1
        """
        # Create list of (index, distance) and sort by distance (descending)
        indexed_distances = [(i, d) for i, d in enumerate(distances)]
        indexed_distances.sort(key=lambda x: x[1], reverse=True)
        
        # Distribute insertions proportionally to gap size
        insertion_counts = [0] * len(distances)
        
        for k in range(n_to_insert):
            # Insert in the largest remaining gap
            # Find segment with largest distance/insertions ratio
            max_ratio = -1
            max_idx = 0
            for i, d in enumerate(distances):
                ratio = d / (insertion_counts[i] + 1)
                if ratio > max_ratio:
                    max_ratio = ratio
                    max_idx = i
            insertion_counts[max_idx] += 1
        
        # Convert to list of (index, count) tuples
        plan = [(i, count) for i, count in enumerate(insertion_counts) if count > 0]
        return plan

    def _insert_images_by_plan(self, images: List[Atoms], plan: List[Tuple[int, int]]) -> List[Atoms]:
        """
        Insert interpolated images according to the insertion plan.
        
        Parameters
        ----------
        images : List[Atoms]
            Current list of images
        plan : List[Tuple[int, int]]
            Insertion plan from _determine_insertion_plan
        
        Returns
        -------
        List[Atoms]
            New list with interpolated images inserted
        """
        # Sort plan by index (descending) to insert from back to front
        plan_sorted = sorted(plan, key=lambda x: x[0], reverse=True)
        
        new_images = images.copy()
        
        for seg_idx, count in plan_sorted:
            # Insert 'count' images between new_images[seg_idx] and new_images[seg_idx+1]
            pos1 = to_numpy_f64(new_images[seg_idx].get_positions())
            pos2 = to_numpy_f64(new_images[seg_idx + 1].get_positions())
            
            inserted = []
            for k in range(1, count + 1):
                # Linear interpolation
                lam = k / (count + 1)
                new_pos = (1.0 - lam) * pos1 + lam * pos2
                
                # Copy atoms object and set new positions
                new_atom = new_images[seg_idx].copy()
                new_atom.set_positions(new_pos)
                new_atom.calc = new_images[seg_idx].calc  # Inherit calculator
                inserted.append(new_atom)
            
            # Insert all new images after seg_idx
            for idx, img in enumerate(inserted):
                new_images.insert(seg_idx + 1 + idx, img)
        
        return new_images
    
    # ---------------- Climbing Image NEB (CINEB) -------------------------------
    def cineb_forces(self, images: List[Atoms], energies: List[float], k_spring: float) -> Tuple[List[np.ndarray], float, int]:
            """
            Climbing Image NEB projected forces (Henkelman, Uberuaga &
            Jonsson, JCP 2000, 113, 9901, DOI: 10.1063/1.1329672):
            - normal NEB force for all non-endpoints except HEI
            - for HEI: remove spring force and reverse parallel component of true force

            The climbing image index can be "frozen" via self._cineb_fixed_hei:
            - if set and valid, use it
            - otherwise use the current highest-energy internal image
            """
            n_img = len(images)
            forces_proj = [None] * n_img
            max_fp = 0.0

            raw_forces = [to_numpy_f64(at.get_forces()) for at in images]
            coords = [to_numpy_f64(at.get_positions()) for at in images]
            Es = [float(e) for e in energies]

            inner_indices = list(range(1, n_img - 1))

            # --- choose HEI index: fixed one if present, else dynamic ---
            fixed_hei = getattr(self, "_cineb_fixed_hei", None)
            if (fixed_hei is not None) and (fixed_hei in inner_indices):
                hei_idx = fixed_hei
            else:
                hei_idx = max(inner_indices, key=lambda i: Es[i])

            for i in inner_indices:
                Rm1, R, Rp1 = coords[i - 1], coords[i], coords[i + 1]
                Em1, E, Ep1 = Es[i - 1], Es[i], Es[i + 1]

                tau = improved_tangent(Rm1, R, Rp1, Em1, E, Ep1)
                tau_resh = tau.reshape(-1, 3)
                F_true = vec1d(raw_forces[i])

                if i == hei_idx:
                    # ----------------------------------------------------------
                    # Climbing image force (ONLY true force, no projection)
                    # F_CI = F_true - 2 (F_true·tau) tau
                    # ----------------------------------------------------------
                    f_true = F_true                      # 1D (3N,)
                    tau_ci = tau / np.linalg.norm(tau)   # ensure normalized

                    ft = np.dot(f_true, tau_ci)
                    f_ci = f_true - 2.0 * ft * tau_ci

                    forces_proj[i] = f_ci.reshape(-1, 3)
                    continue

                else:
                    # Normal NEB projected force
                    F_true_perp = F_true - np.dot(F_true, tau) * tau
                    d_next = float(np.linalg.norm(Rp1 - R))
                    d_prev = float(np.linalg.norm(R - Rm1))
                    F_spring_par = k_spring * (d_next - d_prev) * tau_resh
                    Fp = F_true_perp.reshape(-1, 3) + F_spring_par

                forces_proj[i] = Fp
                max_fp = max(max_fp, float(np.abs(Fp).max()))

            # endpoints fixed
            forces_proj[0] = np.zeros_like(coords[0])
            forces_proj[-1] = np.zeros_like(coords[-1])

            return forces_proj, max_fp, hei_idx


    def restart_run(self, images: List[Atoms], energies: Optional[List[float]] = None):
        """
        Continue from a converged NEB path and perform:
        - CINEB refinement (always)
        - Optional TS refinement with RFO (if params.refine == 'nebts')

        Logic:
        - regular images: use projected NEB forces Fp for convergence
        - climbing image: use *true* forces F for convergence
        - climbing image index is frozen for the whole CINEB run
        """
        if energies is None:
            energies = [float(at.get_potential_energy(force_consistent=True)) for at in images]

        # reset any previous fixed HEI
        self._cineb_fixed_hei = None

        traj_file = os.path.splitext(self.output)[0] + "_cineb_traj.xyz"
        driver = LBFGSDriver(m=self.params.lbfgs_m, curvature=70.0, maxstep=self.params.cistep0)
        # set convergence thresholds
        driver.fmax_reg = self.params.neb_f_max_th
        driver.frms_reg = self.params.neb_f_rms_th
        driver.fmax_ci  = self.params.cineb_f_max_th
        driver.frms_ci  = self.params.cineb_f_rms_th

        def eval_grad(x_flat: np.ndarray) -> np.ndarray:
            """Return gradient dE/dx (flattened) for all internal images using CINEB forces."""
            self._unpack_internal(x_flat, images)
            Es_local = [float(at.get_potential_energy(force_consistent=True)) for at in images]
            Fp_list_local, _, _ = self.cineb_forces(images, Es_local, self.params.k_max)
            grads = [(-Fp_list_local[i]).reshape(-1) for i in range(1, len(images) - 1)]
            return np.concatenate(grads) if grads else np.zeros_like(x_flat)

        # initial packing / gradient
        x = self._pack_internal(images)
        g = eval_grad(x)
        iteration = 0

        # initial energies / forces for logging & to freeze HEI
        Es = [float(at.get_potential_energy(force_consistent=True)) for at in images]
        Fp_list, maxfp, hei = self.cineb_forces(images, Es, self.params.k_max)

        # freeze HEI index for the whole CINEB run
        self._cineb_fixed_hei = hei


        F_CI_vec = to_numpy_f64(images[hei].get_forces())
        maxF_CI = float(np.max(np.linalg.norm(F_CI_vec, axis=1)))
        rmsF_CI = float(np.sqrt(np.mean(np.linalg.norm(F_CI_vec, axis=1) ** 2)))

        log_info([
            "\nStarting CINEB refinement:\n",
            "Optim.  Iteration  CI   E(CI)-E(0)   max(|Fp|)   RMS(Fp)   max(|FCI|)   RMS(|FCI|)\n",
            f"Convergence thresholds regular: {self.params.neb_f_max_th: .6f}/{self.params.neb_f_rms_th: .6f},  "
            f"CI: {self.params.cineb_f_max_th: .6f}/{self.params.cineb_f_rms_th: .6f}\n"
        ], self.output)

        while iteration < self.params.max_iter:
            # ===== convergence check at current geometry =====
            # regular Fp: all internal non-CI images
            inner_idx = list(range(1, len(images) - 1))
            regular_flat = []
            for i in inner_idx:
                if i == hei:
                    continue
                regular_flat.append(Fp_list[i].reshape(-1))
            Fp_all = np.concatenate(regular_flat) if regular_flat else np.zeros(0, dtype=np.float64)

            # true forces on CI (no projection)
            F_CI = to_numpy_f64(images[hei].get_forces()).reshape(-1)

            if driver.ci_should_stop(Fp_all, F_CI):
                log_info([f"\nCINEB converged after {iteration} iterations.\n"], self.output)
                break

            # ===== L-BFGS step using CINEB gradient =====
            p = driver.two_loop(g)
            p = driver.step_limit(p)

            # take full step (no line search)
            x_new = x + p

            # evaluate new gradient
            g_new = eval_grad(x_new)


            driver.update(x_new - x, g_new - g)

            x = x_new
            g = g_new

            # ===== recompute energies / forces for next iteration & logging =====
            Es = [float(at.get_potential_energy(force_consistent=True)) for at in images]
            Fp_list, maxfp, _ = self.cineb_forces(images, Es, self.params.k_max)

            F_CI_vec = to_numpy_f64(images[hei].get_forces())
            maxF_CI = float(np.max(np.linalg.norm(F_CI_vec, axis=1)))
            rmsF_CI = float(np.sqrt(np.mean(np.linalg.norm(F_CI_vec, axis=1) ** 2)))
            rmsfp = rms_force(Fp_list)
            dE_CI = Es[hei] - Es[0]

            write_all_images_xyz(traj_file, images, energies=Es, iteration=iteration)
            log_info([f"   LBFGS {iteration:>4d} {hei:>6d} {dE_CI:>10.6f} "
                      f"{maxfp:>11.6f} {rmsfp:>10.6f} {maxF_CI:>10.6f} {rmsF_CI:>10.6f}\n"], self.output)

            iteration += 1
        else:
            log_info(["\nCINEB refinement reached maximum iterations.\n"], self.output)

        # Clean up fixed HEI
        self._cineb_fixed_hei = None

        # --- Stage 1 summary: CI part ---
        base, _ = os.path.splitext(self.output)
        cineb_mep = base + "_cineb_mep.xyz"
        cineb_hei = base + "_cineb_hei.xyz"
        write_xyz(cineb_mep, images, energies=Es)
        write_xyz(cineb_hei, [images[hei]], energies=[Es[hei]])

        log_info([
            "\n---------------------------------------------------------------\n",
            "               INFORMATION ABOUT SADDLE POINT     \n",
            "---------------------------------------------------------------\n",
            f"Climbing image                            ....  {hei}\n",
            f"Energy                                    ....  {Es[hei]: .8f} Eh\n",
            f"Max. abs. force                           ....  {maxF_CI: .4e} Eh/Angstrom\n",
            "\n-----------------------------------------\n",
            "  SADDLE POINT (ANGSTROEM)\n",
            "-----------------------------------------\n",
            self.atoms_to_xyz(images[hei])
        ], self.output)

        # --- Stage 2: Optional PRFO refinement ---
        if self.params.refine == 'nebts':
            from .PRFO import PRFO

            # Use CI geometry as TS guess
            ts_guess = images[hei].copy()
            ts_guess.calc = self.atoms_R.calc
            ts_guess.f_max_th = images[0].f_max_th
            ts_guess.f_rms_th = images[0].f_rms_th
            ts_guess.dp_max_th = images[0].dp_max_th
            ts_guess.dp_rms_th = images[0].dp_rms_th

            prfo = PRFO(output=self.output, atoms=ts_guess)
            ts_opt = prfo.run()

            E_TS = ts_opt.get_potential_energy(force_consistent=True)
            maxF_TS = np.max(np.linalg.norm(ts_opt.get_forces(), axis=1))
            rmsF_TS = np.sqrt(np.mean(np.linalg.norm(ts_opt.get_forces(), axis=1) ** 2))

            # Insert TS right after CI
            images.insert(hei + 1, ts_opt)
            Es.insert(hei + 1, E_TS)

            nebts_mep = base + "_nebts_mep.xyz"
            nebts_ts = base + "_nebts_ts.xyz"
            write_xyz(nebts_mep, images, energies=Es)
            write_xyz(nebts_ts, [ts_opt], energies=[E_TS])

            log_info([
                "\n---------------------------------------------------------------\n",
                "                      PATH SUMMARY FOR NEB-TS             \n",
                "---------------------------------------------------------------\n",
                "All forces in Eh/Angstrom. Global forces for TS.\n\n",
                "Image     E(Eh)   dE(kcal/mol)  max(|Fp|)  RMS(Fp)\n"
            ], self.output)

            kcal_per_Eh = 627.509
            for i, E in enumerate(Es):
                dE = (E - Es[0]) * kcal_per_Eh
                label = " TS" if i == hei + 1 else f"{i:3d}"
                marker = " <= TS" if i == hei + 1 else (" <= CI" if i == hei else "")
                maxF = np.max(np.linalg.norm(images[i].get_forces(), axis=1))
                rmsF = np.sqrt(np.mean(np.linalg.norm(images[i].get_forces(), axis=1) ** 2))
                log_info([
                    f"{label:>4s} {E:12.5f} {dE:11.2f} {maxF:11.5f} {rmsF:10.5f}{marker}\n"
                ], self.output)

            log_info([
                "\n-----------------------------------------\n",
                "  REFINED TS STRUCTURE (ANGSTROEM)\n",
                "-----------------------------------------\n",
                self.atoms_to_xyz(ts_opt)
            ], self.output)

            log_info([
                f"\nWrote NEB-TS MEP to: {nebts_mep}\n",
                f"Wrote TS structure to: {nebts_ts}\n"
            ], self.output)



    # ==============================================================
    # IDPP smoothing for initial path
    # ==============================================================
    def _run_idpp_smoothing(self, images, idpp_params=IDPPParams()):
        """
        Strong IDPP smoothing (BNEB-style):
        - LBFGS optimization instead of GD
        - damping + curvature correction
        - step clipping
        - projected force formulation identical to pysisyphus/BNEB

        This function preserves your image list + ASE Atoms interface.
        """

        # ----- extract coords -----
        coords = [to_numpy_f64(img.get_positions()) for img in images]
        R0 = coords[0]
        R1 = coords[-1]

        n_img = len(images)
        n_inner = n_img - 2
        n_atoms = R0.shape[0]

        # pair list (upper-triangle)
        pairs = _pair_indices(n_atoms)
        n_pair = pairs.shape[0]

        # ----- IDPP target 1/dist -----
        targets = _idpp_targets(R0, R1, n_inner)  # list length = n_inner

        # ----- flatten internal images -----
        x = np.concatenate([coords[i].reshape(-1) for i in range(1, n_img - 1)])

        # ----- LBFGS state -----
        m = 7
        S = []
        Y = []
        rho = []
        maxstep = 0.1
        curvature = 1.0

        def lbfgs_direction(g):
            """Two-loop recursion."""
            q = g.copy()
            alpha = []

            for s, y, r in reversed(list(zip(S, Y, rho))):
                a = r * np.dot(s, q)
                alpha.append(a)
                q -= a * y

            if len(Y) > 0:
                gamma = np.dot(S[-1], Y[-1]) / (np.dot(Y[-1], Y[-1]) + 1e-20)
            else:
                gamma = 1.0 / curvature

            z = gamma * q

            for (s, y, r), a in zip(zip(S, Y, rho), reversed(alpha)):
                b = r * np.dot(y, z)
                z += s * (a - b)

            return -z

        def compute_grad(x):
            """Compute IDPP gradient for all internal images."""
            grad = np.zeros_like(x)

            offset = 0
            for k in range(n_inner):
                Xi = x[offset:offset + n_atoms * 3].reshape(n_atoms, 3)

                # pair distances
                Rij = Xi[pairs[:, 0]] - Xi[pairs[:, 1]]      # (M,3)
                dij = np.linalg.norm(Rij, axis=1) + 1e-12
                inv = 1.0 / dij

                diff = inv - targets[k]                     # (M,)
                dE_dd = -2.0 * diff / (dij**2)              # derivative wrt dij

                g_pair = (dE_dd[:, None] * (Rij / dij[:, None]))  # (M,3)

                # accumulate forces
                g_atoms = np.zeros_like(Xi)
                for p in range(n_pair):
                    i, j = pairs[p]
                    g_atoms[i] +=  g_pair[p]
                    g_atoms[j] += -g_pair[p]

                grad[offset:offset + n_atoms * 3] = g_atoms.reshape(-1)
                offset += n_atoms * 3

            return grad

        # ----- optimization loop -----
        grad = compute_grad(x)

        for _ in range(idpp_params.max_steps):
            g = grad.copy()
            step = lbfgs_direction(g)

            # step clipping
            max_disp = np.max(np.abs(step))
            if max_disp > maxstep:
                step *= maxstep / max_disp

            x_new = x + step
            grad_new = compute_grad(x_new)

            # LBFGS update
            s = x_new - x
            y = grad_new - grad

            sy = np.dot(s, y)
            if sy > 1e-12:
                if len(S) == m:
                    S.pop(0); Y.pop(0); rho.pop(0)
                S.append(s); Y.append(y); rho.append(1.0 / sy)

            x = x_new
            grad = grad_new

            # stopping based on RMS gradient
            if np.sqrt(np.mean(grad * grad)) < idpp_params.eps:
                break

        # ----- write back optimized coords -----
        offset = 0
        for i in range(1, n_img - 1):
            Xi = x[offset:offset + n_atoms * 3].reshape(n_atoms, 3)
            images[i].set_positions(Xi)
            offset += n_atoms * 3

        return images



    # ------------------------------- main flow --------------------------------

    def run(self):
        # ===================================================================
        # Step 0: Process input images and determine if interpolation needed
        # ===================================================================
        n_input = len(self.input_images)
        n_required = self.params.n_images  # This is already n_images + 2
        
        log_info([
            f"\n{'='*70}\n",
            f"NEB Initialization\n",
            f"{'='*70}\n",
            f"Input images:    {n_input}\n",
            f"Required images: {n_required} (including endpoints)\n"
        ], self.output)
        
        # Check input validity
        if n_input < 2:
            raise ValueError(f"Need at least 2 images (reactant + product), got {n_input}")
        
        if n_input > n_required:
            raise ValueError(
                f"Too many input images: got {n_input}, but n_images={self.params.n_images-2} "
                f"requires exactly {n_required} images (including endpoints). "
                f"Please reduce input images or increase n_images parameter."
            )
        
        # Assign endpoints
        self.atoms_R = self.input_images[0]
        self.atoms_P = self.input_images[-1]
        
        # ===================================================================
        # CRITICAL: Align all input images BEFORE any interpolation/processing
        # This ensures interpolation happens in the correct coordinate system
        # ===================================================================
        log_info(["\nAligning input images to reactant reference...\n"], self.output)
        
        # Align product endpoint to reactant
        R_ref = to_numpy_f64(self.atoms_R.get_positions())
        P = to_numpy_f64(self.atoms_P.get_positions())
        P_aligned, rmsd_product, _, _ = kabsch_align(R_ref, P)
        self.atoms_P.set_positions(P_aligned)
        self.input_images[-1] = self.atoms_P  # Update in input list
        
        log_info([f"Product aligned to reactant. RMSD: {rmsd_product:.6f} Angstrom\n"], self.output)
        
        # Align all intermediate input images (if any exist)
        if n_input > 2:
            log_info([f"Aligning {n_input - 2} intermediate input image(s)...\n"], self.output)
            for i in range(1, n_input - 1):
                Q = to_numpy_f64(self.input_images[i].get_positions())
                Q_aligned, rmsd_i, _, _ = kabsch_align(R_ref, Q)
                self.input_images[i].set_positions(Q_aligned)
                log_info([f"  Image {i} aligned. RMSD: {rmsd_i:.6f} Angstrom\n"], self.output)
        
        log_info(["Input alignment completed.\n"], self.output)
        
        # ===================================================================
        # Determine if interpolation is needed
        # ===================================================================
        
        # Case 1: Exact number of images - use directly (already aligned)
        if n_input == n_required:
            log_info([f"\nExact number of images provided. Using input directly.\n"], self.output)
            images = self.input_images
            
        # Case 2: Need to insert additional images via interpolation
        else:
            n_to_insert = n_required - n_input
            log_info([
                f"\nNeed to insert {n_to_insert} additional image(s).\n",
                f"Analyzing distances to determine optimal insertion points...\n"
            ], self.output)
            
            # Compute straight-line distances between consecutive input images
            distances = self._compute_distances(self.input_images)
            
            log_info(["\nDistances between consecutive input images:\n"], self.output)
            for i, d in enumerate(distances):
                log_info([f"  Image {i} -> {i+1}: {d:.4f} Angstrom\n"], self.output)
            
            # Determine optimal insertion plan (prioritize largest gaps)
            insertion_plan = self._determine_insertion_plan(distances, n_to_insert)
            
            log_info(["\nInsertion plan:\n"], self.output)
            for seg_idx, count in sorted(insertion_plan, key=lambda x: x[0]):
                log_info([f"  Insert {count} image(s) between image {seg_idx} and {seg_idx+1}\n"], self.output)
            
            # Perform linear interpolation according to plan
            images = self._insert_images_by_plan(self.input_images, insertion_plan)
            
            log_info([f"\nTotal images after insertion: {len(images)}\n"], self.output)
            
            # Apply IDPP smoothing to refine interpolated path (optional)
            if self.params.ifidpp == 1:
                log_info(["\nApplying IDPP smoothing to interpolated images...\n"], self.output)
                images = self._run_idpp_smoothing(images)
                log_info(["IDPP smoothing completed.\n"], self.output)
            else:
                log_info(["\nIDPP smoothing disabled (ifidpp=0). Using linear interpolation only.\n"], self.output)
        
        # ===================================================================
        # CRITICAL: Re-align entire path after interpolation/IDPP
        # This ensures the complete path is properly aligned before NEB starts
        # ===================================================================
        log_info(["\nPerforming final Kabsch alignment of all images to reactant...\n"], self.output)
        self._align_path(images, ref_mode="reactant")
        log_info(["Final alignment completed.\n"], self.output)
        

        # ===================================================================
        # Step 1: Optional endpoint optimization
        # ===================================================================
        if self.params.initial_opt and self._use_batch:
            log_info([
                "\ninitial_opt requested but skipped on the batched-calculator path\n",
                "(endpoint pre-optimization uses per-image serial ASE forces; not\n",
                "wired for the batch calculator in this phase). Endpoints used as-is.\n"
            ], self.output)
        elif self.params.initial_opt:
            self.atoms_R, self.atoms_P = self.optimize_endpoints(
                images[0], images[-1],
                f_max_th=self.params.neb_f_max_th,
                f_rms_th=self.params.neb_f_rms_th,
                max_iter=self.params.max_iter
            )
            images[0] = self.atoms_R
            images[-1] = self.atoms_P
        
        # ===================================================================
        # Step 2: Endpoint properties and alignment
        # ===================================================================
        def forces_info(F):
            F = to_numpy_f64(F)
            maxF = np.max(np.linalg.norm(F, axis=1))
            rmsF = np.sqrt(np.mean(np.linalg.norm(F, axis=1) ** 2))
            return maxF, rmsF

        if self._use_batch:
            # One batched forward over the whole band; pull endpoint props from it.
            Es_band0, F_band0 = self._band_eval(images)
            E_R, E_P = Es_band0[0], Es_band0[-1]
            maxF_R, rmsF_R = forces_info(F_band0[0])
            maxF_P, rmsF_P = forces_info(F_band0[-1])
        else:
            E_R = images[0].get_potential_energy(force_consistent=True)
            E_P = images[-1].get_potential_energy(force_consistent=True)
            maxF_R, rmsF_R = forces_info(images[0].get_forces())
            maxF_P, rmsF_P = forces_info(images[-1].get_forces())

        log_info([
            "\nProperties of fixed NEB end points:\n",
            "               Reactant:\n",
            f"                         E               ....   {E_R: .6f} Eh\n",
            f"                         RMS(F)          ....   {rmsF_R: .6f} Eh/Angstrom\n",
            f"                         MAX(|F|)        ....   {maxF_R: .6f} Eh/Angstrom\n",
            "               Product:\n",
            f"                         E               ....   {E_P: .6f} Eh\n",
            f"                         RMS(F)          ....   {rmsF_P: .6f} Eh/Angstrom\n",
            f"                         MAX(|F|)        ....   {maxF_P: .6f} Eh/Angstrom\n",
            "\nReactant XYZ (Angstrom):\n",
            self.atoms_to_xyz(images[0]),
            "\nProduct XYZ (Angstrom):\n",
            self.atoms_to_xyz(images[-1]),
            "\n"
        ], self.output)

        # Alignment
        log_info(["\nPerforming Kabsch alignment of all images...\n"], self.output)
        self._align_path(images, ref_mode="reactant")
        log_info(["Alignment completed.\n"], self.output)
        
        # Ensure all images have calculators
        for img in images:
            if img.calc is None:
                img.calc = self.atoms_R.calc

        # ===================================================================
        # Step 3: Normal NEB optimization loop
        # ===================================================================
        
        traj_file = os.path.splitext(self.output)[0] + "_image_traj.xyz"
        
        driver = LBFGSDriver(
            m=self.params.lbfgs_m,
            curvature=70.0,
            maxstep=self.params.step0
        )
        
        self._k_springs_history = []
        
        def eval_grad(x_flat):
            self._unpack_internal(x_flat, images)
            # ONE prepare + ONE forward over the whole band per geometry; both the
            # energies and the raw forces come from this single batched evaluation
            # (serial fallback when no batch calculator).
            Es, raw_F = self._band_eval(images) if self._use_batch else (self.get_energies(images), None)

            Fp_list, _, _ = neb_forces(
                images, Es,
                k_spring=None,
                k_springs=None,
                use_dynamic_k=self.params.use_dynamic_k,
                k_min=self.params.k_min,
                k_max=self.params.k_max,
                k_decay=self.params.k_decay,
                raw_forces=raw_F
            )

            grads = [(-Fp_list[i]).reshape(-1) for i in range(1, len(images) - 1)]
            return np.concatenate(grads) if grads else np.zeros_like(x_flat)

        x = self._pack_internal(images)
        g = eval_grad(x)
        iteration = 0

        Es = self.get_energies(images)
        raw_F = self._band_forces(images)

        if self.params.use_dynamic_k:
            k_springs = compute_dynamic_k(Es, self.params.k_min, self.params.k_max, self.params.k_decay)
            self._k_springs_history.append(k_springs.copy())
        else:
            k_springs = [self.params.k_max] * len(images)

        Fp_list, maxfp, hei = neb_forces(
            images, Es,
            k_spring=None,
            k_springs=k_springs,
            use_dynamic_k=False,
            raw_forces=raw_F
        )
        rmsfp = rms_force(Fp_list)
        dE_hei = Es[hei] - Es[0]
        
        # Log header with initial state
        if self.params.use_dynamic_k:
            k_hei = k_springs[hei]
            log_info([
                "\nStarting NEB iterations with ORCA-style dynamic spring constants:\n",
                f"k_min = {self.params.k_min:.4f}, k_max = {self.params.k_max:.4f}, k_decay = {self.params.k_decay:.4f}\n",
                "Optim.  Iteration  HEI  E(HEI)-E(0)  max(|Fp|)   RMS(Fp)   k(HEI)\n",
                f"Convergence thresholds         {self.params.neb_f_max_th: .6f}   {self.params.neb_f_rms_th: .6f}\n",
                f"Initial {iteration:>4d} {hei:>6d} {dE_hei:>10.6f} {maxfp:>11.6f} {rmsfp:>10.6f}   {k_hei:.4f}\n"
            ], self.output)
        else:
            log_info([
                "\nStarting NEB iterations with fixed spring constant:\n",
                f"k_spring = {self.params.k_max:.4f}\n",
                "Optim.  Iteration  HEI  E(HEI)-E(0)  max(|Fp|)   RMS(Fp)\n",
                f"Convergence thresholds         {self.params.neb_f_max_th: .6f}   {self.params.neb_f_rms_th: .6f}\n",
                f"Initial {iteration:>4d} {hei:>6d} {dE_hei:>10.6f} {maxfp:>11.6f} {rmsfp:>10.6f}\n"
            ], self.output)
        
        # Write initial trajectory
        write_all_images_xyz(traj_file, images, energies=Es, iteration=iteration)
        
        # ===== Main optimization loop =====
        while iteration < self.params.max_iter and not driver.should_stop(g, self.params.neb_f_max_th, self.params.neb_f_rms_th):
            iteration += 1  # Increment at start to avoid confusion
            
            # L-BFGS step
            p = driver.two_loop(g)
            p = driver.step_limit(p)
            
            x_new = x + p
            g_new = eval_grad(x_new)
            
            driver.update(x_new - x, g_new - g)
            
            x = x_new
            g = g_new
            
            # Compute forces and energies for logging. Same geometry as the just
            # finished eval_grad(x_new) -> the per-geometry cache returns without a
            # second forward (still ONE forward per NEB iteration).
            Es = self.get_energies(images)
            raw_F = self._band_forces(images)

            if self.params.use_dynamic_k:
                k_springs = compute_dynamic_k(Es, self.params.k_min, self.params.k_max, self.params.k_decay)
                self._k_springs_history.append(k_springs.copy())
            else:
                k_springs = [self.params.k_max] * len(images)

            Fp_list, maxfp, hei = neb_forces(
                images, Es,
                k_spring=None,
                k_springs=k_springs,
                use_dynamic_k=False,
                raw_forces=raw_F
            )
            
            rmsfp = rms_force(Fp_list)
            dE_hei = Es[hei] - Es[0]
            
            write_all_images_xyz(traj_file, images, energies=Es, iteration=iteration)
            
            if self.params.use_dynamic_k:
                k_hei = k_springs[hei]
                log_info([f"   LBFGS {iteration:>4d} {hei:>6d} {dE_hei:>10.6f} {maxfp:>11.6f} {rmsfp:>10.6f}   {k_hei:.4f}\n"], self.output)
            else:
                log_info([f"   LBFGS {iteration:>4d} {hei:>6d} {dE_hei:>10.6f} {maxfp:>11.6f} {rmsfp:>10.6f}\n"], self.output)
        
        if iteration == 0:
            log_info(["\nNEB already converged at initial geometry (iteration 0).\n"], self.output)
        elif iteration == self.params.max_iter:
            log_info(["\nNEB optimization reached maximum iterations.\n"], self.output)
        else:
            log_info([f"\nNEB optimization converged after {iteration} iterations.\n"], self.output)


        # ---------------------------------------------------------------
        #  Final path summary
        # ---------------------------------------------------------------
        base, _ = os.path.splitext(self.output)
        mep_path = base + "_mep.xyz"
        hip_path = base + "_hei.xyz"
        write_xyz(mep_path, images, energies=Es)
        write_xyz(hip_path, [images[hei]], energies=[Es[hei]])

        # Compute straight-line distances
        distances = [np.linalg.norm(images[i+1].get_positions() - images[i].get_positions()) for i in range(len(images)-1)]

        kcal_per_Eh = 627.509
        summary = [
            "\n---------------------------------------------------------------\n",
            "                         PATH SUMMARY              \n",
            "---------------------------------------------------------------\n",
            "All forces in Eh/Angstrom.\n\n",
            "Image Dist.(Ang.)    E(Eh)   dE(kcal/mol)  max(|Fp|)  RMS(Fp)\n"
        ]

        for i, (E, Fp) in enumerate(zip(Es, Fp_list)):
            dE = (E - Es[0]) * kcal_per_Eh
            dist = 0.0 if i == 0 else distances[i-1]
            maxF = np.max(np.linalg.norm(Fp, axis=1))
            rmsF = np.sqrt(np.mean(np.linalg.norm(Fp, axis=1) ** 2))
            marker = " <= HEI" if i == hei else ""
            summary.append(f"{i:3d} {dist:9.3f} {E:13.5f} {dE:11.2f} {maxF:11.5f} {rmsF:10.5f}{marker}\n")

        summary.append("\nStraight line distance between images along the path:\n")
        for i, d in enumerate(distances):
            summary.append(f"        D({i:2d}-{i+1:2d}) = {d:7.4f} Ang.\n")

        summary.append(
            "\n---------------------------------------------------------------\n"
            "           INFORMATION ABOUT HIGHEST ENERGY IMAGE\n"
            "---------------------------------------------------------------\n"
            f"Highest energy image                      ....  {hei}\n"
            f"Energy                                    ....  {Es[hei]: .8f} Eh\n"
            f"Max. abs. force                           ....  {np.max(np.linalg.norm(Fp_list[hei], axis=1)) : .4e} Eh/Angstrom\n"
            "\n-----------------------------------------\n"
            "  HIGHEST ENERGY IMAGE (ANGSTROEM)\n"
            "-----------------------------------------\n"
        )
        summary.append(self.atoms_to_xyz(images[hei]))
        log_info(summary, self.output)

        log_info([
            f"\nWrote MEP to: {mep_path}\n",
            f"Wrote HEI to: {hip_path}\n",
            f"Wrote trajectory to: {traj_file}\n"
        ], self.output)
        
        
        # 3) Optional: CINEB refinement
        if self.params.refine == 'cineb' or self.params.refine == 'nebts':
            if iteration == self.params.max_iter:
                log_info([
                    "\nNEB did not converge. Skipping CINEB refinement.\n",
                    "You may try to increase max_iter or check the initial path.\n"
                ], self.output)
            else:
                self.restart_run(images, energies=Es)
