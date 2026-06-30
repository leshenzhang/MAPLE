# -*- coding: utf-8 -*-
"""
Dimer implementation with:
- Minimum-mode following (rotation: minimize kappa = n^T H n)
- Translation with parallel flip once kappa < 0
- Optional HVP (autograd) callback to replace finite-difference (no Δ tuning)
- Trust-radius / max-step control
- Detailed human-readable logging and XYZ outputs per iteration
"""
from __future__ import annotations

import os
import math
import torch
from dataclasses import dataclass
from typing import Optional, Callable, List

import numpy as np
from ase import Atoms

from .logger import log_info
from ...jobABC import JobABC

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
                f.write(f"Image {i}  Energy = {energies[i]:.10f}\n")
            else:
                f.write(f"Image {i}\n")
            for s, (x, y, z) in zip(symbols, pos):
                f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")

def write_all_images_xyz(filename: str, atoms: Atoms, energy: Optional[float] = None, iteration: int = 0):
    """
    Append current single structure to an xyz trajectory file (for Dimer debug).
    Each block corresponds to one iteration of Dimer optimization.
    """
    if iteration == 0 and os.path.exists(filename):
        os.remove(filename)
    pos = to_numpy_f64(atoms.get_positions())
    symbols = atoms.get_chemical_symbols()
    with open(filename, "a") as f:
        f.write(f"{len(symbols)}\n")
        if energy is not None:
            f.write(f"Iter {iteration}  Energy = {energy:.10f}\n")
        else:
            f.write(f"Iter {iteration}\n")
        for s, (x, y, z) in zip(symbols, pos):
            f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")

# =============================================================================
# ------------------------------ Dimer Params ---------------------------------
# =============================================================================

@dataclass
class DimerParams:
    # Rotation / curvature estimation
    use_hvp: bool = False               # if True, call hvp_fn(atoms, n_flat) for Hn
    delta: float = 0.005                 # Angstrom; used only if not use_hvp
    rot_max_iter: int = 5               # rotation inner iterations per outer step
    rot_alpha: float = 0.5              # rotation step factor on F_rot (unitless); small ~ (0.1~1)
    rot_f_max_th: float = 1.0e-3        # convergence threshold on max|F_rot| (Eh/Ang)
    rot_f_rms_th: float = 5.0e-4        # convergence threshold on RMS(F_rot)

    # Translation / trust region
    step0: float = 0.2                  # initial step scaling on search direction
    step_max: float = 0.15              # absolute max Cartesian displacement (Ang)
    trust_radius: float = 0.15          # same role as max_step; kept both for clarity

    # Convergence (translation forces)
    f_max_th: float = 5.0e-3            # max(|F_trans|) Eh/Ang
    f_rms_th: float = 1.0e-3            # RMS(F_trans) Eh/Ang
    kappa_to_flip: float = 0.0          # if kappa < this value, flip parallel component

    # Iterations
    max_iter: int = 200

    # Metric / projections
    use_mass_weight: bool = False       # if True, use M-metric for dot/proj (mass weighted)
    remove_rigid: bool = True           # remove global translation/rotation from n (molecular)

    # Initialization of n
    n_init: str = "random"              # "random" | "force" | "given"
    n_given: Optional[np.ndarray] = None

    # Outputs
    save_traj: bool = True
    save_metrics: bool = True

# =============================================================================
# ------------------------------ Metric helpers -------------------------------
# =============================================================================

def _get_metric(atoms: Atoms, use_mass_weight: bool):
    if not use_mass_weight:
        return None
    m = to_numpy_f64(atoms.get_masses()).reshape(-1, 1)  # (N,1)
    M = np.repeat(m, 3, axis=1).reshape(-1)              # (3N,)
    return M                                             # diagonal metric entries

def _dot(v, w, M=None):
    """Metric dot: if M is None -> Euclidean; else mass-weighted sum(M * v * w)."""
    if M is None:
        return float(np.dot(v, w))
    return float(np.dot(M * v, w))

def _norm(v, M=None):
    val = _dot(v, v, M=M)
    return math.sqrt(max(val, 0.0))

def _proj_parallel(F, n, M=None):
    """Return parallel component of F along n (unit under metric)."""
    # assume n is normalized in metric (n^T M n = 1) if M given
    c = _dot(F, n, M=M)
    return c * n

def _proj_perp(F, n, M=None):
    return F - _proj_parallel(F, n, M=M)

def _normalize(n, M=None, eps=1e-20):
    dn = _norm(n, M=M)
    if dn < eps:
        raise ValueError("Zero-length direction encountered during normalization.")
    return n / dn

def _remove_rigid_body_components(n, atoms: Atoms, M=None):
    """
    Remove global translation & rotation components from direction n.
    Only meaningful for molecules (non-periodic). For simplicity:
    - Remove translation: subtract mean per axis (mass-weighted average if M given)
    - Remove rotation: project out 3 rotational modes around COM using cross r x axis
      (simple approximate projector; robust enough for search direction).
    """
    # translation
    N = len(atoms)
    X = to_numpy_f64(atoms.get_positions()).reshape(-1, 3)
    v = n.reshape(-1, 3).copy()
    if M is None:
        t = v.mean(axis=0, keepdims=True)
    else:
        mw = to_numpy_f64(atoms.get_masses()).reshape(-1, 1)
        t = (mw * v).sum(axis=0, keepdims=True) / (mw.sum() + 1e-20)
    v -= t

    # rotation (approx): project out components proportional to r x omega, omega = basis unit vectors
    # center positions
    rc = X.mean(axis=0)
    r = X - rc
    # three axes basis
    axes = np.eye(3)
    for k in range(3):
        rot_mode = np.cross(r, axes[k])  # (N,3)
        # project each component of v onto rot_mode
        a = (v * rot_mode).sum() / ((rot_mode * rot_mode).sum() + 1e-20)
        v -= a * rot_mode
    return v.reshape(-1)

# =============================================================================
# ------------------------------ Dimer class ----------------------------------
# =============================================================================

class Dimer(JobABC):
    """
    Dimer saddle search with optional HVP (autograd) backend.

    hvp_fn: Optional[Callable[[Atoms, np.ndarray], np.ndarray]]
        If provided and DimerParams.use_hvp=True, returns H @ n (shape (3N,))
        Else: we use finite-difference via two force calls at R ± Δ n.
    """
    def __init__(self,
                 output: str,
                 atoms_init: Atoms,
                 paras: Optional[dict] = None,
                 hvp_fn: Optional[Callable[[Atoms, np.ndarray], np.ndarray]] = None):
        super().__init__(output)

        self.atoms = atoms_init
        if self.atoms.calc is None:
            raise ValueError("atoms_init must have a working calculator set (atoms.calc).")

        # Initialize params from paras dict
        self.params = self._init_params(DimerParams, paras, ("dimer", "DIMER", "ts"))

        self.hvp_fn = hvp_fn if self.params.use_hvp else None

        # metric
        self.M = _get_metric(self.atoms, self.params.use_mass_weight)

        # init direction n
        self.n = self._init_direction()

        # trust region bookkeeping
        self.alpha = float(self.params.step0)

    # ------------------------------ init n -----------------------------------

    def _init_direction(self) -> np.ndarray:
        N = len(self.atoms)
        D = 3 * N
        p = self.params

        if p.n_init.lower() == "given" and (p.n_given is not None):
            n = vec1d(p.n_given, D)
        elif p.n_init.lower() == "force":
            F = -vec1d(self.atoms.get_forces(), D)  # gradient = -F; here use force itself
            n = F
        else:  # random
            rng = np.random.default_rng()
            n = rng.normal(size=D)

        if p.remove_rigid:
            n = _remove_rigid_body_components(n, self.atoms, M=self.M)
        n = _normalize(n, M=self.M)
        return n

    # ------------------------------- main flow --------------------------------

    def atoms_to_xyz_block(self, atoms: Atoms) -> str:
        lines = []
        syms = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        for idx, (s, (x, y, z)) in enumerate(zip(syms, pos)):
            lines.append(f"{idx:<4d}{s:>2s}{x:18.4f}{y:18.4f}{z:18.4f}")
        return "\n".join(lines) + "\n"

    def _packed_hvp(self, n):
        """Return (Hn, forces, energy) torch tensors for direction ``n``.

        D2 autodiff wiring: when an autograd HVP callback was supplied and
        ``DimerParams.use_hvp=True`` (``self.hvp_fn`` set), use it -- Hn = H@n by
        double-backward, with forces/energy from the SAME autograd pass (no finite
        difference, no delta tuning). The callback may return either
        ``(Hn, forces, energy)`` or just ``Hn`` (forces/energy then come from the
        calculator). Falls back to ``self.atoms.calc.get_hvp`` when no callback.
        """
        if self.hvp_fn is not None:
            out = self.hvp_fn(self.atoms, n)
            if isinstance(out, (tuple, list)) and len(out) == 3:
                Hn, forces, energy = out
            else:
                Hn = out
                _, forces, energy = self.atoms.calc.get_hvp(self.atoms, n)
            ref = Hn if isinstance(Hn, torch.Tensor) else None
            dev = ref.device if ref is not None else torch.device("cpu")
            dty = ref.dtype if ref is not None else torch.float64

            def _t(x, shape):
                if isinstance(x, torch.Tensor):
                    t = x.to(device=dev, dtype=dty)
                else:
                    t = torch.as_tensor(np.asarray(x, dtype=np.float64),
                                        device=dev, dtype=dty)
                return t.reshape(shape) if shape is not None else t.reshape(())

            return _t(Hn, (-1,)), _t(forces, (-1,)), _t(energy, None)
        return self.atoms.calc.get_hvp(self.atoms, n)

    def run(self):
        """
        Main Dimer optimization loop with autograd-based Hessian-vector product (Hn).
        - Rotation uses atoms.calc.get_hvp(atoms, n) to compute Hn (no finite difference)
        - Translation also uses get_hvp to obtain (Hn, forces, energy) in one pass
        - No duplicate energy/force evaluations
        - Logs curvature, rotational force, mode flip, 4 PRFO-style criteria, and trust region
        """
        import torch

        p = self.params
        base, _ = os.path.splitext(self.output)
        traj_file = base + "_dimer_traj.xyz"
        ts_file = base + "_dimer_ts.xyz"
        kcal_per_Eh = 627.509

        # ------------------ init direction & step size ------------------
        n = self.n.copy()        # initial dimer orientation (assumed normalized & rigid-body removed if requested)
        alpha = float(self.alpha)

        # ------------------ initial eval via autograd HVP ------------------
        # get forces & energy once (Hn unused for initial report)
        _, forces_t, energy_t = self._packed_hvp(n)
        forces_np = forces_t.detach().cpu().numpy()
        E0 = float(energy_t.detach().cpu().item())
        maxF0 = float(np.max(np.linalg.norm(forces_np.reshape(-1, 3), axis=1)))
        rmsF0  = float(np.sqrt(np.mean(np.linalg.norm(forces_np.reshape(-1, 3), axis=1) ** 2)))

        log_info([
            "\n---------------------------------------------------------------\n",
            "                       DIMER INITIAL STATE\n",
            "---------------------------------------------------------------\n",
            f"Energy (initial)                        ....  {E0: .8f} Eh\n",
            f"RMS(|F|)                                ....  {rmsF0: .6f} Eh/Angstrom\n",
            f"MAX(|F|)                                ....  {maxF0: .6f} Eh/Angstrom\n",
            "\nINITIAL COORDINATES (ANGSTROEM):\n",
            self.atoms_to_xyz_block(self.atoms),
        ], self.output)

        # ------------------ ensure PRFO-style thresholds exist ------------------
        if not hasattr(self.atoms, "f_max_th"):  self.atoms.f_max_th  = p.f_max_th
        if not hasattr(self.atoms, "f_rms_th"):  self.atoms.f_rms_th  = p.f_rms_th
        if not hasattr(self.atoms, "dp_max_th"): self.atoms.dp_max_th = 1.8e-3
        if not hasattr(self.atoms, "dp_rms_th"): self.atoms.dp_rms_th = 1.2e-3

        log_info([
            "\n----------------------------------------------------------------------\n",
            "                           Dimer Iterations                           \n",
            "----------------------------------------------------------------------\n"
        ], self.output)

        # ------------------ rotation helper (HVP-based) ------------------
        def rotate_minimize_kappa(n_vec: np.ndarray):
            """
            Up to rot_max_iter inner steps to reduce kappa, using autograd Hn.
            Returns: n_new (np.ndarray), max|F_rot| (float), RMS(F_rot) (float)
            """
            n_curr = n_vec.copy()
            max_frot = np.inf
            rms_frot = np.inf

            for _ in range(p.rot_max_iter):
                # Hn from autograd; discard forces/energy here
                Hn_t, _, _ = self._packed_hvp(n_curr)
                dev, dty = Hn_t.device, Hn_t.dtype
                n_th = torch.tensor(n_curr, device=dev, dtype=dty)

                # rotational force: (I - n n^T) Hn
                Hn_par = torch.dot(Hn_t, n_th) * n_th
                F_rot_t = Hn_t - Hn_par

                # metrics
                max_frot = float(torch.max(torch.abs(F_rot_t)).detach().cpu().item())
                rms_frot = float(torch.sqrt(torch.mean(F_rot_t * F_rot_t)).detach().cpu().item())

                # convergence of rotation
                if (max_frot <= p.rot_f_max_th) and (rms_frot <= p.rot_f_rms_th):
                    break

                # gradient descent on kappa in orientation space, then re-normalize (and rigid-body remove if requested)
                n_next = (n_th - p.rot_alpha * F_rot_t).detach().cpu().numpy()
                if p.remove_rigid:
                    n_next = _remove_rigid_body_components(n_next, self.atoms, M=self.M)
                n_next = _normalize(n_next, M=self.M)
                n_curr = n_next

            return n_curr, max_frot, rms_frot

        # ======================= main iteration loop =======================
        for it in range(1, p.max_iter + 1):

            # (1) rotation step using autograd HVP
            n, max_frot, rms_frot = rotate_minimize_kappa(n)

            # (2) translation-side evaluation in one pass
            Hn_t, forces_t, energy_t = self._packed_hvp(n)
            forces_np = forces_t.detach().cpu().numpy()
            E = float(energy_t.detach().cpu().item())

            # curvature kappa = n^T H n
            dev, dty = Hn_t.device, Hn_t.dtype
            n_th = torch.tensor(n, device=dev, dtype=dty)
            kappa = float(torch.dot(n_th, Hn_t).detach().cpu().item())

            # project forces parallel / perpendicular to n (torch tensors)
            Fpar_t  = torch.dot(forces_t, n_th) * n_th
            Fperp_t = forces_t - Fpar_t
            mode_flip = (kappa < p.kappa_to_flip)
            Ftrans_t = (Fperp_t - Fpar_t) if mode_flip else Fperp_t

            # (3) trust-region step (convert only the step to numpy)
            step_vec = (alpha * Ftrans_t).detach().cpu().numpy()
            step_norm_inf = float(np.max(np.abs(step_vec)))
            on_boundary = False
            max_allow = min(p.trust_radius, p.step_max)
            if step_norm_inf > max_allow:
                step_vec *= (max_allow / (step_norm_inf + 1e-20))
                on_boundary = True

            new_positions = self.atoms.get_positions() + step_vec.reshape(-1, 3)
            self.atoms.set_positions(new_positions)
            alpha = (max(0.5 * alpha, 0.1 * p.step0) if on_boundary
                    else min(1.2 * alpha, p.step_max))

            # (4) PRFO-style metrics
            max_dp = float(np.max(np.linalg.norm(step_vec.reshape(-1, 3), axis=1)))
            rms_dp = float(np.sqrt(np.mean(np.linalg.norm(step_vec.reshape(-1, 3), axis=1) ** 2)))
            max_f  = float(np.max(np.linalg.norm(forces_np.reshape(-1, 3), axis=1)))
            rms_f  = float(np.sqrt(np.mean(np.linalg.norm(forces_np.reshape(-1, 3), axis=1) ** 2)))

            self.atoms.max_f  = max_f
            self.atoms.rms_f  = rms_f
            self.atoms.max_dp = max_dp
            self.atoms.rms_dp = rms_dp

            # (5) report one iteration block
            coords_block = self.atoms_to_xyz_block(self.atoms)
            info = [
                "\n----------------------------------------------------------------------\n",
                f"                             Iteration: {it:<3d}                              \n\n",
                "                             Coordinates                               \n",
                "----------------------------------------------------------------------\n",
                coords_block,
                "\n",
                f"Energy:                  {E: .6f} Convergence criteria  Is converged \n",
                f"Maximum Force:         {max_f:>12.6f} {self.atoms.f_max_th:>12.6f}                {'Yes' if max_f <= self.atoms.f_max_th else 'No'}\n",
                f"RMS Force:             {rms_f:>12.6f} {self.atoms.f_rms_th:>12.6f}                {'Yes' if rms_f <= self.atoms.f_rms_th else 'No'}\n",
                f"Maximum Displacement:  {max_dp:>12.6f} {self.atoms.dp_max_th:>12.6f}                {'Yes' if max_dp <= self.atoms.dp_max_th else 'No'}\n",
                f"RMS Displacement:      {rms_dp:>12.6f} {self.atoms.dp_rms_th:>12.6f}                {'Yes' if rms_dp <= self.atoms.dp_rms_th else 'No'}\n",
                f"\nTrust radius (MW): {max_allow: .6f}  Step norm (MW): {step_norm_inf: .6f}  On boundary: {on_boundary}\n"
                f"Curvature (kappa):      {kappa:>12.6f} Eh/Å²\n",
                f"Max Rotational Force:   {max_frot:>12.6f} Eh/Angstrom\n",
                f"RMS Rotational Force:   {rms_frot:>12.6f} Eh/Angstrom\n",
                f"Mode Flip:              {'Yes' if mode_flip else 'No'}\n\n",
            ]
            log_info(info, self.output)

            if p.save_traj:
                # Your local helper supports single Atoms; keeping your current call signature
                write_all_images_xyz(traj_file, self.atoms, energy=E, iteration=it)

            # (6) convergence: all 4 PRFO-style criteria
            if (max_f <= self.atoms.f_max_th and
                rms_f <= self.atoms.f_rms_th and
                max_dp <= self.atoms.dp_max_th and
                rms_dp <= self.atoms.dp_rms_th):
                log_info([f"\nDimer optimization converged at iteration {it}.\n"], self.output)
                break

        # ------------------ final write & brief summary ------------------
        _, _, E_final_t = self._packed_hvp(n)
        E_final = float(E_final_t.detach().cpu().item())
        write_xyz(ts_file, [self.atoms], energies=[E_final])

        log_info([
            "\n---------------------------------------------------------------\n",
            "                 INFORMATION ABOUT SADDLE POINT                \n",
            "---------------------------------------------------------------\n",
            f"Energy (final)                           ....  {E_final: .8f} Eh\n",
            "\n-----------------------------------------\n",
            "  SADDLE GUESS (ANGSTROEM)\n",
            "-----------------------------------------\n",
            self.atoms_to_xyz_block(self.atoms),
            f"\nWrote Dimer trajectory to: {traj_file}\n",
            f"Wrote TS guess to:         {ts_file}\n"
        ], self.output)


# =============================================================================
# ====================== BatchDimer (additive, batched GPU) ===================
# =============================================================================
# B independent dimer min-mode saddle searches advanced in lockstep on the GPU,
# analogous to BatchPRFO (BPRFO.py): one batched force forward serves all B
# systems, per-structure state lives in (B, nmax_dof) padded tensors, a
# per-structure convergence mask freezes finished systems while tensor shapes
# stay static, and per-structure optimizer state is NEVER shared.
#
# The single-structure ``Dimer`` class above is KEPT UNCHANGED as the fallback
# path + parity oracle (validate: BatchDimer must converge to the SAME saddle
# as the serial single-structure loop).
#
# Method / references (cited per 铁则):
#   * Dimer minimum-mode following (finite-difference curvature + rotation):
#       G. Henkelman, H. Jonsson, J. Chem. Phys. 1999, 111, 7010.
#       DOI: 10.1063/1.480097
#   * Improved dimer (single trigonometric "Fourier" rotation to the optimal
#     angle from the rotational force + curvature, instead of many SD rotations):
#       A. Heyden, A. T. Bell, F. J. Keil, J. Chem. Phys. 2005, 123, 224101.
#       DOI: 10.1063/1.2104507
#   * Superlinear dimer (opt-in): gradient extrapolation during rotation
#     (~1 gradient per rotation) + L-BFGS for translation (and a 1-D L-BFGS /
#     secant for rotation), cutting cost to ~2 grad/iter near convergence:
#       J. Kastner, P. Sherwood, J. Chem. Phys. 2008, 128, 014106.
#       DOI: 10.1063/1.2815812
#
# Batched HVP: UMA (and the other MAPLE batch calculators) expose batched
# forces only (``get_ef_gpu``) + a batched numerical Hessian (``get_efh_gpu``),
# NOT a batched autograd HVP. The dimer never needs a full Hessian -- it needs
# only H@N along the current axis, which is obtained by a finite difference of
# the batched force field along N (this IS the dimer construction of HJ-1999):
#       forward difference (default, 1 force eval, reuses the midpoint force F0):
#           H@N ~= -(F(R + dN) - F0) / d            (g = -F)
#       central difference (opt-in ``central_hvp``, 2 force evals, more accurate):
#           H@N ~= -(F(R + dN) - F(R - dN)) / (2 d)
# Every force eval is ONE batched forward over all active systems.

DTYPE_BD = torch.float64


@dataclass
class BatchDimerParams:
    """Parameters for :class:`BatchDimer` (batched dimer saddle search)."""
    # --- curvature / HVP (finite difference of the batched force field) ---
    delta: float = 0.005               # Angstrom dimer half-length for the FD HVP
    central_hvp: bool = False          # False -> 1-eval forward diff (classic HJ);
                                       # True  -> 2-eval central diff (more accurate)
    # --- rotation (Heyden single trigonometric rotation) ---
    rot_max_iter: int = 5              # max rotation iters per outer step
    rot_f_tol: float = 1.0e-3          # stop rotating a system once max|F_rot| < tol
    rot_phi_trial: float = math.pi / 4.0  # Heyden trial angle for the curvature fit
    rot_phi_max: float = math.pi / 12.0   # superlinear: max |rotation| per rot-iter
    rot_kappa_min: float = 0.2            # superlinear: positive floor on rot curvature
    # --- translation / trust region ---
    step0: float = 0.2                 # initial step scaling on the translation force
    step_max: float = 0.15             # absolute max Cartesian displacement (Ang)
    trust_radius: float = 0.15         # trust radius (same role as step_max)
    kappa_to_flip: float = 0.0         # flip the parallel force once curvature < this
    # --- convergence (per structure; same defaults as single-structure Dimer) ---
    f_max_th: float = 5.0e-3
    f_rms_th: float = 1.0e-3
    dp_max_th: float = 1.8e-3
    dp_rms_th: float = 1.2e-3
    max_iter: int = 200
    # --- direction init / projections ---
    remove_rigid: bool = True          # project out global translation/rotation from N
    n_init: str = "random"             # "random" | "given"
    n_given: Optional[np.ndarray] = None  # (B, nmax) padded axes when n_init="given"
    seed: int = 0                      # RNG seed for reproducible random N init
    # --- superlinear (opt-in, Kastner & Sherwood 2008) ---
    superlinear: bool = False          # gradient-extrapolation rotation + L-BFGS translation
    lbfgs_history: int = 8             # L-BFGS memory (number of (s, y) pairs)
    # --- convergence-shrink (mirror BatchPRFO's default) ---
    shrink_on_converge: bool = True    # True  -> slice converged systems OUT of the batch
                                       #          so the next forward + every rotation HVP
                                       #          stop evaluating them (tail dead-compute
                                       #          removed; UMA is block-diagonal so a
                                       #          survivor's math is unchanged beyond fp32
                                       #          reduction-reorder noise).
                                       # False -> legacy: freeze (zero the step) but keep
                                       #          converged systems in the batch forever
                                       #          (= the parity oracle).
    # --- outputs ---
    save_traj: bool = True


class BatchDimer(JobABC):
    """Run B independent dimer saddle searches in lockstep on the GPU.

    Mirrors the BatchPRFO per-system-state / convergence-mask / one-batched-
    forward pattern, but follows the minimum-curvature mode with a dimer
    (finite-difference HVP + rotation) instead of an explicit Hessian RS-PRFO.

    Parameters
    ----------
    output : str
        Log file path; per-structure TS guesses are written next to it.
    device : str
        "cuda" | "cpu".
    paras : dict, optional
        Maps onto :class:`BatchDimerParams` (aliases: "batchdimer", "dimer", "ts").

    Usage
    -----
        m = Molecules(atoms_list); m.calc = batch_calc      # e.g. UMABatchCalc
        BatchDimer(output="bd.out", device="cuda").run(m)

    The calculator must provide the batched contract used by BatchPRFO:
    ``prepare`` / ``get_ef_gpu`` / ``step_cart_`` / ``backup_coords`` /
    ``restore_coords`` and expose ``coord`` / ``_cols`` / ``_n_b`` / ``nmax_dof``.
    """

    def __init__(self, output: str, device: str = "cuda",
                 paras: Optional[dict] = None):
        super().__init__(output)
        self.device = torch.device(device if torch.cuda.is_available()
                                   or device == "cpu" else "cpu")
        self.params = self._init_params(
            BatchDimerParams, paras, ("batchdimer", "dimer", "ts"))

        # topology (set in run())
        self._B = 0
        self._nmax = 0
        self._A = 0                      # padded atom count = nmax // 3
        self._real_mask = None           # (B, nmax) bool  -- real Cartesian DOFs
        self._atom_mask = None           # (B, A)    bool  -- real atoms
        self._nat = None                 # (B, 1, 1) float -- atoms per structure
        self._Leff = None                # (B,) float      -- 3 * n_atoms (real DOFs)

        # per-structure L-BFGS state (superlinear translation); allocated in run()
        self._lb_S = None
        self._lb_Y = None
        self._lb_rho = None
        self._lb_k = 0
        self._rot_kappa = None           # persisted 1-D rotation curvature (superlinear)

    # ------------------------------------------------------------------ helpers
    def _coord_padded(self, calc) -> torch.Tensor:
        """Current geometry scattered into the (B, nmax) padded DOF layout."""
        flat = torch.zeros(self._B * self._nmax, dtype=DTYPE_BD, device=self.device)
        if calc.N_atoms > 0:
            flat[calc._cols] = calc.coord.reshape(-1).to(DTYPE_BD)
        return flat.reshape(self._B, self._nmax)

    def _atomview(self, v: torch.Tensor) -> torch.Tensor:
        """(B, nmax) -> (B, A, 3)."""
        return v.reshape(self._B, self._A, 3)

    def _mdot(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Masked Euclidean dot over real DOFs -> (B,)."""
        return (a * b * self._real_mask).sum(dim=-1)

    def _mnorm(self, a: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self._mdot(a, a).clamp(min=0.0))

    def _mnormalize(self, a: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
        a = a * self._real_mask
        nrm = self._mnorm(a).clamp(min=eps)
        return a / nrm.unsqueeze(-1)

    def _remove_rigid(self, V: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Project global translation (+ rotation) out of each system's direction.

        V, pos: (B, nmax). Per-structure, mass-unweighted (matches the single-
        structure Dimer default ``use_mass_weight=False``)."""
        Va = self._atomview(V).clone()
        Pa = self._atomview(pos)
        m = self._atom_mask.unsqueeze(-1).to(DTYPE_BD)          # (B, A, 1)
        nat = m.sum(dim=1, keepdim=True).clamp(min=1.0)         # (B, 1, 1)
        # translation: subtract per-axis mean over real atoms
        Va = Va - (Va * m).sum(dim=1, keepdim=True) / nat
        Va = Va * m
        # rotation: project out the 3 infinitesimal rotations r x e_k about the COM
        com = (Pa * m).sum(dim=1, keepdim=True) / nat
        r = (Pa - com) * m
        eye = torch.eye(3, dtype=DTYPE_BD, device=self.device)
        for k in range(3):
            rot = torch.cross(r, eye[k].view(1, 1, 3).expand_as(r), dim=-1) * m
            denom = (rot * rot).sum(dim=(1, 2), keepdim=True) + 1e-20
            a = (Va * rot).sum(dim=(1, 2), keepdim=True) / denom
            Va = (Va - a * rot) * m
        return Va.reshape(self._B, self._nmax)

    # ------------------------------------------------------------- batched HVP
    def _hvp(self, calc, N, F0, delta, central):
        """Batched H@N along the unit dimer axis N via finite difference.

        N  : (B, nmax) unit, masked, per structure.
        F0 : (B, nmax) midpoint forces (reused by the forward difference).
        Returns HN (B, nmax), masked. One (central=False) or two (central=True)
        batched forwards. Each system uses its OWN N -> the perturbations are
        block-diagonal isolated (same isolation BatchPRFO relies on)."""
        calc.backup_coords()
        calc.step_cart_((delta * N))
        _, Fp = calc.get_ef_gpu()
        calc.restore_coords()
        Fp = Fp.to(DTYPE_BD)
        if central:
            calc.backup_coords()
            calc.step_cart_((-delta * N))
            _, Fm = calc.get_ef_gpu()
            calc.restore_coords()
            Fm = Fm.to(DTYPE_BD)
            HN = -(Fp - Fm) / (2.0 * delta)
        else:
            HN = -(Fp - F0) / delta
        return HN * self._real_mask

    def _rotate(self, N, Theta, phi):
        """Rotate unit axis N toward unit, N-orthogonal Theta by angle phi (B,)."""
        c = torch.cos(phi).unsqueeze(-1)
        s = torch.sin(phi).unsqueeze(-1)
        return self._mnormalize(N * c + Theta * s)

    # ------------------------------------------------------------- rotation
    def _rotation_step(self, calc, N, F0, active):
        """One outer rotation cycle: align N with the lowest-curvature mode.

        Returns (N, curvature C (B,), max|F_rot| (B,), n_force_evals).
        Default: Heyden single trigonometric rotation (2 grad / rot-iter).
        Superlinear: secant / 1-D L-BFGS rotation (~1 grad / rot-iter after seed).
        """
        p = self.params
        B = self._B
        nfe = 0
        C = torch.zeros(B, dtype=DTYPE_BD, device=self.device)
        max_frot = torch.zeros(B, dtype=DTYPE_BD, device=self.device)
        rot_done = ~active                       # frozen systems never rotate
        dC_prev = None                           # for the superlinear secant
        phi_prev = None
        # superlinear: persisted per-structure 1-D rotation curvature (carried
        # ACROSS outer iterations) so NO Heyden trial HVP is needed -> 1 grad/iter
        kap = self._rot_kappa if p.superlinear else None

        for _it in range(p.rot_max_iter):
            HN = self._hvp(calc, N, F0, p.delta, p.central_hvp); nfe += (2 if p.central_hvp else 1)
            C = self._mdot(N, HN)                                  # curvature N^T H N
            # rotational force F_rot = -(I - N N^T) H N  (steepest descent on C)
            F_rot = -(HN - C.unsqueeze(-1) * N) * self._real_mask
            # max per-atom |F_rot| as the rotation convergence metric
            frot_atom = torch.linalg.norm(self._atomview(F_rot), dim=-1)
            max_frot = (frot_atom * self._atom_mask).amax(dim=-1)
            newly = (max_frot < p.rot_f_tol) & (~rot_done)
            rot_done = rot_done | newly
            run_mask = ~rot_done
            if not bool(run_mask.any()):
                break

            Theta = self._mnormalize(F_rot)                       # unit, N-orthogonal
            dC0 = 2.0 * self._mdot(Theta, HN)                     # dC/dphi at phi=0

            if not p.superlinear:
                # --- Heyden trigonometric optimal angle (1 trial HVP / rot-iter) ---
                phi1 = p.rot_phi_trial
                N1 = self._rotate(N, Theta, torch.full((B,), phi1, dtype=DTYPE_BD,
                                                       device=self.device))
                HN1 = self._hvp(calc, N1, F0, p.delta, p.central_hvp); nfe += (2 if p.central_hvp else 1)
                C1 = self._mdot(N1, HN1)
                b1 = 0.5 * dC0
                c2, s2 = math.cos(2 * phi1), math.sin(2 * phi1)
                # C(phi) = a0 + a1 cos2phi + b1 sin2phi ; solve a1 from the trial
                a1 = (C - C1 + b1 * s2) / (1.0 - c2 + 1e-30)
                psi = torch.atan2(b1, a1)
                phi = 0.5 * psi + math.pi / 2.0                   # angle of the minimum
                # wrap to (-pi/2, pi/2] (minimal equivalent rotation)
                phi = torch.atan2(torch.sin(phi), torch.cos(phi))
                phi = torch.where(phi > math.pi / 2.0, phi - math.pi, phi)
                phi = phi.clamp(-math.pi / 2.0, math.pi / 2.0)
            else:
                # --- superlinear: 1-D L-BFGS / damped-secant rotation, NO trial HVP ---
                # Update the persisted rotation curvature from the slope change over
                # the previous rotation (1-D quasi-Newton); positive floor + a bounded
                # Newton step keep the dimer locked onto the minimum mode (raw secant
                # over-rotated and lost the negative mode on floppy systems).
                if dC_prev is not None:
                    denom = torch.where(phi_prev.abs() < 1e-6,
                                        torch.full_like(phi_prev, 1e-6), phi_prev)
                    sec = (dC0 - dC_prev) / denom
                    upd = phi_prev.abs() > 1e-4
                    kap = torch.where(upd, 0.5 * kap + 0.5 * sec.abs().clamp(min=p.rot_kappa_min), kap)
                phi = -dC0 / (2.0 * kap.clamp(min=p.rot_kappa_min))   # bounded Newton on C(phi)
                phi = phi.clamp(-p.rot_phi_max, p.rot_phi_max)

            phi = torch.where(run_mask, phi, torch.zeros_like(phi))
            dC_prev = dC0
            phi_prev = phi
            N = self._rotate(N, Theta, phi)

        if p.superlinear:
            self._rot_kappa = kap.detach()       # persist across outer iterations

        return N, C, max_frot, nfe

    # ------------------------------------------------------------- L-BFGS (translation)
    def _lbfgs_reset(self):
        self._lb_S = torch.zeros(self._B, self.params.lbfgs_history, self._nmax,
                                 dtype=DTYPE_BD, device=self.device)
        self._lb_Y = torch.zeros_like(self._lb_S)
        self._lb_rho = torch.zeros(self._B, self.params.lbfgs_history,
                                   dtype=DTYPE_BD, device=self.device)
        self._lb_k = 0

    def _lbfgs_push(self, s, y):
        """Append a per-structure (s, y) pair (s.y>0 systems only); ring buffer."""
        m = self.params.lbfgs_history
        sy = (s * y * self._real_mask).sum(dim=-1)
        good = sy > 1e-12
        rho = torch.where(good, 1.0 / sy.clamp(min=1e-12), torch.zeros_like(sy))
        s = torch.where(good.unsqueeze(-1), s, torch.zeros_like(s))
        y = torch.where(good.unsqueeze(-1), y, torch.zeros_like(y))
        if self._lb_k < m:
            i = self._lb_k
            self._lb_S[:, i] = s; self._lb_Y[:, i] = y; self._lb_rho[:, i] = rho
            self._lb_k += 1
        else:
            self._lb_S = torch.roll(self._lb_S, -1, dims=1); self._lb_S[:, -1] = s
            self._lb_Y = torch.roll(self._lb_Y, -1, dims=1); self._lb_Y[:, -1] = y
            self._lb_rho = torch.roll(self._lb_rho, -1, dims=1); self._lb_rho[:, -1] = rho

    def _lbfgs_dir(self, grad):
        """Two-loop recursion -> search direction d ~ -H_inv grad (per structure)."""
        k = self._lb_k
        q = grad.clone()
        alphas = [None] * k
        for i in range(k - 1, -1, -1):
            a = self._lb_rho[:, i] * (self._lb_S[:, i] * q * self._real_mask).sum(-1)
            q = q - a.unsqueeze(-1) * self._lb_Y[:, i]
            alphas[i] = a
        if k > 0:
            s = self._lb_S[:, k - 1]; y = self._lb_Y[:, k - 1]
            sy = (s * y * self._real_mask).sum(-1)
            yy = (y * y * self._real_mask).sum(-1).clamp(min=1e-12)
            gamma = (sy / yy).clamp(min=1e-6, max=1e6)
        else:
            gamma = torch.ones(self._B, dtype=DTYPE_BD, device=self.device)
        r = gamma.unsqueeze(-1) * q
        for i in range(k):
            b = self._lb_rho[:, i] * (self._lb_Y[:, i] * r * self._real_mask).sum(-1)
            r = r + (alphas[i] - b).unsqueeze(-1) * self._lb_S[:, i]
        return (-r) * self._real_mask

    # ------------------------------------------------------- topology / shrink
    def _build_topology(self, calc, atoms_list):
        """(Re)build the padded masks for the CURRENT active set.

        ``_nmax`` / ``_A`` stay FIXED at their startup values (``prepare`` is always
        called with ``fixed_nmax=self._nmax``), so a convergence-shrink only changes
        the batch dimension B -- every (B, nmax) per-structure state tensor is kept
        column-aligned and is updated by simple row-slicing with ``survive_local``.
        """
        device = self.device
        self._B = len(atoms_list)
        n_b = calc._n_b.to(device)                                   # (B,) atoms/struct
        arange_dof = torch.arange(self._nmax, device=device)
        self._real_mask = (arange_dof[None, :] < (3 * n_b)[:, None])
        arange_a = torch.arange(self._A, device=device)
        self._atom_mask = (arange_a[None, :] < n_b[:, None])
        self._Leff = (3 * n_b).clamp(min=1).to(DTYPE_BD)

    def _sync_atoms(self, calc, atoms_list):
        """Commit the calculator's CURRENT geometry back into the (pre-slice) atoms
        objects so survivors keep their optimized coords across the ``prepare()``
        rebuild (mirror ``BatchPRFO._sync_atoms_from_calc``)."""
        pos = calc.coord.detach().cpu().numpy()
        ptr = calc._ptr.detach().cpu().numpy()
        for i, at in enumerate(atoms_list):
            s, t = ptr[i], ptr[i + 1]
            at.positions[:] = pos[s:t]

    def _record(self, atoms_orig, calc, leaving_mask, E_now, C_now,
                result_E, result_C):
        """Write the FINAL geometry / energy / curvature of the systems flagged in
        ``leaving_mask`` (a current-batch bool mask) back by their ORIGINAL index.

        Used both for systems leaving mid-loop on a shrink and for whatever remains
        in the batch at loop exit, so the per-original-index result arrays are filled
        exactly once per structure regardless of when it left the batch."""
        pos = calc.coord.detach().cpu().numpy()
        ptr = calc._ptr.detach().cpu().numpy()
        base, _ = os.path.splitext(self.output)
        for i_local in leaving_mask.nonzero(as_tuple=False).flatten().cpu().tolist():
            oi = int(self._orig_index[i_local].item())
            s, t = ptr[i_local], ptr[i_local + 1]
            atoms_orig[oi].positions[:] = pos[s:t]
            result_E[oi] = float(E_now[i_local])
            result_C[oi] = float(C_now[i_local])
            if self.params.save_traj:
                write_xyz(f"{base}_bd_ts_{oi}.xyz", [atoms_orig[oi]],
                          energies=[result_E[oi]])

    # ------------------------------------------------------------------- run
    def run(self, mols):
        """Drive B dimers to their saddles in lockstep. Reads mols.multiatoms / mols.calc."""
        p = self.params
        device = self.device
        atoms_list = list(mols.multiatoms)
        calc = mols.calc
        B = len(atoms_list)
        if B == 0:
            return
        base, _ = os.path.splitext(self.output)

        # --- topology: one prepare() fixes nmax (kept FIXED for the whole run so a
        #     convergence-shrink only changes the batch dimension B), build masks ---
        calc.prepare(atoms_list)
        _, F_probe = calc.get_ef_gpu()
        self._nmax = int(F_probe.shape[1])
        self._A = self._nmax // 3
        self._build_topology(calc, atoms_list)

        # --- per-ORIGINAL-index bookkeeping (survive-and-shrink writes each system's
        #     final result back by its original index; the map is the identity when
        #     shrink_on_converge=False, so that path reproduces the legacy collection) ---
        atoms_orig = atoms_list                 # original mols.multiatoms (kept order)
        B0 = B
        self._orig_index = torch.arange(B0, dtype=torch.long, device=device)
        result_status = ["max_iter"] * B0
        result_E = [float("nan")] * B0
        result_C = [float("nan")] * B0
        shrink = bool(getattr(p, "shrink_on_converge", True))

        f_max_th = torch.tensor([getattr(a, "f_max_th", p.f_max_th) for a in atoms_list],
                                dtype=DTYPE_BD, device=device)
        f_rms_th = torch.tensor([getattr(a, "f_rms_th", p.f_rms_th) for a in atoms_list],
                                dtype=DTYPE_BD, device=device)
        dp_max_th = torch.tensor([getattr(a, "dp_max_th", p.dp_max_th) for a in atoms_list],
                                 dtype=DTYPE_BD, device=device)
        dp_rms_th = torch.tensor([getattr(a, "dp_rms_th", p.dp_rms_th) for a in atoms_list],
                                 dtype=DTYPE_BD, device=device)

        # --- per-structure dimer axis N (unit, rigid-body-removed) ---
        g = torch.Generator(device="cpu").manual_seed(int(p.seed))
        if p.n_init.lower() == "given" and getattr(p, "n_given", None) is not None:
            N = torch.as_tensor(p.n_given, dtype=DTYPE_BD, device=device).reshape(B, self._nmax)
        else:
            N = torch.randn(B, self._nmax, generator=g).to(device=device, dtype=DTYPE_BD)
        N = N * self._real_mask
        pos0 = self._coord_padded(calc)
        if p.remove_rigid:
            N = self._remove_rigid(N, pos0)
        N = self._mnormalize(N)

        # --- per-structure translation/optimizer state (NOT shared) ---
        alpha = torch.full((B,), float(p.step0), dtype=DTYPE_BD, device=device)
        active = torch.ones(B, dtype=torch.bool, device=device)
        last_step = torch.zeros(B, self._nmax, dtype=DTYPE_BD, device=device)
        g_prev = None                                # -Ftrans buffer for L-BFGS
        self._rot_kappa = torch.full((B,), 1.0, dtype=DTYPE_BD, device=device)
        if p.superlinear:
            self._lbfgs_reset()

        max_allow = min(p.trust_radius, p.step_max)
        total_fe = 0

        log_info([
            "\n========================================================================\n",
            "                 BatchDimer  (batched GPU dimer saddle search)          \n",
            "========================================================================\n",
            f"B systems                : {B}\n",
            f"padded DOF (nmax)        : {self._nmax}\n",
            f"HVP                      : {'central FD (2 eval)' if p.central_hvp else 'forward FD (1 eval, classic HJ)'}\n",
            f"rotation                 : {'superlinear secant/L-BFGS (Kastner-Sherwood 2008)' if p.superlinear else 'Heyden trig (J.Chem.Phys.2005,123,224101)'}\n",
            f"translation              : {'L-BFGS (Kastner-Sherwood 2008)' if p.superlinear else 'trust-radius steepest descent'}\n",
            "Refs: HJ J.Chem.Phys.1999,111,7010 (10.1063/1.480097); "
            "Heyden 2005 (10.1063/1.2104507); Kastner-Sherwood 2008 (10.1063/1.2815812)\n",
            "------------------------------------------------------------------------\n",
        ], self.output)

        for it in range(1, p.max_iter + 1):
            # (0) midpoint energy + forces -- ONE batched forward, all systems
            E0, F0 = calc.get_ef_gpu()
            E0 = E0.to(DTYPE_BD); F0 = (F0.to(DTYPE_BD)) * self._real_mask
            total_fe += 1

            # (1) rotation: align N with the lowest-curvature mode
            N, C, max_frot, nfe = self._rotation_step(calc, N, F0, active)
            total_fe += nfe

            # (2) translation force: F_perp, flip F_par once curvature < 0
            Fpar = (self._mdot(F0, N)).unsqueeze(-1) * N
            Fperp = (F0 - Fpar) * self._real_mask
            flip = (C < p.kappa_to_flip)
            Ftrans = torch.where(flip.unsqueeze(-1), Fperp - Fpar, Fperp) * self._real_mask

            # (3) step (frozen systems contribute nothing)
            if p.superlinear:
                grad = -Ftrans
                if g_prev is not None and self._lb_k >= 0:
                    self._lbfgs_push(last_step, grad - g_prev)
                d = self._lbfgs_dir(grad) if self._lb_k > 0 else (alpha.unsqueeze(-1) * Ftrans)
                step = d
                g_prev = grad
            else:
                step = alpha.unsqueeze(-1) * Ftrans

            step = step * active.unsqueeze(-1) * self._real_mask
            step_atom = torch.linalg.norm(self._atomview(step), dim=-1)         # (B, A)
            max_step = (step_atom * self._atom_mask).amax(dim=-1)               # (B,)
            on_boundary = max_step > max_allow
            scale = torch.where(on_boundary, max_allow / max_step.clamp(min=1e-20),
                                torch.ones_like(max_step))
            step = step * scale.unsqueeze(-1)

            calc.step_cart_(step)

            # per-structure trust update (steepest-descent path only)
            if not p.superlinear:
                alpha = torch.where(on_boundary & active,
                                    torch.clamp(0.5 * alpha, min=0.1 * p.step0),
                                    torch.minimum(1.2 * alpha,
                                                  torch.full_like(alpha, p.step_max)))
            last_step = step

            # (4) convergence (per-atom-norm metric, same as single-structure Dimer)
            f_atom = torch.linalg.norm(self._atomview(F0), dim=-1)
            max_f = (f_atom * self._atom_mask).amax(dim=-1)
            rms_f = torch.sqrt((f_atom ** 2 * self._atom_mask).sum(-1)
                               / self._atom_mask.sum(-1).clamp(min=1))
            dp_atom = step_atom
            max_dp = (dp_atom * self._atom_mask).amax(dim=-1)
            rms_dp = torch.sqrt((dp_atom ** 2 * self._atom_mask).sum(-1)
                                / self._atom_mask.sum(-1).clamp(min=1))
            conv = (max_f <= f_max_th) & (rms_f <= f_rms_th) & \
                   (max_dp <= dp_max_th) & (rms_dp <= dp_rms_th) & active
            for b in conv.nonzero(as_tuple=False).flatten().cpu().tolist():
                result_status[int(self._orig_index[b].item())] = "converged"
            active = active & (~conv)

            if (it <= 5) or (it % 10 == 0) or (not bool(active.any())):
                na = int(active.sum().item())
                log_info([
                    f"[iter {it:4d}] active={na:3d}/{B}  "
                    f"E[min/max]={float(E0.min()):.5f}/{float(E0.max()):.5f}  "
                    f"curv<0={int((C < 0).sum().item())}/{B}  "
                    f"max|F|={float(max_f.max()):.5f}  "
                    f"max|Frot|={float(max_frot.max()):.5f}  "
                    f"rot_fe={nfe} cum_fe={total_fe}\n"
                ], self.output)

            # (5) survive-and-shrink: drop the just-converged systems OUT of the batch
            #     so the next midpoint forward + every rotation HVP stop evaluating them
            #     (mirror BatchPRFO's default convergence-shrink). UMA is a LOCAL
            #     block-diagonal potential, so removing a converged row cannot change a
            #     survivor's math beyond the fp32 reduction-reorder noise floor.
            #     shrink_on_converge=False keeps the legacy no-shrink behaviour intact
            #     (the parity oracle: converged systems are merely frozen via active).
            if shrink and bool(conv.any()):
                # Record the just-converged systems' FINAL results at their frozen,
                # post-step geometry -- IDENTICAL to what the no-shrink oracle reports
                # at loop end (a converged system's geometry + axis N never move again).
                # Cost: ONE extra batched forward + one curvature HVP over the current
                # (pre-slice) batch, amortised across the whole tail it removes.
                E_now, F_now = calc.get_ef_gpu(); total_fe += 1
                E_now = E_now.to(DTYPE_BD); F_now = F_now.to(DTYPE_BD) * self._real_mask
                HN_now = self._hvp(calc, N, F_now, p.delta, p.central_hvp)
                total_fe += (2 if p.central_hvp else 1)
                C_now = self._mdot(N, HN_now)
                self._record(atoms_orig, calc, conv, E_now, C_now, result_E, result_C)

                survive_local = active.nonzero(as_tuple=False).flatten()
                if int(survive_local.numel()) < self._B:
                    # commit survivor geometries, then slice EVERY per-structure tensor
                    # in lockstep (nmax pinned => column layout unchanged, rows only).
                    self._sync_atoms(calc, atoms_list)
                    atoms_list = [atoms_list[i] for i in survive_local.cpu().tolist()]
                    self._orig_index = self._orig_index[survive_local]
                    N = N[survive_local]
                    alpha = alpha[survive_local]
                    active = active[survive_local]
                    last_step = last_step[survive_local]
                    if g_prev is not None:
                        g_prev = g_prev[survive_local]
                    self._rot_kappa = self._rot_kappa[survive_local]
                    if p.superlinear:
                        self._lb_S = self._lb_S[survive_local]
                        self._lb_Y = self._lb_Y[survive_local]
                        self._lb_rho = self._lb_rho[survive_local]
                    f_max_th = f_max_th[survive_local]
                    f_rms_th = f_rms_th[survive_local]
                    dp_max_th = dp_max_th[survive_local]
                    dp_rms_th = dp_rms_th[survive_local]
                    # rebuild the calculator topology + padded masks for the smaller
                    # active set (nmax kept FIXED so the sliced tensors stay aligned).
                    if len(atoms_list) > 0:
                        calc.prepare(atoms_list, fixed_nmax=self._nmax)
                        self._build_topology(calc, atoms_list)
                    else:
                        self._B = 0

            if not bool(active.any()):
                log_info([f"\nAll {B0} dimers converged at iteration {it}.\n"], self.output)
                break

        # --- final: record whatever is STILL in the batch by original index.
        #     shrink=False -> the FULL batch (reproduces the legacy collection: every
        #                     system, converged-and-frozen or max_iter).
        #     shrink=True  -> only the max_iter stragglers (every converged system was
        #                     already recorded at its convergence iteration above).
        if self._B > 0:
            E_final, F_final = calc.get_ef_gpu(); total_fe += 1
            E_final = E_final.to(DTYPE_BD)
            # final curvature sign per structure (one batched HVP at the converged axis)
            HN_final = self._hvp(calc, N, F_final.to(DTYPE_BD) * self._real_mask,
                                 p.delta, p.central_hvp)
            total_fe += (2 if p.central_hvp else 1)
            C_final = self._mdot(N, HN_final)
            remaining = torch.ones(self._B, dtype=torch.bool, device=device)
            self._record(atoms_orig, calc, remaining, E_final, C_final,
                         result_E, result_C)

        n_conv = sum(1 for s in result_status if s == "converged")
        n_negcurv = sum(1 for c in result_C if c == c and c < 0.0)
        log_info([
            "\n------------------------------------------------------------------------\n",
            "                       BatchDimer summary                               \n",
            "------------------------------------------------------------------------\n",
            f"converged                : {n_conv}/{B0}\n",
            f"shrink_on_converge       : {shrink}\n",
            f"negative final curvature : {n_negcurv}/{B0}\n",
            f"total batched force evals: {total_fe}\n",
            f"per-structure status     : {result_status}\n",
            f"per-structure curvature  : {[round(c, 5) if c == c else None for c in result_C]}\n",
        ], self.output)

        self.final_curvature = np.array(result_C, dtype=float)
        self.final_status = result_status
        self.final_energy = np.array(result_E, dtype=float)
        self.total_force_evals = total_fe
        self.n_iter = it
        return
