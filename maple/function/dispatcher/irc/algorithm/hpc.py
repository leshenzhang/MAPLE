# -*- coding: utf-8 -*-
"""
Intrinsic Reaction Coordinate (IRC) integrator using Hessian-based Predictor-Corrector(HPC):

- Mass-weighted coordinates and Hessian
- Predictor step via local quadratic approximation
- DWI surface fitting and mBS (modified Bulirsch–Stoer) corrector integration
- Hessian updated by BFGS/Bofill (optional periodic full recalculation)
- Forward and backward paths from the TS geometry, starting along the
  lowest negative eigenmode of the mass-weighted Hessian
- Path merge with the lower-energy endpoint set as dE=0 and the TS
  marked at the maximum energy point

Units:
- Cartesian coordinates: Å
- Forces: Eh/Å
- Energies: Eh
- Mass-weighted coordinates: sqrt(amu) * Å (via masses_D)
"""

import os
from collections import deque
from dataclasses import dataclass, fields
from typing import List, Optional, Dict, Tuple

import numpy as np
from ase import Atoms

from .logger import log_info, log_error

# =============================== Utilities ===============================
BOHR_TO_ANG = 0.529177210903
KCAL_PER_EH = 627.509474


def to_f64(x):
    """Convert input to float64 numpy array or scalar."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            return x.astype(np.float64, copy=False)
    except Exception:
        pass
    if np.isscalar(x):
        return float(x)
    return np.asarray(x, dtype=np.float64)


def v1(x, n: Optional[int] = None) -> np.ndarray:
    """Flatten to 1D float64 array, optionally enforcing size."""
    v = to_f64(x).reshape(-1)
    if n is not None and v.size != n:
        raise ValueError(f"Expected size {n}, got {v.size}")
    return v


def masses_D(atoms: Atoms) -> np.ndarray:
    """
    Return diagonal scaling vector D (length 3N) for mass-weighted quantities.

    Convention here (consistent with earlier code):

        q_mw = q_cart / D       where D = 1/sqrt(m_i)

    so that q_mw has units sqrt(amu) * Å, and the mass-weighted Hessian is

        H_mw = D * H_cart * D

    Gradients in MW coordinates follow

        g_mw = D * g_cart
    """
    m = to_f64(atoms.get_masses())
    m = np.where(m > 0.0, m, 1.0)
    return v1(1.0 / np.sqrt(np.repeat(m, 3)))


def write_xyz(path: str, atoms_list: List[Atoms], energies: Optional[List[float]] = None):
    """Write a list of structures to an XYZ file."""
    with open(path, "w") as f:
        for i, at in enumerate(atoms_list):
            pos = at.get_positions()
            symbols = at.get_chemical_symbols()
            f.write(f"{len(symbols)}\n")
            if energies is not None and i < len(energies):
                f.write(f"Image {i}  Energy = {energies[i]:.10f}\n")
            else:
                f.write(f"Image {i}\n")
            for s, (x, y, z) in zip(symbols, pos):
                f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    n = _norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _taylor_energy(energy: float, gradient: np.ndarray, hessian: np.ndarray, step: np.ndarray) -> float:
    """Second-order Taylor energy expansion."""
    return float(energy + step @ gradient + 0.5 * step @ hessian @ step)


def _taylor_grad(gradient: np.ndarray, hessian: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Gradient of the second-order Taylor expansion."""
    return gradient + hessian @ step


class DWI:
    """
    Distance-weighted interpolation (DWI) of a local quadratic PES.
    Keeps the most recent two points (coords, energy, gradient, hessian).
    """

    def __init__(self, n: int = 4, maxlen: int = 2):
        self.n = int(n)
        if self.n <= 0 or (self.n % 2) != 0:
            raise ValueError("DWI n must be a positive even integer.")
        self.maxlen = int(maxlen)
        if self.maxlen != 2:
            raise ValueError("DWI currently supports maxlen=2 only.")

        self.coords = deque(maxlen=self.maxlen)
        self.energies = deque(maxlen=self.maxlen)
        self.gradients = deque(maxlen=self.maxlen)
        self.hessians = deque(maxlen=self.maxlen)

    def update(self, coords: np.ndarray, energy: float, gradient: np.ndarray, hessian: np.ndarray) -> None:
        self.coords.append(coords)
        self.energies.append(float(energy))
        self.gradients.append(gradient)
        self.hessians.append(hessian)

        if not (len(self.coords) == len(self.energies) == len(self.gradients) == len(self.hessians)):
            raise RuntimeError("DWI internal buffers are inconsistent.")

    def interpolate(self, at_coords: np.ndarray, gradient: bool = False):
        """Distance-weighted interpolation of energy (and optionally gradient)."""
        if len(self.coords) < 2:
            raise RuntimeError("DWI requires two points before interpolation.")

        c1, c2 = self.coords
        dx1 = at_coords - c1
        dx2 = at_coords - c2

        dx1_norm = _norm(dx1)
        dx2_norm = _norm(dx2)
        dx1_norm_n = dx1_norm ** self.n
        dx2_norm_n = dx2_norm ** self.n

        denom = dx1_norm_n + dx2_norm_n
        if denom <= 0.0:
            raise RuntimeError("DWI interpolation denominator is zero.")

        w1 = dx2_norm_n / denom
        w2 = dx1_norm_n / denom

        e1, e2 = self.energies
        g1, g2 = self.gradients
        h1, h2 = self.hessians

        t1 = _taylor_energy(e1, g1, h1, dx1)
        t2 = _taylor_energy(e2, g2, h2, dx2)
        e_dwi = w1 * t1 + w2 * t2

        if not gradient:
            return e_dwi

        t1_grad = _taylor_grad(g1, h1, dx1)
        t2_grad = _taylor_grad(g2, h2, dx2)

        n_2 = self.n // 2
        dx1_norm_n_grad = 2 * n_2 * (dx1_norm ** (2 * n_2 - 2)) * dx1
        dx2_norm_n_grad = 2 * n_2 * (dx2_norm ** (2 * n_2 - 2)) * dx2
        w1_grad = (dx2_norm_n_grad * dx1_norm_n - dx1_norm_n_grad * dx2_norm_n) / (denom ** 2)
        w2_grad = -w1_grad

        g_dwi = w1_grad * t1 + w1 * t1_grad + w2_grad * t2 + w2 * t2_grad
        return e_dwi, g_dwi


# =============================== Parameters ===============================
@dataclass
class HPCParams:
    # Which negative eigenmode (1 = most negative) to use for initial direction
    target_mode: int = 1

    # HPC step length in Bohr
    step_length_bohr: float = 0.10

    # Number of macro steps per direction
    max_steps: int = 50

    # LQA predictor integration sub-steps
    euler_n: int = 5000

    # Recalculate Hessian every N micro-steps (None = never)
    hessian_recalc: Optional[int] = None

    # Hessian update method: "bfgs" or "bofill"
    hessian_update: str = "bofill"

    # DWI / mBS corrector controls
    dwi_n: int = 4
    mbs_max_k: int = 15
    mbs_points: int = 20
    mbs_tol: float = 1e-5

    # Convergence on forces in Cartesian space (Eh/Å)
    f_max_th: float = 2e-3
    f_rms_th: float = 5e-4

    # Output controls
    print_each: bool = True
    write_traj: bool = True


# ================================== HPC ===================================
class HPC:
    """
    Hessian-based Predictor-Corrector(HPC):

    - Works in mass-weighted coordinates.
    - Each macro step uses LQA predictor propagation followed by mBS
      corrector integration on a DWI-fitted surface.
    - Uses BFGS/Bofill updates of the mass-weighted Hessian, with optional
      periodic full recalculation.
    - Integrates forward and backward from the TS along the lowest
      negative eigenmode of H_mw, then merges paths.
    """

    def __init__(self, atoms: Atoms, output: str,
                 params: Optional[HPCParams] = None,
                 paras: Optional[dict] = None):
        self.atoms = atoms
        self.output = output
        self.p = params if params is not None else HPCParams()

        # Parse optional dict overrides, supporting both {"hpc": {...}}
        # and flat dict style. Keep backward compatibility aliases.
        if isinstance(paras, dict):
            low = {k.lower(): v for k, v in paras.items()}
            sub = None
            for key in ("hpc", "irc"):
                if key in low and isinstance(low[key], dict):
                    sub = low[key]
                    break
            if sub is None:
                sub = low
            sub_low = {k.lower(): v for k, v in sub.items()}

            # Aliases / compatibility mapping
            aliases = {
                "sd_len_bohr": "step_length_bohr",
                "steplength_bohr": "step_length_bohr",
                "max_points": "max_steps",
                "euler_n": "euler_n",
                "hessian_update": "hessian_update",
                "dwi_n": "dwi_n",
                "mbs_max_k": "mbs_max_k",
                "mbs_points": "mbs_points",
                "mbs_tol": "mbs_tol",
                "hessian_recalc": "hessian_recalc",
                "target_mode": "target_mode",
                "f_max_th": "f_max_th",
                "f_rms_th": "f_rms_th",
                "tol_maxf": "f_max_th",       # backward compatibility
                "tol_rmsf": "f_rms_th",       # backward compatibility
                "print_each": "print_each",
                "write_traj": "write_traj",
            }

            for k, v in sub_low.items():
                if k in aliases:
                    setattr(self.p, aliases[k], v)
                elif hasattr(self.p, k):
                    setattr(self.p, k, v)

        # Internal state for HPC integration
        self._D: Optional[np.ndarray] = None  # mass-weight scaling vector
        self._step_len_mw: float = float(self.p.step_length_bohr * BOHR_TO_ANG)
        self._step_len_umw: float = float(self.p.step_length_bohr * BOHR_TO_ANG)
        self.mw_coords: Optional[np.ndarray] = None
        self.mw_hessian: Optional[np.ndarray] = None
        self.prev_coords: Optional[np.ndarray] = None
        self.prev_grad: Optional[np.ndarray] = None
        self.micro_counter: int = 0
        self._dwi: Optional[DWI] = None

    # ------------------------------ Public API ------------------------------
    def run(self) -> Dict[str, any]:
        """
        Compute forward and backward HPC IRC paths and produce a merged summary.

        Returns:
            {
                "forward": {...},
                "backward": {...},
                "summary": {...}
            }
        """
        # Prepare mass weights once at TS geometry
        self._D = masses_D(self.atoms)
        self._step_len_mw = float(self.p.step_length_bohr * BOHR_TO_ANG)
        self._step_len_umw = float(self.p.step_length_bohr * BOHR_TO_ANG)

        # Diagonalize mass-weighted Hessian at TS to get negative mode
        H_cart_ts = self._get_hessian_cart()
        H_mw_ts = (self._D[:, None] * H_cart_ts) * self._D[None, :]
        w, V = np.linalg.eigh(H_mw_ts)

        neg_idx = np.where(w < 0.0)[0]
        if len(neg_idx) == 0:
            log_error(
                ["[ERROR] HPC-IRC: No negative eigenvalues found — "
                 "starting geometry is not a saddle point.\n"],
                self.output,
            )
            raise RuntimeError("HPC-IRC: no negative eigenvalues at TS.")

        if len(neg_idx) < self.p.target_mode:
            log_error(
                [f"[ERROR] HPC-IRC: Requested mode {self.p.target_mode}, "
                 f"but only {len(neg_idx)} negative modes found.\n"],
                self.output,
            )
            raise RuntimeError("HPC-IRC: requested negative mode does not exist.")

        sorted_neg = neg_idx[np.argsort(w[neg_idx])]  # most negative first
        idx = sorted_neg[self.p.target_mode - 1]
        eigval = w[idx]
        v_neg_mw = V[:, idx]

        log_info(
            [
                "\n[INFO] HPC-IRC: Selected negative eigenmode "
                f"#{self.p.target_mode} with λ = {eigval:.6e} (MW basis)\n"
            ],
            self.output,
        )

        # Reference TS energy
        E_ts = float(self.atoms.get_potential_energy(force_consistent=True))

        # Store original TS Cartesian positions, reused for both directions
        R_ts_cart = self.atoms.get_positions().copy().reshape(-1)

        # Forward and backward HPC-IRC
        forward_log = self._one_side(
            forward=True,
            sign=+1.0,
            q_ts_cart=R_ts_cart,
            v_neg_mw=v_neg_mw,
            E_ts=E_ts,
        )
        backward_log = self._one_side(
            forward=False,
            sign=-1.0,
            q_ts_cart=R_ts_cart,
            v_neg_mw=v_neg_mw,
            E_ts=E_ts,
        )

        merged = self._merge_and_mark_ts(forward_log, backward_log)

        if self.p.write_traj:
            self._write_trajs(forward_log, backward_log)

        return {"forward": forward_log, "backward": backward_log, "summary": merged}

    # --------------------------- Low-level helpers --------------------------
    def _cart_from_mw(self, q_mw: np.ndarray) -> np.ndarray:
        """Convert mass-weighted coordinates back to Cartesian (Å)."""
        return (q_mw * self._D).reshape(-1)

    def _mw_from_cart(self, q_cart: np.ndarray) -> np.ndarray:
        """Convert Cartesian coordinates (Å) to mass-weighted coordinates."""
        return (q_cart / self._D).reshape(-1)

    def _unweight_len(self, dq_mw: np.ndarray) -> float:
        """Return unweighted length of a MW displacement."""
        return _norm(dq_mw * self._D)

    def _scale_mw_step(self, direction_mw: np.ndarray, step_umw: float) -> np.ndarray:
        """Scale a MW direction to a target unweighted step length."""
        denom = self._unweight_len(direction_mw)
        if denom <= 1e-16:
            return np.zeros_like(direction_mw)
        return direction_mw * (step_umw / denom)

    def _unweight_grad(self, g_mw: np.ndarray) -> np.ndarray:
        """Convert MW gradient to Cartesian gradient."""
        return g_mw / self._D

    def _get_conv_fact(self, g_mw: np.ndarray, min_fact: float = 2.0) -> float:
        """Estimate MW-to-unweighted conversion factor for step length."""
        norm_mw = _norm(g_mw)
        if norm_mw <= 1e-16:
            return min_fact
        norm_cart = _norm(self._unweight_grad(g_mw))
        return max(min_fact, norm_cart / norm_mw)

    def _get_hessian_cart(self) -> np.ndarray:
        """
        Get Cartesian Hessian (3N x 3N) from the calculator.

        Assumes atoms.calc implements get_hessian(atoms) and returns 3N x 3N
        or shape (1, 3N, 3N).
        """
        H = self.atoms.calc.get_hessian(self.atoms)
        H = to_f64(H)
        if H.ndim == 3 and H.shape[0] == 1:
            H = H[0]
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError(f"Hessian must be square 2D, got shape {H.shape}")
        return H

    def _energy_forces_from_mw(self, q_mw: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Set positions from MW coordinates, then return (E, F_cart).

        E: Eh
        F_cart: (3N,) Eh/Å
        """
        q_cart = self._cart_from_mw(q_mw)
        self.atoms.set_positions(q_cart.reshape(-1, 3))
        E = float(self.atoms.get_potential_energy(force_consistent=True))
        F_cart = to_f64(self.atoms.get_forces()).reshape(-1)
        return E, F_cart

    def _gradient_mw_from_forces(self, F_cart: np.ndarray) -> np.ndarray:
        """
        Convert Cartesian forces (Eh/Å) to MW gradient:

            g_cart = dE/dR = -F_cart
            g_mw = D * g_cart
        """
        g_cart = -F_cart.reshape(-1)
        return self._D * g_cart

    @staticmethod
    def _perp_component(vec: np.ndarray, perp_to: np.ndarray) -> np.ndarray:
        """Return component of vec perpendicular to perp_to."""
        denom = float(np.dot(perp_to, perp_to))
        if denom <= 0.0:
            return np.zeros_like(vec)
        return vec - np.dot(perp_to, vec) * perp_to / denom

    @staticmethod
    def _bfgs_update(H: np.ndarray,
                     s: np.ndarray,
                     y: np.ndarray) -> np.ndarray:
        """
        Symmetric BFGS update:

            H_{k+1} = H_k + (y y^T)/(y·s) - (H s s^T H)/(s^T H s)

        Safeguards if denominators are too small or curvature condition fails.
        """
        ys = float(np.dot(y, s))
        if ys <= 1e-12:
            return H

        Hy = H.dot(s)
        sTHs = float(np.dot(s, Hy))
        if sTHs <= 1e-12:
            return H

        term1 = np.outer(y, y) / ys
        term2 = np.outer(Hy, Hy) / sTHs
        return H + term1 - term2

    @staticmethod
    def _bofill_update(H: np.ndarray,
                       s: np.ndarray,
                       y: np.ndarray) -> np.ndarray:
        """
        Bofill update:
        Combines Murtagh-Sargent and Powell-symmetric-Broyden updates using the Bofill mixing factor.
        """
        dx, dg = s, y

        z = dg - H.dot(dx)
        # MS
        ms = np.outer(z, z) / z.dot(dx)
        # PSB
        dx2 = dx.dot(dx)
        psb = ((np.outer(dx, z) + np.outer(z, dx)) / dx2) - (z.dot(dx)) * np.outer(dx, dx) / (dx2 * dx2)
        # Bofill mixing
        mix = (z.dot(dx) ** 2) / (z.dot(z) * dx2)

        dH = mix * ms + (1.0 - mix) * psb
        return H + dH

    @staticmethod
    def _newton_1d(on_sphere,
                   lambda_0: float,
                   maxiter: int = 50,
                   tol: float = 1e-10) -> float:
        """
        Simple 1D Newton root finder with finite-difference derivative.
        Used instead of scipy.optimize.newton to avoid extra dependency.
        """
        lam = float(lambda_0)
        for _ in range(maxiter):
            f = float(on_sphere(lam))
            if abs(f) < tol:
                break
            h = 1e-4 * max(1.0, abs(lam))
            f1 = float(on_sphere(lam + h))
            df = (f1 - f) / h
            if abs(df) < 1e-16:
                lam *= 0.5
                continue
            lam -= f / df
        return lam

    # --------------------------- Micro-step (HPC) ----------------------------
    def _micro_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one HPC step using LQA predictor + mBS corrector.

        Updates:
            self.mw_coords, self.mw_hessian, self.prev_coords,
            self.prev_grad, self.micro_counter

        Returns:
            dx: MW displacement for this step
            g_pred: predicted MW gradient (for diagnostics)
        """
        init_mw = self.mw_coords.copy()

        # Current energy / gradient
        e_curr, F_cart = self._energy_forces_from_mw(init_mw)
        g_curr = self._gradient_mw_from_forces(F_cart)
        if _norm(g_curr) < 1e-12:
            return np.zeros_like(g_curr), g_curr

        # Update Hessian (or optionally recalc)
        recalc = (
            self.p.hessian_recalc is not None
            and self.p.hessian_recalc > 0
            and (self.micro_counter % self.p.hessian_recalc == 0)
        )
        if recalc and self.micro_counter > 0:
            H_cart = self._get_hessian_cart()
            self.mw_hessian = (self._D[:, None] * H_cart) * self._D[None, :]
        elif (self.prev_coords is not None) and (self.prev_grad is not None):
            gradient_diff = g_curr - self.prev_grad
            coords_diff = init_mw - self.prev_coords
            if str(self.p.hessian_update).lower() == "bofill":
                self.mw_hessian = self._bofill_update(self.mw_hessian, coords_diff, gradient_diff)
            else:
                self.mw_hessian = self._bfgs_update(self.mw_hessian, coords_diff, gradient_diff)

        # Store for next update
        self.prev_coords = init_mw
        self.prev_grad = g_curr

        # Update DWI with current point (skip the very first step)
        if self._dwi is not None and self.micro_counter > 0:
            self._dwi.update(init_mw.copy(), e_curr, g_curr, self.mw_hessian.copy())

        # Predictor: LQA propagation using current Hessian
        eigvals, eigvecs = np.linalg.eigh(self.mw_hessian)
        mask = np.abs(eigvals) > 1e-8
        if not np.any(mask):
            return np.zeros_like(g_curr), g_curr

        eigvals = eigvals[mask]
        eigvecs = eigvecs[:, mask]

        g_star = eigvecs.T.dot(g_curr)
        g_norm = _norm(g_curr)
        if g_norm < 1e-12:
            return np.zeros_like(g_curr), g_curr

        dt = self._step_len_mw / (float(self.p.euler_n) * g_norm)
        t = dt
        cur_length = 0.0
        for _ in range(int(self.p.euler_n)):
            dsdt = np.sqrt(np.sum((g_star ** 2) * np.exp(-2.0 * eigvals * t)))
            cur_length += dsdt * dt
            if cur_length >= self._step_len_mw:
                break
            t += dt

        alphas = (np.exp(-eigvals * t) - 1.0) / eigvals
        dx_pred = eigvecs.dot(alphas * g_star)
        pred_mw = init_mw + dx_pred

        # Evaluate predicted point and update Hessian
        E_pred, F_pred = self._energy_forces_from_mw(pred_mw)
        g_pred = self._gradient_mw_from_forces(F_pred)

        dg_pred = g_pred - g_curr
        if str(self.p.hessian_update).lower() == "bofill":
            self.mw_hessian = self._bofill_update(self.mw_hessian, dx_pred, dg_pred)
        else:
            self.mw_hessian = self._bfgs_update(self.mw_hessian, dx_pred, dg_pred)

        # Update DWI with predicted point
        if self._dwi is not None:
            self._dwi.update(pred_mw.copy(), E_pred, g_pred, self.mw_hessian.copy())

        # Corrector: mBS integration on DWI surface
        if self._dwi is not None:
            corr_mw = self._corrector_step(init_mw, self._step_len_umw, self._dwi)
        else:
            corr_mw = pred_mw

        self.mw_coords = corr_mw
        self.micro_counter += 1

        dx = self.mw_coords - init_mw
        return dx, g_pred

    def _corrector_step(self, init_mw: np.ndarray, step_length: float, dwi: DWI) -> np.ndarray:
        """mBS (modified Bulirsch–Stoer) corrector integration on DWI PES."""
        errors = []
        richardson: Dict[Tuple[int, int], np.ndarray] = {}

        for k in range(int(self.p.mbs_max_k)):
            points = int(self.p.mbs_points) * (2 ** k)
            corr_step = step_length / (points - 1)
            cur_coords = init_mw.copy()
            k_coords: List[np.ndarray] = []
            cur_length = 0.0

            while True:
                k_coords.append(cur_coords.copy())
                if abs(step_length - cur_length) < 0.5 * corr_step:
                    break

                _, gradient = dwi.interpolate(cur_coords, gradient=True)
                grad_norm = _norm(gradient)
                if grad_norm < 1e-12:
                    break
                cur_coords = cur_coords + corr_step * (-gradient / grad_norm)
                cur_length = self._unweight_len(cur_coords - init_mw)

                # Oscillation check
                if len(k_coords) > 1:
                    prev_coords = k_coords[-2]
                    if _norm(cur_coords - prev_coords) <= corr_step:
                        return prev_coords

            richardson[(k, 0)] = cur_coords

            for j in range(1, k + 1):
                richardson[(k, j)] = (
                    (2 ** j) * richardson[(k, j - 1)] - richardson[(k - 1, j - 1)]
                ) / (2 ** j - 1)

            if k > 0:
                error = float(np.sqrt(np.mean((richardson[(k, k)] - richardson[(k, k - 1)]) ** 2)))
                errors.append(error)
                if error <= float(self.p.mbs_tol):
                    break
        else:
            raise RuntimeError("mBS Richardson extrapolation did not converge.")

        return richardson[(k, k)]

    # --------------------------- One direction ------------------------------
    def _one_side(self,
                  forward: bool,
                  sign: float,
                  q_ts_cart: np.ndarray,
                  v_neg_mw: np.ndarray,
                  E_ts: float) -> Dict[str, any]:
        """
        Single-sided HPC IRC integration.

        Args:
            forward: True for forward path, False for backward.
            sign: +1.0 or -1.0 to choose direction along the negative mode.
            q_ts_cart: TS Cartesian coordinates (flattened, Å).
            v_neg_mw: selected negative eigenmode in MW basis (3N, normalized later).
            E_ts: TS reference energy (Eh).
        """
        p = self.p
        title = "FORWARD HPC-IRC" if forward else "BACKWARD HPC-IRC"

        self._print_header(title)
        self._print_conv_thresholds(p.f_max_th, p.f_rms_th)

        # Initial MW coordinates at TS
        q_ts_mw = self._mw_from_cart(q_ts_cart)

        # Initial displacement along negative mode in MW
        v_dir = _unit(v_neg_mw) * sign
        q0_mw = q_ts_mw + self._scale_mw_step(v_dir, 0.5 * self._step_len_mw)

        # Gradient at TS (for initial Hessian update)
        _, F_ts_cart = self._energy_forces_from_mw(q_ts_mw)
        g_ts_mw = self._gradient_mw_from_forces(F_ts_cart)

        # Initial Hessian at TS (MW)
        H_ts_cart = self._get_hessian_cart()
        H0_mw = (self._D[:, None] * H_ts_cart) * self._D[None, :]

        # Initial energy / forces / gradient at displaced point
        E0, F0_cart = self._energy_forces_from_mw(q0_mw)
        g0_mw = self._gradient_mw_from_forces(F0_cart)

        maxF0 = float(np.max(np.abs(F0_cart)))
        rmsF0 = float(np.sqrt(np.mean(F0_cart ** 2)))

        # Hessian update using TS -> displaced point
        if str(self.p.hessian_update).lower() == "bofill":
            H0_mw = self._bofill_update(H0_mw, q0_mw - q_ts_mw, g0_mw - g_ts_mw)
        else:
            H0_mw = self._bfgs_update(H0_mw, q0_mw - q_ts_mw, g0_mw - g_ts_mw)

        # Initialize HPC state
        self.mw_coords = q0_mw.copy()
        self.mw_hessian = H0_mw.copy()
        self.prev_coords = None
        self.prev_grad = None
        self.micro_counter = 0
        self._dwi = DWI(n=self.p.dwi_n, maxlen=2)
        self._dwi.update(q0_mw.copy(), E0, g0_mw, self.mw_hessian.copy())

        # Iteration 0 logging
        if p.print_each:
            self._print_iter_line(0, E0, (E0 - E_ts) * KCAL_PER_EH, maxF0, rmsF0)

        records: List[Dict[str, any]] = []
        records.append(
            {
                "E": E0,
                "maxG": maxF0,
                "rmsG": rmsF0,
                "x": self.atoms.get_positions().copy(),
            }
        )

        # Macro steps
        for it in range(1, p.max_steps + 1):
            dx, _ = self._micro_step()
            if _norm(dx) <= 1e-12:
                log_info(
                    [f"[INFO] {title}: step too small at step {it}, stopping.\n"],
                    self.output,
                )
                break

            # Energy / forces at new point
            E_new, F_new = self._energy_forces_from_mw(self.mw_coords)
            maxF = float(np.max(np.abs(F_new)))
            rmsF = float(np.sqrt(np.mean(F_new ** 2)))

            if p.print_each:
                self._print_iter_line(it, E_new, (E_new - E_ts) * KCAL_PER_EH, maxF, rmsF)

            records.append(
                {
                    "E": E_new,
                    "maxG": maxF,
                    "rmsG": rmsF,
                    "x": self.atoms.get_positions().copy(),
                }
            )

            # Convergence in terms of Cartesian forces
            if (maxF <= p.f_max_th) and (rmsF <= p.f_rms_th):
                self._print_hurray()
                break

        return {"title": title, "records": records, "E_ts": E_ts}

    # --------------------------- Merge & summary ----------------------------
    def _merge_and_mark_ts(self, f: Dict, b: Dict) -> Dict:
        fR, bR = f["records"], b["records"]
        if not fR or not bR:
            log_error(["[ERROR] HPC-IRC: One path side is empty.\n"], self.output)
            raise RuntimeError("HPC-IRC: one path side is empty.")

        Ef_end, Eb_end = fR[-1]["E"], bR[-1]["E"]
        if Ef_end <= Eb_end:
            first, second = fR, bR
        else:
            first, second = bR, fR

        merged = []
        # Reverse first (to go from minimum to TS), then append second
        for k in range(len(first) - 1, -1, -1):
            merged.append(first[k])
        for k in range(0, len(second)):
            merged.append(second[k])

        # Find TS index as maximum energy point
        E_list = [rec["E"] for rec in merged]
        ts_idx = int(np.argmax(E_list))
        E_ref = float(min(Ef_end, Eb_end))

        log_info(
            [
                "\n---------------------------------------------------------------\n",
                "                       HPC-IRC PATH SUMMARY           \n",
                "---------------------------------------------------------------\n",
                "All forces are in Eh/Å.\n\n",
                "Step        E(Eh)      dE(kcal/mol)  max(|G|)   RMS(G) \n",
            ],
            self.output,
        )

        rows = []
        for i, rec in enumerate(merged, start=1):
            dE_kcal = (rec["E"] - E_ref) * KCAL_PER_EH
            line = (
                f"{i:4d}  {rec['E']:14.6f}  {dE_kcal:12.6f}    "
                f"{rec['maxG']:8.6f}  {rec['rmsG']:8.6f}"
            )
            if i - 1 == ts_idx:
                line += " <= TS"
            rows.append(line + "\n")
        log_info(rows, self.output)

        return {
            "E_ref": E_ref,
            "ts_index": ts_idx + 1,  # 1-based
            "merged_rows": rows,
        }

    # --------------------------- Trajectories -----------------------------
    def _write_trajs(self, f: Dict, b: Dict):
        """Write full, forward, and backward trajectories as XYZ files."""
        base, _ = os.path.splitext(self.output)
        full_path = base + "_full.xyz"
        fwd_path = base + "_forward.xyz"
        bwd_path = base + "_backward.xyz"

        # Forward
        f_atoms, f_E = [], []
        for rec in f["records"]:
            a = self.atoms.copy()
            a.set_positions(rec["x"])
            f_atoms.append(a)
            f_E.append(rec["E"])
        write_xyz(fwd_path, f_atoms, f_E)

        # Backward
        b_atoms, b_E = [], []
        for rec in b["records"]:
            a = self.atoms.copy()
            a.set_positions(rec["x"])
            b_atoms.append(a)
            b_E.append(rec["E"])
        write_xyz(bwd_path, b_atoms, b_E)

        # Full (concatenate forward then backward)
        full_atoms = f_atoms + b_atoms
        full_E = f_E + b_E
        write_xyz(full_path, full_atoms, full_E)

        log_info(
            [
                f"\n[INFO] HPC-IRC forward trajectory written to: {fwd_path}\n",
                f"[INFO] HPC-IRC backward trajectory written to: {bwd_path}\n",
                f"[INFO] HPC-IRC full trajectory written to: {full_path}\n",
            ],
            self.output,
        )

    # ----------------------------- Printing ------------------------------
    def _print_header(self, title: str):
        log_info(
            [
                f"\n         {'*' * 61}\n",
                f"         *{title.center(59)}*\n",
                f"         {'*' * 61}\n\n",
            ],
            self.output,
        )

    def _print_conv_thresholds(self, f_max_th: float, f_rms_th: float):
        log_info(
            [
                "Iteration    E(Eh)      dE(kcal/mol)  max(|G|)   RMS(G) \n",
                f"Convergence thresholds                {f_max_th:0.6f}  {f_rms_th:0.6f}\n",
            ],
            self.output,
        )

    def _print_iter_line(self,
                         i: int,
                         E: float,
                         dE_kcal: float,
                         maxF: float,
                         rmsF: float):
        log_info(
            [f"{i:5d}  {E:14.6f}  {dE_kcal:12.6f}    {maxF:8.6f}  {rmsF:8.6f}\n"],
            self.output,
        )

    def _print_hurray(self):
        log_info(
            [
                "\n                      *************************************************\n",
                "                      ***          THE HPC-IRC HAS CONVERGED        ***\n",
                "                      *************************************************\n\n",
            ],
            self.output,
        )


# ======================================================================== #
#                            BATCHED HPC-IRC                                #
# ======================================================================== #
# Batched Hessian-based Predictor-Corrector IRC over B transition-state
# structures sharing ONE GPU *batch* calculator (prepare / get_ef_gpu /
# get_efh_gpu / set_coords_). All B paths are advanced in lockstep; the
# per-step force (and, at the TS anchor / optional recalc, Hessian) evaluations
# for every structure are fused into ONE batched GPU forward. Per the project
# profiling the MLIP forward is 95-99% of wall, so batching those calls is the
# win -- the per-path CONTROL FLOW (LQA predictor, DWI/mBS corrector, Bofill/
# BFGS Hessian updates) stays per-item numpy, IDENTICAL to the single-structure
# ``HPC`` (parity oracle). Mirrors ``LQABatch`` / ``GSBatch``.
#
# Design (why per-item numpy + batched forward, NOT full torch vectorization):
# HPC's DWI-fitted mBS (modified Bulirsch-Stoer) corrector runs a per-structure
# variable-length Richardson-extrapolation loop and two Bofill updates per micro
# step -- these do not vectorize cleanly across B without diverging from the
# byte-exact single-structure math. Instead each structure runs the EXACT single
# ``HPC`` numpy (reusing ``HPC``'s own helpers/static methods), expressed as a
# generator that ``yield``s an evaluation request at every point the serial code
# would touch the calculator. A driver ``_pump`` collects the B still-active
# requests at each phase (they stay phase-aligned by construction: identical,
# data-independent call sequence per structure) and fulfils them with ONE
# batched ``get_ef_gpu`` / ``get_efh_gpu``. This makes the batch trajectory of
# structure i bit-equivalent (to fp32 forward noise) to running ``HPC`` on i
# alone, because the block-diagonal batch calc returns each structure's forces
# independent of its batch mates (the perturb-one byte-isolation the sibling
# batched optimizers rely on).
#
# Units are identical to the single-structure HPC: positions Angstrom, forces
# Eh/Angstrom, Hessian Eh/Angstrom^2, energy Eh, MW coords sqrt(amu)*Angstrom
# with q_mw = q_cart / D, H_mw = D (x) D * H, g_mw = D * g_cart  (D = 1/sqrt(m)).
import torch as _torch


def _is_batch_calc_hpc(calc) -> bool:
    """A batch calculator exposes prepare() + get_ef_gpu() (shared predicate)."""
    from ..._batch_calc_utils import is_batch_calc
    return is_batch_calc(calc)


def _is_coupled_batch_calc(calc) -> bool:
    """True for globally-coupled / polarizable batch calcs (MACE-POL, AIMNet2-NSE)
    where a block-diagonal multi-structure batch is PHYSICALLY WRONG (molecules
    share charge / polarization globally). Duck-typed, no imports:

      * ``coupling_mode`` present and != 'sequential' -> MACEPolBatchCalc (its
        'raise'/'approx' modes globally couple; only 'sequential' is per-molecule
        correct).
      * class name or model tag carries 'pol' / 'nse'.
    """
    cm = getattr(calc, "coupling_mode", None)
    if cm is not None and str(cm).lower() != "sequential":
        return True
    name = type(calc).__name__.lower()
    if "pol" in name or "nse" in name:
        return True
    mn = str(getattr(calc, "_model_name", None)
             or getattr(calc, "model_name", None) or "").lower()
    if "pol" in mn or "nse" in mn:
        return True
    return False


def _apply_paras_hpc(p: "HPCParams", paras):
    """Apply a {'hpc'|'irc': {...}} or flat dict of overrides onto an HPCParams.

    Same alias table as HPC.__init__ (kept separate so the batch path reuses it
    without touching the single-structure constructor)."""
    if not isinstance(paras, dict):
        return p
    low = {k.lower(): v for k, v in paras.items()}
    sub = None
    for key in ("hpc", "irc"):
        if key in low and isinstance(low[key], dict):
            sub = low[key]
            break
    if sub is None:
        sub = low
    sub_low = {k.lower(): v for k, v in sub.items()}
    aliases = {
        "sd_len_bohr": "step_length_bohr",
        "steplength_bohr": "step_length_bohr",
        "max_points": "max_steps",
        "euler_n": "euler_n",
        "hessian_update": "hessian_update",
        "dwi_n": "dwi_n",
        "mbs_max_k": "mbs_max_k",
        "mbs_points": "mbs_points",
        "mbs_tol": "mbs_tol",
        "hessian_recalc": "hessian_recalc",
        "target_mode": "target_mode",
        "f_max_th": "f_max_th",
        "f_rms_th": "f_rms_th",
        "tol_maxf": "f_max_th",
        "tol_rmsf": "f_rms_th",
        "print_each": "print_each",
        "write_traj": "write_traj",
    }
    for k, v in sub_low.items():
        if k in aliases:
            setattr(p, aliases[k], v)
        elif hasattr(p, k):
            setattr(p, k, v)
    return p


class HPCBatch:
    """Batched HPC-IRC integrator over B transition-state structures.

    Consumes a GPU *batch* calculator (NOT a per-atoms ASE calculator):
        calc.prepare(atoms_list, fixed_nmax)
        calc.set_coords_(coord (N,3))                 # packed Cartesian, calc order
        calc.get_ef_gpu()  -> (E (B,), F (B,M))       [Hartree, Eh/A]
        calc.get_efh_gpu() -> (E (B,), F (B,M), H (B,M,M), P (B,))  [Hartree]

    Every macro step's per-structure force evaluations (and the TS-anchor / opt-in
    periodic-recalc Hessians) are fused into ONE batched forward via ``_pump``.
    The per-path algorithm is the EXACT single-structure ``HPC`` numpy (LQA
    predictor + DWI/mBS corrector + Bofill/BFGS-updated MW Hessian), so a batch
    path is numerically identical (to fp32 forward noise) to the serial ``HPC``
    oracle on the same structure.

    Backend gate (REQUIRED, ai-maple-gpu 各势函数适配):
      * Batchable (UMA-local / standard MACE / MACE-OFF / AIMNet2-decoupled / ANI):
        full N-path batch.
      * Coupled/polarizable (MACE-POL, AIMNet2-NSE): molecules couple globally, so
        a multi-structure block-diagonal batch is physically WRONG -> raises a
        clear NotImplementedError when B > 1 (B == 1 is allowed: one structure has
        no cross-coupling; use a 'sequential' calc or the serial ``HPC`` oracle).
    """

    def __init__(self, atoms_list, calc, output: str = "hpc_batch.out",
                 params=None, paras=None, device=None):
        self.atoms_list = [a.copy() for a in atoms_list]
        self.calc = calc
        self.output = output
        self.B = len(self.atoms_list)
        if self.B == 0:
            raise ValueError("HPCBatch: empty atoms_list")

        # --- backend gate ------------------------------------------------------
        if not _is_batch_calc_hpc(calc):
            raise ValueError(
                "HPCBatch requires a batched calculator implementing "
                "prepare()+get_ef_gpu() (got a non-batch calc). Use the "
                "single-structure HPC for a per-Atoms ASE calculator.")
        if not callable(getattr(calc, "get_efh_gpu", None)):
            raise NotImplementedError(
                "HPCBatch is Hessian-based and needs calc.get_efh_gpu() for the "
                "TS-anchor Hessian; this batch calc exposes only get_ef_gpu().")
        if _is_coupled_batch_calc(calc) and self.B > 1:
            raise NotImplementedError(
                "HPCBatch: this calculator globally couples molecules "
                "(MACE-POL / AIMNet2-NSE); a multi-structure block-diagonal batch "
                "is physically WRONG (charge/polarization would redistribute "
                "across structures). Run the paths one-at-a-time with the serial "
                "HPC oracle, or use a per-molecule 'sequential' calculator.")

        # --- params (shared, read-only during run) -----------------------------
        self.p = params if params is not None else HPCParams()
        _apply_paras_hpc(self.p, paras)

        self.device = (device if device is not None
                       else getattr(calc, "device", None)
                       or _torch.device("cpu"))

        # --- per-structure serial HPC "engines" (reuse EXACT single-HPC numpy) --
        # We never call engine.run()/engine.atoms.calc; the engines only lend their
        # byte-identical helpers (_gradient_mw_from_forces, _scale_mw_step,
        # _corrector_step, _mw_from_cart, static Bofill/BFGS, DWI) + hold per-path
        # state. Force/Hessian touches are intercepted by the generator's yields.
        self._engs = []
        self._n = []
        self._ts_cart = []          # (n_i, 3) float64 TS Cartesian per structure
        for a in self.atoms_list:
            eng = HPC(a, output, params=self.p)     # paras=None -> no re-aliasing
            eng._D = masses_D(a)
            self._engs.append(eng)
            ni = len(a)
            self._n.append(ni)
            self._ts_cart.append(
                np.asarray(a.get_positions(), dtype=np.float64).reshape(ni, 3))
        self.nmax_dof = 3 * max(self._n)

        # last Cartesian placed per structure (finished paths keep their last point
        # so the always-B batched forward has valid coords for every row).
        self._last_cart = [c.copy() for c in self._ts_cart]

        # diagnostics: batched-forward call counts (the whole point of batching)
        self._ef_calls = 0          # batched get_ef_gpu calls
        self._efh_calls = 0         # batched get_efh_gpu calls

    # ----------------------------- batched forward ------------------------------
    def _pack_coords(self):
        """Concatenate current per-structure Cartesian into the calc's packed
        (N, 3) order (== concat of atoms_list)."""
        rows = np.concatenate([c.reshape(-1, 3) for c in self._last_cart], axis=0)
        return _torch.as_tensor(rows, dtype=_torch.float64, device=self.device)

    def _batched_eval(self, requests):
        """Fulfil a phase of per-structure eval requests with ONE batched forward.

        ``requests``: {i: (kind, q_mw)} with kind in {'ef','efh'}. Places every
        requesting structure's Cartesian (finished ones keep last), then returns
        {i: (E_i, F_cart_i)} for 'ef' and {i: H_cart_i} for 'efh' (Hartree / Eh/A /
        Eh/A^2, unpadded to the structure's real DOFs).

        Force sourcing (reproducibility contract): 'ef' paths ALWAYS take their
        (E, F) from a pure ``get_ef_gpu`` forward, and ``get_efh_gpu`` is called
        ONLY to supply the Hessian ``H`` to 'efh' paths (its center E/F are
        discarded). The two are SEPARATE forwards over the same coord buffer, so a
        path's forces never depend on whether a sibling happened to need a Hessian
        this tick -> batch-composition-independent and bit-consistent with the
        single-HPC oracle (which always evaluates forces via get_ef_gpu). In the
        lockstep pump the two kinds never actually co-occur in one call, but the
        split keeps the contract robust regardless."""
        for i, (kind, q_mw) in requests.items():
            self._last_cart[i] = (q_mw * self._engs[i]._D).reshape(self._n[i], 3)
        self.calc.set_coords_(self._pack_coords())
        ef_idx = [i for i, (k, _q) in requests.items() if k == "ef"]
        efh_idx = [i for i, (k, _q) in requests.items() if k == "efh"]
        out = {}
        if ef_idx:
            E, F = self.calc.get_ef_gpu()
            self._ef_calls += 1
            for i in ef_idx:
                d3 = 3 * self._n[i]
                out[i] = (float(E[i].item()), to_f64(F[i, :d3]).reshape(-1))
        if efh_idx:
            _E, _F, H, _P = self.calc.get_efh_gpu()   # H only; center E/F discarded
            self._efh_calls += 1
            for i in efh_idx:
                d3 = 3 * self._n[i]
                out[i] = to_f64(H[i, :d3, :d3])
        return out

    def _ts_setup(self):
        """One batched get_efh_gpu at the TS geometries -> per-structure mode
        selection, mirroring HPC.run() but flagging (not raising) invalid TSs.

        Returns lists over B: E_ts, H_ts_cart, v_neg (or None), eigval, valid,
        n_strong_neg."""
        for i in range(self.B):
            self._last_cart[i] = self._ts_cart[i].copy()
        self.calc.set_coords_(self._pack_coords())
        E, F, H, _P = self.calc.get_efh_gpu()
        self._efh_calls += 1

        E_ts, H_ts, v_negs, eigvals, valids, nstrong = [], [], [], [], [], []
        k = int(self.p.target_mode)
        for i in range(self.B):
            d3 = 3 * self._n[i]
            D = self._engs[i]._D
            H_cart = to_f64(H[i, :d3, :d3])
            H_mw = (D[:, None] * H_cart) * D[None, :]
            w, V = np.linalg.eigh(H_mw)
            neg_idx = np.where(w < 0.0)[0]
            n_strong = int(np.sum(w < -1e-4))
            if len(neg_idx) == 0 or len(neg_idx) < k:
                E_ts.append(float(E[i].item())); H_ts.append(H_cart)
                v_negs.append(None); eigvals.append(float("nan"))
                valids.append(False); nstrong.append(n_strong)
                continue
            sorted_neg = neg_idx[np.argsort(w[neg_idx])]   # most negative first
            idx = sorted_neg[k - 1]
            E_ts.append(float(E[i].item()))
            H_ts.append(H_cart)
            v_negs.append(V[:, idx].copy())
            eigvals.append(float(w[idx]))
            valids.append(True)
            nstrong.append(n_strong)
        return E_ts, H_ts, v_negs, eigvals, valids, nstrong

    # ------------------------------ generators ----------------------------------
    # Literal transcriptions of HPC._micro_step / HPC._one_side with the two
    # calculator touches replaced by ``x = yield (kind, q_mw)``. Everything else
    # delegates to the per-structure engine's byte-identical numpy helpers.
    def _micro_step_gen(self, eng):
        """One HPC micro step for engine ``eng``; ``yield`` fetches (E,F)/H from the
        batched driver. Returns the MW displacement ``dx`` (StopIteration value)."""
        p = eng.p
        init_mw = eng.mw_coords.copy()

        e_curr, F_cart = yield ("ef", init_mw)
        g_curr = eng._gradient_mw_from_forces(F_cart)
        if _norm(g_curr) < 1e-12:
            return np.zeros_like(g_curr)

        recalc = (
            p.hessian_recalc is not None
            and p.hessian_recalc > 0
            and (eng.micro_counter % p.hessian_recalc == 0)
        )
        if recalc and eng.micro_counter > 0:
            H_cart = yield ("efh", init_mw)
            eng.mw_hessian = (eng._D[:, None] * H_cart) * eng._D[None, :]
        elif (eng.prev_coords is not None) and (eng.prev_grad is not None):
            gradient_diff = g_curr - eng.prev_grad
            coords_diff = init_mw - eng.prev_coords
            if str(p.hessian_update).lower() == "bofill":
                eng.mw_hessian = HPC._bofill_update(eng.mw_hessian, coords_diff, gradient_diff)
            else:
                eng.mw_hessian = HPC._bfgs_update(eng.mw_hessian, coords_diff, gradient_diff)

        eng.prev_coords = init_mw
        eng.prev_grad = g_curr

        if eng._dwi is not None and eng.micro_counter > 0:
            eng._dwi.update(init_mw.copy(), e_curr, g_curr, eng.mw_hessian.copy())

        eigvals, eigvecs = np.linalg.eigh(eng.mw_hessian)
        mask = np.abs(eigvals) > 1e-8
        if not np.any(mask):
            return np.zeros_like(g_curr)
        eigvals = eigvals[mask]
        eigvecs = eigvecs[:, mask]

        g_star = eigvecs.T.dot(g_curr)
        g_norm = _norm(g_curr)
        if g_norm < 1e-12:
            return np.zeros_like(g_curr)

        dt = eng._step_len_mw / (float(p.euler_n) * g_norm)
        t = dt
        cur_length = 0.0
        for _ in range(int(p.euler_n)):
            dsdt = np.sqrt(np.sum((g_star ** 2) * np.exp(-2.0 * eigvals * t)))
            cur_length += dsdt * dt
            if cur_length >= eng._step_len_mw:
                break
            t += dt

        alphas = (np.exp(-eigvals * t) - 1.0) / eigvals
        dx_pred = eigvecs.dot(alphas * g_star)
        pred_mw = init_mw + dx_pred

        E_pred, F_pred = yield ("ef", pred_mw)
        g_pred = eng._gradient_mw_from_forces(F_pred)

        dg_pred = g_pred - g_curr
        if str(p.hessian_update).lower() == "bofill":
            eng.mw_hessian = HPC._bofill_update(eng.mw_hessian, dx_pred, dg_pred)
        else:
            eng.mw_hessian = HPC._bfgs_update(eng.mw_hessian, dx_pred, dg_pred)

        if eng._dwi is not None:
            eng._dwi.update(pred_mw.copy(), E_pred, g_pred, eng.mw_hessian.copy())

        if eng._dwi is not None:
            corr_mw = eng._corrector_step(init_mw, eng._step_len_umw, eng._dwi)
        else:
            corr_mw = pred_mw

        eng.mw_coords = corr_mw
        eng.micro_counter += 1
        return eng.mw_coords - init_mw

    def _one_side_gen(self, eng, sign, ts_cart, v_neg_mw, E_ts, H_ts_cart):
        """One-sided HPC-IRC path for engine ``eng`` (transcription of
        HPC._one_side). Returns the per-side records as a DICT-OF-LISTS
        ``{"E":[...], "maxG":[...], "rmsG":[...], "x":[...]}`` -- the SAME schema
        LQABatch / EulerPCBatch return so the shared consumer
        ``irc/irc.py::_write_batched_lqa`` (which does ``records.get('x')``) treats
        every batched IRC integrator uniformly (StopIteration value)."""
        p = eng.p
        D = eng._D
        title = "FORWARD HPC-IRC" if sign > 0 else "BACKWARD HPC-IRC"

        q_ts_mw = eng._mw_from_cart(ts_cart)
        v_dir = _unit(v_neg_mw) * sign
        q0_mw = q_ts_mw + eng._scale_mw_step(v_dir, 0.5 * eng._step_len_mw)

        # gradient at TS (for the initial Hessian update)
        _e_ts, F_ts_cart = yield ("ef", q_ts_mw)
        g_ts_mw = eng._gradient_mw_from_forces(F_ts_cart)

        # initial Hessian at TS (MW) -- reuse the TS-anchor Hessian (same geometry
        # / deterministic FD as HPC._one_side's own get_hessian recompute).
        H0_mw = (D[:, None] * H_ts_cart) * D[None, :]

        # energy / forces / gradient at the displaced start point
        E0, F0_cart = yield ("ef", q0_mw)
        g0_mw = eng._gradient_mw_from_forces(F0_cart)
        maxF0 = float(np.max(np.abs(F0_cart)))
        rmsF0 = float(np.sqrt(np.mean(F0_cart ** 2)))

        if str(p.hessian_update).lower() == "bofill":
            H0_mw = HPC._bofill_update(H0_mw, q0_mw - q_ts_mw, g0_mw - g_ts_mw)
        else:
            H0_mw = HPC._bfgs_update(H0_mw, q0_mw - q_ts_mw, g0_mw - g_ts_mw)

        eng.mw_coords = q0_mw.copy()
        eng.mw_hessian = H0_mw.copy()
        eng.prev_coords = None
        eng.prev_grad = None
        eng.micro_counter = 0
        eng._dwi = DWI(n=p.dwi_n, maxlen=2)
        eng._dwi.update(q0_mw.copy(), E0, g0_mw, eng.mw_hessian.copy())

        ni = self._n_of(eng)
        rec = {"E": [], "maxG": [], "rmsG": [], "x": []}   # dict-of-lists (canonical)
        rec["E"].append(E0)
        rec["maxG"].append(maxF0)
        rec["rmsG"].append(rmsF0)
        rec["x"].append((q0_mw * D).reshape(ni, 3).copy())

        for _it in range(1, p.max_steps + 1):
            dx = yield from self._micro_step_gen(eng)
            if _norm(dx) <= 1e-12:
                break
            E_new, F_new = yield ("ef", eng.mw_coords)
            maxF = float(np.max(np.abs(F_new)))
            rmsF = float(np.sqrt(np.mean(F_new ** 2)))
            rec["E"].append(E_new)
            rec["maxG"].append(maxF)
            rec["rmsG"].append(rmsF)
            rec["x"].append((eng.mw_coords * D).reshape(ni, 3).copy())
            if (maxF <= p.f_max_th) and (rmsF <= p.f_rms_th):
                break

        return rec

    def _n_of(self, eng):
        return len(eng.atoms)

    # -------------------------------- pump --------------------------------------
    def _pump(self, gens):
        """Drive a list of B side-generators (None = skipped/invalid) in lockstep,
        fusing each phase's per-structure eval requests into ONE batched forward.
        Returns a list of each generator's StopIteration value (or None)."""
        results = [None] * self.B
        pending = {}
        for i, g in enumerate(gens):
            if g is None:
                continue
            try:
                pending[i] = next(g)
            except StopIteration as e:
                results[i] = e.value
                gens[i] = None
            except Exception as exc:  # per-path failure must NOT poison the batch
                log_error([f"[hpc-batch] path {i} raised at init ({exc!r}); "
                           f"marked invalid, batch continues\n"], self.output)
                results[i] = None
                gens[i] = None
        while pending:
            out = self._batched_eval(pending)
            nxt = {}
            for i in pending:
                try:
                    nxt[i] = gens[i].send(out[i])
                except StopIteration as e:
                    results[i] = e.value
                    gens[i] = None
                except Exception as exc:  # isolate a diverged path; keep the batch
                    log_error([f"[hpc-batch] path {i} raised mid-run ({exc!r}); "
                               f"marked invalid, batch continues\n"], self.output)
                    results[i] = None
                    gens[i] = None
            pending = nxt
        return results

    # -------------------------------- run ---------------------------------------
    def run(self):
        """Run batched forward+backward HPC-IRC for all B structures.

        Returns a list (len B) of per-structure result dicts -- schema IDENTICAL to
        LQABatch / EulerPCBatch so ``irc/irc.py::_write_batched_lqa`` consumes all
        three uniformly:
            {"index": i, "valid": bool, "neg_eigval": float, "n_strong_neg": int,
             "E_ts": float,
             "forward":  {"records": {"E":[...],"maxG":[...],"rmsG":[...],"x":[...]}},
             "backward": {"records": {"E":[...],"maxG":[...],"rmsG":[...],"x":[...]}}}
        Invalid structures (no clean negative TS mode) get valid=False and empty
        forward/backward record lists (flagged, not propagated)."""
        self.calc.prepare(self.atoms_list, fixed_nmax=self.nmax_dof)

        E_ts, H_ts, v_negs, eigvals, valids, nstrong = self._ts_setup()

        # forward (+1) then backward (-1); each side is a full lockstep pump.
        fwd_gens = [
            self._one_side_gen(self._engs[i], +1.0, self._ts_cart[i].reshape(-1),
                               v_negs[i], E_ts[i], H_ts[i]) if valids[i] else None
            for i in range(self.B)
        ]
        fwd = self._pump(fwd_gens)

        bwd_gens = [
            self._one_side_gen(self._engs[i], -1.0, self._ts_cart[i].reshape(-1),
                               v_negs[i], E_ts[i], H_ts[i]) if valids[i] else None
            for i in range(self.B)
        ]
        bwd = self._pump(bwd_gens)

        results = []
        for i in range(self.B):
            results.append({
                "index": i,
                "valid": bool(valids[i]),
                "neg_eigval": float(eigvals[i]),
                "n_strong_neg": int(nstrong[i]),
                "E_ts": float(E_ts[i]),
                "forward": {"records": fwd[i]} if fwd[i] is not None
                           else {"records": {"E": [], "maxG": [], "rmsG": [], "x": []}},
                "backward": {"records": bwd[i]} if bwd[i] is not None
                            else {"records": {"E": [], "maxG": [], "rmsG": [], "x": []}},
            })
        return results


def run_hpc_irc(atoms_or_list, output: str = "hpc.out", paras=None,
                calc=None, params=None, device=None):
    """Unified HPC-IRC entry point (single-structure backward compatible).

    - Single ASE ``Atoms`` (with an attached ``atoms.calc``): dispatch to the
      original single-structure ``HPC`` (unchanged behaviour / units / outputs).
    - list/tuple of ``Atoms``: dispatch to the batched ``HPCBatch``, consuming the
      supplied GPU batch calculator ``calc`` (prepare/get_ef_gpu/get_efh_gpu API).
    """
    if isinstance(atoms_or_list, (list, tuple)):
        if calc is None:
            raise ValueError("run_hpc_irc(batch): a batch calculator `calc` is required")
        return HPCBatch(list(atoms_or_list), calc, output=output,
                        params=params, paras=paras, device=device).run()
    hpc = HPC(atoms_or_list, output=output, params=params, paras=paras)
    return hpc.run()
