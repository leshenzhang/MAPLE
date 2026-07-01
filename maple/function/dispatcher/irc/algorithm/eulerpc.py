# -*- coding: utf-8 -*-
"""
Intrinsic Reaction Coordinate (IRC) integrator using Euler-based Predictor-Corrector(EulerPC):

- Mass-weighted coordinates and Hessian
- Predictor Euler integration with a Hessian-based gradient model
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
class EulerPCParams:
    # Which negative eigenmode (1 = most negative) to use for initial direction
    target_mode: int = 1

    # EulerPC step length in Bohr
    step_length_bohr: float = 0.10

    # Number of macro steps per direction
    max_steps: int = 50

    # Predictor Euler integration sub-steps
    max_pred_steps: int = 500
    loose_cycles: int = 3

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


# ================================== EulerPC ===================================
class EulerPC:
    """
    Euler integration with a Hessian-based Predictor-Corrector:

    - Works in mass-weighted coordinates.
    - Each macro step uses Euler predictor integration followed by
      DWI + mBS corrector integration.
    - Uses BFGS/Bofill updates of the mass-weighted Hessian, with optional
      periodic full recalculation.
    - Integrates forward and backward from the TS along the lowest
      negative eigenmode of H_mw, then merges paths.
    """

    def __init__(self, atoms: Atoms, output: str,
                 params: Optional[EulerPCParams] = None,
                 paras: Optional[dict] = None):
        self.atoms = atoms
        self.output = output
        self.p = params if params is not None else EulerPCParams()

        # Parse optional dict overrides, supporting both {"EulerPC": {...}}
        # and flat dict style. Keep backward compatibility aliases.
        if isinstance(paras, dict):
            low = {k.lower(): v for k, v in paras.items()}
            sub = None
            for key in ("EulerPC", "irc"):
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
                "max_pred_steps": "max_pred_steps",
                "loose_cycles": "loose_cycles",
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

        # Internal state for EulerPC integration
        self._D: Optional[np.ndarray] = None  # mass-weight scaling vector
        self._step_len_mw: float = float(self.p.step_length_bohr * BOHR_TO_ANG)
        self._step_len_umw: float = float(self.p.step_length_bohr * BOHR_TO_ANG)
        self.mw_coords: Optional[np.ndarray] = None
        self.mw_hessian: Optional[np.ndarray] = None
        self.prev_coords: Optional[np.ndarray] = None
        self.prev_grad: Optional[np.ndarray] = None
        self.micro_counter: int = 0
        self._dwi: Optional[DWI] = None
        self.cur_cycle: int = 0
        self._early_converged: bool = False

    # ------------------------------ Public API ------------------------------
    def run(self) -> Dict[str, any]:
        """
        Compute forward and backward EulerPC IRC paths and produce a merged summary.

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
                ["[ERROR] EulerPC-IRC: No negative eigenvalues found — "
                 "starting geometry is not a saddle point.\n"],
                self.output,
            )
            raise RuntimeError("EulerPC-IRC: no negative eigenvalues at TS.")

        if len(neg_idx) < self.p.target_mode:
            log_error(
                [f"[ERROR] EulerPC-IRC: Requested mode {self.p.target_mode}, "
                 f"but only {len(neg_idx)} negative modes found.\n"],
                self.output,
            )
            raise RuntimeError("EulerPC-IRC: requested negative mode does not exist.")

        sorted_neg = neg_idx[np.argsort(w[neg_idx])]  # most negative first
        idx = sorted_neg[self.p.target_mode - 1]
        eigval = w[idx]
        v_neg_mw = V[:, idx]

        log_info(
            [
                "\n[INFO] EulerPC-IRC: Selected negative eigenmode "
                f"#{self.p.target_mode} with λ = {eigval:.6e} (MW basis)\n"
            ],
            self.output,
        )

        # Reference TS energy
        E_ts = float(self.atoms.get_potential_energy(force_consistent=True))

        # Store original TS Cartesian positions, reused for both directions
        R_ts_cart = self.atoms.get_positions().copy().reshape(-1)

        # Forward and backward EulerPC-IRC
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

    # --------------------------- Micro-step (EulerPC) ----------------------------
    def _micro_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one EulerPC step using DWI + mBS corrector.

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

        # Predictor: Euler integration with Hessian-based gradient model
        conv_fact = self._get_conv_fact(g_curr)
        euler_step_len = self._step_len_mw / (float(self.p.max_pred_steps) / conv_fact)

        euler_mw = init_mw.copy()
        euler_grad = g_curr.copy()
        pred_converged = False
        for _ in range(int(self.p.max_pred_steps)):
            if self._unweight_len(euler_mw - init_mw) >= self._step_len_umw:
                pred_converged = True
                break
            grad_norm = _norm(euler_grad)
            if grad_norm < 1e-12:
                break
            step = euler_step_len * (-euler_grad) / grad_norm
            euler_mw = euler_mw + step
            euler_step = euler_mw - init_mw
            euler_grad = g_curr + self.mw_hessian.dot(euler_step)

        pred_mw = euler_mw

        if not pred_converged:
            # If predictor misses target length, keep going: either early-converge
            # (small Cartesian gradient) or continue with corrector.
            euler_grad_cart = self._unweight_grad(euler_grad)
            rms_grad = float(np.sqrt(np.mean(euler_grad_cart ** 2)))
            if self.cur_cycle < int(self.p.loose_cycles):
                log_info(
                    [
                        "[INFO] EulerPC-IRC: Predictor did not reach target length; "
                        "entering loose-cycle mode (relaxed convergence).\n"
                    ],
                    self.output,
                )
            elif rms_grad <= float(self.p.f_rms_th):
                self._early_converged = True
                self.mw_coords = pred_mw
                dx = self.mw_coords - init_mw
                return dx, euler_grad
            else:
                log_info(
                    [
                        "[INFO] EulerPC-IRC: Predictor did not converge within max_pred_steps; "
                        f"rms_grad={rms_grad:.6e} > tol={float(self.p.f_rms_th):.6e}. "
                        "Continuing with corrector.\n"
                    ],
                    self.output,
                )

        # Evaluate predicted point and update Hessian
        E_pred, F_pred = self._energy_forces_from_mw(pred_mw)
        g_pred = self._gradient_mw_from_forces(F_pred)

        dx_pred = pred_mw - init_mw
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
        Single-sided EulerPC IRC integration.

        Args:
            forward: True for forward path, False for backward.
            sign: +1.0 or -1.0 to choose direction along the negative mode.
            q_ts_cart: TS Cartesian coordinates (flattened, Å).
            v_neg_mw: selected negative eigenmode in MW basis (3N, normalized later).
            E_ts: TS reference energy (Eh).
        """
        p = self.p
        title = "FORWARD EulerPC-IRC" if forward else "BACKWARD EulerPC-IRC"

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

        # Initialize EulerPC state
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
            self.cur_cycle = it - 1
            self._early_converged = False
            dx, _ = self._micro_step()
            if _norm(dx) <= 1e-12:
                log_info(
                    [f"[INFO] {title}: step too small at step {it}, stopping.\n"],
                    self.output,
                )
                break

            if self._early_converged:
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
                self._print_hurray()
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
            log_error(["[ERROR] EulerPC-IRC: One path side is empty.\n"], self.output)
            raise RuntimeError("EulerPC-IRC: one path side is empty.")

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
                "                       EulerPC-IRC PATH SUMMARY           \n",
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
                f"\n[INFO] EulerPC-IRC forward trajectory written to: {fwd_path}\n",
                f"[INFO] EulerPC-IRC backward trajectory written to: {bwd_path}\n",
                f"[INFO] EulerPC-IRC full trajectory written to: {full_path}\n",
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
                "                      ***          THE EulerPC-IRC HAS CONVERGED        ***\n",
                "                      *************************************************\n\n",
            ],
            self.output,
        )


# ======================================================================== #
#                          BATCHED EulerPC-IRC                              #
# ======================================================================== #
# Batched IRC over B transition-state structures sharing ONE *batch* calculator
# (prepare / get_ef_gpu [+ get_efh_gpu] / set_coords_). All B independent paths
# are propagated in LOCKSTEP: the per-path EulerPC control flow (predictor-Euler,
# DWI + mBS corrector, BFGS/Bofill Hessian history, per-path convergence) stays
# BYTE-for-BYTE the single-structure oracle's numpy -- ONLY the force / Hessian
# EVALUATION is batched. Per the project profiling the MLIP forward is 95-99% of
# wall time, so co-batching every path's force call into ONE GPU forward per
# rendezvous is the entire win.
#
# Mechanism (single-threaded, deterministic, no locks): each path runs as a
# Python generator that mirrors EulerPC.run/_one_side/_micro_step but replaces
#   E, F = self._energy_forces_from_mw(q_mw)    ->   E, F = yield ('ef', cart)
#   H    = self._get_hessian_cart()             ->   H    = yield ('h',  cart)
# The driver collects every live generator's pending request, packs all cartesian
# coords into the calc's (N,3) buffer, and fires ONE batched get_ef_gpu() -- forces
# ALWAYS come from this pure forward (exactly like the single oracle) -- plus, only
# on ticks where some path needs a Hessian, ONE get_efh_gpu() whose H alone is used
# (its E,F discarded, since a backend's FD-Hessian center force can differ from the
# standalone get_ef_gpu forward). Each path is then `.send()`ed its own slice.
# Because every path executes the identical oracle numpy given its own get_ef_gpu
# forces, the batched paths reproduce the single-structure EulerPC oracle to the
# calculator's B=1-vs-B=N determinism (the fp32 batching-noise floor).
#
# Units identical to single-structure EulerPC: positions Angstrom, forces Eh/A,
# Hessian Eh/A^2, energy Eh, MW coords sqrt(amu)*A with q_mw = q_cart / D,
# H_mw = D (x) D * H, g_mw = D * g_cart  (D = 1/sqrt(m)).
import warnings as _warnings
import torch as _torch


# Coupled / globally-polarizable batch calculators: co-batching independent
# molecules into ONE graph makes MACE-POL do a global total-charge-constrained
# charge equilibration + long-range Coulomb, and AIMNet2-NSE a global charge
# equilibration, that transfer charge BETWEEN molecules -> a multi-structure
# batch is PHYSICALLY WRONG. These fall back to a sequential B=1 loop (correct
# single-molecule IRC, no batch speedup). Decoupled variants are safe.
_COUPLED_BATCH_CALC_NAMES = {"MACEPolBatchCalc", "AIMNet2BatchCalc"}


def _is_batch_calc(calc) -> bool:
    """Duck-typed test mirroring dispatcher._is_batch_calc: a batched calculator
    exposes prepare() + get_ef_gpu()."""
    return (calc is not None
            and callable(getattr(calc, "prepare", None))
            and callable(getattr(calc, "get_ef_gpu", None)))


def _is_coupled_calc(calc) -> bool:
    """True for globally-coupled / polarizable batch calcs (MACE-POL, AIMNet2-NSE)
    whose multi-structure batch couples molecules. Decoupled variants -> False."""
    name = type(calc).__name__
    if "decoupled" in name.lower():
        return False
    if getattr(calc, "batch_decoupled", False) is True:
        return False
    if name in _COUPLED_BATCH_CALC_NAMES:
        return True
    if hasattr(calc, "coupling_mode"):          # MACE-POL exposes this
        return True
    if getattr(calc, "couples_molecules", False) is True:
        return True
    nl = name.lower()                           # class-name tag (matches hpc gate)
    if "pol" in nl or "nse" in nl:
        return True
    mn = str(getattr(calc, "_model_name", None)
             or getattr(calc, "model_name", None) or "").lower()
    if "pol" in mn or "nse" in mn:
        return True
    return False


def _apply_paras_eulerpc(p: "EulerPCParams", paras):
    """Apply a {'EulerPC'|'irc': {...}} or flat dict of overrides onto an
    EulerPCParams. Same alias table as EulerPC.__init__ (kept separate so the
    batch path reuses it without touching the single-structure constructor)."""
    if not isinstance(paras, dict):
        return p
    low = {k.lower(): v for k, v in paras.items()}
    sub = None
    for key in ("eulerpc", "irc"):
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
        "max_pred_steps": "max_pred_steps",
        "loose_cycles": "loose_cycles",
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


class EulerPCBatch:
    """Batched EulerPC-IRC integrator over B transition-state structures.

    OPT-IN: the single-structure ``EulerPC`` above is untouched and remains the
    parity oracle. ``EulerPCBatch`` runs N independent IRC paths in lockstep and
    batches every force / Hessian evaluation into ONE GPU forward per rendezvous.

    Consumes the *batch* calculator API (NOT a per-atoms ASE calculator):
        calc.prepare(atoms_list, fixed_nmax)
        calc.get_ef_gpu()  -> (E (B,), F (B, nmax_dof))            [Hartree, Eh/A]
        calc.get_efh_gpu() -> (E (B,), F (B,M), H (B,M,M), P (B,)) [+ Eh/A^2]
        calc.set_coords_(coord (N,3))    # packed Cartesian, calc atom order

    Parameters mirror ``EulerPCParams`` (same defaults / ``paras`` overrides). The
    per-path algorithm is IDENTICAL to the single-structure oracle (predictor-Euler
    + DWI/mBS corrector + BFGS/Bofill Hessian history + per-path convergence); only
    the force evaluation is co-batched. Backend gate: a batchable calc runs the full
    N-path batch; a globally-coupled / polarizable calc (MACE-POL, AIMNet2-NSE)
    falls back to a sequential B=1 loop (correct single-molecule physics).
    """

    def __init__(self, atoms_list, calc, output: str = "eulerpc_batch.out",
                 params=None, paras=None, device=None):
        self.atoms_list = list(atoms_list)
        self.calc = calc
        self.output = output
        # Resolve to a concrete torch.device so the packed coord buffer lives on
        # the same device the batched calc consumes (mirrors LQABatch).
        self.device = (_torch.device(device) if device is not None
                       else _torch.device("cuda" if _torch.cuda.is_available() else "cpu"))
        self.p = params if params is not None else EulerPCParams()
        _apply_paras_eulerpc(self.p, paras)

        self.B = len(self.atoms_list)
        if self.B == 0:
            raise ValueError("EulerPCBatch: empty atoms_list")

        if not _is_batch_calc(calc):
            raise NotImplementedError(
                "EulerPCBatch requires a batched calculator exposing "
                "prepare()+get_ef_gpu() (UMA / standard MACE / MACE-OFF / "
                "AIMNet2-decoupled / ANI). Got "
                f"{type(calc).__name__}; use the single-structure EulerPC for a "
                "per-Atoms ASE calculator.")
        if not callable(getattr(calc, "get_efh_gpu", None)):
            raise NotImplementedError(
                "EulerPCBatch needs get_efh_gpu() for the TS Hessian / negative-"
                f"mode selection; {type(calc).__name__} does not provide it.")

        # diagnostics (mirror LQABatch._efh_calls / _ef_calls)
        self._ef_calls = 0        # cheap get_ef_gpu forwards (the 95-99% hot path)
        self._efh_calls = 0       # exact get_efh_gpu Hessian forwards (setup)

        # coupled/polarizable backends -> a multi-structure batch is physically
        # wrong (see module note). Fall back to a sequential B=1 loop.
        self._coupled = _is_coupled_calc(calc)
        self._sequential = bool(self._coupled and self.B > 1)
        if self._sequential:
            _warnings.warn(
                f"EulerPCBatch: {type(calc).__name__} globally couples molecules "
                "(MACE-POL / AIMNet2-NSE class); a multi-structure batch would be "
                "physically wrong. Falling back to a sequential B=1 loop (correct "
                "single-molecule IRC, no GPU-batch speedup).", RuntimeWarning)

    # ------------------------------- run ----------------------------------
    def run(self):
        """Run batched forward+backward EulerPC-IRC for all B structures.

        Returns a list (len B) of per-structure result dicts (schema mirrors
        LQABatch so the central IRC writer consumes both):
            {"index": i, "valid": bool, "neg_eigval": float, "n_strong_neg": int,
             "E_ts": float, "forward": {"records": {E,maxG,rmsG,x}},
             "backward": {"records": {E,maxG,rmsG,x}}}
        Invalid structures (no clean negative TS mode) get valid=False + empty
        forward/backward paths (flagged, not propagated) instead of raising.
        """
        if self._sequential:
            results = []
            for i, at in enumerate(self.atoms_list):
                results.extend(self._run_group([(i, at)]))
            return results
        return self._run_group([(i, at) for i, at in enumerate(self.atoms_list)])

    # ------------------------------------------------------------ one group
    def _run_group(self, items):
        """Prepare the calc for `items` (list of (global_index, Atoms)) and drive
        all their paths in lockstep. B == len(items) here (B=1 in the sequential
        fallback, B=N in the batched path)."""
        atoms_sub = [at for _, at in items]
        nmax_dof = 3 * max(len(at) for at in atoms_sub)
        self.calc.prepare(atoms_sub, fixed_nmax=nmax_dof)

        pcs = [EulerPC(at, self.output, params=self.p) for _, at in items]
        gens = [self._path_gen(pc) for pc in pcs]
        local = self._drive(items, gens)

        out = []
        for (gidx, _), r in zip(items, local):
            r = dict(r)
            r["index"] = gidx
            out.append(r)
        return out

    # -------------------------------- driver ------------------------------
    def _drive(self, items, gens):
        """Lockstep coroutine driver: rendezvous every live generator's pending
        force/Hessian request into ONE batched forward per tick."""
        B = len(items)
        n_atoms = [len(at) for _, at in items]
        row0 = [0]
        for n in n_atoms:
            row0.append(row0[-1] + n)
        N = row0[-1]
        dev = self.device

        # packed Cartesian buffer (N,3) on the batch device, atom order ==
        # concat(atoms_sub) (== the calc's own coord layout after prepare()).
        pack = _torch.zeros((N, 3), dtype=_torch.float64, device=dev)
        for k, (_, at) in enumerate(items):
            pack[row0[k]:row0[k + 1]] = _torch.as_tensor(
                np.asarray(at.get_positions(), dtype=np.float64), device=dev)

        results = [None] * B
        pending = {}
        for k, g in enumerate(gens):
            try:
                pending[k] = next(g)
            except StopIteration as e:
                results[k] = e.value
            except Exception as exc:  # per-path failure must NOT poison the batch
                log_error([f"[eulerpc-batch] path {k} raised at init ({exc!r}); "
                           f"marked invalid, batch continues\n"], self.output)
                results[k] = None

        while pending:
            need_h = any(req[0] == 'h' for req in pending.values())
            for k, (_typ, cart) in pending.items():
                nk = n_atoms[k]
                pack[row0[k]:row0[k + 1]] = _torch.as_tensor(
                    np.asarray(cart, dtype=np.float64).reshape(nk, 3), device=dev)
            self.calc.set_coords_(pack)

            # FORCES ALWAYS come from a PURE get_ef_gpu() -- the single EulerPC
            # oracle's forces do too. A backend's get_efh_gpu() returns a center
            # force that can differ from get_ef_gpu() (e.g. UMA's FD-Hessian center
            # vs the standalone forward), so an 'ef'-requesting path must never be
            # served forces out of the Hessian forward, or its first/every step
            # walks a slightly different trajectory than the oracle. When any path
            # needs a Hessian this tick, ALSO fire get_efh_gpu() but use ONLY its
            # H (its E,F are discarded).
            E, F = self.calc.get_ef_gpu()
            self._ef_calls += 1
            if need_h:
                _E2, _F2, H, _P = self.calc.get_efh_gpu()
                self._efh_calls += 1
            else:
                H = None

            E_cpu = E.detach().cpu()
            F_cpu = F.detach().cpu()
            H_cpu = H.detach().cpu() if H is not None else None

            new_pending = {}
            for k, (typ, _cart) in pending.items():
                nk = n_atoms[k]
                if typ == 'h':
                    send_val = H_cpu[k, :3 * nk, :3 * nk].numpy().astype(np.float64)
                else:
                    Ek = float(E_cpu[k])
                    Fk = F_cpu[k, :3 * nk].numpy().astype(np.float64)  # (3nk,) flat
                    send_val = (Ek, Fk)
                try:
                    new_pending[k] = gens[k].send(send_val)
                except StopIteration as e:
                    results[k] = e.value
                except Exception as exc:  # isolate a diverged path; keep the batch
                    log_error([f"[eulerpc-batch] path {k} raised mid-run ({exc!r}); "
                               f"marked invalid, batch continues\n"], self.output)
                    results[k] = None
            pending = new_pending

        return results

    # ---------------------------- record helper ---------------------------
    @staticmethod
    def _append_record(records, E, maxG, rmsG, cart_flat, natoms):
        records["E"].append(float(E))
        records["maxG"].append(float(maxG))
        records["rmsG"].append(float(rmsG))
        records["x"].append(
            np.asarray(cart_flat, dtype=np.float64).reshape(natoms, 3).copy())

    # ------------------------- per-path generators ------------------------
    # These mirror EulerPC.run / _one_side / _micro_step VERBATIM in numpy,
    # reusing the oracle instance `pc` for every pure (force-free) helper (DWI,
    # BFGS/Bofill, predictor, DWI+mBS corrector, MW<->Cartesian conversions), and
    # replacing each force/Hessian evaluation with a `yield`. Per-path logging is
    # suppressed (numerically inert side effect) to avoid interleaved file writes;
    # the batched summary is written centrally by the IRC dispatcher.

    def _path_gen(self, pc):
        """Full EulerPC run for one path (mirror of EulerPC.run)."""
        pc._D = masses_D(pc.atoms)
        pc._step_len_mw = float(pc.p.step_length_bohr * BOHR_TO_ANG)
        pc._step_len_umw = float(pc.p.step_length_bohr * BOHR_TO_ANG)

        R_ts_cart = pc.atoms.get_positions().copy().reshape(-1)

        # Hessian at TS (batched) -> negative mode
        H_cart_ts = to_f64((yield ('h', R_ts_cart)))
        H_mw_ts = (pc._D[:, None] * H_cart_ts) * pc._D[None, :]
        w, V = np.linalg.eigh(H_mw_ts)
        neg_idx = np.where(w < 0.0)[0]
        n_strong_neg = int(np.sum(w < -1e-4))

        # Reference TS energy (mirror oracle order: after mode diagonalization)
        E_ts_val, _F = yield ('ef', R_ts_cart)
        E_ts = float(E_ts_val)

        empty = {"E": [], "maxG": [], "rmsG": [], "x": []}
        if len(neg_idx) == 0 or len(neg_idx) < pc.p.target_mode:
            didx = min(pc.p.target_mode - 1, len(w) - 1)
            return {"index": None, "valid": False,
                    "neg_eigval": float(w[didx]), "n_strong_neg": n_strong_neg,
                    "E_ts": E_ts,
                    "forward": {"records": dict(empty)},
                    "backward": {"records": dict(empty)}}

        sorted_neg = neg_idx[np.argsort(w[neg_idx])]     # most negative first
        idx = sorted_neg[pc.p.target_mode - 1]
        eigval = float(w[idx])
        # Use the RAW np.linalg.eigh eigenvector -- BYTE-IDENTICAL to the single
        # EulerPC oracle's neg-mode selection (which does not canonicalize the
        # sign). Given the same TS Hessian, batch and oracle then pick the SAME
        # eigenvector with the SAME LAPACK sign, so batch(B=1) reproduces the
        # oracle exactly (no reliance on downstream side-matching to undo a sign).
        # [Earlier a canonical-phase flip was applied here; it made batch's
        # forward/backward *labeling* deviate from the oracle -- removed.]
        v_neg_mw = V[:, idx].copy()

        forward_records = yield from self._one_side_gen(pc, +1.0, R_ts_cart, v_neg_mw, E_ts)
        backward_records = yield from self._one_side_gen(pc, -1.0, R_ts_cart, v_neg_mw, E_ts)

        return {"index": None, "valid": True,
                "neg_eigval": eigval, "n_strong_neg": n_strong_neg, "E_ts": E_ts,
                "neg_eigvec_mw": v_neg_mw.copy(),   # diagnostic: MW neg-mode eigenvector
                "forward": {"records": forward_records},
                "backward": {"records": backward_records}}

    def _one_side_gen(self, pc, sign, q_ts_cart, v_neg_mw, E_ts):
        """One-sided EulerPC integration (mirror of EulerPC._one_side)."""
        p = pc.p
        natoms = len(pc.atoms)

        q_ts_mw = pc._mw_from_cart(q_ts_cart)
        v_dir = _unit(v_neg_mw) * sign
        q0_mw = q_ts_mw + pc._scale_mw_step(v_dir, 0.5 * pc._step_len_mw)

        # Gradient at TS
        _E, F_ts_cart = yield ('ef', pc._cart_from_mw(q_ts_mw))
        g_ts_mw = pc._gradient_mw_from_forces(to_f64(F_ts_cart))

        # Initial Hessian at TS (MW)
        H_ts_cart = to_f64((yield ('h', pc._cart_from_mw(q_ts_mw))))
        H0_mw = (pc._D[:, None] * H_ts_cart) * pc._D[None, :]

        # Energy / forces at displaced start point
        E0, F0_cart = yield ('ef', pc._cart_from_mw(q0_mw))
        E0 = float(E0)
        F0_cart = to_f64(F0_cart)
        g0_mw = pc._gradient_mw_from_forces(F0_cart)
        maxF0 = float(np.max(np.abs(F0_cart)))
        rmsF0 = float(np.sqrt(np.mean(F0_cart ** 2)))

        # Hessian update TS -> displaced point
        if str(p.hessian_update).lower() == "bofill":
            H0_mw = pc._bofill_update(H0_mw, q0_mw - q_ts_mw, g0_mw - g_ts_mw)
        else:
            H0_mw = pc._bfgs_update(H0_mw, q0_mw - q_ts_mw, g0_mw - g_ts_mw)

        # Initialize EulerPC state
        pc.mw_coords = q0_mw.copy()
        pc.mw_hessian = H0_mw.copy()
        pc.prev_coords = None
        pc.prev_grad = None
        pc.micro_counter = 0
        pc._dwi = DWI(n=p.dwi_n, maxlen=2)
        pc._dwi.update(q0_mw.copy(), E0, g0_mw, pc.mw_hessian.copy())

        records = {"E": [], "maxG": [], "rmsG": [], "x": []}
        self._append_record(records, E0, maxF0, rmsF0, pc._cart_from_mw(q0_mw), natoms)

        for it in range(1, p.max_steps + 1):
            pc.cur_cycle = it - 1
            pc._early_converged = False
            dx, _g = yield from self._micro_step_gen(pc)
            if _norm(dx) <= 1e-12:
                break

            if pc._early_converged:
                E_new, F_new = yield ('ef', pc._cart_from_mw(pc.mw_coords))
                E_new = float(E_new)
                F_new = to_f64(F_new)
                maxF = float(np.max(np.abs(F_new)))
                rmsF = float(np.sqrt(np.mean(F_new ** 2)))
                self._append_record(records, E_new, maxF, rmsF,
                                     pc._cart_from_mw(pc.mw_coords), natoms)
                break

            E_new, F_new = yield ('ef', pc._cart_from_mw(pc.mw_coords))
            E_new = float(E_new)
            F_new = to_f64(F_new)
            maxF = float(np.max(np.abs(F_new)))
            rmsF = float(np.sqrt(np.mean(F_new ** 2)))
            self._append_record(records, E_new, maxF, rmsF,
                                 pc._cart_from_mw(pc.mw_coords), natoms)

            if (maxF <= p.f_max_th) and (rmsF <= p.f_rms_th):
                break

        return records

    def _micro_step_gen(self, pc):
        """One EulerPC predictor-corrector micro step (mirror of
        EulerPC._micro_step). Yields for the current-point and predicted-point
        force evals (and optional Hessian recalc); predictor + corrector are pure
        numpy (no force eval)."""
        p = pc.p
        init_mw = pc.mw_coords.copy()

        # Current energy / gradient
        e_curr, F_cart = yield ('ef', pc._cart_from_mw(init_mw))
        e_curr = float(e_curr)
        g_curr = pc._gradient_mw_from_forces(to_f64(F_cart))
        if _norm(g_curr) < 1e-12:
            return np.zeros_like(g_curr), g_curr

        # Update Hessian (or optionally recalc)
        recalc = (
            p.hessian_recalc is not None
            and p.hessian_recalc > 0
            and (pc.micro_counter % p.hessian_recalc == 0)
        )
        if recalc and pc.micro_counter > 0:
            H_cart = to_f64((yield ('h', pc._cart_from_mw(init_mw))))
            pc.mw_hessian = (pc._D[:, None] * H_cart) * pc._D[None, :]
        elif (pc.prev_coords is not None) and (pc.prev_grad is not None):
            gradient_diff = g_curr - pc.prev_grad
            coords_diff = init_mw - pc.prev_coords
            if str(p.hessian_update).lower() == "bofill":
                pc.mw_hessian = pc._bofill_update(pc.mw_hessian, coords_diff, gradient_diff)
            else:
                pc.mw_hessian = pc._bfgs_update(pc.mw_hessian, coords_diff, gradient_diff)

        pc.prev_coords = init_mw
        pc.prev_grad = g_curr

        if pc._dwi is not None and pc.micro_counter > 0:
            pc._dwi.update(init_mw.copy(), e_curr, g_curr, pc.mw_hessian.copy())

        # Predictor: Euler integration with Hessian-based gradient model (no eval)
        conv_fact = pc._get_conv_fact(g_curr)
        euler_step_len = pc._step_len_mw / (float(p.max_pred_steps) / conv_fact)

        euler_mw = init_mw.copy()
        euler_grad = g_curr.copy()
        pred_converged = False
        for _ in range(int(p.max_pred_steps)):
            if pc._unweight_len(euler_mw - init_mw) >= pc._step_len_umw:
                pred_converged = True
                break
            grad_norm = _norm(euler_grad)
            if grad_norm < 1e-12:
                break
            step = euler_step_len * (-euler_grad) / grad_norm
            euler_mw = euler_mw + step
            euler_step = euler_mw - init_mw
            euler_grad = g_curr + pc.mw_hessian.dot(euler_step)

        pred_mw = euler_mw

        if not pred_converged:
            euler_grad_cart = pc._unweight_grad(euler_grad)
            rms_grad = float(np.sqrt(np.mean(euler_grad_cart ** 2)))
            if pc.cur_cycle < int(p.loose_cycles):
                pass  # loose-cycle mode (log suppressed in batch)
            elif rms_grad <= float(p.f_rms_th):
                pc._early_converged = True
                pc.mw_coords = pred_mw
                dx = pc.mw_coords - init_mw
                return dx, euler_grad
            else:
                pass  # continue with corrector (log suppressed in batch)

        # Evaluate predicted point and update Hessian
        E_pred, F_pred = yield ('ef', pc._cart_from_mw(pred_mw))
        E_pred = float(E_pred)
        g_pred = pc._gradient_mw_from_forces(to_f64(F_pred))

        dx_pred = pred_mw - init_mw
        dg_pred = g_pred - g_curr
        if str(p.hessian_update).lower() == "bofill":
            pc.mw_hessian = pc._bofill_update(pc.mw_hessian, dx_pred, dg_pred)
        else:
            pc.mw_hessian = pc._bfgs_update(pc.mw_hessian, dx_pred, dg_pred)

        if pc._dwi is not None:
            pc._dwi.update(pred_mw.copy(), E_pred, g_pred, pc.mw_hessian.copy())

        # Corrector: mBS integration on DWI surface (no eval)
        if pc._dwi is not None:
            corr_mw = pc._corrector_step(init_mw, pc._step_len_umw, pc._dwi)
        else:
            corr_mw = pred_mw

        pc.mw_coords = corr_mw
        pc.micro_counter += 1

        dx = pc.mw_coords - init_mw
        return dx, g_pred


def run_eulerpc_irc(atoms_or_list, output: str = "eulerpc.out", paras=None,
                    calc=None, params=None, device=None):
    """Unified EulerPC-IRC entry point (single-structure backward compatible).

    - Single ASE ``Atoms`` with an attached ASE calculator -> the original
      single-structure ``EulerPC`` oracle (unchanged behaviour / units / outputs).
    - list / tuple of ``Atoms`` -> the batched ``EulerPCBatch``, consuming the
      supplied batch calculator ``calc`` (prepare / get_ef_gpu[/get_efh_gpu] API).
    """
    if isinstance(atoms_or_list, (list, tuple)):
        if calc is None:
            raise ValueError("run_eulerpc_irc(batch): a batch calculator `calc` is required")
        return EulerPCBatch(list(atoms_or_list), calc, output=output,
                            params=params, paras=paras, device=device).run()
    eulerpc = EulerPC(atoms_or_list, output=output, params=params, paras=paras)
    return eulerpc.run()
