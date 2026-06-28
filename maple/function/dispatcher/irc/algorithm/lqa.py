# -*- coding: utf-8 -*-
"""
Intrinsic Reaction Coordinate (IRC) integrator using Local Quadratic Approximation (LQA):

- Mass-weighted coordinates and Hessian
- One LQA propagation step per macro step
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


# =============================== Parameters ===============================
@dataclass
class LQAParams:
    # Which negative eigenmode (1 = most negative) to use for initial direction
    target_mode: int = 1

    # LQA step length in Bohr
    step_length_bohr: float = 0.10

    # Number of macro steps per direction
    max_steps: int = 50

    # Euler integration sub-steps to estimate the LQA propagation parameter
    euler_n: int = 5000

    # Recalculate Hessian every N micro-steps (None = never)
    hessian_recalc: Optional[int] = None

    # Hessian update method: "bfgs" or "bofill"
    hessian_update: str = "bofill"

    # Convergence on forces in Cartesian space (Eh/Å)
    f_max_th: float = 2e-3
    f_rms_th: float = 5e-4

    # Output controls
    print_each: bool = True
    write_traj: bool = True


# ================================== LQA ===================================
class LQA:
    """
    Local quadratic approximation IRC integrator:

    - Works in mass-weighted coordinates.
    - Each macro step uses a local quadratic approximation.
    - Uses BFGS/Bofill updates of the mass-weighted Hessian, with optional periodic
      full recalculation.
    - Integrates forward and backward from the TS along the lowest
      negative eigenmode of H_mw, then merges paths.
    """

    def __init__(self, atoms: Atoms, output: str,
                 params: Optional[LQAParams] = None,
                 paras: Optional[dict] = None):
        self.atoms = atoms
        self.output = output
        self.p = params if params is not None else LQAParams()

        # Parse optional dict overrides, supporting both {"lqa": {...}}
        # and flat dict style. Keep backward compatibility aliases.
        if isinstance(paras, dict):
            low = {k.lower(): v for k, v in paras.items()}
            sub = None
            for key in ("lqa", "irc"):
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
                "hessian_update": "hessian_update",
                "euler_n": "euler_n",
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

        # Internal state for LQA integration
        self._D: Optional[np.ndarray] = None  # mass-weight scaling vector
        self._step_len_mw: float = float(self.p.step_length_bohr * BOHR_TO_ANG)

        self.mw_coords: Optional[np.ndarray] = None
        self.mw_hessian: Optional[np.ndarray] = None
        self.prev_coords: Optional[np.ndarray] = None
        self.prev_grad: Optional[np.ndarray] = None
        self.micro_counter: int = 0

    # ------------------------------ Public API ------------------------------
    def run(self) -> Dict[str, any]:
        """
        Compute forward and backward LQA IRC paths and produce a merged summary.

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

        # Diagonalize mass-weighted Hessian at TS to get negative mode
        H_cart_ts = self._get_hessian_cart()
        H_mw_ts = (self._D[:, None] * H_cart_ts) * self._D[None, :]
        w, V = np.linalg.eigh(H_mw_ts)

        neg_idx = np.where(w < 0.0)[0]
        if len(neg_idx) == 0:
            log_error(
                ["[ERROR] LQA-IRC: No negative eigenvalues found — "
                 "starting geometry is not a saddle point.\n"],
                self.output,
            )
            raise RuntimeError("LQA-IRC: no negative eigenvalues at TS.")

        if len(neg_idx) < self.p.target_mode:
            log_error(
                [f"[ERROR] LQA-IRC: Requested mode {self.p.target_mode}, "
                 f"but only {len(neg_idx)} negative modes found.\n"],
                self.output,
            )
            raise RuntimeError("LQA-IRC: requested negative mode does not exist.")

        sorted_neg = neg_idx[np.argsort(w[neg_idx])]  # most negative first
        idx = sorted_neg[self.p.target_mode - 1]
        eigval = w[idx]
        v_neg_mw = V[:, idx]

        log_info(
            [
                "\n[INFO] LQA-IRC: Selected negative eigenmode "
                f"#{self.p.target_mode} with λ = {eigval:.6e} (MW basis)\n"
            ],
            self.output,
        )

        # Reference TS energy
        E_ts = float(self.atoms.get_potential_energy(force_consistent=True))

        # Store original TS Cartesian positions, reused for both directions
        R_ts_cart = self.atoms.get_positions().copy().reshape(-1)

        # Forward and backward LQA-IRC
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

    # --------------------------- Micro-step (LQA) ----------------------------
    def _micro_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one LQA propagation step in mass-weighted coordinates.

        Updates:
            self.mw_coords, self.mw_hessian, self.prev_coords,
            self.prev_grad, self.micro_counter

        Returns:
            dx: MW displacement for this step
            g_curr: current MW gradient (for diagnostics)
        """
        # Current gradient at MW coordinates
        _, F_cart = self._energy_forces_from_mw(self.mw_coords)
        g_curr = self._gradient_mw_from_forces(F_cart)
        g_norm = _norm(g_curr)
        if g_norm < 1e-12:
            return np.zeros_like(g_curr), g_curr

        coords_curr = self.mw_coords.copy()

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
            coords_diff = coords_curr - self.prev_coords
            if str(self.p.hessian_update).lower() == "bofill":
                self.mw_hessian = self._bofill_update(self.mw_hessian, coords_diff, gradient_diff)
            else:
                self.mw_hessian = self._bfgs_update(self.mw_hessian, coords_diff, gradient_diff)

        # Store for next update
        self.prev_coords = coords_curr
        self.prev_grad = g_curr

        # LQA propagation (Eigen-decomposition of MW Hessian)
        eigvals, eigvecs = np.linalg.eigh(self.mw_hessian)
        mask = np.abs(eigvals) > 1e-8
        if not np.any(mask):
            return np.zeros_like(g_curr), g_curr

        eigvals = eigvals[mask]
        eigvecs = eigvecs[:, mask]

        g_star = eigvecs.T.dot(g_curr)

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
        dx = eigvecs.dot(alphas * g_star)

        self.mw_coords = self.mw_coords + dx
        self.micro_counter += 1

        return dx, g_curr

    # --------------------------- One direction ------------------------------
    def _one_side(self,
                  forward: bool,
                  sign: float,
                  q_ts_cart: np.ndarray,
                  v_neg_mw: np.ndarray,
                  E_ts: float) -> Dict[str, any]:
        """
        Single-sided LQA IRC integration.

        Args:
            forward: True for forward path, False for backward.
            sign: +1.0 or -1.0 to choose direction along the negative mode.
            q_ts_cart: TS Cartesian coordinates (flattened, Å).
            v_neg_mw: selected negative eigenmode in MW basis (3N, normalized later).
            E_ts: TS reference energy (Eh).
        """
        p = self.p
        title = "FORWARD LQA-IRC" if forward else "BACKWARD LQA-IRC"

        self._print_header(title)
        self._print_conv_thresholds(p.f_max_th, p.f_rms_th)

        # Initial MW coordinates at TS
        q_ts_mw = self._mw_from_cart(q_ts_cart)

        # Initial displacement along negative mode in MW
        v_dir = _unit(v_neg_mw) * sign
        q0_mw = q_ts_mw + self._scale_mw_step(v_dir, 0.5 * self._step_len_mw)

        # Initial energy / forces / gradient at starting point
        E0, F0_cart = self._energy_forces_from_mw(q0_mw)
        g0_mw = self._gradient_mw_from_forces(F0_cart)

        maxF0 = float(np.max(np.abs(F0_cart)))
        rmsF0 = float(np.sqrt(np.mean(F0_cart ** 2)))

        # Initial Hessian at starting point (MW)
        H0_cart = self._get_hessian_cart()
        H0_mw = (self._D[:, None] * H0_cart) * self._D[None, :]

        # Initialize LQA state
        self.mw_coords = q0_mw.copy()
        self.mw_hessian = H0_mw.copy()
        self.prev_coords = None
        self.prev_grad = None
        self.micro_counter = 0

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
            log_error(["[ERROR] LQA-IRC: One path side is empty.\n"], self.output)
            raise RuntimeError("LQA-IRC: one path side is empty.")

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
                "                       LQA-IRC PATH SUMMARY           \n",
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
                f"\n[INFO] LQA-IRC forward trajectory written to: {fwd_path}\n",
                f"[INFO] LQA-IRC backward trajectory written to: {bwd_path}\n",
                f"[INFO] LQA-IRC full trajectory written to: {full_path}\n",
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
                "                      ***          THE LQA-IRC HAS CONVERGED        ***\n",
                "                      *************************************************\n\n",
            ],
            self.output,
        )


# ======================================================================== #
#                            BATCHED LQA-IRC                                #
# ======================================================================== #
# Batched IRC over B transition-state structures sharing ONE UMA *batch*
# calculator (prepare / get_efh_gpu / step_cart_ / set_coords_). All B paths
# are propagated in lockstep: ONE batched get_efh_gpu() per macro step drives
# every structure's LQA propagation. Pure local-quadratic (fresh mass-weighted
# Hessian each step, no BFGS/Bofill history) -- the natural batch form of LQA.
#
# Padding layout matches the calculator's (B, nmax_dof) buffer: structure i
# fills DOF block [0, 3*n_i); the remainder is zero. Padding DOFs carry zero
# force and a zero Hessian block, so their eigenvalues are exactly 0 and are
# dropped by the same |lambda|>1e-8 mode mask used in the single-structure path.
#
# Units identical to the single-structure LQA: positions Angstrom, forces
# Eh/Angstrom, Hessian Eh/Angstrom^2, energy Eh, MW coords sqrt(amu)*Angstrom
# with q_mw = q_cart / D, H_mw = D (x) D * H, g_mw = D * g_cart  (D = 1/sqrt(m)).
import torch as _torch


def _apply_paras(p: "LQAParams", paras):
    """Apply a {'lqa'|'irc': {...}} or flat dict of overrides onto an LQAParams.

    Same alias table as LQA.__init__ (kept separate so the batch path can reuse
    it without touching the single-structure constructor).
    """
    if not isinstance(paras, dict):
        return p
    low = {k.lower(): v for k, v in paras.items()}
    sub = None
    for key in ("lqa", "irc"):
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
        "hessian_update": "hessian_update",
        "euler_n": "euler_n",
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


class LQABatch:
    """Batched LQA-IRC integrator over B transition-state structures.

    Consumes the UMA *batch* calculator API (NOT a per-atoms ASE calculator):
        calc.prepare(atoms_list, fixed_nmax)
        calc.get_efh_gpu() -> (E (B,), F (B,M), H (B,M,M), P (B,))   [Hartree]
        calc.step_cart_(s (B,M))         # in-place padded Cartesian displacement
        calc.set_coords_(coord (N,3))    # reset packed Cartesian coords

    Parameters mirror LQAParams (same defaults / overrides via `paras`). The
    algorithm is the pure local-quadratic LQA step recomputed from the fresh
    mass-weighted Hessian at every macro step; this is the batch-natural form
    (the single-structure LQA's optional BFGS/Bofill Hessian *update* needs a
    per-structure history that does not vectorize cleanly -- see module docstring
    "no DWI interp/history").
    """

    def __init__(self, atoms_list, calc, output: str = "lqa_batch.out",
                 params=None, paras=None, device=None):
        self.atoms_list = list(atoms_list)
        self.calc = calc
        self.output = output
        self.p = params if params is not None else LQAParams()
        _apply_paras(self.p, paras)

        self.B = len(self.atoms_list)
        if self.B == 0:
            raise ValueError("LQABatch: empty atoms_list")

        self.device = _torch.device(
            device if device is not None
            else ("cuda" if _torch.cuda.is_available() else "cpu")
        )
        self._step_len_mw = float(self.p.step_length_bohr * BOHR_TO_ANG)
        self._build_padding()

        # --- OPTIONAL FIX #4: amortized Hessian (method-consistent, NOT byte-id) -
        # Default p.hessian_recalc=None -> exact get_efh_gpu() EVERY macro step =
        # the current behaviour (byte-identical). When set to K>0 the batched LQA
        # mirrors the single-structure LQA: exact Hessian only every K steps,
        # Bofill/BFGS-updated (cheap get_ef_gpu) in between -- amortizing the
        # expensive FD Hessian (3N+ forwards) over K steps. OPT-IN.
        self._efh_calls = 0       # diagnostics: # exact get_efh_gpu Hessian forwards
        self._ef_calls = 0        # diagnostics: # cheap get_ef_gpu forwards (amortized steps)

    # ----------------------------- setup ----------------------------------
    def _build_padding(self):
        """Build padded mass-weight vector D, DOF mask, DOF counts, TS coords."""
        B = self.B
        dev = self.device
        nmax_a = max(len(a) for a in self.atoms_list)
        M = 3 * nmax_a
        self.nmax_dof = M

        Dpad = _torch.ones((B, M), dtype=_torch.float64, device=dev)
        dof_mask = _torch.zeros((B, M), dtype=_torch.bool, device=dev)
        n_dof = _torch.zeros(B, dtype=_torch.float64, device=dev)
        qcart_ts = _torch.zeros((B, M), dtype=_torch.float64, device=dev)
        coord_rows = []

        for i, a in enumerate(self.atoms_list):
            m = np.asarray(a.get_masses(), dtype=np.float64)
            m = np.where(m > 0.0, m, 1.0)
            d = 1.0 / np.sqrt(np.repeat(m, 3))          # (3 n_i,)
            ni3 = d.shape[0]
            Dpad[i, :ni3] = _torch.tensor(d, dtype=_torch.float64, device=dev)
            dof_mask[i, :ni3] = True
            n_dof[i] = float(ni3)
            pos = np.asarray(a.get_positions(), dtype=np.float64)
            qcart_ts[i, :ni3] = _torch.tensor(pos.reshape(-1), dtype=_torch.float64, device=dev)
            coord_rows.append(_torch.tensor(pos, dtype=_torch.float64, device=dev))

        self.Dpad = Dpad
        self.dof_mask = dof_mask
        self.n_dof = n_dof                               # (B,) = 3 n_i
        self._qcart_ts = qcart_ts                        # (B, M) padded TS Cartesian
        self._q_ts_mw = qcart_ts / Dpad                  # (B, M) padded TS MW coords
        self._ts_coord_N3 = _torch.cat(coord_rows, dim=0)  # (N, 3) packed, calc atom order

    # --------------------------- small helpers ----------------------------
    @staticmethod
    def _unit_rows(v):
        """Row-wise L2 normalize (B, M); zero rows stay zero."""
        n = v.norm(dim=1, keepdim=True)
        return v / n.clamp_min(1e-16)

    def _scale_mw_step(self, direction_mw, step_umw):
        """Scale each MW direction row to a target *unweighted* step length."""
        denom = (direction_mw * self.Dpad).norm(dim=1)        # (B,)
        scale = step_umw / denom.clamp_min(1e-16)
        return direction_mw * scale[:, None]

    def _force_metrics(self, F):
        """Per-structure max|F| and rms(F) over REAL DOFs only. F: (B, M) Eh/A."""
        maxG = F.abs().amax(dim=1)                            # padding is 0 -> ok
        sumsq = (F * F).sum(dim=1)                            # padding 0
        rmsG = _torch.sqrt(sumsq / self.n_dof.clamp_min(1.0))
        return maxG, rmsG

    # ------------------------- TS mode selection --------------------------
    def _ts_modes(self):
        """One batched get_efh_gpu() at TS -> batched eigh -> per-structure mode.

        Returns (E_ts (B,), vneg (B,M) phase-canonical, eigval (B,), valid (B,),
                 n_strong_neg (B,)).
        """
        self.calc.set_coords_(self._ts_coord_N3)
        E_ts, F_ts, H_ts, _ = self.calc.get_efh_gpu()
        self._efh_calls += 1
        Hmw = self.Dpad[:, :, None] * H_ts * self.Dpad[:, None, :]
        Hmw = 0.5 * (Hmw + Hmw.transpose(1, 2))
        w, V = _torch.linalg.eigh(Hmw)                       # ascending eigvals
        k = int(self.p.target_mode)
        idx = k - 1
        eigval = w[:, idx]                                   # selected mode eigval
        vneg = V[:, :, idx].clone()                          # (B, M)

        # how many *strong* negative modes (clean first-order saddle == k)
        n_strong_neg = (w < -1e-4).sum(dim=1)

        # success: the selected mode is genuinely negative and exists
        n_neg = (w < 0.0).sum(dim=1)
        valid = (eigval < -1e-6) & (n_neg >= k)

        # deterministic phase: make the largest-magnitude component positive
        am = vneg.abs().argmax(dim=1)
        rows = _torch.arange(self.B, device=self.device)
        sgn = _torch.sign(vneg[rows, am])
        sgn = _torch.where(sgn == 0, _torch.ones_like(sgn), sgn)
        vneg = vneg * sgn[:, None]

        return E_ts, vneg, eigval, valid, n_strong_neg

    # --------------------------- batched LQA step -------------------------
    def _lqa_step(self, F, H, active):
        """Vectorized LQA propagation step. F (B,M), H (B,M,M); returns dx (B,M)."""
        B, M = self.B, self.nmax_dof
        dev = self.device
        Dpad = self.Dpad
        step = self._step_len_mw
        euler_n = int(self.p.euler_n)

        g = Dpad * (-F)                                      # MW gradient (B, M)
        Hmw = Dpad[:, :, None] * H * Dpad[:, None, :]
        Hmw = 0.5 * (Hmw + Hmw.transpose(1, 2))
        w, V = _torch.linalg.eigh(Hmw)                       # (B,M),(B,M,M)

        gstar = _torch.einsum('bmk,bm->bk', V, g)            # V^T g  (B, M)
        modemask = w.abs() > 1e-8                            # drop padding/near-zero
        gstar = gstar * modemask

        gnorm = g.norm(dim=1)                                # (B,)
        dt = step / (float(euler_n) * gnorm.clamp_min(1e-30))  # (B,)

        # per-structure Euler arc-length integration to find propagation time t
        t = dt.clone()
        cur = _torch.zeros(B, dtype=_torch.float64, device=dev)
        done = (~active) | (gnorm < 1e-12)
        for _ in range(euler_n):
            expo = _torch.exp(-2.0 * w * t[:, None])         # (B, M)
            dsdt = _torch.sqrt((gstar * gstar * expo).sum(dim=1).clamp_min(0.0))
            cur = _torch.where(done, cur, cur + dsdt * dt)
            reach = cur >= step
            done = done | reach
            if bool(done.all()):
                break
            t = _torch.where(done, t, t + dt)

        w_safe = _torch.where(modemask, w, _torch.ones_like(w))
        alphas = ((_torch.exp(-w * t[:, None]) - 1.0) / w_safe) * modemask
        dx = _torch.einsum('bmk,bk->bm', V, alphas * gstar)  # (B, M)

        act = active & (gnorm >= 1e-12)
        dx = dx * act[:, None].to(_torch.float64)
        return dx

    # ---------------- batched Bofill Hessian update (FIX #4) -----------------
    @staticmethod
    def _bofill_update_batched(H, s_cart, g_prev, g_new, real_mask,
                               step_accepted=None,
                               step_tol: float = 1e-8, grad_tol: float = 1e-8,
                               sr1_tol: float = 1e-8):
        """Batched Bofill (MS/SR1 + PSB) update of the CARTESIAN Hessian.

        Ported verbatim from BPRFO._bofill_update_batched (same Gaussian-style
        residual Z = dg - H.dq, phi mixing, masked to real DOFs, per-structure
        gating on step_accepted). Used by the FIX #4 amortized-Hessian LQA path to
        carry curvature between exact get_efh_gpu() recalcs (method-consistent with
        the single-structure LQA's BFGS/Bofill micro-step Hessian update)."""
        DTYPE = H.dtype
        rm = real_mask.to(DTYPE)
        mask_ij = (real_mask.unsqueeze(-1) & real_mask.unsqueeze(-2)).to(DTYPE)

        dq = s_cart.to(DTYPE) * rm
        dg = (g_new - g_prev).to(DTYPE) * rm
        dq2 = (dq * dq).sum(-1)
        dg2 = (dg * dg).sum(-1)
        upd_mask = (dq2 > step_tol**2) & (dg2 > grad_tol**2)
        if step_accepted is not None:
            upd_mask = upd_mask & step_accepted
        if not bool(upd_mask.any()):
            return H

        H_new = H.clone()
        idx = upd_mask.nonzero(as_tuple=False).flatten()
        dq_m = dq[idx]; dg_m = dg[idx]; HH = H_new[idx]
        Hdq = _torch.einsum("mij,mj->mi", HH, dq_m)
        Z = dg_m - Hdq
        dq2_m = (dq_m * dq_m).sum(-1)
        zz_m = (Z * Z).sum(-1)
        qz_m = (dq_m * Z).sum(-1)

        Z_dqT = _torch.einsum("mi,mj->mij", Z, dq_m) * mask_ij[idx]
        dq_ZT = _torch.einsum("mi,mj->mij", dq_m, Z) * mask_ij[idx]
        dq_dqT = _torch.einsum("mi,mj->mij", dq_m, dq_m) * mask_ij[idx]
        dH_PSB = (Z_dqT + dq_ZT) / dq2_m.view(-1, 1, 1) \
            - (qz_m / (dq2_m * dq2_m)).view(-1, 1, 1) * dq_dqT

        use_sr1 = (qz_m.abs() > sr1_tol) & (zz_m > sr1_tol**2)
        dH_SR1_full = _torch.zeros_like(dH_PSB)
        if bool(use_sr1.any()):
            Z_ZT = _torch.einsum("mi,mj->mij", Z[use_sr1], Z[use_sr1]) * mask_ij[idx][use_sr1]
            dH_SR1_full[use_sr1] = Z_ZT / qz_m[use_sr1].view(-1, 1, 1)

        phi = _torch.ones_like(qz_m)
        good_phi = (dq2_m > sr1_tol**2) & (zz_m > sr1_tol**2)
        if bool(good_phi.any()):
            ratio = (qz_m[good_phi] * qz_m[good_phi]) / (dq2_m[good_phi] * zz_m[good_phi])
            phi[good_phi] = (1.0 - ratio).clamp(0.0, 1.0)
        phi = _torch.where(use_sr1, phi, _torch.ones_like(phi))

        inc = (1.0 - phi).view(-1, 1, 1) * dH_SR1_full + phi.view(-1, 1, 1) * dH_PSB
        HH = HH + inc
        HH = 0.5 * (HH + HH.transpose(-1, -2))
        H_new[idx] = HH
        return H_new

    # --------------------------- one direction ----------------------------
    def _propagate_side(self, sign, vneg, active_init):
        """Propagate all B structures one side (sign=+1 forward / -1 backward).

        Returns list (len B) of per-structure record dicts {E,maxG,rmsG,x}.
        """
        B, M = self.B, self.nmax_dof
        p = self.p
        Dpad = self.Dpad
        step = self._step_len_mw

        # reset calculator + state to TS, then displace 0.5*step along neg mode
        self.calc.set_coords_(self._ts_coord_N3)
        vdir = self._unit_rows(vneg) * sign
        dq0_mw = self._scale_mw_step(vdir, 0.5 * step)
        # zero the displacement for invalid structures so they sit at TS
        dq0_mw = dq0_mw * active_init[:, None].to(_torch.float64)
        self.calc.step_cart_(dq0_mw * Dpad)
        q_mw = self._q_ts_mw + dq0_mw

        store = [{"E": [], "maxG": [], "rmsG": [], "x": []} for _ in range(B)]
        active = active_init.clone()

        # FIX #4: amortize the Hessian when p.hessian_recalc=K>0 (default None ->
        # exact every step = current behaviour, byte-identical). The displaced
        # start point is ALWAYS an exact Hessian anchor.
        recalc_k = getattr(p, "hessian_recalc", None)
        amortize = (recalc_k is not None and int(recalc_k) > 0)

        # initial eval at the displaced start point (exact Hessian anchor)
        E, F, H, _ = self.calc.get_efh_gpu()
        self._efh_calls += 1
        self._record(store, E, F, q_mw, active)
        maxG, rmsG = self._force_metrics(F)
        conv = (maxG <= p.f_max_th) & (rmsG <= p.f_rms_th)
        active = active & ~conv
        g_prev = -F   # Cartesian gradient, threaded for the Bofill update

        for _it in range(1, p.max_steps + 1):
            if not bool(active.any()):
                break
            dx = self._lqa_step(F, H, active)
            q_mw = q_mw + dx
            s_cart = dx * Dpad
            self.calc.step_cart_(s_cart)
            if (not amortize) or (_it % int(recalc_k) == 0):
                # exact Hessian recalc (anchor): default path takes this EVERY step
                E, F, H, _ = self.calc.get_efh_gpu()
                self._efh_calls += 1
            else:
                # amortized step: cheap E/F forward + Bofill-update the Hessian
                E, F = self.calc.get_ef_gpu()
                self._ef_calls += 1
                g_new = -F
                H = self._bofill_update_batched(
                    H, s_cart, g_prev, g_new, self.dof_mask, step_accepted=active)
            g_prev = -F
            self._record(store, E, F, q_mw, active)
            maxG, rmsG = self._force_metrics(F)
            conv = (maxG <= p.f_max_th) & (rmsG <= p.f_rms_th)
            small = dx.abs().amax(dim=1) <= 1e-12
            active = active & ~conv & ~small

        return store

    def _record(self, store, E, F, q_mw, active):
        cart = q_mw * self.Dpad                               # (B, M) padded Cartesian
        maxG, rmsG = self._force_metrics(F)
        Ecpu = E.detach().cpu()
        mGcpu = maxG.detach().cpu()
        rGcpu = rmsG.detach().cpu()
        cart_cpu = cart.detach().cpu().numpy()
        for i in range(self.B):
            if not bool(active[i]):
                continue
            ni = len(self.atoms_list[i])
            store[i]["E"].append(float(Ecpu[i]))
            store[i]["maxG"].append(float(mGcpu[i]))
            store[i]["rmsG"].append(float(rGcpu[i]))
            store[i]["x"].append(cart_cpu[i, :3 * ni].reshape(ni, 3).copy())

    # ------------------------------- run ----------------------------------
    def run(self):
        """Run batched forward+backward LQA-IRC for all B structures.

        Returns a list (len B) of per-structure result dicts:
            {"index": i, "valid": bool, "neg_eigval": float, "n_strong_neg": int,
             "E_ts": float, "forward": {records}, "backward": {records}}
        Invalid structures (no clean negative TS mode) get valid=False and empty
        forward/backward paths (flagged, not propagated).
        """
        # prepare topology once for the whole batch
        self.calc.prepare(self.atoms_list, fixed_nmax=self.nmax_dof)

        E_ts, vneg, eigval, valid, n_strong = self._ts_modes()

        fwd = self._propagate_side(+1.0, vneg, valid)
        bwd = self._propagate_side(-1.0, vneg, valid)

        E_ts_cpu = E_ts.detach().cpu()
        eig_cpu = eigval.detach().cpu()
        nstr_cpu = n_strong.detach().cpu()
        valid_cpu = valid.detach().cpu()

        results = []
        for i in range(self.B):
            results.append({
                "index": i,
                "valid": bool(valid_cpu[i]),
                "neg_eigval": float(eig_cpu[i]),
                "n_strong_neg": int(nstr_cpu[i]),
                "E_ts": float(E_ts_cpu[i]),
                "forward": {"records": fwd[i]},
                "backward": {"records": bwd[i]},
            })
        return results


def run_lqa_irc(atoms_or_list, output: str = "lqa.out", paras=None,
                calc=None, params=None, device=None):
    """Unified LQA-IRC entry point (single-structure backward compatible).

    - If `atoms_or_list` is a single ASE Atoms with an attached ASE calculator
      (`atoms.calc`), dispatch to the original single-structure `LQA` (unchanged
      behaviour / units / outputs).
    - If `atoms_or_list` is a list/tuple of Atoms, dispatch to the batched
      `LQABatch`, consuming the supplied batch calculator `calc` (UMA batch API).
    """
    if isinstance(atoms_or_list, (list, tuple)):
        if calc is None:
            raise ValueError("run_lqa_irc(batch): a batch calculator `calc` is required")
        return LQABatch(list(atoms_or_list), calc, output=output,
                        params=params, paras=paras, device=device).run()
    lqa = LQA(atoms_or_list, output=output, params=params, paras=paras)
    return lqa.run()
