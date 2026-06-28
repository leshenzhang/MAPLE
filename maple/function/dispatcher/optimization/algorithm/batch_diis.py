# -*- coding: utf-8 -*-
"""
Batched Geometry-DIIS (GDIIS / Pulay) optimizer with fixed padded dimension nmax.

Cross-molecule GPU-batched geometry optimization. Each structure keeps its OWN
DIIS error-vector subspace (error vectors = forces history, the GDIIS convention
of DIIS.py::DIISAccelerator). The per-structure Pulay B-matrix least-squares
solve is BATCHED over the whole batch (torch.linalg.lstsq over (B, m+1, m+1))
with per-structure history-validity masks. Structures exit the batch when
converged (dynamic batch shrinking / eviction), identical to BatchLBFGS.

Algorithm (per structure, IDENTICAL to the single-structure GDIIS in DIIS.py):
  error vector  e_i = forces f_i
  B[i,j]        = <e_i | e_j>                         (+ Tikhonov regularization)
  augmented     [[B, -1],[-1^T, 0]] [c; lambda] = [0; -1]      (sum_i c_i = 1)
  corrected     x_tilde_i = x_i + step_scale * f_i    (step_scale = BB estimate)
  extrapolated  x_new = sum_i c_i * x_tilde_i
  step          = clip(x_new - x_cur, max_step)
For the first (memory < min_vectors) steps a steepest-descent bootstrap step is
taken (step = clip(max_step * f, max_step)); the same SD step is the per-structure
fallback whenever the batched solve yields a non-finite coefficient.

This is a NEW class; the single-structure DIIS (DIIS.py / SDCG.py) is untouched.

Reference:
  Csaszar & Pulay, J. Mol. Struct. (Theochem) 114, 31-34 (1984)
  Farkas & Schlegel, PCCP 4, 11-15 (2002)
"""

from typing import List, Optional
import os
import numpy as np
import torch
from ase import Atoms

# Reuse the BatchLBFGS skeleton helpers verbatim (nmax padding / topology / coord).
from .blbfgs import (
    DTYPE,
    _ptr_from_atoms,
    _symbols_flat,
    _get_coord_gpu,
)


def _x_padded_from_atoms(at_list, nmax, device):
    """Pack per-structure Cartesian coords into a (B, nmax) padded flat buffer
    matching the calculator's (B, nmax_dof) step/force layout."""
    B = len(at_list)
    x = torch.zeros((B, nmax), dtype=DTYPE, device=device)
    for b, at in enumerate(at_list):
        flat = torch.from_numpy(
            np.asarray(at.get_positions(), dtype=np.float64).reshape(-1)
        ).to(device=device, dtype=DTYPE)
        x[b, : flat.numel()] = flat
    return x


class BatchDIIS:
    """
    Batched geometry-DIIS (GDIIS / Pulay) optimizer with dynamic batch shrinking.

    Each structure maintains its own DIIS subspace (position history x_i and
    error/force history e_i). The Pulay coefficient systems are assembled and
    solved as one batched least-squares problem; per-structure validity masks
    keep evicted / degenerate history slots out of each structure's solve.
    """

    def __init__(self,
                 output: str,
                 memory: int = 6,
                 min_vectors: int = 3,
                 regularization: float = 1e-10,
                 maxstep: float = 0.2,
                 maxiter: int = 256,
                 device: str = "cuda",
                 write_traj: bool = False,
                 traj_every: int = 1,
                 verbose: int = 1):

        self.memory = memory
        self.min_vectors = min_vectors
        self.regularization = regularization
        self.maxstep = maxstep
        self.maxiter = maxiter
        self.device = torch.device(device)
        self.write_traj = write_traj
        self.traj_every = traj_every
        self.verbose = verbose

        self.output = os.path.abspath(output)
        self.out_dir = os.path.dirname(self.output) or "."
        os.makedirs(self.out_dir, exist_ok=True)

        self.log_fp = None
        self.xyz_paths = []
        self.frame_counts = []

        self._ptr = None
        self._L_vec = None
        self._L_eff = None
        self._nmax = 0
        self._B = 0
        self._symbols_per_batch = None
        self._arange_n = None
        self._real_mask = None
        self._orig_index = None

        # Per-structure convergence thresholds (rebuilt only on batch shrink).
        self._f_max_th = None
        self._f_rms_th = None
        self._dp_max_th = None
        self._dp_rms_th = None

        # Per-structure DIIS history (FIFO ring, depth <= memory). Each entry is a
        # (B, nmax) tensor; the python list index is the history slot.
        self.x_history: List[torch.Tensor] = []   # position vectors x_i
        self.e_history: List[torch.Tensor] = []    # error vectors  e_i = forces f_i

        # Barzilai-Borwein step-scale state (per structure).
        self._prev_x = None    # (B, nmax)
        self._prev_f = None    # (B, nmax)

        # Padded master coordinates kept in sync with calc.coord (B, nmax).
        self._x_cur = None

    # ===================================================
    # PUBLIC RUN
    # ===================================================
    def run(self, mols) -> None:
        device = self.device
        atoms_list = list(mols.multiatoms)
        calc = mols.calc

        B0 = len(atoms_list)
        if B0 == 0:
            return

        self._orig_index = torch.arange(B0, dtype=torch.long, device=device)

        # Per-structure harvested results (keyed by original batch index).
        self.result_pos = [None] * B0       # (N,3) np.float64 final geometry
        self.result_E = [None] * B0         # float Ha final energy
        self.result_niter = [None] * B0     # int iteration at exit
        self.result_conv = [False] * B0     # bool converged flag

        self._open_log()
        self._w("# Batch GDIIS (geometry-DIIS / Pulay) optimization start\n")
        self._w(f"# Memory: {self.memory}, MinVectors: {self.min_vectors}, "
                f"Reg: {self.regularization:.1e}, Maxstep: {self.maxstep}\n")
        self._w(f"# Maxiter: {self.maxiter}, Device: {device}\n\n")

        self._init_xyz_paths(B0)
        self._symbols_per_batch = _symbols_flat(atoms_list)

        # Single initial forward fixes nmax AND provides E0/F0.
        calc.prepare(atoms_list)
        E0, F0 = calc.get_ef_gpu()
        self._nmax = int(F0.shape[1])
        self._arange_n = torch.arange(self._nmax, device=device)

        self._rebuild_topology(atoms_list)
        self._dump_xyz_all(calc, atoms_list, tag="init")

        # Padded master coords (kept identical to calc.coord throughout).
        self._x_cur = _x_padded_from_atoms(atoms_list, self._nmax, device)
        self._prev_x = None
        self._prev_f = None
        self.x_history = []
        self.e_history = []

        iteration = 0
        E_old = E0.to(dtype=DTYPE)
        F_old = F0.to(dtype=DTYPE)

        while iteration < self.maxiter and len(atoms_list) > 0:
            iteration += 1
            rmask = self._real_mask.to(DTYPE)

            # Error vectors for this point = forces (masked to real DOF).
            f_cart = F_old * rmask

            # Store current (x, e=force) snapshot into each structure's subspace.
            self._store_history(self._x_cur, f_cart)

            # Barzilai-Borwein per-structure step scale (approx H^-1).
            step_scale = self._bb_step_scale(self._x_cur, f_cart)

            # Update BB history for next iteration.
            self._prev_x = self._x_cur.clone()
            self._prev_f = f_cart.clone()

            # GDIIS extrapolated step (SD bootstrap / fallback inside).
            step_cart = self._gdiis_step_batched(f_cart, step_scale)

            # Clip per structure to maxstep.
            step_cart = self._clip_step_batched(step_cart)

            # Take the step (calc + padded master in lock-step).
            calc.backup_coords()
            calc.step_cart_(step_cart)
            self._x_cur = self._x_cur + step_cart

            # The single calculator forward / iteration.
            E_new, F_new = calc.get_ef_gpu()
            E_new = E_new.to(dtype=DTYPE)
            F_new = F_new.to(dtype=DTYPE)

            # Shared batched reductions (force from F_new; displacement from step).
            L_eff = self._L_eff
            max_f = F_new.abs().amax(dim=-1)
            rms_f = torch.sqrt((F_new ** 2).sum(-1) / L_eff)
            max_dp = step_cart.abs().amax(dim=-1)
            rms_dp = torch.sqrt((step_cart ** 2).sum(-1) / L_eff)

            self._log_iteration(iteration, E_new, max_f, rms_f, max_dp, rms_dp)

            if self.write_traj and iteration % self.traj_every == 0:
                self._dump_xyz_all(calc, atoms_list, tag=f"iter={iteration}")

            done = self._check_convergence(
                it=iteration, E=E_new,
                max_f=max_f, rms_f=rms_f, max_dp=max_dp, rms_dp=rms_dp,
            )

            # Dynamic batch shrinking (eviction of converged structures).
            survive_local = (~done).nonzero(as_tuple=False).flatten()
            if survive_local.numel() < len(done):
                self._sync_atoms_from_calc(calc, atoms_list)

                # Harvest the just-converged (evicted) structures.
                done_local = done.nonzero(as_tuple=False).flatten().cpu().tolist()
                E_l = E_new.detach().cpu().tolist()
                for il in done_local:
                    oi = int(self._orig_index[il].item())
                    self.result_pos[oi] = atoms_list[il].positions.copy()
                    self.result_E[oi] = float(E_l[il])
                    self.result_niter[oi] = iteration
                    self.result_conv[oi] = True

                atoms_list = [atoms_list[i] for i in survive_local.cpu().tolist()]
                self._orig_index = self._orig_index[survive_local]

                self._shrink_state(survive_local)

                calc.prepare(atoms_list, fixed_nmax=self._nmax)
                self._rebuild_topology(atoms_list)

                E_old = E_new[survive_local]
                F_old = F_new[survive_local]
            else:
                E_old = E_new
                F_old = F_new

            if len(atoms_list) == 0:
                self._w("\n# All batches converged!\n")
                break
        else:
            self._w("\n# Maximum iterations reached.\n")

        # Harvest any survivors that hit maxiter without converging.
        if len(atoms_list) > 0:
            self._sync_atoms_from_calc(calc, atoms_list)
            E_l = E_old.detach().cpu().tolist()
            for il in range(len(atoms_list)):
                oi = int(self._orig_index[il].item())
                self.result_pos[oi] = atoms_list[il].positions.copy()
                self.result_E[oi] = float(E_l[il])
                self.result_niter[oi] = iteration
                self.result_conv[oi] = False

        self._close_log()

    # ===================================================
    # DIIS HISTORY
    # ===================================================
    def _store_history(self, x_vec: torch.Tensor, e_vec: torch.Tensor):
        """Append current (position, error) snapshot; FIFO drop oldest > memory."""
        self.x_history.append(x_vec.clone())
        self.e_history.append(e_vec.clone())
        if len(self.x_history) > self.memory:
            self.x_history.pop(0)
            self.e_history.pop(0)

    def _shrink_state(self, survive_idx: torch.Tensor):
        """Evict converged structures from history + BB + master coords."""
        for i in range(len(self.x_history)):
            self.x_history[i] = self.x_history[i][survive_idx]
            self.e_history[i] = self.e_history[i][survive_idx]
        if self._prev_x is not None:
            self._prev_x = self._prev_x[survive_idx]
            self._prev_f = self._prev_f[survive_idx]
        self._x_cur = self._x_cur[survive_idx]

    # ===================================================
    # BARZILAI-BORWEIN STEP SCALE
    # ===================================================
    def _bb_step_scale(self, x_cur: torch.Tensor, f_cart: torch.Tensor) -> torch.Tensor:
        """Per-structure BB step scale, clamped to [0.01, 2.0]; maxstep on the
        first iteration or when df.df is degenerate. Mirrors SDCG._estimate_step_scale."""
        B = x_cur.shape[0]
        if self._prev_x is None:
            return torch.full((B,), self.maxstep, dtype=DTYPE, device=self.device)
        dx = x_cur - self._prev_x
        df = f_cart - self._prev_f
        df2 = (df * df).sum(dim=-1)               # (B,)
        dxdf = (dx * df).sum(dim=-1)              # (B,)
        alpha = dxdf / df2.clamp(min=1e-300)
        alpha = alpha.abs().clamp(min=0.01, max=2.0)
        return torch.where(df2 < 1e-20,
                           torch.full_like(alpha, self.maxstep), alpha)

    # ===================================================
    # BATCHED GDIIS UPDATE (Pulay)
    # ===================================================
    def _gdiis_step_batched(self, f_cart: torch.Tensor,
                            step_scale: torch.Tensor) -> torch.Tensor:
        """
        Batched GDIIS extrapolation. Builds the per-structure Pulay B-matrix from
        the shared force-history subspace and solves the augmented Lagrange system
        for ALL structures at once (one batched torch.linalg.lstsq), with a
        per-structure history-validity mask. Returns the per-structure proposed
        step (x_new - x_cur); the steepest-descent bootstrap / fallback is applied
        for structures that have too few vectors or whose solve is non-finite.
        """
        device = self.device
        B, nmax = f_cart.shape
        m = len(self.e_history)

        # Steepest-descent step (bootstrap < min_vectors AND non-finite fallback).
        sd_step = self.maxstep * f_cart

        if m < self.min_vectors:
            return sd_step

        # History subspace -> (B, m, nmax).
        E_stack = torch.stack(self.e_history, dim=1)   # error/force vectors
        X_stack = torch.stack(self.x_history, dim=1)   # position vectors

        # Per-structure validity mask of history slots. All currently-stored slots
        # are valid; a slot with (near) zero error norm is masked out (degenerate),
        # which generalizes the solve to per-structure subspaces of unequal depth.
        e_norm2 = (E_stack ** 2).sum(dim=-1)           # (B, m)
        valid = e_norm2 > 1e-30                        # (B, m) bool

        # Pulay B-matrix: B[b,i,j] = <e_i | e_j> (real-DOF dot, padding is zero).
        Bmat = torch.einsum("bik,bjk->bij", E_stack, E_stack)   # (B, m, m)

        # Tikhonov regularization: reg_b = reg * max_i |B[b,i,i]| over valid slots.
        diag = torch.diagonal(Bmat, dim1=1, dim2=2)             # (B, m)
        diag_valid = torch.where(valid, diag.abs(), torch.zeros_like(diag))
        reg_b = self.regularization * diag_valid.amax(dim=-1)   # (B,)
        eye_m = torch.eye(m, dtype=DTYPE, device=device).unsqueeze(0)
        Bmat = Bmat + reg_b[:, None, None] * eye_m

        # Mask invalid slots: force c_i = 0 and remove them from the constraint.
        invalid = ~valid                                        # (B, m)
        # zero the rows/cols of invalid slots, then put 1.0 on their diagonal
        keep = valid.to(DTYPE)                                  # (B, m)
        Bmat = Bmat * keep[:, :, None] * keep[:, None, :]
        Bmat = Bmat + torch.diag_embed(invalid.to(DTYPE))       # identity on invalid

        # Assemble augmented system A (B, m+1, m+1), b (B, m+1).
        A = torch.zeros((B, m + 1, m + 1), dtype=DTYPE, device=device)
        A[:, :m, :m] = Bmat
        cons = -keep                                            # -1 on valid, 0 on invalid
        A[:, m, :m] = cons
        A[:, :m, m] = cons
        rhs = torch.zeros((B, m + 1, 1), dtype=DTYPE, device=device)
        rhs[:, m, 0] = -1.0

        # One batched least-squares solve for the whole batch.
        sol = torch.linalg.lstsq(A, rhs).solution.squeeze(-1)   # (B, m+1)
        coeffs = sol[:, :m]                                     # (B, m)
        coeffs = torch.where(valid, coeffs, torch.zeros_like(coeffs))

        # GDIIS extrapolation: x_new = sum_i c_i (x_i + step_scale * e_i).
        x_tilde = X_stack + step_scale[:, None, None] * E_stack     # (B, m, nmax)
        x_new = (coeffs[:, :, None] * x_tilde).sum(dim=1)           # (B, nmax)
        diis_step = x_new - self._x_cur

        # Per-structure fallback to SD when the solve is non-finite / unusable.
        good = torch.isfinite(diis_step).all(dim=-1) & torch.isfinite(coeffs).all(dim=-1)
        use_diis = good.view(-1, 1)
        return torch.where(use_diis, diis_step, sd_step)

    # ===================================================
    # STEP CLIPPING
    # ===================================================
    def _clip_step_batched(self, step: torch.Tensor) -> torch.Tensor:
        max_disp = step.abs().amax(dim=-1)
        scale = torch.clamp(self.maxstep / (max_disp + 1e-20), max=1.0)
        return step * scale.unsqueeze(-1)

    # ===================================================
    # CONVERGENCE CHECK
    # ===================================================
    def _check_convergence(self, it, E, max_f, rms_f, max_dp, rms_dp):
        done = (
            (max_f <= self._f_max_th)
            & (rms_f <= self._f_rms_th)
            & (max_dp <= self._dp_max_th)
            & (rms_dp <= self._dp_rms_th)
        )
        self._w(self._fmt_convergence_table(
            it=it, E=E, max_f=max_f, rms_f=rms_f,
            max_dp=max_dp, rms_dp=rms_dp, done=done))
        return done

    # ===================================================
    # TOPOLOGY
    # ===================================================
    def _rebuild_topology(self, atoms_list):
        device = self.device
        B = len(atoms_list)
        self._B = B

        self._ptr = _ptr_from_atoms(atoms_list, device)
        L_list = [3 * len(at) for at in atoms_list]
        self._L_vec = torch.tensor(L_list, dtype=torch.int64, device=device)
        self._L_eff = self._L_vec.clamp(min=1).to(DTYPE)
        self._real_mask = (self._arange_n[None, :] < self._L_vec[:, None])

        self._f_max_th = torch.tensor(
            [getattr(at, "f_max_th", 2e-3) for at in atoms_list],
            dtype=DTYPE, device=device)
        self._f_rms_th = torch.tensor(
            [getattr(at, "f_rms_th", 1e-3) for at in atoms_list],
            dtype=DTYPE, device=device)
        self._dp_max_th = torch.tensor(
            [getattr(at, "dp_max_th", 1e-3) for at in atoms_list],
            dtype=DTYPE, device=device)
        self._dp_rms_th = torch.tensor(
            [getattr(at, "dp_rms_th", 5e-4) for at in atoms_list],
            dtype=DTYPE, device=device)

    def _sync_atoms_from_calc(self, calc, atoms_list):
        with torch.no_grad():
            pos = _get_coord_gpu(calc).detach().cpu().numpy()
        ptr = self._ptr.detach().cpu().numpy()
        for i, at in enumerate(atoms_list):
            s, t = ptr[i], ptr[i + 1]
            at.positions[:] = pos[s:t]

    # ===================================================
    # LOGGING
    # ===================================================
    def _open_log(self):
        self.log_fp = open(self.output, "w", encoding="utf-8")

    def _close_log(self):
        if self.log_fp:
            self.log_fp.close()
            self.log_fp = None

    def _w(self, s):
        if self.log_fp:
            self.log_fp.write(s)
            self.log_fp.flush()

    def _log_iteration(self, it, E, max_f, rms_f, max_dp, rms_dp):
        if self.verbose == 0:
            return
        means = torch.stack([
            E.mean(), max_f.mean(), rms_f.mean(), max_dp.mean(), rms_dp.mean()
        ]).tolist()
        e_m, maxf_m, rmsf_m, maxdp_m, rmsdp_m = means
        msg = (f"\nIter {it}: E_mean={e_m:.6f} max|F|_mean={maxf_m:.6f} "
               f"rms|F|_mean={rmsf_m:.6f} max|dX|_mean={maxdp_m:.6f} "
               f"rms|dX|_mean={rmsdp_m:.6f}\n")
        self._w(msg)

    def _fmt_convergence_table(self, it, E, max_f, rms_f, max_dp, rms_dp, done):
        B = E.shape[0]
        E_l, maxf_l, rmsf_l, maxdp_l, rmsdp_l = torch.stack(
            [E, max_f, rms_f, max_dp, rms_dp], dim=0).tolist()
        done_l = done.tolist()
        orig_l = self._orig_index.tolist()

        lines = []
        lines.append("-" * 70 + "\n")
        lines.append(f"{('Batch GDIIS Iteration ' + str(it)).center(70)}\n")
        lines.append("-" * 70 + "\n")
        lines.append(
            "Batch   Energy(Ha)   Max|F|(H/A)  RMS|F|  Max|dX|(A)  RMS|dX|  Conv\n")
        lines.append("-" * 70 + "\n")
        for b in range(B):
            lines.append(
                f"[{orig_l[b]:2d}] "
                f"{E_l[b]:14.6f} {maxf_l[b]:12.6f} {rmsf_l[b]:10.6f} "
                f"{maxdp_l[b]:12.6f} {rmsdp_l[b]:10.6f} "
                f"{('YES' if done_l[b] else 'NO')}\n")
        lines.append("\n")
        return "".join(lines)

    # ===================================================
    # XYZ OUTPUT
    # ===================================================
    def _init_xyz_paths(self, B_all):
        self.xyz_paths = [
            os.path.join(self.out_dir, f"diis_batch{i + 1}.xyz")
            for i in range(B_all)
        ]
        self.frame_counts = [0 for _ in range(B_all)]
        for p in self.xyz_paths:
            open(p, "w").close()

    def _dump_xyz_all(self, calc, atoms_list, tag="init"):
        with torch.no_grad():
            pos = _get_coord_gpu(calc).detach().cpu().numpy()
        ptr = self._ptr.detach().cpu().numpy()
        for i_local, at in enumerate(atoms_list):
            s, t = ptr[i_local], ptr[i_local + 1]
            idx_orig = int(self._orig_index[i_local].item())
            symbols = self._symbols_per_batch[idx_orig]
            self._append_xyz(idx_orig, symbols, pos[s:t], tag)

    def _append_xyz(self, idx_orig, symbols, pos_np, comment=""):
        path = self.xyz_paths[idx_orig]
        n = pos_np.shape[0]
        with open(path, "a") as f:
            f.write(f"{n}\n")
            f.write(f"{comment}\n")
            for k in range(n):
                x, y, z = pos_np[k]
                f.write(f"{symbols[k]:<2s} {x:20.10f} {y:20.10f} {z:20.10f}\n")
        self.frame_counts[idx_orig] += 1


# ============================================================================
# SINGLE-STRUCTURE GDIIS REFERENCE (parity oracle, reuses DIIS.py unmodified)
# ============================================================================
def single_diis_run(atoms: Atoms, calc, output: str,
                    memory: int = 6, min_vectors: int = 3,
                    regularization: float = 1e-10,
                    maxstep: float = 0.2, maxiter: int = 256):
    """
    Single-structure GDIIS optimization driving the UNMODIFIED repo
    ``DIISAccelerator`` (DIIS.py). The driver loop (BB step-scale, SD bootstrap,
    clip, convergence) is bit-for-bit the same protocol BatchDIIS batches, so this
    is the serial parity oracle for a single structure. ``calc`` must be a B=1
    UMABatchCalc so the potential-energy surface is identical to the batched run.

    Returns dict(positions=(N,3) np.float64, energy=float Ha, niter=int, converged=bool).
    """
    from .DIIS import DIISAccelerator, DIISParams

    acc = DIISAccelerator(DIISParams(memory=memory, min_vectors=min_vectors,
                                     regularization=regularization))

    f_max_th = getattr(atoms, "f_max_th", 2e-3)
    f_rms_th = getattr(atoms, "f_rms_th", 1e-3)
    dp_max_th = getattr(atoms, "dp_max_th", 1e-3)
    dp_rms_th = getattr(atoms, "dp_rms_th", 5e-4)

    def _clip(step):
        md = float(np.abs(step).max())
        if md > maxstep:
            step = step * (maxstep / md)
        return step

    calc.prepare([atoms])
    n = len(atoms)
    x = np.asarray(atoms.get_positions(), dtype=np.float64).copy()   # (N,3)

    def ef_at(xcoord):
        # set calc coords to xcoord and evaluate (B=1)
        coord = torch.from_numpy(xcoord).to(device=calc.device, dtype=calc.dtype)
        calc.set_coords_(coord)
        E, F = calc.get_ef_gpu()                  # E (1,), F (1, nmax)
        E = float(E[0].item())
        F = F[0, : 3 * n].detach().cpu().numpy().astype(np.float64).reshape(n, 3)
        return E, F

    E, F = ef_at(x)
    prev_x = None
    prev_f = None
    converged = False
    it = 0
    while it < maxiter:
        it += 1
        # store snapshot (positions, error=forces)
        acc.store(x, F)

        # BB step scale
        if prev_x is None:
            ss = maxstep
        else:
            dx = (x - prev_x).ravel()
            df = (F - prev_f).ravel()
            d2 = float(df @ df)
            if d2 < 1e-20:
                ss = maxstep
            else:
                ss = min(max(abs(float(dx @ df) / d2), 0.01), 2.0)
        prev_x = x.copy()
        prev_f = F.copy()

        # GDIIS step (else SD bootstrap / fallback)
        step = None
        if acc.can_extrapolate():
            res = acc.extrapolate(step_scale=ss)
            if res is not None:
                x_new, _ = res
                step = x_new - x
        if step is None:
            step = maxstep * F
        step = _clip(step)

        x = x + step
        E, F = ef_at(x)

        max_f = float(np.abs(F).max())
        rms_f = float(np.sqrt((F ** 2).sum() / F.size))
        max_dp = float(np.abs(step).max())
        rms_dp = float(np.sqrt((step ** 2).sum() / step.size))
        if (max_f <= f_max_th and rms_f <= f_rms_th
                and max_dp <= dp_max_th and rms_dp <= dp_rms_th):
            converged = True
            break

    if output:
        with open(output, "w") as fh:
            fh.write(f"# single GDIIS niter={it} converged={converged} E={E:.10f}\n")
    return dict(positions=x, energy=E, niter=it, converged=converged)
