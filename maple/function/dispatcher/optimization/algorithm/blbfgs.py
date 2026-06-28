# -*- coding: utf-8 -*-
"""
Batch L-BFGS optimizer with fixed padded dimension nmax.
Highly parallelized optimization using GPU tensors.
Individual batches exit when converged (dynamic batch shrinking).
"""

from typing import List, Optional
import os
import numpy as np
import torch
from ase import Atoms

DTYPE = torch.float64


def _ptr_from_atoms(at_list, device):
    ptr = [0]
    for at in at_list:
        ptr.append(ptr[-1] + len(at))
    return torch.tensor(ptr, dtype=torch.long, device=device)


def _masses_flat(at_list, nmax, device):
    B = len(at_list)
    mass = torch.ones((B, nmax), dtype=DTYPE, device=device)
    for b, at in enumerate(at_list):
        m = np.asarray(at.get_masses(), dtype=np.float64)
        mass[b, :3 * len(at)] = torch.from_numpy(np.repeat(m, 3)).to(device=device, dtype=DTYPE)
    return mass


def _symbols_flat(at_list):
    return [at.get_chemical_symbols() for at in at_list]


def _get_coord_gpu(calc) -> torch.Tensor:
    if hasattr(calc, "coord"):
        return calc.coord
    elif hasattr(calc, "coord32"):
        return calc.coord32
    raise AttributeError("calculator has no coord buffer")


class BatchLBFGS:
    """
    Batched L-BFGS optimizer with dynamic batch shrinking.
    Each batch maintains its own L-BFGS history (S, Y, rho).
    """

    def __init__(self,
                 output: str,
                 memory: int = 5,
                 curvature: float = 70.0,
                 maxstep: float = 0.2,
                 maxiter: int = 256,
                 device: str = "cuda",
                 write_traj: bool = False,
                 traj_every: int = 1,
                 verbose: int = 1,
                 precon: Optional[str] = None,
                 precon_exp_A: float = 3.0,
                 precon_exp_rcut_mult: float = 2.0,
                 precon_stabilize: float = 0.1):

        self.memory = memory
        self.curvature = curvature
        self.maxstep = maxstep
        self.maxiter = maxiter
        self.device = torch.device(device)
        self.write_traj = write_traj
        self.traj_every = traj_every
        self.verbose = verbose

        # ---- OPT-IN preconditioner (default None = current oracle, byte-identical).
        # When set ('exp' or 'lindh') the L-BFGS initial inverse-Hessian H_0^{-1} in
        # the two-loop recursion is replaced by a cheap geometry-only preconditioner
        # P^{-1} (P approximates the Hessian), dropping the effective condition number
        # -> far fewer steps on large/floppy minimizations (gain ~sqrt(N)). P is built
        # from geometry ONLY -> ZERO extra MLIP forwards. The L-BFGS (s, y) history is
        # preconditioner-independent, so only H_0 changes: the stationary point (g=0)
        # is IDENTICAL to the oracle; only the path (condition number) changes.
        #   Refs: Packwood, Kermode, Mones, Bernstein, Woolley, Gould, Ortner, Csanyi,
        #   J. Chem. Phys. 144, 164109 (2016), DOI 10.1063/1.4947024 (Exp precon);
        #   Lindh, Bernhardsson, Karlstrom, Malmqvist, Chem. Phys. Lett. 241, 423
        #   (1995), DOI 10.1016/0009-2614(95)00646-L (Lindh model Hessian, reused from
        #   ts/algorithm/initial_hessian.py); ASE PreconLBFGS.
        self._precon_mode = (str(precon).lower() if precon is not None else None)
        if self._precon_mode not in (None, "exp", "lindh"):
            raise ValueError(
                f"precon must be None | 'exp' | 'lindh', got {precon!r}")
        self._precon_exp_A = float(precon_exp_A)
        self._precon_exp_rcut_mult = float(precon_exp_rcut_mult)
        self._precon_stab = float(precon_stabilize)
        # (B, nmax, nmax) batched lower-Cholesky factor of P, or None when disabled.
        self._precon_L = None

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

        # Per-structure convergence thresholds, precomputed once per topology
        # (rebuilt only on batch shrink) instead of every iteration.
        self._f_max_th = None
        self._f_rms_th = None
        self._dp_max_th = None
        self._dp_rms_th = None

        # Per-batch L-BFGS history
        self.S_history = []  # List of (B, nmax) tensors per history step
        self.Y_history = []  # List of (B, nmax) tensors per history step
        self.rho_history = []  # List of (B,) tensors per history step
        self.history_valid = None  # (B, memory) mask indicating valid history slots

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

        self._open_log()
        self._w("# Batch L-BFGS optimization start\n")
        self._w(f"# Memory: {self.memory}, Curvature: {self.curvature}, Maxstep: {self.maxstep}\n")
        self._w(f"# Maxiter: {self.maxiter}, Device: {device}\n\n")

        self._init_xyz_paths(B0)
        self._symbols_per_batch = _symbols_flat(atoms_list)

        # === Single initial forward: fixes nmax AND provides E0/F0 ===
        # (OPT: the original did two get_ef_gpu() forwards on the *same* initial
        #  geometry -- one to read nmax, one for E0/F0. They are bit-identical, so
        #  one forward suffices.)
        calc.prepare(atoms_list)
        E0, F0 = calc.get_ef_gpu()
        self._nmax = int(F0.shape[1])
        self._arange_n = torch.arange(self._nmax, device=device)

        # Build topology (nmax fixed) + precompute per-structure thresholds/masks
        self._rebuild_topology(atoms_list)
        self._dump_xyz_all(calc, atoms_list, tag="init")

        # Initialize history tracking
        B = self._B
        self.history_valid = torch.zeros((B, self.memory), dtype=torch.bool, device=device)

        iteration = 0

        # Get initial energy and forces
        E_old = E0.to(dtype=DTYPE)
        F_old = F0.to(dtype=DTYPE)

        while iteration < self.maxiter and len(atoms_list) > 0:
            iteration += 1
            real_mask = self._real_mask
            rmask = real_mask.to(DTYPE)              # cast once, reused for g_cart & g_new

            # Current gradient (negative force)
            g_cart = -F_old * rmask

            # Compute L-BFGS search direction
            search_dir = self._two_loop_batched(g_cart)

            # Clip step size
            step_cart = self._clip_step_batched(search_dir)

            # Backup coordinates
            calc.backup_coords()

            # Take step
            calc.step_cart_(step_cart)

            # Evaluate new (committed) point -- THE single calculator forward/iter.
            # (OPT: the committed-geometry energy E_new is threaded straight into
            #  the convergence logging below; the original recomputed E via a second
            #  full get_ef_gpu() forward at the identical geometry inside the
            #  convergence check -- a 100% redundant forward+backward every iter.)
            E_new, F_new = calc.get_ef_gpu()
            E_new = E_new.to(dtype=DTYPE)
            F_new = F_new.to(dtype=DTYPE)
            g_new = -F_new * rmask

            # Update L-BFGS history
            s_vec = step_cart
            y_vec = g_new - g_cart
            self._update_history_batched(s_vec, y_vec)

            # Convergence/log metrics: compute the four batched reductions ONCE and
            # share between the iteration log and the convergence check (the original
            # computed each of them twice -- once per helper).
            L_eff = self._L_eff
            max_f = F_new.abs().amax(dim=-1)
            rms_f = torch.sqrt((F_new ** 2).sum(-1) / L_eff)
            max_dp = step_cart.abs().amax(dim=-1)
            rms_dp = torch.sqrt((step_cart ** 2).sum(-1) / L_eff)

            # Log iteration info
            self._log_iteration(iteration, E_new, max_f, rms_f, max_dp, rms_dp)

            # Write trajectory for accepted steps
            if self.write_traj and iteration % self.traj_every == 0:
                self._dump_xyz_all(calc, atoms_list, tag=f"iter={iteration}")

            # Check convergence (uses threaded E_new + shared metrics; no extra fwd)
            done = self._check_convergence(
                it=iteration,
                E=E_new,
                max_f=max_f, rms_f=rms_f,
                max_dp=max_dp, rms_dp=rms_dp,
            )

            # Dynamic batch shrinking
            survive_local = (~done).nonzero(as_tuple=False).flatten()
            if survive_local.numel() < len(done):
                self._sync_atoms_from_calc(calc, atoms_list)

                atoms_list = [atoms_list[i] for i in survive_local.cpu().tolist()]
                self._orig_index = self._orig_index[survive_local]

                # Shrink history
                self._shrink_history(survive_local)

                calc.prepare(atoms_list, fixed_nmax=self._nmax)
                self._rebuild_topology(atoms_list)

                # Update old values
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

        self._close_log()

    # ===================================================
    # L-BFGS TWO-LOOP RECURSION (BATCHED)
    # ===================================================
    def _two_loop_batched(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Batched L-BFGS two-loop recursion.
        
        Args:
            grad: (B, nmax) gradient tensor
            
        Returns:
            search_dir: (B, nmax) search direction tensor
        """
        device = self.device
        B, nmax = grad.shape
        
        q = grad.clone()
        alpha_list = []

        # Backward pass through history (most recent first)
        num_history = len(self.S_history)
        for t in range(num_history - 1, -1, -1):
            s_t = self.S_history[t]  # (B, nmax)
            y_t = self.Y_history[t]  # (B, nmax)
            rho_t = self.rho_history[t]  # (B,)
            valid_t = self.history_valid[:, t]  # (B,)

            # alpha = rho * (s^T q)
            alpha = rho_t * (s_t * q).sum(dim=-1)  # (B,)
            alpha = torch.where(valid_t, alpha, torch.zeros_like(alpha))
            alpha_list.append(alpha)

            # q = q - alpha * y
            q = q - alpha.unsqueeze(-1) * y_t

        # Reverse alpha_list for forward pass
        alpha_list = list(reversed(alpha_list))

        # Initial Hessian approximation H_0^{-1}.
        #
        # Default (no preconditioner): H_0^{-1} = gamma * I with the standard
        # Oren-Luenberger scalar gamma = (y^T s)/(y^T y)  -- ORACLE PATH, unchanged.
        #
        # OPT-IN preconditioner: H_0^{-1} = gamma * P^{-1}, P = geometry-only Hessian
        # model (exp/Lindh) factored once per topology. gamma is the *preconditioned*
        # Oren-Luenberger refinement gamma = (y^T s)/(y^T P^{-1} y) (-> 1 when P
        # matches the true Hessian); the no-history seed is gamma=1 because P is
        # normalized to carry the 1/curvature scale itself.
        precon_on = self._precon_L is not None
        if num_history > 0:
            # gamma = (y^T s) / (y^T y)   [precon: (y^T s)/(y^T P^{-1} y)]
            s_last = self.S_history[-1]
            y_last = self.Y_history[-1]
            valid_last = self.history_valid[:, -1]

            ys = (y_last * s_last).sum(dim=-1)
            if precon_on:
                ypy = (y_last * self._apply_pinv(y_last)).sum(dim=-1)
                gamma = ys / (ypy + 1e-20)
                gamma = torch.where(valid_last, gamma, torch.ones_like(gamma))
            else:
                yy = (y_last * y_last).sum(dim=-1)
                gamma = ys / (yy + 1e-20)
                gamma = torch.where(valid_last, gamma, torch.ones_like(gamma) / self.curvature)
        else:
            if precon_on:
                gamma = torch.ones((B,), dtype=DTYPE, device=device)
            else:
                gamma = torch.full((B,), 1.0 / self.curvature, dtype=DTYPE, device=device)

        if precon_on:
            z = gamma.unsqueeze(-1) * self._apply_pinv(q)
        else:
            z = gamma.unsqueeze(-1) * q

        # Forward pass through history (oldest first)
        for t in range(num_history):
            s_t = self.S_history[t]
            y_t = self.Y_history[t]
            rho_t = self.rho_history[t]
            valid_t = self.history_valid[:, t]
            alpha_t = alpha_list[t]

            # beta = rho * (y^T z)
            beta = rho_t * (y_t * z).sum(dim=-1)
            beta = torch.where(valid_t, beta, torch.zeros_like(beta))

            # z = z + s * (alpha - beta)
            z = z + s_t * (alpha_t - beta).unsqueeze(-1)

        return -z

    # ===================================================
    # PRECONDITIONER (OPT-IN; default path never reaches here)
    # ===================================================
    def _apply_pinv(self, v: torch.Tensor) -> torch.Tensor:
        """Apply P^{-1} to a (B, nmax) batch via the cached Cholesky factor of P.

        P is block-diagonal (a real SPD (3n_i x 3n_i) block + curvature*I on the pad
        DOFs); v is 0 on the pad DOFs (gradients are real-masked), so the pad solve
        returns 0 and the real DOFs get the exact preconditioner solve. Fully
        batched over B via ``torch.cholesky_solve`` (NO MLIP forward)."""
        sol = torch.cholesky_solve(v.unsqueeze(-1), self._precon_L).squeeze(-1)
        return sol * self._real_mask.to(sol.dtype)

    def _build_precon(self, atoms_list):
        """(Re)build the per-structure preconditioner P and cache its Cholesky factor.

        No-op (sets ``_precon_L=None``) when ``precon`` is disabled -> the default
        optimizer path is byte-identical to the oracle. Otherwise P (B, nmax, nmax)
        is assembled from the CURRENT geometry by either the Lindh model Hessian
        (reused verbatim from ts/algorithm/initial_hessian.py) or the exponential
        (Packwood-Csanyi) graph-Laplacian model, then per structure:
          (i)  symmetrized;
          (ii) stabilized -- add ``stab * mean(diag) * I`` on the real block to lift
               the translational/rotational null space so the block is SPD;
          (iii)normalized so the real-DOF diagonal mean == ``curvature`` -- this makes
               P^{-1} carry the SAME 1/curvature scale as the unpreconditioned
               H_0^{-1}=I/curvature seed, so only the *conditioning* (off-diagonal
               structure), not the step magnitude, is changed;
          (iv) pad DOFs set to ``curvature * I`` (decoupled; gradients are 0 there).
        Built once per topology (run start + each batch shrink); the inner per-
        structure assembly loop is acceptable (validation batches are small, matching
        the Lindh builder's own per-structure loop), while the hot per-iteration
        apply (``_apply_pinv``) is fully batched."""
        if self._precon_mode is None:
            self._precon_L = None
            return
        device = self.device
        B, nmax = self._B, self._nmax
        if B == 0 or nmax == 0:
            self._precon_L = None
            return

        if self._precon_mode == "lindh":
            from maple.function.dispatcher.ts.algorithm.initial_hessian import (
                lindh_initial_hessian,
            )
            P = lindh_initial_hessian(atoms_list, nmax, device, dtype=DTYPE)
        else:  # "exp"
            P = self._build_exp_precon(atoms_list, nmax)

        eye_n = torch.eye(nmax, dtype=DTYPE, device=device)
        stab = self._precon_stab
        curv = float(self.curvature)
        diag_idx = torch.arange(nmax, device=device)
        for b, at in enumerate(atoms_list):
            L = 3 * len(at)
            Pb = P[b]
            if L < nmax:                       # kill any real<->pad coupling
                Pb[L:, :] = 0.0
                Pb[:, L:] = 0.0
            if L == 0:
                Pb[diag_idx, diag_idx] = curv
                continue
            sub = 0.5 * (Pb[:L, :L] + Pb[:L, :L].t())
            dmean = sub.diagonal().mean().clamp(min=1e-12)
            sub = sub + (stab * dmean) * eye_n[:L, :L]
            sub = sub * (curv / sub.diagonal().mean().clamp(min=1e-12))
            Pb[:L, :L] = sub
            if L < nmax:                       # pad block = curvature * I
                Pb[diag_idx[L:], diag_idx[L:]] = curv
        self._precon_L = self._batched_cholesky(P, curv)

    def _build_exp_precon(self, atoms_list, nmax) -> torch.Tensor:
        """Exponential (Packwood-Csanyi) preconditioner as an isotropic graph
        Laplacian, padded to (B, nmax, nmax).

        For each structure: pairwise distances r_ij; nearest-neighbour distance
        r_nn = min positive r; coupling c_ij = exp(-A (r_ij/r_nn - 1)) zeroed beyond
        r_cut = rcut_mult * r_nn and on the diagonal; graph Laplacian L = diag(sum_j
        c) - c (PSD, single translational null space lifted later by the common
        stabilizer); expanded isotropically to 3N via the Kronecker product
        ``L (x) I_3``. Geometry-only. DOI 10.1063/1.4947024."""
        device = self.device
        B = len(atoms_list)
        A = self._precon_exp_A
        rcut_mult = self._precon_exp_rcut_mult
        P = torch.zeros((B, nmax, nmax), dtype=DTYPE, device=device)
        for b, at in enumerate(atoms_list):
            N = len(at)
            if N == 0:
                continue
            Ldof = 3 * N
            pos = torch.tensor(np.asarray(at.get_positions(), dtype=np.float64),
                               dtype=DTYPE, device=device)          # (N,3)
            d = pos.unsqueeze(0) - pos.unsqueeze(1)                  # (N,N,3)
            r = d.norm(dim=-1)                                       # (N,N)
            eye = torch.eye(N, dtype=torch.bool, device=device)
            r_nn = r.masked_fill(eye, float("inf")).min().clamp(min=1e-6)
            r_cut = rcut_mult * r_nn
            C = torch.exp(-A * (r / r_nn - 1.0))                     # (N,N)
            C = C.masked_fill(eye, 0.0).masked_fill(r > r_cut, 0.0)
            Lap = torch.diag(C.sum(dim=1)) - C                       # (N,N) PSD
            # isotropic 3N expansion: kron(Lap, I_3)
            Lap3 = Lap.repeat_interleave(3, dim=0).repeat_interleave(3, dim=1)
            comp = torch.arange(Ldof, device=device) % 3
            iso = (comp.unsqueeze(0) == comp.unsqueeze(1)).to(DTYPE)
            P[b, :Ldof, :Ldof] = Lap3 * iso
        return P

    @staticmethod
    def _batched_cholesky(P: torch.Tensor, scale: float) -> torch.Tensor:
        """Batched lower-Cholesky factor of an SPD batch, with a diagonal-jitter
        fallback for numerical safety (the assembled P is SPD by construction)."""
        n = P.shape[-1]
        eye = torch.eye(n, dtype=P.dtype, device=P.device)
        base = max(float(scale), 1.0) * 1e-10
        jit = 0.0
        for _ in range(8):
            try:
                return torch.linalg.cholesky(P if jit == 0.0 else P + jit * eye)
            except Exception:
                jit = base if jit == 0.0 else jit * 10.0
        d = P.diagonal(dim1=-2, dim2=-1).clamp(min=base)
        return torch.linalg.cholesky(torch.diag_embed(d))

    # ===================================================
    # STEP CLIPPING
    # ===================================================
    def _clip_step_batched(self, step: torch.Tensor) -> torch.Tensor:
        """
        Clip step size per batch to maxstep.
        
        Args:
            step: (B, nmax) step tensor
            
        Returns:
            clipped_step: (B, nmax) clipped step tensor
        """
        max_disp = step.abs().amax(dim=-1)  # (B,)
        scale = torch.clamp(self.maxstep / (max_disp + 1e-20), max=1.0)
        return step * scale.unsqueeze(-1)

    # ===================================================
    # HISTORY UPDATE
    # ===================================================
    def _update_history_batched(self, s_vec: torch.Tensor, y_vec: torch.Tensor):
        """
        Update L-BFGS history for all batches.
        
        Args:
            s_vec: (B, nmax) position change
            y_vec: (B, nmax) gradient change
        """
        device = self.device
        B = s_vec.shape[0]

        # Compute rho = 1 / (y^T s)
        ys = (y_vec * s_vec).sum(dim=-1)  # (B,)
        rho = 1.0 / (ys + 1e-20)
        
        # Check validity (finite rho and positive curvature)
        valid = torch.isfinite(rho) & (ys > 1e-12)

        # Add to history
        self.S_history.append(s_vec.clone())
        self.Y_history.append(y_vec.clone())
        self.rho_history.append(rho)

        # Update validity mask
        if len(self.S_history) > self.memory:
            # Remove oldest
            self.S_history.pop(0)
            self.Y_history.pop(0)
            self.rho_history.pop(0)
            # Shift validity
            self.history_valid = torch.cat([
                self.history_valid[:, 1:],
                valid.unsqueeze(-1)
            ], dim=-1)
        else:
            # Append new validity column
            if self.history_valid.shape[1] < self.memory:
                new_col = torch.zeros((B, self.memory - self.history_valid.shape[1]), 
                                     dtype=torch.bool, device=device)
                self.history_valid = torch.cat([self.history_valid, new_col], dim=-1)
            self.history_valid[:, len(self.S_history) - 1] = valid

    # ===================================================
    # HISTORY SHRINKING
    # ===================================================
    def _shrink_history(self, survive_idx: torch.Tensor):
        """Shrink history when batches are removed."""
        for i in range(len(self.S_history)):
            self.S_history[i] = self.S_history[i][survive_idx]
            self.Y_history[i] = self.Y_history[i][survive_idx]
            self.rho_history[i] = self.rho_history[i][survive_idx]
        self.history_valid = self.history_valid[survive_idx]

    # ===================================================
    # CONVERGENCE CHECK
    # ===================================================
    def _check_convergence(self, it, E, max_f, rms_f, max_dp, rms_dp):
        """
        Check convergence criteria for each batch.

        Thresholds and the four batched metrics are precomputed/passed in:
          * thresholds (self._f_max_th, ...) are built ONCE per topology in
            _rebuild_topology (they only change when the batch shrinks), instead
            of being rebuilt from python lists every iteration;
          * (max_f, rms_f, max_dp, rms_dp) are the shared reductions already
            computed in run();
          * the per-structure energy E is the committed-geometry energy threaded
            from run() -- the original recomputed it with a second full forward.

        Returns:
            done: (B,) boolean tensor indicating converged batches
        """
        # Check convergence (fully vectorized over the batch)
        done = (
            (max_f <= self._f_max_th)
            & (rms_f <= self._f_rms_th)
            & (max_dp <= self._dp_max_th)
            & (rms_dp <= self._dp_rms_th)
        )

        # Log convergence table (sync-light)
        self._w(self._fmt_convergence_table(
            it=it,
            E=E,
            max_f=max_f, rms_f=rms_f,
            max_dp=max_dp, rms_dp=rms_dp,
            done=done
        ))

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
        # effective DOF count per structure (for RMS denominators); cached so the
        # hot loop never re-derives it.
        self._L_eff = self._L_vec.clamp(min=1).to(DTYPE)

        # real_mask padded to fixed nmax
        self._real_mask = (self._arange_n[None, :] < self._L_vec[:, None])

        # Precompute per-structure convergence thresholds ONCE per topology.
        # These only change when the batch shrinks (which re-calls this method),
        # so rebuilding them from python lists every iteration (host->device copy
        # + sync x4) was pure overhead.
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

        # OPT-IN: (re)build the per-structure preconditioner for the current
        # topology + geometry. Cheap, geometry-only, no MLIP forward. No-op (sets
        # _precon_L=None) when precon is disabled, so the default path is untouched.
        self._build_precon(atoms_list)

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
        """Log iteration information. Metrics are passed in (already computed in
        run()). The five batch means are stacked into ONE tensor so the whole line
        costs a single device->host sync instead of five .item() calls."""
        if self.verbose == 0:
            return

        means = torch.stack([
            E.mean(), max_f.mean(), rms_f.mean(), max_dp.mean(), rms_dp.mean()
        ]).tolist()
        e_m, maxf_m, rmsf_m, maxdp_m, rmsdp_m = means

        msg = f"\nIter {it}: "
        msg += f"E_mean={e_m:.6f} "
        msg += f"max|F|_mean={maxf_m:.6f} "
        msg += f"rms|F|_mean={rmsf_m:.6f} "
        msg += f"max|dX|_mean={maxdp_m:.6f} "
        msg += f"rms|dX|_mean={rmsdp_m:.6f}\n"

        self._w(msg)

    def _fmt_convergence_table(self, it, E, max_f, rms_f, max_dp, rms_dp, done):
        """Format convergence table (byte-identical output to the original).

        The original did ~6*B `.item()` calls (one device->host sync each) per
        iteration -- 384 syncs/iter at B=64. Here every per-structure scalar is
        moved to host in batched `.tolist()` transfers (3 syncs total: the 5
        stacked float metrics, the bool `done` vector, the original-index map),
        then formatted purely on CPU.
        """
        B = E.shape[0]
        # one batched D2H transfer for all five float metrics
        E_l, maxf_l, rmsf_l, maxdp_l, rmsdp_l = torch.stack(
            [E, max_f, rms_f, max_dp, rms_dp], dim=0).tolist()
        done_l = done.tolist()
        orig_l = self._orig_index.tolist()

        lines = []
        lines.append("-" * 70 + "\n")
        lines.append(f"{('Batch L-BFGS Iteration ' + str(it)).center(70)}\n")
        lines.append("-" * 70 + "\n")
        lines.append(
            "Batch   Energy(Ha)   Max|F|(H/A)  RMS|F|  Max|dX|(A)  RMS|dX|  Conv\n"
        )
        lines.append("-" * 70 + "\n")

        for b in range(B):
            lines.append(
                f"[{orig_l[b]:2d}] "
                f"{E_l[b]:14.6f} "
                f"{maxf_l[b]:12.6f} "
                f"{rmsf_l[b]:10.6f} "
                f"{maxdp_l[b]:12.6f} "
                f"{rmsdp_l[b]:10.6f} "
                f"{('YES' if done_l[b] else 'NO')}\n"
            )

        lines.append("\n")
        return "".join(lines)

    # ===================================================
    # XYZ OUTPUT
    # ===================================================
    def _init_xyz_paths(self, B_all):
        self.xyz_paths = [
            os.path.join(self.out_dir, f"lbfgs_batch{i + 1}.xyz")
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