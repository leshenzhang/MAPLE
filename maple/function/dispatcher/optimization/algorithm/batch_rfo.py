# -*- coding: utf-8 -*-
"""
BatchRFO: GPU-batched Rational Function Optimization MINIMIZER with fixed padded
dimension nmax, per-structure trust-region masking, and dynamic-shrink eviction.

This is the MINIMIZATION analog of ``dispatcher/ts/algorithm/BPRFO.py`` (BatchPRFO).
It reuses BPRFO's batched machinery verbatim -- fixed-nmax padding, mass-weighted
Hessian eigendecomposition, per-structure trust-region inner loop, the
``get_efh_gpu`` Hessian pull, dynamic batch shrinking -- but DROPS the
transition-state mode partition (1 mode uphill / rest downhill). Instead it follows
ALL modes DOWNHILL to a minimum via the standard single-shift RFO / More-Sorensen
trust-region step:

  * unconstrained Newton step  s = -g/w  (in the mass-weighted eigenbasis) when it
    lies inside the trust radius;
  * otherwise a single eigenvalue shift  mu < min(w)  found by bracket+bisection so
    that ``||s(mu)|| = R`` with  s(mu) = -g/(w - mu)  (every mode descends, the
    shift makes the shifted Hessian positive-definite -> guaranteed energy decrease).

The per-structure update rule, thresholds, regularization (evals_eps=1e-10,
mu_margin=1e-8), trust logic (eta_shrink=0.75, eta_expand=1.75) and 60-iter
bisection are byte-for-byte the same arithmetic as the SINGLE-structure oracle
``dispatcher/optimization/algorithm/RFO.py`` (class RFO, method ``_rfo_step_mw`` +
``run`` accept/reject), so the batched run matches the serial single-RFO run on each
structure (parity oracle).

Calculator contract (UMABatchCalc, reused, not modified):
    prepare(atoms_list, fixed_nmax=None) ; get_ef_gpu() -> (E (B,), F (B,nmax_dof))
    get_efh_gpu() -> (E (B,), F (B,nmax_dof), H (B,nmax_dof,nmax_dof), P (B,))
    step_cart_(delta) / backup_coords() / restore_coords() / coord buffer.
All Hartree / Hartree-Ang. float64 math throughout.
"""

from typing import List, Optional
import os
import numpy as np
import torch
from ase import Atoms

DTYPE = torch.float64
BIG = 1e8


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


class BatchRFO:
    """
    Batched single-shift RFO minimizer with fixed padded dimension nmax.

    Parameters mirror the single-structure RFOParams so the batched update rule is
    numerically identical to the serial oracle ``RFO``.
    """

    def __init__(self,
                 output: str,
                 trust_init: float = 0.20,
                 trust_min: float = 1e-3,
                 trust_max: float = 1.00,
                 eta_shrink: float = 0.75,
                 eta_expand: float = 1.75,
                 evals_eps: float = 1e-10,
                 mu_margin: float = 1e-8,
                 max_bisect_it: int = 60,
                 max_inner_attempts: int = 12,
                 max_outer_iter: int = 300,
                 device: str = "cuda",
                 verbose: int = 1):
        self.trust_init = float(trust_init)
        self.trust_min = float(trust_min)
        self.trust_max = float(trust_max)
        self.eta_shrink = float(eta_shrink)
        self.eta_expand = float(eta_expand)
        self.evals_eps = float(evals_eps)
        self.mu_margin = float(mu_margin)
        self.max_bisect_it = int(max_bisect_it)
        self.max_inner_attempts = int(max_inner_attempts)
        self.max_outer_iter = int(max_outer_iter)
        self.device = torch.device(device)
        self.verbose = int(verbose)

        # === Forward-reuse optimization (mirror BPRFO FIX #2 + FIX #1) ===
        # BatchRFO's single-structure oracle (optimization/algorithm/RFO.py) keeps the
        # accepted trial's forces and threads the committed forward forward; BatchRFO
        # regressed by (a) dropping the inner trial forces and recomputing them with a
        # fresh forward purely for the convergence check (the :233 get_ef_gpu), and
        # (b) re-evaluating the SAME committed geometry's base forward at the top of the
        # next outer iter inside get_efh_gpu. Both are eliminated below (UMA is
        # block-diagonal => the reused E/F are bit-identical to a fresh forward at the
        # byte-identical committed geometry, up to fp32 run-to-run noise). Default ON;
        # set MAPLE_RFO_NO_FWD_REUSE=1 to restore the legacy recompute path = the
        # byte-parity oracle (precedent: BPRFO's MAPLE_NO_FWD_REUSE).
        self._reuse_forward = (os.environ.get("MAPLE_RFO_NO_FWD_REUSE", "0") != "1")
        self._fwd_reused = 0   # diagnostics: # forwards eliminated by reuse

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
        self._D = None
        self._orig_index = None

        # Per-structure convergence thresholds, precomputed once per topology
        # (rebuilt only on batch shrink). Minimization defaults match blbfgs.
        self._f_max_th = None
        self._f_rms_th = None
        self._dp_max_th = None
        self._dp_rms_th = None

        self._final_status = None    # per ORIGINAL index: converged / max_iter

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
        self._w("# Batched single-shift RFO minimization start\n")
        self._w(f"# trust_init={self.trust_init} trust_min={self.trust_min} "
                f"trust_max={self.trust_max} eta_shrink={self.eta_shrink} "
                f"eta_expand={self.eta_expand}\n")
        self._w(f"# max_outer_iter={self.max_outer_iter} "
                f"max_inner_attempts={self.max_inner_attempts} device={device}\n\n")

        self._init_xyz_paths(B0)
        self._symbols_per_batch = _symbols_flat(atoms_list)

        # === First prepare to fix nmax ===
        calc.prepare(atoms_list)
        # nmax_dof is set inside prepare() BEFORE any forward (== F.shape[1]); read
        # it directly instead of burning a full forward+backward just to get a shape.
        self._nmax = int(calc.nmax_dof)
        self._arange_n = torch.arange(self._nmax, device=device)

        # Build topology (nmax fixed) + per-structure masks/thresholds/mass-weights
        self._rebuild_topology(atoms_list)
        self._dump_xyz_all(calc, atoms_list, tag="init")

        B = self._B
        trust_r = torch.full((B,), self.trust_init, dtype=DTYPE, device=device)
        last_step = torch.zeros((B, self._nmax), dtype=DTYPE, device=device)
        self._final_status = [None] * B0

        outer_it = 0
        # FIX #1: committed (E,F) at the geometry that will be the TOP of the next outer
        # iteration (== the committed geometry of this iteration; no step happens between
        # the end of an outer iter and the get_efh_gpu at the start of the next one). When
        # set, it is fed to get_efh_gpu(base_ef=...) so its base forward (the gradient
        # point) is skipped. Reset to None on the first iter and after any batch shrink
        # (geometry/batch ordering changed -> a real base forward must run).
        base_ef_next = None
        while outer_it < self.max_outer_iter and len(atoms_list) > 0:
            outer_it += 1
            real_mask = self._real_mask
            rmask = real_mask.to(DTYPE)

            # --- Exact (numerical FD) Hessian + E/F at the current geometry.
            # Single RFO recomputes the exact Hessian every accepted step (geometry-
            # keyed cache); the batch does ONE get_efh_gpu per outer iteration, and an
            # in-iteration reject restores the SAME geometry -> the cached H stays valid
            # across inner retries, exactly like the single oracle.
            # FIX #1: when base_ef_next is set (reuse ON), the committed (E,F) already
            # evaluated at THIS exact geometry (end of last outer iter) is passed in so
            # get_efh_gpu skips its base forward; the FD Hessian is still computed fresh.
            if self._reuse_forward and base_ef_next is not None:
                E_old, F_raw, H_raw, _ = calc.get_efh_gpu(base_ef=base_ef_next)
                self._fwd_reused += 1
            else:
                E_old, F_raw, H_raw, _ = calc.get_efh_gpu()
            E_old = E_old.to(dtype=DTYPE)
            F_raw = F_raw.to(dtype=DTYPE)
            H_raw = 0.5 * (H_raw + H_raw.transpose(-1, -2)).to(dtype=DTYPE)

            # pad to fixed nmax if the active topology shrank to a smaller raw width
            Bcur, Lcur = F_raw.shape
            if Lcur != self._nmax:
                F_use = torch.zeros((Bcur, self._nmax), dtype=DTYPE, device=device)
                F_use[:, :Lcur] = F_raw
                H_use = torch.zeros((Bcur, self._nmax, self._nmax), dtype=DTYPE, device=device)
                H_use[:, :Lcur, :Lcur] = H_raw
            else:
                F_use, H_use = F_raw, H_raw

            # Cartesian Hessian (masked) + gradient g = -F (single RFO convention)
            mask_ij = (real_mask.unsqueeze(-1) & real_mask.unsqueeze(-2)).to(DTYPE)
            H_cart = H_use * mask_ij
            g_cart = -F_use * rmask

            # Mass-weighting (D = 1/sqrt(m)); pad modes get +BIG on the diagonal and a
            # zeroed gradient so they decouple (eigenvalue ~BIG, projected grad ~0) and
            # never contribute to the step -- the real-block eigenpairs are then
            # identical to the single oracle's mass-weighted Hessian.
            H_mw, g_mw = self._mass_weight_hg(H_cart, g_cart, real_mask)

            # Eigendecomposition + projected gradient (eigenbasis)
            w, V = torch.linalg.eigh(H_mw)                       # (B,n),(B,n,n)
            g_proj = (V.transpose(-1, -2) @ g_mw.unsqueeze(-1)).squeeze(-1)  # (B,n)

            # Regularize tiny eigenvalues (verbatim single-RFO rule)
            eps = self.evals_eps
            tiny = w.abs() < eps
            w = torch.where(tiny & (w == 0.0), torch.full_like(w, eps), w)
            w = torch.where(tiny & (w != 0.0), torch.sign(w) * eps, w)

            do_log = (self.verbose == 1)

            # Inner trust-region loop (per-structure masked accept/reject/shrink/expand).
            # FIX #2: it also returns the committed-geometry (E,F) it already evaluated --
            # the accepted trial's forward sits at base+s_cart == the committed geometry,
            # and the legacy code dropped its forces via `_`, forcing the :233 forward to
            # recompute them at the identical geometry.
            trust_r, last_step, last_rho, step_accepted, E_committed, F_committed = \
                self._inner_loop(
                    it=outer_it, calc=calc, w=w, V=V, g_proj=g_proj,
                    H=H_cart, g_cart=g_cart, trust_r=trust_r, last_step=last_step,
                    real_mask=real_mask, E_old=E_old, F_base=F_use, log=do_log,
                )

            # Committed-geometry E/F (for convergence). FIX #2: reuse the inner loop's
            # committed (E,F) -- block-diagonal UMA forward => bit-identical to a fresh
            # forward at the same coords. On reuse-OFF, run the legacy fresh forward
            # (the byte-parity oracle = the old :233 get_ef_gpu).
            if self._reuse_forward:
                E_fin = E_committed
                F_fin = F_committed
                self._fwd_reused += 1
            else:
                E_fin, F_fin = calc.get_ef_gpu()
                E_fin = E_fin.to(dtype=DTYPE)
                F_fin = F_fin.to(dtype=DTYPE)
            g_new = -F_fin * rmask

            # FIX #1: thread committed (E,F) to next iter's get_efh_gpu base forward.
            # calc.coord now sits at the committed geometry (accepted commits in the
            # inner loop were NOT restored), so this is the geometry at the top of the
            # next outer iter. Invalidated (None) below if a batch shrink reorders/resizes
            # the batch (the threaded rows would no longer align with the new ordering).
            base_ef_next = (E_fin, F_fin) if self._reuse_forward else None

            done = self._check_convergence(
                it=outer_it, E=E_fin, g_new=g_new, last_step=last_step,
                trust_r=trust_r, last_rho=last_rho,
            )

            # ---- Dynamic batch shrinking = eviction of converged structures ----
            leaving = done.nonzero(as_tuple=False).flatten().cpu().tolist()
            for i_local in leaving:
                oi = int(self._orig_index[i_local].item())
                self._final_status[oi] = "converged"

            survive_local = (~done).nonzero(as_tuple=False).flatten()
            if survive_local.numel() < done.numel():
                # batch shrink -> the committed (E,F) rows no longer align with the
                # reindexed survivor batch; force a real base forward next iter.
                base_ef_next = None
                # commit current geometries back into the (pre-slice) atoms objects
                self._sync_atoms_from_calc(calc, atoms_list)

                atoms_list = [atoms_list[i] for i in survive_local.cpu().tolist()]
                self._orig_index = self._orig_index[survive_local]
                trust_r = trust_r[survive_local]
                last_step = last_step[survive_local]

                if len(atoms_list) == 0:
                    self._w("\n# All structures converged.\n")
                    break

                calc.prepare(atoms_list, fixed_nmax=self._nmax)
                self._rebuild_topology(atoms_list)

        else:
            if len(atoms_list) > 0:
                self._w("\n# Maximum iterations reached.\n")

        # Any structure still active at loop exit hit the iteration cap.
        if self._orig_index is not None:
            self._sync_atoms_from_calc(calc, atoms_list)
            for i_local in range(self._orig_index.numel()):
                oi = int(self._orig_index[i_local].item())
                if self._final_status[oi] is None:
                    self._final_status[oi] = "max_iter"

        n_conv = sum(1 for s in self._final_status if s == "converged")
        n_max = sum(1 for s in self._final_status if s == "max_iter")
        self._w(f"\n# Final status: converged={n_conv} max_iter={n_max}\n")
        self._w(f"# fwd_reused (forwards eliminated by reuse)={self._fwd_reused} "
                f"reuse_forward={self._reuse_forward}\n")
        self._close_log()

    # ===================================================
    # MASS-WEIGHTING  (mirror BatchPRFO._mass_weight_hg)
    # ===================================================
    def _mass_weight_hg(self, H, g_cart, real_mask):
        D = self._D
        arange_n = self._arange_n

        g_mw = D * g_cart
        H_mw = D.unsqueeze(-1) * H * D.unsqueeze(-2)

        pad_mask = ~real_mask
        if pad_mask.any():
            H_mw = H_mw.clone()
            diag = H_mw[..., arange_n, arange_n]
            H_mw[..., arange_n, arange_n] = diag + pad_mask.to(DTYPE) * BIG
            g_mw = g_mw * (~pad_mask).to(DTYPE)
        return H_mw, g_mw

    # ===================================================
    # SINGLE-SHIFT RFO MINIMIZATION STEP (batched)
    # ===================================================
    @torch.no_grad()
    def _min_step(self, w, V, g_proj, trust_r, real_mask):
        """Batched single-shift RFO minimization step.

        Exact batched replica of single-RFO ``_rfo_step_mw``:
          1. unconstrained Newton  s_unc = -g_proj/w  (eigenbasis); use it when
             ||s_unc|| <= R (on_boundary = False);
          2. else shift mu < min(w) by bracket+bisection so ||s(mu)|| = R, with
             s(mu) = -g_proj/(w - mu), then clamp exactly to the boundary.
        Pad eigenmodes carry w ~ BIG and g_proj ~ 0, so they contribute a ~0 step and
        never become min(w); no explicit pad masking is needed (parity-exact).

        Returns (s_cart (B,n), on_boundary (B,) bool).
        """
        eps = self.evals_eps
        B, n = w.shape

        # --- unconstrained Newton step in eigenbasis ---
        denom0 = torch.where(w.abs() < eps, torch.sign(w) * eps, w)
        s_unc_eig = -g_proj / denom0                              # (B,n)
        norm_unc2 = (s_unc_eig ** 2).sum(dim=-1)                  # (B,) (V orthonormal)
        R2 = trust_r ** 2
        inside = norm_unc2 <= R2

        # --- constrained branch: shift mu < min(w), bracket + bisection ---
        w_min = w.min(dim=1).values                              # most-negative real mode
        hi = w_min - self.mu_margin                              # upper bound on mu

        def Fmu(mu):                                            # (B,) -> (B,)  sum (g/(w-mu))^2
            denom = w - mu.unsqueeze(1)
            denom = torch.where(denom.abs() < eps, torch.sign(denom) * eps, denom)
            return ((g_proj / denom) ** 2).sum(dim=-1)

        # expand the lower bracket downward while F(lo) > R2 (lockstep, <= max_bisect_it)
        lo = hi - 1.0
        for _ in range(self.max_bisect_it):
            need = Fmu(lo) > R2
            if not bool(need.any()):
                break
            step = torch.clamp(lo.abs() * 0.5, min=1.0)
            lo = torch.where(need, lo - step, lo)

        # bisection (F is monotonically increasing in mu on mu < w_min)
        hi_b = hi.clone()
        lo_b = lo
        for _ in range(self.max_bisect_it):
            mid = 0.5 * (lo_b + hi_b)
            go_hi = Fmu(mid) > R2
            hi_b = torch.where(go_hi, mid, hi_b)
            lo_b = torch.where(~go_hi, mid, lo_b)
        mu_star = 0.5 * (lo_b + hi_b)

        denom = w - mu_star.unsqueeze(1)
        denom = torch.where(denom.abs() < eps, torch.sign(denom) * eps, denom)
        s_con_eig = -g_proj / denom
        norm_con = torch.sqrt((s_con_eig ** 2).sum(dim=-1))
        scale = torch.where(norm_con > 0, trust_r / norm_con, torch.ones_like(norm_con))
        s_con_eig = s_con_eig * scale.unsqueeze(1)               # clamp to boundary

        # --- select unconstrained vs constrained ---
        s_eig = torch.where(inside.unsqueeze(1), s_unc_eig, s_con_eig)
        on_boundary = ~inside

        # eigenbasis -> mass-weighted -> Cartesian
        s_mw = (V @ s_eig.unsqueeze(-1)).squeeze(-1) * real_mask
        s_cart = self._D * s_mw
        return s_cart, on_boundary

    # ===================================================
    # INNER TRUST-REGION LOOP  (mirror BatchPRFO._inner_rs_prfo_loop)
    # ===================================================
    @torch.no_grad()
    def _inner_loop(self, it, calc, w, V, g_proj, H, g_cart,
                    trust_r, last_step, real_mask, E_old, F_base, log=True):
        device = self.device
        B, n = g_proj.shape

        accepted = torch.zeros(B, dtype=torch.bool, device=device)
        step_accepted = torch.zeros(B, dtype=torch.bool, device=device)
        last_rho = torch.full((B,), float("nan"), dtype=DTYPE, device=device)

        # === FIX #2: committed-geometry E/F, threaded back to run() so it can skip the
        # post-step convergence forward (legacy :233 get_ef_gpu). Init to the base point
        # (E_old / F_base at entry): a row that never accepts a step stays at base, so its
        # committed E/F == base. On each accept the row is overwritten with the accepted
        # trial's E and forces (the trial forward at base+s_cart IS the committed geometry,
        # whose forces the legacy code dropped via `_`).
        E_committed = E_old.clone()
        F_committed = F_base.clone()

        for _try in range(self.max_inner_attempts):
            pend = ~accepted
            if not bool(pend.any()):
                break

            s_cart, on_boundary = self._min_step(w, V, g_proj, trust_r, real_mask)

            # Trial only the pending structures
            s_try = torch.zeros_like(s_cart)
            s_try[pend] = s_cart[pend]

            calc.backup_coords()
            calc.step_cart_(s_try)
            # FIX #2: KEEP the trial forces (legacy dropped them via `_`). For an accepted
            # row, base+s_cart IS the committed geometry, so F_trial == the post-step
            # forward's forces (block-diagonal UMA) -> run() reuses them, skipping :233.
            E_new, F_trial = calc.get_ef_gpu()
            E_new = E_new.to(dtype=DTYPE)
            F_trial = F_trial.to(dtype=DTYPE)
            calc.restore_coords()

            # quadratic model change (cartesian == mass-weighted, parity-exact)
            Hs = torch.einsum("bij,bj->bi", H, s_try)
            model_change = (g_cart * s_try).sum(-1) + 0.5 * (s_try * Hs).sum(-1)
            actual_change = E_new - E_old

            rho = torch.full_like(model_change, float("nan"))
            ok = (model_change.abs() > 1e-16) & pend
            rho[ok] = actual_change[ok] / model_change[ok]
            last_rho = torch.where(pend, rho, last_rho)

            force_accept = trust_r <= (self.trust_min * 1.000000000001)
            bad = pend & ((~torch.isfinite(rho)) | (rho < self.eta_shrink))
            acc = pend & (~bad | force_accept)
            rej = pend & (~acc)

            if bool(acc.any()):
                s_commit = torch.zeros_like(s_cart)
                s_commit[acc] = s_cart[acc]
                calc.step_cart_(s_commit)
                last_step[acc] = s_commit[acc]
                step_accepted |= acc
                # FIX #2: accepted rows now sit at base+s_cart == committed geometry, the
                # exact point the trial forward (above) evaluated -> record its E/F there.
                E_committed[acc] = E_new[acc]
                F_committed[acc] = F_trial[acc]
                self._dump_xyz_subset(calc, acc, it)

            # trust-radius update (legacy == single-RFO run(): expand on good rho at
            # boundary; shrink on reject)
            grow = (rho > self.eta_expand) & on_boundary & acc
            trust_r = torch.where(grow, torch.clamp(2.0 * trust_r, max=self.trust_max), trust_r)
            trust_r = torch.where(rej, torch.clamp(0.5 * trust_r, min=self.trust_min), trust_r)

            if log:
                self._w(self._fmt_iter_head(it, acc, rej, rho, trust_r, E_new))
            accepted |= acc

        return trust_r, last_step, last_rho, step_accepted, E_committed, F_committed

    # ===================================================
    # CONVERGENCE
    # ===================================================
    def _check_convergence(self, it, E, g_new, last_step, trust_r, last_rho):
        L_eff = self._L_eff
        max_f = g_new.abs().amax(dim=-1)
        rms_f = torch.sqrt((g_new ** 2).sum(-1) / L_eff)
        max_dp = last_step.abs().amax(dim=-1)
        rms_dp = torch.sqrt((last_step ** 2).sum(-1) / L_eff)

        done = (
            (max_f <= self._f_max_th)
            & (rms_f <= self._f_rms_th)
            & (max_dp <= self._dp_max_th)
            & (rms_dp <= self._dp_rms_th)
        )

        self._w(self._fmt_cycle_table(it, E, last_rho, trust_r,
                                      max_f, rms_f, max_dp, rms_dp, done))
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
        mass = _masses_flat(atoms_list, self._nmax, device)
        self._D = 1.0 / torch.sqrt(torch.clamp(mass, min=1e-12))

        # minimization convergence thresholds (match blbfgs defaults; per-atoms
        # overridable -> identical to the single-RFO oracle which reads atoms.*_th)
        self._f_max_th = torch.tensor(
            [getattr(at, "f_max_th", 2e-3) for at in atoms_list], dtype=DTYPE, device=device)
        self._f_rms_th = torch.tensor(
            [getattr(at, "f_rms_th", 1e-3) for at in atoms_list], dtype=DTYPE, device=device)
        self._dp_max_th = torch.tensor(
            [getattr(at, "dp_max_th", 1e-3) for at in atoms_list], dtype=DTYPE, device=device)
        self._dp_rms_th = torch.tensor(
            [getattr(at, "dp_rms_th", 5e-4) for at in atoms_list], dtype=DTYPE, device=device)

    def _sync_atoms_from_calc(self, calc, atoms_list):
        with torch.no_grad():
            pos = _get_coord_gpu(calc).detach().cpu().numpy()
        ptr = self._ptr.detach().cpu().numpy()
        for i, at in enumerate(atoms_list):
            s, t = ptr[i], ptr[i + 1]
            at.positions[:] = pos[s:t]

    # ===================================================
    # LOGGING / XYZ
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

    def _init_xyz_paths(self, B_all):
        self.xyz_paths = [
            os.path.join(self.out_dir, f"rfo_batch{i + 1}.xyz") for i in range(B_all)
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

    def _dump_xyz_subset(self, calc, accept_mask, it):
        if not bool(accept_mask.any()):
            return
        with torch.no_grad():
            pos = _get_coord_gpu(calc).detach().cpu().numpy()
        ptr = self._ptr.detach().cpu().numpy()
        for i_local in accept_mask.nonzero(as_tuple=False).flatten().cpu().tolist():
            s, t = ptr[i_local], ptr[i_local + 1]
            idx_orig = int(self._orig_index[i_local].item())
            symbols = self._symbols_per_batch[idx_orig]
            self._append_xyz(idx_orig, symbols, pos[s:t], f"iter={it}")

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

    def _fmt_iter_head(self, it, acc, rej, rho, trust_r, E_new):
        acc_idx = acc.nonzero(as_tuple=False).flatten().cpu().tolist()
        rej_idx = rej.nonzero(as_tuple=False).flatten().cpu().tolist()
        acc_list = [int(self._orig_index[i]) for i in acc_idx]
        rej_list = [int(self._orig_index[i]) for i in rej_idx]
        rhos = rho.detach().cpu().numpy()
        Rs = trust_r.detach().cpu().numpy()
        En = E_new.detach().cpu().numpy()

        def head(arr, k=3, fmt="{:.3f}"):
            out = []
            for v in arr[:k]:
                try:
                    out.append(fmt.format(float(v)))
                except Exception:
                    out.append("nan")
            return "[" + ", ".join(out) + "]"

        return (f"Iter {it}: accepted={acc_list} rejected={rej_list} "
                f"rho_head={head(rhos)} R_head={head(Rs)} "
                f"E_head={head(En, fmt='{:.6f}')}\n")

    def _fmt_cycle_table(self, it, E, rho, R, max_f, rms_f, max_dp, rms_dp, done):
        B = E.shape[0]
        E_l, rho_l, R_l, maxf_l, rmsf_l, maxdp_l, rmsdp_l = torch.stack(
            [E, rho, R, max_f, rms_f, max_dp, rms_dp], dim=0).tolist()
        done_l = done.tolist()
        orig_l = self._orig_index.tolist()

        lines = ["-" * 78 + "\n",
                 f"{('Batch RFO Cycle ' + str(it)).center(78)}\n",
                 "-" * 78 + "\n",
                 "Batch  Energy(Ha)    rho     R(MW)   Max|F|     RMS|F|    Max|dX|    RMS|dX|  Conv\n",
                 "-" * 78 + "\n"]
        for b in range(B):
            lines.append(
                f"[{orig_l[b]:2d}] {E_l[b]:13.6f} {rho_l[b]:8.3f} {R_l[b]:7.3f} "
                f"{maxf_l[b]:10.6f} {rmsf_l[b]:9.6f} {maxdp_l[b]:10.6f} {rmsdp_l[b]:9.6f} "
                f"{('YES' if done_l[b] else 'NO')}\n")
        lines.append("\n")
        return "".join(lines)
