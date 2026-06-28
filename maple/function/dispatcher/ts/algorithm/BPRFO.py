# -*- coding: utf-8 -*-
"""
BatchPRFO with fixed padded dimension nmax.
All EFH are padded to nmax determined from FIRST prepare/get_ef_gpu.
Calculator.prepare() is called with fixed_nmax afterwards.

FIXES:
1. Bofill update now happens after EVERY accepted step (not just non-recalc)
2. Gradient consistency: use same gradient source throughout iteration
3. Increased SR1 tolerance from 1e-10 to 1e-8
4. Better step acceptance tracking

===========================================================================
TRANSITION-STATE-OPTIMIZER UPGRADES (all OPT-IN; default = legacy full-Hessian
path, which is kept verbatim as both the fallback and the parity oracle).
Every new method is gated by a constructor flag that defaults to OLD behavior.
Literature each upgrade is grounded in (cited again at each method):

  [1] Iterative leftmost-eigenpair solver via matrix-free Hessian-vector
      products (Sella). Hermes, Sargent, Head-Gordon, Slavicek?  -- actually:
      Hermes, E. D.; Sargsyan, K.; Najm, H. N.; Zador, J. "Accelerated
      Saddle Point Refinement through Full Exploitation of Partial Hessian
      Diagonalization", J. Chem. Theory Comput. 2019, 15, 6536.
      DOI: 10.1021/acs.jctc.9b00869  and the Sella follow-up
      "Sella, an Open-Source Automation-Friendly Molecular Saddle Point
      Optimizer", J. Chem. Theory Comput. 2022, 18, 2.
      DOI: 10.1021/acs.jctc.2c00395
      => hessian_mode='iterative' (vs default 'full').
  [2] Trust-radius controller. Nocedal & Wright, "Numerical Optimization",
      2nd ed., Springer 2006, Ch. 4, Algorithm 4.1.
      => trust_mode='nw' (vs default 'legacy').
  [3] Eigenvector-following / mode tracking + negative-eigenvalue guard.
      Banerjee, Adams, Simons, Shepard, J. Phys. Chem. 1985, 89, 52.
      DOI: 10.1021/j100247a015 ; Baker, J. Comput. Chem. 1986, 7, 385.
      DOI: 10.1002/jcc.540070402
      => mode_follow_guard=True (vs default False).
  [4] ts_hessian curvature injection (Swart-Bickelhaupt; pysisyphus).
      DOI: 10.1002/qua.21049  => ts_hessian_inject=True (default False).
  [5] Lindh model initial Hessian. Lindh et al., Chem. Phys. Lett. 1995,
      241, 423. DOI: 10.1016/0009-2614(95)00646-L
      => initial_hessian='lindh' (vs default 'identity'); see initial_hessian.py.
  [6] Adaptive Hessian recalculation (pysisyphus hessian_recalc_adapt):
      recompute the exact Hessian once ||g|| drops below 1/n of its initial
      value, instead of a fixed RecalcFC cadence.
      => hessian_recalc_adapt=True (default False).
  [7] Reduced device->host sync: convergence/trust kept on-GPU, convergence +
      verbose logging only sampled every K iters.
      => conv_check_interval=K (default 1 = every iter = legacy).
===========================================================================
"""

from typing import List, Optional
import os
import numpy as np
import torch
from ase import Atoms

# Opt-in model-Hessian seeds (Lindh + ts_hessian injection). Imported defensively
# so BPRFO still imports standalone (e.g. kernprof microbench) if the sibling file
# is missing; the seeds are only reached when their opt-in flags are set.
try:
    from .initial_hessian import lindh_initial_hessian, ts_hessian_inject
except Exception:  # pragma: no cover - standalone import fallback
    try:
        from initial_hessian import lindh_initial_hessian, ts_hessian_inject
    except Exception:
        lindh_initial_hessian = None
        ts_hessian_inject = None

# --- profile shim: BPRFO carries @profile (line_profiler/kernprof). Under a plain
# `python` import, `profile` is undefined -> NameError. Provide a no-op fallback so
# the module is importable + benchmarkable standalone (kernprof still injects its own).
try:
    profile  # type: ignore[name-defined]
except NameError:
    def profile(f):
        return f

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


class BatchPRFO:
    """
    Batched RS-PRFO with fixed padded dimension nmax.
    """

    def __init__(self,
                 output: str,
                 trust_init: float = 0.20,
                 trust_min: float = 1e-3,
                 trust_max: float = 1.00,
                 eta_shrink: float = 0.75,
                 eta_expand: float = 1.75,
                 max_inner_attempts: int = 10,
                 max_outer_iter: int = 256,
                 device: str = "cuda",
                 recalc: int = 4,
                 pool_queue: Optional[list] = None,
                 B_target: Optional[int] = None,
                 pool_size_bucket: bool = False,
                 hessian_update: str = "bofill",
                 # ---- OPT-IN upgrades (all default to legacy behavior) ----
                 hessian_mode: str = "full",          # [1] 'full'(default,oracle)|'iterative'
                 iter_n_leftmost: int = 1,            # [1] # leftmost eigenpairs to resolve
                 iter_lanczos_m: int = 16,            # [1] max Krylov dim
                 iter_lanczos_m_min: int = 4,         # [1] min steps before residual early-stop
                 iter_gamma: float = 0.4,             # [1] loose conv: ||r|| < gamma*|lambda|
                 iter_fd_eta: float = 1e-3,           # [1] finite-diff HVP step (Angstrom)
                 hvp_source: str = "fd",              # [1] 'fd'(default)|'auto'|'autograd' HVP backend
                 trust_mode: str = "legacy",          # [2] 'legacy'(default)|'nw'
                 nw_eta_accept: float = 0.10,         # [2] accept step if rho > this
                 nw_rho_lo: float = 0.25,             # [2] rho<lo -> trust x1/4
                 nw_rho_hi: float = 0.75,             # [2] rho>hi & boundary -> trust x2
                 mode_follow_guard: bool = False,     # [3] restrict overlap to neg subspace + neg_num==0 guard
                 ts_hessian_inject: bool = False,     # [4] sign-flip+damp rxn mode of model/identity seed
                 ts_inject_scale: float = 0.25,       # [4] diag[rxn] = -scale*|diag|
                 initial_hessian: str = "identity",   # [5] 'identity'(default)|'lindh'
                 hessian_recalc_adapt: bool = False,  # [6] recalc when ||g|| < ||g0||/n
                 recalc_adapt_n: float = 10.0,        # [6] the 'n'
                 conv_check_interval: int = 1):       # [7] check convergence + verbose-log every K iters
        self.trust_init = trust_init
        self.trust_min = trust_min
        self.trust_max = trust_max
        self.eta_shrink = eta_shrink
        self.eta_expand = eta_expand
        self.max_inner_attempts = max_inner_attempts
        self.max_outer_iter = max_outer_iter
        self.device = torch.device(device)
        self.recalc = max(1, int(recalc))
        self.hessian_update = str(hessian_update).lower()
        if self.hessian_update not in {"bofill", "bfgs"}:
            raise ValueError("hessian_update must be 'bofill' or 'bfgs'.")

        # ---- OPT-IN upgrade config ----
        self.hessian_mode = str(hessian_mode).lower()
        if self.hessian_mode not in {"full", "iterative"}:
            raise ValueError("hessian_mode must be 'full' or 'iterative'.")
        self.iter_n_leftmost = max(1, int(iter_n_leftmost))
        self.iter_lanczos_m = max(2, int(iter_lanczos_m))
        self.iter_lanczos_m_min = max(2, int(iter_lanczos_m_min))
        self.iter_gamma = float(iter_gamma)
        self.iter_fd_eta = float(iter_fd_eta)
        # [1] HVP backend. DEFAULT 'fd' (finite-difference) because UMA's eSCN-MoE backbone
        # has an INCOMPLETE double-backward: its forces are exact but its autograd Hessian/HVP
        # are wrong by ~1.2e-2 Ha/A^2 (delta-independent => a real error, not FD truncation),
        # which would make the eigensolver follow the wrong mode. FD HVP is the accurate path
        # for UMA. Only switch to 'autograd' for models with a reliable double-backward (e.g.
        # MACE-OFF). 'auto' = use analytic calc.hvp/hvp_batch IF advertised, else FD.
        self.hvp_source = str(hvp_source).lower()
        if self.hvp_source not in {"fd", "auto", "autograd"}:
            raise ValueError("hvp_source must be 'fd', 'auto' or 'autograd'.")
        self.trust_mode = str(trust_mode).lower()
        if self.trust_mode not in {"legacy", "nw"}:
            raise ValueError("trust_mode must be 'legacy' or 'nw'.")
        self.nw_eta_accept = float(nw_eta_accept)
        self.nw_rho_lo = float(nw_rho_lo)
        self.nw_rho_hi = float(nw_rho_hi)
        self.mode_follow_guard = bool(mode_follow_guard)
        self.ts_hessian_inject = bool(ts_hessian_inject)
        self.ts_inject_scale = float(ts_inject_scale)
        self.initial_hessian = str(initial_hessian).lower()
        if self.initial_hessian not in {"identity", "lindh"}:
            raise ValueError("initial_hessian must be 'identity' or 'lindh'.")
        self.hessian_recalc_adapt = bool(hessian_recalc_adapt)
        self.recalc_adapt_n = float(recalc_adapt_n)
        self.conv_check_interval = max(1, int(conv_check_interval))
        if (self.initial_hessian == "lindh" or self.ts_hessian_inject) \
                and lindh_initial_hessian is None:
            raise ImportError(
                "initial_hessian.py (lindh/ts_hessian_inject) not importable; "
                "cannot use initial_hessian='lindh' / ts_hessian_inject=True.")
        # --- iterative-eigensolver diagnostics ---
        self._hvp_kind = None             # 'hvp' | 'hvp_batch' | 'fd' (detected lazily)
        self._hvp_calls = 0               # total HVP evaluations (forwards in FD mode)
        self._lanczos_calls = 0           # # of _leftmost_eigpairs_mw invocations
        self._lanczos_steps = 0           # total Lanczos steps (sum of m used)
        self._lanczos_unconverged = 0     # Ritz pairs that missed gamma*|lambda| at m_max
        self._neg_guard_hits = 0          # # (system,iter) flagged neg_num==0 by mode guard
        # adaptive-recalc state (Task 6); scalar batch-level
        self._g0_norm_mean = None
        self._adapt_level = 0

        self.output = os.path.abspath(output)
        self.out_dir = os.path.dirname(self.output) or "."
        os.makedirs(self.out_dir, exist_ok=True)

        self.log_fp = None
        self.xyz_paths = []
        self.frame_counts = []

        self.tracked_mode_vec_mw = None
        self.tracked_mode_idx = None

        self._ptr = None
        self._L_vec = None
        self._nmax = 0
        self._B = 0
        self._symbols_per_batch = None
        self._arange_n = None
        self._real_mask = None
        self._D = None
        # Per-structure convergence thresholds, built ONCE per topology in
        # _rebuild_topology (they only change when the batch shrinks/refills)
        # instead of being rebuilt from python lists every _check_convergence.
        self._f_max_th = None
        self._f_rms_th = None
        self._dp_max_th = None
        self._dp_rms_th = None

        self._orig_index = None
        self._H_work = None
        self._g_cart_prev = None
        self._recent_acceptance_rate = 0.0
        self._enable_vectorized_mu = True

        # --- vectorized-mu diagnostics ---
        self._mu_unbracketed = 0          # # of (structure,call) left un-bracketed after 60 iters

        # --- outer/inner robustness (per-structure straggler control) ---
        # Counters live as (B,) tensors allocated in run(); these are the policy knobs.
        self._bad_streak_limit = 8        # consecutive rho-bad iters before "stuck" candidate
        self._pin_streak_limit = 8        # consecutive trust-pinned-at-min iters before "stuck"
        self._max_restarts = 2            # bounded inner restarts before eviction
        self._force_recalc_next = False   # set when a restart needs a fresh exact Hessian
        self._iter_count = None
        self._recalc_count = None
        self._bad_streak = None
        self._pin_streak = None
        self._restart_count = None
        self._final_status = None         # per ORIGINAL index: converged / evicted_straggler / max_iter

        # --- streaming pool (optional; None => no pooling => baseline behavior) ---
        # A list of pending ASE Atoms TS-guesses used to refill the active batch
        # back up to `B_target` after structures converge/evict and the batch shrinks.
        self._pool_queue_init = list(pool_queue) if pool_queue is not None else None
        self._B_target_init = int(B_target) if B_target is not None else None
        self._pool_size_bucket = bool(pool_size_bucket)
        self._pool_queue = None            # live queue (set in run())
        self._B_target = None              # live target (set in run())
        self._next_orig = 0                # next free ORIGINAL index for refilled guesses
        self._pool_refilled = 0            # diagnostics: total guesses pulled from queue

        # --- ACCURACY-PRESERVING forward reuse (FIX #1 + #2) -----------------
        # Default ON. The committed-geometry E/F threaded out of iter N-1 (end-of-
        # iter FORWARD-B + the inner loop's committed trial forward) are reused at
        # the top of iter N (FORWARD-A) and as the post-step gradient, because the
        # geometry does NOT change between FORWARD-B(N-1) and FORWARD-A(N) (only
        # backup_coords clones + convergence/shrink logic), and the UMA forward is
        # block-diagonal (one molecule's perturbation cannot leak to another). On
        # recalc/refill/forced iters the reuse is SKIPPED -> a real forward runs.
        # Set MAPLE_NO_FWD_REUSE=1 (or _reuse_forward=False) to force the legacy
        # 3-forward/iter path = the byte-parity oracle.
        self._reuse_forward = (os.environ.get("MAPLE_NO_FWD_REUSE", "0") != "1")
        # in-run audit: when on, FORWARD-A still does a fresh forward and the abs
        # diff vs the reused tensor is accumulated (proves geometry-unchanged,
        # isolating reuse-legitimacy from fp32 run-to-run calculator noise).
        self._audit_reuse = (os.environ.get("MAPLE_AUDIT_FWD_REUSE", "0") == "1")
        self._audit_max_dE = 0.0
        self._audit_max_dF = 0.0
        self._fwd_calls = 0               # diagnostics: # get_ef_gpu forwards
        self._fwd_reused = 0              # diagnostics: # forwards eliminated by reuse
        # committed-geometry E threaded from FORWARD-B(N-1) -> FORWARD-A(N).
        # (g is already carried by self._g_cart_prev = -F_committed*mask.)
        self._reuse_E = None


    # ===================================================
    # PUBLIC RUN
    # ===================================================
    @profile
    def run(self, mols,
            pool_queue: Optional[list] = None,
            B_target: Optional[int] = None) -> None:
        device = self.device
        atoms_list = list(mols.multiatoms)
        calc = mols.calc

        B0 = len(atoms_list)
        if B0 == 0:
            return

        # --- resolve streaming-pool config (run() args override __init__ defaults) ---
        pq = pool_queue if pool_queue is not None else self._pool_queue_init
        bt = B_target if B_target is not None else self._B_target_init
        self._pool_queue = list(pq) if pq is not None else None
        self._B_target = int(bt) if bt is not None else None
        if (self._pool_queue is not None and self._B_target is not None
                and self._pool_size_bucket):
            # D1.5 size-bucketing: optimize most-homogeneous refills first (less padding
            # waste) by draining the queue in ascending atom-count order.
            self._pool_queue.sort(key=lambda a: len(a))
        self._pool_refilled = 0

        self._orig_index = torch.arange(B0, dtype=torch.long, device=device)
        # ORIGINAL-index counter; refilled guesses get fresh indices >= B0.
        self._next_orig = B0

        self._open_log()
        self._w("# RS-PRFO batched TS search start\n")
        self._w(f"# RecalcFC interval: {self.recalc}\n")
        self._w(f"# Hessian update method: {self.hessian_update}\n")
        self._w(f"# hessian_mode={self.hessian_mode} trust_mode={self.trust_mode} "
                f"initial_hessian={self.initial_hessian} ts_hessian_inject={self.ts_hessian_inject} "
                f"mode_follow_guard={self.mode_follow_guard} "
                f"hessian_recalc_adapt={self.hessian_recalc_adapt} "
                f"conv_check_interval={self.conv_check_interval}\n")
        if self.hessian_mode == "iterative":
            self._w(f"# iterative eigensolver: n_leftmost={self.iter_n_leftmost} "
                    f"lanczos_m<={self.iter_lanczos_m} gamma={self.iter_gamma} "
                    f"fd_eta={self.iter_fd_eta} hvp_source={self.hvp_source}\n")
        # reset adaptive-recalc + iterative diagnostics for this run
        self._g0_norm_mean = None
        self._adapt_level = 0
        self._hvp_kind = None
        self._hvp_calls = 0
        self._lanczos_calls = 0
        self._lanczos_steps = 0
        self._lanczos_unconverged = 0
        self._neg_guard_hits = 0

        # reset forward-reuse state + diagnostics for this run
        self._reuse_E = None
        self._audit_max_dE = 0.0
        self._audit_max_dF = 0.0
        self._fwd_calls = 0
        self._fwd_reused = 0

        self._init_xyz_paths(B0)
        self._symbols_per_batch = _symbols_flat(atoms_list)

        # === First prepare to fix nmax ===
        calc.prepare(atoms_list)
        self._fwd_calls += 1
        _, F0 = calc.get_ef_gpu()
        self._nmax = int(F0.shape[1])
        self._arange_n = torch.arange(self._nmax, device=device)

        # Build topology (nmax fixed)
        self._rebuild_topology(atoms_list)
        self._dump_xyz_all(calc, atoms_list, tag="init")

        B = self._B
        trust_r = torch.full((B,), self.trust_init, dtype=DTYPE, device=device)
        last_step = torch.zeros((B, self._nmax), dtype=DTYPE, device=device)

        self.tracked_mode_vec_mw = None
        self.tracked_mode_idx = None

        # Per-structure robustness counters (sliced together with the batch on shrink).
        self._iter_count = torch.zeros((B,), dtype=torch.long, device=device)
        self._recalc_count = torch.zeros((B,), dtype=torch.long, device=device)
        self._bad_streak = torch.zeros((B,), dtype=torch.long, device=device)
        self._pin_streak = torch.zeros((B,), dtype=torch.long, device=device)
        self._restart_count = torch.zeros((B,), dtype=torch.long, device=device)
        self._final_status = [None] * B0
        self._force_recalc_next = False

        outer_it = 0
        self._H_work = None
        self._g_cart_prev = None

        while outer_it < self.max_outer_iter and len(atoms_list) > 0:
            outer_it += 1
            real_mask = self._real_mask

            calc.backup_coords()

            # Per-structure outer-iteration counter.
            self._iter_count = self._iter_count + 1

            # Decide whether to rebuild Hessian from calculator (Gaussian RecalcFC-like).
            # A bounded inner restart (straggler control) can force an extra fresh Hessian.
            forced = bool(self._force_recalc_next)
            self._force_recalc_next = False

            if self.hessian_mode == "iterative":
                # ---- [1] Matrix-free iterative path: NO full numerical Hessian here.
                # The working (model) Hessian carries bulk curvature (Lindh/identity seed +
                # Bofill updates); the leftmost eigenpair(s) of the TRUE Hessian are resolved
                # exactly each iteration by the Lanczos HVP solver in _eigh_and_track_modes.
                # The model seed is built ONCE (or on a forced straggler restart) -- the
                # RecalcFC cadence does NOT re-seed (that would wipe Bofill curvature).
                seed_now = (self._H_work is None) or forced
                # FIX #1: on a non-seed iter the committed geometry is identical to the
                # one already evaluated by FORWARD-B at the end of iter N-1 (only
                # backup_coords clones + convergence/shrink ran in between) -> thread its
                # E (self._reuse_E) and g (self._g_cart_prev) instead of a fresh forward.
                # SKIP on seed/forced/refill (geometry/batch changed -> real forward).
                if self._reuse_forward and (not seed_now) and (self._reuse_E is not None) \
                        and (self._reuse_E.shape[0] == real_mask.shape[0]):
                    E_old = self._reuse_E
                    g_cart = self._g_cart_prev
                    self._fwd_reused += 1
                    if self._audit_reuse:
                        self._audit_forward(calc, E_old, g_cart, real_mask)
                else:
                    E_old, F_now = self._ef(calc)
                    E_old = E_old.to(dtype=DTYPE)
                    F_now = F_now.to(dtype=DTYPE)
                    g_cart = -F_now * real_mask.to(DTYPE)
                if seed_now:
                    self._recalc_count = self._recalc_count + 1
                    self._H_work = self._build_seed_hessian(atoms_list, real_mask)
                    self._g_cart_prev = g_cart.clone()
                    if self._g0_norm_mean is None:
                        self._g0_norm_mean = float(g_cart.norm(dim=-1).mean().item())
                    self._w(f"[Iter {outer_it}] seeded '{self.initial_hessian}' model Hessian "
                            f"(ts_hessian_inject={self.ts_hessian_inject})\n")
                H_cart = self._H_work
                # gate the post-step Bofill block: skip ONLY on the (re)seed iteration.
                need_recalc = seed_now
                E_old_is_set = True
            else:
                E_old_is_set = False
                # [6] adaptive-recalc (opt-in, full mode): trigger when batch-mean ||g|| has
                # dropped below 1/n of its initial value (one trigger per geometric level),
                # using the previous committed gradient available at loop top. Replaces cadence.
                adapt_fire = False
                if (self.hessian_recalc_adapt
                        and self._g_cart_prev is not None and self._g0_norm_mean is not None):
                    cur = float(self._g_cart_prev.norm(dim=-1).mean().item())
                    thr = self._g0_norm_mean * (1.0 / self.recalc_adapt_n) ** (self._adapt_level + 1)
                    if cur < thr:
                        adapt_fire = True
                        self._adapt_level += 1
                        self._w(f"[Iter {outer_it}] adaptive recalc: <|g|>={cur:.3e} < "
                                f"g0/n^{self._adapt_level} ({thr:.3e})\n")

                if self.hessian_recalc_adapt:
                    need_recalc = ((outer_it == 1) or forced or adapt_fire)
                else:
                    need_recalc = ((outer_it == 1)
                                   or ((outer_it - 1) % self.recalc == 0)
                                   or forced)

            if E_old_is_set:
                pass  # iterative path already produced E_old / H_cart / g_cart above
            elif need_recalc:
                self._recalc_count = self._recalc_count + 1
                # Pull EFH (true or numerical) from calculator; pad to nmax.
                # A real forward (the Hessian base point) -- reuse is NOT applicable.
                self._fwd_calls += 1
                E_old, F_tmp, H_tmp, _ = calc.get_efh_gpu()
                E_old = E_old.to(dtype=DTYPE)
                F_tmp = F_tmp.to(dtype=DTYPE)
                H_tmp = 0.5 * (H_tmp + H_tmp.transpose(-1, -2)).to(dtype=DTYPE)

                # pad to nmax if needed
                Bcur, Lcur = F_tmp.shape
                if Lcur != self._nmax:
                    H_pad = torch.zeros((Bcur, self._nmax, self._nmax), dtype=DTYPE, device=H_tmp.device)
                    H_pad[:, :Lcur, :Lcur] = H_tmp
                    H_use = H_pad
                    F_use = torch.zeros((Bcur, self._nmax), dtype=DTYPE, device=F_tmp.device)
                    F_use[:, :Lcur] = F_tmp
                else:
                    H_use = H_tmp
                    F_use = F_tmp

                # Build Cartesian H and g
                H_cart, g_cart = self._build_cartesian_hg(F_use, H_use, real_mask)

                # Store exact Hessian and gradient
                self._H_work = H_cart.clone()
                self._g_cart_prev = g_cart.clone()
                # capture the initial batch-mean ||g|| once (adaptive-recalc baseline).
                if self._g0_norm_mean is None:
                    self._g0_norm_mean = float(g_cart.norm(dim=-1).mean().item())

                self._w(f"[Iter {outer_it}] Recalculated exact Hessian\n")

            else:
                # Non-recalc step: use working Hessian; get current EF.
                # Use working Hessian; gradient from the committed geometry.
                H_cart = self._H_work
                # FIX #1: this branch is reached ONLY when not (recalc|forced|refill)
                # (need_recalc is False), so the committed geometry equals the one
                # FORWARD-B already evaluated at the end of iter N-1 -> reuse its E/g
                # (block-diagonal forward => bit-identical) instead of FORWARD-A.
                if self._reuse_forward and (self._reuse_E is not None) \
                        and (self._reuse_E.shape[0] == real_mask.shape[0]):
                    E_old = self._reuse_E
                    g_cart = self._g_cart_prev
                    self._fwd_reused += 1
                    if self._audit_reuse:
                        self._audit_forward(calc, E_old, g_cart, real_mask)
                else:
                    E_old, F_now = self._ef(calc)
                    E_old = E_old.to(dtype=DTYPE)
                    F_now = F_now.to(dtype=DTYPE)
                    g_cart = -F_now * real_mask.to(DTYPE)

            # Mass-weighting and eigen-decomposition
            H_mw, g_mw = self._mass_weight_hg(H_cart, g_cart, real_mask)
            w, V, gp = self._eigh_and_track_modes(
                H_mw, g_mw, it=outer_it, calc=calc, g_cart=g_cart, real_mask=real_mask)

            # [7] D2H-sync reduction: sample convergence + verbose logging only every K
            # iters (and on the last outer iter). Default K=1 -> every iter == legacy.
            do_check = ((outer_it % self.conv_check_interval == 0)
                        or (outer_it == self.max_outer_iter))

            # Inner RS-PRFO loop. FIX #2: it now ALSO returns the committed-geometry
            # E/g it already evaluated (the accepted trial forward at line ~1204 is at
            # base+s_cart == the committed geometry; the legacy code dropped its forces
            # via `_` and FORWARD-B below recomputed them at the identical geometry).
            trust_r, last_step, last_rho, step_accepted, E_committed, g_committed = \
                self._inner_rs_prfo_loop(
                    it=outer_it,
                    calc=calc,
                    w=w, V=V, gp=gp,
                    H=H_cart, g_cart=g_cart,
                    trust_r=trust_r,
                    last_step=last_step,
                    real_mask=real_mask,
                    E_old=E_old,
                    log=do_check,
                )

            # After step: new E/gradient at the committed geometry. FIX #2: reuse the
            # inner loop's committed E/g (block-diagonal forward => bit-identical to a
            # fresh forward at the same coords) instead of FORWARD-B. The convergence
            # check below already reuses these (E_precomp/g_precomp). On reuse-OFF the
            # legacy real forward (FORWARD-B) runs -- the byte-parity oracle.
            if self._reuse_forward:
                E_fin = E_committed
                g_new_cart = g_committed
                self._fwd_reused += 1
                if self._audit_reuse:
                    self._audit_forward(calc, E_fin, g_new_cart, real_mask)
            else:
                E_fin, F_fin = self._ef(calc)
                F_fin = F_fin.to(dtype=DTYPE)
                g_new_cart = -F_fin * real_mask.to(DTYPE)
                E_fin = E_fin.to(dtype=DTYPE)

            # Thread the committed E forward for FIX #1 (FORWARD-A of the next iter).
            self._reuse_E = E_fin

            # Bofill update: apply ONLY if we didn't just recalculate AND at least one step was accepted
            if not need_recalc and step_accepted.any():
                self._w(
                    f"[Iter {outer_it}] Applying {self.hessian_update.upper()} "
                    f"update to {step_accepted.sum().item()} batches\n"
                )
                update_fn = (
                    self._bfgs_update_batched
                    if self.hessian_update == "bfgs"
                    else self._bofill_update_batched
                )
                self._H_work = update_fn(
                    H=self._H_work,
                    s_cart=last_step,
                    g_prev=self._g_cart_prev,
                    g_new=g_new_cart,
                    real_mask=real_mask,
                    step_accepted=step_accepted
                )

            # Always update gradient buffer for next iteration
            self._g_cart_prev = g_new_cart.clone()

            # Convergence check (reuse committed-geometry E/g instead of recomputing).
            # [7] Only evaluated on sampled iters; on skipped iters done=all-False (no shrink,
            # no per-B device->host table sync). Default K=1 -> evaluated every iter (legacy).
            if do_check:
                done = self._check_convergence(
                    it=outer_it,
                    calc=calc,
                    atoms_list=atoms_list,
                    trust_r=trust_r,
                    last_step=last_step,
                    real_mask=real_mask,
                    last_rho=last_rho,
                    E_precomp=E_fin,
                    g_precomp=g_new_cart,
                )
            else:
                done = torch.zeros(real_mask.shape[0], dtype=torch.bool, device=device)

            # ---- Outer/inner robustness: per-structure straggler detection ----
            # rho-bad streak (oscillation / poor model) + trust pinned-at-min streak.
            bad = (~torch.isfinite(last_rho)) | (last_rho < self.eta_shrink)
            zero_l = torch.zeros_like(self._bad_streak)
            self._bad_streak = torch.where(bad, self._bad_streak + 1, zero_l)
            pinned = trust_r <= (self.trust_min * 1.000000000001)
            self._pin_streak = torch.where(pinned, self._pin_streak + 1, zero_l)

            stuck = ((self._bad_streak >= self._bad_streak_limit)
                     & (self._pin_streak >= self._pin_streak_limit)
                     & (~done))
            restartable = stuck & (self._restart_count < self._max_restarts)
            evict = stuck & (self._restart_count >= self._max_restarts)

            if bool(restartable.any()):
                # Bounded inner restart: kick trust back up + force a fresh exact Hessian
                # so the structure escapes the pinned/oscillating basin (does NOT touch others).
                trust_r = torch.where(restartable,
                                      torch.full_like(trust_r, self.trust_init), trust_r)
                self._bad_streak = torch.where(restartable, zero_l, self._bad_streak)
                self._pin_streak = torch.where(restartable, zero_l, self._pin_streak)
                self._restart_count = self._restart_count + restartable.to(self._restart_count.dtype)
                self._force_recalc_next = True
                self._w(f"[Iter {outer_it}] Restart {int(restartable.sum().item())} straggler(s): "
                        f"trust->{self.trust_init}, force fresh Hessian\n")

            # Evicted stragglers leave the batch flagged (NOT silently wrong).
            final_done = done | evict
            if bool(evict.any()):
                self._w(f"[Iter {outer_it}] Evicting {int(evict.sum().item())} unrecoverable "
                        f"straggler(s) after {self._max_restarts} restarts\n")

            # Record per-original-index exit status before slicing.
            leaving = final_done.nonzero(as_tuple=False).flatten().cpu().tolist()
            done_cpu = done.detach().cpu()
            for i_local in leaving:
                oi = int(self._orig_index[i_local].item())
                self._final_status[oi] = "converged" if bool(done_cpu[i_local]) else "evicted_straggler"

            # Dynamic batch shrinking (on-GPU mask drives the shrink).
            survive_local = (~final_done).nonzero(as_tuple=False).flatten()

            # Streaming pool: after the shrink, the active batch can be refilled from
            # `self._pool_queue` back up to `self._B_target` (keeps the GPU saturated).
            # Pooling disabled (queue/target None) => want_refill is always False =>
            # this whole block reduces EXACTLY to the original shrink-only behavior.
            pooling = (self._pool_queue is not None and self._B_target is not None)
            n_room = (self._B_target - int(survive_local.numel())) if pooling else 0
            want_refill = pooling and (n_room > 0) and (len(self._pool_queue) > 0)

            if (survive_local.numel() < final_done.numel()) or want_refill:
                # Commit current geometries back into the (pre-slice) atoms objects so
                # survivors keep their optimized coords across the prepare() rebuild.
                self._sync_atoms_from_calc(calc, atoms_list)

                # ---- Slice survivors (every per-structure state in lockstep) ----
                atoms_list = [atoms_list[i] for i in survive_local.cpu().tolist()]
                self._orig_index = self._orig_index[survive_local]

                trust_r = trust_r[survive_local]
                last_step = last_step[survive_local]

                if self.tracked_mode_idx is not None:
                    self.tracked_mode_idx = self.tracked_mode_idx[survive_local]
                if self.tracked_mode_vec_mw is not None:
                    self.tracked_mode_vec_mw = self.tracked_mode_vec_mw[survive_local]

                # Slice per-structure robustness counters in lockstep.
                self._iter_count = self._iter_count[survive_local]
                self._recalc_count = self._recalc_count[survive_local]
                self._bad_streak = self._bad_streak[survive_local]
                self._pin_streak = self._pin_streak[survive_local]
                self._restart_count = self._restart_count[survive_local]

                # Slice working buffers in lockstep.
                if self._H_work is not None:
                    self._H_work = self._H_work[survive_local]
                if self._g_cart_prev is not None:
                    self._g_cart_prev = self._g_cart_prev[survive_local]
                # FIX #1: keep the threaded committed-E aligned with the shrunk batch
                # (survivors' geometry is unchanged, so the sliced value stays valid
                # for the next FORWARD-A). A refill below sets _force_recalc_next=True,
                # so the post-refill iter does a real forward anyway (no stale reuse).
                if self._reuse_E is not None:
                    self._reuse_E = self._reuse_E[survive_local]

                # ---- Streaming pool refill (INVERSE of the shrink above) ----
                # Pull up to (B_target - active) fresh guesses and torch.cat new rows
                # onto EVERY per-structure tensor so the batch stays in lockstep.
                if want_refill:
                    k_want = self._B_target - len(atoms_list)
                    new_atoms = []
                    while len(new_atoms) < k_want and self._pool_queue:
                        cand = self._pool_queue.pop(0)
                        if 3 * len(cand) > self._nmax:
                            # Cannot fit the fixed padded layout (nmax fixed at startup).
                            self._w(f"[Iter {outer_it}] pool: skip guess with "
                                    f"{len(cand)} atoms (> nmax_atoms {self._nmax // 3})\n")
                            continue
                        new_atoms.append(cand)
                    k = len(new_atoms)
                    if k > 0:
                        # Fresh ORIGINAL indices for the newcomers (unique, monotonic).
                        new_oi = list(range(self._next_orig, self._next_orig + k))
                        self._next_orig += k
                        self._pool_refilled += k

                        # Extend python-side per-ORIGINAL-index bookkeeping + xyz sinks.
                        for j, at in enumerate(new_atoms):
                            oi = new_oi[j]
                            self._symbols_per_batch.append(at.get_chemical_symbols())
                            self._final_status.append(None)
                            p = os.path.join(self.out_dir, f"ts_batch{oi + 1}.xyz")
                            self.xyz_paths.append(p)
                            self.frame_counts.append(0)
                            open(p, "w").close()

                        # Extend the active set + its ORIGINAL-index map.
                        atoms_list = atoms_list + new_atoms
                        self._orig_index = torch.cat([
                            self._orig_index,
                            torch.tensor(new_oi, dtype=torch.long, device=device),
                        ])

                        # New per-structure (B,) state: trust=trust_init, last_step=0,
                        # all robustness counters = 0.
                        trust_r = torch.cat([
                            trust_r,
                            torch.full((k,), self.trust_init, dtype=DTYPE, device=device),
                        ])
                        last_step = torch.cat([
                            last_step,
                            torch.zeros((k, self._nmax), dtype=DTYPE, device=device),
                        ])
                        # One zero tensor feeds all five cats (torch.cat copies into
                        # fresh storage and never mutates its inputs -> no clones).
                        zl = torch.zeros((k,), dtype=torch.long, device=device)
                        self._iter_count = torch.cat([self._iter_count, zl])
                        self._recalc_count = torch.cat([self._recalc_count, zl])
                        self._bad_streak = torch.cat([self._bad_streak, zl])
                        self._pin_streak = torch.cat([self._pin_streak, zl])
                        self._restart_count = torch.cat([self._restart_count, zl])

                        # Mode tracking: seed newcomers with a ZERO reference vector so the
                        # next _eigh_and_track_modes (overlap branch) auto-selects eigen-
                        # mode 0 (the lowest/most-negative mode after ascending eigh) = the
                        # TS reaction coordinate. No special-casing needed.
                        if self.tracked_mode_idx is not None:
                            self.tracked_mode_idx = torch.cat([
                                self.tracked_mode_idx,
                                torch.zeros((k,), dtype=torch.long, device=device),
                            ])
                        if self.tracked_mode_vec_mw is not None:
                            self.tracked_mode_vec_mw = torch.cat([
                                self.tracked_mode_vec_mw,
                                torch.zeros((k, self._nmax), dtype=DTYPE, device=device),
                            ])

                        # New working Hessian seed, prev grad = 0 (like the startup init);
                        # the RecalcFC schedule replaces it with the exact numerical Hessian
                        # within `self.recalc` iterations. [4][5] When a model seed / curvature
                        # injection is requested, build a Lindh / inject seed for the newcomers
                        # (fixes the wave-1 degenerate-identity-spectrum issue); otherwise keep
                        # the legacy identity seed verbatim (parity).
                        if self._H_work is not None:
                            if self.initial_hessian == "lindh" or self.ts_hessian_inject:
                                Lnew = torch.tensor(
                                    [3 * len(a) for a in new_atoms],
                                    dtype=torch.long, device=device)
                                mask_new = (self._arange_n[None, :] < Lnew[:, None])
                                H_new_seed = self._build_seed_hessian(new_atoms, mask_new)
                                self._H_work = torch.cat([self._H_work, H_new_seed])
                            else:
                                # cat copies into fresh storage, so the 0-stride expand
                                # is fine here (no .contiguous() materialization needed).
                                eye = torch.eye(self._nmax, dtype=DTYPE, device=device)
                                self._H_work = torch.cat([
                                    self._H_work,
                                    eye.unsqueeze(0).expand(k, -1, -1),
                                ])
                        if self._g_cart_prev is not None:
                            self._g_cart_prev = torch.cat([
                                self._g_cart_prev,
                                torch.zeros((k, self._nmax), dtype=DTYPE, device=device),
                            ])

                        self._w(f"[Iter {outer_it}] pool refill +{k} "
                                f"(active={len(atoms_list)}, queue_left="
                                f"{len(self._pool_queue)})\n")

                        # D1-1: newcomers are seeded with an IDENTITY working
                        # Hessian and a ZERO previous-gradient placeholder. Force
                        # a full exact-Hessian recalc on the NEXT outer iteration
                        # so their first active step uses a real Hessian (correct
                        # TS reaction mode instead of the degenerate identity
                        # spectrum), and the need_recalc path skips the Bofill
                        # update + resets _g_cart_prev to the true exact gradient
                        # (a g_prev=0 Bofill would corrupt the curvature update).
                        self._force_recalc_next = True

                # ---- Rebuild calculator topology ONCE for the new active set ----
                calc.prepare(atoms_list, fixed_nmax=self._nmax)
                self._rebuild_topology(atoms_list)

                # Dump the init frame of any freshly refilled structures (trajectory
                # completeness; survivors already have their running trajectory).
                if want_refill and len(atoms_list) > survive_local.numel():
                    n_new = len(atoms_list) - int(survive_local.numel())
                    new_local = list(range(len(atoms_list) - n_new, len(atoms_list)))
                    self._dump_xyz_locals(calc, new_local, tag="refill_init")

        else:
            self._w("# Maximum iterations reached.\n")

        # Any structure still in the batch at loop exit hit the iteration cap.
        if self._orig_index is not None:
            for i_local in range(self._orig_index.numel()):
                oi = int(self._orig_index[i_local].item())
                if self._final_status[oi] is None:
                    self._final_status[oi] = "max_iter"
        n_conv = sum(1 for s in self._final_status if s == "converged")
        n_evict = sum(1 for s in self._final_status if s == "evicted_straggler")
        n_max = sum(1 for s in self._final_status if s == "max_iter")
        self._w(f"# Final status: converged={n_conv} evicted_straggler={n_evict} "
                f"max_iter={n_max} (mu_unbracketed={self._mu_unbracketed}, "
                f"pool_refilled={self._pool_refilled})\n")
        if self.hessian_mode == "iterative":
            self._w(f"# iterative: hvp_kind={self._hvp_kind} hvp_calls={self._hvp_calls} "
                    f"lanczos_calls={self._lanczos_calls} lanczos_steps={self._lanczos_steps} "
                    f"unconverged_ritz={self._lanczos_unconverged}\n")
        if self.mode_follow_guard:
            self._w(f"# mode_follow_guard: neg_num==0 flags={self._neg_guard_hits}\n")
        # FIX #1/#2 forward accounting: real forwards executed vs reuses eliminated.
        self._w(f"# forward_reuse: reuse_forward={self._reuse_forward} "
                f"fwd_calls={self._fwd_calls} fwd_reused={self._fwd_reused}\n")
        if self._audit_reuse:
            self._w(f"# forward_reuse_audit: max|dE|={self._audit_max_dE:.3e} Ha "
                    f"max|dg|={self._audit_max_dF:.3e} Ha/A\n")

        self._close_log()


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

        # real_mask padded to fixed nmax
        self._real_mask = (self._arange_n[None, :] < self._L_vec[:, None])
        mass = _masses_flat(atoms_list, self._nmax, device)
        self._D = 1.0 / torch.sqrt(torch.clamp(mass, min=1e-12))

        # Precompute per-structure convergence thresholds ONCE per topology
        # (mirror blbfgs). These only change when the batch shrinks/refills,
        # which re-calls this method, so rebuilding them from python lists every
        # _check_convergence (host->device copy + sync x4) was pure overhead.
        # TS-search defaults (looser |F| than minimization) are preserved.
        self._f_max_th = torch.tensor(
            [getattr(at, "f_max_th", 9.5e-3) for at in atoms_list],
            dtype=DTYPE, device=device)
        self._f_rms_th = torch.tensor(
            [getattr(at, "f_rms_th", 5e-3) for at in atoms_list],
            dtype=DTYPE, device=device)
        self._dp_max_th = torch.tensor(
            [getattr(at, "dp_max_th", 1.8e-3) for at in atoms_list],
            dtype=DTYPE, device=device)
        self._dp_rms_th = torch.tensor(
            [getattr(at, "dp_rms_th", 1.2e-3) for at in atoms_list],
            dtype=DTYPE, device=device)

    def _sync_atoms_from_calc(self, calc, atoms_list):
        with torch.no_grad():
            pos = _get_coord_gpu(calc).detach().cpu().numpy()
        ptr = self._ptr.detach().cpu().numpy()
        for i, at in enumerate(atoms_list):
            s, t = ptr[i], ptr[i+1]
            at.positions[:] = pos[s:t]

    # ===================================================
    # EFH with padding to fixed nmax
    # ===================================================

    @profile
    def _compute_efh(self, calc):
        """
        EFH must be padded to the fixed nmax from the first iteration.
        """
        E_old, F_raw, H_raw, _ = calc.get_efh_gpu()

        F_raw = F_raw.to(dtype=DTYPE)
        H_raw = 0.5 * (H_raw + H_raw.transpose(-1, -2)).to(dtype=DTYPE)

        nmax = int(self._nmax)
        B, L = F_raw.shape

        if L != nmax:
            F_pad = torch.zeros((B, nmax), dtype=DTYPE, device=F_raw.device)
            F_pad[:, :L] = F_raw
            F_raw = F_pad

            H_pad = torch.zeros((B, nmax, nmax), dtype=DTYPE, device=H_raw.device)
            H_pad[:, :L, :L] = H_raw
            H_raw = H_pad

        return E_old.to(dtype=DTYPE), F_raw, H_raw

    def _build_cartesian_hg(self, F_raw, H_raw, real_mask):
        mask_ij = (real_mask.unsqueeze(-1) & real_mask.unsqueeze(-2)).to(DTYPE)
        H = H_raw * mask_ij
        g_cart = -F_raw * real_mask.to(DTYPE)
        return H, g_cart

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
    # EIGEN + TRACKING
    # ===================================================

    def _eigh_and_track_modes(self, H_mw, g_mw, it=0, calc=None,
                              g_cart=None, real_mask=None):
        B, n = g_mw.shape

        w, V = torch.linalg.eigh(H_mw)

        # ---- [1] Iterative leftmost-eigenpair substitution (opt-in) ----
        # Rather than trust the (approximate model) Hessian's lowest modes, resolve the
        # iter_n_leftmost leftmost eigenpairs of the TRUE Hessian matrix-free (Lanczos on
        # batched Hessian-vector products) and overwrite the corresponding lowest model
        # eigenpairs. The exact eigvec(s) are then fed to the existing RS-P-RFO eigenvector-
        # following unchanged -- no full Hessian is built.
        # Sella: Hermes et al. JCTC 2019 (10.1021/acs.jctc.9b00869) +
        #        Hermes et al. JCTC 2022 (10.1021/acs.jctc.2c00395).
        if (self.hessian_mode == "iterative" and calc is not None
                and g_cart is not None and real_mask is not None):
            lam_lo, vec_lo = self._leftmost_eigpairs_mw(
                calc, g_mw, g_cart, real_mask, self.iter_n_leftmost)
            n_lo = int(lam_lo.shape[1])
            V = V.clone()
            w = w.clone()
            for p in range(n_lo):
                V[:, :, p] = vec_lo[:, :, p]
                w[:, p] = lam_lo[:, p]

        gp = (V.transpose(-1, -2) @ g_mw.unsqueeze(-1)).squeeze(-1)

        tiny = w.abs() < 1e-10
        w = torch.where(tiny, torch.sign(w) * 1e-10, w)

        # ---- [3] negative-eigenvalue ('imaginary subspace') bookkeeping ----
        # Banerjee et al. JPC 1985 (10.1021/j100247a015); Baker JCC 1986
        # (10.1002/jcc.540070402).
        neg = w < -1e-6
        neg_num = neg.sum(dim=1)

        if self.tracked_mode_vec_mw is None:
            neg_idx = torch.argmin(w, dim=1)
            has_neg = w.gather(1, neg_idx[:, None]).squeeze(1) < -1e-6
            alt_idx = torch.argmax(gp.abs(), dim=1)
            tracked_idx = torch.where(has_neg, neg_idx, alt_idx)
            if self.mode_follow_guard:
                self._flag_neg_guard(it, neg_num)

            self.tracked_mode_idx = tracked_idx
            # gather column tracked_idx[b] from V[b] -> (B, n); avoids python B-loop host sync
            self.tracked_mode_vec_mw = V.gather(
                2, tracked_idx.view(B, 1, 1).expand(B, n, 1)
            ).squeeze(2)

        else:
            overlap = torch.matmul(
                V.transpose(-1, -2),
                self.tracked_mode_vec_mw.unsqueeze(-1)
            ).squeeze(-1)

            if self.mode_follow_guard:
                # [3] Restrict the max-overlap mode selection to the negative (imaginary)
                # subspace wherever one exists; for a structure with NO negative mode, fall
                # back to the full-spectrum overlap and FLAG it (stricter than pysisyphus,
                # which has no neg_num==0 guard).
                ov = overlap.abs()
                masked_ov = torch.where(neg, ov, torch.full_like(ov, -1.0))
                idx_neg = torch.argmax(masked_ov, dim=-1)
                idx_full = torch.argmax(ov, dim=-1)
                no_neg = (neg_num == 0)
                idx = torch.where(no_neg, idx_full, idx_neg)
                self._flag_neg_guard(it, neg_num)
            else:
                idx = torch.argmax(overlap.abs(), dim=-1)

            signs = torch.sign(overlap.gather(1, idx[:, None]).squeeze(1))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)

            self.tracked_mode_idx = idx
            # gather column idx[b] from V[b] -> (B, n), then sign-fix; no python B-loop
            vec = V.gather(2, idx.view(B, 1, 1).expand(B, n, 1)).squeeze(2)
            self.tracked_mode_vec_mw = vec * signs.unsqueeze(1)

        return w, V, gp

    def _flag_neg_guard(self, it, neg_num):
        """[3] Count + log structures with NO negative Hessian eigenvalue (not yet in a
        saddle region). Opt-in (mode_follow_guard); the .item() sync is gated by that flag."""
        no_neg = (neg_num == 0)
        c = int(no_neg.sum().item())
        if c > 0:
            self._neg_guard_hits += c
            self._w(f"[Iter {it}] mode_follow_guard: {c} structure(s) with NO negative "
                    f"Hessian eigenvalue (not a saddle region yet)\n")

    # ===================================================
    # ITERATIVE LEFTMOST-EIGENPAIR SOLVER (matrix-free, Sella) + SEED HESSIAN
    # ===================================================

    def _build_seed_hessian(self, atoms_list, real_mask):
        """[4][5] Cartesian seed Hessian for `atoms_list` (padded to nmax). Identity (legacy
        default) or Lindh model (10.1016/0009-2614(95)00646-L), optionally with ts_hessian
        curvature injection (10.1002/qua.21049) so the seed has EXACTLY one negative
        mass-weighted eigenvalue. Self-contained (derives its own mass-weighting) so it serves
        both the startup batch and pool-newcomer sub-batches before topology rebuild."""
        B = len(atoms_list)
        n = self._nmax
        device = self.device
        if self.initial_hessian == "lindh":
            H = lindh_initial_hessian(atoms_list, n, device, DTYPE)
        else:
            eye = torch.eye(n, dtype=DTYPE, device=device)
            H = eye.unsqueeze(0).expand(B, n, n).contiguous()
        mask_ij = (real_mask.unsqueeze(-1) & real_mask.unsqueeze(-2)).to(DTYPE)
        H = H * mask_ij
        if self.ts_hessian_inject:
            mass = _masses_flat(atoms_list, n, device)
            D = 1.0 / torch.sqrt(torch.clamp(mass, min=1e-12))
            H = ts_hessian_inject(H, real_mask, D, scale=self.ts_inject_scale, big=BIG)
        return H

    @torch.no_grad()
    def _hvp_cart(self, calc, u_cart, g0_cart, real_mask):
        """Cartesian Hessian-vector product H @ u (Hartree/Angstrom). Prefers an analytic
        calculator HVP (``calc.hvp`` / ``calc.hvp_batch``) when exposed; otherwise a forward
        finite-difference HVP  H@u = ||u|| * [g(x+eta*uhat) - g(x)] / eta  (ONE extra forward).
        ``g0_cart`` is the gradient (-F) at the current geometry. The base geometry is saved
        and restored in-place WITHOUT touching the calculator's own backup buffer."""
        rm = real_mask.to(DTYPE)
        u = u_cart * rm
        if self._hvp_kind is None:
            # DEFAULT 'fd' (UMA double-backward is unreliable -> FD is the accurate HVP).
            if self.hvp_source == "fd":
                self._hvp_kind = "fd"
            else:  # 'auto' or 'autograd': prefer an advertised analytic HVP, else FD
                if hasattr(calc, "hvp"):
                    self._hvp_kind = "hvp"
                elif hasattr(calc, "hvp_batch"):
                    self._hvp_kind = "hvp_batch"
                elif self.hvp_source == "autograd":
                    self._w("# [hvp] hvp_source='autograd' but calc exposes no hvp/hvp_batch; "
                            "using FD\n")
                    self._hvp_kind = "fd"
                else:
                    self._hvp_kind = "fd"
        self._hvp_calls += 1
        if self._hvp_kind in ("hvp", "hvp_batch"):
            try:
                out = (calc.hvp(u) if self._hvp_kind == "hvp" else calc.hvp_batch(u))
                Hu = out[0] if isinstance(out, (tuple, list)) else out
                return Hu.to(dtype=DTYPE) * rm
            except Exception as exc:  # permanent fallback to FD on any analytic failure
                self._w(f"# [hvp] analytic HVP failed ({exc!r}); falling back to FD\n")
                self._hvp_kind = "fd"
        # finite-difference HVP (linear in u -> scale a unit-step displacement back up)
        un = u.norm(dim=-1, keepdim=True)
        uhat = u / torch.clamp(un, min=1e-30)
        eta = self.iter_fd_eta
        d = eta * uhat
        coord = _get_coord_gpu(calc)
        save = coord.clone()
        calc.step_cart_(d)
        _, F_plus = calc.get_ef_gpu()
        coord.copy_(save)
        g_plus = -F_plus.to(dtype=DTYPE) * rm
        Hu = ((g_plus - g0_cart) / eta) * un
        return Hu * rm

    @torch.no_grad()
    def _leftmost_eigpairs_mw(self, calc, g_mw, g_cart, real_mask, n_lo):
        """[1] The ``n_lo`` leftmost (most-negative) eigenpairs of the mass-weighted TRUE
        Hessian, via batched Lanczos on matrix-free HVPs (Sella). The operator is confined to
        the real (non-pad) subspace -- mirroring the +BIG pad penalty of the full-eigh path,
        pad modes are never leftmost, so they are simply excluded. Full reorthogonalization
        (twice) keeps the small Krylov basis orthonormal; small m x m Gram (tridiagonal)
        eigh is batched over systems. Loose stop ||r|| < gamma*|lambda| (gamma~0.4).
        Returns (lam (B,n_lo) ascending, vec (B,n,n_lo) mass-weighted unit eigvecs)."""
        device = g_mw.device
        B, n = g_mw.shape
        rm = real_mask.to(DTYPE)
        n_lo = min(int(n_lo), n)
        m_min_eff = max(self.iter_lanczos_m_min, n_lo)
        m_max = min(self.iter_lanczos_m, n)
        m_max = max(m_max, min(n, n_lo + 2))
        gamma = self.iter_gamma

        def A_mw(v):  # mass-weighted Hessian action, (B,n)->(B,n)
            v = v * rm
            u_cart = self._D * v                       # D = 1/sqrt(m): mw-dir -> cart-dir
            Hu = self._hvp_cart(calc, u_cart, g_cart, real_mask)
            return (self._D * Hu) * rm

        gen = torch.Generator(device=device)
        gen.manual_seed(1234)
        v = torch.randn(B, n, generator=gen, dtype=DTYPE, device=device) * rm
        v = v / torch.clamp(v.norm(dim=-1, keepdim=True), min=1e-30)

        Vk = []
        alphas = []
        betas = []
        lam_out = None
        vec_out = None
        resid_lo = None
        tol = None
        for j in range(m_max):
            Vk.append(v)
            wv = A_mw(v)
            alpha = (wv * v).sum(dim=-1)
            alphas.append(alpha)
            for _pass in range(2):                     # full reorthogonalization x2
                for vk in Vk:
                    wv = wv - (wv * vk).sum(dim=-1, keepdim=True) * vk
            beta = wv.norm(dim=-1)
            m_used = j + 1

            if m_used >= 2:
                T = torch.diag_embed(torch.stack(alphas, dim=1))
                off = torch.stack(betas, dim=1)        # (B, m_used-1)
                ar = torch.arange(m_used - 1, device=device)
                T[:, ar, ar + 1] = off
                T[:, ar + 1, ar] = off
            else:
                T = torch.stack(alphas, dim=1).unsqueeze(-1)
            theta, S = torch.linalg.eigh(T)            # ascending
            Vk_stack = torch.stack(Vk, dim=1)          # (B, m_used, n)
            Y = torch.einsum("bjn,bjp->bnp", Vk_stack, S)   # (B, n, m_used)
            resid = beta.unsqueeze(1) * S[:, -1, :].abs()   # Ritz residual estimate (B,m_used)

            last = (j == m_max - 1)
            if (m_used >= n_lo) and ((m_used >= m_min_eff) or last):
                lam_out = theta[:, :n_lo]
                vec_out = Y[:, :, :n_lo]
                resid_lo = resid[:, :n_lo]
                tol = gamma * lam_out.abs().clamp(min=1e-8)
                if (m_used >= m_min_eff) and bool((resid_lo < tol).all()):
                    break
            if j < m_max - 1:
                v = wv / torch.clamp(beta, min=1e-30).unsqueeze(-1)
                v = v * rm
                betas.append(beta)

        self._lanczos_calls += 1
        self._lanczos_steps += len(Vk)
        if resid_lo is not None and tol is not None:
            self._lanczos_unconverged += int((resid_lo >= tol).sum().item())

        vec_out = vec_out * rm.unsqueeze(-1)
        vec_out = vec_out / vec_out.norm(dim=1, keepdim=True).clamp(min=1e-30)
        return lam_out, vec_out

    # ===================================================
    # FORWARD ACCOUNTING + REUSE AUDIT
    # ===================================================
    def _ef(self, calc):
        """calc.get_ef_gpu() with a forward counter (diagnostics / forward-count
        parity gate). Every REAL forward in the run loop goes through here."""
        self._fwd_calls += 1
        return calc.get_ef_gpu()

    @torch.no_grad()
    def _audit_forward(self, calc, E_reuse, g_reuse, real_mask):
        """Audit the legitimacy of a reused forward: do a FRESH forward at the
        current (supposedly unchanged) coords and accumulate max|reuse-fresh| for
        E and g. If the geometry truly did not move, this is at the UMA fp32 run-to-
        run noise floor (~1e-7), proving the reuse is exact up to calculator noise
        (NOT an algorithmic change). Enabled only by MAPLE_AUDIT_FWD_REUSE=1; it
        ADDS a forward so it is never on in production."""
        self._fwd_calls += 1
        E_fresh, F_fresh = calc.get_ef_gpu()
        g_fresh = -F_fresh.to(dtype=DTYPE) * real_mask.to(DTYPE)
        dE = float((E_reuse.to(DTYPE) - E_fresh.to(DTYPE)).abs().max().item())
        dF = float((g_reuse - g_fresh).abs().max().item())
        self._audit_max_dE = max(self._audit_max_dE, dE)
        self._audit_max_dF = max(self._audit_max_dF, dF)

    # ===================================================
    # INNER RS-PRFO LOOP
    # ===================================================
    @profile
    def _inner_rs_prfo_loop(self, it, calc, w, V, gp, H, g_cart,
                            trust_r, last_step, real_mask, E_old, log=True):
        """
        Optimized inner loop with ~1.5-2x speedup.

        ``log`` (default True == legacy) gates the per-attempt host-side iteration log +
        the acceptance-rate EMA .item() sync (Task 7 D2H-sync reduction). Step acceptance,
        trust update, and the actual optimization math are unchanged regardless of ``log``.
        """
        device = self.device
        B, n = gp.shape

        minus_mask = torch.nn.functional.one_hot(
            self.tracked_mode_idx, num_classes=n).to(torch.bool)
        plus_mask = ~minus_mask

        accepted = torch.zeros(B, dtype=torch.bool, device=device)
        step_accepted = torch.zeros(B, dtype=torch.bool, device=device)
        last_rho = torch.full((B,), float("nan"), dtype=DTYPE, device=device)

        # === FIX #2: committed-geometry E/g, threaded back to run() so it can skip
        # FORWARD-B. Init to the base point (E_old / g_cart at entry): a structure
        # that never accepts a step stays at base, so its committed E/g == base.
        # On each accept the row is overwritten with the accepted trial's E and
        # forces (the trial forward at base+s_cart IS the committed geometry).
        E_committed = E_old.clone()
        g_committed = g_cart.clone()

        # === OPTIMIZATION #4: Adaptive max attempts ===
        if self._recent_acceptance_rate > 0.7 and it > 5:
            max_attempts = max(10, self.max_inner_attempts // 2)
        else:
            max_attempts = self.max_inner_attempts

        attempts_used = 0
        for _try in range(max_attempts):
            attempts_used = _try + 1
            pend = ~accepted
            
            # === OPTIMIZATION #2: Early termination ===
            if not pend.any():
                break

            # Build unconstrained steps
            s_unc_minus = torch.zeros_like(gp)
            s_unc_plus = torch.zeros_like(gp)

            denom_m0 = -w.masked_select(minus_mask)
            denom_m0 = torch.where(denom_m0.abs() < 1e-10,
                                   torch.sign(denom_m0) * 1e-10, denom_m0)
            s_unc_minus[minus_mask] = -(-gp[minus_mask]) / denom_m0

            denom_p0 = w.masked_select(plus_mask)
            denom_p0 = torch.where(denom_p0.abs() < 1e-10,
                                   torch.sign(denom_p0) * 1e-10, denom_p0)
            s_unc_plus[plus_mask] = -(gp[plus_mask]) / denom_p0

            norm2_minus = (s_unc_minus ** 2).sum(-1)
            norm2_plus  = (s_unc_plus ** 2).sum(-1)
            total_unc   = norm2_minus + norm2_plus
            alpha = torch.where(
                total_unc > 0,
                norm2_minus / total_unc,
                torch.full_like(total_unc, 0.5)
            ).clamp(0.05, 0.95)

            R2 = trust_r ** 2
            R2_minus = alpha * R2
            R2_plus  = (1.0 - alpha) * R2

            # === Choose μ solver ===
            if self._enable_vectorized_mu:
                mu_minus, s_part_minus = self._solve_mu_vectorized(
                    w, gp, minus_mask, R2_minus, sigma=-1, only=pend
                )
                mu_plus, s_part_plus = self._solve_mu_vectorized(
                    w, gp, plus_mask, R2_plus, sigma=+1, only=pend
                )
            else:
                mu_minus, s_part_minus = self._solve_mu_batched(
                    w, gp, minus_mask, R2_minus, sigma=-1, only=pend
                )
                mu_plus, s_part_plus = self._solve_mu_batched(
                    w, gp, plus_mask, R2_plus, sigma=+1, only=pend
                )

            s_p = s_part_minus + s_part_plus
            norm_mw = torch.linalg.norm(s_p, dim=-1)
            s_mw = (V @ s_p.unsqueeze(-1)).squeeze(-1) * real_mask
            s_cart = self._D * s_mw

            # === OPTIMIZATION #5: Only compute for pending ===
            s_try = torch.zeros_like(s_cart)
            s_try[pend] = s_cart[pend]

            # Trial evaluation. FIX #2: KEEP the trial forces (the legacy code dropped
            # them via `_`); for an accepted structure base+s_cart IS the committed
            # geometry, so these forces == FORWARD-B's forces (block-diagonal) and let
            # run() skip FORWARD-B. The trial forward itself is unavoidable (rho test).
            calc.backup_coords()
            calc.step_cart_(s_try)
            self._fwd_calls += 1
            E_new, F_trial = calc.get_ef_gpu()
            E_new = E_new.to(dtype=DTYPE)
            F_trial = F_trial.to(dtype=DTYPE)
            calc.restore_coords()

            # Model change - could be further optimized
            Hs = torch.einsum("bij,bj->bi", H, s_try)
            model_change = (g_cart * s_try).sum(-1) + 0.5 * (s_try * Hs).sum(-1)
            actual_change = (E_new - E_old)

            rho = torch.full_like(model_change, float("nan"))
            ok = (model_change.abs() > 1e-16) & pend
            rho[ok] = actual_change[ok] / model_change[ok]
            last_rho = torch.where(pend, rho, last_rho)

            on_boundary = (norm_mw - trust_r).abs() <= (
                1e-6 * torch.clamp(trust_r, min=1.0)
            )
            force_accept = trust_r <= (self.trust_min * 1.000000000001)

            # ---- step acceptance (legacy vs [2] Nocedal-Wright) ----
            if self.trust_mode == "nw":
                # accept if rho > nw_eta_accept (or trust pinned at min).
                bad = pend & ((~torch.isfinite(rho)) | (rho < self.nw_eta_accept))
            else:
                bad = pend & ((~torch.isfinite(rho)) | (rho < self.eta_shrink))
            acc = pend & (~bad | force_accept)
            rej = pend & (~acc)

            if acc.any():
                s_commit = torch.zeros_like(s_cart)
                s_commit[acc] = s_cart[acc]
                calc.step_cart_(s_commit)
                last_step[acc] = s_commit[acc]
                step_accepted |= acc
                # FIX #2: the accepted rows now sit at base+s_cart == the geometry the
                # trial forward (above) evaluated -> record its E and g there so run()
                # reuses them instead of FORWARD-B (g = -F*mask, the run() convention).
                E_committed[acc] = E_new[acc]
                g_committed[acc] = (-F_trial * real_mask.to(DTYPE))[acc]
                self._dump_xyz_subset(calc, acc, it)

            # ---- trust-radius update ----
            if self.trust_mode == "nw":
                # [2] Nocedal & Wright, Numerical Optimization 2e (2006), Ch.4 Alg.4.1:
                #   rho < rho_lo                 -> trust x1/4 (floor trust_min)
                #   rho > rho_hi AND on boundary -> trust x2   (cap   trust_max)
                #   else                          -> keep
                shrink = pend & ((~torch.isfinite(rho)) | (rho < self.nw_rho_lo))
                trust_r = torch.where(
                    shrink, torch.clamp(0.25 * trust_r, min=self.trust_min), trust_r)
                grow = (pend & torch.isfinite(rho) & (rho > self.nw_rho_hi)
                        & on_boundary & (~shrink))
                trust_r = torch.where(
                    grow, torch.clamp(2.0 * trust_r, max=self.trust_max), trust_r)
            else:
                grow = (rho > self.eta_expand) & on_boundary & acc
                trust_r = torch.where(
                    grow, torch.clamp(2.0 * trust_r, max=self.trust_max), trust_r)
                trust_r = torch.where(
                    rej, torch.clamp(0.5 * trust_r, min=self.trust_min), trust_r)

            if log:
                self._w(self._fmt_iter_head(it, acc, rej, rho, trust_r, E_new))
            accepted |= acc

        # Update acceptance rate (exponential moving average). [7] gated by `log`; with the
        # default conv_check_interval=1 this fires every iter == legacy behavior.
        if log:
            acceptance_this_iter = accepted.float().mean().item()
            self._recent_acceptance_rate = (
                0.85 * self._recent_acceptance_rate + 0.15 * acceptance_this_iter)
            if attempts_used < max_attempts:
                self._w(f"  [Efficiency] Used {attempts_used}/{max_attempts} attempts\n")

        return trust_r, last_step, last_rho, step_accepted, E_committed, g_committed
    # ===================================================
    # CONVERGENCE
    # ===================================================

    def _check_convergence(self, it, calc, atoms_list,
                           trust_r, last_step, real_mask, last_rho,
                           E_precomp=None, g_precomp=None):

        device = self.device
        # Coalesce: the committed-geometry forward (run loop) already evaluated THIS
        # exact geometry. Reuse its E/g instead of recomputing a forward+backward.
        if E_precomp is not None and g_precomp is not None:
            E_final = E_precomp.to(dtype=DTYPE)
            g_last = g_precomp
        else:
            E_final, F_final = calc.get_ef_gpu()
            F_final = F_final.to(dtype=DTYPE)
            g_last = -F_final * real_mask.to(DTYPE)

        # Thresholds are built ONCE per topology in _rebuild_topology (mirror
        # blbfgs); reuse the cached (B,) tensors instead of rebuilding from
        # python lists every iteration.
        f_max_th = self._f_max_th
        f_rms_th = self._f_rms_th
        dp_max_th = self._dp_max_th
        dp_rms_th = self._dp_rms_th

        L_eff = self._L_vec.clamp(min=1).to(DTYPE)

        max_f = g_last.abs().amax(dim=-1)
        rms_f = torch.sqrt((g_last**2).sum(-1) / L_eff)
        max_dp = last_step.abs().amax(dim=-1)
        rms_dp = torch.sqrt((last_step**2).sum(-1) / L_eff)

        done = (
            (max_f <= f_max_th)
            & (rms_f <= f_rms_th)
            & (max_dp <= dp_max_th)
            & (rms_dp <= dp_rms_th)
        )

        self._w(self._fmt_orca_cycle_table(
            it=it,
            E=E_final.to(dtype=DTYPE),
            rho=last_rho,
            R=trust_r,
            max_f=max_f, rms_f=rms_f,
            max_dp=max_dp, rms_dp=rms_dp,
            f_max_th=f_max_th, f_rms_th=f_rms_th,
            dp_max_th=dp_max_th, dp_rms_th=dp_rms_th,
            done=done
        ))

        return done

    # ===================================================
    # MU SOLVER
    # ===================================================

    @torch.no_grad()
    def _solve_mu_vectorized(self, w, gp, mask, R2, sigma, only):
        """Fully vectorized RS trust-region shift solver (no python B-loop / no .item()).

        Parity target: _solve_mu_batched. For each structure b with only[b]:
          - subspace = mask[b] (minus_mask -> 1 uphill mode; plus_mask -> downhill+pad).
          - lam = sigma*w, num = sigma*gp.
          - Newton (mu=0) step s_unc = -num/lam (eval floor 1e-10). If ||s_unc||^2 <= R2
            -> mu=0 (unconstrained branch).
          - Else solve F(mu)=sum_sub (num/(lam-mu))^2 = R2 with mu < wt_min(subspace):
              * sigma<0 (single uphill mode): closed form mu = lam_s - |num_s|/sqrt(R2),
                clamped to <= lam_s-1e-6 (matches the batched hi when the root is < 1e-6
                from lam_s, the un-bracketed edge).
              * sigma>0 (downhill, many modes): lockstep bisection identical to the batched
                bracket(hi=wt_min-1e-6, expand lo while F>R2)+60-iter bisection, but all-B.
        Pad modes (eigenvalue ~ BIG) and the excluded uphill mode are MASKED OUT of the
        wt_min reduction (a naive (B,n) min collapses the bracket); they stay in the F sum
        where their num~0 contributes ~0 (parity-exact). evals_eps=1e-10, denom floor 1e-12
        replicated exactly. Returns (mu_out (B,), s_part (B,n)) in eigen-coordinates.
        """
        device = w.device
        B, n = w.shape

        lam = sigma * w
        num = sigma * gp
        sub = mask
        subf = sub.to(DTYPE)
        zeros_bn = torch.zeros((B, n), dtype=DTYPE, device=device)

        # --- unconstrained (mu=0) step, restricted to the subspace ---
        denom0 = torch.where(lam.abs() < 1e-10, torch.sign(lam) * 1e-10, lam)
        s_unc = torch.where(sub, -num / denom0, zeros_bn)
        norm2_unc = (s_unc * s_unc).sum(dim=1)
        is_unc = norm2_unc <= R2                       # (B,) bool

        def Fvec(mu):                                  # mu: (B,) -> (B,)
            denom = lam - mu.unsqueeze(1)
            denom = torch.where(denom.abs() < 1e-12, torch.sign(denom) * 1e-12, denom)
            val = (num / denom) ** 2
            return (val * subf).sum(dim=1)

        if sigma < 0:
            # uphill: exactly one mode per structure -> closed form
            num_s = (num * subf).sum(dim=1)            # the single tracked num
            lam_s = (lam * subf).sum(dim=1)            # the single tracked lam
            root = num_s.abs() / torch.sqrt(torch.clamp(R2, min=1e-300))
            mu_star = lam_s - root
            mu_star = torch.minimum(mu_star, lam_s - 1e-6)   # clamp to bracket upper bound
        else:
            # downhill: vectorized bracket + bisection
            pad_mode = w.abs() > (BIG * 0.5)
            min_mask = sub & (~pad_mode)
            BIG_SENT = torch.full_like(lam, 1e30)
            lam_for_min = torch.where(min_mask, lam, BIG_SENT)
            wt_min = lam_for_min.min(dim=1).values      # min real lam over subspace

            hi = wt_min - 1e-6
            Fhi = Fvec(hi)
            hi = torch.where(~torch.isfinite(Fhi), wt_min - 1e-4, hi)
            lo = hi - 1.0
            active = only & (~is_unc)

            # bracket expansion (lockstep, <=60), push lo down until F(lo) <= R2
            for _ in range(60):
                Flo = Fvec(lo)
                need = active & (Flo > R2)
                if not bool(need.any()):
                    break
                step = torch.clamp(lo.abs() * 0.5, min=1.0)
                lo = torch.where(need, lo - step, lo)

            # flag un-bracketed-after-60 (clamp, don't crash)
            unbr = active & (Fvec(lo) > R2)
            if bool(unbr.any()):
                self._mu_unbracketed += int(unbr.sum().item())

            # 60-iter lockstep bisection (F increasing in mu on mu < wt_min)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                go_hi = Fvec(mid) > R2
                hi = torch.where(active & go_hi, mid, hi)
                lo = torch.where(active & (~go_hi), mid, lo)
            mu_star = 0.5 * (lo + hi)

        # --- constrained step at mu_star, restricted to subspace ---
        denom = lam - mu_star.unsqueeze(1)
        denom = torch.where(denom.abs() < 1e-12, torch.sign(denom) * 1e-12, denom)
        s_con = torch.where(sub, -num / denom, zeros_bn)

        # select unconstrained vs constrained, then zero out non-`only` structures
        s_part = torch.where(is_unc.unsqueeze(1), s_unc, s_con)
        mu_out = torch.where(is_unc, torch.zeros_like(mu_star), mu_star)
        s_part = torch.where(only.unsqueeze(1), s_part, zeros_bn)
        mu_out = torch.where(only, mu_out, torch.zeros_like(mu_out))
        return mu_out, s_part

    @staticmethod
    @torch.no_grad()
    def _solve_mu_batched(w, gp, mask, R2, sigma, only):
        device = w.device
        B, n = w.shape

        lam_all = (sigma * w)
        num_all = (sigma * gp)

        mu_out = torch.zeros(B, dtype=DTYPE, device=device)
        s_part = torch.zeros((B, n), dtype=DTYPE, device=device)

        for b in range(B):
            if not only[b]:
                continue

            m_b = mask[b]
            lam_b = lam_all[b][m_b]
            num_b = num_all[b][m_b]

            if lam_b.numel() == 0:
                mu_out[b] = 0
                continue

            R2_b = float(R2[b].item())
            denom0 = torch.where(lam_b.abs() < 1e-10,
                                 torch.sign(lam_b) * 1e-10, lam_b)
            s_unc = -num_b / denom0
            norm2_unc = float((s_unc * s_unc).sum().item())

            if norm2_unc <= R2_b:
                mu_out[b] = 0.0
                s_tmp = s_unc
            else:
                def F(mu_val):
                    mu_t = torch.tensor(mu_val, dtype=DTYPE, device=device)
                    denom = lam_b - mu_t
                    denom = torch.where(
                        denom.abs() < 1e-12,
                        torch.sign(denom) * 1e-12,
                        denom
                    )
                    return float(((num_b / denom)**2).sum().item())

                wt_min = float(lam_b.min().item())
                hi = wt_min - 1e-6
                if not np.isfinite(F(hi)):
                    hi = wt_min - 1e-4

                lo = hi - 1
                Fa = F(lo)
                it_ex = 0
                while Fa > R2_b and it_ex < 60:
                    lo -= max(1.0, abs(lo) * 0.5)
                    Fa = F(lo)
                    it_ex += 1

                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    Fm = F(mid)
                    if abs(Fm - R2_b) <= 1e-12 * max(1, R2_b) or abs(hi - lo) < 1e-12:
                        lo = hi = mid
                        break
                    if Fm > R2_b:
                        hi = mid
                    else:
                        lo = mid

                mu_star = 0.5 * (lo + hi)
                mu_out[b] = mu_star
                mu_t = torch.tensor(mu_star, dtype=DTYPE, device=device)
                denom = lam_b - mu_t
                denom = torch.where(
                    denom.abs() < 1e-12,
                    torch.sign(denom) * 1e-12,
                    denom
                )
                s_tmp = -num_b / denom

            s_full = torch.zeros(n, dtype=DTYPE, device=device)
            s_full[m_b] = s_tmp
            s_part[b] = s_full

        return mu_out, s_part

    @staticmethod
    @torch.no_grad()
    def _bfgs_update_batched(
        H, s_cart, g_prev, g_new, real_mask,
        step_accepted=None,
        step_tol: float = 1e-8,
        grad_tol: float = 1e-8,
        curvature_tol: float = 1e-12,
    ):
        """Symmetric BFGS Hessian update for accepted batched PRFO steps."""
        DTYPE = H.dtype
        rm = real_mask.to(DTYPE)
        mask_ij = (real_mask.unsqueeze(-1) & real_mask.unsqueeze(-2)).to(DTYPE)

        s = s_cart.to(DTYPE) * rm
        y = (g_new - g_prev).to(DTYPE) * rm
        upd_mask = ((s * s).sum(-1) > step_tol**2) & ((y * y).sum(-1) > grad_tol**2)
        if step_accepted is not None:
            upd_mask = upd_mask & step_accepted
        if not bool(upd_mask.any()):
            return H

        H_new = H.clone()
        for b in upd_mask.nonzero(as_tuple=False).flatten().tolist():
            sb = s[b]
            yb = y[b]
            ys = torch.dot(yb, sb)
            if ys <= curvature_tol:
                continue
            Hs = H[b].matmul(sb)
            sHs = torch.dot(sb, Hs)
            if sHs <= curvature_tol:
                continue
            Hb = H[b] + torch.outer(yb, yb) / ys - torch.outer(Hs, Hs) / sHs
            Hb = 0.5 * (Hb + Hb.transpose(-1, -2))
            H_new[b] = Hb * mask_ij[b] + H_new[b] * (1.0 - mask_ij[b])
        return H_new

    @staticmethod
    @torch.no_grad()
    def _bofill_update_batched(
        H, s_cart, g_prev, g_new, real_mask,
        step_accepted=None,  # NEW: mask of which batches actually took a step
        step_tol: float = 1e-8, 
        grad_tol: float = 1e-8, 
        sr1_tol: float = 1e-8  # CHANGED: from 1e-10 to 1e-8
    ):
        """
        Bofill update (Cartesian, batched) with Gaussian-style logic:
        - Residual Z = dg - H * delta
        - phi = 1 - ( (dq^T Z)^2 / ( (dq^T dq) * (Z^T Z) ) )
        - H_{k+1} = H + (1-phi) * (Z Z^T) / (dq^T Z)   [MS/SR1 term]
                            +  phi * ( (Z dq^T + dq Z^T)/(dq^T dq) - (dq^T Z) * (dq dq^T)/(dq^T dq)^2 ) [PSB term]
        
        NEW: Only update batches that actually accepted a step (step_accepted mask)
        """
        DTYPE = H.dtype
        B, n, _ = H.shape

        rm = real_mask.to(DTYPE)
        mask_ij = (real_mask.unsqueeze(-1) & real_mask.unsqueeze(-2)).to(DTYPE)

        # Restrict vectors to real DOFs
        dq = s_cart.to(DTYPE) * rm                 # dq := delta (Cartesian)
        dg = (g_new - g_prev).to(DTYPE) * rm       # dg := grad change (Cartesian)

        # Per-batch norms to decide if we update
        dq2 = (dq * dq).sum(-1)                    # dq·dq
        dg2 = (dg * dg).sum(-1)                    # dg·dg
        
        # Update only if: (1) step was accepted, (2) non-trivial step/grad change
        upd_mask = (dq2 > step_tol**2) & (dg2 > grad_tol**2)
        if step_accepted is not None:
            upd_mask = upd_mask & step_accepted
        
        if not bool(upd_mask.any()):
            return H

        # Work on a copy; slice only the batches we will update
        H_new = H.clone()
        idx = upd_mask.nonzero(as_tuple=False).flatten()

        # Slice helpers
        dq_m = dq[idx]                 # (M, n)
        dg_m = dg[idx]                 # (M, n)
        HH   = H_new[idx]              # (M, n, n)

        # Z residual: Z = dg - H * dq
        Hdq  = torch.einsum("mij,mj->mi", HH, dq_m)
        Z    = dg_m - Hdq

        # Scalars
        dq2_m = (dq_m * dq_m).sum(-1)
        zz_m  = (Z * Z).sum(-1)
        qz_m  = (dq_m * Z).sum(-1)

        # ---- Build PSB increment ----
        Z_dqT = torch.einsum("mi,mj->mij", Z,    dq_m) * mask_ij[idx]
        dq_ZT = torch.einsum("mi,mj->mij", dq_m, Z   ) * mask_ij[idx]
        dq_dqT= torch.einsum("mi,mj->mij", dq_m, dq_m) * mask_ij[idx]
        dH_PSB = (Z_dqT + dq_ZT) / dq2_m.view(-1,1,1) - (qz_m / (dq2_m * dq2_m)).view(-1,1,1) * dq_dqT

        # ---- SR1/MS term ----
        use_sr1 = (qz_m.abs() > sr1_tol) & (zz_m > sr1_tol**2)
        dH_SR1_full = torch.zeros_like(dH_PSB)
        if bool(use_sr1.any()):
            Z_ZT = torch.einsum("mi,mj->mij", Z[use_sr1], Z[use_sr1]) * mask_ij[idx][use_sr1]
            dH_SR1 = Z_ZT / qz_m[use_sr1].view(-1,1,1)
            dH_SR1_full[use_sr1] = dH_SR1

        # ---- Bofill mixing weight ----
        phi = torch.ones_like(qz_m)
        good_phi = (dq2_m > sr1_tol**2) & (zz_m > sr1_tol**2)
        if bool(good_phi.any()):
            ratio = (qz_m[good_phi] * qz_m[good_phi]) / (dq2_m[good_phi] * zz_m[good_phi])
            phi_val = (1.0 - ratio).clamp(0.0, 1.0)
            phi[good_phi] = phi_val
        phi = torch.where(use_sr1, phi, torch.ones_like(phi))

        # ---- Combine increments ----
        inc = (1.0 - phi).view(-1,1,1) * dH_SR1_full + phi.view(-1,1,1) * dH_PSB

        # Apply and symmetrize
        HH = HH + inc
        HH = 0.5 * (HH + HH.transpose(-1, -2))
        H_new[idx] = HH

        return H_new


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
        self.log_fp.write(s)
        self.log_fp.flush()

    def _init_xyz_paths(self, B_all):
        self.xyz_paths = [
            os.path.join(self.out_dir, f"ts_batch{i+1}.xyz")
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
            s, t = ptr[i_local], ptr[i_local+1]
            idx_orig = int(self._orig_index[i_local].item())
            symbols = self._symbols_per_batch[idx_orig]
            self._append_xyz(idx_orig, symbols, pos[s:t], tag)

    def _dump_xyz_subset(self, calc, accept_mask, it):
        if not accept_mask.any():
            return
        with torch.no_grad():
            pos = _get_coord_gpu(calc).detach().cpu().numpy()
        ptr = self._ptr.detach().cpu().numpy()
        for i_local in accept_mask.nonzero(as_tuple=False).flatten().cpu().tolist():
            s, t = ptr[i_local], ptr[i_local+1]
            idx_orig = int(self._orig_index[i_local].item())
            symbols = self._symbols_per_batch[idx_orig]
            self._append_xyz(idx_orig, symbols, pos[s:t], f"iter={it}")

    def _dump_xyz_locals(self, calc, local_indices, tag=""):
        """Dump current geometry for an explicit list of LOCAL batch indices.

        Used to record the initial frame of structures freshly pulled into the
        active batch by the streaming pool (after calc.prepare/_rebuild_topology).
        """
        if not local_indices:
            return
        with torch.no_grad():
            pos = _get_coord_gpu(calc).detach().cpu().numpy()
        ptr = self._ptr.detach().cpu().numpy()
        for i_local in local_indices:
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
                except:
                    out.append("nan")
            return "[" + ", ".join(out) + "]"

        return (
            f"Iter {it}: accepted={acc_list} rejected={rej_list} "
            f"rho_head={head(rhos)} R_head={head(Rs)} E_head={head(En, fmt='{:.6f}')}\n"
        )

    def _fmt_orca_cycle_table(self, it, E, rho, R,
                              max_f, rms_f, max_dp, rms_dp,
                              f_max_th, f_rms_th, dp_max_th, dp_rms_th,
                              done):

        lines = []
        lines.append("-"*70 + "\n")
        lines.append(f"{('RS-PRFO Cycle ' + str(it)).center(70)}\n")
        lines.append("-"*70 + "\n")
        lines.append(
            "Batch   Energy(Ha)      rho      R(MW)   Max|F|(H/A)  RMS|F|  "
            "Max|dX|(A)  RMS|dX|  Conv\n"
        )
        lines.append("-"*70 + "\n")

        def fmt(v, wid=12, p=6):
            x = float(v)
            return f"{x:>{wid}.{p}f}"

        B = E.shape[0]
        for b in range(B):
            idx_orig = int(self._orig_index[b].item())
            lines.append(
                f"[{idx_orig:2d}] "
                f"{fmt(E[b], 14, 6)} "
                f"{fmt(rho[b], 8, 3)} "
                f"{fmt(R[b], 8, 3)} "
                f"{fmt(max_f[b], 12, 6)} "
                f"{fmt(rms_f[b], 10, 6)} "
                f"{fmt(max_dp[b], 12, 6)} "
                f"{fmt(rms_dp[b], 10, 6)} "
                f"{('YES' if bool(done[b]) else 'NO')}\n"
            )

        lines.append("\n")
        lines.append(" Convergence criteria:\n")
        lines.append(
            f"   Max|F| ≤ {float(f_max_th.max()):.6f}   "
            f"RMS|F| ≤ {float(f_rms_th.max()):.6f}   "
            f"Max|dX| ≤ {float(dp_max_th.max()):.6f}   "
            f"RMS|dX| ≤ {float(dp_rms_th.max()):.6f}\n"
        )

        return "".join(lines)