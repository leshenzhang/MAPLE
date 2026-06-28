# -*- coding: utf-8 -*-
"""
Batch SDCG (Steepest-Descent + PRP+ Conjugate-Gradient) optimizer.

Batched, GPU-resident, GRADIENT-ONLY (uses calc.get_ef_gpu(); no Hessian) fusion
optimizer -- the BATCHED counterpart of the single-structure ``SDCG`` optimizer
(``SDCG.py``) with GDIIS disabled.  Each structure independently runs:

    Phase 1 (SD): step = clip(max_step * forces)
    Phase 2 (CG): PRP+ conjugate gradient, force-proportional step + clip
        beta = max(f_k . (f_k - f_{k-1}) / ||f_{k-1}||^2, 0)              (PRP+)
        Powell restart: if |f_k . f_{k-1}| >= cg_restart_threshold ||f_k||^2 -> beta=0
        descent guard: if d_k . f_k <= 0 -> reset d_k = f_k, beta=0
        step = clip(max_step * d_k); store d_{k} rescaled to the force magnitude

Per-structure phase transition (identical to SDCG.run): a structure switches
SD -> CG when its own  max|F| < cg_switch_threshold  (auto = 0.5 * initial max|F|)
OR after sd_max_iter SD iterations -- whichever comes first.  The phase, the
SD-iteration counter, the CG history (f_{k-1}, d_{k-1}) and the switch threshold
are all per-structure tensors.

Per-structure convergence masking + dynamic batch shrinking (eviction) and all
topology / padding / metric / logging / final-geometry machinery are reused from
the shared ``_BatchGradOpt`` base in ``batch_sd.py`` (itself reusing the
BatchLBFGS skeleton).  float64 algorithm math.

GDIIS, Barzilai-Borwein step-scale and trajectory-rejection (present in the
single SDCG) are intentionally omitted: BatchSDCG is gradient-only, so it is the
batched form of single SDCG run with ``diis_enabled=False`` -- the parity oracle.
"""

import torch

from .batch_sd import _BatchGradOpt, DTYPE


class BatchSDCG(_BatchGradOpt):
    """
    Batched SD+CG fusion optimizer (gradient-only, per-structure phase + eviction).

    method='sdcg' (default) -> SD phase then PRP+ CG phase (per structure)
    method='sd'             -> SD only       (cg_enabled=False)
    method='cg'             -> CG only       (sd_enabled=False)
    """
    _algo_name = "Batch-SDCG"

    def __init__(self,
                 output: str,
                 max_step: float = 0.2,
                 max_iter: int = 256,
                 method: str = "sdcg",
                 sd_max_iter: int = 50,
                 cg_switch_fmax: float = 0.0,
                 cg_restart_threshold: float = 0.2,
                 device: str = "cuda",
                 write_traj: bool = False,
                 traj_every: int = 1,
                 verbose: int = 1):
        super().__init__(output, maxstep=max_step, maxiter=max_iter, device=device,
                         write_traj=write_traj, traj_every=traj_every, verbose=verbose)
        m = str(method).lower()
        if m == "sd":
            self.sd_enabled, self.cg_enabled = True, False
        elif m == "cg":
            self.sd_enabled, self.cg_enabled = False, True
        else:                          # sdcg / default: both phases
            self.sd_enabled, self.cg_enabled = True, True
        self.method = m
        self.sd_max_iter = int(sd_max_iter)
        self.cg_switch_fmax = float(cg_switch_fmax)
        self.cg_restart_threshold = float(cg_restart_threshold)

        # per-structure state (allocated in _init_algo_state)
        self._phase_cg = None          # (B,) bool : True => CG phase
        self._sd_iter_count = None     # (B,) long
        self._cg_switch_th = None      # (B,) f64
        self._cg_init = None           # (B,) bool : CG history valid
        self._cg_prev_forces = None    # (B, nmax) f64
        self._cg_prev_direction = None # (B, nmax) f64

    # ===================================================
    # per-structure algorithm state
    # ===================================================
    def _init_algo_state(self, B, nmax, F0):
        device = self.device
        initial_max_f = F0.abs().amax(dim=-1)            # (B,) masked F0

        # phase: start SD if enabled, else CG
        self._phase_cg = torch.full((B,), not self.sd_enabled,
                                    dtype=torch.bool, device=device)

        # auto cg switch threshold = 0.5 * initial max|F| (only in true fusion);
        # an explicit cg_switch_fmax>0 overrides (mirrors SDCG.run auto logic).
        if self.cg_switch_fmax > 0.0:
            self._cg_switch_th = torch.full((B,), self.cg_switch_fmax,
                                            dtype=DTYPE, device=device)
        elif self.sd_enabled and self.cg_enabled:
            self._cg_switch_th = 0.5 * initial_max_f
        else:
            self._cg_switch_th = torch.zeros(B, dtype=DTYPE, device=device)

        self._sd_iter_count = torch.zeros(B, dtype=torch.long, device=device)
        self._cg_init = torch.zeros(B, dtype=torch.bool, device=device)
        self._cg_prev_forces = torch.zeros((B, nmax), dtype=DTYPE, device=device)
        self._cg_prev_direction = torch.zeros((B, nmax), dtype=DTYPE, device=device)

    def _pre_step(self, forces, iteration):
        """Per-structure SD -> CG phase switch (checked on current committed forces)."""
        if not self.cg_enabled:
            return
        max_f = forces.abs().amax(dim=-1)                # (B,)
        in_sd = ~self._phase_cg
        should = (self._sd_iter_count >= self.sd_max_iter) | \
                 ((self._cg_switch_th > 0) & (max_f < self._cg_switch_th))
        switch_now = in_sd & should
        if switch_now.any():
            self._phase_cg = self._phase_cg | switch_now
            # fresh CG start for the just-switched structures
            self._cg_init = self._cg_init & ~switch_now

    def _compute_direction(self, forces):
        f_curr = forces                                  # (B, nmax) masked
        if not self.cg_enabled:
            return f_curr                                # pure SD

        zeros = torch.zeros(forces.shape[0], dtype=DTYPE, device=self.device)
        f_prev = self._cg_prev_forces
        df = f_curr - f_prev
        denom = (f_prev * f_prev).sum(dim=-1)            # (B,)
        num = (f_curr * df).sum(dim=-1)
        safe_denom = torch.where(denom < 1e-20, torch.ones_like(denom), denom)
        beta = torch.where(denom < 1e-20, zeros, num / safe_denom)
        beta = beta.clamp(min=0.0)                       # PRP+ clamp

        # Powell restart
        fcfp = (f_curr * f_prev).sum(dim=-1).abs()
        fcfc = (f_curr * f_curr).sum(dim=-1)
        restart = fcfp >= self.cg_restart_threshold * fcfc
        beta = torch.where(restart, zeros, beta)

        direction_cg = f_curr + beta.unsqueeze(-1) * self._cg_prev_direction

        # descent guard: d . f <= 0 -> reset to SD direction
        nondescent = (direction_cg * f_curr).sum(dim=-1) <= 0
        direction_cg = torch.where(nondescent.unsqueeze(-1), f_curr, direction_cg)

        # use the real CG direction only where the structure is in CG phase AND has
        # a valid history; SD phase / first CG step -> steepest descent (forces).
        use_cg = (self._phase_cg & self._cg_init).unsqueeze(-1)
        return torch.where(use_cg, direction_cg, f_curr)

    def _post_step(self, forces, direction):
        if self.cg_enabled:
            # rescale stored direction to the force magnitude (anti-accumulation),
            # exactly as SDCG._cg_step:  prev_direction = direction * (f_max / d_max)
            f_max = forces.abs().amax(dim=-1)
            d_max = direction.abs().amax(dim=-1)
            both = (d_max > 0) & (f_max > 0)
            safe_dmax = torch.where(d_max > 0, d_max, torch.ones_like(d_max))
            scale = torch.where(both, f_max / safe_dmax, torch.ones_like(d_max))
            new_prev_dir = direction * scale.unsqueeze(-1)

            pc = self._phase_cg.unsqueeze(-1)
            self._cg_prev_direction = torch.where(pc, new_prev_dir, self._cg_prev_direction)
            self._cg_prev_forces = torch.where(pc, forces, self._cg_prev_forces)
            self._cg_init = self._cg_init | self._phase_cg

        # SD-iteration counter advances only for structures still in SD phase
        self._sd_iter_count = self._sd_iter_count + (~self._phase_cg).long()

    def _shrink_algo_state(self, survive_idx):
        self._phase_cg = self._phase_cg[survive_idx]
        self._sd_iter_count = self._sd_iter_count[survive_idx]
        self._cg_switch_th = self._cg_switch_th[survive_idx]
        self._cg_init = self._cg_init[survive_idx]
        self._cg_prev_forces = self._cg_prev_forces[survive_idx]
        self._cg_prev_direction = self._cg_prev_direction[survive_idx]
