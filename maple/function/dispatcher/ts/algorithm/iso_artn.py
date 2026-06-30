# -*- coding: utf-8 -*-
"""Batched iso-ARTn -- single-ended (reactant-only) saddle search on the GPU.

NEW MAPLE capability: find a first-order transition state starting from ONE
minimum (no product / no near-saddle guess needed). The dimer / P-RFO modules
need a guess already inside the saddle's negative-curvature region; NEB / GSM
need both reactant AND product. iso-ARTn needs only the reactant minimum + an
initial activation direction, so it is genuinely single-ended.

Method
------
iso-ARTn (modified Activation-Relaxation Technique nouveau), Kim, Kim & Han,
"Modified Activation-Relaxation Technique (ARTn) Method Tuned for Efficient
Identification of Transition States in Surface Reactions", J. Chem. Theory
Comput. 2024, 20 (18), 8024-8034. DOI: 10.1021/acs.jctc.4c00767.
Background ARTn: Barkema & Mousseau, Phys. Rev. Lett. 1996, 77, 4358; Malek &
Mousseau, Phys. Rev. E 2000, 62, 7723. pARTn: Poberznik et al., Comput. Phys.
Commun. 2023, 295, 108961 (10.1016/j.cpc.2023.108961). Minimum-mode following:
Henkelman & Jonsson dimer, J. Chem. Phys. 1999, 111, 7010 (10.1063/1.480097).

The protocol has two phases per structure:

  (1) ACTIVATION -- leave the harmonic basin. From the minimum we climb along an
      ACTIVATION direction (the lowest-curvature / soft mode, tracked by the same
      batched dimer rotation used in convergence) by a fixed increment each step,
      WHILE relaxing the structure on the ORTHOGONAL HYPERPLANE perpendicular to
      that direction (iso-ARTn's "constraints on an orthogonal hyperplane"). The
      activation step for a system is

          dx = push_step * N_hat  -  activate_relax_alpha * grad_perp
             = push_step * N_hat  +  activate_relax_alpha * F_perp ,
          F_perp = F - (F . N_hat) N_hat ,

      i.e. push UP the soft mode, relax everything orthogonal to it. Activation
      ENDS for a structure the instant its lowest Hessian eigenvalue (curvature
      along the min mode, C = N^T H N, obtained by the batched FD-HVP) goes
      NEGATIVE -- the system has crossed the inflection ridge into the saddle's
      convex-down region. This is the iso-ARTn activation criterion.

  (2) CONVERGENCE -- minimum-mode following to the first-order saddle. Once a
      negative mode exists this is EXACTLY MAPLE's existing batched dimer
      machinery: rotate N onto the lowest mode (Heyden trig rotation, FD-HVP of
      the batched force field), INVERT the force component parallel to N, and do
      a trust-radius (or L-BFGS) step on F_trans = F_perp - F_par. Converged when
      the dimer's force + displacement criteria are met AND the curvature is
      negative (a real first-order saddle).

iso-ARTn's "adaptive active volume" (restrict the moving region to a sphere
around the most-displaced atom) is implemented as an OPTIONAL static active-atom
mask (``active_volume_radius``); the default (None) activates the whole system,
which is appropriate for the small isolated molecules in ts1x.

Batching / reuse
----------------
``BatchIsoARTn`` SUBCLASSES :class:`BatchDimer` (dimer.py) and REUSES, unchanged:
``_hvp`` (batched finite-difference H@N over the UMA force field -- the ONLY HVP
MAPLE exposes batched), ``_rotation_step`` (batched Heyden min-mode rotation),
``_mdot`` / ``_mnorm`` / ``_mnormalize`` / ``_remove_rigid`` (masked metric ops),
``_build_topology`` / ``_coord_padded`` / ``_atomview`` (padded (B, nmax) DOF
layout), the L-BFGS translation, and ``_record`` (write final geom/E/curv back by
original index). iso-ARTn is therefore almost pure PROTOCOL on top of the dimer:
B structures advance in lockstep, ONE batched ``get_ef_gpu`` + the rotation HVPs
per outer iteration serve all B, and a per-structure PHASE flag (activating vs
converging vs done) is masked exactly like ``BatchDimer``'s convergence mask /
``GSBatch``'s per-structure micro-cycle freezing.
"""
from __future__ import annotations

import os
import math
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

from .logger import log_info
from .dimer import BatchDimer, BatchDimerParams, DTYPE_BD, write_xyz


# ---- phase codes (per-structure, masked) ----
PH_ACTIVATE = 0
PH_CONVERGE = 1


@dataclass
class BatchIsoARTnParams(BatchDimerParams):
    """iso-ARTn params = all :class:`BatchDimerParams` (rotation / FD-HVP / trust /
    convergence thresholds, reused verbatim by the convergence phase) PLUS the
    activation-phase controls below."""
    # --- activation phase (climb the soft mode + orthogonal-hyperplane relax) ---
    push_step: float = 0.10            # Ang climb increment along the soft mode / iter
    activate_relax_alpha: float = 0.5  # SD factor for the orthogonal-hyperplane relax
    activate_max_iter: int = 120       # give up activating a system after this many iters
    curv_activate_th: float = -1.0e-3  # leave activation once C = N^T H N < this (<0)
    activate_mode_follow: bool = True  # climb the lowest-curvature mode (vs the fixed
                                       # initial direction); mode-following is what makes
                                       # the single-ended search actually reach a ridge.
    # --- adaptive active volume (optional; None => whole system, ts1x default) ---
    active_volume_radius: Optional[float] = None  # Ang sphere around the seed atom
    active_volume_seed: int = 0        # atom index used as the active-volume centre
    # iso-ARTn needs more outer iterations than a near-saddle dimer (it must first
    # walk out of the basin), so raise the default ceiling.
    max_iter: int = 400
    # single-ended search must NOT declare the starting minimum converged: require a
    # genuine negative curvature before the dimer force/disp criteria count. The gate
    # is a real (not numerically-zero) negative mode so the search does not "converge"
    # on a flat shoulder where C ~ -1e-5 (a higher-/zero-order stationary point).
    require_negative_curvature: bool = True
    conv_min_neg_curv: float = -1.0e-3   # converge only when C = N^T H N < this


class BatchIsoARTn(BatchDimer):
    """B independent single-ended iso-ARTn saddle searches in lockstep on the GPU.

    Usage (identical wiring to :class:`BatchDimer`, but the inputs are MINIMA /
    near-minima -- e.g. ts1x reactant geometries -- not near-saddle guesses)::

        m = Molecules(reactant_atoms_list); m.calc = UMABatchCalc(...)
        job = BatchIsoARTn(output="isoartn.out", device="cuda",
                           paras=dict(push_step=0.10, seed=1))
        job.run(m)
        job.final_status      # per-structure: 'converged' | 'no_activation' | 'max_iter'
        job.final_curvature   # per-structure final curvature (saddle => < 0)
        job.final_energy      # per-structure final energy (Hartree)
        # converged saddle geometries are written back into m.multiatoms in place
        # and to <output_base>_isoartn_ts_<i>.xyz

    The calculator contract is the one :class:`BatchDimer` already uses:
    ``prepare`` / ``get_ef_gpu`` / ``step_cart_`` / ``backup_coords`` /
    ``restore_coords`` + ``coord`` / ``_cols`` / ``_ptr`` / ``_n_b``.
    """

    def __init__(self, output: str, device: str = "cuda",
                 paras: Optional[dict] = None):
        # NOTE: skip BatchDimer.__init__ (it hard-codes BatchDimerParams); replicate
        # its body but parse into BatchIsoARTnParams so the activation knobs bind.
        from ...jobABC import JobABC
        JobABC.__init__(self, output)
        self.device = torch.device(device if torch.cuda.is_available()
                                   or device == "cpu" else "cpu")
        self.params = self._init_params(
            BatchIsoARTnParams, paras, ("isoartn", "iso_artn", "artn", "dimer", "ts"))
        # topology / state placeholders (filled in run(), mirror BatchDimer)
        self._B = 0
        self._nmax = 0
        self._A = 0
        self._real_mask = None
        self._atom_mask = None
        self._nat = None
        self._Leff = None
        self._lb_S = None
        self._lb_Y = None
        self._lb_rho = None
        self._lb_k = 0
        self._rot_kappa = None

    # ------------------------------------------------------------- active volume
    def _active_volume_mask(self, calc):
        """Optional iso-ARTn adaptive active volume: real-DOF mask restricted to a
        sphere of ``active_volume_radius`` around ``active_volume_seed`` (per
        structure). Returns (B, nmax) bool. ``None`` radius => the full real mask."""
        p = self.params
        if p.active_volume_radius is None:
            return self._real_mask
        pos = self._atomview(self._coord_padded(calc))            # (B, A, 3)
        seed = int(p.active_volume_seed)
        seed = max(0, min(seed, self._A - 1))
        c = pos[:, seed:seed + 1, :]                              # (B, 1, 3)
        d = torch.linalg.norm(pos - c, dim=-1)                   # (B, A)
        within = (d <= float(p.active_volume_radius)) & self._atom_mask
        return within.unsqueeze(-1).expand(self._B, self._A, 3).reshape(
            self._B, self._nmax) & self._real_mask

    # ------------------------------------------------------------------- run
    def run(self, mols):
        """Drive B single-ended iso-ARTn searches to their saddles in lockstep."""
        p = self.params
        device = self.device
        atoms_list = list(mols.multiatoms)
        calc = mols.calc
        B = len(atoms_list)
        if B == 0:
            return
        base, _ = os.path.splitext(self.output)

        # --- topology: ONE prepare() fixes nmax; build masks (reused helper) ---
        calc.prepare(atoms_list)
        _, F_probe = calc.get_ef_gpu()
        self._nmax = int(F_probe.shape[1])
        self._A = self._nmax // 3
        self._build_topology(calc, atoms_list)

        atoms_orig = atoms_list
        B0 = B
        self._orig_index = torch.arange(B0, dtype=torch.long, device=device)
        result_status = ["max_iter"] * B0
        result_E = [float("nan")] * B0
        result_C = [float("nan")] * B0

        f_max_th = torch.tensor([getattr(a, "f_max_th", p.f_max_th) for a in atoms_list],
                                dtype=DTYPE_BD, device=device)
        f_rms_th = torch.tensor([getattr(a, "f_rms_th", p.f_rms_th) for a in atoms_list],
                                dtype=DTYPE_BD, device=device)
        dp_max_th = torch.tensor([getattr(a, "dp_max_th", p.dp_max_th) for a in atoms_list],
                                 dtype=DTYPE_BD, device=device)
        dp_rms_th = torch.tensor([getattr(a, "dp_rms_th", p.dp_rms_th) for a in atoms_list],
                                 dtype=DTYPE_BD, device=device)

        # --- per-structure activation / dimer axis N (unit, rigid-body removed) ---
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
        N_prev = N.clone()                                  # for push-sign continuity

        # --- per-structure phase / optimizer state (NEVER shared) ---
        phase = torch.full((B,), PH_ACTIVATE, dtype=torch.long, device=device)
        act_iters = torch.zeros(B, dtype=torch.long, device=device)
        alpha = torch.full((B,), float(p.step0), dtype=DTYPE_BD, device=device)
        active = torch.ones(B, dtype=torch.bool, device=device)
        last_step = torch.zeros(B, self._nmax, dtype=DTYPE_BD, device=device)
        g_prev = None
        self._rot_kappa = torch.full((B,), 1.0, dtype=DTYPE_BD, device=device)
        if p.superlinear:
            self._lbfgs_reset()

        max_allow = min(p.trust_radius, p.step_max)
        total_fe = 0

        log_info([
            "\n========================================================================\n",
            "          BatchIsoARTn  (batched GPU single-ended saddle search)        \n",
            "========================================================================\n",
            f"B systems                : {B}\n",
            f"padded DOF (nmax)        : {self._nmax}\n",
            f"activation               : climb soft mode (push_step={p.push_step} Ang) + "
            f"orthogonal-hyperplane relax (alpha={p.activate_relax_alpha})\n",
            f"activation criterion     : C = N^T H N < {p.curv_activate_th} (lowest eigenvalue < 0)\n",
            f"convergence              : dimer min-mode following (REUSES BatchDimer FD-HVP + rotation)\n",
            f"HVP                      : {'central FD (2 eval)' if p.central_hvp else 'forward FD (1 eval, classic HJ)'}\n",
            f"active volume            : {'whole system' if p.active_volume_radius is None else str(p.active_volume_radius)+' Ang sphere'}\n",
            "Method: iso-ARTn, Kim/Kim/Han JCTC 2024, 20, 8024 (10.1021/acs.jctc.4c00767); "
            "min-mode: HJ dimer 1999 (10.1063/1.480097)\n",
            "------------------------------------------------------------------------\n",
        ], self.output)

        for it in range(1, p.max_iter + 1):
            # (0) midpoint energy + forces -- ONE batched forward, all systems
            E0, F0 = calc.get_ef_gpu()
            E0 = E0.to(DTYPE_BD); F0 = (F0.to(DTYPE_BD)) * self._real_mask
            total_fe += 1

            # (1) rotation: align N with the lowest-curvature mode + curvature C
            #     (REUSED BatchDimer batched FD-HVP min-mode rotation, all B at once).
            N, C, max_frot, nfe = self._rotation_step(calc, N, F0, active)
            total_fe += nfe

            # keep the push direction continuous across iters (rotation may flip N's
            # sign); flip N so it stays on the same side it was last iteration.
            sgn = torch.sign(self._mdot(N, N_prev))
            sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
            N = N * sgn.unsqueeze(-1)
            N_prev = N.clone()

            # (2) phase transition: an ACTIVATING system enters CONVERGENCE the moment
            #     its lowest curvature goes negative (iso-ARTn activation criterion).
            activating = active & (phase == PH_ACTIVATE)
            newly_conv = activating & (C < p.curv_activate_th)
            phase = torch.where(newly_conv, torch.full_like(phase, PH_CONVERGE), phase)

            # give up systems that never activate within activate_max_iter
            stuck = active & (phase == PH_ACTIVATE) & (act_iters >= p.activate_max_iter)
            for b in stuck.nonzero(as_tuple=False).flatten().cpu().tolist():
                result_status[int(self._orig_index[b].item())] = "no_activation"
            active = active & (~stuck)

            act_mask = active & (phase == PH_ACTIVATE)
            conv_mask = active & (phase == PH_CONVERGE)
            act_iters = act_iters + act_mask.long()

            # (3a) ACTIVATION step: push up the soft mode + relax orthogonal hyperplane
            Fpar = (self._mdot(F0, N)).unsqueeze(-1) * N
            Fperp = (F0 - Fpar) * self._real_mask
            avol = self._active_volume_mask(calc)               # (B, nmax) bool
            push = float(p.push_step) * N
            act_step = (push + float(p.activate_relax_alpha) * Fperp) * avol

            # (3b) CONVERGENCE step: dimer translation (invert F_par once C<0)
            flip = (C < p.kappa_to_flip)
            Ftrans = torch.where(flip.unsqueeze(-1), Fperp - Fpar, Fperp) * self._real_mask
            if p.superlinear:
                grad = -Ftrans
                if g_prev is not None and self._lb_k >= 0:
                    self._lbfgs_push(last_step, grad - g_prev)
                d = self._lbfgs_dir(grad) if self._lb_k > 0 else (alpha.unsqueeze(-1) * Ftrans)
                conv_step = d
                g_prev = grad
            else:
                conv_step = alpha.unsqueeze(-1) * Ftrans

            # (3c) select per-structure step by phase; freeze inactive
            step = torch.where(act_mask.unsqueeze(-1), act_step,
                               torch.where(conv_mask.unsqueeze(-1), conv_step,
                                           torch.zeros_like(act_step)))
            step = step * active.unsqueeze(-1) * self._real_mask

            # per-structure trust clamp (same as BatchDimer)
            step_atom = torch.linalg.norm(self._atomview(step), dim=-1)          # (B, A)
            max_step = (step_atom * self._atom_mask).amax(dim=-1)                # (B,)
            on_boundary = max_step > max_allow
            scale = torch.where(on_boundary, max_allow / max_step.clamp(min=1e-20),
                                torch.ones_like(max_step))
            step = step * scale.unsqueeze(-1)

            calc.step_cart_(step)

            # per-structure trust update (steepest-descent path, convergence only)
            if not p.superlinear:
                upd = on_boundary & conv_mask
                alpha = torch.where(upd, torch.clamp(0.5 * alpha, min=0.1 * p.step0),
                                    torch.where(conv_mask,
                                                torch.minimum(1.2 * alpha,
                                                              torch.full_like(alpha, p.step_max)),
                                                alpha))
            last_step = step

            # (4) convergence test -- ONLY for converging-phase systems (never declare
            #     the starting minimum converged) AND require a real negative curvature.
            f_atom = torch.linalg.norm(self._atomview(F0), dim=-1)
            max_f = (f_atom * self._atom_mask).amax(dim=-1)
            rms_f = torch.sqrt((f_atom ** 2 * self._atom_mask).sum(-1)
                               / self._atom_mask.sum(-1).clamp(min=1))
            dp_atom = step_atom
            max_dp = (dp_atom * self._atom_mask).amax(dim=-1)
            rms_dp = torch.sqrt((dp_atom ** 2 * self._atom_mask).sum(-1)
                                / self._atom_mask.sum(-1).clamp(min=1))
            neg_ok = (C < p.conv_min_neg_curv) if p.require_negative_curvature \
                     else torch.ones_like(C, dtype=torch.bool)
            conv = conv_mask & neg_ok & (max_f <= f_max_th) & (rms_f <= f_rms_th) & \
                   (max_dp <= dp_max_th) & (rms_dp <= dp_rms_th)
            for b in conv.nonzero(as_tuple=False).flatten().cpu().tolist():
                result_status[int(self._orig_index[b].item())] = "converged"
            active = active & (~conv)

            if (it <= 5) or (it % 10 == 0) or (not bool(active.any())):
                na = int(active.sum().item())
                n_act = int((active & (phase == PH_ACTIVATE)).sum().item())
                n_cnv = int((active & (phase == PH_CONVERGE)).sum().item())
                log_info([
                    f"[iter {it:4d}] active={na:3d}/{B} (activating={n_act} converging={n_cnv})  "
                    f"E[min/max]={float(E0.min()):.5f}/{float(E0.max()):.5f}  "
                    f"curv<0={int((C < 0).sum().item())}/{B}  "
                    f"max|F|={float(max_f.max()):.5f}  rot_fe={nfe} cum_fe={total_fe}\n"
                ], self.output)

            if not bool(active.any()):
                log_info([f"\nAll active systems resolved at iteration {it}.\n"], self.output)
                break

        # --- final: one batched forward + HVP, record EVERY system by orig index.
        #     Frozen systems never moved after they resolved, so their recorded geom
        #     is exactly their resolution geometry (reuse BatchDimer._record). ---
        E_final, F_final = calc.get_ef_gpu(); total_fe += 1
        E_final = E_final.to(DTYPE_BD)
        HN_final = self._hvp(calc, N, F_final.to(DTYPE_BD) * self._real_mask,
                             p.delta, p.central_hvp)
        total_fe += (2 if p.central_hvp else 1)
        C_final = self._mdot(N, HN_final)
        remaining = torch.ones(self._B, dtype=torch.bool, device=device)
        # rename the per-structure TS files to an isoartn-specific stem
        prev_save = self.params.save_traj
        self._record(atoms_orig, calc, remaining, E_final, C_final, result_E, result_C)
        if prev_save:
            for oi in range(B0):
                src = f"{base}_bd_ts_{oi}.xyz"
                dst = f"{base}_isoartn_ts_{oi}.xyz"
                if os.path.exists(src):
                    try:
                        os.replace(src, dst)
                    except Exception:
                        pass

        n_conv = sum(1 for s in result_status if s == "converged")
        n_noact = sum(1 for s in result_status if s == "no_activation")
        n_negcurv = sum(1 for c in result_C if c == c and c < 0.0)
        log_info([
            "\n------------------------------------------------------------------------\n",
            "                       BatchIsoARTn summary                             \n",
            "------------------------------------------------------------------------\n",
            f"converged (reached saddle): {n_conv}/{B0}\n",
            f"failed to activate        : {n_noact}/{B0}\n",
            f"negative final curvature  : {n_negcurv}/{B0}\n",
            f"total batched force evals : {total_fe}\n",
            f"per-structure status      : {result_status}\n",
            f"per-structure curvature   : {[round(c, 5) if c == c else None for c in result_C]}\n",
        ], self.output)

        self.final_curvature = np.array(result_C, dtype=float)
        self.final_status = result_status
        self.final_energy = np.array(result_E, dtype=float)
        self.total_force_evals = total_fe
        self.n_iter = it
        return
