# -*- coding: utf-8 -*-
"""
Batch steepest-descent (SD) optimizer with fixed padded dimension nmax.

Batched, GPU-resident, GRADIENT-ONLY geometry optimizer (uses calc.get_ef_gpu();
no Hessian).  This is the BATCHED counterpart of the single-structure SD optimizer
-- SD.py is a thin stub that maps to ``SDCG`` with ``method='sd'`` (sd_enabled=True,
cg_enabled=False, GDIIS off).  The update rule is exactly ``SDCG._sd_step``:

    step = max_step * forces            # forces = -dE/dx           [Ha/Angstrom]
    clip so max|step| <= max_step       # per structure

Per-structure convergence masking + eviction (dynamic batch shrinking) is reused
verbatim from the BatchLBFGS skeleton (``blbfgs.py``): nmax padding, ``_real_mask``
(padded valid-DOF mask), topology build, the per-iter ``calc.get_ef_gpu()`` loop,
the four batched convergence reductions and the per-structure threshold tensors.

Convergence (per structure, identical to ``_common.is_converged``):
    max|F| <= f_max_th AND rms|F| <= f_rms_th AND
    max|dX| <= dp_max_th AND rms|dX| <= dp_rms_th

float64 algorithm math (DTYPE).  The shared base ``_BatchGradOpt`` is defined here
and reused by ``batch_sdcg.BatchSDCG``; only the search-DIRECTION + per-structure
algorithm state differ between SD and SD+CG, everything else (loop / eviction /
metrics / logging / final-geometry harvesting) is shared.
"""

from typing import List, Optional
import os
import numpy as np
import torch
from ase import Atoms

DTYPE = torch.float64


# --------------------------------------------------------------------------- #
# small topology helpers (mirrors blbfgs.py)                                    #
# --------------------------------------------------------------------------- #
def _ptr_from_atoms(at_list, device):
    ptr = [0]
    for at in at_list:
        ptr.append(ptr[-1] + len(at))
    return torch.tensor(ptr, dtype=torch.long, device=device)


def _symbols_flat(at_list):
    return [at.get_chemical_symbols() for at in at_list]


def _get_coord_gpu(calc) -> torch.Tensor:
    if hasattr(calc, "coord"):
        return calc.coord
    elif hasattr(calc, "coord32"):
        return calc.coord32
    raise AttributeError("calculator has no coord buffer")


# --------------------------------------------------------------------------- #
# shared batched gradient-only optimizer base                                   #
# --------------------------------------------------------------------------- #
class _BatchGradOpt:
    """
    Shared batched, gradient-only optimizer skeleton (steepest-descent family).

    Subclasses implement the per-structure search DIRECTION + any per-structure
    algorithm state (phase, CG history, ...) via the hooks:
        _init_algo_state(B, nmax, F0)        : allocate per-structure state
        _pre_step(forces, iteration)         : pre-step state update (phase switch)
        _compute_direction(forces)           : (B, nmax) raw search direction
        _post_step(forces, direction)        : post-step state update (CG history)
        _shrink_algo_state(survive_idx)      : index per-structure state on eviction

    The base owns: topology / padding / masks, per-structure convergence
    thresholds, the maxstep step clip, the main iteration loop, dynamic batch
    shrinking (per-structure convergence eviction) and final-geometry harvesting.
    """

    _algo_name = "Batch-SD"

    def __init__(self,
                 output: str,
                 maxstep: float = 0.2,
                 maxiter: int = 256,
                 device: str = "cuda",
                 write_traj: bool = False,
                 traj_every: int = 1,
                 verbose: int = 1):
        self.maxstep = float(maxstep)
        self.maxiter = int(maxiter)
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

        # per-structure convergence thresholds (rebuilt only on shrink)
        self._f_max_th = None
        self._f_rms_th = None
        self._dp_max_th = None
        self._dp_rms_th = None

        # harvested final state, keyed by ORIGINAL batch index
        self.final_positions = {}   # orig_idx -> (n,3) float64 np.ndarray  [Angstrom]
        self.final_energy = {}      # orig_idx -> float  [Hartree]
        self.final_iter = {}        # orig_idx -> int    (iteration converged / stopped)
        self.final_converged = {}   # orig_idx -> bool

    # ===================================================
    # subclass hooks (default = pure steepest descent)
    # ===================================================
    def _init_algo_state(self, B, nmax, F0):
        pass

    def _pre_step(self, forces, iteration):
        pass

    def _compute_direction(self, forces):
        # steepest descent: direction = forces (= -grad).  maxstep scaling + clip
        # is applied by the base (step = clip(maxstep * direction)).
        return forces

    def _post_step(self, forces, direction):
        pass

    def _shrink_algo_state(self, survive_idx):
        pass

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
        self._w(f"# {self._algo_name} optimization start\n")
        self._w(f"# Maxstep: {self.maxstep}, Maxiter: {self.maxiter}, Device: {device}\n\n")

        self._init_xyz_paths(B0)
        self._symbols_per_batch = _symbols_flat(atoms_list)

        # single initial forward: fixes nmax AND provides E0/F0
        calc.prepare(atoms_list)
        E0, F0 = calc.get_ef_gpu()
        self._nmax = int(F0.shape[1])
        self._arange_n = torch.arange(self._nmax, device=device)

        self._rebuild_topology(atoms_list)
        self._dump_xyz_all(calc, atoms_list, tag="init")

        E_old = E0.to(dtype=DTYPE)
        F_old = (F0.to(dtype=DTYPE)) * self._real_mask.to(DTYPE)

        # per-structure algorithm state (uses the masked initial forces)
        self._init_algo_state(self._B, self._nmax, F_old)

        iteration = 0
        while iteration < self.maxiter and len(atoms_list) > 0:
            iteration += 1
            rmask = self._real_mask.to(DTYPE)

            # forces at the current committed geometry (= F_new of previous iter)
            forces = F_old * rmask

            # pre-step per-structure state update (e.g. SD->CG phase switch)
            self._pre_step(forces, iteration)

            # search direction + maxstep clip (step = clip(maxstep * direction))
            direction = self._compute_direction(forces) * rmask
            step_cart = self._clip_step_batched(self.maxstep * direction)

            # take the step; single calculator forward/iter at the new geometry
            calc.step_cart_(step_cart)
            E_new, F_new = calc.get_ef_gpu()
            E_new = E_new.to(dtype=DTYPE)
            F_new = (F_new.to(dtype=DTYPE)) * rmask

            # post-step per-structure state update (e.g. CG history store)
            self._post_step(forces, direction)

            # convergence / log metrics (committed forces + the step just taken)
            L_eff = self._L_eff
            max_f = F_new.abs().amax(dim=-1)
            rms_f = torch.sqrt((F_new ** 2).sum(-1) / L_eff)
            max_dp = step_cart.abs().amax(dim=-1)
            rms_dp = torch.sqrt((step_cart ** 2).sum(-1) / L_eff)

            self._log_iteration(iteration, E_new, max_f, rms_f, max_dp, rms_dp)
            if self.write_traj and iteration % self.traj_every == 0:
                self._dump_xyz_all(calc, atoms_list, tag=f"iter={iteration}")

            done = self._check_convergence(iteration, E_new, max_f, rms_f, max_dp, rms_dp)

            # harvest converged structures BEFORE the topology is rebuilt
            self._harvest(calc, done, E_new, iteration, converged=True)

            # dynamic batch shrinking (drop converged structures)
            survive_local = (~done).nonzero(as_tuple=False).flatten()
            if survive_local.numel() < len(done):
                atoms_list = [atoms_list[i] for i in survive_local.cpu().tolist()]
                self._orig_index = self._orig_index[survive_local]
                self._shrink_algo_state(survive_local)
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

        # harvest any structure that never converged (still in the batch)
        if len(atoms_list) > 0:
            full = torch.ones(len(atoms_list), dtype=torch.bool, device=device)
            self._harvest(calc, full, E_old, iteration, converged=False)

        # Propagate the optimized geometry back to the caller's Molecules, so the
        # standard dispatch path (optimization.py ``_run_batched`` -> ``return
        # mols``) and any ``mols.multiatoms[i].get_positions()`` reader see the
        # SD/SDCG result. BatchLBFGS / BatchRFO / BatchDIIS already update
        # ``mols.multiatoms`` in place; previously the _BatchGradOpt family wrote
        # the final geometry ONLY to ``self.final_positions`` and the dispatcher
        # discarded the optimizer object, so the batched SD/SDCG result was lost.
        # ``self.final_positions`` is preserved unchanged (additive, root-cause at
        # the shared base run()). (algorithm-audit fix; default = the optimized
        # geometry, matching the single-structure optimizers.)
        for orig, pos in self.final_positions.items():
            if 0 <= orig < len(mols.multiatoms):
                mols.multiatoms[orig].set_positions(pos)

        self._close_log()

    # ===================================================
    # STEP CLIPPING (per structure, identical to SDCG._clip_step)
    # ===================================================
    def _clip_step_batched(self, step: torch.Tensor) -> torch.Tensor:
        max_disp = step.abs().amax(dim=-1)               # (B,)
        scale = torch.clamp(self.maxstep / (max_disp + 1e-20), max=1.0)
        return step * scale.unsqueeze(-1)

    # ===================================================
    # CONVERGENCE
    # ===================================================
    def _check_convergence(self, it, E, max_f, rms_f, max_dp, rms_dp):
        done = (
            (max_f <= self._f_max_th)
            & (rms_f <= self._f_rms_th)
            & (max_dp <= self._dp_max_th)
            & (rms_dp <= self._dp_rms_th)
        )
        if self.verbose:
            self._w(self._fmt_convergence_table(
                it, E, max_f, rms_f, max_dp, rms_dp, done))
        return done

    # ===================================================
    # FINAL-GEOMETRY HARVEST
    # ===================================================
    def _harvest(self, calc, mask, E, iteration, converged):
        idx = mask.nonzero(as_tuple=False).flatten().cpu().tolist()
        if not idx:
            return
        with torch.no_grad():
            coord = _get_coord_gpu(calc).detach().cpu().numpy()
        ptr = self._ptr.detach().cpu().numpy()
        E_l = E.detach().cpu().tolist()
        for b in idx:
            orig = int(self._orig_index[b].item())
            s, t = int(ptr[b]), int(ptr[b + 1])
            self.final_positions[orig] = coord[s:t].astype(np.float64, copy=True)
            self.final_energy[orig] = float(E_l[b])
            self.final_iter[orig] = int(iteration)
            self.final_converged[orig] = bool(converged)

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
            [getattr(at, "f_max_th", 2e-3) for at in atoms_list], dtype=DTYPE, device=device)
        self._f_rms_th = torch.tensor(
            [getattr(at, "f_rms_th", 1e-3) for at in atoms_list], dtype=DTYPE, device=device)
        self._dp_max_th = torch.tensor(
            [getattr(at, "dp_max_th", 1e-3) for at in atoms_list], dtype=DTYPE, device=device)
        self._dp_rms_th = torch.tensor(
            [getattr(at, "dp_rms_th", 5e-4) for at in atoms_list], dtype=DTYPE, device=device)

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
        if not self.verbose:
            return
        means = torch.stack([
            E.mean(), max_f.mean(), rms_f.mean(), max_dp.mean(), rms_dp.mean()
        ]).tolist()
        e_m, maxf_m, rmsf_m, maxdp_m, rmsdp_m = means
        self._w(
            f"\nIter {it}: E_mean={e_m:.6f} max|F|_mean={maxf_m:.6f} "
            f"rms|F|_mean={rmsf_m:.6f} max|dX|_mean={maxdp_m:.6f} "
            f"rms|dX|_mean={rmsdp_m:.6f}\n"
        )

    def _fmt_convergence_table(self, it, E, max_f, rms_f, max_dp, rms_dp, done):
        B = E.shape[0]
        E_l, maxf_l, rmsf_l, maxdp_l, rmsdp_l = torch.stack(
            [E, max_f, rms_f, max_dp, rms_dp], dim=0).tolist()
        done_l = done.tolist()
        orig_l = self._orig_index.tolist()

        lines = ["-" * 70 + "\n",
                 f"{(self._algo_name + ' Iteration ' + str(it)).center(70)}\n",
                 "-" * 70 + "\n",
                 "Batch   Energy(Ha)   Max|F|(H/A)  RMS|F|  Max|dX|(A)  RMS|dX|  Conv\n",
                 "-" * 70 + "\n"]
        for b in range(B):
            lines.append(
                f"[{orig_l[b]:2d}] {E_l[b]:14.6f} {maxf_l[b]:12.6f} "
                f"{rmsf_l[b]:10.6f} {maxdp_l[b]:12.6f} {rmsdp_l[b]:10.6f} "
                f"{('YES' if done_l[b] else 'NO')}\n"
            )
        lines.append("\n")
        return "".join(lines)

    # ===================================================
    # XYZ OUTPUT
    # ===================================================
    def _init_xyz_paths(self, B_all):
        self.xyz_paths = [
            os.path.join(self.out_dir, f"{self._algo_name.lower().replace('-', '')}_batch{i + 1}.xyz")
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
            f.write(f"{n}\n{comment}\n")
            for k in range(n):
                x, y, z = pos_np[k]
                f.write(f"{symbols[k]:<2s} {x:20.10f} {y:20.10f} {z:20.10f}\n")
        self.frame_counts[idx_orig] += 1


# --------------------------------------------------------------------------- #
# BatchSD                                                                       #
# --------------------------------------------------------------------------- #
class BatchSD(_BatchGradOpt):
    """
    Batched steepest-descent optimizer (gradient-only, per-structure eviction).

    Direction = forces; the base applies  step = clip(maxstep * forces).  This is
    the exact batched form of the single-structure SD update (SDCG with
    method='sd', GDIIS off).  No per-structure algorithm state beyond the shared
    topology, so the state hooks are the no-op defaults of ``_BatchGradOpt``.
    """
    _algo_name = "Batch-SD"

    def __init__(self,
                 output: str,
                 max_step: float = 0.2,
                 max_iter: int = 256,
                 device: str = "cuda",
                 write_traj: bool = False,
                 traj_every: int = 1,
                 verbose: int = 1):
        super().__init__(output, maxstep=max_step, maxiter=max_iter, device=device,
                         write_traj=write_traj, traj_every=traj_every, verbose=verbose)
