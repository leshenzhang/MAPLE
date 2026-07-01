# -*- coding: utf-8 -*-
"""
Batched MACE-POL calculator (GPU throughput-oriented) + isolation guard.

================================================================================
CRITICAL FINDING (measured 2026-06-26, macepols.pt, cxtorch torch 2.11) -- READ THIS
================================================================================
MACE-POL CANNOT be safely block-diagonal-batched. Two independent obstacles:

(1) The traced model is **B=1-locked**: `torch.jit.trace` baked `dim_size=[1]` into
    the per-graph reference-energy (`e0`) scatter, so passing a true multi-graph
    `ptr`/`batch` (B>1) raises
        `index 1 is out of bounds for dimension 0 with size 1`.
    => a real multi-graph batch (the AIMNet2 contract design) is impossible.

(2) The only way to push >1 molecule through the trace is to pack them as ONE graph
    (`batch=zeros(N)`, `ptr=[0,N]`, one 3x3 cell, one total_charge). But MACE-POL has
    a long-range electrostatics head (`model.coulomb_energy`: Gaussian densities,
    sigma~1.5 A, real-space erf-screened Coulomb with a 1/r monopole tail) AND it does
    a **global charge equilibration constrained to the single total_charge**. Packed in
    one graph, molecules COUPLE:
      * B=1 (one molecule alone)         -> matches single MACEPolCalculator EXACTLY
                                            (dE=0, dFmax~1e-8 Ha/A). Inputs are correct.
      * B=3, molecules 50-100 A apart    -> each molecule's force differs from its
                                            isolated value by ~1-5e-3 Ha/A, and the
                                            per-molecule monopole charge becomes nonzero
                                            (+0.011/+0.011/-0.022 e; sum=0): charge is
                                            transferred BETWEEN far-apart molecules.
      * perturb-one byte-isolation test  -> FAILS: perturbing mol 0 changes mol 1/2
                                            forces (leak ~1e-3 Ha/A @5A decaying to
                                            ~1e-5 Ha/A @50A; never byte-zero).
    The coupling (~1e-3 Ha/A) exceeds the 1e-4 Ha/A parity gate and is comparable to /
    larger than typical optimizer fmax, so a packed batch does NOT reproduce isolated
    single-molecule physics.

CONSEQUENCE -> this class defaults to `coupling_mode="raise"`. Modes:
  * "raise"      (default): prepare() raises a clear error. SAFE default.
  * "sequential": correct fallback -- loops the single MACEPolCalculator per molecule
                  (full contract, results identical to single calc, but NO GPU-batch
                  speedup). Use when you need the batch contract but correct numbers.
  * "approx"     : opt-in single-graph block-diagonal batched forward (the throughput
                  path). FAST but carries the ~1e-3 Ha/A cross-molecule coupling error
                  above; only for loose screening where that error is acceptable. Emits
                  a warning and the measured isolation leak at prepare().

`isolation_check()` quantifies the cross-molecule force leak for the current batch so a
caller can decide. Standard (non-POL) MACE-OFF does NOT have this problem (native
multi-graph batching) -- see the companion standard-MACE batch calculator.
================================================================================

Contract (matches AIMNet2BatchCalc): prepare / step_cart_ / set_coords_ /
backup_coords / restore_coords / get_ef_gpu -> (E_Ha(B,), F_Ha(B,nmax_dof)) /
get_efh_gpu -> (E_Ha, F_Ha, H_Ha(B,nmax,nmax), P(B,) padding-atom count).
Units: Hartree, Hartree/Angstrom (model returns eV; EV2HARTREE=1/27.211386245988).
Gas phase ONLY (raises if a solvent is requested).
"""

import os
import warnings
import torch
import numpy as np
from typing import List, Sequence
from ase import Atoms

EV2HARTREE = 1.0 / 27.211386245988
EH2EV = 27.211386245988

_MACEPOL_MODEL_FILES = {
    'macepols': 'macepols.pt',
    'macepolm': 'macepolm.pt',
    'macepoll': 'macepoll.pt',
}


def _one_hot_node_attrs(Z: torch.Tensor, atomic_number_table: Sequence[int],
                        dtype=torch.float32) -> torch.Tensor:
    table = torch.tensor(list(atomic_number_table), dtype=torch.long, device=Z.device)
    eq = (Z[:, None] == table[None, :])
    if not torch.all(eq.any(dim=1)):
        miss = Z[~eq.any(dim=1)].unique().tolist()
        raise ValueError(f"Atomic number(s) {miss} not in AtomicNumberTable "
                         f"{list(atomic_number_table)}")
    return eq.to(dtype)


class MACEPolBatchCalc:
    """Batched MACE-POL calculator. See module docstring for the coupling finding.

    coupling_mode: 'raise' (default, safe) | 'sequential' (correct, no speedup) |
                   'approx' (fast single-graph batch, ~1e-3 Ha/A coupling error).
    isolation_tol: Ha/A force-leak threshold used by the warning/guard logic.
    """

    def __init__(self,
                 device: str = "cuda",
                 model: str = "macepols",
                 model_path: str = None,
                 dtype: torch.dtype = torch.float64,
                 implicit: str = "none",
                 solvent: str = "none",
                 coupling_mode: str = "raise",
                 isolation_tol: float = 1e-4):
        if str(solvent).lower() not in ("none", "", "vacuum", "gas"):
            raise ValueError(
                f"MACEPolBatchCalc is gas-phase ONLY (no GBSA in batch); "
                f"got solvent={solvent!r}.")
        if coupling_mode not in ("raise", "sequential", "approx"):
            raise ValueError("coupling_mode must be 'raise'|'sequential'|'approx'")

        self.device = torch.device(device)
        self.dtype = dtype
        self.mdtype = torch.float32                  # traced MACE-POL is f32
        self.coupling_mode = coupling_mode
        self.isolation_tol = float(isolation_tol)
        self._model_name = model

        if model_path is None:
            model_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.dirname(model_dir)
            filename = _MACEPOL_MODEL_FILES.get(model, f'{model}.pt')
            model_path = os.path.join(model_dir, 'model', filename)
        self._model_path = model_path

        self.model = torch.jit.load(model_path, map_location=self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.r_max = float(self.model.r_max)
        self.atomic_numbers = [int(z) for z in self.model.atomic_numbers]

        self._hess_mode = None                       # 'analytic'|'fd' (approx mode)
        self._seq_calc = None                        # lazy single MACEPolCalculator

        # buffers
        self._prepared = False
        self._atoms_B = 0
        self._ptr = None
        self.numbers = None
        self.node_attrs = None
        self.mol_idx = None
        self.coord = None
        self.N_atoms = 0
        self.Nmax_atoms = 0
        self.nmax_dof = 0
        self.total_charge = None
        self.total_spin = None
        self._charges_host = None
        self._mults_host = None
        self.cand_i = None
        self.cand_j = None
        self._base = None
        self._n_b = None
        self._coord_backup = None
        # R3-5: loop-invariant single-graph forward tensors (set in prepare())
        self._fwd_batch = None
        self._fwd_ptr = None
        self._fwd_cell = None
        self._fwd_tc = None
        self._fwd_ts = None
        self._fwd_ext = None
        self._fwd_log = None

        try:
            self._probe_double_backward()
        except Exception:
            self._hess_mode = None

    # =====================================================================
    def prepare(self, atoms_list: List[Atoms], fixed_nmax: int = None):
        device, dtype = self.device, self.dtype
        B = len(atoms_list)
        self._atoms_B = B

        ptr = [0]
        nums, mids, coords, charges, spins = [], [], [], [], []
        for i, at in enumerate(atoms_list):
            Z = torch.tensor(at.get_atomic_numbers(), dtype=torch.int64, device=device)
            n = int(Z.shape[0])
            ptr.append(ptr[-1] + n)
            nums.append(Z)
            mids.append(torch.full((n,), i, dtype=torch.int64, device=device))
            coords.append(torch.tensor(at.get_positions(), dtype=dtype, device=device))
            charges.append(float(at.info.get('charge', 0)))
            spins.append(float(int(at.info.get('mult', 1)) - 1))

        self._ptr = torch.tensor(ptr, dtype=torch.int64, device=device)
        self.numbers = torch.cat(nums) if nums else torch.zeros((0,), dtype=torch.int64, device=device)
        self.mol_idx = torch.cat(mids) if mids else torch.zeros((0,), dtype=torch.int64, device=device)
        self.coord = (torch.cat(coords).contiguous() if coords
                      else torch.zeros((0, 3), dtype=dtype, device=device))
        self.N_atoms = int(self.numbers.numel())
        self.Nmax_atoms = int(max((len(at) for at in atoms_list), default=0))
        self.nmax_dof = (3 * self.Nmax_atoms) if fixed_nmax is None else int(fixed_nmax)
        self._charges_host = charges
        self._mults_host = [int(at.info.get('mult', 1)) for at in atoms_list]

        self.node_attrs = (_one_hot_node_attrs(self.numbers, self.atomic_numbers, self.mdtype)
                           if self.N_atoms > 0
                           else torch.zeros((0, len(self.atomic_numbers)), dtype=self.mdtype, device=device))
        self.total_charge = torch.tensor(charges, dtype=self.mdtype, device=device)
        self.total_spin = torch.tensor(spins, dtype=self.mdtype, device=device)

        s = self._ptr[:-1]; t = self._ptr[1:]
        self._n_b = (t - s)
        if self.N_atoms > 0:
            local_idx = torch.arange(self.N_atoms, device=device) - s[self.mol_idx]
            self._base = self.mol_idx * self.nmax_dof + 3 * local_idx
        else:
            self._base = torch.zeros((0,), dtype=torch.int64, device=device)

        ci, cj = [], []
        for b in range(B):
            n = int(self._n_b[b].item()); off = int(s[b].item())
            if n >= 2:
                iu, ju = torch.triu_indices(n, n, offset=1, device=device)
                ci.append(iu + off); cj.append(ju + off)
        self.cand_i = torch.cat(ci) if ci else torch.zeros((0,), dtype=torch.int64, device=device)
        self.cand_j = torch.cat(cj) if cj else torch.zeros((0,), dtype=torch.int64, device=device)

        # R3-5 opt: hoist the 7 loop-invariant tensors consumed by the per-step
        # _forward_single_graph model call out of the hot path (batch/ptr/cell are
        # geometry-invariant; tc/ts derive from the fixed total_charge/total_spin;
        # ext/log are the constant zero external-field / unit log-weight). Values are
        # byte-identical to the former per-forward allocation.
        N = self.N_atoms
        self._fwd_batch = torch.zeros(N, dtype=torch.int64, device=device)
        self._fwd_ptr = torch.tensor([0, N], dtype=torch.int64, device=device)
        self._fwd_cell = torch.zeros((3, 3), dtype=self.mdtype, device=device)
        self._fwd_tc = self.total_charge.sum().reshape(1)
        self._fwd_ts = self.total_spin.sum().reshape(1)
        self._fwd_ext = torch.zeros((N, 3), dtype=self.mdtype, device=device)
        self._fwd_log = torch.ones((N,), dtype=self.mdtype, device=device)

        self._coord_backup = None
        self._prepared = True

        # ---- mode gating ----------------------------------------------------
        # single-graph packing carries one total_charge; mixed per-mol charge is
        # NOT representable (would be globally redistributed) -> always reject.
        if self.coupling_mode in ("raise", "approx") and B > 1:
            if len(set(self._charges_host)) > 1:
                raise ValueError(
                    "MACEPolBatchCalc single-graph packing supports only a single "
                    "total_charge: mixed per-molecule charges "
                    f"{sorted(set(self._charges_host))} cannot be batched (the model "
                    "would redistribute charge globally). Use coupling_mode='sequential'.")

        if self.coupling_mode == "sequential":
            self._ensure_seq_calc()
            return

        if self.coupling_mode == "raise" and B > 1:
            raise RuntimeError(
                "MACE-POL globally couples molecules and CANNOT be safely block-diagonal "
                "batched (the traced model is B=1-locked, so molecules must be packed into "
                "ONE graph; MACE-POL then does a global total-charge-constrained charge "
                "equilibration + long-range Coulomb that transfers charge BETWEEN molecules "
                "even at >50 A, ~1-5e-3 Ha/A force coupling -- perturb-one byte-isolation "
                "FAILS). Refusing to return silently-coupled results. Options: "
                "coupling_mode='sequential' (correct, no GPU-batch speedup) or "
                "coupling_mode='approx' (fast single-graph batch, accepts the coupling "
                "error for loose screening). See module docstring + isolation_check().")

        if self.coupling_mode == "approx" and B > 1:
            leak = None
            try:
                leak = self.isolation_check()
            except Exception:
                pass
            msg = ("MACEPolBatchCalc coupling_mode='approx': MACE-POL globally couples "
                   "molecules (global charge equilibration + long-range Coulomb). Batched "
                   "results carry a cross-molecule coupling error")
            if leak is not None:
                msg += f" (measured perturb-one force leak = {leak:.2e} Ha/A)"
            msg += ". Use only for loose screening; use 'sequential' for correct numbers."
            warnings.warn(msg, RuntimeWarning)

    # =====================================================================
    # coordinate ops (shared, vectorized)
    # =====================================================================
    @torch.no_grad()
    def step_cart_(self, s_cart: torch.Tensor):
        assert self._prepared, "call prepare() first"
        B = self._atoms_B
        assert s_cart.shape == (B, self.nmax_dof), \
            f"step_cart_ expects (B,{self.nmax_dof}), got {tuple(s_cart.shape)}"
        if self.N_atoms == 0:
            return
        s_flat = s_cart.reshape(-1).to(self.device, self.dtype)
        base = self._base
        disp = torch.stack([s_flat[base], s_flat[base + 1], s_flat[base + 2]], dim=1)
        self.coord.add_(disp)

    @torch.no_grad()
    def set_coords_(self, coord: torch.Tensor):
        assert self._prepared, "call prepare() first"
        assert coord.shape == (self.N_atoms, 3)
        self.coord.copy_(coord.to(self.device, self.dtype))

    @torch.no_grad()
    def backup_coords(self):
        if self._prepared:
            self._coord_backup = self.coord.clone()

    @torch.no_grad()
    def restore_coords(self):
        if self._coord_backup is not None:
            self.coord.copy_(self._coord_backup)
            self._coord_backup = None

    # =====================================================================
    # single-graph block-diagonal forward (approx mode + isolation_check)
    # =====================================================================
    def _build_edges(self, coord_f32: torch.Tensor):
        """Block-diagonal radius graph from precomputed intra-mol candidate pairs.
        EXACT squared distance (NOT cdist). Note: this isolates the *message-passing*
        edges; the model's internal Coulomb head still couples molecules globally."""
        device = self.device
        if self.cand_i.numel() == 0:
            ei = torch.zeros((2, 0), dtype=torch.int64, device=device)
            sh = torch.zeros((0, 3), dtype=self.mdtype, device=device)
            return ei, sh, sh.clone()
        rij = coord_f32[self.cand_i] - coord_f32[self.cand_j]
        d2 = (rij * rij).sum(dim=-1)
        keep = d2 <= (self.r_max + 1e-12) ** 2
        ci = self.cand_i[keep]; cj = self.cand_j[keep]
        src = torch.cat([ci, cj], dim=0); dst = torch.cat([cj, ci], dim=0)
        edge_index = torch.stack([src, dst], dim=0)
        shifts = torch.zeros((edge_index.size(1), 3), dtype=self.mdtype, device=device)
        return edge_index, shifts, shifts.clone()

    def _forward_single_graph(self, coord: torch.Tensor, need_graph: bool):
        """Pack ALL molecules into ONE graph (the only shape the B=1 trace accepts).
        Returns (E_eV(B,) per-mol from node_energy scatter, F_all_eV(N,3), coord_leaf).
        NOTE: long-range Coulomb energy is graph-global and NOT split into the per-mol
        E here (it IS present in the forces via autograd of the true total energy)."""
        assert self._prepared, "call prepare() first"
        device = self.device
        B, N = self._atoms_B, self.N_atoms
        coord_leaf = coord.detach().to(device=device, dtype=self.mdtype).requires_grad_(True)
        edge_index, shifts, unit_shifts = self._build_edges(coord_leaf)
        # R3-5 opt: reuse the loop-invariant forward tensors hoisted in prepare().
        out = self.model(coord_leaf, self.node_attrs, edge_index, shifts, unit_shifts,
                         self._fwd_batch, self._fwd_ptr, self._fwd_cell,
                         self._fwd_tc, self._fwd_ts, self._fwd_ext, self._fwd_log)
        total_energy, node_energy, _density = out[0], out[1], out[2]

        ne = node_energy.reshape(-1)
        E_eV = torch.zeros(B, dtype=ne.dtype, device=device).index_add(0, self.mol_idx, ne)

        # forces from the TRUE total energy (includes coulomb)
        grad = torch.autograd.grad(total_energy.sum(), coord_leaf,
                                   create_graph=need_graph, retain_graph=need_graph)[0]
        F_all_eV = -grad
        return E_eV.to(self.dtype), F_all_eV, coord_leaf

    def _scatter_forces(self, F_all: torch.Tensor) -> torch.Tensor:
        B = self._atoms_B
        F_flat = torch.zeros(B * self.nmax_dof, dtype=self.dtype, device=self.device)
        if self.N_atoms > 0:
            base = self._base
            F_flat[base] = F_all[:, 0]; F_flat[base + 1] = F_all[:, 1]; F_flat[base + 2] = F_all[:, 2]
        return F_flat.view(B, self.nmax_dof)

    # =====================================================================
    # partial-Hessian (movable-atom subspace) helpers -- mirror UMABatchCalc
    # =====================================================================
    def _resolve_movable(self, movable_masks):
        """Per-structure list[int] of movable atom indices. None = all atoms;
        per-structure None/bool-mask/index-list otherwise. Frozen atoms still appear
        in every forward (they exert forces); only their Hessian rows/cols are
        dropped (exact FixAtoms-constrained block). Mirrors UMABatchCalc."""
        B = self._atoms_B
        n_b = (self._ptr[1:] - self._ptr[:-1]).tolist()
        if movable_masks is None:
            return [list(range(n_b[b])) for b in range(B)]
        out = []
        for b in range(B):
            m = movable_masks[b]
            if m is None:
                out.append(list(range(n_b[b])))
                continue
            m_arr = np.asarray(m)
            if m_arr.dtype == bool:
                out.append([int(i) for i in np.nonzero(m_arr)[0]])
            else:
                idxs = [int(i) for i in m_arr.reshape(-1)]
                assert all(0 <= a < n_b[b] for a in idxs), (
                    f"movable index out of range for structure {b} "
                    f"(n_atoms={n_b[b]}): {idxs}")
                out.append(idxs)
        return out

    def _movable_tensors(self, movable_masks):
        """Return (movable_atom (N,) bool, movable_la (B,nmax_a) bool). Both reduce to
        the full real-atom set (-> byte-identical full Hessian) when movable_masks=None."""
        N, B, nmax_a = self.N_atoms, self._atoms_B, self.Nmax_atoms
        device = self.device
        mov = self._resolve_movable(movable_masks)
        movable_atom = torch.zeros(N, dtype=torch.bool, device=device)
        movable_la = torch.zeros((B, nmax_a), dtype=torch.bool, device=device)
        ptr = self._ptr.tolist()
        for i in range(B):
            base = ptr[i]
            for a in mov[i]:
                movable_atom[base + a] = True
                movable_la[i, a] = True
        return movable_atom, movable_la

    # =====================================================================
    def isolation_check(self, perturb: float = 0.05) -> float:
        """Perturb-one byte-isolation probe on the CURRENT batch.
        Returns the max cross-molecule force leak (Ha/A) when mol 0's first atom is
        displaced by `perturb` A. 0.0 => byte-isolated; >0 => molecules couple.
        Uses the single-graph forward (approx path)."""
        assert self._prepared and self._atoms_B >= 2
        c0 = self.coord.clone()
        _, F0, _ = self._forward_single_graph(c0, need_graph=False)
        c1 = c0.clone(); c1[0, 0] += perturb
        _, F1, _ = self._forward_single_graph(c1, need_graph=False)
        leak = 0.0
        for j in range(1, self._atoms_B):
            rows = (self.mol_idx == j)
            d = (F1[rows] - F0[rows]).abs().max().item()
            leak = max(leak, d)
        return leak * EV2HARTREE

    # =====================================================================
    # public: E + F
    # =====================================================================
    def get_ef_gpu(self):
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return (torch.zeros((0,), dtype=dtype, device=device),
                    torch.zeros((0, 0), dtype=dtype, device=device))
        if self.coupling_mode == "raise" and B > 1:
            raise RuntimeError("coupling_mode='raise'; call with 'sequential' or 'approx'.")
        if self.coupling_mode == "sequential":
            return self._ef_sequential()
        E_eV, F_all_eV, _ = self._forward_single_graph(self.coord, need_graph=False)
        F_eV = self._scatter_forces(F_all_eV.detach().to(dtype))
        return (E_eV.detach() * EV2HARTREE, F_eV * EV2HARTREE)

    def get_efh_gpu(self, movable_masks=None):
        """Energy + forces + per-structure Hessian. ``movable_masks`` (mirrors
        UMABatchCalc): None = full Hessian (byte-identical current behavior); a
        per-structure spec restricts the Hessian to a movable-atom subspace -> only
        movable rows/cols filled (frozen atoms still exert forces). Threaded through
        all three paths: sequential (single-calc full Hessian masked to the subspace),
        approx seeded-analytic, and approx batched-FD."""
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return (torch.zeros((0,), dtype=dtype, device=device),
                    torch.zeros((0, 0), dtype=dtype, device=device),
                    torch.zeros((0, 0, 0), dtype=dtype, device=device),
                    torch.zeros((0,), dtype=torch.int64, device=device))
        if self.coupling_mode == "raise" and B > 1:
            raise RuntimeError("coupling_mode='raise'; call with 'sequential' or 'approx'.")
        if self.coupling_mode == "sequential":
            return self._efh_sequential(movable_masks)
        if self._hess_mode is None:
            try:
                self._probe_double_backward()
            except Exception:
                self._hess_mode = 'fd'
        if self._hess_mode == 'analytic':
            try:
                return self._efh_analytic(movable_masks)
            except Exception:
                self._hess_mode = 'fd'
        return self._efh_fd(movable_masks=movable_masks)

    # =====================================================================
    # sequential (correct) fallback -- loops the single MACEPolCalculator
    # =====================================================================
    def _ensure_seq_calc(self):
        if self._seq_calc is None:
            from ._macepol_calculator import MACEPolCalculator
            self._seq_calc = MACEPolCalculator(device=self.device, model=self._model_name,
                                               model_path=self._model_path,
                                               implicit="none", solvent="none")

    def _mol_atoms(self, i: int) -> Atoms:
        s = int(self._ptr[i].item()); t = int(self._ptr[i + 1].item())
        Z = self.numbers[s:t].detach().cpu().numpy()
        pos = self.coord[s:t].detach().cpu().numpy()
        at = Atoms(numbers=Z, positions=pos)
        at.info['charge'] = self._charges_host[i]
        at.info['mult'] = self._mults_host[i]
        return at

    def _ef_sequential(self):
        self._ensure_seq_calc()
        B, dtype, device = self._atoms_B, self.dtype, self.device
        E = torch.zeros(B, dtype=dtype, device=device)
        F = torch.zeros((B, self.nmax_dof), dtype=dtype, device=device)
        s = self._ptr
        for i in range(B):
            at = self._mol_atoms(i)
            at.calc = self._seq_calc
            E[i] = float(at.get_potential_energy())      # Hartree
            f = at.get_forces()                          # Hartree/A
            n = int((s[i + 1] - s[i]).item())
            F[i, :3 * n] = torch.tensor(f.reshape(-1), dtype=dtype, device=device)
        return E, F

    def _efh_sequential(self, movable_masks=None):
        self._ensure_seq_calc()
        B, dtype, device = self._atoms_B, self.dtype, self.device
        nmax, nmax_a = self.nmax_dof, self.Nmax_atoms
        E = torch.zeros(B, dtype=dtype, device=device)
        F = torch.zeros((B, nmax), dtype=dtype, device=device)
        H = torch.zeros((B, nmax, nmax), dtype=dtype, device=device)
        P = torch.zeros(B, dtype=torch.int64, device=device)
        mov = self._resolve_movable(movable_masks)        # per-structure movable atom idx
        s = self._ptr
        for i in range(B):
            at = self._mol_atoms(i)
            at.calc = self._seq_calc
            E[i] = float(at.get_potential_energy())
            f = at.get_forces()
            n = int((s[i + 1] - s[i]).item())
            F[i, :3 * n] = torch.tensor(f.reshape(-1), dtype=dtype, device=device)
            Hi = torch.tensor(np.asarray(self._seq_calc.get_hessian(at)),  # (3n,3n) Ha/A^2
                              dtype=dtype, device=device)
            if movable_masks is not None:
                # mask the full single-calc block to the movable rows/cols (frozen=0)
                keep = torch.zeros(3 * n, dtype=torch.bool, device=device)
                for a in mov[i]:
                    keep[3 * a:3 * a + 3] = True
                Hi = Hi * keep[:, None].to(dtype) * keep[None, :].to(dtype)
            H[i, :3 * n, :3 * n] = Hi
            P[i] = nmax_a - n
        return E, F, H, P

    # =====================================================================
    # approx-mode batched Hessian (seeded analytic / batched FD)
    # =====================================================================
    def _efh_analytic(self, movable_masks=None):
        B, device, dtype = self._atoms_B, self.device, self.dtype
        N, nmax, nmax_a = self.N_atoms, self.nmax_dof, self.Nmax_atoms
        E_eV, F_all_eV, coord_leaf = self._forward_single_graph(self.coord, need_graph=True)
        s = self._ptr[:-1]; n_b = self._n_b; n_b_list = n_b.tolist()
        F_eV = self._scatter_forces(F_all_eV.to(dtype))
        H_eV = torch.zeros((B, nmax, nmax), dtype=dtype, device=device)
        P = (nmax_a - n_b).to(torch.int64)
        movable_atom, movable_la = self._movable_tensors(movable_masks)
        row_scale = movable_atom[:, None].to(dtype)
        for k in range(3 * nmax_a):
            a_local, c = k // 3, k % 3
            valid = movable_la[:, a_local]          # was (n_b > a_local); == it when None
            if not bool(valid.any()):
                continue
            rows = s[valid] + a_local
            go = torch.zeros((N, 3), dtype=F_all_eV.dtype, device=device)
            go[rows, c] = 1.0
            col = torch.autograd.grad(F_all_eV, coord_leaf, grad_outputs=go,
                                      retain_graph=True, create_graph=False)[0].to(dtype)
            col = col * row_scale                   # zero frozen response rows (no-op when None)
            for i in valid.nonzero(as_tuple=False).flatten().tolist():
                dof = 3 * n_b_list[i]
                if k < dof:
                    H_eV[i, :dof, k] = -col[s[i]:s[i] + n_b_list[i], :].reshape(-1)
        H_eV = 0.5 * (H_eV + H_eV.transpose(1, 2))
        return (E_eV.detach() * EV2HARTREE, F_eV * EV2HARTREE, H_eV * EV2HARTREE, P)

    def _efh_fd(self, delta: float = 2e-3, movable_masks=None):
        B, device, dtype = self._atoms_B, self.device, self.dtype
        nmax, nmax_a = self.nmax_dof, self.Nmax_atoms
        E_eV, F_all_eV, _ = self._forward_single_graph(self.coord, need_graph=False)
        base_coord = self.coord.clone()
        s = self._ptr[:-1]; n_b = self._n_b; n_b_list = n_b.tolist()
        F_eV = self._scatter_forces(F_all_eV.detach().to(dtype))
        H_eV = torch.zeros((B, nmax, nmax), dtype=dtype, device=device)
        P = (nmax_a - n_b).to(torch.int64)
        movable_atom, movable_la = self._movable_tensors(movable_masks)
        row_scale = movable_atom[:, None].to(dtype)
        for k in range(3 * nmax_a):
            a_local, c = k // 3, k % 3
            valid = movable_la[:, a_local]          # was (n_b > a_local); == it when None
            if not bool(valid.any()):
                continue
            rows = s[valid] + a_local
            cp = base_coord.clone(); cp[rows, c] += delta
            cm = base_coord.clone(); cm[rows, c] -= delta
            _, Fp, _ = self._forward_single_graph(cp, need_graph=False)
            _, Fm, _ = self._forward_single_graph(cm, need_graph=False)
            col = ((-(Fp - Fm) / (2.0 * delta)).to(dtype)) * row_scale  # zero frozen rows (no-op None)
            for i in valid.nonzero(as_tuple=False).flatten().tolist():
                dof = 3 * n_b_list[i]
                if k < dof:
                    H_eV[i, :dof, k] = col[s[i]:s[i] + n_b_list[i], :].reshape(-1)
        self.coord.copy_(base_coord)
        H_eV = 0.5 * (H_eV + H_eV.transpose(1, 2))
        return (E_eV.detach() * EV2HARTREE, F_eV * EV2HARTREE, H_eV * EV2HARTREE, P)

    # =====================================================================
    def _probe_double_backward(self):
        device = self.device
        z = self.atomic_numbers[0]
        d = min(1.0, self.r_max * 0.5)
        coord = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, d]], dtype=self.mdtype,
                             device=device).requires_grad_(True)
        Z = torch.tensor([z, z], dtype=torch.int64, device=device)
        na = _one_hot_node_attrs(Z, self.atomic_numbers, self.mdtype)
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64, device=device)
        sh = torch.zeros((2, 3), dtype=self.mdtype, device=device)
        batch = torch.zeros(2, dtype=torch.int64, device=device)
        ptr = torch.tensor([0, 2], dtype=torch.int64, device=device)
        cell = torch.zeros((3, 3), dtype=self.mdtype, device=device)
        tc = torch.zeros(1, dtype=self.mdtype, device=device)
        ts = torch.zeros(1, dtype=self.mdtype, device=device)
        ext = torch.zeros((2, 3), dtype=self.mdtype, device=device)
        log = torch.ones(2, dtype=self.mdtype, device=device)
        out = self.model(coord, na, ei, sh, sh.clone(), batch, ptr, cell, tc, ts, ext, log)
        E = out[0].reshape(-1).sum()
        try:
            F = -torch.autograd.grad(E, coord, create_graph=True)[0]
            go = torch.zeros_like(coord); go[0, 2] = 1.0
            torch.autograd.grad(F, coord, grad_outputs=go, retain_graph=False)[0]
            self._hess_mode = 'analytic'
        except Exception:
            self._hess_mode = 'fd'
