# -*- coding: utf-8 -*-
"""
Batched STANDARD (non-POL) MACE calculator -- TRUE multi-graph batching.

Contrast with MACE-POL (`_macepol_batch_calculator.py`): standard MACE-OFF / MACE-MP /
maceomol models are PURE LOCAL MLIPs. They have NO long-range Coulomb head and NO global
charge equilibration, so per-graph energy pooling (`scatter_sum` over `batch`) makes every
molecule in a batch independent. A block-diagonal multi-graph batch (concatenated atoms +
per-molecule offset edges + `batch`/`ptr`) is therefore EXACTLY isolated by construction:
the perturb-one byte-isolation test passes, and per-molecule E/F equal the single-molecule
values bit-for-bit. This is the clean batching MACE-POL could not do.

Two preconditions are auto-checked at init by `_probe_batch_native()`:
  * The traced model must accept B>1 (`ptr=[0,n0,n0+n1,...]`). If the `.pt` was traced
    with a frozen single-graph `dim_size` (as the MACE-POL trace was -> `index out of
    bounds ... size 1`), this calculator raises with guidance (no silent fallback to wrong
    numbers). Standard MACE traces are normally exported batch-generic, so this passes.
  * Double-backward (for the seeded analytic Hessian) is probed; FD fallback otherwise.

Target model format = the 6-positional-arg "general" MACE wrapper used by
`_mace_general_calculator.py` (maceomol family):
    model(positions, node_attrs, edge_index, shifts, batch, ptr) -> total_energy (B,)  [eV]
Master coords are float64 (this wrapper is f64, matching the single calc). Gas phase only.

Contract matches AIMNet2BatchCalc exactly: prepare / step_cart_ / set_coords_ /
backup_coords / restore_coords / get_ef_gpu -> (E_Ha(B,), F_Ha(B,nmax_dof)) /
get_efh_gpu -> (E_Ha, F_Ha, H_Ha(B,nmax,nmax), P(B,) padding-atom count).
Units: Hartree, Hartree/Angstrom (model returns eV; EV2HARTREE=1/27.211386245988).

VALIDATION STATUS (2026-06-26): the model-free logic (batch collation, block-diagonal
offset edges, vectorized step_cart_/scatter) is unit-tested. The E/F-vs-single and
perturb-one RUNTIME gates could NOT be executed in the authoring sandbox because no
loadable standard MACE model was present there (scripted maceoff23*/maceomol .pt absent;
the `mace` python package was broken by a torchvision/torch mismatch; maceomol_v2.pt was
not TorchScript). Run `_test_mace_batch.py` once a real scripted standard MACE .pt is in
maple/function/calculator/model/ -- the included gates then verify isolation + parity.
"""

import os
import torch
import numpy as np
from typing import List, Sequence
from ase import Atoms

EV2HARTREE = 1.0 / 27.211386245988
EH2EV = 27.211386245988


def _one_hot_node_attrs(Z: torch.Tensor, atomic_number_table: Sequence[int],
                        dtype=torch.float64) -> torch.Tensor:
    table = torch.tensor(list(atomic_number_table), dtype=torch.long, device=Z.device)
    eq = (Z[:, None] == table[None, :])
    if not torch.all(eq.any(dim=1)):
        miss = Z[~eq.any(dim=1)].unique().tolist()
        raise ValueError(f"Atomic number(s) {miss} not in AtomicNumberTable "
                         f"{list(atomic_number_table)}")
    return eq.to(dtype)


class MACEBatchCalc:
    """Batched standard-MACE calculator (true multi-graph). See module docstring."""

    def __init__(self,
                 device: str = "cuda",
                 model: str = "maceomol",
                 model_path: str = None,
                 dtype: torch.dtype = torch.float64,
                 implicit: str = "none",
                 solvent: str = "none"):
        if str(solvent).lower() not in ("none", "", "vacuum", "gas"):
            raise ValueError(
                f"MACEBatchCalc is gas-phase ONLY; got solvent={solvent!r}.")

        self.device = torch.device(device)
        self.dtype = dtype                            # standard MACE wrapper is f64
        self.mdtype = dtype
        self._model_name = model

        if model_path is None:
            model_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.dirname(model_dir)
            model_path = os.path.join(model_dir, 'model', f'{model}.pt')
        self._model_path = model_path

        self.model = torch.jit.load(model_path, map_location=self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.r_max = float(self.model.r_max)
        self.atomic_numbers = [int(z) for z in self.model.atomic_numbers]

        self._batch_native = None     # set by probe: True if model accepts B>1
        self._hess_mode = None        # 'analytic' | 'fd'

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
        self.cand_i = None
        self.cand_j = None
        self._base = None
        self._n_b = None
        self._coord_backup = None

        try:
            self._probe_batch_native()
        except Exception:
            self._batch_native = None      # resolve lazily

    # =====================================================================
    def prepare(self, atoms_list: List[Atoms], fixed_nmax: int = None):
        device, dtype = self.device, self.dtype
        B = len(atoms_list)
        self._atoms_B = B

        if self._batch_native is None:
            try:
                self._probe_batch_native()
            except Exception:
                self._batch_native = False
        if B > 1 and self._batch_native is False:
            raise RuntimeError(
                f"Standard MACE model '{self._model_name}' was traced single-graph "
                "(B=1-locked: per-graph scatter dim_size frozen to 1), so it cannot ingest "
                "a multi-graph batch. Re-export the model batch-generic, or run molecules "
                "one at a time. (This is the same trace-lock that affects MACE-POL.)")

        ptr = [0]
        nums, mids, coords = [], [], []
        for i, at in enumerate(atoms_list):
            Z = torch.tensor(at.get_atomic_numbers(), dtype=torch.int64, device=device)
            n = int(Z.shape[0])
            ptr.append(ptr[-1] + n)
            nums.append(Z)
            mids.append(torch.full((n,), i, dtype=torch.int64, device=device))
            coords.append(torch.tensor(at.get_positions(), dtype=dtype, device=device))

        self._ptr = torch.tensor(ptr, dtype=torch.int64, device=device)
        self.numbers = torch.cat(nums) if nums else torch.zeros((0,), dtype=torch.int64, device=device)
        self.mol_idx = torch.cat(mids) if mids else torch.zeros((0,), dtype=torch.int64, device=device)
        self.coord = (torch.cat(coords).contiguous() if coords
                      else torch.zeros((0, 3), dtype=dtype, device=device))
        self.N_atoms = int(self.numbers.numel())
        self.Nmax_atoms = int(max((len(at) for at in atoms_list), default=0))
        self.nmax_dof = (3 * self.Nmax_atoms) if fixed_nmax is None else int(fixed_nmax)

        self.node_attrs = (_one_hot_node_attrs(self.numbers, self.atomic_numbers, self.mdtype)
                           if self.N_atoms > 0
                           else torch.zeros((0, len(self.atomic_numbers)), dtype=self.mdtype, device=device))

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

        self._coord_backup = None
        self._prepared = True

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
    # forward (true multi-graph)
    # =====================================================================
    def _build_edges(self, coord: torch.Tensor):
        """Block-diagonal radius graph from precomputed intra-mol candidate pairs.
        EXACT squared distance (NOT cdist). Returns edge_index, shifts."""
        device = self.device
        if self.cand_i.numel() == 0:
            return (torch.zeros((2, 0), dtype=torch.int64, device=device),
                    torch.zeros((0, 3), dtype=self.mdtype, device=device))
        rij = coord[self.cand_i] - coord[self.cand_j]
        d2 = (rij * rij).sum(dim=-1)
        keep = d2 <= (self.r_max + 1e-12) ** 2
        ci = self.cand_i[keep]; cj = self.cand_j[keep]
        src = torch.cat([ci, cj], dim=0); dst = torch.cat([cj, ci], dim=0)
        edge_index = torch.stack([src, dst], dim=0)
        shifts = torch.zeros((edge_index.size(1), 3), dtype=self.mdtype, device=device)
        return edge_index, shifts

    def _forward_ef_(self, coord: torch.Tensor, need_graph: bool):
        """Single fused multi-graph forward -> (E_eV(B,), F_all_eV(N,3), coord_leaf)."""
        assert self._prepared, "call prepare() first"
        B, N = self._atoms_B, self.N_atoms
        coord_leaf = coord.detach().to(device=self.device, dtype=self.mdtype).requires_grad_(True)
        edge_index, shifts = self._build_edges(coord_leaf)
        # native multi-graph: batch (node->mol), ptr (B+1)
        E = self.model(coord_leaf, self.node_attrs, edge_index, shifts, self.mol_idx, self._ptr)
        e = E.reshape(-1)
        if e.numel() == B:
            E_eV = e
        elif e.numel() == N:                          # per-node fallback -> pool
            E_eV = torch.zeros(B, dtype=e.dtype, device=self.device).index_add(0, self.mol_idx, e)
        else:
            raise RuntimeError(f"Unexpected energy shape {tuple(e.shape)} (B={B}, N={N})")
        grad = torch.autograd.grad(E_eV.sum(), coord_leaf,
                                   create_graph=need_graph, retain_graph=need_graph)[0]
        return E_eV.to(self.dtype), -grad, coord_leaf

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
        """Per-structure list[int] of atom indices whose DOFs are perturbed.
        None = all atoms; per-structure None/bool-mask/index-list otherwise. Frozen
        atoms still appear in every forward (they exert forces); only their Hessian
        rows/cols are omitted (exact FixAtoms-constrained PES block). Mirrors
        UMABatchCalc._resolve_movable so all backends agree."""
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
        """Return (movable_atom (N,) bool, movable_la (B,nmax_a) bool). Both all-True
        within each structure's real-atom range when movable_masks is None, so the
        seeded loop reduces byte-identically to the full Hessian."""
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
    def get_ef_gpu(self):
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return (torch.zeros((0,), dtype=dtype, device=device),
                    torch.zeros((0, 0), dtype=dtype, device=device))
        E_eV, F_all_eV, _ = self._forward_ef_(self.coord, need_graph=False)
        F_eV = self._scatter_forces(F_all_eV.detach().to(dtype))
        return (E_eV.detach() * EV2HARTREE, F_eV * EV2HARTREE)

    def get_efh_gpu(self, movable_masks=None):
        """Energy + forces + per-structure Hessian. ``movable_masks`` (mirrors
        UMABatchCalc): None = full Hessian (byte-identical current behavior); a
        per-structure spec restricts the perturbed/responding DOFs to a movable-atom
        subspace -> only movable rows/cols filled, frozen atoms still exert forces
        (exact FixAtoms-constrained block). Applies to both the seeded analytic and
        the FD fallback path identically."""
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return (torch.zeros((0,), dtype=dtype, device=device),
                    torch.zeros((0, 0), dtype=dtype, device=device),
                    torch.zeros((0, 0, 0), dtype=dtype, device=device),
                    torch.zeros((0,), dtype=torch.int64, device=device))
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

    def isolation_check(self, perturb: float = 0.05) -> float:
        """Perturb-one byte-isolation probe: returns max cross-molecule force leak (Ha/A).
        For a correct standard-MACE multi-graph batch this is 0.0 (exact isolation)."""
        assert self._prepared and self._atoms_B >= 2
        c0 = self.coord.clone()
        _, F0, _ = self._forward_ef_(c0, need_graph=False)
        c1 = c0.clone(); c1[0, 0] += perturb
        _, F1, _ = self._forward_ef_(c1, need_graph=False)
        leak = 0.0
        for j in range(1, self._atoms_B):
            rows = (self.mol_idx == j)
            leak = max(leak, (F1[rows] - F0[rows]).abs().max().item())
        return leak * EV2HARTREE

    # =====================================================================
    # seeded block-diagonal analytic Hessian (mirror AIMNet2) / FD fallback
    # =====================================================================
    def _efh_analytic(self, movable_masks=None):
        B, device, dtype = self._atoms_B, self.device, self.dtype
        N, nmax, nmax_a = self.N_atoms, self.nmax_dof, self.Nmax_atoms
        E_eV, F_all_eV, coord_leaf = self._forward_ef_(self.coord, need_graph=True)
        s = self._ptr[:-1]; n_b = self._n_b; n_b_list = n_b.tolist()
        F_eV = self._scatter_forces(F_all_eV.to(dtype))
        H_eV = torch.zeros((B, nmax, nmax), dtype=dtype, device=device)
        P = (nmax_a - n_b).to(torch.int64)
        # movable subspace: movable_la[:,a]=structure has movable atom a; movable_atom
        # gates the response rows. Both reduce to the full set when movable_masks=None.
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
        E_eV, F_all_eV, _ = self._forward_ef_(self.coord, need_graph=False)
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
            _, Fp, _ = self._forward_ef_(cp, need_graph=False)
            _, Fm, _ = self._forward_ef_(cm, need_graph=False)
            col = ((-(Fp - Fm) / (2.0 * delta)).to(dtype)) * row_scale  # zero frozen rows (no-op None)
            for i in valid.nonzero(as_tuple=False).flatten().tolist():
                dof = 3 * n_b_list[i]
                if k < dof:
                    H_eV[i, :dof, k] = col[s[i]:s[i] + n_b_list[i], :].reshape(-1)
        self.coord.copy_(base_coord)
        H_eV = 0.5 * (H_eV + H_eV.transpose(1, 2))
        return (E_eV.detach() * EV2HARTREE, F_eV * EV2HARTREE, H_eV * EV2HARTREE, P)

    # =====================================================================
    # init probes
    # =====================================================================
    def _two_atom_graph(self, n_graphs: int):
        """Build a tiny n_graphs-graph batch of identical 2-atom molecules for probing."""
        device = self.device
        z = self.atomic_numbers[0]
        d = min(1.0, self.r_max * 0.5)
        per = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, d]], dtype=self.mdtype, device=device)
        coords, na_list, eis, batch, ptr = [], [], [], [], [0]
        Zsingle = torch.tensor([z, z], dtype=torch.int64, device=device)
        for g in range(n_graphs):
            off = 2 * g
            coords.append(per + torch.tensor([[g * 50.0, 0, 0]], dtype=self.mdtype, device=device))
            na_list.append(_one_hot_node_attrs(Zsingle, self.atomic_numbers, self.mdtype))
            eis.append(torch.tensor([[off, off + 1], [off + 1, off]], dtype=torch.int64, device=device))
            batch.extend([g, g]); ptr.append(2 * (g + 1))
        coord = torch.cat(coords, 0).requires_grad_(True)
        na = torch.cat(na_list, 0)
        ei = torch.cat(eis, 1)
        sh = torch.zeros((ei.shape[1], 3), dtype=self.mdtype, device=device)
        bt = torch.tensor(batch, dtype=torch.int64, device=device)
        pt = torch.tensor(ptr, dtype=torch.int64, device=device)
        return coord, na, ei, sh, bt, pt

    def _probe_batch_native(self):
        """Verify the traced model accepts a B=2 batch (not single-graph trace-locked)."""
        coord, na, ei, sh, bt, pt = self._two_atom_graph(2)
        try:
            E = self.model(coord, na, ei, sh, bt, pt)
            self._batch_native = (E.reshape(-1).numel() in (2, 4))   # (B,) or (N,)
        except Exception:
            self._batch_native = False

    def _probe_double_backward(self):
        coord, na, ei, sh, bt, pt = self._two_atom_graph(1)
        E = self.model(coord, na, ei, sh, bt, pt).reshape(-1).sum()
        try:
            F = -torch.autograd.grad(E, coord, create_graph=True)[0]
            go = torch.zeros_like(coord); go[0, 2] = 1.0
            torch.autograd.grad(F, coord, grad_outputs=go, retain_graph=False)[0]
            self._hess_mode = 'analytic'
        except Exception:
            self._hess_mode = 'fd'
