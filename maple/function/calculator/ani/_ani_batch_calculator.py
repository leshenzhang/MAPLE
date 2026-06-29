# -*- coding: utf-8 -*-
"""Batched ANI (torchani) calculator for the MAPLE GPU-batch comp-chem library.

Goal
----
Batch many small molecules through an ANI TorchScript model in ONE forward to
maximize GPU utilization, mirroring the duck-typed batch contract of
``UMABatchCalc`` / ``MACEBatchCalc`` / ``MACEAutogradBatchCalc`` /
``AIMNet2BatchCalc``. There is NO hand-written CUDA -- parallelism comes entirely
from torchani's *native* (species, coordinates) batch dimension.

Why ANI batches EXACTLY (local potential, cleaner than MACE)
-----------------------------------------------------------
ANI (ANI-1x / ANI-1ccx / ANI-2x / ANI-1xnr) is a PURE LOCAL potential: a per-atom
AEV descriptor feeds per-element neural networks and the molecular energy is the
SUM of per-atom energies. There is NO charge equilibration and NO long-range
Coulomb head. torchani represents a batch as ``species (B, A)`` + ``coordinates
(B, A, 3)`` where every ROW is an independent molecule -- atoms in different rows
NEVER enter each other's neighbour list (the AEV is computed per row). So a padded
multi-molecule batch is isolated *by construction* -- even more trivially than the
standard-MACE block-diagonal multi-graph batch (which needs per-molecule offset
edges). Padding atoms use torchani's universal dummy convention ``species = -1``
(the wrapped ``SpeciesConverter`` maps atomic-number -1 to internal index -1, and
``AEVComputer`` / ``ANIModel`` skip dummy atoms), so a padded row's dummy atoms
contribute zero energy and receive zero force. The perturb-one / 3-identical-far-
apart decouple gate gives dE_others == 0 and batch E/F == isolated E/F to fp tol.

Hartree-native units
--------------------
ANI's TorchScript checkpoints return Hartree directly (MODEL_ENERGY_UNIT =
'hartree' in ``_ani_calculator.py``); coordinates are Angstrom. So ``E`` is Ha,
``F = -dE/dx`` is Ha/A, and the Hessian ``d^2E/dx^2`` is Ha/A^2 with NO eV->Ha
conversion (contrast UMA/MACE which multiply by EV2HARTREE). The model wrapper is
float32 (matching the single ``ANICalculator``), so the forward bridges through
f32 while the master coordinate tensor is kept f64 for the contract; E/F/H are
returned in f64.

Forces and Hessian are BOTH autograd (torchani is fully twice-differentiable)
-----------------------------------------------------------------------------
  * forces      = ``-autograd.grad(E.sum(), coord_leaf)``                 (1 backward)
  * Hessian     = seeded block-diagonal double-backward of ``F = -dE/dx``
                  over the unit basis ``eye(3*nmax_a)`` (ONE forward + 1 + 3*nmax_a
                  backward), the same pattern as ``MACEAutogradBatchCalc``.
  * H @ v        = ``autograd.grad((g . v).sum(), x)`` -- one forward + two backward,
                  no dense Hessian (BPRFO / dimer consumer).
Unlike UMA's eSCN backbone (whose custom ops give an INCOMPLETE double-backward,
so its autograd Hessian deviates ~1e-2 Ha/A^2 from FD), ANI is a plain MLP stack
-> the autograd Hessian matches the finite-difference oracle to ~1e-5 with clean
delta->0 convergence.

Hessian mode (opt-in; OLD numerical FD is the DEFAULT + byte-parity oracle)
---------------------------------------------------------------------------
``get_efh_gpu`` dispatches on ``hessian_mode`` (user 铁则: every new method is
opt-in, the default is the old behaviour and doubles as the parity oracle):
  * ``'numerical'`` (DEFAULT) -- ``_efh_fd``, the 6N central finite-difference
    Hessian; kept verbatim as both the fallback AND the oracle the autograd path
    is validated against.
  * ``'autograd'`` (OPT-IN)   -- ``_efh_analytic``, the seeded block-diagonal
    double-backward Hessian, with an optional ``torch.vmap`` over the seed basis
    (``hessian_chunk_size``) and a hard try/except fall-back to the FD oracle on
    any failure. Model is run EAGER (torch.jit traced graph still supports
    double-backward for these MLP models; on any failure it falls back).

Contract (mirrors UMABatchCalc / MACEBatchCalc / AIMNet2BatchCalc EXACTLY)
-------------------------------------------------------------------------
    prepare(atoms_list, fixed_nmax=None)
    step_cart_(s_cart: (B, nmax_dof)) / set_coords_(coord: (N,3))
    backup_coords() / restore_coords()
    get_ef_gpu()  -> (E_Ha (B,), F_Ha (B, nmax_dof))
    get_efh_gpu(movable_masks=None)
                  -> (E_Ha (B,), F_Ha (B, nmax_dof),
                      H_Ha (B, nmax_dof, nmax_dof), P (B,) int64)
    hvp(v: (B, nmax_dof)) -> Hn_Ha (B, nmax_dof)
Units: Hartree, Hartree/Angstrom, Hartree/Angstrom^2. Gas phase only (no PBC).

References for the autograd Hessian / HVP
-----------------------------------------
  * Yuan et al., Nat. Commun. 2024, DOI 10.1038/s41467-024-52481-5 -- autograd
    (double-backward) molecular Hessians, exact and free of FD truncation error.
  * MACE ``compute_hessians_vmap`` (github ACEsuit/mace, mace/modules/utils.py):
    ``torch.vmap`` of ``autograd.grad`` over ``eye(3N)`` + per-row loop fallback
    (issue #488) -- the vmap+fallback pattern mirrored here.
  * torchani (github aiqm/torchani) -- native (species, coordinates) batching with
    species == -1 dummy-atom padding; energies/forces/Hessians via torch autograd.
"""

import os
from typing import List, Optional, Sequence

import numpy as np
import torch
from ase import Atoms


class ANIBatchCalc:
    """Batched ANI (torchani) calculator -- native (species, coords) batching.

    One ``prepare()`` fixes the topology (padded species + scatter maps) of a
    batch of B molecules; thereafter ``get_ef_gpu`` / ``get_efh_gpu`` run a single
    batched forward (Hessian = seeded double-backward or chunked FD). The ANI
    TorchScript wrapper is f32; the master coordinate tensor is f64 and bridges
    through f32 for the forward (matching the single ``ANICalculator``). ANI is
    Hartree-native, so outputs are Hartree / Hartree-per-Angstrom directly.
    """

    supported_hessian_modes = ("numerical", "autograd")

    def __init__(self,
                 model_path: str,
                 device: str = "cuda",
                 model: str = "ani2x",
                 dtype: torch.dtype = torch.float64,
                 hessian_mode: Optional[str] = None,
                 hessian: Optional[str] = None,
                 hessian_chunk_size: Optional[int] = None,
                 d4: bool = False,
                 implicit: str = "none",
                 solvent: str = "none"):
        if str(solvent).lower() not in ("none", "", "vacuum", "gas"):
            raise ValueError(
                f"ANIBatchCalc is gas-phase ONLY; got solvent={solvent!r}.")
        if str(implicit).lower() not in ("none", "", "false", "0"):
            raise ValueError(
                f"ANIBatchCalc does not support implicit solvent; got "
                f"implicit={implicit!r}.")
        if d4:
            raise NotImplementedError(
                "ANIBatchCalc does not implement batched D4 dispersion; use the "
                "single ANICalculator for d4=True, or run d4 as a separate "
                "per-molecule correction.")

        dev = str(device)
        dev = "cuda" if dev.startswith("cuda") else "cpu"
        self.device = torch.device(dev)
        self.dtype = dtype                 # master coord dtype (f64)
        self.mdtype = torch.float32        # ANI TorchScript wrapper is f32
        self._model_name = model

        if model_path is None:
            model_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.dirname(model_dir)
            model_path = os.path.join(model_dir, 'model', f'{model}.pt')
        self._model_path = model_path

        # ---- Hessian mode resolution (OPT-IN discipline) -------------------
        # Canonical knob = ``hessian_mode`` in {'numerical','autograd'}. Legacy
        # ``hessian`` ('analytic'/'numerical') is honoured for back-compat. Default
        # 'numerical' = the 6N finite-difference path that doubles as the oracle.
        _raw = hessian_mode if hessian_mode is not None else hessian
        if _raw is None:
            _raw = "numerical"
        _m = str(_raw).lower()
        self.hessian_mode = "autograd" if _m in ("autograd", "analytic") else "numerical"
        self.hessian = self.hessian_mode             # back-compat attribute
        self._hessian_chunk_size = (int(hessian_chunk_size)
                                    if hessian_chunk_size is not None else None)

        self.model = torch.jit.load(model_path, map_location=self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._n_forward = 0   # model-forward counter (algorithmic-cost verification)

        # prepare() state
        self._prepared = False
        self._atoms_B = 0
        self._ptr = None
        self.numbers = None
        self.mol_idx = None
        self._local_idx = None
        self.coord = None
        self.N_atoms = 0
        self.Nmax_atoms = 0
        self.nmax_dof = 0
        self._base = None
        self._n_b = None
        self._species_pad = None
        self._coord_backup = None

    # ------------------------------------------------------------------ utils
    def reset_fwd_count(self):
        self._n_forward = 0

    @property
    def n_forward(self) -> int:
        return self._n_forward

    # =====================================================================
    def prepare(self, atoms_list: List[Atoms], fixed_nmax: int = None):
        """Fix topology (padded species + scatter maps) + initial coords.

        ``fixed_nmax`` (optional) overrides the padded per-structure DOF size so a
        batch-PRFO driver can share one padded layout across iterations (matches
        the UMA/MACE/AIMNet2 batch semantics).
        """
        device, dtype = self.device, self.dtype
        B = len(atoms_list)
        self._atoms_B = B

        if any(bool(np.any(getattr(at, "pbc", False))) for at in atoms_list):
            raise NotImplementedError(
                "ANIBatchCalc is a no-PBC molecular batch calculator.")

        ptr = [0]
        nums, mids, locs, coords = [], [], [], []
        for i, at in enumerate(atoms_list):
            Z = torch.tensor(at.get_atomic_numbers(), dtype=torch.int64, device=device)
            n = int(Z.shape[0])
            ptr.append(ptr[-1] + n)
            nums.append(Z)
            mids.append(torch.full((n,), i, dtype=torch.int64, device=device))
            locs.append(torch.arange(n, dtype=torch.int64, device=device))
            coords.append(torch.tensor(at.get_positions(), dtype=dtype, device=device))

        self._ptr = torch.tensor(ptr, dtype=torch.int64, device=device)
        self.numbers = torch.cat(nums) if nums else torch.zeros((0,), dtype=torch.int64, device=device)
        self.mol_idx = torch.cat(mids) if mids else torch.zeros((0,), dtype=torch.int64, device=device)
        self._local_idx = torch.cat(locs) if locs else torch.zeros((0,), dtype=torch.int64, device=device)
        self.coord = (torch.cat(coords).contiguous() if coords
                      else torch.zeros((0, 3), dtype=dtype, device=device))
        self.N_atoms = int(self.numbers.numel())
        self.Nmax_atoms = int(max((len(at) for at in atoms_list), default=0))
        self.nmax_dof = (3 * self.Nmax_atoms) if fixed_nmax is None else int(fixed_nmax)

        s = self._ptr[:-1]
        t = self._ptr[1:]
        self._n_b = (t - s)

        # scatter map: global atom g (mol b, local a) -> flat (B, nmax_dof) base col
        if self.N_atoms > 0:
            self._base = self.mol_idx * self.nmax_dof + 3 * self._local_idx
        else:
            self._base = torch.zeros((0,), dtype=torch.int64, device=device)

        # padded species (B, Amax) with -1 dummy padding (geometry independent).
        nmax_a = self.Nmax_atoms
        self._species_pad = torch.full((B, nmax_a), -1, dtype=torch.int64, device=device)
        if self.N_atoms > 0:
            self._species_pad[self.mol_idx, self._local_idx] = self.numbers

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
    # forward (native species/coordinates batching)
    # =====================================================================
    def _padded_positions(self, coord_leaf: torch.Tensor) -> torch.Tensor:
        """Differentiable scatter of flat real-atom coords (N,3) -> (B, Amax, 3).

        Dummy padding rows keep coords 0 (their species is -1 -> torchani skips
        them, so the value is irrelevant). The advanced-index assignment keeps the
        autograd graph from ``coord_leaf`` through ``pos_pad`` to the energy, so
        forces differentiate back to the real-atom flat tensor directly.
        """
        B, nmax_a = self._atoms_B, self.Nmax_atoms
        pos_pad = coord_leaf.new_zeros((B, nmax_a, 3))
        if self.N_atoms > 0:
            pos_pad = pos_pad.index_put((self.mol_idx, self._local_idx), coord_leaf)
        return pos_pad

    def _model_energy(self, species_pad: torch.Tensor, pos_pad: torch.Tensor) -> torch.Tensor:
        """One ANI forward -> per-structure energy (B,) [Hartree]. Counts a forward.

        Mirrors the single ``ANICalculator``: ``self.model(species, coordinates)[0]``
        is the energy. For a B-row batch torchani returns energies of shape (B,).
        """
        self._n_forward += 1
        out = self.model(species_pad, pos_pad)
        e = out[0] if isinstance(out, (tuple, list)) else out
        return e.reshape(-1)

    def _forward_ef_(self, coord: torch.Tensor, need_graph: bool):
        """Single fused batched forward -> (E_Ha (B,), F_all_Ha (N,3), coord_leaf).

        ``coord_leaf`` is the flat (N,3) real-atom tensor in f32 (model dtype);
        ``F_all = -dE/dx`` carries a graph when ``need_graph`` (for the seeded
        Hessian / HVP). Hartree-native: no unit conversion.
        """
        assert self._prepared, "call prepare() first"
        B, N = self._atoms_B, self.N_atoms
        coord_leaf = coord.detach().to(device=self.device, dtype=self.mdtype).requires_grad_(True)
        pos_pad = self._padded_positions(coord_leaf)
        E = self._model_energy(self._species_pad, pos_pad)
        if E.numel() != B:
            raise RuntimeError(
                f"ANI batch forward returned energy of shape {tuple(E.shape)} "
                f"(expected B={B}); the model wrapper may not accept a B>1 batch.")
        if N == 0:
            return E.to(self.dtype), torch.zeros((0, 3), dtype=self.dtype, device=self.device), coord_leaf
        g = torch.autograd.grad(E.sum(), coord_leaf,
                                create_graph=need_graph, retain_graph=need_graph)[0]
        return E.to(self.dtype), (-g), coord_leaf

    def _scatter_forces(self, F_all: torch.Tensor) -> torch.Tensor:
        """(N,3) per-atom -> (B, nmax_dof) padded per structure."""
        B = self._atoms_B
        F_flat = torch.zeros(B * self.nmax_dof, dtype=self.dtype, device=self.device)
        if self.N_atoms > 0:
            base = self._base
            Fd = F_all.to(self.dtype)
            F_flat[base] = Fd[:, 0]
            F_flat[base + 1] = Fd[:, 1]
            F_flat[base + 2] = Fd[:, 2]
        return F_flat.view(B, self.nmax_dof)

    def _gather_pad_to_atoms(self, v_pad: torch.Tensor) -> torch.Tensor:
        """Inverse of _scatter_forces: (B, nmax_dof) -> (N,3) per-atom."""
        v_flat = v_pad.reshape(-1).to(self.device, self.mdtype)
        base = self._base
        return torch.stack([v_flat[base], v_flat[base + 1], v_flat[base + 2]], dim=1)

    # =====================================================================
    def get_ef_gpu(self):
        """Energy + forces from one batched forward (1 forward + 1 backward)."""
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return (torch.zeros((0,), dtype=dtype, device=device),
                    torch.zeros((0, 0), dtype=dtype, device=device))
        E_Ha, F_all_Ha, _ = self._forward_ef_(self.coord, need_graph=False)
        F_Ha = self._scatter_forces(F_all_Ha.detach())
        return (E_Ha.detach(), F_Ha)

    def get_efh_gpu(self, movable_masks=None):
        """Energy + forces + per-structure Hessian.

        Mode dispatch (OPT-IN): default 'numerical' = the central-FD Hessian
        (oracle); 'autograd' = the seeded double-backward Hessian (tried with the
        optional vmap chunk first), hard-falling back to FD on any failure.
        ``movable_masks`` (mirrors UMA/MACE) restricts the perturbed/responding DOFs
        to a per-structure movable-atom subspace (None = full Hessian).
        """
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return (torch.zeros((0,), dtype=dtype, device=device),
                    torch.zeros((0, 0), dtype=dtype, device=device),
                    torch.zeros((0, 0, 0), dtype=dtype, device=device),
                    torch.zeros((0,), dtype=torch.int64, device=device))
        if self.hessian_mode == "numerical":
            return self._efh_fd(movable_masks=movable_masks)
        if self._hessian_chunk_size is not None:
            try:
                return self._efh_analytic(movable_masks, chunk_size=self._hessian_chunk_size)
            except Exception:
                pass
        try:
            return self._efh_analytic(movable_masks)
        except Exception:
            return self._efh_fd(movable_masks=movable_masks)

    # =====================================================================
    # partial-Hessian (movable-atom subspace) helpers -- mirror MACEBatchCalc
    # =====================================================================
    def _resolve_movable(self, movable_masks):
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
    # seeded block-diagonal analytic Hessian (double-backward) / FD fallback
    # =====================================================================
    def _efh_analytic(self, movable_masks=None, chunk_size: Optional[int] = None):
        """Full block-diagonal Hessian by seeded DOUBLE-BACKWARD. 1 forward +
        (1 + 3*nmax_a) backward (vs FD's 6N forwards). ANI is Hartree-native."""
        B, device, dtype = self._atoms_B, self.device, self.dtype
        N, nmax, nmax_a = self.N_atoms, self.nmax_dof, self.Nmax_atoms
        E_Ha, F_all_Ha, coord_leaf = self._forward_ef_(self.coord, need_graph=True)
        s = self._ptr[:-1]
        n_b = self._n_b
        n_b_list = n_b.tolist()
        F_Ha = self._scatter_forces(F_all_Ha)
        H_Ha = torch.zeros((B, nmax, nmax), dtype=dtype, device=device)
        P = (nmax_a - n_b).to(torch.int64)
        movable_atom, movable_la = self._movable_tensors(movable_masks)
        row_scale = movable_atom[:, None].to(F_all_Ha.dtype)
        K = 3 * nmax_a

        def _scatter_col(k, col):
            a_local = k // 3
            for i in movable_la[:, a_local].nonzero(as_tuple=False).flatten().tolist():
                dof = 3 * n_b_list[i]
                if k < dof:
                    H_Ha[i, :dof, k] = -col[s[i]:s[i] + n_b_list[i], :].reshape(-1).to(dtype)

        if chunk_size is not None and K > 0:
            seeds = []
            for k in range(K):
                a_local, c = k // 3, k % 3
                go = torch.zeros((N, 3), dtype=F_all_Ha.dtype, device=device)
                valid = movable_la[:, a_local]
                if bool(valid.any()):
                    go[s[valid] + a_local, c] = 1.0
                seeds.append(go)
            GO = torch.stack(seeds, dim=0)

            def _get_vjp(v):
                return torch.autograd.grad(F_all_Ha, coord_leaf, grad_outputs=v,
                                           retain_graph=True, create_graph=False)[0]

            COLS = torch.vmap(_get_vjp, in_dims=0, out_dims=0,
                              chunk_size=int(chunk_size))(GO)
            COLS = COLS * row_scale[None]
            for k in range(K):
                if bool(movable_la[:, k // 3].any()):
                    _scatter_col(k, COLS[k])
        else:
            for k in range(K):
                a_local, c = k // 3, k % 3
                valid = movable_la[:, a_local]
                if not bool(valid.any()):
                    continue
                rows = s[valid] + a_local
                go = torch.zeros((N, 3), dtype=F_all_Ha.dtype, device=device)
                go[rows, c] = 1.0
                col = torch.autograd.grad(F_all_Ha, coord_leaf, grad_outputs=go,
                                          retain_graph=True, create_graph=False)[0]
                col = col * row_scale
                _scatter_col(k, col)
        H_Ha = 0.5 * (H_Ha + H_Ha.transpose(1, 2))
        return (E_Ha.detach(), F_Ha, H_Ha, P)

    def _efh_fd(self, delta: float = 2e-3, movable_masks=None):
        """Full numerical Hessian (central FD). 1 + 2*3*nmax_a forwards. Oracle."""
        B, device, dtype = self._atoms_B, self.device, self.dtype
        nmax, nmax_a = self.nmax_dof, self.Nmax_atoms
        E0, F0, _ = self._forward_ef_(self.coord, need_graph=False)
        base_coord = self.coord.clone()
        s = self._ptr[:-1]
        n_b = self._n_b
        n_b_list = n_b.tolist()
        F_Ha = self._scatter_forces(F0.detach())
        H_Ha = torch.zeros((B, nmax, nmax), dtype=dtype, device=device)
        P = (nmax_a - n_b).to(torch.int64)
        movable_atom, movable_la = self._movable_tensors(movable_masks)
        row_scale = movable_atom[:, None].to(dtype)
        for k in range(3 * nmax_a):
            a_local, c = k // 3, k % 3
            valid = movable_la[:, a_local]
            if not bool(valid.any()):
                continue
            rows = s[valid] + a_local
            cp = base_coord.clone(); cp[rows, c] += delta
            cm = base_coord.clone(); cm[rows, c] -= delta
            _, Fp, _ = self._forward_ef_(cp, need_graph=False)
            _, Fm, _ = self._forward_ef_(cm, need_graph=False)
            col = ((-(Fp - Fm) / (2.0 * delta)).to(dtype)) * row_scale
            for i in valid.nonzero(as_tuple=False).flatten().tolist():
                dof = 3 * n_b_list[i]
                if k < dof:
                    H_Ha[i, :dof, k] = col[s[i]:s[i] + n_b_list[i], :].reshape(-1)
        self.coord.copy_(base_coord)
        H_Ha = 0.5 * (H_Ha + H_Ha.transpose(1, 2))
        return (E0.detach(), F_Ha, H_Ha, P)

    # =====================================================================
    # batched autograd HVP (no dense Hessian) -- BPRFO / dimer consumer
    # =====================================================================
    def hvp_batch(self, v_pad: torch.Tensor):
        """Batched Hessian-vector product H@v for all B structures at once.

        Cost = 1 model forward + 2 backward. ``v_pad`` is (B, nmax_dof) padded
        (forces layout); returns (Hn_pad (B, nmax_dof) [Ha/A^2], E_Ha (B,) [Ha]).
        """
        assert self._prepared, "call prepare() first"
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return (torch.zeros((0, 0), dtype=dtype, device=device),
                    torch.zeros((0,), dtype=dtype, device=device))
        E_Ha, F_all_Ha, coord_leaf = self._forward_ef_(self.coord, need_graph=True)
        g = -F_all_Ha                                              # dE/dx (N,3)
        v_atom = self._gather_pad_to_atoms(v_pad)                  # (N,3)
        Hn_atom = torch.autograd.grad((g * v_atom).sum(), coord_leaf,
                                      retain_graph=False)[0]        # H@v (N,3)
        Hn_pad = self._scatter_forces(Hn_atom)
        return Hn_pad, E_Ha.detach()

    def hvp(self, v: torch.Tensor) -> torch.Tensor:
        """Canonical batched Hessian-vector product ``H @ v`` (forces layout)."""
        Hn_pad, _ = self.hvp_batch(v)
        return Hn_pad

    # =====================================================================
    # diagnostics: the decouple / isolation gate
    # =====================================================================
    def isolation_check(self, perturb: float = 0.05) -> float:
        """Perturb-one cross-molecule force leak (Ha/A). 0.0 for a correct local
        batch (ANI rows are independent -> exact isolation by construction)."""
        assert self._prepared and self._atoms_B >= 2
        c0 = self.coord.clone()
        _, F0, _ = self._forward_ef_(c0, need_graph=False)
        c1 = c0.clone(); c1[0, 0] += perturb
        _, F1, _ = self._forward_ef_(c1, need_graph=False)
        leak = 0.0
        for j in range(1, self._atoms_B):
            rows = (self.mol_idx == j)
            leak = max(leak, (F1[rows] - F0[rows]).abs().max().item())
        return leak

    def energy_of(self, mol_index: int) -> float:
        """Per-molecule energy [Ha] (for the perturb-one dE isolation gate)."""
        E_Ha, _, _ = self._forward_ef_(self.coord, need_graph=False)
        return float(E_Ha[mol_index].item())
