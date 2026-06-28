# -*- coding: utf-8 -*-
"""Decoupled AIMNet2 batch calculator (cross-reaction batchable backend).

Out-of-scope for the unified MAPLE calculator protocol (same status as the
sibling ``_aimnet2_batch_calculator.AIMNet2BatchCalc``): it implements only the
duck-typed *batch contract* consumed by ``NEB.run_multiband`` / ``BatchPRFO`` /
``BatchLBFGS``::

    prepare(atoms_list, fixed_nmax=None)   # set topology + initial coords
    get_ef_gpu()  -> (E (B,) Ha, F (B, nmax_dof) Ha/Angstrom)
    get_e_gpu()   -> E (B,) Ha                      # energy-only single point

WHY A SEPARATE CALC.  The bundled ``aimnet2.pt`` (SanderAIMNet2_2025) performs a
*global* charge-equilibration over the whole forward, so co-batching independent
molecules CObiquitously COUPLES them (verified: 3 identical far-apart copies
disagree by ~0.16 Ha; even NEB images within one band corrupt each other ->
garbage ~30 eV barriers). That model is therefore NOT cross-reaction batchable.

This calc instead wraps the upstream ``aimnet2calc.AIMNet2Calculator`` loaded
with a NON charge-equilibration model (default ``aimnet2`` ->
``aimnet2_wb97m_0.jpt``, cutoff 5.0). Its neighbour graph is segmented by
``mol_idx`` (``radius_graph`` batch=mol_idx) so NO edge ever crosses a molecule
boundary; with a short-range (non-Ewald) energy this makes every structure's
E/F exactly independent of its co-batched peers -- i.e. cross-reaction
batching is decoupled (perturb-one probe: dE_other < 1e-6 Ha).

The wrapped high-level API casts coordinates to float32 internally, so energies
are float32-precision (the decoupling is *structural*, i.e. zero cross terms,
not precision-limited, so the 1e-4 Ha decouple gate still passes by orders of
magnitude). Results are returned as float64 tensors for the MAPLE contract.

Runtime: requires ``aimnet2calc`` importable and ``PYTHONNOUSERSITE=1``.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from ase import Atoms

EH2EV = 27.211386245988  # eV per Hartree


def _ptr_from_atoms(atoms_list: List[Atoms], device) -> torch.Tensor:
    ptr = [0]
    for at in atoms_list:
        ptr.append(ptr[-1] + len(at))
    return torch.tensor(ptr, dtype=torch.long, device=device)


class AIMNet2DecoupledBatchCalc:
    """Batched, cross-reaction-decoupled AIMNet2 calculator.

    Parameters
    ----------
    model : str
        aimnet2calc model name or path. Default ``"aimnet2"`` ->
        ``aimnet2_wb97m_0.jpt`` (the validated decoupled model). Pass an
        explicit ``.jpt`` path to pin a specific checkpoint.
    device : str | None
        Torch device. ``None`` -> the aimnet2calc auto-selected device.
    dtype : torch.dtype
        Output dtype for the returned tensors (default float64 per the MAPLE
        contract). Note the underlying forward runs in float32.
    """

    def __init__(self, model: str = "aimnet2", device: Optional[str] = "cuda",
                 dtype: torch.dtype = torch.float64):
        from aimnet2calc import AIMNet2Calculator  # lazy: keep maple import light

        self.aim = AIMNet2Calculator(model)
        self.device = torch.device(self.aim.device if device is None else device)
        self.dtype = dtype
        self.cutoff = float(self.aim.cutoff)

        # prepared-topology buffers
        self._prepared = False
        self._atoms_B = 0
        self._ptr = None
        self.numbers = None         # (N,) int64
        self.mol_idx = None         # (N,) int64
        self.coord = None           # (N,3) dtype
        self.charge_B = None        # (B,) per-structure total charge
        self.mult_B = None          # (B,) per-structure multiplicity
        self.N_atoms = 0
        self.Nmax_atoms = 0
        self.nmax_dof = 0
        self._cart_idx_flat = None  # (3N,) scatter map real-atom DOF -> (B,nmax_dof) flat
        self._coord_backup = None

    # ------------------------------------------------------------------ prepare
    def prepare(self, atoms_list: List[Atoms], fixed_nmax: Optional[int] = None):
        device, dtype = self.device, self.dtype
        if any(bool(np.any(getattr(at, "pbc", False))) for at in atoms_list):
            raise NotImplementedError(
                "AIMNet2DecoupledBatchCalc is a no-PBC batch wrapper.")

        self._atoms_B = len(atoms_list)
        self._ptr = _ptr_from_atoms(atoms_list, device)

        nums, mids, charges, mults = [], [], [], []
        for i, at in enumerate(atoms_list):
            Z = torch.tensor(at.get_atomic_numbers(), dtype=torch.int64, device=device)
            nums.append(Z)
            mids.append(torch.full((Z.shape[0],), i, dtype=torch.int64, device=device))
            info = getattr(at, "info", {}) or {}
            charges.append(float(info.get("charge", 0.0)))
            mults.append(float(info.get("mult", 1.0)))

        self.numbers = (torch.cat(nums) if nums
                        else torch.zeros((0,), dtype=torch.int64, device=device))
        self.mol_idx = (torch.cat(mids) if mids
                        else torch.zeros((0,), dtype=torch.int64, device=device))
        self.charge_B = torch.tensor(charges, dtype=dtype, device=device)
        self.mult_B = torch.tensor(mults, dtype=dtype, device=device)

        self.N_atoms = int(self.numbers.numel())
        self.Nmax_atoms = int(max((len(at) for at in atoms_list), default=0))

        if fixed_nmax is None:
            self.nmax_dof = 3 * self.Nmax_atoms
        else:
            self.nmax_dof = int(fixed_nmax)
            req = 3 * self.Nmax_atoms
            if self.nmax_dof < req:
                raise ValueError(f"fixed_nmax={self.nmax_dof} < required {req}")
            if self.nmax_dof % 3 != 0:
                raise ValueError(f"fixed_nmax={self.nmax_dof} not a multiple of 3")

        if self.N_atoms > 0:
            pos = torch.cat([torch.tensor(at.get_positions(), dtype=dtype)
                             for at in atoms_list], dim=0)
        else:
            pos = torch.zeros((0, 3), dtype=dtype)
        self.coord = pos.to(device, non_blocking=True).contiguous()

        self._build_static_maps()
        self._coord_backup = None
        self._prepared = True

    def _build_static_maps(self):
        device, N = self.device, self.N_atoms
        if N == 0:
            self._cart_idx_flat = torch.zeros((0,), dtype=torch.int64, device=device)
            return
        a_local = (torch.arange(N, device=device)
                   - self._ptr[:-1].to(torch.int64)[self.mol_idx])
        base = self.mol_idx.to(torch.int64) * self.nmax_dof + 3 * a_local
        cart_idx = base[:, None] + torch.arange(3, device=device)[None, :]
        self._cart_idx_flat = cart_idx.reshape(-1)

    # ----------------------------------------------------------- coord updates
    @torch.no_grad()
    def step_cart_(self, s_cart: torch.Tensor):
        assert self._prepared, "call prepare() first"
        B = self._atoms_B
        assert s_cart.shape == (B, self.nmax_dof), \
            f"step_cart_ expects (B,{self.nmax_dof}), got {tuple(s_cart.shape)}"
        if self.N_atoms == 0:
            return
        s_cart = s_cart.to(self.device, dtype=self.dtype)
        delta = s_cart.reshape(-1)[self._cart_idx_flat].reshape(self.N_atoms, 3)
        self.coord.add_(delta)

    @torch.no_grad()
    def set_coords_(self, coord: torch.Tensor):
        assert self._prepared, "call prepare() first"
        assert coord.shape == (self.N_atoms, 3)
        self.coord.copy_(coord.to(self.device, dtype=self.dtype))

    @torch.no_grad()
    def backup_coords(self):
        if self._prepared:
            self._coord_backup = self.coord.clone()

    @torch.no_grad()
    def restore_coords(self):
        if self._coord_backup is not None:
            self.coord.copy_(self._coord_backup)
            self._coord_backup = None

    # ---------------------------------------------------------------- forward
    def _build_data(self):
        return {
            "coord": self.coord,
            "numbers": self.numbers,
            "charge": self.charge_B,
            "mult": self.mult_B,
            "mol_idx": self.mol_idx,
        }

    def _reduce_energy(self, out) -> torch.Tensor:
        B = self._atoms_B
        # detach: the high-level API returns ``energy`` still attached to the
        # autograd graph when forces=True; the MAPLE contract expects plain
        # result tensors (NEB/PRFO detach downstream, but be self-contained).
        e = out["energy"].detach().to(self.dtype).reshape(-1)
        if e.numel() == B:
            return e
        if e.numel() == B + 1:        # padded sentinel structure
            return e[:B]
        raise RuntimeError(f"Unexpected energy shape {tuple(e.shape)} (B={B})")

    def _scatter_forces(self, F_atom_eV: torch.Tensor) -> torch.Tensor:
        B, nmax = self._atoms_B, self.nmax_dof
        F_flat = torch.zeros(B * nmax, dtype=self.dtype, device=self.device)
        if self.N_atoms > 0:
            F_flat[self._cart_idx_flat] = F_atom_eV.reshape(-1).to(self.dtype)
        return F_flat.view(B, nmax)

    def get_e_gpu(self):
        assert self._prepared, "call prepare() first"
        B = self._atoms_B
        if B == 0:
            return torch.zeros((0,), dtype=self.dtype, device=self.device)
        out = self.aim(self._build_data(), forces=False)
        return self._reduce_energy(out) / EH2EV

    def get_ef_gpu(self):
        assert self._prepared, "call prepare() first"
        B = self._atoms_B
        if B == 0:
            return (torch.zeros((0,), dtype=self.dtype, device=self.device),
                    torch.zeros((0, 0), dtype=self.dtype, device=self.device))
        out = self.aim(self._build_data(), forces=True)
        E_eV = self._reduce_energy(out)
        F_atom_eV = out["forces"].detach().to(self.dtype).reshape(self.N_atoms, 3)
        F_eV = self._scatter_forces(F_atom_eV)
        return E_eV / EH2EV, F_eV / EH2EV
