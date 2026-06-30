# -*- coding: utf-8 -*-
"""
Runnable MACE-OFF batched calculator built on the installed ``mace`` package's
NATIVE multi-graph batching (mace.data.AtomicData + torch_geometric.Batch).

Why this exists (vs ``_mace_batch_calculator.py`` / ``MACEBatchCalc``):
``MACEBatchCalc`` ingests a *traced* 6-positional-arg general model
(``model/maceomol.pt``). No such traced ``.pt`` is shipped in this tree; the only
MACE-OFF asset present is the mace-package checkpoint
``~/.cache/mace/MACE-OFF23_medium.model`` (a ``ScaleShiftMACE`` whose ``forward``
takes a data dict). This calculator loads that checkpoint directly and batches
B independent (isolated, non-periodic) systems with ONE forward per call, so the
batched-MD core (``ensemble/nvt_batched.py``) has a real MACE-OFF batch backend.

Isolation: standard MACE-OFF is a PURE LOCAL MLIP (no global charge
equilibration). A block-diagonal multi-graph batch therefore has ZERO
cross-system coupling -- the perturb-one byte-isolation probe returns 0.0 Ha
(verified: rep1 ΔE = 0.0 Ha when rep0 atom is moved). Hence ``batch_isolated =
True`` and B>1 co-batching is exact.

Contract (matches the batched-forward primitive the MD core drives):
    prepare(atoms_list, fixed_nmax=None)
    step_cart_(s_cart: (B, nmax_dof))   in-place Cartesian displacement [Angstrom]
    set_coords_(coord: (N_atoms, 3))
    backup_coords() / restore_coords()
    get_ef_gpu() -> (E_Ha (B,), F_Ha (B, nmax_dof))   Hartree, Hartree/Angstrom
    isolation_check(perturb) -> float   max cross-system ENERGY leak [Ha]
Padded layout: replica ``b``, atom ``a``, axis ``c`` -> flat column ``3*a + c``;
columns ``3*n_b .. nmax_dof`` are zero padding. Units: mace returns eV / eV·Å⁻¹;
this calc returns Hartree / Hartree·Å⁻¹ (EV2HARTREE = 1/27.211386245988).

ponytail: gas-phase / isolated systems only (the mace AtomicData here is built
without a cell). Periodic batching is out of scope for this backend.
"""

import os

import numpy as np
import torch

EV2HARTREE = 1.0 / 27.211386245988
_DEFAULT_MODEL = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")


class MaceOffBatchCalc:
    """Batched MACE-OFF calculator (mace-package native multi-graph batching)."""

    # Pure local MLIP: a block-diagonal batch is exactly isolated (no charge eq.).
    batch_isolated = True

    def __init__(self,
                 model_path: str = _DEFAULT_MODEL,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float64,
                 allow_tf32: bool = False):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu"
                                   else "cpu")
        self.dtype = dtype
        # GPU-opt Lever 1: TF32 tensor-core matmul is an EXPLICIT opt-in (only ever
        # active for an fp32 forward; fp64 matmuls ignore the flag). Default OFF so
        # the fp64 default path is byte-for-byte unchanged.
        self.allow_tf32 = bool(allow_tf32)
        self._model_path = model_path

        model = torch.load(model_path, map_location=self.device, weights_only=False)
        # Explicit device move AFTER load (mirrors mace.calculators.MACECalculator):
        # map_location alone does NOT relocate the e3nn TorchScript tensor-product
        # constants (baked Wigner-3j buffers), so on GPU they stay on CPU and the
        # forward fails with a cuda/cpu device mismatch. .to(device) recurses into
        # the scripted submodules and moves them.
        self.model = model.to(self.device).to(self.dtype).eval()
        self._maybe_set_tf32()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.r_max = float(self.model.r_max)
        self.atomic_numbers = [int(z) for z in self.model.atomic_numbers]
        self.heads = list(getattr(self.model, "heads", ["Default"]))

        # mace data helpers (imported here so package import stays light).
        from mace.data.utils import config_from_atoms
        from mace.data import AtomicData
        from mace.tools import AtomicNumberTable, torch_geometric
        self._config_from_atoms = config_from_atoms
        self._AtomicData = AtomicData
        self._tg = torch_geometric
        self._z_table = AtomicNumberTable(self.atomic_numbers)

        self._prepared = False
        self._coord_backup = None

    # ----------------------------------------------------------------- precision
    def _maybe_set_tf32(self):
        """Enable TF32 tensor-core matmul -- ONLY for an opted-in fp32 CUDA forward.
        TF32 truncates fp32 mantissas in the matmul accumulate, so it must be an
        explicit choice (energy drift fp32->TF32 ~1.2e-3 Ha, acceptable for NVT per
        B-33). It is a no-op for fp64 (the flag does not affect double-precision
        matmuls), so leaving it set never perturbs the fp64 default path."""
        if self.allow_tf32 and self.dtype == torch.float32 and self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def set_precision(self, dtype: torch.dtype, allow_tf32: bool = False):
        """Switch the forward precision in place (Lever 1 wiring from BatchedNVT).

        Recasts the loaded model to ``dtype`` and (for fp32+CUDA) optionally turns on
        TF32. Safe to call before OR after prepare(): if already prepared, the master
        coord buffers are recast too. fp32 halves the model+activation VRAM (~2x max
        batch) and runs ~1.37x (fp32) / ~1.83x (fp32+TF32) on the compute-bound case;
        energy drift fp64->fp32 ~7e-4 Ha (B-33). fp64 stays the default (NVE needs it)."""
        self.dtype = dtype
        self.allow_tf32 = bool(allow_tf32)
        self.model = self.model.to(dtype).eval()
        self._maybe_set_tf32()
        if self._prepared:
            self.coord = self.coord.to(dtype)
            if self._coord_backup is not None:
                self._coord_backup = self._coord_backup.to(dtype)
            # R2 opt (B-81): the cached static batch fields are dtype-typed (built at
            # the prior dtype) -> recast the floating tensors so the per-step forward
            # stays type-consistent with the recast model. Integer fields (batch/ptr/
            # head/edge bookkeeping) and None entries are left untouched.
            if getattr(self, "_static", None):
                for k, v in self._static.items():
                    if torch.is_tensor(v) and v.is_floating_point():
                        self._static[k] = v.to(dtype)
        return self

    # ------------------------------------------------------------------ prepare
    def prepare(self, atoms_list, fixed_nmax: int = None):
        self._atoms = [a.copy() for a in atoms_list]      # topology (Z) holders
        self.B = len(atoms_list)
        self.n_b = np.array([len(a) for a in atoms_list], dtype=int)        # (B,)
        self.N_atoms = int(self.n_b.sum())
        self.Nmax_atoms = int(self.n_b.max()) if self.B else 0
        self.nmax_dof = (3 * self.Nmax_atoms) if fixed_nmax is None else int(fixed_nmax)

        # master coords (N_atoms, 3) in Angstrom on device.
        pos = np.concatenate([a.get_positions() for a in atoms_list], axis=0)
        self.coord = torch.tensor(pos, dtype=self.dtype, device=self.device)

        # per-replica atom-offset bookkeeping.
        self._ptr = np.concatenate([[0], np.cumsum(self.n_b)]).astype(int)   # (B+1,)
        # flat index into the (B*nmax_dof) padded buffer for each REAL atom's x-col.
        base = []
        for i, n in enumerate(self.n_b):
            for a in range(int(n)):
                base.append(i * self.nmax_dof + 3 * a)
        self._base = torch.tensor(base, dtype=torch.long, device=self.device)
        self._coord_backup = None
        self._prepared = True

        # R2 opt (B-81): build the STATIC batch dict ONCE (node_attrs / head / batch /
        # ptr / cell / weight fields are geometry-INVARIANT) and the block-diagonal
        # intra-replica candidate-pair index ONCE, both resident on the GPU. The
        # per-step forward then only (a) rebuilds edge_index on-GPU from the candidate
        # pairs (cheap radius graph; positions change, the build does not leave the
        # GPU) and (b) reuses the cached statics -- no D2H coord copy, no per-replica
        # ASE/matscipy neighbour list, no Batch.from_data_list re-collate/re-transfer,
        # no per-step global set_default_dtype toggle. Mirrors MACEBatchCalc.prepare().
        self._build_static_cache(atoms_list)

    def _build_static_cache(self, atoms_list):
        """One-time: capture the geometry-invariant batch dict + GPU candidate pairs.

        Uses the SAME mace AtomicData/Batch path the original per-step forward used,
        so node_attrs/head/batch/ptr/cell/weight tensors are BYTE-identical to what
        the validated path produced -- only positions/edge_index/shifts are refreshed
        per step. The one-time AtomicData build keeps the set_default_dtype context
        (correct & cheap once); the per-step hot path has NO dtype toggle."""
        device = self.device
        # block-diagonal intra-replica candidate pairs (global atom indices).
        ci_l, cj_l = [], []
        for b in range(self.B):
            n = int(self.n_b[b]); off = int(self._ptr[b])
            if n >= 2:
                iu, ju = torch.triu_indices(n, n, offset=1, device=device)
                ci_l.append(iu + off); cj_l.append(ju + off)
        self._cand_i = (torch.cat(ci_l) if ci_l
                        else torch.zeros((0,), dtype=torch.long, device=device))
        self._cand_j = (torch.cat(cj_l) if cj_l
                        else torch.zeros((0,), dtype=torch.long, device=device))

        if self.B == 0:
            self._static = None
            return
        # build the batch ONCE (prepare-time geometry) to harvest the static fields.
        prev = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        try:
            datas = []
            for i in range(self.B):
                cfg = self._config_from_atoms(self._atoms[i])
                datas.append(self._AtomicData.from_config(
                    cfg, z_table=self._z_table, cutoff=self.r_max, heads=self.heads))
            batch = self._tg.Batch.from_data_list(datas).to(self.device)
        finally:
            torch.set_default_dtype(prev)
        d = batch.to_dict()
        # drop the geometry-dependent keys (refreshed every step); keep the rest.
        self._dyn_keys = ("positions", "edge_index", "shifts", "unit_shifts")
        self._static = {k: v for k, v in d.items() if k not in self._dyn_keys}
        # cache dtype-typed templates for the per-step dynamic fields.
        self._unit_shifts0 = torch.zeros((0, 3), dtype=self.dtype, device=device)

    # --------------------------------------------------------------- coord ops
    @torch.no_grad()
    def step_cart_(self, s_cart: torch.Tensor):
        assert self._prepared, "call prepare() first"
        assert tuple(s_cart.shape) == (self.B, self.nmax_dof), \
            f"step_cart_ expects (B,{self.nmax_dof}), got {tuple(s_cart.shape)}"
        if self.N_atoms == 0:
            return
        s = s_cart.reshape(-1).to(self.device, self.dtype)
        base = self._base
        disp = torch.stack([s[base], s[base + 1], s[base + 2]], dim=1)
        self.coord.add_(disp)

    @torch.no_grad()
    def set_coords_(self, coord: torch.Tensor):
        assert self._prepared, "call prepare() first"
        assert tuple(coord.shape) == (self.N_atoms, 3)
        self.coord.copy_(coord.to(self.device, self.dtype))

    @torch.no_grad()
    def backup_coords(self):
        if self._prepared:
            self._coord_backup = self.coord.clone()

    @torch.no_grad()
    def restore_coords(self):
        if self._coord_backup is not None:
            self.coord.copy_(self._coord_backup)

    # ------------------------------------------------------------------ forward
    def _build_edges_gpu(self, coord):
        """Block-diagonal radius graph from the cached intra-replica candidate pairs,
        entirely on the GPU (no host neighbour list). Returns (edge_index, shifts,
        unit_shifts). Non-periodic isolated replicas -> shifts == unit_shifts == 0.

        Cutoff convention matches mace's neighbour list (matscipy ``neighbour_list``,
        strict ``distance < cutoff``); edge ORDER is irrelevant because the model
        aggregates messages with order-invariant scatter, so only the edge SET (which
        this reproduces exactly for separated molecular geometries) drives E/F."""
        device = self.device
        if self._cand_i.numel() == 0:
            z = torch.zeros((2, 0), dtype=torch.long, device=device)
            s = torch.zeros((0, 3), dtype=self.dtype, device=device)
            return z, s, s
        rij = coord[self._cand_i] - coord[self._cand_j]
        d2 = (rij * rij).sum(dim=-1)
        keep = d2 < (self.r_max * self.r_max)            # matscipy strict-<
        ci = self._cand_i[keep]; cj = self._cand_j[keep]
        src = torch.cat([ci, cj], dim=0); dst = torch.cat([cj, ci], dim=0)
        edge_index = torch.stack([src, dst], dim=0)
        shifts = torch.zeros((edge_index.size(1), 3), dtype=self.dtype, device=device)
        return edge_index, shifts, shifts

    def _batched_ef_eV(self):
        """ONE native multi-graph forward -> (E_eV (B,), F_eV (N_atoms,3)).

        R2 opt (B-81): coords stay GPU-resident; edges are rebuilt on-GPU; the static
        batch fields are reused from prepare(); no per-step set_default_dtype toggle."""
        coord = self.coord.detach().to(self.device, self.dtype).requires_grad_(True)
        edge_index, shifts, unit_shifts = self._build_edges_gpu(coord)
        d = dict(self._static)                       # shallow copy; static tensors reused
        d["positions"] = coord
        d["edge_index"] = edge_index
        d["shifts"] = shifts
        d["unit_shifts"] = unit_shifts
        out = self.model(d, compute_force=True, training=False)
        E_eV = out["energy"].detach().reshape(-1).to(self.dtype)          # (B,)
        F_eV = out["forces"].detach().to(self.dtype)                      # (N,3)
        return E_eV, F_eV

    def get_ef_gpu(self):
        if self.B == 0:
            z = torch.zeros((0,), dtype=self.dtype, device=self.device)
            return z, torch.zeros((0, 0), dtype=self.dtype, device=self.device)
        E_eV, F_eV = self._batched_ef_eV()
        F_flat = torch.zeros(self.B * self.nmax_dof, dtype=self.dtype, device=self.device)
        if self.N_atoms > 0:
            base = self._base
            F_flat[base] = F_eV[:, 0]
            F_flat[base + 1] = F_eV[:, 1]
            F_flat[base + 2] = F_eV[:, 2]
        F_Ha = F_flat.view(self.B, self.nmax_dof) * EV2HARTREE
        E_Ha = E_eV * EV2HARTREE
        return E_Ha, F_Ha

    # --------------------------------------------------------------- isolation
    def isolation_check(self, perturb: float = 0.05) -> float:
        """Perturb replica-0 atom 0 and return the max ENERGY leak into the OTHER
        replicas [Ha]. Exact (0.0) for a pure-local block-diagonal MACE batch."""
        assert self._prepared and self.B >= 2
        E0, _ = self.get_ef_gpu()
        self.backup_coords()
        self.coord[0, 0] += perturb
        E1, _ = self.get_ef_gpu()
        self.restore_coords()
        return float((E1[1:] - E0[1:]).abs().max().item())
