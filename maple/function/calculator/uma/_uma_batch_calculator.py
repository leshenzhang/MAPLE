# -*- coding: utf-8 -*-
"""Batched UMA (fairchem) calculator for the MAPLE GPU-batch comp-chem library.

Goal
----
Batch many small molecules (<= ~50 atoms) through UMA in ONE fairchem forward to
maximize throughput and GPU utilization. Parallelism comes entirely from
fairchem's *native* graph batching -- there is NO hand-written CUDA.

Native batching path
--------------------
    AtomicData.from_ase(atoms, task_name="omol", ...)        # one per molecule
    atomicdata_list_to_batch(ad_list) -> batched AtomicData  # one combined graph
    predict_unit.predict(batch) -> {"energy": (B,), "forces": (N_total, 3), ...}

The batched graph carries a per-atom ``batch`` index, so each molecule's edges
stay within its own node block (block-diagonal); perturbing one molecule cannot
leak into another (verified by the perturb-one byte-isolation parity gate).

Contract
--------
Mirrors ``AIMNet2BatchCalc``:
    prepare(atoms_list, fixed_nmax=None)
    step_cart_(s_cart: (B, nmax_dof))
    set_coords_(coord: (N, 3)) / backup_coords() / restore_coords()
    get_ef_gpu()  -> (E_Ha: (B,), F_Ha: (B, nmax_dof))
    get_efh_gpu() -> (E_Ha: (B,), F_Ha: (B, nmax_dof),
                      H_Ha: (B, nmax_dof, nmax_dof), P: (B,) int64)

Hartree units (UMA returns eV / eV.A^-1; EV2HARTREE = 1/27.211386245988).
UMA supports a NUMERICAL Hessian only -> get_efh_gpu() uses a batched central
finite-difference Hessian (H = -(F(x+d) - F(x-d)) / (2d)); each perturbed replica
is an isolated graph in a chunked super-batch.

Self-contained fairchem env-compat preamble
-------------------------------------------
This module makes ``import fairchem.core`` work in environments where it is
otherwise broken, WITHOUT modifying the environment or any other file:
  (1) ``torch.serialization.add_safe_globals([slice])`` -- torch>=2.6 defaults
      ``weights_only=True``; e3nn's ``constants.pt`` and the UMA checkpoint store
      a ``slice`` object and fail to load otherwise.
  (2) A ``ray.serve`` stub installed into ``sys.modules`` before importing
      fairchem -- ``fairchem.core.__init__`` pulls in a ray.serve batch-serve
      shim whose fastapi dependency needs pydantic v2; on pydantic-v1 envs the
      import raises even though direct inference never touches ray.serve.
"""

# --------------------------------------------------------------------------- #
# Self-contained fairchem env-compat preamble (must run before importing       #
# fairchem). Both steps are idempotent and only take effect when needed.       #
# --------------------------------------------------------------------------- #
import sys as _sys
import types as _types

import torch

# (1) Allowlist `slice` for torch.load weights_only=True (e3nn / UMA checkpoint).
try:
    torch.serialization.add_safe_globals([slice])
except Exception:
    pass


# (2) Stub ray.serve so fairchem.core import does not pull a pydantic-v2 fastapi
#     chain. We never use ray.serve for direct inference.
def _install_ray_serve_stub() -> None:
    try:
        import ray  # noqa: F401
    except Exception:
        return  # ray not present -> fairchem import path differs; nothing to do
    existing = _sys.modules.get("ray.serve")
    if existing is not None and getattr(existing, "_maple_stub", False):
        return
    try:
        import ray.serve  # noqa: F401  -- imports cleanly -> leave it alone
        return
    except Exception:
        pass
    serve = _types.ModuleType("ray.serve")
    serve._maple_stub = True
    serve.deployment = lambda *a, **k: (lambda cls: cls)
    serve.batch = lambda *a, **k: (lambda fn: fn)
    serve.handle = None
    serve.run = lambda *a, **k: None
    serve.start = lambda *a, **k: None
    schema = _types.ModuleType("ray.serve.schema")
    schema.LoggingConfig = lambda *a, **k: None
    serve.schema = schema
    import ray
    ray.serve = serve
    _sys.modules["ray.serve"] = serve
    _sys.modules["ray.serve.schema"] = schema


_install_ray_serve_stub()
# --------------------------------------------------------------------------- #

from functools import partial
from typing import List, Optional

from ase import Atoms

try:
    from fairchem.core.calculate.ase_calculator import AtomicData
    from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
    from fairchem.core.units.mlip_unit import load_predict_unit
except ImportError as exc:  # pragma: no cover
    raise ImportError(f"fairchem-core is not importable: {exc}")


EV2HARTREE = 1.0 / 27.211386245988
EH2EV = 27.211386245988


class UMABatchCalc:
    """Batched UMA calculator using fairchem native graph batching.

    One ``prepare()`` fixes the topology of a batch of B molecules; thereafter
    ``get_ef_gpu`` / ``get_efh_gpu`` run a single batched forward (plus, for the
    Hessian, chunked batched finite-difference forwards). Coordinates are kept as
    an f64 master tensor; the forward bridges through f32 (UMA runs in f32).
    """

    supported_hessian_modes = ("numerical",)

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float64,
        task: str = "omol",
        hessian_delta: float = 2e-3,
        hessian_max_atoms: int = 4096,
    ):
        dev = str(device)
        dev = "cuda" if dev.startswith("cuda") else "cpu"
        self.device = torch.device(dev)
        self.dtype = dtype
        self.task_name = str(task).lower()
        self.hessian = "numerical"
        self._delta = float(hessian_delta)
        self._h_max_atoms = int(hessian_max_atoms)

        self._predictor = load_predict_unit(
            model_path, inference_settings="default", device=dev
        )
        ext = bool(getattr(self._predictor.inference_settings, "external_graph_gen", False))
        self._r_edges = ext
        self._max_neigh = 300 if ext else None
        self._a2g = partial(
            AtomicData.from_ase,
            task_name=self.task_name,
            r_edges=self._r_edges,
            r_data_keys=["spin", "charge"],
            max_neigh=self._max_neigh,
            radius=6.0,
        )

        # prepare() state
        self._prepared = False
        self._atoms_B = 0
        self._ptr = None
        self.numbers = None
        self.mol_idx = None
        self._local_atom = None
        self.coord = None
        self.N_atoms = 0
        self.Nmax_atoms = 0
        self.nmax_dof = 0
        self._n_b = None
        self._cols = None
        self._ad_list = None
        self._batch_ad = None
        self._coord_backup = None

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _ad_atoms(at: Atoms) -> Atoms:
        """Copy with UMA-convention info: spin = multiplicity, charge = charge."""
        at = at.copy()
        at.info["spin"] = int(at.info.get("mult", at.info.get("spin", 1)))
        at.info["charge"] = int(at.info.get("charge", 0))
        return at

    # ---------------------------------------------------------------- prepare
    def prepare(self, atoms_list: List[Atoms], fixed_nmax: Optional[int] = None):
        """Fix topology + initial coords for a batch of molecules.

        ``fixed_nmax`` (optional) overrides the padded per-structure DOF size so a
        batch-PRFO driver can share one padded layout across iterations (matches
        AIMNet2BatchCalc semantics).
        """
        device, dtype = self.device, self.dtype
        B = len(atoms_list)
        self._atoms_B = B

        ptr = [0]
        nums, mids, locs = [], [], []
        for i, at in enumerate(atoms_list):
            Z = torch.tensor(at.get_atomic_numbers(), dtype=torch.int64, device=device)
            n = int(Z.shape[0])
            ptr.append(ptr[-1] + n)
            nums.append(Z)
            mids.append(torch.full((n,), i, dtype=torch.int64, device=device))
            locs.append(torch.arange(n, dtype=torch.int64, device=device))

        self._ptr = torch.tensor(ptr, dtype=torch.long, device=device)
        self.numbers = torch.cat(nums) if nums else torch.zeros(0, dtype=torch.int64, device=device)
        self.mol_idx = torch.cat(mids) if mids else torch.zeros(0, dtype=torch.int64, device=device)
        self._local_atom = torch.cat(locs) if locs else torch.zeros(0, dtype=torch.int64, device=device)

        self.N_atoms = int(self.numbers.numel())
        self.Nmax_atoms = int(max((len(at) for at in atoms_list), default=0))
        self.nmax_dof = 3 * self.Nmax_atoms if fixed_nmax is None else int(fixed_nmax)

        if self.N_atoms > 0:
            pos = torch.cat(
                [torch.tensor(at.get_positions(), dtype=dtype) for at in atoms_list], dim=0
            )
        else:
            pos = torch.zeros((0, 3), dtype=dtype)
        self.coord = pos.to(device).contiguous()

        self._n_b = (self._ptr[1:] - self._ptr[:-1])  # (B,) atoms per structure

        # Vectorized scatter columns: global atom g (mol b, local a) maps to the
        # flat index  b*nmax_dof + 3*a + {0,1,2}  inside a (B, nmax_dof) buffer.
        if self.N_atoms > 0:
            base = self.mol_idx * self.nmax_dof + 3 * self._local_atom  # (N,)
            self._cols = (
                base[:, None] + torch.arange(3, device=device)[None, :]
            ).reshape(-1)  # (3N,)
        else:
            self._cols = torch.zeros(0, dtype=torch.int64, device=device)

        # Per-molecule AtomicData templates + reusable batched template.
        self._ad_list = [self._a2g(self._ad_atoms(at)) for at in atoms_list]
        self._batch_ad = atomicdata_list_to_batch(self._ad_list) if B > 0 else None

        self._coord_backup = None
        self._prepared = True

    # ------------------------------------------------------------ coord ops
    @torch.no_grad()
    def step_cart_(self, s_cart: torch.Tensor):
        """In-place displacement. ``s_cart`` is (B, nmax_dof), padded per structure."""
        assert self._prepared, "call prepare() first"
        B = self._atoms_B
        assert s_cart.shape == (B, self.nmax_dof), (
            f"step_cart_ expects (B,{self.nmax_dof}), got {tuple(s_cart.shape)}"
        )
        if self.N_atoms == 0:
            return
        s = s_cart.to(self.device, dtype=self.dtype).reshape(-1)
        disp = s[self._cols].reshape(self.N_atoms, 3)  # vectorized gather, no .item()
        self.coord.add_(disp)

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

    # -------------------------------------------------------------- forward
    def _clone_batch(self):
        ad = self._batch_ad
        if hasattr(ad, "clone"):
            return ad.clone()
        return atomicdata_list_to_batch(self._ad_list)

    def _to_device(self, ad):
        if hasattr(ad, "to"):
            try:
                return ad.to(self.device)
            except Exception:
                return ad
        return ad

    def _predict_forces(self, batch_ad):
        """Run predict; return (energy (Bc,), forces (Nc, 3), batch_idx (Nc,)).

        UMA returns forces directly (computed inside ``predict``), so the outputs
        carry an autograd graph; this calculator is inference-only -> detach.
        """
        batch_ad = self._to_device(batch_ad)
        out = self._predictor.predict(batch_ad)
        E = out["energy"].detach().to(self.dtype).reshape(-1)
        F = out["forces"].detach().to(self.dtype).to(self.device)
        bidx = batch_ad.batch.to(self.device)
        return E, F, bidx

    def _forward(self, coord: torch.Tensor):
        """coord (N,3) f64 -> (E_eV (B,), F_eV (N,3)), single batched forward."""
        ad = self._to_device(self._clone_batch())
        ad.pos = coord.to(device=self.device, dtype=torch.float32)
        E, F, _ = self._predict_forces(ad)
        return E, F.reshape(self.N_atoms, 3)

    # ------------------------------------------------------------- get_ef_gpu
    def get_ef_gpu(self):
        """(E_Ha: (B,), F_Ha: (B, nmax_dof)) from one batched forward."""
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return (
                torch.zeros(0, dtype=dtype, device=device),
                torch.zeros((0, 0), dtype=dtype, device=device),
            )
        E_eV, F_eV = self._forward(self.coord)
        F_pad = torch.zeros((B, self.nmax_dof), dtype=dtype, device=device)
        if self.N_atoms > 0:
            F_pad.reshape(-1)[self._cols] = F_eV.reshape(-1)  # vectorized scatter
        return E_eV * EV2HARTREE, F_pad * EV2HARTREE

    # ------------------------------------------------------------ get_efh_gpu
    def get_efh_gpu(self):
        """Energy + forces + per-structure NUMERICAL Hessian (batched central FD).

        Returns
        -------
        (E_Ha: (B,), F_Ha: (B, nmax_dof),
         H_Ha: (B, nmax_dof, nmax_dof), P: (B,) int64)

        H is block-diagonal-padded: each structure fills its own (3*n_i, 3*n_i)
        top-left block. UMA exposes a numerical Hessian only, so every column is a
        central finite difference of forces. Each +/- displacement of each DOF of
        each owning molecule is an *isolated* graph; replicas are packed into a
        chunked super-batch (<= hessian_max_atoms atoms per forward).
        """
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return (
                torch.zeros(0, dtype=dtype, device=device),
                torch.zeros((0, 0), dtype=dtype, device=device),
                torch.zeros((0, 0, 0), dtype=dtype, device=device),
                torch.zeros(0, dtype=torch.int64, device=device),
            )

        delta = self._delta
        nmax_a = self.Nmax_atoms
        nmax = self.nmax_dof  # = 3 * nmax_a
        n_b_list = self._n_b.tolist()
        ptr_list = self._ptr.tolist()

        # base energy + padded forces
        E_eV, F_eV = self._forward(self.coord)
        F_pad = torch.zeros((B, nmax), dtype=dtype, device=device)
        if self.N_atoms > 0:
            F_pad.reshape(-1)[self._cols] = F_eV.reshape(-1)

        H_eV = torch.zeros((B, nmax, nmax), dtype=dtype, device=device)
        P = (nmax_a - self._n_b).to(torch.int64)

        # Enumerate perturbation replicas: (mol i, dof k, sign s).
        # delta is tiny -> graph topology is unchanged, so clone the per-mol
        # AtomicData template and only overwrite its positions.
        replicas = []  # list[AtomicData]
        meta = []      # list[(i, k, s)]
        for k in range(3 * nmax_a):
            a, c = k // 3, k % 3
            for i in range(B):
                if n_b_list[i] <= a:
                    continue
                g0, g1 = ptr_list[i], ptr_list[i + 1]
                for s in (1.0, -1.0):
                    pos_i = self.coord[g0:g1].clone()
                    pos_i[a, c] += s * delta
                    tmpl = self._ad_list[i]
                    ad = tmpl.clone() if hasattr(tmpl, "clone") else self._a2g(self._ad_atoms(
                        Atoms(numbers=self.numbers[g0:g1].tolist(),
                              positions=pos_i.detach().cpu().numpy())
                    ))
                    # keep replica on the template's device (CPU); the whole
                    # chunk is moved to self.device in one shot before predict.
                    ad.pos = pos_i.detach().to(device="cpu", dtype=torch.float32)
                    replicas.append(ad)
                    meta.append((i, k, s))

        # Forces store: (i, k) -> (force_plus, force_minus), each (n_i, 3) eV/A.
        f_plus, f_minus = {}, {}

        def _flush(ads, metas):
            if not ads:
                return
            bb = atomicdata_list_to_batch(ads)
            _, F, bidx = self._predict_forces(bb)
            for r, (i, k, s) in enumerate(metas):
                fr = F[bidx == r]
                (f_plus if s > 0 else f_minus)[(i, k)] = fr

        chunk_ads, chunk_meta, cur_atoms = [], [], 0
        for ad, (i, k, s) in zip(replicas, meta):
            na = n_b_list[i]
            if cur_atoms + na > self._h_max_atoms and chunk_ads:
                _flush(chunk_ads, chunk_meta)
                chunk_ads, chunk_meta, cur_atoms = [], [], 0
            chunk_ads.append(ad)
            chunk_meta.append((i, k, s))
            cur_atoms += na
        _flush(chunk_ads, chunk_meta)

        # Assemble per-structure Hessian columns: H[:, k] = -(F+ - F-)/(2 delta).
        for (i, k), fp in f_plus.items():
            fm = f_minus[(i, k)]
            dof_i = 3 * n_b_list[i]
            col = (-(fp - fm) / (2.0 * delta)).reshape(-1)  # (dof_i,)
            H_eV[i, :dof_i, k] = col

        H_eV = 0.5 * (H_eV + H_eV.transpose(1, 2))  # symmetrize per structure

        return (
            E_eV * EV2HARTREE,
            F_pad * EV2HARTREE,
            H_eV * EV2HARTREE,
            P,
        )
