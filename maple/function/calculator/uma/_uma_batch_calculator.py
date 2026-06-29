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

Hessian modes (opt-in; OLD numerical FD is the DEFAULT + parity oracle)
----------------------------------------------------------------------
``get_efh_gpu`` dispatches on ``hessian_mode``:
  * ``'numerical'`` (DEFAULT) -- the batched central finite-difference Hessian
    ``H = -(F(x+d) - F(x-d)) / (2d)``; each perturbed replica is an isolated graph
    in a chunked super-batch. This is the OLD path, KEPT verbatim as both the
    fallback AND the byte-parity ORACLE.
  * ``'autograd'`` (OPT-IN) -- an exact AUTOGRAD Hessian from UMA's energy graph
    (double-backward). UMA-omol is a conservative energy model, so the Hessian is
    ``d^2E/dx^2`` obtained by ``autograd.grad`` of the energy graph's force field
    ``F_all = -dE/dx`` (built with ``create_graph=True``) row-looped over the
    block-diagonal unit basis ``eye(3N)``, with an optional ``torch.vmap``
    (``hessian_chunk_size``) acceleration and a try/except loop fallback, and a
    hard fall-back to the numerical oracle on any failure. If the loaded model
    exposes a DIRECT force head (``direct_forces=True``) the same seeded backward
    differentiates that head (matching the FD-of-direct-forces quantity exactly).
    The model is run EAGER (no torch.compile / cuEq; double-backward is
    unsupported there, pytorch#91469); activation checkpointing is auto-disabled
    for this mode (checkpoint reentrancy breaks double-backward).

    ACCURACY CAVEAT (validated 2026-06-27, uma-s-1p1 eSCN-MoE, fp32 native).
    The autograd path is mechanically correct and SELF-CONSISTENT (HVP == H@v and
    grad(grad(E)) == grad(forces) to ~5e-7; forces == predict to ~4e-8), BUT the
    eSCN backbone's custom autograd ops give an INCOMPLETE double-backward: the
    autograd Hessian is correct to FIRST order (forces) yet deviates from the
    finite-difference Hessian by a delta-INDEPENDENT ~1e-2 Ha/A^2 (~1%) that does
    NOT shrink as delta->0 (so it is NOT FD truncation -- it is missing 2nd-order
    curvature in the model graph). Therefore for UMA the NUMERICAL FD Hessian (the
    default) remains the recommended/accurate path; the autograd opt-in is exact
    only on fully twice-differentiable models (e.g. MACE-OFF, validated to ~1e-5
    vs FD with clean delta->0 convergence). A one-time warning is emitted when the
    autograd Hessian is first used.

References for the autograd Hessian / HVP
-----------------------------------------
  * Yuan et al., Nat. Commun. 2024, DOI 10.1038/s41467-024-52481-5 -- autograd
    (double-backward) molecular Hessians, exact and free of the O(delta^2) FD
    truncation error.
  * fairchem UMA ``compute_hessian_vmap`` / ``predict_untrained_hessian``
    (fairchem core uma/outputs.py) -- UMA ships an autograd Hessian via the energy
    graph's double-backward; this calculator wires the same energy-graph
    double-backward locally (the helper is absent in this fairchem build).
  * MACE ``compute_hessians_vmap`` (github ACEsuit/mace, mace/modules/utils.py):
    ``torch.vmap`` of ``autograd.grad`` over ``eye(3N)`` + per-row loop fallback
    (MACE issue #488) -- the vmap+fallback pattern mirrored here.
  * pytorch#91469 -- double-backward unsupported under torch.compile / fused
    kernels -> eager-only requirement.

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

# Inference-settings handle for the OPT-IN activation_checkpointing toggle
# (Task 3) and for the autograd-Hessian mode (checkpoint reentrancy breaks
# double-backward). Optional: absence only disables those opt-ins, never the
# default numerical path.
try:
    from fairchem.core.units.mlip_unit.api.inference import (
        InferenceSettings,
        inference_settings_default,
    )
except Exception:  # pragma: no cover
    InferenceSettings = None
    inference_settings_default = None


EV2HARTREE = 1.0 / 27.211386245988
EH2EV = 27.211386245988


class UMABatchCalc:
    """Batched UMA calculator using fairchem native graph batching.

    One ``prepare()`` fixes the topology of a batch of B molecules; thereafter
    ``get_ef_gpu`` / ``get_efh_gpu`` run a single batched forward (plus, for the
    Hessian, chunked batched finite-difference forwards). Coordinates are kept as
    an f64 master tensor; the forward bridges through f32 (UMA runs in f32).
    """

    supported_hessian_modes = ("numerical", "autograd")

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float64,
        task: str = "omol",
        hessian_delta: float = 2e-3,
        hessian_max_atoms: int = 4096,
        hessian_mode: str = "numerical",
        hessian_chunk_size: Optional[int] = None,
        disable_activation_checkpointing: bool = False,
        fast_inference: bool = False,
        compile_model: bool = False,
        # ---- VRAM-adaptive FD-Hessian chunk sizing (OPT-IN; default OFF = oracle) ----
        auto_chunk: bool = False,
        vram_safety: float = 0.8,
        vram_slope_mib_per_atom: float = 1.5,
        vram_chunk_cap: int = 200000,
    ):
        dev = str(device)
        dev = "cuda" if dev.startswith("cuda") else "cpu"
        self.device = torch.device(dev)
        self.dtype = dtype
        self.task_name = str(task).lower()
        self._delta = float(hessian_delta)
        self._h_max_atoms = int(hessian_max_atoms)

        # ---- Hessian mode (OPT-IN; default 'numerical' = OLD FD path = oracle).
        _m = str(hessian_mode).lower()
        self.hessian_mode = "autograd" if _m in ("autograd", "analytic") else "numerical"
        self.hessian = self.hessian_mode             # back-compat attribute
        self._hessian_chunk_size = (int(hessian_chunk_size)
                                    if hessian_chunk_size is not None else None)
        self._disable_ac = bool(disable_activation_checkpointing)
        self._warned_autograd = False   # one-time eSCN double-backward caveat
        # ---- OPT-IN fast forward (umas_fast_gpu, ACCURACY-PRESERVING subset) ----
        # default OFF -> string "default" -> byte-identical to the base (oracle).
        # fast_inference=True flips on the parity-safe speedups of fairchem's turbo
        # preset but DELIBERATELY KEEPS tf32=False so the fp32 matmul precision is
        # unchanged: merge_mole (exact fusion of the omol mixture-of-experts into
        # dense ops) + activation_checkpointing=False (recompute->store, exact, ~2x).
        # compile_model=True additionally torch.compiles the model graph (numerically
        # ~1e-6, helps STABLE batch shapes; recompiles when B changes, so off by
        # default for the shrinking-batch NEB loop). NONE of these change the model
        # math or the algorithm -- they are kernel/graph optimizations only.
        self._fast_inference = bool(fast_inference)
        self._compile_model = bool(compile_model)

        # ---- VRAM-adaptive FD-Hessian chunk budget (OPT-IN; default OFF) ----
        # When auto_chunk is on, get_efh_gpu sizes the per-forward chunk atom budget
        # (self._h_max_atoms) from the CURRENT free VRAM: a forward of A atoms
        # allocates ~ vram_slope_mib_per_atom * A on top of what is already reserved,
        # so the budget is (vram_safety*total - used) / slope. An OOM halve-and-retry
        # guard makes it crash-proof. Default OFF -> _h_max_atoms stays the fixed
        # hessian_max_atoms (byte-identical oracle). Sizing the chunk changes ONLY how
        # many isolated FD replicas share a forward, never any force value (the FD
        # columns are block-diagonal per replica) -> results are bit-for-bit identical
        # to the fixed-chunk path, regardless of the chosen budget.
        self.auto_chunk = bool(auto_chunk)
        self.vram_safety = float(vram_safety)
        self.vram_slope_mib_per_atom = max(1e-6, float(vram_slope_mib_per_atom))
        self.vram_chunk_cap = int(vram_chunk_cap)
        self._auto_chunk_retries = 0   # diagnostics: total OOM halve-and-retry events

        # ---- inference settings ------------------------------------------
        # Default path keeps the string "default" -> byte-identical to the base.
        # OPT-IN: build an InferenceSettings object only to flip
        # activation_checkpointing off, either because the user asked (Task 3,
        # ~2x speed, parity-safe) OR because the autograd Hessian needs it off
        # (torch.utils.checkpoint reentrancy is incompatible with double-backward).
        _need_ac_off = self._disable_ac or (self.hessian_mode == "autograd")
        if (_need_ac_off or self._fast_inference) and inference_settings_default is not None:
            _isett = inference_settings_default()
            _isett.activation_checkpointing = False
            if self._fast_inference:
                # graph speedups only; tf32 stays False (precision unchanged).
                # NOTE: merge_mole is INCOMPATIBLE with a multi-molecule batch
                # (fairchem escn_md asserts natoms.numel()==1) -> NOT used here.
                # activation_checkpointing=False (set above) is the ~2x parity-safe
                # win; compile=torch.compile (exact ~1e-6) on top, multi-system OK.
                _isett.merge_mole = False
                _isett.tf32 = False
                _isett.compile = bool(self._compile_model)
            self._inference_settings_arg = _isett
        else:
            self._inference_settings_arg = "default"

        self._predictor = load_predict_unit(
            model_path, inference_settings=self._inference_settings_arg, device=dev
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
        # Phase-1b numerical-Hessian plan cache (D3): block-diagonal batch
        # containers + vectorized perturb/scatter index tensors, geometry
        # independent so they are built once per prepare() and reused across
        # every get_efh_gpu() call (e.g. a BatchPRFO RecalcFC loop).
        self._h_plan = None
        self._h_plan_key = None

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
        self._h_plan = None          # invalidate cached Hessian plan (topology changed)
        self._h_plan_key = None
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

    # ------------------------------------------------- partial-Hessian helper
    def _movable_key(self, movable_masks):
        """Hashable cache key for a Hessian plan.

        Includes the FD step ``self._delta`` and chunk budget ``self._h_max_atoms``
        because the cached plan bakes BOTH in (pert_val = s*delta, fac =
        -s/(2*delta), and the chunk boundaries are derived from h_max). An
        adaptive-delta change (or an h_max change) must therefore invalidate a
        stale plan -- otherwise get_efh_gpu would silently reuse a plan built for
        the old step size. The movable spec ('full' or per-structure index
        tuples) is the third component.
        """
        mv = ("full" if movable_masks is None
              else tuple(tuple(int(x) for x in m)
                         for m in self._resolve_movable(movable_masks)))
        return (self._delta, self._h_max_atoms, mv)

    def _resolve_movable(self, movable_masks):
        """Per-structure list[int] of atom indices whose DOFs are perturbed.

        ``movable_masks`` may be None (all atoms of every structure), or a
        sequence of length B where entry b is None (all atoms of structure b),
        a bool mask of length n_b, or an explicit list/array of atom indices.
        Frozen atoms still appear in every replica (they exert forces); only
        their columns/rows are omitted from the Hessian (the exact second-
        derivative block of the FixAtoms-constrained PES).
        """
        import numpy as _np
        B = self._atoms_B
        n_b = self._n_b.tolist()
        if movable_masks is None:
            return [list(range(n_b[b])) for b in range(B)]
        out = []
        for b in range(B):
            m = movable_masks[b]
            if m is None:
                out.append(list(range(n_b[b])))
                continue
            m_arr = _np.asarray(m)
            if m_arr.dtype == bool:
                out.append([int(i) for i in _np.nonzero(m_arr)[0]])
            else:
                idxs = [int(i) for i in m_arr.reshape(-1)]
                assert all(0 <= a < n_b[b] for a in idxs), (
                    f"movable index out of range for structure {b} "
                    f"(n_atoms={n_b[b]}): {idxs}"
                )
                out.append(idxs)
        return out

    # --------------------------------------------------- Hessian plan builder
    def _build_hessian_plan(self, movable_masks=None):
        """Build (once, cached) the block-diagonal central-FD Hessian plan.

        Eliminates the per-replica AtomicData clone + per-call ``...list_to_batch``
        rebuild + per-DOF python scatter of the legacy path. All pieces here are
        geometry INDEPENDENT (topology fixed by prepare()), so the plan is built
        once and reused across every get_efh_gpu() call (BatchPRFO RecalcFC loop):

          * chunk the (mol i, movable atom a, axis c, sign s) replicas under the
            same <= hessian_max_atoms atom budget as the legacy path;
          * for each chunk, collate the per-mol AtomicData *templates* into ONE
            block-diagonal batch container ONCE and park it on-device (reused; the
            untouched topology is never rebuilt again);
          * precompute the vectorized gather (chunk node -> master coord row), the
            vectorized perturbation (one (node,axis) += s*delta per replica), and
            the vectorized scatter (chunk force row -> flat H index) so the whole
            assembly is index ops with NO python loop over DOFs.
        """
        device = self.device
        B = self._atoms_B
        delta = self._delta
        nmax = self.nmax_dof
        nmax2 = nmax * nmax
        n_b = self._n_b.tolist()
        ptr = self._ptr.tolist()
        h_max = self._h_max_atoms

        mov = self._resolve_movable(movable_masks)               # list[list[int]]
        # responding atom indices per structure = perturbed (movable) set; for the
        # full Hessian this is every atom, so all nodes of a replica respond.
        mov_resp = mov

        # ---- enumerate replicas: (i, a, c, s) ; column k = 3*a + c -----------
        rep_i, rep_a, rep_c, rep_s = [], [], [], []
        for i in range(B):
            for a in mov[i]:
                for c in range(3):
                    rep_i.append(i); rep_a.append(a); rep_c.append(c); rep_s.append(1.0)
                    rep_i.append(i); rep_a.append(a); rep_c.append(c); rep_s.append(-1.0)
        R = len(rep_i)

        # ---- chunk replicas under the atom budget (legacy-identical rule) -----
        chunks = []
        cur, cur_atoms = [], 0
        for r in range(R):
            na = n_b[rep_i[r]]
            if cur and cur_atoms + na > h_max:
                chunks.append(cur); cur, cur_atoms = [], 0
            cur.append(r); cur_atoms += na
        if cur:
            chunks.append(cur)

        plan_chunks = []
        max_atoms_in_chunk = 0
        for chunk in chunks:
            Rc = len(chunk)
            # block-diagonal batch container from shared per-mol templates (ONCE)
            mol_refs = [self._ad_list[rep_i[r]] for r in chunk]
            cont = atomicdata_list_to_batch(mol_refs)
            cont = self._to_device(cont)

            cn = [n_b[rep_i[r]] for r in chunk]                  # atoms per replica
            coff = [0]
            for v in cn:
                coff.append(coff[-1] + v)
            Nc = coff[-1]
            max_atoms_in_chunk = max(max_atoms_in_chunk, Nc)

            # gather: chunk node -> master coord row (block order == replica order)
            node_src = []
            for slot, r in enumerate(chunk):
                g0 = ptr[rep_i[r]]
                node_src.extend(range(g0, g0 + cn[slot]))
            node_src = torch.tensor(node_src, dtype=torch.long, device=device)

            # perturbation: one (node,axis) += s*delta per replica (disjoint blocks)
            pert_node = torch.tensor(
                [coff[slot] + rep_a[r] for slot, r in enumerate(chunk)],
                dtype=torch.long, device=device,
            )
            pert_comp = torch.tensor(
                [rep_c[r] for r in chunk], dtype=torch.long, device=device,
            )
            pert_val = torch.tensor(
                [rep_s[r] * delta for r in chunk], dtype=self.dtype, device=device,
            )

            # scatter: responding (movable) atom rows -> flat H index; fac=-s/(2d)
            resp_node, tgt, fac = [], [], []
            for slot, r in enumerate(chunk):
                i = rep_i[r]; k = 3 * rep_a[r] + rep_c[r]
                f = -rep_s[r] / (2.0 * delta)
                base_i = i * nmax2
                for ap in mov_resp[i]:
                    resp_node.append(coff[slot] + ap)
                    fac.append(f)
                    row3 = 3 * ap
                    tgt.append(base_i + (row3 + 0) * nmax + k)
                    tgt.append(base_i + (row3 + 1) * nmax + k)
                    tgt.append(base_i + (row3 + 2) * nmax + k)
            resp_node = torch.tensor(resp_node, dtype=torch.long, device=device)
            tgt = torch.tensor(tgt, dtype=torch.long, device=device)
            fac = torch.tensor(fac, dtype=self.dtype, device=device)

            plan_chunks.append(dict(
                cont=cont, node_src=node_src,
                pert_node=pert_node, pert_comp=pert_comp, pert_val=pert_val,
                resp_node=resp_node, tgt=tgt, fac=fac,
            ))

        return dict(
            chunks=plan_chunks, n_replicas=R, n_chunks=len(chunks),
            max_atoms_in_chunk=max_atoms_in_chunk, nmax=nmax,
        )

    # ------------------------------------------------------------ get_efh_gpu
    def get_efh_gpu(self, movable_masks=None):
        """Energy + forces + per-structure NUMERICAL Hessian (batched central FD).

        Returns
        -------
        (E_Ha: (B,), F_Ha: (B, nmax_dof),
         H_Ha: (B, nmax_dof, nmax_dof), P: (B,) int64)

        H is block-diagonal-padded: each structure fills its own (3*n_i, 3*n_i)
        top-left block. UMA exposes a numerical Hessian only, so every column is a
        central finite difference of forces. Each +/- displacement of each DOF of
        each owning molecule is an *isolated* single-molecule graph; replicas are
        packed into a chunked super-batch (<= hessian_max_atoms atoms per forward).

        Phase-1b (D3) optimizations vs the legacy path, ALL parity-preserving (the
        model force evaluations are byte-identical -- same templates, same chunk
        budget, same central FD):
          (a) block-diagonal container reuse -- the geometry-independent batch
              topology (atomic_numbers/batch/natoms/charge/spin/...) is collated
              ONCE per prepare() and parked on-device, instead of cloning every
              per-replica AtomicData and re-collating every call;
          (b) vectorized perturb + vectorized scatter -- positions are gathered
              and displaced with index ops and the (i,k)-column assembly is ONE
              ``index_add_`` instead of a python loop over DOFs;
          (c) the base-point E/F forward is computed ONCE and returned as the
              gradient (central FD intrinsically uses only the +/- replicas, so
              the base point is not, and cannot be, an FD center without changing
              the discretization and breaking byte parity).

        ``movable_masks`` (D3.2) optionally restricts the perturbed/responding DOFs
        to a per-structure flexible-atom subspace (None = full Hessian); frozen
        atoms still appear in every replica and exert forces (exact constrained-PES
        Hessian block).

        Mode dispatch (OPT-IN): ``hessian_mode='autograd'`` routes to the exact
        energy-graph double-backward Hessian; ANY failure hard-falls to the
        numerical FD body below (the default + parity oracle).
        """
        # OPT-IN autograd Hessian; on any error fall through to the numerical oracle.
        if self.hessian_mode == "autograd":
            try:
                return self._efh_gpu_autograd(movable_masks)
            except Exception:
                pass
        # OPT-IN VRAM-adaptive chunk sizing (default OFF -> fixed-chunk oracle).
        if self.auto_chunk and self.device.type == "cuda":
            return self._get_efh_numerical_auto(movable_masks)
        return self._get_efh_numerical(movable_masks)

    # ---------------------------------------------- VRAM-adaptive chunk helpers
    def _vram_budget_atoms(self) -> int:
        """Max atoms-per-FD-forward to keep total VRAM under vram_safety*total.

        Uses the LIVE free VRAM (torch.cuda.mem_get_info reflects the caching
        allocator's reserved pool + context), so the budget self-regulates as the
        run's reserved high-water mark grows: a forward of A atoms allocates
        ~vram_slope_mib_per_atom * A on top of what is already used, so
        A_max = (vram_safety*total - used) / slope. Floored at the largest single
        structure (one whole replica must fit a chunk) and capped at vram_chunk_cap.
        """
        MiB = 1024.0 ** 2
        free_b, total_b = torch.cuda.mem_get_info()
        total_mib = total_b / MiB
        used_mib = total_mib - free_b / MiB
        headroom_mib = self.vram_safety * total_mib - used_mib
        amax = int(headroom_mib / self.vram_slope_mib_per_atom)
        floor = max(1, int(self.Nmax_atoms))
        return max(floor, min(self.vram_chunk_cap, amax))

    def _get_efh_numerical_auto(self, movable_masks):
        """VRAM-adaptive FD Hessian: size the chunk atom budget to fill VRAM toward
        vram_safety, then on a CUDA OOM halve the budget and retry (never crashes).
        Math is identical to the fixed-chunk oracle -- chunk size changes only how
        many isolated replicas share a forward, not any force value. NOTE: while
        auto_chunk is ON, self._h_max_atoms is the LIVE adaptive budget (it is
        overwritten here each call); a caller that toggles auto_chunk back OFF and
        wants the old fixed budget must reset self._h_max_atoms itself."""
        budget = self._vram_budget_atoms()
        if budget != self._h_max_atoms:
            # only invalidate the cached plan when the budget actually changed, so a
            # stable-VRAM RecalcFC loop still reuses one plan across recalcs.
            self._h_max_atoms = budget
            self._h_plan = None
            self._h_plan_key = None
        attempts = 0
        while True:
            try:
                return self._get_efh_numerical(movable_masks)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or attempts >= 12:
                    raise
                torch.cuda.empty_cache()
                new_budget = max(int(self.Nmax_atoms), self._h_max_atoms // 2)
                if new_budget >= self._h_max_atoms:
                    raise  # cannot shrink further (single structure already too big)
                self._h_max_atoms = new_budget
                self._h_plan = None
                self._h_plan_key = None
                self._auto_chunk_retries += 1
                attempts += 1

    def _get_efh_numerical(self, movable_masks=None):
        """Numerical central-FD Hessian body (the default + byte-parity oracle).

        Chunk atom budget = self._h_max_atoms (fixed hessian_max_atoms by default, or
        set by _get_efh_numerical_auto when auto_chunk is on). See get_efh_gpu's
        docstring for the full contract / optimizations."""
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return (
                torch.zeros(0, dtype=dtype, device=device),
                torch.zeros((0, 0), dtype=dtype, device=device),
                torch.zeros((0, 0, 0), dtype=dtype, device=device),
                torch.zeros(0, dtype=torch.int64, device=device),
            )

        nmax_a = self.Nmax_atoms
        nmax = self.nmax_dof
        nmax2 = nmax * nmax

        # (c) base-point energy + padded forces -- ONE forward, reused as gradient
        E_eV, F_eV = self._forward(self.coord)
        F_pad = torch.zeros((B, nmax), dtype=dtype, device=device)
        if self.N_atoms > 0:
            F_pad.reshape(-1)[self._cols] = F_eV.reshape(-1)
        P = (nmax_a - self._n_b).to(torch.int64)

        if self.N_atoms == 0:
            H_eV = torch.zeros((B, nmax, nmax), dtype=dtype, device=device)
            return (E_eV * EV2HARTREE, F_pad * EV2HARTREE, H_eV * EV2HARTREE, P)

        # (a) build-or-reuse the cached block-diagonal Hessian plan
        key = self._movable_key(movable_masks)
        if self._h_plan is None or self._h_plan_key != key:
            self._h_plan = self._build_hessian_plan(movable_masks)
            self._h_plan_key = key
        plan = self._h_plan

        coord = self.coord  # (N,3) f64 master
        H_flat = torch.zeros(B * nmax2, dtype=dtype, device=device)

        for ch in plan["chunks"]:
            # vectorized gather of the perturbation positions for the whole chunk
            pos = coord[ch["node_src"]].clone()                  # (Nc,3) f64
            pos[ch["pert_node"], ch["pert_comp"]] += ch["pert_val"]
            # reuse the cached container; overwrite ONLY positions (clone keeps the
            # cached template pristine against any in-place use inside predict)
            cont = ch["cont"]
            cont = cont.clone() if hasattr(cont, "clone") else cont
            cont.pos = pos.to(device=device, dtype=torch.float32)
            _, F, _ = self._predict_forces(cont)                 # (Nc,3) f64
            # (b) ONE vectorized scatter: H[i,row,k] += -s * F[row] / (2 delta)
            contrib = (F[ch["resp_node"]] * ch["fac"][:, None]).reshape(-1)
            H_flat.index_add_(0, ch["tgt"], contrib)

        H_eV = H_flat.reshape(B, nmax, nmax)
        H_eV = 0.5 * (H_eV + H_eV.transpose(1, 2))               # symmetrize

        return (
            E_eV * EV2HARTREE,
            F_pad * EV2HARTREE,
            H_eV * EV2HARTREE,
            P,
        )

    # ------------------------------------------------ legacy reference (parity)
    def _get_efh_gpu_legacy(self):
        """Verbatim pre-D3 implementation, kept ONLY as the byte-parity oracle.

        Identical math to get_efh_gpu(); used by the parity gate to prove the
        Phase-1b vectorization changes nothing numerically.
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

        E_eV, F_eV = self._forward(self.coord)
        F_pad = torch.zeros((B, nmax), dtype=dtype, device=device)
        if self.N_atoms > 0:
            F_pad.reshape(-1)[self._cols] = F_eV.reshape(-1)

        H_eV = torch.zeros((B, nmax, nmax), dtype=dtype, device=device)
        P = (nmax_a - self._n_b).to(torch.int64)

        replicas = []
        meta = []
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
                    ad.pos = pos_i.detach().to(device="cpu", dtype=torch.float32)
                    replicas.append(ad)
                    meta.append((i, k, s))

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

        for (i, k), fp in f_plus.items():
            fm = f_minus[(i, k)]
            dof_i = 3 * n_b_list[i]
            col = (-(fp - fm) / (2.0 * delta)).reshape(-1)
            H_eV[i, :dof_i, k] = col

        H_eV = 0.5 * (H_eV + H_eV.transpose(1, 2))

        return (
            E_eV * EV2HARTREE,
            F_pad * EV2HARTREE,
            H_eV * EV2HARTREE,
            P,
        )

    # ===================================================================== #
    # OPT-IN autograd (energy-graph double-backward) Hessian + batched HVP.  #
    # All of the below is reachable ONLY via hessian_mode='autograd' / hvp() #
    # and never touches the default numerical path above.                    #
    # Refs: Yuan et al. Nat. Commun. 2024 DOI 10.1038/s41467-024-52481-5;    #
    # fairchem compute_hessian_vmap (uma/outputs.py); MACE compute_hessians_ #
    # vmap (ACEsuit/mace, issue #488); eager-only per pytorch#91469.         #
    # ===================================================================== #
    def _warn_autograd_once(self):
        """One-time RuntimeWarning about the eSCN double-backward accuracy caveat."""
        if not self._warned_autograd:
            import warnings as _warnings
            _warnings.warn(
                "UMABatchCalc autograd mode: the autograd Hessian/HVP is self-"
                "consistent but the eSCN-MoE backbone's custom ops give an INCOMPLETE "
                "double-backward (~1e-2 Ha/A^2, delta-independent, vs numerical FD). "
                "Use hessian_mode='numerical' (default) for an accurate UMA Hessian. "
                "See module docstring ACCURACY CAVEAT.",
                RuntimeWarning, stacklevel=2)
            self._warned_autograd = True

    def _collate_proc(self, data, preds):
        """Collate task-keyed model outputs -> {'energy':(B,), 'forces':(N,3)}.

        Mirrors fairchem's ``collate_predictions`` (per-system / per-atom gather)
        but keeps the autograd graph (we run the model under ``enable_grad``,
        bypassing ``predict``'s ``no_grad`` + ``.clone()`` which would detach the
        position leaf). ``preds`` already had ``normalizer.denorm`` + element-ref
        undo applied by ``_process_outputs``; both are affine in positions so the
        second derivative is exact up to the (positive) denorm scale they carry.
        """
        from collections import defaultdict
        pu = self._predictor
        collated = defaultdict(list)
        for i, dataset in enumerate(data.dataset):
            for task in pu.dataset_to_tasks[dataset]:
                if task.level == "system":
                    collated[task.property].append(preds[task.name][i].unsqueeze(0))
                elif task.level == "atom":
                    collated[task.property].append(preds[task.name][data.batch == i])
        return {prop: torch.cat(val) for prop, val in collated.items()}

    def _forward_fall_graph(self, coord_leaf):
        """ONE grad-enabled forward at ``coord_leaf`` (N,3, requires_grad, f32).

        Returns ``(E_eV (B,), F_all (N,3))`` where ``F_all`` is the energy graph's
        force field, carrying a graph w.r.t. ``coord_leaf`` for a further backward
        (Hessian / HVP):
          * conservative UMA-omol (``direct_forces=False``): the model's energy
            HEAD computes ``forces = -grad(E, pos, create_graph=self.training)``
            INTERNALLY and, in eval, frees the energy graph -- so we flip ONLY the
            head modules to ``training=True`` (``create_graph=True``) so
            ``out['forces']`` (= ``-dE/dx``) is itself twice-differentiable
            (fairchem's ``predict_untrained_hessian`` trick). The backbone + MoE
            router stay in EVAL: MoE expert routing depends on ``self.training``
            (escn_moe.py), so flipping the whole model would change the energy;
            head-only flip keeps the energy byte-identical to the numerical predict
            and makes ``d^2E/dx^2`` the exact second derivative of THAT energy;
          * direct-force head: ``out['forces']`` is a first-order function of the
            positions; differentiating it once reproduces the FD-of-direct-forces
            Hessian exactly.
        Either way ``F_all = out['forces']`` (the SAME force field the numerical
        path differentiates). Model is run EAGER (no compile/cuEq) under
        ``torch.enable_grad()``; train mode only flips ``self.training`` (UMA-eSCN
        has no dropout/batchnorm), so energy/force VALUES are unchanged.
        """
        pu = self._predictor
        # Ensure fairchem's lazy init ran (prepare_for_inference: eval mode, MOLE
        # merge, graph-gen, activation_checkpointing flag) before the direct model
        # call -- the autograd path may be the very first forward on this predictor.
        if not getattr(pu, "lazy_model_intialized", False):
            with torch.no_grad():
                ad0 = self._to_device(self._clone_batch())
                ad0.pos = self.coord.to(device=self.device, dtype=torch.float32)
                pu.predict(ad0)
        ad = self._to_device(self._clone_batch())
        ad.pos = coord_leaf.to(device=self.device, dtype=torch.float32)
        model = pu.model
        # Flip ONLY the head modules to training=True (so their internal
        # autograd.grad uses create_graph=True); keep backbone + MoE in eval so the
        # energy is identical to the numerical predict (MoE routing is training-
        # dependent: escn_moe.py). Restore afterwards.
        flipped = [m for m in model.modules()
                   if ("Head" in type(m).__name__) and (m.training is False)]
        try:
            for m in flipped:
                m.training = True
            with torch.enable_grad():
                output = model(ad)
                proc = pu._process_outputs(ad, output, True)
                out = self._collate_proc(ad, proc)
        finally:
            for m in flipped:
                m.training = False
        E_eV = out["energy"].reshape(-1)
        if ("forces" in out) and out["forces"].requires_grad:
            F_all = out["forces"].reshape(self.N_atoms, 3)
        else:
            # no differentiable force field (energy-only model): differentiate the
            # energy externally (works when the model does not free its own graph).
            g = torch.autograd.grad(E_eV.sum(), coord_leaf, create_graph=True)[0]
            F_all = -g
        return E_eV, F_all

    def _efh_gpu_autograd(self, movable_masks=None):
        """Energy + forces + per-structure AUTOGRAD Hessian (double-backward).

        Reverse-mode: seed a RESPONSE DOF and the backward fills that Hessian ROW.
        ``F_all`` is the energy graph's force field (``-dE/dx``, create_graph=True)
        for the conservative UMA-omol, so ``H[i, r, k] = -dF_r/dx_k = d^2E/dx^2``,
        exactly the quantity the numerical FD path differentiates -> same return
        contract / padded block-diagonal layout / sign / units. Seeds are
        block-diagonal so ONE backward per local DOF index fills the row for every
        molecule at once (the molecule batching). ``movable_masks`` restricts
        perturbed/responding DOFs identically to the numerical path. Optional
        ``torch.vmap`` over the seed basis when ``hessian_chunk_size`` is set (MACE
        ``compute_hessians_vmap`` pattern) with a per-row loop fallback.
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

        self._warn_autograd_once()
        nmax_a, nmax = self.Nmax_atoms, self.nmax_dof
        N = self.N_atoms
        ptr = self._ptr.tolist()
        mov = self._resolve_movable(movable_masks)        # per-mol movable atom idx

        # ONE grad-enabled forward; F_all carries the graph for the 2nd backward.
        coord_leaf = self.coord.detach().to(torch.float32).requires_grad_(True)
        E_eV, F_all = self._forward_fall_graph(coord_leaf)
        E_eV = E_eV.to(dtype)

        # padded forces -- F_all is the SAME force field the numerical Hessian
        # differentiates (conservative F_all=-dE/dx=F; direct F_all=F_head=F), so
        # this scatter is byte-equivalent to get_ef_gpu's force output.
        F_pad = torch.zeros((B, nmax), dtype=dtype, device=device)
        if N > 0:
            F_pad.reshape(-1)[self._cols] = F_all.reshape(-1).to(dtype)

        H = torch.zeros((B, nmax, nmax), dtype=dtype, device=device)
        P = (nmax_a - self._n_b).to(torch.int64)

        # local response-DOF index ld = 3*slot + c over the movable-atom slots;
        # owners[ld] = list of (mol i, real local atom idx ap) that own slot.
        n_movable_max = max((len(m) for m in mov), default=0)
        K = 3 * n_movable_max
        seeds, owners = [], {}
        for ld in range(K):
            slot, c = ld // 3, ld % 3
            go = torch.zeros((N, 3), dtype=F_all.dtype, device=device)
            own = []
            for i in range(B):
                if slot < len(mov[i]):
                    ap = mov[i][slot]
                    go[ptr[i] + ap, c] = 1.0
                    own.append((i, ap))
            seeds.append(go)
            owners[ld] = own

        def _scatter(ld, col):
            slot, c = ld // 3, ld % 3
            for (i, ap) in owners[ld]:
                r = 3 * ap + c                            # response row DOF index
                # fill row r across the molecule's movable column atoms:
                for aq in mov[i]:
                    H[i, r, 3 * aq:3 * aq + 3] = -col[ptr[i] + aq, :].to(dtype)

        if self._hessian_chunk_size is not None and K > 0:
            GO = torch.stack(seeds, dim=0)                # (K,N,3)

            def _get_vjp(v):
                return torch.autograd.grad(F_all, coord_leaf, grad_outputs=v,
                                           retain_graph=True, create_graph=False)[0]

            COLS = torch.vmap(_get_vjp, in_dims=0, out_dims=0,
                              chunk_size=int(self._hessian_chunk_size))(GO)
            for ld in range(K):
                _scatter(ld, COLS[ld])
        else:
            for ld in range(K):
                col = torch.autograd.grad(F_all, coord_leaf, grad_outputs=seeds[ld],
                                          retain_graph=True, create_graph=False)[0]
                _scatter(ld, col)

        H = 0.5 * (H + H.transpose(1, 2))                 # symmetrize
        return (E_eV * EV2HARTREE, F_pad * EV2HARTREE, H * EV2HARTREE, P)

    def hvp(self, v: torch.Tensor) -> torch.Tensor:
        """Batched Hessian-vector product ``H @ v`` via energy double-backward.

        Block-diagonal per system; ONE forward + TWO backward, no dense Hessian:
        ``g = dE/dx (create_graph=True); Hn = -grad((F_all . v).sum(), x)`` which
        equals ``grad((g . v).sum(), x) = d^2E/dx^2 . v`` for the conservative
        model (and ``-d(F_head)/dx . v`` for a direct head -- the matching
        FD-of-forces operator). ``v`` is ``(B, nmax_dof)`` padded (forces layout);
        returns ``Hn`` ``(B, nmax_dof)`` [Ha/A^2]. Consumed by the BPRFO iterative
        eigensolver (sibling agent). Ref: Yuan et al. Nat. Commun. 2024,
        DOI 10.1038/s41467-024-52481-5.
        """
        assert self._prepared, "call prepare() first"
        self._warn_autograd_once()
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return torch.zeros((0, 0), dtype=dtype, device=device)
        assert v.shape == (B, self.nmax_dof), \
            f"hvp expects v (B,{self.nmax_dof}), got {tuple(v.shape)}"
        coord_leaf = self.coord.detach().to(torch.float32).requires_grad_(True)
        _E, F_all = self._forward_fall_graph(coord_leaf)
        # gather padded v -> per-atom (N,3) using the same flat-column map as forces
        if self.N_atoms > 0:
            v_atom = v.to(device, F_all.dtype).reshape(-1)[self._cols].reshape(self.N_atoms, 3)
        else:
            v_atom = torch.zeros((0, 3), dtype=F_all.dtype, device=device)
        Hn_atom = -torch.autograd.grad((F_all * v_atom).sum(), coord_leaf,
                                       retain_graph=False)[0]
        Hn_pad = torch.zeros((B, self.nmax_dof), dtype=dtype, device=device)
        if self.N_atoms > 0:
            Hn_pad.reshape(-1)[self._cols] = Hn_atom.reshape(-1).to(dtype)
        return Hn_pad * EV2HARTREE
