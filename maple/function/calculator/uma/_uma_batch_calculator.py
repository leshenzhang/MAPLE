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


from ..batch_calculator_base import (  # noqa: E402
    BatchCalcABC,
    EV2HARTREE,
    register_batch_calculator,
)


@register_batch_calculator
class UMABatchCalc(BatchCalcABC):
    """Batched UMA calculator using fairchem native graph batching.

    One ``prepare()`` fixes the topology of a batch of B molecules; thereafter
    ``get_ef_gpu`` / ``get_efh_gpu`` run a single batched forward (plus, for the
    Hessian, chunked batched finite-difference forwards). Coordinates are kept as
    an f64 master tensor; the forward bridges through f32 (UMA runs in f32).

    PBC (Phase-1A): UMA is natively periodic (``AtomicData.from_ase`` reads
    ``atoms.cell``/``atoms.pbc``; the fairchem periodic tasks omat/oc20/oc22/oc25/
    odac build minimum-image edges). When the prepared replicas carry a cell this
    calc (i) requires a PERIODIC task (rejects the molecular ``omol`` task on
    periodic input, mirroring the single-system gate), (ii) PRESERVES cell+pbc on
    the per-replica AtomicData, and (iii) REBUILDS the batched AtomicData from the
    current coordinates EACH forward (``_forward``) so the periodic edges are never
    stale -- this sidesteps the precomputed-edge (``external_graph_gen``/r_edges)
    staleness hazard entirely rather than trusting internal regen. The MD loop and
    ``get_ef_gpu`` return contract are UNCHANGED (cell fixed -> NVT only).
    """

    # ---- capability protocol (BatchCalcABC declarative attrs) --------------
    MODEL_NAMES = ("uma",)
    MODEL_ENERGY_UNIT = "eV"                 # UMA returns eV / eV.A^-1; base -> Hartree once
    MODEL_DTYPE = torch.float32              # UMA runs in f32 (master coord stays f64)
    SUPPORTS_PBC = True                      # UMA is the only PBC-capable backend
    SUPPORTED_HESSIAN_MODES = ("numerical", "autograd")  # numerical = default + parity oracle
    HAS_HVP = True                           # autograd HVP + FD get_hvp (single-struct Dimer path)
    SUPPORTS_COUPLING = False                # block-diagonal batch; molecules are isolated
    BATCHABLE = True

    # ---- batched-MD stack attrs (Phase-B; read outside the BatchCalcABC contract) --
    # fairchem's per-atom ``batch`` index makes the co-batched graph block-diagonal:
    # replica energies/forces are independent (no global charge equilibration), so
    # BatchedNVT/REMD/... accept B>1 (read by nvt_batched._assert_batch_isolated).
    batch_isolated = True
    # Graph edge cutoff (matches the ``radius=6.0`` passed to AtomicData.from_ase in
    # _a2g). box_guard.get_calculator_r_max reads this to enforce side >= 2*r_max for
    # the periodic minimum-image guard; absent it, the guard silently no-ops.
    r_max = 6.0
    # fairchem periodic task names (single-system gate parity, _uma_calculator.py):
    # periodic replicas require one of these (checked in _build_topology).
    _PERIODIC_TASKS = ("omat", "oc20", "oc22", "oc25", "odac")

    # Back-compat lowercase alias (pre-retrofit callers / sibling backends read this).
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
        fd_mode: str = "central",
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
        # base.__init__ resolves device (torch.device) + dtype and inits the
        # generic prepared-state; derive the 'cuda'|'cpu' string it settled on for
        # load_predict_unit / the inference-settings path below.
        super().__init__(device, dtype)
        dev = self.device.type
        self.task_name = str(task).lower()
        self._delta = float(hessian_delta)
        self._h_max_atoms = int(hessian_max_atoms)

        # ---- Hessian mode (OPT-IN; default 'numerical' = OLD FD path = oracle).
        _m = str(hessian_mode).lower()
        self.hessian_mode = "autograd" if _m in ("autograd", "analytic") else "numerical"
        self.hessian = self.hessian_mode             # back-compat attribute

        # ---- FD discretization for the NUMERICAL Hessian (OPT-IN; default
        # 'central' = OLD 6m-forward path = byte-identical oracle). 'forward' =
        # forward-difference H[:,j] = -(F(x0+delta e_j) - F(x0))/delta : ~2x fewer
        # forwards (3m+1 vs 6m) by sharing ONE unperturbed reference F(x0) across
        # all columns, at the cost of O(delta) truncation error (vs central
        # O(delta^2)). Useful fast mode for TS-validation / RRHO screening where
        # ~1 cm^-1 is fine; NOT for high-accuracy IR. Baked into the cached plan
        # key so a mode flip invalidates a stale plan.
        self._fd_mode = "forward" if str(fd_mode).lower() == "forward" else "central"
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

        # ---- model-specific prepared-state ONLY. base.__init__ already inits the
        #      generic layout state (_prepared / _atoms_B / _ptr / numbers /
        #      mol_idx / _local_atom / _n_b / _cols / coord / N_atoms /
        #      Nmax_atoms / nmax_dof / _coord_backup); do NOT re-init those here.
        self._fwd_count = 0   # diagnostics: total batched model forwards (_forward calls)
        self._ad_list = None
        self._batch_ad = None
        # PBC (Phase-1A): cell-carrying source atoms cached so the periodic forward can
        # REBUILD the batch with fresh minimum-image edges each step. base.__init__
        # already owns _periodic (False default) / _coord_backup.
        self._src_atoms = None
        # Phase-1b numerical-Hessian plan cache (D3): block-diagonal batch
        # containers + vectorized perturb/scatter index tensors, geometry
        # independent so they are built once per prepare() and reused across
        # every get_efh_gpu() call (e.g. a BatchPRFO RecalcFC loop).
        self._h_plan = None
        self._h_plan_key = None

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _ad_atoms(at: Atoms) -> Atoms:
        """Copy with UMA-convention info: spin = multiplicity, charge = charge.

        PBC (Phase-1A): ``at.copy()`` already carries ``cell``/``pbc``; they are kept
        verbatim so ``AtomicData.from_ase`` builds a PERIODIC neighbourhood for a
        periodic replica (the previous code only copied ``info`` but ASE ``copy()``
        preserves cell+pbc, so no extra work is needed -- this is asserted at
        prepare() and the cell is the calc-internal periodic state)."""
        at = at.copy()
        at.info["spin"] = int(at.info.get("mult", at.info.get("spin", 1)))
        at.info["charge"] = int(at.info.get("charge", 0))
        return at

    # -------------------------------------------------------- topology hook
    def _build_topology(self, atoms_list):
        """Cache UMA's per-molecule AtomicData templates + the reusable batched
        template, and reset the numerical-Hessian plan cache.

        Called at the END of ``BatchCalcABC.prepare()`` -- which already built
        ptr/numbers/mol_idx/_local_atom/_n_b/_cols/coord/nmax_dof (with the shared
        ``fixed_nmax`` validation) and reset ``_coord_backup``. The batched template
        is geometry-INDEPENDENT topology, so it is collated + moved to device ONCE
        here (like the FD-Hessian ``cont`` cache) instead of cloning-on-CPU + H2D on
        every forward; per-forward we then only D2D-clone and overwrite ``pos``.
        """
        B = self._atoms_B

        # PBC (Phase-1A): the base already set ``self._periodic`` and enforced the
        # homogeneous-pbc gate against SUPPORTS_PBC. UMA adds the periodic-TASK gate
        # (reject the molecular ``omol`` task on periodic input, mirroring the
        # single-system gate) since that constraint is model-specific, not generic.
        if self._periodic:
            if self.task_name not in self._PERIODIC_TASKS:
                raise NotImplementedError(
                    f"UMABatchCalc got periodic replicas but task='{self.task_name}' is "
                    f"molecular. Periodic UMA needs a periodic task "
                    f"{self._PERIODIC_TASKS} (e.g. 'omat'). Rebuild the calc with "
                    f"task='omat' for condensed-phase PBC.")
            # ANTI-STALE-NEIGHBOR (R3-1): the periodic forward REBUILDS the batched
            # AtomicData from current coords every step (_rebuild_periodic_batch), so
            # periodic edges+shifts are ALWAYS fresh -- this holds even when
            # external_graph_gen/r_edges=True (the precomputed edges are recomputed by
            # from_ase each step) and is immune to the pos-overwrite-only staleness bug.
            # Caveat: with torch.compile/fast_inference the per-step edge COUNT varies
            # as atoms drift, so the compiled graph recompiles (correctness preserved,
            # throughput may drop) -- warn once.
            if (self._fast_inference or self._compile_model) and not getattr(
                    self, "_warned_pbc_compile", False):
                import warnings
                warnings.warn(
                    "UMABatchCalc periodic path rebuilds the neighbour graph each "
                    "forward (fresh edges/shifts); torch.compile/fast_inference will "
                    "recompile on the varying edge count. Correct but slower -- consider "
                    "compile_model=False for periodic NVT.")
                self._warned_pbc_compile = True

        # R3-1 GUARD (stale neighbor list, opt campaign 2026-07-28): when the
        # predictor resolved ``external_graph_gen=True`` (checkpoint default with
        # inference_settings=None/InferenceSettings(external_graph_gen=True/None)),
        # ``_a2g`` bakes the edge list INTO the AtomicData at build time
        # (``r_edges=True``). The MOLECULAR fast path below then D2D-clones that
        # prepare-time template and overwrites ONLY ``pos`` every forward, so the
        # baked edges go STALE as the optimizer/MD moves atoms -> silently wrong
        # E/F (the R3-1 trap). Both fairchem presets ('default'/'turbo') set
        # external_graph_gen=False, so this raise is unreachable on the supported
        # configs; it exists to turn a silent-wrong-number config into a loud
        # error. The PERIODIC path is exempt here (it rebuilds the AtomicData from
        # current coords each forward) but its Hessian paths are NOT -> guarded in
        # get_efh_gpu/hvp.
        if self._r_edges and not self._periodic:
            raise NotImplementedError(
                "UMABatchCalc: this predictor resolved external_graph_gen=True, i.e. "
                "edges are precomputed on the AtomicData at prepare() time. The batched "
                "molecular fast path reuses that template and overwrites positions only, "
                "so the precomputed edges would go STALE as geometries move (silently "
                "wrong E/F -- R3-1). Rebuild the predictor with "
                "InferenceSettings(external_graph_gen=False) (both fairchem presets "
                "'default' and 'turbo' already do).")

        # Per-molecule AtomicData templates + reusable device-resident batched template.
        # PBC: keep the per-replica UMA-convention ASE atoms (cell+pbc preserved by
        # ASE ``copy()``) so the periodic forward can REBUILD the batch with fresh edges
        # each step; the molecular path only D2D-clones this template + overwrites pos.
        self._src_atoms = [self._ad_atoms(at) for at in atoms_list]
        self._ad_list = [self._a2g(a) for a in self._src_atoms]
        self._batch_ad = (
            self._to_device(atomicdata_list_to_batch(self._ad_list)) if B > 0 else None
        )
        self._h_plan = None          # invalidate cached Hessian plan (topology changed)
        self._h_plan_key = None

    # step_cart_ / set_coords_ / backup_coords / restore_coords are identical to
    # BatchCalcABC's -> inherited (deleted here).

    # -------------------------------------------------------------- forward
    def _clone_batch(self):
        ad = self._batch_ad
        if hasattr(ad, "clone"):
            return ad.clone()                       # _batch_ad is device-resident -> D2D clone
        return self._to_device(atomicdata_list_to_batch(self._ad_list))

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

    def _rebuild_periodic_batch(self, coord: torch.Tensor):
        """PBC: rebuild the batched AtomicData from the CURRENT coordinates so the
        periodic neighbour list is regenerated fresh every forward.

        Overwriting only ``ad.pos`` on a cloned template (the isolated fast path)
        would reuse a STALE periodic edge list once atoms drift across the box face
        (the design's correctness hazard #5). Rebuilding via ``from_ase`` on the
        cell-carrying source atoms emits the correct minimum-image edges each step
        regardless of whether the model uses external or internal graph generation.
        Cell/pbc are fixed (NVT) and taken from the prepare-time source atoms.
        """
        cpu = coord.detach().to("cpu", torch.float64).numpy()
        ad_list = []
        for i, at in enumerate(self._src_atoms):
            a = at.copy()
            a.set_positions(cpu[int(self._ptr[i]):int(self._ptr[i + 1])])
            ad_list.append(self._a2g(a))
        return atomicdata_list_to_batch(ad_list)

    def _forward(self, coord: torch.Tensor, need_graph: bool = False):
        """ONE batched forward -> BatchCalcABC contract (E_eV (B,), F_eV (N,3),
        leaf|None) in NATIVE eV (the base converts once via MODEL_ENERGY_UNIT; do
        NOT scale by EV2HARTREE here).

        need_graph=False (default): inference-only predict path -- byte-identical to
        the pre-retrofit ``_forward`` (single batched forward, detached). UMA's own
        overrides (get_ef_gpu / _get_efh_numerical / _get_efh_gpu_legacy) call this
        variant and pack the eV->Hartree conversion themselves, so they are
        unchanged. PBC (Phase-1A): when the batch is periodic the inference path
        REBUILDS the batch from current coords (``_rebuild_periodic_batch``) so the
        minimum-image edges stay fresh (anti-stale-neighbor); the molecular path
        D2D-clones the device-resident template + overwrites pos. need_graph=True:
        return the grad-enabled energy-graph force field + the requires_grad position
        leaf (delegates to _forward_fall_graph, the SAME construction the autograd
        Hessian/HVP path uses), satisfying the base's generic _efh_analytic path.
        """
        if need_graph:
            leaf = coord.detach().to(torch.float32).requires_grad_(True)
            E_eV, F_all = self._forward_fall_graph(leaf)
            return E_eV, F_all, leaf
        # _clone_batch() is already on-device (topology moved in prepare()); the
        # _to_device inside _predict_forces is the only one left (no-op here, kept
        # for the CPU-built batches passed at the partial/full-Hessian call sites).
        self._fwd_count += 1
        if getattr(self, "_periodic", False):
            ad = self._to_device(self._rebuild_periodic_batch(coord))
        else:
            ad = self._clone_batch()
            ad.pos = coord.to(device=self.device, dtype=torch.float32)
        E, F, _ = self._predict_forces(ad)
        return E, F.reshape(self.N_atoms, 3), None

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
        E_eV, F_eV, _ = self._forward(self.coord)
        F_pad = torch.zeros((B, self.nmax_dof), dtype=dtype, device=device)
        if self.N_atoms > 0:
            F_pad.reshape(-1)[self._cols] = F_eV.reshape(-1)  # vectorized scatter
        return E_eV * EV2HARTREE, F_pad * EV2HARTREE

    # --------------------------------------------------- block-diagonal isolation
    def isolation_check(self, perturb: float = 0.05) -> float:
        """Perturb replica-0 atom-0 and return the max ENERGY leak into the OTHER
        replicas [Ha] (mirrors MaceOffBatchCalc.isolation_check).

        fairchem's ``atomicdata_list_to_batch`` tags every atom with a per-replica
        ``batch`` index, so the co-batched graph is BLOCK-DIAGONAL: a displacement
        inside replica 0 can only re-wire replica 0's own node block (its periodic
        edges are rebuilt from replica 0's coords alone) and cannot reach another
        replica. The residual leak floor is UMA's fp32 GPU-forward nondeterminism
        (~1e-6, D-59), NOT exactly 0.0 as for the fp64-deterministic MACE-OFF path.
        """
        assert self._prepared and self._atoms_B >= 2
        E0, _ = self.get_ef_gpu()
        self.backup_coords()
        self.coord[0, 0] += float(perturb)
        E1, _ = self.get_ef_gpu()
        self.restore_coords()
        return float((E1[1:] - E0[1:]).abs().max().item())

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
        return (self._delta, self._h_max_atoms, self._fd_mode, mv)

    # _resolve_movable is identical to BatchCalcABC's -> inherited (deleted here).

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
        # central: +delta AND -delta per DOF (2 replicas/DOF -> 6m).
        # forward (OPT-IN): only +delta per DOF (1 replica/DOF -> 3m) and reuse the
        # ONE base-point forward F(x0) as the shared subtrahend (added in
        # _get_efh_numerical), so total forwards = 3m+1 vs central 6m+1 (~2x).
        forward = (self._fd_mode == "forward")
        rep_i, rep_a, rep_c, rep_s = [], [], [], []
        for i in range(B):
            for a in mov[i]:
                for c in range(3):
                    rep_i.append(i); rep_a.append(a); rep_c.append(c); rep_s.append(1.0)
                    if not forward:
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
            # central: H[:,k] += -s*F(x0+s*delta)/(2 delta) (two replicas/DOF).
            # forward: H[:,k] += -F(x0+delta)/delta (one replica/DOF); the +F(x0)/delta
            #          base term is added once per structure in _get_efh_numerical.
            resp_node, tgt, fac = [], [], []
            for slot, r in enumerate(chunk):
                i = rep_i[r]; k = 3 * rep_a[r] + rep_c[r]
                f = (-rep_s[r] / delta) if forward else (-rep_s[r] / (2.0 * delta))
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
            forward=forward, mov=mov,
        )

    # ------------------------------------------------------------ get_efh_gpu
    def get_efh_gpu(self, movable_masks=None, mode=None, delta=None, chunk_size=None,
                    base_ef=None):
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

        ``base_ef`` (OPT-IN, BatchRFO FIX #1): when the caller has ALREADY evaluated
        (E, F) at the current ``self.coord`` (e.g. the committed-geometry forward of the
        previous optimizer step, geometry unchanged), pass ``base_ef=(E_Ha (B,),
        F_Ha (B, nmax_dof))`` (Hartree, padded) to SKIP the base-point forward (line
        ~779) and reuse it as the returned gradient. The central-FD Hessian is computed
        FRESH from ``self.coord`` regardless (it never uses the base force), so this is
        parity-exact -- it only removes one redundant model forward. Caller is
        responsible for guaranteeing ``self.coord`` is unchanged since ``base_ef`` was
        measured. Honored only on the numerical FD path (the default + oracle); the
        autograd path ignores it and recomputes.

        ``mode`` / ``delta`` / ``chunk_size`` complete the documented BatchCalcABC
        signature (this override used to DROP them -> get_efh_gpu(mode=...) raised
        TypeError). All three are None-sentinel: when None the constructor's knob is
        used and behavior is BIT-IDENTICAL to before.
          * mode=None -> self.hessian_mode ('numerical' default, with the documented
            autograd->numerical hard-fallback). mode='numerical'/'fd' forces FD;
            mode='autograd'/'analytic' forces the double-backward Hessian and RAISES
            instead of silently downgrading (an explicit request is not a hint).
          * delta -> the central-FD step (else self._delta from hessian_delta). The
            Hessian-plan cache key covers _delta (c3af78d), so changing it here
            correctly rebuilds the plan instead of reusing a stale one.
          * chunk_size -> the FD replica-packing budget in ATOMS (else
            self._h_max_atoms from hessian_max_atoms). UMA's Hessian is FD-based, so
            this is its analogue of the base's vmap seed budget; chunking only changes
            how many isolated replicas share a forward, never a force value
            (parity-preserving by construction).
        """
        # R3-1 GUARD (see _build_topology): every Hessian path below (FD plan
        # containers, legacy chunks, autograd _clone_batch) reuses prepare-time
        # AtomicData templates and overwrites ``pos`` only. With
        # external_graph_gen=True the baked edges would be STALE at the perturbed/
        # current geometry -> silently wrong H. Only reachable for a PERIODIC batch
        # (the molecular case already raised at prepare()).
        if self._r_edges:
            raise NotImplementedError(
                "UMABatchCalc.get_efh_gpu: external_graph_gen=True bakes edges into "
                "the prepare-time AtomicData templates; the batched Hessian paths "
                "overwrite positions only, so those edges would be STALE (silently "
                "wrong Hessian -- R3-1). Rebuild the predictor with "
                "InferenceSettings(external_graph_gen=False).")

        # Per-call overrides of the ctor knobs. Both are components of the
        # Hessian-plan cache key, so a change correctly invalidates the cached plan.
        if delta is not None:
            self._delta = float(delta)
        if chunk_size is not None:
            self._h_max_atoms = int(chunk_size)

        req = None if mode is None else str(mode).lower()
        if req in ("autograd", "analytic"):
            return self._efh_gpu_autograd(movable_masks)   # explicit -> no silent fallback
        if req is not None and req not in ("numerical", "fd"):
            raise ValueError(
                f"{type(self).__name__}.get_efh_gpu: unknown mode {mode!r}; expected one "
                f"of {self.SUPPORTED_HESSIAN_MODES} (or None to use hessian_mode).")
        # OPT-IN autograd Hessian; on any error fall through to the numerical oracle.
        if req is None and self.hessian_mode == "autograd":
            try:
                return self._efh_gpu_autograd(movable_masks)
            except Exception:
                pass
        # OPT-IN VRAM-adaptive chunk sizing (default OFF -> fixed-chunk oracle).
        if self.auto_chunk and self.device.type == "cuda":
            return self._get_efh_numerical_auto(movable_masks, base_ef=base_ef)
        return self._get_efh_numerical(movable_masks, base_ef=base_ef)

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

    def _get_efh_numerical_auto(self, movable_masks, base_ef=None):
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
                return self._get_efh_numerical(movable_masks, base_ef=base_ef)
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

    def _get_efh_numerical(self, movable_masks=None, base_ef=None):
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

        # (c) base-point energy + padded forces -- ONE forward, reused as gradient.
        # FIX #1 (base_ef): if the caller already evaluated (E,F) at THIS exact
        # self.coord, reuse it and SKIP this forward. base_ef = (E_Ha (B,), F_Ha
        # (B,nmax)) in Hartree/padded layout; we divide back to eV so the final
        # *EV2HARTREE below round-trips to the caller's values (fp round-trip ~1 ULP,
        # far under the ~1e-6 noise floor). The FD Hessian below is unaffected -- it
        # perturbs self.coord and never reads this base force -> parity-exact.
        if base_ef is not None:
            E_base_Ha, F_base_Ha = base_ef
            E_eV = (E_base_Ha / EV2HARTREE).to(dtype=dtype, device=device)
            F_pad = (F_base_Ha / EV2HARTREE).to(dtype=dtype, device=device)
        else:
            E_eV, F_eV, _ = self._forward(self.coord)
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

        # forward-difference base term: the chunk loop deposited only the perturbed
        # -F(x0+delta e_k)/delta columns; complete H[row,k] = -(F(x0+delta e_k)-F(x0))
        # /delta by adding the shared +F(x0)/delta (the ONE base forward, already in
        # F_pad eV) to every movable (row,k) cell. Column-independent -> broadcast the
        # base force vector across all movable columns. (central path leaves H as-is.)
        if plan.get("forward"):
            inv_d = 1.0 / self._delta
            mov = plan["mov"]
            for b in range(B):
                if not mov[b]:
                    continue
                idx = torch.tensor(
                    [3 * a + c for a in mov[b] for c in range(3)],
                    dtype=torch.long, device=device,
                )
                base = F_pad[b, idx] * inv_d                       # (d,) eV/A / A
                H_eV[b].index_put_(
                    (idx[:, None], idx[None, :]),
                    H_eV[b][idx[:, None], idx[None, :]] + base[:, None],
                )

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

        E_eV, F_eV, _ = self._forward(self.coord)
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
        # R3-1 GUARD: _forward_fall_graph clones the prepare-time template and
        # overwrites pos only -> stale baked edges under external_graph_gen=True.
        if self._r_edges:
            raise NotImplementedError(
                "UMABatchCalc.hvp: external_graph_gen=True bakes edges into the "
                "prepare-time AtomicData template (stale at the current geometry -- "
                "R3-1). Rebuild the predictor with "
                "InferenceSettings(external_graph_gen=False).")
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

    @torch.no_grad()
    def get_hvp(self, atoms, n, delta: float = 0.005):
        """Single-structure finite-difference Hessian-vector product ``H @ n``.

        OPT-IN, additive: provides the ``calc.get_hvp(atoms, n) -> (Hn, forces,
        energy)`` contract that the single-structure :class:`Dimer` (dimer.py)
        consumes, so the serial Dimer can run *natively* on a UMA potential and
        serve as the convergence-equivalence parity oracle for :class:`BatchDimer`.
        UMA's eSCN-MoE autograd double-backward is known-incomplete (see
        :meth:`hvp` caveat), so this uses a *finite difference of the batched
        force field* -- the SAME construction (and SAME ``get_ef_gpu`` forward)
        that ``BatchDimer._hvp`` uses, guaranteeing both follow the identical PES:

            H @ n  ~=  -(F(R + delta*n) - F(R)) / delta        (forward diff)

        because ``F = -grad E`` => ``dF = -H dx`` => ``-(F1-F0)/delta = H n``.

        Parameters
        ----------
        atoms : ase.Atoms
            Geometry whose current positions define ``R`` (B=1).
        n : array-like, shape (3N,)
            Dimer axis / direction vector (unit, as the Dimer supplies it).
        delta : float
            FD half-length in Angstrom (default 0.005, matching ``DimerParams``).

        Returns
        -------
        (Hn, forces, energy) : tuple of torch.Tensor
            ``Hn`` (3N,) [Ha/A^2], ``forces`` (3N,) [Ha/A], ``energy`` scalar [Ha]
            -- all on this calculator's device/dtype, in Hartree units.
        """
        import numpy as _np
        self.prepare([atoms])
        N = len(atoms); dof = 3 * N
        E0, F0 = self.get_ef_gpu()                       # (1,), (1, nmax_dof) Hartree
        v = torch.zeros((self._atoms_B, self.nmax_dof),
                        dtype=self.dtype, device=self.device)
        nt = torch.as_tensor(_np.asarray(n, dtype=_np.float64).reshape(-1),
                             dtype=self.dtype, device=self.device)
        v[0, :dof] = nt[:dof]
        self.backup_coords()
        self.step_cart_(delta * v)
        _, F1 = self.get_ef_gpu()
        self.restore_coords()
        Hn = (-(F1 - F0) / delta)[0, :dof].clone()       # = H @ n, Ha/A^2
        forces = F0[0, :dof].clone()                     # Ha/A
        energy = E0[0].clone()                           # Ha
        return Hn, forces, energy
