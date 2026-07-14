# -*- coding: utf-8 -*-
"""Abstract base for MAPLE GPU-batched calculators (the ``get_ef_gpu`` path).

Mirrors ``CalcABC`` (calculator_base.py) for the single-structure path, but for
the *batch* path: one ``prepare()`` fixes B molecules' topology; thereafter
``get_ef_gpu``/``get_efh_gpu`` run batched forwards over the whole set. The
contract is ``get_ef_gpu()``/``get_efh_gpu()`` (Hartree, ``(B, nmax_dof)``
layout), NOT ASE ``calculate()`` — see BATCH_AUTHORING.md.

This base OWNS the surface that was duplicated across the 7 hand-rolled batch
calculators (coord ops, ``(B, nmax_dof)`` pack/scatter, unit conversion, padding,
the generic finite-difference + seeded-autograd Hessian, PBC fail-fast, the
registry). A backend subclass implements ONLY the model-specific surface:

  - ``__init__``: load the model, then call ``super().__init__(device, dtype)``.
  - ``_build_topology(atoms_list)``: cache the model-specific graph / species /
    AtomicData for the fixed batch (called at the end of ``prepare``).
  - ``_forward(coord, need_graph) -> (E, F_all, leaf)``: ONE batched forward.
    ``E`` is ``(B,)`` and ``F_all`` is ``(N, 3)`` in the backend's native energy
    unit (declared by ``MODEL_ENERGY_UNIT``); ``leaf`` is the position tensor
    with ``requires_grad=True`` when ``need_graph`` else ``None``. The base never
    converts inside ``_forward`` — it converts once, here, via
    ``MODEL_ENERGY_UNIT`` (killing the divide-EH2EV / multiply-EV2HARTREE / none
    three-way split the standalone classes had).

Capability class attributes let the base + dispatchers reason declaratively
instead of by hard-coded class-name sets:
  MODEL_NAMES, MODEL_ENERGY_UNIT, MODEL_DTYPE, SUPPORTS_PBC,
  SUPPORTED_HESSIAN_MODES, HAS_HVP, SUPPORTS_COUPLING, BATCHABLE.

Layout contract (canonical, adopted from UMABatchCalc — the vectorized one):
  * master ``coord`` is ``(N, 3)`` in ``dtype`` (f64) on ``device``.
  * ``_cols`` (3N,) is the flat scatter map: global atom g (mol b, local a) ->
    ``b*nmax_dof + 3*a + {0,1,2}`` inside a ``(B, nmax_dof)`` buffer.
  * forces ``(B, nmax_dof)``; per-structure Hessian block ``(B, nmax_dof,
    nmax_dof)``; ``P`` ``(B,)`` = padding-atom count ``Nmax_atoms - n_i``.

Duck-typing by design: like UMA under CalcABC, a backend may satisfy this
protocol by attribute presence without inheriting (do NOT ``isinstance`` gate in
dispatchers — use the shared predicate in ``dispatcher/_batch_calc_utils.py``).
"""
from __future__ import annotations

import numpy as np
import torch

# Reuse the single-structure chokepoint's constants/helpers — do NOT redefine
# EH2EV/EV2HARTREE here (inconsistency #1: the 7 classes each hardcoded one).
from .calculator_base import (  # noqa: F401
    EV2HARTREE,
    _convert_energy_force_units,
    reject_periodic_atoms,
)


# --------------------------------------------------------------------------- #
# Batch-calculator registry (mirror of _REGISTRY / register_calculator).       #
# --------------------------------------------------------------------------- #
_BATCH_REGISTRY: dict[str, type] = {}


def register_batch_calculator(cls):
    """Register a batch-calculator class under each lowercase name in MODEL_NAMES.

    Raises ValueError on a real collision (a different class already owns the
    name) so an accidental duplicate surfaces instead of silently overwriting.
    """
    for raw_name in cls.MODEL_NAMES:
        name = str(raw_name).lower()
        existing = _BATCH_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Batch calculator name '{name}' is already registered to "
                f"{existing.__name__}; refusing to overwrite with {cls.__name__}."
            )
        _BATCH_REGISTRY[name] = cls
    return cls


def get_registered_batch_calculator(name: str) -> type:
    return _BATCH_REGISTRY[str(name).lower()]


# --------------------------------------------------------------------------- #
# Lazy backend import + generic factory. Importing a backend module runs its    #
# @register_batch_calculator (populating _BATCH_REGISTRY). Each import is        #
# guarded so one missing optional dep (torchani / aimnet2calc / a MACE build)   #
# is skipped instead of sinking registration of the backends that ARE           #
# importable. Imports stay lazy (inside the helper) so importing THIS module     #
# never eagerly drags in every backend.                                         #
# --------------------------------------------------------------------------- #
_BACKENDS_IMPORTED = False

# Dotted paths for the batch backends (order irrelevant; each self-registers).
_BATCH_BACKEND_MODULES = (
    "maple.function.calculator.uma._uma_batch_calculator",
    "maple.function.calculator.aimnet._aimnet2_batch_calculator",
    "maple.function.calculator.aimnet._aimnet2_decoupled_batch_calculator",
    "maple.function.calculator.ani._ani_batch_calculator",
    "maple.function.calculator.mace._mace_batch_calculator",
    "maple.function.calculator.mace._mace_autograd_batch_calculator",
    "maple.function.calculator.mace._macepol_batch_calculator",
    "maple.function.calculator.mace._maceoff_batch_calculator",
)


def _import_batch_backends(force: bool = False) -> list:
    """Import every backend module so its @register_batch_calculator runs.

    Idempotent (guarded by ``_BACKENDS_IMPORTED``). Each import is wrapped in
    try/except so a backend whose optional dependency is missing (torchani /
    aimnet2calc / a specific MACE build) is skipped rather than breaking
    registration of the importable backends. Returns the module paths that
    failed to import (for diagnostics).
    """
    global _BACKENDS_IMPORTED
    failed = []
    if _BACKENDS_IMPORTED and not force:
        return failed
    import importlib
    for mod in _BATCH_BACKEND_MODULES:
        try:
            importlib.import_module(mod)
        except Exception:  # noqa: BLE001 - optional-dep guard; keep the registry alive
            failed.append(mod)
    _BACKENDS_IMPORTED = True
    return failed


def make_batch_calc(model, model_path=None, device="cuda", dtype=None, **opts):
    """Build a GPU-batched calculator by registered ``model`` name.

    Backend-agnostic front door to the batch registry: lazily imports the 7
    backend modules (so their ``@register_batch_calculator`` populates
    ``_BATCH_REGISTRY``), looks up the class for ``model`` (case-insensitive),
    and constructs it, forwarding ONLY the arguments that class's ``__init__``
    actually accepts (the 7 constructors are heterogeneous).

    Args:
      model: registered batch-calc name -- 'uma', 'aimnet2', 'aimnet2_decoupled',
        'ani2x'/'ani1x'/..., 'mace', 'mace_autograd', 'macepols'/'macepolm'/'macepoll'.
      model_path: checkpoint / traced-model path; forwarded (as ``None`` when
        None -- NOT the string 'None') whenever the backend declares a
        ``model_path`` arg, so a path-less ANI/MACE/MACE-POL falls back to its
        own local model dir. Backends without the arg (AIMNet2-decoupled) ignore it.
      device: 'cuda'|'cpu'|'gpu*'|'auto' token (backend _resolve_device normalizes).
      dtype: torch dtype; forwarded only when not None (else backend default, f64).
      **opts: backend-specific extras (task=, cutoff=, coupling_mode=,
        hessian_mode=, ...) forwarded verbatim; a caller-supplied opt wins over
        the derived device/dtype/model_path/model defaults.

    Raises ValueError when ``model`` is not a registered batch-calc name (lists
    the available names).
    """
    import inspect

    key = str(model).lower().strip()
    _import_batch_backends()
    try:
        cls = get_registered_batch_calculator(key)
    except KeyError:
        avail = ", ".join(sorted(_BATCH_REGISTRY)) or "<none importable>"
        raise ValueError(
            f"No batch calculator registered for model {model!r}. Available: {avail}."
        )

    params = inspect.signature(cls).parameters   # __init__ minus self
    _has = params.__contains__                    # forward only EXPLICIT ctor args

    kwargs = dict(opts)                           # caller opts win over derived below
    if _has("device"):
        kwargs.setdefault("device", device)
    if dtype is not None and _has("dtype"):
        kwargs.setdefault("dtype", dtype)
    # Forward model_path whenever the ctor declares it -- INCLUDING None, which is
    # the local-model-dir sentinel for ANI/MACE/MACE-POL (ANI's model_path is a
    # required positional, so omitting it would raise). Never stringify None.
    if _has("model_path"):
        kwargs.setdefault(
            "model_path", str(model_path) if model_path is not None else None)
    # A class registered under >1 name selects its variant by the lookup name
    # (ANI: ani2x/ani1x/...; MACE-POL: macepols/macepolm/macepoll). Single-name
    # backends keep their own ``model`` default unless the caller overrides via opts.
    if len(getattr(cls, "MODEL_NAMES", ())) > 1 and _has("model"):
        kwargs.setdefault("model", key)

    return cls(**kwargs)


def _resolve_device(device) -> torch.device:
    """Map a 'cuda'/'gpu*'/'cpu'/'auto'/'' token (or torch.device) to a device.

    The dispatcher's ``batch_device_str`` already normalizes to 'cuda'|'cpu';
    this is the calc-side fallback so a hand-built calc still does the right
    thing.
    """
    if isinstance(device, torch.device):
        return device
    d = str(device).lower()
    want_cuda = d.startswith("cuda") or d.startswith("gpu") or d in ("", "auto")
    return torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")


# --- environment vs model-capability exception triage (shared) ---------------
# Capability probes ("can this traced model ingest B>1?", "does it double-backward?")
# run the model inside try/except. A BARE ``except Exception`` there conflates two
# categories with opposite correct responses:
#   * ENVIRONMENT / HARDWARE failure (CUDA kernel-arch mismatch, OOM, driver, no
#     device) -- says NOTHING about the model. Swallowing it mislabels a perfectly
#     batchable model as "trace-locked" and reports a wrong diagnosis to the user.
#     Observed: on a V100 (sm_70) an sm_80-only torch build raises "no kernel image
#     is available for execution on the device" -> every MACE model was falsely
#     reported B=1-trace-locked. MUST propagate.
#   * MODEL / TRACE limitation (per-graph scatter dim_size frozen to 1, shape
#     mismatch on a B>1 graph) -- the real signal the probe is looking for.
_ENV_ERROR_MARKERS = (
    "no kernel image",            # binary has no cubin for this SM (arch mismatch)
    "out of memory",              # CUDA / host OOM
    "cuda error",                 # generic driver/runtime failure
    "cuda driver",
    "cuda runtime",
    "no cuda-capable device",
    "cudnn",
    "cublas",
    "device-side assert",
    "invalid device",
    "peer mapping",
    "initialization error",
)


def is_environment_error(exc: BaseException) -> bool:
    """True when ``exc`` is a hardware/driver/OOM failure, NOT a model limitation.

    Conservative by design: only exceptions that either are a known OOM type or
    carry an unambiguous environment marker are classified as environment. Anything
    else (shape/scatter/trace errors) stays a model-capability signal, so the
    trace-lock detection keeps working exactly as before.
    """
    if isinstance(exc, MemoryError):
        return True
    oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom is not None and isinstance(exc, oom):
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _ENV_ERROR_MARKERS)


def raise_if_environment_error(exc: BaseException) -> None:
    """Re-raise ``exc`` when it is an environment/hardware failure (see above).

    Use inside capability probes:  ``except Exception as e: raise_if_environment_error(e)``
    then treat the surviving exception as the model limitation being probed for.
    """
    if is_environment_error(exc):
        raise exc


class BatchCalcABC:
    """Protocol base for GPU-batched calculators. Subclass overrides the model
    surface (``_build_topology`` + ``_forward`` + class attrs); the rest is here.
    """

    # ---- capability protocol (subclass overrides what applies) -------------
    MODEL_NAMES: tuple = ()
    MODEL_ENERGY_UNIT: str = "eV"           # 'eV' | 'hartree'; base converts once
    MODEL_DTYPE: torch.dtype = torch.float32  # model compute dtype (forward bridge)
    SUPPORTS_PBC: bool = False
    SUPPORTED_HESSIAN_MODES: tuple = ("numerical",)  # subset of ('numerical','autograd')
    HAS_HVP: bool = False
    # True => perturbing one molecule leaks into the others (Coulomb-coupled
    # MACE-POL / charge-equilibration AIMNet2-native): a block-diagonal batch is
    # numerically unsafe. Replaces the hard-coded _COUPLED_BATCH_CALC_NAMES set.
    SUPPORTS_COUPLING: bool = False
    # False => this potential can only run one structure at a time (coupled);
    # dispatchers gate a B>1 batch on this instead of by class name.
    BATCHABLE: bool = True
    # False => the batched path applies NO implicit-solvent correction, so a job that
    # asked for #solv(method=gbsa, implicit=...) must NOT be silently routed here (it
    # would return gas-phase numbers AND bypass the single-structure path's
    # reject_implicit_solvent_derivatives gate). Every batch backend is gas-phase-only
    # today; the dispatcher-layer gate (dispatcher/_batch_calc_utils.py ::
    # reject_batched_implicit_solvent) fails fast on False and lets a future
    # solvent-capable batch backend through by flipping this to True.
    SUPPORTS_IMPLICIT_SOLVENT: bool = False

    def __init__(self, device="cuda", dtype: torch.dtype = torch.float64):
        self.device = _resolve_device(device)
        self.dtype = dtype
        # prepare() state (all set in prepare)
        self._prepared = False
        self._atoms_B = 0
        self._ptr = None
        self.numbers = None
        self.mol_idx = None
        self._local_atom = None
        self._n_b = None
        self._cols = None
        self.coord = None
        self.N_atoms = 0
        self.Nmax_atoms = 0
        self.nmax_dof = 0
        self._coord_backup = None
        self._periodic = False  # set in prepare(); shared hook for the PBC MD stack

    # =====================================================================  #
    # prepare — topology fixed once per batch (COMMON; hooks _build_topology) #
    # =====================================================================  #
    def prepare(self, atoms_list, fixed_nmax=None):
        """Fix topology + initial coords for a batch of B molecules.

        ``fixed_nmax`` (optional): pin the padded per-structure DOF size so a
        batch optimizer (BatchPRFO) shares one layout across iterations. Must be
        a multiple of 3 and >= 3*max_atoms (validated for ALL backends here —
        inconsistency #5: only 2 of 7 classes validated it).
        """
        device, dtype = self.device, self.dtype
        # PBC gate (aligned with ai-maple-md Phase-B). Molecular backends
        # (SUPPORTS_PBC=False) reject any periodic replica (inconsistency #6:
        # 3 of 7 raised, 4 silently ignored). A PBC-capable backend
        # (SUPPORTS_PBC=True -- UMA-periodic / MACE with _build_edges_pbc) instead
        # requires a HOMOGENEOUS pbc state across the batch and keeps cell/pbc as
        # calc-internal state, carried on atoms_list into _build_topology / _forward
        # (the get_ef_gpu return contract is unchanged; cell fixed = NVT).
        # ``self._periodic`` is the shared hook the sibling batched-MD stack reads.
        pbc_flags = [bool(np.any(np.asarray(getattr(at, "pbc", False)))) for at in atoms_list]
        self._periodic = any(pbc_flags)
        if self._periodic and not self.SUPPORTS_PBC:
            reject_periodic_atoms(atoms_list[pbc_flags.index(True)], type(self).__name__)
        if self._periodic and not all(pbc_flags):
            raise NotImplementedError(
                f"{type(self).__name__}: heterogeneous PBC across the batch is unsupported; "
                "all replicas must share one periodic state (homogeneous-pbc gate).")

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
        self._n_b = self._ptr[1:] - self._ptr[:-1]  # (B,)

        self.N_atoms = int(self.numbers.numel())
        self.Nmax_atoms = int(max((len(at) for at in atoms_list), default=0))
        req = 3 * self.Nmax_atoms
        if fixed_nmax is None:
            self.nmax_dof = req
        else:
            self.nmax_dof = int(fixed_nmax)
            if self.nmax_dof % 3 != 0:
                raise ValueError(f"fixed_nmax={self.nmax_dof} is not a multiple of 3 Cartesian DOFs.")
            if self.nmax_dof < req:
                raise ValueError(
                    f"fixed_nmax={self.nmax_dof} too small for this batch; need >= {req}."
                )

        if self.N_atoms > 0:
            pos = torch.cat(
                [torch.tensor(at.get_positions(), dtype=dtype) for at in atoms_list], dim=0
            )
        else:
            pos = torch.zeros((0, 3), dtype=dtype)
        self.coord = pos.to(device).contiguous()

        # flat scatter map: global atom g (mol b, local a) -> b*nmax_dof + 3*a + k
        if self.N_atoms > 0:
            base = self.mol_idx * self.nmax_dof + 3 * self._local_atom  # (N,)
            self._cols = (base[:, None] + torch.arange(3, device=device)[None, :]).reshape(-1)
        else:
            self._cols = torch.zeros(0, dtype=torch.int64, device=device)

        self._coord_backup = None
        self._build_topology(atoms_list)  # subclass hook (edges / AtomicData / species)
        self._prepared = True

    # ---------------------------------------------------------- coord ops ---
    @torch.no_grad()
    def step_cart_(self, s_cart: torch.Tensor):
        """In-place displacement; ``s_cart`` is (B, nmax_dof), padded per structure."""
        assert self._prepared, "call prepare() first"
        assert s_cart.shape == (self._atoms_B, self.nmax_dof), (
            f"step_cart_ expects (B,{self.nmax_dof}), got {tuple(s_cart.shape)}"
        )
        if self.N_atoms == 0:
            return
        s = s_cart.to(self.device, dtype=self.dtype).reshape(-1)
        self.coord.add_(s[self._cols].reshape(self.N_atoms, 3))  # vectorized, no .item()

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

    # --------------------------------------------------- pack / unit helpers
    def _pad_forces(self, F_all: torch.Tensor) -> torch.Tensor:
        """(N,3) per-atom -> (B, nmax_dof) padded, via the flat _cols scatter."""
        B, nmax = self._atoms_B, self.nmax_dof
        F_pad = torch.zeros((B, nmax), dtype=self.dtype, device=self.device)
        if self.N_atoms > 0:
            F_pad.reshape(-1)[self._cols] = F_all.reshape(-1).to(self.dtype)
        return F_pad

    def _to_hartree(self, energy, forces=None):
        """Convert model-unit (E, F) to Hartree once, per MODEL_ENERGY_UNIT."""
        return _convert_energy_force_units(energy, forces, source_unit=self.MODEL_ENERGY_UNIT)

    def _padding_counts(self) -> torch.Tensor:
        return (self.Nmax_atoms - self._n_b).to(torch.int64)  # (B,)

    def _empty_ef(self):
        z = lambda *s: torch.zeros(s, dtype=self.dtype, device=self.device)
        return z(0), z(0, 0)

    def _empty_efh(self):
        z = lambda *s: torch.zeros(s, dtype=self.dtype, device=self.device)
        return z(0), z(0, 0), z(0, 0, 0), torch.zeros(0, dtype=torch.int64, device=self.device)

    # =====================================================================  #
    # get_ef_gpu — one batched forward -> (E, F) Hartree, padded (COMMON)     #
    # =====================================================================  #
    def get_ef_gpu(self):
        assert self._prepared, "call prepare() first"
        if self._atoms_B == 0:
            return self._empty_ef()
        E, F_all, _ = self._forward(self.coord, need_graph=False)
        F_pad = self._pad_forces(F_all)
        E_ha, F_ha = self._to_hartree(E.to(self.dtype), F_pad)
        return E_ha, F_ha

    # =====================================================================  #
    # get_efh_gpu — energy + forces + per-structure Hessian (COMMON dispatch) #
    # =====================================================================  #
    def get_efh_gpu(self, movable_masks=None, mode=None, delta: float = 2e-3, chunk_size=None):
        """Return (E (B,), F (B,nmax_dof), H (B,nmax_dof,nmax_dof), P (B,)) Hartree.

        ``mode`` defaults to the backend's first SUPPORTED_HESSIAN_MODES entry
        (inconsistency #10: the classes drifted between a hessian_mode kwarg, an
        internal _hess_mode, and coupling_mode). H is per-structure block-padded.
        """
        assert self._prepared, "call prepare() first"
        if self._atoms_B == 0:
            return self._empty_efh()
        m = (mode or self.SUPPORTED_HESSIAN_MODES[0]).lower()
        m = "autograd" if m in ("autograd", "analytic") else "numerical"
        if m == "autograd" and "autograd" in self.SUPPORTED_HESSIAN_MODES:
            return self._efh_analytic(movable_masks=movable_masks, chunk_size=chunk_size)
        return self._efh_fd(movable_masks=movable_masks, delta=delta)

    # ------------------------------- numerical central-FD Hessian (COMMON) --
    def _efh_fd(self, movable_masks=None, delta: float = 2e-3):
        """Batched central finite-difference Hessian, parallel over molecules.

        For each local DOF k=(a,c) it perturbs that DOF in EVERY molecule that
        owns atom ``a`` (and, under movable_masks, for which ``a`` is movable) in
        ONE batched forward, so a single (+/-) pair yields column k for the whole
        batch. Vectorized scatter, no per-atom host-sync in the hot path.

        ponytail: O(6*Nmax_atoms) full-batch forwards — correct and molecule-
        parallel. A backend that can pack the +/- replicas into ONE super-batch
        (UMA's chunked plan) overrides this for extra DOF-level parallelism.
        """
        B, N = self._atoms_B, self.N_atoms
        nmax, device, dtype = self.nmax_dof, self.device, self.dtype
        # base point (reused as the returned gradient)
        E0, F0_all, _ = self._forward(self.coord, need_graph=False)
        F_pad = self._pad_forces(F0_all)
        P = self._padding_counts()
        if N == 0:
            E_ha, F_ha = self._to_hartree(E0.to(dtype), F_pad)
            return E_ha, F_ha, torch.zeros((B, nmax, nmax), dtype=dtype, device=device), P

        mov = self._resolve_movable(movable_masks)          # per-mol list[int] local atoms
        # movable global-atom mask (rows whose Hessian entries we keep)
        mv_atom = torch.zeros(N, dtype=torch.bool, device=device)
        ptr = self._ptr.tolist()
        for i in range(B):
            for a in mov[i]:
                mv_atom[ptr[i] + a] = True

        H = torch.zeros((B, nmax, nmax), dtype=dtype, device=device)
        coord0 = self.coord
        # union of movable local-atom indices across molecules -> DOFs to sweep
        max_mov_a = max((max(m) + 1 for m in mov if m), default=0)
        for a in range(max_mov_a):
            # molecules that own local atom a AND for which a is movable
            owners = [i for i in range(B) if a < int(self._n_b[i].item()) and a in mov[i]]
            if not owners:
                continue
            g_rows = torch.tensor([ptr[i] + a for i in owners], dtype=torch.long, device=device)
            own_idx = torch.tensor(owners, dtype=torch.long, device=device)
            for c in range(3):
                k = 3 * a + c
                cp = coord0.clone(); cp[g_rows, c] += delta
                _, Fp, _ = self._forward(cp, need_graph=False)
                cm = coord0.clone(); cm[g_rows, c] -= delta
                _, Fm, _ = self._forward(cm, need_graph=False)
                col = (-(Fp - Fm) / (2.0 * delta))          # (N,3) = dF/dx_k rows
                col = col * mv_atom[:, None]                 # zero non-movable rows
                col_pad = self._pad_forces(col)              # (B, nmax)
                # OWNER-MASKED column write. Column k belongs ONLY to the molecules
                # that own+move local atom a. A non-owner (atom a frozen there, or the
                # molecule has < a+1 atoms) was NEVER perturbed, so its Fp-Fm is 0 in
                # exact arithmetic -- but Fp/Fm are two INDEPENDENT forwards, so in fp
                # they differ by ~1e-9 and the 1/(2*delta) division amplifies that by
                # ~500x into ~1e-6. Writing the whole column (H[:, :, k] = col_pad)
                # leaked that FD noise into the frozen / padding columns, which the
                # PHVA contract requires to be EXACTLY 0 ("the remaining atoms carry
                # infinite masses"). Measured 3.29e-6 Ha/A^2 on heterogeneous batches
                # (aimnet2_decoupled); 0 after this masking. Owner columns are
                # bit-identical to before -- non-owner columns simply stay at their
                # zero-initialized value.
                H[own_idx, :, k] = col_pad[own_idx]
        H = 0.5 * (H + H.transpose(1, 2))                    # symmetrize
        E_ha, F_ha = self._to_hartree(E0.to(dtype), F_pad)
        _, H_ha = self._to_hartree(E0.to(dtype), H)          # H uses same eV->Ha scale
        return E_ha, F_ha, H_ha, P

    # ------------------------------ seeded double-backward Hessian (COMMON) -
    def _efh_analytic(self, movable_masks=None, chunk_size=None):
        """Exact autograd Hessian from the energy graph's force field.

        Requires ``_forward(coord, need_graph=True)`` to return a ``F_all`` that
        carries a graph w.r.t. ``leaf``. Seeds each local response DOF; one
        backward fills that Hessian row for every molecule (molecule batching).
        Optional ``torch.vmap`` over the seed basis (MACE compute_hessians_vmap
        pattern) with a per-row loop fallback.
        """
        B, N = self._atoms_B, self.N_atoms
        nmax, device, dtype = self.nmax_dof, self.device, self.dtype
        E, F_all, leaf = self._forward(self.coord, need_graph=True)
        if leaf is None:
            raise RuntimeError(
                f"{type(self).__name__}._forward returned leaf=None under need_graph=True; "
                "analytic Hessian needs the position leaf."
            )
        F_pad = self._pad_forces(F_all.detach())
        P = self._padding_counts()
        if N == 0:
            E_ha, F_ha = self._to_hartree(E.detach().to(dtype), F_pad)
            return E_ha, F_ha, torch.zeros((B, nmax, nmax), dtype=dtype, device=device), P

        mov = self._resolve_movable(movable_masks)
        ptr = self._ptr.tolist()
        H = torch.zeros((B, nmax, nmax), dtype=dtype, device=device)
        K = 3 * max((len(m) for m in mov), default=0)
        # local response DOF ld = 3*slot + c; owners[ld] = [(mol i, local atom ap)]
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
            seeds.append(go); owners[ld] = own

        def _scatter(ld, colN3):
            slot, c = ld // 3, ld % 3
            for (i, ap) in owners[ld]:
                r = 3 * ap + c
                for aq in mov[i]:
                    H[i, r, 3 * aq:3 * aq + 3] = -colN3[ptr[i] + aq, :].to(dtype)

        if chunk_size is not None and K > 0:
            GO = torch.stack(seeds, dim=0)  # (K,N,3)
            vjp = lambda v: torch.autograd.grad(
                F_all, leaf, grad_outputs=v, retain_graph=True, create_graph=False)[0]
            COLS = torch.vmap(vjp, in_dims=0, out_dims=0, chunk_size=int(chunk_size))(GO)
            for ld in range(K):
                _scatter(ld, COLS[ld])
        else:
            for ld in range(K):
                col = torch.autograd.grad(
                    F_all, leaf, grad_outputs=seeds[ld], retain_graph=True, create_graph=False)[0]
                _scatter(ld, col)
        H = 0.5 * (H + H.transpose(1, 2))
        E_ha, F_ha = self._to_hartree(E.detach().to(dtype), F_pad)
        _, H_ha = self._to_hartree(E.detach().to(dtype), H)
        return E_ha, F_ha, H_ha, P

    # ----------------------------------------------------- movable helpers --
    def _resolve_movable(self, movable_masks):
        """Per-structure list[int] of movable atom indices (None = all atoms).

        ``movable_masks`` is None, or a length-B sequence where entry b is None
        (all atoms), a bool mask of length n_b, or an explicit index list.
        """
        B = self._atoms_B
        n_b = self._n_b.tolist()
        if movable_masks is None:
            return [list(range(n_b[b])) for b in range(B)]
        out = []
        for b in range(B):
            m = movable_masks[b]
            if m is None:
                out.append(list(range(n_b[b]))); continue
            m_arr = np.asarray(m)
            if m_arr.dtype == bool:
                out.append([int(i) for i in np.nonzero(m_arr)[0]])
            else:
                idxs = [int(i) for i in m_arr.reshape(-1)]
                assert all(0 <= a < n_b[b] for a in idxs), (
                    f"movable index out of range for structure {b} (n={n_b[b]}): {idxs}")
                out.append(idxs)
        return out

    # ======================================================= abstract hooks #
    def _build_topology(self, atoms_list):
        """Cache model-specific batch topology (edges / AtomicData / species).

        Called at the END of prepare() with the fixed atoms_list. Default no-op
        for backends that build the graph inside _forward each call.
        """
        return None

    def _forward(self, coord: torch.Tensor, need_graph: bool):
        """ONE batched forward. MUST override.

        Returns (E (B,), F_all (N,3), leaf|None) in the backend's native energy
        unit (MODEL_ENERGY_UNIT). ``leaf`` is the requires_grad position tensor
        when need_graph else None. The base converts units + pads + scatters.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _forward(coord, need_graph).")

    def hvp(self, v: torch.Tensor) -> torch.Tensor:
        """Batched Hessian-vector product H@v, (B,nmax_dof)->(B,nmax_dof) Hartree.

        Optional; only backends with HAS_HVP=True override. Default raises so a
        Dimer HVP path fails loudly rather than misreading a differently-shaped
        forward (mirrors CalcABC.get_hvp).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement hvp; set HAS_HVP and override "
            "for the HVP-enabled Dimer path.")


# --------------------------------------------------------------------------- #
# In-tree runnable check: a trivial harmonic "model" (no ML backend) exercises #
# the base plumbing (prepare/step/pad/scatter/get_ef_gpu + FD & autograd       #
# Hessian). E = 0.5*k*sum(r^2) about the origin -> F = -k*r, H = k*I. Verifies  #
# the base independent of any real model. Needs torch (runs on the Ibex env).  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from ase import Atoms

    K = 1.7

    class _HarmonicBatch(BatchCalcABC):
        MODEL_NAMES = ("_harmonic_test",)
        MODEL_ENERGY_UNIT = "eV"
        SUPPORTED_HESSIAN_MODES = ("numerical", "autograd")

        def _forward(self, coord, need_graph):
            leaf = coord.detach().to(self.MODEL_DTYPE).requires_grad_(need_graph)
            E_atoms = 0.5 * K * (leaf ** 2).sum(dim=1)          # (N,)
            E = torch.zeros(self._atoms_B, dtype=leaf.dtype, device=leaf.device)
            E.index_add_(0, self.mol_idx, E_atoms)              # (B,) per-mol energy
            if need_graph:
                F_all = -torch.autograd.grad(E_atoms.sum(), leaf, create_graph=True)[0]
            else:
                F_all = (-K * leaf)
            return E, F_all, (leaf if need_graph else None)

    mols = [Atoms("H2O", positions=np.random.RandomState(0).randn(3, 3)),
            Atoms("H2", positions=np.random.RandomState(1).randn(2, 3))]
    c = _HarmonicBatch(device="cpu", dtype=torch.float64)
    c.prepare(mols)
    E, F = c.get_ef_gpu()
    # F for a 3-atom mol should equal -K*coord scattered into first 9 slots.
    assert E.shape == (2,) and F.shape == (2, c.nmax_dof), (E.shape, F.shape)
    # analytic vs FD Hessian both = K on the diagonal of each atom's block.
    _, _, H_fd, P = c.get_efh_gpu(mode="numerical")
    _, _, H_ag, _ = c.get_efh_gpu(mode="autograd")
    diag0 = torch.diagonal(H_fd[0])[:9]                         # H2O = 9 real DOFs
    kHa = K * EV2HARTREE                                        # H is returned in Hartree
    assert torch.allclose(diag0, torch.full_like(diag0, kHa), atol=1e-6), (diag0, kHa)
    assert torch.allclose(H_fd[0][:9, :9], H_ag[0][:9, :9], atol=1e-6), "FD vs autograd mismatch"
    assert int(P[0]) == 0 and int(P[1]) == 1, P                 # mol1 has 1 padding atom vs Nmax=3
    print("BatchCalcABC self-test PASS: E", tuple(E.shape), "F", tuple(F.shape),
          "| H diag ~= K*EV2HARTREE =", round(kHa, 6), "| FD==autograd | P", P.tolist())
