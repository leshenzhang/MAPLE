# -*- coding: utf-8 -*-
"""
Batched AUTOGRAD (native ``F = -nabla E``) standard-MACE calculator + autodiff HVP.

Direction D2 (automatic differentiation) for the MAPLE GPU-batch library.

WHY A SEPARATE FILE
-------------------
The other batch calculators are inference-only:
  * ``_uma_batch_calculator.py`` -- forces come from fairchem's predict HEAD (NOT
    a differentiated energy), Hessian is NUMERICAL finite-difference only
    (``supported_hessian_modes = ("numerical",)``).
  * ``_mace_batch_calculator.py`` -- targets a *TorchScript-traced* ``maceomol``
    wrapper with the 6-positional-arg signature
    ``model(positions, node_attrs, edge_index, shifts, batch, ptr)``.

This file is the AUTODIFF backend. It loads a *standard, un-traced* MACE-OFF23
foundation model through the ``mace`` python package
(``mace.calculators.mace_off(..., return_raw_model=True)``), keeps the full torch
autograd graph, and computes
    E                       -> model(data)["energy"]                  (B,)  [eV]
    F = -dE/dx              -> -autograd.grad(E.sum(), positions)      (N,3) [eV/A]
    H@v = d(dE/dx . v)/dx   -> double-backward (NO full 3N x 3N Hessian)
all by automatic differentiation. MACE-OFF is a PURE LOCAL MLIP (per-graph
``scatter_sum`` energy pooling, no long-range Coulomb / charge equilibration), so
a block-diagonal multi-graph batch is EXACTLY isolated by construction.

ALGORITHMIC COST (the point of autodiff over finite difference)
---------------------------------------------------------------
ONE energy forward builds the graph; forces are its FIRST backward, the
Hessian-vector product H@v is the SECOND backward of that SAME graph. So a full
HVP costs **1 model forward + 2 backward** -- INDEPENDENT of system size N.
Finite-difference HVP needs 2 force evaluations per displaced direction and a
full numerical Hessian needs 2*3N force evaluations (= 6N forwards). All
second-order consumers (``get_efh_gpu`` analytic, ``hvp_batch``, the dimer
``hvp_fn``) share ONE ``_forward_with_graph()`` -- no redundant forward. The
distance-filtered neighbour list is cached and rebuilt only when the positions
actually change (``_edges_dirty``), so repeated HVPs at a fixed geometry (e.g.
dimer rotation inner loop) reuse it. ``self._n_forward`` counts model forwards
for verification.

HESSIAN MODE (opt-in, OLD numerical FD is the default + parity oracle)
---------------------------------------------------------------------
``get_efh_gpu`` dispatches on ``hessian_mode``:
  * ``'numerical'`` (DEFAULT) -- ``_efh_fd``, the 6N central-FD Hessian. This is
    the OLD path and is KEPT verbatim as both the fallback AND the byte-parity
    ORACLE the autograd Hessian is validated against.
  * ``'autograd'`` (OPT-IN)   -- ``_efh_analytic``, the seeded block-diagonal
    DOUBLE-BACKWARD Hessian (the energy graph's ``autograd.grad(forces, pos)``
    row-loop over the unit basis ``eye(3N)`` with ``create_graph=True`` on the
    forces). Optional ``torch.vmap`` acceleration over the seed basis is tried
    first when ``hessian_chunk_size`` is set, with a hard try/except fall-back to
    the row-loop, and the whole autograd attempt falls back to ``_efh_fd`` on any
    error. The model is loaded RAW/EAGER (no ``torch.compile``, no cuEq) because
    double-backward is unsupported under those (pytorch#91469).

References for the autograd Hessian / HVP
-----------------------------------------
  * Yuan et al., "Analytical ML Hessians for ...", Nat. Commun. 2024,
    DOI 10.1038/s41467-024-52481-5 -- autograd (double-backward) molecular
    Hessians as the exact, O(delta)-error-free alternative to finite difference.
  * MACE ``compute_hessians_vmap`` (github ACEsuit/mace, mace/modules/utils.py):
    ``torch.vmap`` of ``autograd.grad`` over ``eye(3N)`` with a per-row loop
    fallback (MACE issue #488) -- the pattern mirrored here.
  * pytorch#91469 -- double-backward is NOT supported under ``torch.compile`` /
    fused kernels, hence the eager raw-model requirement.

CONTRACT (mirrors ``UMABatchCalc`` / ``MACEBatchCalc`` / ``AIMNet2BatchCalc``)
-----------------------------------------------------------------------------
    prepare(atoms_list, fixed_nmax=None)
    step_cart_(s_cart: (B, nmax_dof)) / set_coords_(coord: (N,3))
    backup_coords() / restore_coords()
    get_ef_gpu()  -> (E_Ha (B,), F_Ha (B, nmax_dof))
    get_efh_gpu() -> (E_Ha (B,), F_Ha (B, nmax_dof),
                      H_Ha (B, nmax_dof, nmax_dof), P (B,) int64)
NEW autodiff extensions:
    hvp_batch(v: (B, nmax_dof)) -> (Hn_Ha (B, nmax_dof), E_Ha (B,))   # H@v, no full H
    make_hvp_fn()  -> hvp_fn(atoms, n_flat) -> (Hn, forces, energy)   # dimer.py callback
Units: Hartree, Hartree/Angstrom, Hartree/Angstrom^2 (MACE returns eV;
EV2HARTREE = 1/27.211386245988). f64 master coords + f64 model for HVP accuracy.

ENV-COMPAT PREAMBLE
-------------------
``import mace.calculators`` pulls ``mace.tools.train`` -> ``from torchmetrics
import Metric`` -> a torchvision build whose ``torchvision::nms`` op is broken in
this env. We never train, so a tiny ``torchmetrics`` stub exposing ``Metric`` is
installed into ``sys.modules`` before importing mace (idempotent). Mirrors the
ray.serve stub in the UMA calculator.
"""

# --------------------------------------------------------------------------- #
# Env-compat preamble (must run before importing mace). Idempotent.            #
# --------------------------------------------------------------------------- #
import sys as _sys
import types as _types

import torch

try:  # e3nn / MACE checkpoints pickle a `slice`; allow it for weights_only loads
    torch.serialization.add_safe_globals([slice])
except Exception:
    pass


def _install_torchmetrics_stub() -> None:
    """Stub ``torchmetrics`` so ``mace.tools.train`` imports without torchvision."""
    if "torchmetrics" in _sys.modules:
        return
    try:
        import torchmetrics  # noqa: F401  -- imports cleanly -> leave it alone
        return
    except Exception:
        pass
    tm = _types.ModuleType("torchmetrics")
    tm._maple_stub = True

    class _StubMetric:  # only needed as a base-name for class definitions
        pass

    tm.Metric = _StubMetric
    _sys.modules["torchmetrics"] = tm


_install_torchmetrics_stub()
# --------------------------------------------------------------------------- #

import os
from typing import List, Optional, Sequence

import numpy as np
from ase import Atoms

try:
    from mace.calculators.foundations_models import mace_off
except Exception as exc:  # pragma: no cover
    raise ImportError(f"mace is not importable for the autograd calculator: {exc}")

# Reuse the batch base's registry + the single EV2HARTREE chokepoint (do NOT
# redefine EV2HARTREE/EH2EV here -- the base imports them from calculator_base).
from ..batch_calculator_base import (  # noqa: E402
    BatchCalcABC,
    EV2HARTREE,
    register_batch_calculator,
)


def _one_hot_node_attrs(Z: torch.Tensor, atomic_number_table: Sequence[int],
                        dtype=torch.float64) -> torch.Tensor:
    """One-hot node features over the model's atomic-number table (its ordering)."""
    table = torch.tensor(list(atomic_number_table), dtype=torch.long, device=Z.device)
    eq = (Z[:, None] == table[None, :])
    if not torch.all(eq.any(dim=1)):
        miss = Z[~eq.any(dim=1)].unique().tolist()
        raise ValueError(f"Atomic number(s) {miss} not in MACE table "
                         f"{list(atomic_number_table)}")
    return eq.to(dtype)


@register_batch_calculator
class MACEAutogradBatchCalc(BatchCalcABC):
    """Batched standard-MACE calculator with NATIVE autograd F/HVP. See module docstring."""

    # ---- capability protocol (BatchCalcABC declarative attrs) --------------
    MODEL_NAMES = ("mace_autograd",)
    MODEL_ENERGY_UNIT = "eV"          # MACE returns eV / eV.A^-1; base -> Hartree once
    MODEL_DTYPE = torch.float64       # f64 model (HVP accuracy); master coord also f64
    SUPPORTS_PBC = False              # pure-local gas-phase MACE-OFF (no PBC)
    SUPPORTED_HESSIAN_MODES = ("numerical", "autograd")  # numerical = default + parity oracle
    HAS_HVP = True                    # autograd HVP: hvp / hvp_batch / make_hvp_fn
    SUPPORTS_COUPLING = False         # pure-local MLIP; block-diagonal batch is exactly isolated
    BATCHABLE = True                  # local energy pooling -> B>1 batches safely

    # Canonical hessian_mode values; default 'numerical' (FD oracle). Legacy
    # 'analytic' is accepted as an alias of 'autograd' (double-backward). Kept as a
    # back-compat lowercase alias (pre-retrofit callers / sibling backends read this).
    supported_hessian_modes = ("numerical", "autograd")

    def __init__(self,
                 model_path: str,
                 device: str = "cuda",
                 model: str = "maceoff23s",
                 dtype: torch.dtype = torch.float64,
                 implicit: str = "none",
                 solvent: str = "none",
                 hessian: Optional[str] = None,
                 hessian_mode: Optional[str] = None,
                 hessian_chunk_size: Optional[int] = None):
        if str(solvent).lower() not in ("none", "", "vacuum", "gas"):
            raise ValueError(
                f"MACEAutogradBatchCalc is gas-phase ONLY; got solvent={solvent!r}.")

        # base.__init__ resolves device (torch.device) + dtype and inits the generic
        # prepared-state; derive the 'cuda'|'cpu' string it settled on for mace_off.
        super().__init__(device, dtype)
        dev = self.device.type
        self.mdtype = dtype         # model dtype (f64 for HVP accuracy)
        self._model_name = model
        self._model_path = model_path

        # ---- Hessian mode resolution (OPT-IN discipline) -------------------
        # Canonical knob = ``hessian_mode`` in {'numerical','autograd'}. The
        # legacy ``hessian`` kwarg ('analytic'/'numerical') is still honoured for
        # back-compat. When NEITHER is given the default is 'numerical' -- the OLD
        # 6N finite-difference path that doubles as the parity ORACLE (user 铁则:
        # every new method opt-in, default = old behaviour, old path = oracle).
        _raw = hessian_mode if hessian_mode is not None else hessian
        if _raw is None:
            _raw = "numerical"
        _m = str(_raw).lower()
        self.hessian_mode = "autograd" if _m in ("autograd", "analytic") else "numerical"
        self.hessian = self.hessian_mode             # back-compat attribute
        # vmap seed-basis chunk size for the autograd Hessian (None = pure
        # row-loop, the validated default; an int enables the opt-in vmap try).
        self._hessian_chunk_size = (int(hessian_chunk_size)
                                    if hessian_chunk_size is not None else None)

        # Load the *raw* (un-traced) MACE foundation model -> full autograd graph.
        raw = mace_off(model=model_path, device=dev, return_raw_model=True)
        self.model = raw.to(self.mdtype).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.r_max = float(self.model.r_max)
        self.atomic_numbers = [int(z) for z in self.model.atomic_numbers]

        self._n_forward = 0   # model-forward counter (algorithmic-cost verification)

        # ---- model-specific prepared-state ONLY. base.__init__ already inits the
        #      generic layout state (_prepared / _atoms_B / _ptr / numbers / mol_idx /
        #      _local_atom / _n_b / _cols / coord / N_atoms / Nmax_atoms / nmax_dof /
        #      _coord_backup); do NOT re-init those here.
        self.node_attrs = None          # one-hot node features over the MACE Z-table
        self._base = None               # flat first-component scatter index (N,)
        self.cand_i = None
        self.cand_j = None
        self._cell = None               # per-graph cell (unused non-periodic; data dict)
        # neighbour-list cache (rebuilt only when positions change)
        self._edges_dirty = True
        self._ei_cache = None
        self._sh_cache = None

    # ------------------------------------------------------------------ utils
    def reset_fwd_count(self):
        self._n_forward = 0

    @property
    def n_forward(self) -> int:
        return self._n_forward

    # =====================================================================
    # topology hook -- called at the END of BatchCalcABC.prepare(), which has
    # already built ptr/numbers/mol_idx/_local_atom/_n_b/_cols/coord/N_atoms/
    # Nmax_atoms/nmax_dof (with the shared PBC reject + fixed_nmax validation) and
    # reset _coord_backup. Here we cache ONLY the MACE-specific batch topology.
    # =====================================================================
    def _build_topology(self, atoms_list):
        device = self.device
        B = self._atoms_B

        # one-hot node features over the model's atomic-number table
        self.node_attrs = (_one_hot_node_attrs(self.numbers, self.atomic_numbers, self.mdtype)
                           if self.N_atoms > 0
                           else torch.zeros((0, len(self.atomic_numbers)),
                                            dtype=self.mdtype, device=device))

        # flat first-component scatter index (MACE layout uses base, base+1, base+2);
        # _local_atom (built by the base) == global row - ptr[mol] == local atom idx,
        # so this is byte-identical to the pre-retrofit prepare()'s _base.
        if self.N_atoms > 0:
            self._base = self.mol_idx * self.nmax_dof + 3 * self._local_atom
        else:
            self._base = torch.zeros((0,), dtype=torch.int64, device=device)

        # Candidate intra-molecule pairs = FULL triu per molecule (offset by ptr),
        # filtered by distance each rebuild -> robust to arbitrary geometry moves.
        s = self._ptr[:-1]
        ci, cj = [], []
        for b in range(B):
            n = int(self._n_b[b].item())
            off = int(s[b].item())
            if n >= 2:
                iu, ju = torch.triu_indices(n, n, offset=1, device=device)
                ci.append(iu + off)
                cj.append(ju + off)
        self.cand_i = torch.cat(ci) if ci else torch.zeros((0,), dtype=torch.int64, device=device)
        self.cand_j = torch.cat(cj) if cj else torch.zeros((0,), dtype=torch.int64, device=device)

        # per-graph cell (unused for non-periodic energy/force, kept for the data dict)
        self._cell = torch.zeros((B, 3, 3), dtype=self.mdtype, device=device)

        # neighbour-list cache invalidated (topology / geometry changed)
        self._edges_dirty = True
        self._ei_cache = None
        self._sh_cache = None

    # =====================================================================
    # coordinate ops. Base owns the scatter/backup; here we ALSO mark the
    # neighbour-list cache dirty (the MACE-specific extra) after any move.
    # =====================================================================
    @torch.no_grad()
    def step_cart_(self, s_cart: torch.Tensor):
        """Base in-place displacement (same flat scatter) + mark edges dirty."""
        super().step_cart_(s_cart)
        self._edges_dirty = True

    @torch.no_grad()
    def set_coords_(self, coord: torch.Tensor):
        # NOT a plain super() call: the "positions unchanged -> keep the cache"
        # early-return is a MACE-specific optimization (free repeated HVP at a fixed
        # geometry, e.g. the dimer rotation inner loop) that the base lacks.
        assert self._prepared, "call prepare() first"
        assert coord.shape == (self.N_atoms, 3)
        c = coord.to(self.device, self.dtype)
        if self.coord.shape == c.shape and torch.equal(self.coord, c):
            return
        self.coord.copy_(c)
        self._edges_dirty = True

    # backup_coords is byte-identical to BatchCalcABC's -> inherited (deleted here).

    @torch.no_grad()
    def restore_coords(self):
        """Base restore + mark edges dirty (only when a restore actually happened)."""
        had = self._coord_backup is not None
        super().restore_coords()
        if had:
            self._edges_dirty = True

    # =====================================================================
    # neighbour list + single forward (shared by every second-order consumer)
    # =====================================================================
    def _build_edges(self, coord: torch.Tensor):
        """Block-diagonal radius graph from intra-mol candidate pairs (EXACT d^2).
        edge_index is integer (no grad); shifts are zero constants (non-periodic)."""
        device = self.device
        if self.cand_i.numel() == 0:
            return (torch.zeros((2, 0), dtype=torch.int64, device=device),
                    torch.zeros((0, 3), dtype=self.mdtype, device=device))
        rij = coord[self.cand_i] - coord[self.cand_j]
        d2 = (rij * rij).sum(dim=-1)
        keep = d2 <= (self.r_max + 1e-12) ** 2
        ci = self.cand_i[keep]
        cj = self.cand_j[keep]
        src = torch.cat([ci, cj], dim=0)
        dst = torch.cat([cj, ci], dim=0)
        edge_index = torch.stack([src, dst], dim=0)
        shifts = torch.zeros((edge_index.size(1), 3), dtype=self.mdtype, device=device)
        return edge_index, shifts

    def _cached_edges(self):
        """edge_index/shifts for the CURRENT ``self.coord``; rebuilt only if dirty.
        Edges are non-differentiable index sets, so caching across HVPs at a fixed
        geometry is exact (the gradient flows through ``positions``, not indices)."""
        if self._edges_dirty or self._ei_cache is None:
            with torch.no_grad():
                self._ei_cache, self._sh_cache = self._build_edges(self.coord)
            self._edges_dirty = False
        return self._ei_cache, self._sh_cache

    def _model_energy(self, coord_leaf, edge_index, shifts) -> torch.Tensor:
        """Single MACE forward -> per-graph energy (B,) [eV]. Counts one forward."""
        data = {
            "positions": coord_leaf,
            "node_attrs": self.node_attrs,
            "edge_index": edge_index,
            "shifts": shifts,
            "batch": self.mol_idx,
            "ptr": self._ptr,
            "cell": self._cell,
        }
        # training=True keeps the autograd graph; compute_force=False -> we take
        # the gradient ourselves so a SECOND backward (HVP) is available.
        self._n_forward += 1
        out = self.model(data, training=True, compute_force=False)
        return out["energy"].reshape(-1)

    def _forward_with_graph(self):
        """THE single shared forward at ``self.coord`` (cached neighbour list).

        Returns ``(E_eV (B,), g (N,3), coord_leaf)`` where ``g = dE/dx`` is built
        with ``create_graph=True`` so BOTH forces (= -g) AND a second backward
        (HVP / seeded Hessian column) derive from this ONE forward. Cost so far:
        1 model forward + 1 backward.
        """
        coord_leaf = self.coord.detach().to(self.mdtype).requires_grad_(True)
        ei, sh = self._cached_edges()
        E_eV = self._model_energy(coord_leaf, ei, sh)
        g = torch.autograd.grad(E_eV.sum(), coord_leaf, create_graph=True)[0]
        return E_eV, g, coord_leaf

    def _forward_ef_explicit(self, coord: torch.Tensor, need_graph: bool):
        """Forward at an ARBITRARY ``coord`` (edges rebuilt for that geometry).
        Used by the finite-difference paths (perturbed coordinates differ from
        ``self.coord`` so the cache does not apply). Returns (E_eV, F_all, leaf)."""
        coord_leaf = coord.detach().to(self.mdtype).requires_grad_(True)
        ei, sh = self._build_edges(coord_leaf)
        E_eV = self._model_energy(coord_leaf, ei, sh)
        g = torch.autograd.grad(E_eV.sum(), coord_leaf,
                                create_graph=need_graph, retain_graph=need_graph)[0]
        return E_eV.to(self.dtype), (-g), coord_leaf

    def _forward(self, coord: torch.Tensor, need_graph: bool = False):
        """BatchCalcABC contract: ONE forward at an arbitrary ``coord`` ->
        (E_eV (B,), F_all (N,3), leaf|None) in NATIVE eV (the base converts to
        Hartree once via MODEL_ENERGY_UNIT; do NOT scale by EV2HARTREE here).

        Thin adapter over ``_forward_ef_explicit`` (rebuilds edges for the given
        geometry) -- byte-identical forward math; only nulls the leaf when
        need_graph is False, per the contract. MACE's own overrides (get_ef_gpu /
        _efh_fd / _efh_analytic / hvp*) do NOT route through here -- they use the
        cached-neighbour-list / create_graph forwards directly; this exists to
        satisfy the base's generic get_ef_gpu / _efh_fd / _efh_analytic (the
        model-independent parity oracles)."""
        E_eV, F_all, leaf = self._forward_ef_explicit(coord, need_graph)
        return E_eV, F_all, (leaf if need_graph else None)

    def _scatter_forces(self, F_all: torch.Tensor) -> torch.Tensor:
        """(N,3) per-atom -> (B, nmax_dof) padded per structure."""
        B = self._atoms_B
        F_flat = torch.zeros(B * self.nmax_dof, dtype=self.dtype, device=self.device)
        if self.N_atoms > 0:
            base = self._base
            F_flat[base] = F_all[:, 0]
            F_flat[base + 1] = F_all[:, 1]
            F_flat[base + 2] = F_all[:, 2]
        return F_flat.view(B, self.nmax_dof)

    def _gather_pad_to_atoms(self, v_pad: torch.Tensor) -> torch.Tensor:
        """Inverse of _scatter_forces: (B, nmax_dof) -> (N,3) per-atom."""
        v_flat = v_pad.reshape(-1).to(self.device, self.mdtype)
        base = self._base
        return torch.stack([v_flat[base], v_flat[base + 1], v_flat[base + 2]], dim=1)

    # =====================================================================
    def get_ef_gpu(self):
        """Energy + forces. 1 forward + 1 backward (no create_graph)."""
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return (torch.zeros((0,), dtype=dtype, device=device),
                    torch.zeros((0, 0), dtype=dtype, device=device))
        coord_leaf = self.coord.detach().to(self.mdtype).requires_grad_(True)
        ei, sh = self._cached_edges()
        E_eV = self._model_energy(coord_leaf, ei, sh)
        g = torch.autograd.grad(E_eV.sum(), coord_leaf, create_graph=False)[0]
        F_eV = self._scatter_forces((-g).detach().to(dtype))
        return (E_eV.detach().to(dtype) * EV2HARTREE, F_eV * EV2HARTREE)

    def get_efh_gpu(self, movable_masks=None):
        """Energy + forces + per-structure Hessian.

        ``movable_masks`` (mirrors ANIBatchCalc / UMABatchCalc / AIMNet2BatchCalc):
        None = full Hessian (byte-identical oracle); else a per-structure movable
        atom subset (None / bool mask / index list per structure) that restricts
        BOTH the perturbed (seeded) DOFs AND the responding rows to the movable
        subspace -- the EXACT (3m x 3m) block of the constrained PES, enzyme-safe.
        For the autograd path this is a MOVABLE-AUTOGRAD partial Hessian: only the
        3m movable columns are seeded (1 forward + 1 + 3m backward), NOT the FD
        6m-forward path.
        """
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return (torch.zeros((0,), dtype=dtype, device=device),
                    torch.zeros((0, 0), dtype=dtype, device=device),
                    torch.zeros((0, 0, 0), dtype=dtype, device=device),
                    torch.zeros((0,), dtype=torch.int64, device=device))
        # OLD path (default + oracle): 6N central finite-difference Hessian.
        if self.hessian_mode == "numerical":
            return self._efh_fd(movable_masks=movable_masks)
        # OPT-IN autograd double-backward Hessian. Try (optional) vmap first, then
        # the row-loop, then hard-fall-back to the FD oracle on ANY failure.
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
    # partial-Hessian (movable-atom subspace) helpers -- mirror ANIBatchCalc
    # =====================================================================
    # _resolve_movable is byte-identical to BatchCalcABC's -> inherited (deleted here).

    def _movable_tensors(self, movable_masks):
        """(movable_atom (N,) bool, movable_la (B, nmax_a) bool). All-True when
        movable_masks is None -> byte-identical full Hessian."""
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
    # BATCH HVP via double-backward (NO full 3N x 3N Hessian).  1 forward.
    # =====================================================================
    def hvp_batch(self, v_pad: torch.Tensor):
        """Batched Hessian-vector product H@v for all B structures at once.

        Parameters
        ----------
        v_pad : (B, nmax_dof) torch.Tensor -- per-structure padded direction
            (same layout as the padded forces).

        Returns
        -------
        Hn_pad : (B, nmax_dof)  -- (H @ v) per structure [Ha/A^2]
        E_Ha   : (B,)           -- energy [Ha]

        Cost = 1 model forward + 2 backward (energy grad with create_graph, then
        grad of g . v). The dense Hessian is never materialized.
        """
        assert self._prepared, "call prepare() first"
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return (torch.zeros((0, 0), dtype=dtype, device=device),
                    torch.zeros((0,), dtype=dtype, device=device))
        E_eV, g, coord_leaf = self._forward_with_graph()                  # 1 forward
        v_atom = self._gather_pad_to_atoms(v_pad)                         # (N,3)
        Hn_atom = torch.autograd.grad((g * v_atom).sum(), coord_leaf,
                                      retain_graph=False)[0]              # H@v (N,3)
        Hn_pad = self._scatter_forces(Hn_atom.to(dtype)) * EV2HARTREE
        return Hn_pad, E_eV.detach().to(dtype) * EV2HARTREE

    def hvp(self, v: torch.Tensor) -> torch.Tensor:
        """Canonical batched Hessian-vector product ``H @ v`` (Task-2 name).

        Block-diagonal per system, computed by ONE forward + TWO backward of the
        energy graph: ``g = grad(E, x, create_graph=True); Hn = grad((g*v).sum(),
        x)`` -- the dense Hessian is never formed. Thin wrapper over
        ``hvp_batch``; consumed by the BPRFO iterative eigensolver (sibling agent).
        Autograd-HVP ref: Yuan et al., Nat. Commun. 2024, DOI 10.1038/s41467-024-52481-5.

        Parameters
        ----------
        v : (B, nmax_dof)  -- per-structure padded direction (forces layout).

        Returns
        -------
        Hn_pad : (B, nmax_dof)  -- (H @ v) per structure [Ha/A^2].
        """
        Hn_pad, _ = self.hvp_batch(v)
        return Hn_pad

    # =====================================================================
    # single-structure autograd HVP callback for the dimer (dimer.py hvp_fn).
    # =====================================================================
    def make_hvp_fn(self):
        """Return ``hvp_fn(atoms, n_flat) -> (Hn, forces, energy)`` torch tensors (Hartree).

        The dimer moves ``atoms`` between OUTER steps; this closure re-reads
        ``atoms.get_positions()`` each call via ``set_coords_`` (which keeps the
        neighbour-list cache when positions are unchanged -- e.g. the rotation
        inner loop, where only ``n`` changes -> the edges are built ONCE per
        geometry). Requires ``prepare([single_atoms])`` (B == 1).

        Per call (all on this calculator's device/dtype), from ONE forward:
            Hn      : (3N,)  H @ n      [Ha/A^2]   (energy-Hessian . direction)
            forces  : (3N,)  -dE/dx     [Ha/A]
            energy  : scalar E          [Ha]
        """
        assert self._prepared and self._atoms_B == 1, \
            "make_hvp_fn() requires prepare([single_atoms]) with B == 1"
        N = self.N_atoms

        def hvp_fn(atoms: Atoms, n_flat):
            pos = torch.as_tensor(atoms.get_positions(), dtype=self.dtype, device=self.device)
            self.set_coords_(pos)                       # cache-aware (keeps edges if unchanged)
            E_eV, g, coord_leaf = self._forward_with_graph()             # 1 forward
            v = torch.as_tensor(np.asarray(n_flat, dtype=np.float64).reshape(N, 3),
                                dtype=self.mdtype, device=self.device)
            Hn = torch.autograd.grad((g * v).sum(), coord_leaf, retain_graph=False)[0]
            Hn_ha = (Hn.reshape(-1) * EV2HARTREE).to(self.dtype)         # (3N,) Ha/A^2
            F_ha = ((-g).reshape(-1) * EV2HARTREE).to(self.dtype)        # (3N,) Ha/A
            E_ha = (E_eV.sum() * EV2HARTREE).to(self.dtype)              # scalar Ha
            return Hn_ha.detach(), F_ha.detach(), E_ha.detach()

        return hvp_fn

    # =====================================================================
    # seeded block-diagonal analytic Hessian (double-backward) / FD fallback
    # =====================================================================
    def _efh_analytic(self, movable_masks=None, chunk_size: Optional[int] = None):
        """Block-diagonal Hessian by seeded DOUBLE-BACKWARD. 1 forward +
        (1 + 3*m) backward over the movable subspace (m = movable atoms) -- still
        ONE model forward (vs FD's 6m). ``movable_masks`` (mirrors ANIBatchCalc)
        gates BOTH the seeded columns (only movable local DOFs) AND the responding
        rows (``row_scale`` zeroes non-movable rows) -> the EXACT (3m x 3m) block
        of the constrained PES, with non-movable rows/cols left zero. None ->
        byte-identical full Hessian.

        ``F_all = -dE/dx`` carries ``create_graph=True`` (built in
        ``_forward_with_graph``); each Hessian column is the SECOND backward
        ``autograd.grad(F_all, pos, grad_outputs=seed)`` where ``seed`` is the
        block-diagonal unit cotangent for one movable local DOF over all molecules.

        ``chunk_size`` (OPT-IN): stack the movable seed cotangents and evaluate the
        batched VJP with ``torch.vmap(..., chunk_size)`` -- the MACE
        ``compute_hessians_vmap`` pattern (github ACEsuit/mace mace/modules/utils.py;
        loop fallback per MACE issue #488). ``chunk_size=None`` (DEFAULT) keeps the
        byte-identical row-loop. Autograd Hessian ref: Yuan et al., Nat. Commun.
        2024, DOI 10.1038/s41467-024-52481-5. Model MUST be eager (pytorch#91469).
        """
        B, device, dtype = self._atoms_B, self.device, self.dtype
        N, nmax, nmax_a = self.N_atoms, self.nmax_dof, self.Nmax_atoms
        E_eV, g, coord_leaf = self._forward_with_graph()                 # 1 forward
        F_all = -g
        s = self._ptr[:-1]
        n_b = self._n_b
        n_b_list = n_b.tolist()
        F_eV = self._scatter_forces(F_all.to(dtype))
        H_eV = torch.zeros((B, nmax, nmax), dtype=dtype, device=device)
        P = (nmax_a - n_b).to(torch.int64)
        movable_atom, movable_la = self._movable_tensors(movable_masks)
        row_scale = movable_atom[:, None].to(F_all.dtype)
        K = 3 * nmax_a

        def _scatter_col(k, col):
            a_local = k // 3
            for i in movable_la[:, a_local].nonzero(as_tuple=False).flatten().tolist():
                dof = 3 * n_b_list[i]
                if k < dof:
                    H_eV[i, :dof, k] = -col[s[i]:s[i] + n_b_list[i], :].reshape(-1).to(dtype)

        if chunk_size is not None and K > 0:
            # OPT-IN vmap over the movable block-diagonal seed basis (batched VJP).
            seeds = []
            for k in range(K):
                a_local, c = k // 3, k % 3
                go = torch.zeros((N, 3), dtype=F_all.dtype, device=device)
                valid = movable_la[:, a_local]
                if bool(valid.any()):
                    go[s[valid] + a_local, c] = 1.0
                seeds.append(go)
            GO = torch.stack(seeds, dim=0)                               # (K,N,3)

            def _get_vjp(v):
                return torch.autograd.grad(F_all, coord_leaf, grad_outputs=v,
                                           retain_graph=True, create_graph=False)[0]

            COLS = torch.vmap(_get_vjp, in_dims=0, out_dims=0,
                              chunk_size=int(chunk_size))(GO)            # (K,N,3)
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
                go = torch.zeros((N, 3), dtype=F_all.dtype, device=device)
                go[rows, c] = 1.0
                col = torch.autograd.grad(F_all, coord_leaf, grad_outputs=go,
                                          retain_graph=True, create_graph=False)[0]
                col = col * row_scale
                _scatter_col(k, col)
        H_eV = 0.5 * (H_eV + H_eV.transpose(1, 2))
        return (E_eV.detach().to(dtype) * EV2HARTREE, F_eV * EV2HARTREE, H_eV * EV2HARTREE, P)

    def _efh_fd(self, delta: float = 2e-3, movable_masks=None):
        """Numerical Hessian (central FD). 1 + 2*3*m forwards over the movable
        subspace. Baseline + parity oracle. ``movable_masks`` gates the perturbed
        columns + responding rows (None -> full 6N FD Hessian, unchanged)."""
        B, device, dtype = self._atoms_B, self.device, self.dtype
        nmax, nmax_a = self.nmax_dof, self.Nmax_atoms
        E0, F0, _ = self._forward_ef_explicit(self.coord, need_graph=False)
        base_coord = self.coord.clone()
        s = self._ptr[:-1]
        n_b = self._n_b
        n_b_list = n_b.tolist()
        F_eV = self._scatter_forces(F0.detach().to(dtype))
        H_eV = torch.zeros((B, nmax, nmax), dtype=dtype, device=device)
        P = (nmax_a - n_b).to(torch.int64)
        movable_atom, movable_la = self._movable_tensors(movable_masks)
        row_scale = movable_atom[:, None].to(dtype)
        for k in range(3 * nmax_a):
            a_local, c = k // 3, k % 3
            valid = movable_la[:, a_local]
            if not bool(valid.any()):
                continue
            rows = s[valid] + a_local
            cp = base_coord.clone()
            cp[rows, c] += delta
            cm = base_coord.clone()
            cm[rows, c] -= delta
            _, Fp, _ = self._forward_ef_explicit(cp, need_graph=False)
            _, Fm, _ = self._forward_ef_explicit(cm, need_graph=False)
            col = ((-(Fp - Fm) / (2.0 * delta)).to(dtype)) * row_scale
            for i in valid.nonzero(as_tuple=False).flatten().tolist():
                dof = 3 * n_b_list[i]
                if k < dof:
                    H_eV[i, :dof, k] = col[s[i]:s[i] + n_b_list[i], :].reshape(-1)
        self.coord.copy_(base_coord)
        self._edges_dirty = True
        H_eV = 0.5 * (H_eV + H_eV.transpose(1, 2))
        return (E0.detach().to(dtype) * EV2HARTREE, F_eV * EV2HARTREE, H_eV * EV2HARTREE, P)

    # =====================================================================
    # diagnostics
    # =====================================================================
    def isolation_check(self, perturb: float = 0.05) -> float:
        """Returns max cross-molecule force leak (Ha/A). 0.0 for correct local batch."""
        assert self._prepared and self._atoms_B >= 2
        c0 = self.coord.clone()
        _, F0, _ = self._forward_ef_explicit(c0, need_graph=False)
        c1 = c0.clone()
        c1[0, 0] += perturb
        _, F1, _ = self._forward_ef_explicit(c1, need_graph=False)
        leak = 0.0
        for j in range(1, self._atoms_B):
            rows = (self.mol_idx == j)
            leak = max(leak, (F1[rows] - F0[rows]).abs().max().item())
        return leak * EV2HARTREE

    def energy_of(self, mol_index: int) -> float:
        """Per-molecule energy [Ha] (for the perturb-one dE isolation gate)."""
        E_eV, _, _ = self._forward_ef_explicit(self.coord, need_graph=False)
        return float(E_eV[mol_index].item()) * EV2HARTREE
