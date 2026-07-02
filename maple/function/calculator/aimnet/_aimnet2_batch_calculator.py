# -*- coding: utf-8 -*-
"""Batched AIMNet2 (charge-equilibration / NSE) calculator for the MAPLE GPU-batch path.

Retrofitted onto :class:`BatchCalcABC` (batch_calculator_base.py): the coord ops
(step_cart_/set_coords_/backup_coords/restore_coords), the ``(B, nmax_dof)``
pack/scatter, unit conversion, PBC fail-fast, fixed_nmax validation, the movable
subspace resolver and the registry now come from the base. This subclass keeps
ONLY the AIMNet2-specific surface: the jit model, the block-diagonal neighbour
list, the per-molecule energy pooling, the charge/sentinel graph inputs, the
energy-only ``get_e_gpu`` fast path, and the model-specific SEEDED block-diagonal
ANALYTIC Hessian (``get_efh_gpu``) that exploits the block-diagonal graph to obtain
column k of every per-structure Hessian from ONE backward.

Coupling.  The bundled charge-equilibration (NSE) model does a global charge
equilibration, so this is the COUPLED variant (``SUPPORTS_COUPLING = True``;
matches eulerpc's ``_COUPLED_BATCH_CALC_NAMES``). It still runs B>1 batches
mechanically (``BATCHABLE = True``); the cross-reaction-safe decoupled sibling is
``AIMNet2DecoupledBatchCalc`` (mol_idx-segmented graph, SUPPORTS_COUPLING=False).

Hessian.  AIMNet2's Hessian path is exact autograd (seeded double-backward), so
``SUPPORTED_HESSIAN_MODES = ('autograd',)`` and ``get_efh_gpu`` is kept as an
override rather than the base generic ``_efh_analytic`` (the override is the
block-diagonal-seed optimization: 3*Nmax_atoms backward passes for the WHOLE batch
instead of 3*N_total). Reference kept as ``_get_efh_gpu_global_ref`` for parity.

Units.  AIMNet2 returns eV / eV.A^-1; the base's ``_to_hartree`` / ``EV2HARTREE``
is the ONLY unit conversion (this retrofit centralizes the pre-retrofit per-method
``/EH2EV`` into the single base conversion, killing the divide/multiply split).
"""
import torch

from ..batch_calculator_base import (  # noqa: F401
    BatchCalcABC,
    EV2HARTREE,
    register_batch_calculator,
)


def pad_dim0(a: torch.Tensor, value=0) -> torch.Tensor:
    pad_shape = list(a.shape); pad_shape[0] = 1
    pad_row = torch.full(pad_shape, value, dtype=a.dtype, device=a.device)
    return torch.cat([a, pad_row], dim=0)


# -----------------------------------------------------------------------------
# Neighbour list
# -----------------------------------------------------------------------------
# OPT (D-16): block-diagonal neighbour list.
#
# The molecules are contiguous in `coord` (segmented by `mol_idx`), so the only
# pairs that can ever survive `same-molecule` masking live inside per-molecule
# diagonal blocks. The old dense build materialised a full (N, N, 3) diff and
# (N, N) dist2 + an N-wide argsort + a python `for i in range(N)` fill -- O(N^2)
# memory that OOMs at large B (~5 GB / fwd at B=256, n=50).
#
# The new build computes distances only WITHIN per-molecule blocks via a padded
# (B, nmax_b, nmax_b) tensor, so peak memory is bound by B * max(n_b)^2 instead
# of N^2 = (B*n)^2 (a factor-B reduction). The produced `nbmat` is BYTE-IDENTICAL
# to the dense build (same dtype int64, sentinel pad index = N, neighbours sorted
# by (dist2, ascending global index) exactly as the stable N-wide argsort gives,
# and M = global max degree). The dense version is kept as
# `nblist_dense_padded_multi_ref` for parity validation.
# -----------------------------------------------------------------------------
def nblist_dense_padded_multi_ref(coord: torch.Tensor, mol_idx: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Reference (pre-D16) DENSE neighbour list: (N,N,3) diff + N-wide argsort +
    python fill. O(N^2) memory; OOMs at large B. Kept ONLY for parity validation
    of the block build (nblist_block_padded_multi)."""
    device = coord.device
    dtype  = coord.dtype
    N = coord.shape[0]
    if N == 0:
        return torch.full((1, 1), 0, dtype=torch.int64, device=device)

    diff  = coord[:, None, :] - coord[None, :, :]
    dist2 = (diff * diff).sum(dim=-1)
    same  = (mol_idx[:, None] == mol_idx[None, :])
    eye   = torch.eye(N, dtype=torch.bool, device=device)
    mask  = (dist2 <= cutoff * cutoff) & same & (~eye)

    deg = mask.sum(dim=1)
    M   = int(max(int(deg.max().item()), 1))

    big = torch.finfo(dtype).max / 4.0
    sort_key = torch.where(mask, dist2, dist2.new_full(dist2.shape, big))
    order = torch.argsort(sort_key, dim=1, stable=True)

    nbmat = torch.full((N + 1, M), N, dtype=torch.int64, device=device)
    for i in range(N):
        ki = int(deg[i].item())
        if ki > 0:
            k = min(ki, M)
            nbmat[i, :k] = order[i, :k]
    return nbmat


def _block_meta_from_mol_idx(mol_idx: torch.Tensor, N: int):
    """Derive per-molecule block metadata (molecules assumed contiguous).
    Returns (counts, ptr, nmax_b, valid, local2global). Uses a couple of host
    syncs (B, nmax_b) -- only ever called in prepare(), never in the hot loop."""
    device = mol_idx.device
    if N == 0:
        B = 0
        counts       = torch.zeros((0,), dtype=torch.int64, device=device)
        ptr          = torch.zeros((1,), dtype=torch.int64, device=device)
        nmax_b       = 1
        valid        = torch.zeros((0, 1), dtype=torch.bool, device=device)
        local2global = torch.zeros((0, 1), dtype=torch.int64, device=device)
        return counts, ptr, nmax_b, valid, local2global

    B = int(mol_idx[-1].item()) + 1
    counts = torch.bincount(mol_idx, minlength=B).to(torch.int64)        # (B,)
    ptr = torch.zeros(B + 1, dtype=torch.int64, device=device)
    ptr[1:] = torch.cumsum(counts, dim=0)
    nmax_b = int(counts.max().item())
    ar = torch.arange(nmax_b, device=device)
    valid = ar[None, :] < counts[:, None]                               # (B, nmax_b)
    local2global = ptr[:-1][:, None] + ar[None, :]                      # (B, nmax_b)
    local2global = torch.where(valid, local2global, torch.full_like(local2global, N))
    return counts, ptr, nmax_b, valid, local2global


def _nblist_block_core(coord: torch.Tensor, cutoff: float,
                       ptr: torch.Tensor, nmax_b: int,
                       valid: torch.Tensor, local2global: torch.Tensor,
                       N: int) -> torch.Tensor:
    """Block-diagonal neighbour list from precomputed metadata. Memory bound by
    B * nmax_b^2. Output byte-identical to nblist_dense_padded_multi_ref."""
    device = coord.device
    dtype  = coord.dtype
    if N == 0:
        return torch.full((1, 1), 0, dtype=torch.int64, device=device)

    B = ptr.shape[0] - 1

    # gather per-molecule block coords via a zero pad row (invalid local -> N -> 0)
    coord_pad = torch.cat([coord, coord.new_zeros((1, 3))], dim=0)      # (N+1, 3)
    cb = coord_pad[local2global]                                        # (B, nmax_b, 3)

    diff  = cb[:, :, None, :] - cb[:, None, :, :]                       # (B, nmax_b, nmax_b, 3)
    dist2 = (diff * diff).sum(dim=-1)                                   # (B, nmax_b, nmax_b)

    pair_valid = valid[:, :, None] & valid[:, None, :]
    eye = torch.eye(nmax_b, dtype=torch.bool, device=device)
    mask = (dist2 <= cutoff * cutoff) & pair_valid & (~eye[None, :, :])

    deg = mask.sum(dim=-1)                                              # (B, nmax_b)
    M   = int(max(int(deg.max().item()), 1))

    big = torch.finfo(dtype).max / 4.0
    sort_key = torch.where(mask, dist2, dist2.new_full(dist2.shape, big))
    order = torch.argsort(sort_key, dim=-1, stable=True)               # (B, nmax_b, nmax_b) local idx
    order_M = order[:, :, :M]                                          # (B, nmax_b, M)

    nb_global = ptr[:-1][:, None, None] + order_M                      # local -> global
    keep = torch.arange(M, device=device)[None, None, :] < deg[:, :, None]
    nb_global = torch.where(keep, nb_global, torch.full_like(nb_global, N))

    nbmat = torch.full((N + 1, M), N, dtype=torch.int64, device=device)
    rows = local2global.reshape(-1)                                    # (B*nmax_b,); invalid -> N
    nbmat[rows] = nb_global.reshape(B * nmax_b, M)                     # invalid rows write all-N to row N
    return nbmat


def nblist_block_padded_multi(coord: torch.Tensor, mol_idx: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Self-contained block-diagonal neighbour list (derives metadata from
    mol_idx). Byte-identical to nblist_dense_padded_multi_ref."""
    N = coord.shape[0]
    _, ptr, nmax_b, valid, local2global = _block_meta_from_mol_idx(mol_idx, N)
    return _nblist_block_core(coord, cutoff, ptr, nmax_b, valid, local2global, N)


@register_batch_calculator
class AIMNet2BatchCalc(BatchCalcABC):
    """AIMNet2 (charge-equilibration / NSE) batch calculator.

    One ``prepare()`` (BatchCalcABC's) fixes B molecules' topology; thereafter
    ``get_ef_gpu`` (base generic forward->pad->convert) and ``get_efh_gpu`` (the
    model-specific seeded block-diagonal analytic Hessian kept below) run batched
    forwards. Coordinates are the base's f64 master ``coord`` tensor.
    """

    # ---- capability protocol (BatchCalcABC declarative attrs) --------------
    MODEL_NAMES = ("aimnet2",)
    MODEL_ENERGY_UNIT = "eV"                 # AIMNet2 returns eV; base -> Hartree once
    MODEL_DTYPE = torch.float64              # jit forward runs at the f64 master dtype
    SUPPORTS_PBC = False                     # no-PBC molecular wrapper
    SUPPORTED_HESSIAN_MODES = ("autograd",)  # seeded block-diagonal double-backward
    HAS_HVP = False
    SUPPORTS_COUPLING = True                 # global charge-eq (NSE) couples molecules
    BATCHABLE = True                         # runs B>1 (block nblist); not default-raise

    def __init__(self, model_path: str, device: str = "cuda", cutoff: float = 5.0,
                 dtype: torch.dtype = torch.float64):
        # base.__init__ resolves device (torch.device via _resolve_device) + dtype
        # and inits the generic prepared-state (_prepared/_atoms_B/_ptr/numbers/
        # mol_idx/_local_atom/_n_b/_cols/coord/N_atoms/Nmax_atoms/nmax_dof/
        # _coord_backup); load the jit model onto the resolved device afterward.
        super().__init__(device, dtype)
        self.model = torch.jit.load(model_path, map_location=self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.cutoff = float(cutoff)

        # ---- model-specific prepared-state ONLY (set in prepare/_build_topology).
        #      Do NOT re-init the generic layout state base.__init__ already set.
        self.sentinel_mol = 0
        self.charge       = None

        # block-nblist metadata + flat scatter maps (set in _build_topology).
        self._blk_ptr        = None   # (B+1,)
        self._blk_nmax       = 1
        self._blk_valid      = None   # (B, nmax_b) bool
        self._blk_local2g    = None   # (B, nmax_b) int64
        self._cart_idx_flat  = None   # (3*N,) flat map (== base _cols numerically)
        self._atom_localidx  = None   # (N,) local atom index (== base _local_atom)

    # -------------------------------------------------------- topology hook
    def _build_topology(self, atoms_list):
        """Cache the AIMNet2-specific batch topology.

        Called at the END of ``BatchCalcABC.prepare()`` -- which already built
        ptr/numbers/mol_idx/_local_atom/_n_b/_cols/coord/nmax_dof (with the shared
        fixed_nmax validation + PBC fail-fast) and reset _coord_backup. Here we set
        the model's charge/sentinel graph inputs and the block-diagonal neighbour
        list metadata + the flat scatter / local-atom index maps (topology fixed
        across step/get_* calls; built ONCE to keep the hot path sync-free).
        """
        self.sentinel_mol = (int(self.mol_idx.max().item()) + 1) if self.N_atoms > 0 else 0
        self.charge       = torch.zeros(self._atoms_B + 1, dtype=self.dtype, device=self.device)
        self._build_static_maps()

    def _build_static_maps(self):
        """Build all coordinate/force/nblist index maps that depend only on the
        (fixed) topology. Called once per prepare() (via _build_topology)."""
        device = self.device
        N      = self.N_atoms
        B      = self._atoms_B

        # block-nblist metadata
        (_, self._blk_ptr, self._blk_nmax,
         self._blk_valid, self._blk_local2g) = _block_meta_from_mol_idx(self.mol_idx, N)

        if N == 0:
            self._cart_idx_flat = torch.zeros((0,), dtype=torch.int64, device=device)
            self._atom_localidx = torch.zeros((0,), dtype=torch.int64, device=device)
            return

        # local atom index of each global atom: g -> a (= g - ptr[mol_idx[g]])
        a_local = torch.arange(N, device=device) - self._ptr[:-1].to(torch.int64)[self.mol_idx]
        self._atom_localidx = a_local                                   # (N,)

        # flat map: the 3 components of global atom g <-> (mol_idx[g], 3*a_local) in
        # a (B, nmax_dof) padded buffer flattened. Drives the get_ef_/get_efh_ force
        # scatter; numerically identical to the base's _cols.
        nmax_dof = self.nmax_dof
        base = self.mol_idx.to(torch.int64) * nmax_dof + 3 * a_local    # (N,) start col in flat buf
        cart_idx = base[:, None] + torch.arange(3, device=device)[None, :]   # (N, 3)
        self._cart_idx_flat = cart_idx.reshape(-1)                      # (3*N,)

    # step_cart_ / set_coords_ / backup_coords / restore_coords / _resolve_movable
    # are byte-identical to BatchCalcABC's (_cart_idx_flat == _cols, _n_b builds the
    # same n_b) -> inherited (deleted here).

    # -------------------------------------------------------------------------
    # internal helpers
    # -------------------------------------------------------------------------
    def _nblist(self, coord: torch.Tensor) -> torch.Tensor:
        """Fast block-diagonal neighbour list using precomputed metadata."""
        return _nblist_block_core(coord, self.cutoff, self._blk_ptr, self._blk_nmax,
                                  self._blk_valid, self._blk_local2g, self.N_atoms)

    def _reduce_energy(self, out) -> torch.Tensor:
        """Reduce model output to per-structure energy (B,)."""
        B = self._atoms_B
        N = self.N_atoms
        e_vec = out["energy"].to(self.dtype).reshape(-1)
        if e_vec.numel() == N + 1 or e_vec.numel() == B + 1:
            e_vec = e_vec[:-1]
        if e_vec.numel() == B:
            return e_vec
        elif e_vec.numel() == N:
            return torch.bincount(self.mol_idx, weights=e_vec, minlength=B).to(self.dtype)
        else:
            raise RuntimeError(f"Unexpected energy shape {tuple(e_vec.shape)}")

    def _scatter_forces(self, F_all_eV: torch.Tensor) -> torch.Tensor:
        """Scatter per-atom (N,3) forces into the padded (B, nmax_dof) buffer
        using the precomputed flat index map. Vectorized (no per-b loop).
        Numerically identical to the base's _pad_forces."""
        B    = self._atoms_B
        nmax = self.nmax_dof
        F_flat = torch.zeros(B * nmax, dtype=self.dtype, device=self.device)
        if self.N_atoms > 0:
            F_flat[self._cart_idx_flat] = F_all_eV.reshape(-1)
        return F_flat.view(B, nmax)

    def _movable_atom_mask(self, mov) -> torch.Tensor:
        """(N,) bool: global atom is movable. ``mov`` = _resolve_movable output.
        All-True when movable_masks was None (-> full Hessian, byte-identical)."""
        N = self.N_atoms
        mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        ptr = self._ptr.tolist()
        for i in range(self._atoms_B):
            base = ptr[i]
            for a in mov[i]:
                mask[base + a] = True
        return mask

    # -------------------------------------------------------------------------
    # forward -- BatchCalcABC contract: (E (B,), F_all (N,3), leaf|None) NATIVE eV
    # -------------------------------------------------------------------------
    def _forward(self, coord: torch.Tensor, need_graph: bool = False):
        """ONE batched forward -> (E_eV (B,), F_all_eV (N,3), leaf|None) in NATIVE
        eV (the base converts once via MODEL_ENERGY_UNIT; do NOT scale by EV2HARTREE
        here). AIMNet2 outputs energy; forces come from an autograd backward of the
        per-structure energy sum (create_graph=need_graph).

        Forward math is byte-identical to the pre-retrofit ``_forward_energy_forces_``;
        the ONLY change is returning ``leaf=None`` when need_graph is False (no caller
        uses the leaf in that case -- get_ef_gpu / the base FD path discard it).
        """
        assert self._prepared, "call prepare() first"
        device, dtype = self.device, self.dtype

        coord_leaf = coord.detach().to(device=device, dtype=dtype).requires_grad_(True)
        nbmat = self._nblist(coord_leaf)

        data = {
            "coord":    pad_dim0(coord_leaf, 0.0),
            "numbers":  pad_dim0(self.numbers, 0).to(torch.int64),
            "charge":   self.charge,
            "mol_idx":  pad_dim0(self.mol_idx, self.sentinel_mol).to(torch.int64),
            "nbmat":    nbmat,
            "nbmat_lr": nbmat,
        }

        with torch.jit.optimized_execution(False):
            out = self.model(data)

        E_eV = self._reduce_energy(out)

        grad = torch.autograd.grad(E_eV.sum(), coord_leaf,
                                   create_graph=need_graph, retain_graph=need_graph)[0]
        F_all_eV = -grad

        return E_eV, F_all_eV, (coord_leaf if need_graph else None)

    def _forward_energy_(self, c: torch.Tensor):
        """Energy-ONLY forward (no autograd graph / no force backward)."""
        assert self._prepared, "call prepare() first"
        device, dtype = self.device, self.dtype
        coord = c.detach().to(device=device, dtype=dtype)
        with torch.no_grad():
            nbmat = self._nblist(coord)
            data = {
                "coord":    pad_dim0(coord, 0.0),
                "numbers":  pad_dim0(self.numbers, 0).to(torch.int64),
                "charge":   self.charge,
                "mol_idx":  pad_dim0(self.mol_idx, self.sentinel_mol).to(torch.int64),
                "nbmat":    nbmat,
                "nbmat_lr": nbmat,
            }
            with torch.jit.optimized_execution(False):
                out = self.model(data)
            E_eV = self._reduce_energy(out)
        return E_eV

    # -------------------------------------------------------------------------
    # public results
    # -------------------------------------------------------------------------
    def get_e_gpu(self):
        """Energy-only batched single point (skips the force backward). Same
        per-structure energy as get_ef_gpu(); for SP / inner trial loops. Not part
        of BatchCalcABC (E+F / E+F+H only); model-specific fast path kept here.
        Converts eV->Hartree via the base's centralized EV2HARTREE."""
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return torch.zeros((0,), dtype=dtype, device=device)
        E_eV = self._forward_energy_(self.coord)
        return E_eV * EV2HARTREE

    # get_ef_gpu is the generic forward->pad->convert pattern -> provided by
    # BatchCalcABC (calls self._forward, self._pad_forces, self._to_hartree).

    def get_efh_gpu(self, movable_masks=None, mode=None, delta: float = 2e-3,
                    chunk_size=None, base_ef=None):
        """Energy + forces + per-structure Hessian (batched, padded to nmax).

        MODEL-SPECIFIC OPTIMIZATION -- kept as an override of the base generic
        ``_efh_analytic`` (do NOT delete). Seeded block-diagonal analytic Hessian:
        column k (local DOF) of ALL B per-structure Hessians is obtained from ONE
        backward by seeding a grad_outputs one-hot at local DOF k of every structure
        simultaneously; the inter-molecular blocks are zero (block-diagonal nblist +
        per-mol energy pooling), so each structure recovers its own column. Backward
        passes = 3*Nmax_atoms (NOT 3*N_total) -> ~B-fold fewer, and NO global
        (3*N_total)^2 matrix is ever formed. Reference (slow, global) kept as
        _get_efh_gpu_global_ref for parity validation.

        Analytic (autograd) is AIMNet2's ONLY Hessian path
        (SUPPORTED_HESSIAN_MODES=('autograd',)); ``mode`` / ``delta`` / ``chunk_size``
        / ``base_ef`` are accepted for base-contract + dispatcher call-compatibility
        and IGNORED -- the analytic path reuses its single need_graph=True forward as
        the returned gradient, so there is no separate base forward to skip.

        ``movable_masks`` (base ``_resolve_movable``): None = full Hessian
        (byte-identical oracle). A per-structure spec restricts the perturbed/
        responding DOFs to a movable-atom subspace: only movable atoms are SEEDED (so
        frozen columns stay zero) and only movable RESPONSE rows are scattered.
        Frozen atoms still exert forces (they are in every forward), so the returned
        block is the EXACT (3k x 3k) second-derivative block of the FixAtoms-
        constrained PES.
        """
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return (torch.zeros((0,), dtype=dtype, device=device),
                    torch.zeros((0, 0), dtype=dtype, device=device),
                    torch.zeros((0, 0, 0), dtype=dtype, device=device),
                    torch.zeros((0,), dtype=torch.int64, device=device))

        E_eV, F_all_eV, coord_leaf = self._forward(self.coord, need_graph=True)

        N      = self.N_atoms
        nmax   = self.nmax_dof
        nmax_a = self.Nmax_atoms
        s = self._ptr[:-1]; t = self._ptr[1:]
        n_b = (t - s)                                   # (B,) atom count per structure
        P   = (nmax_a - n_b).to(torch.int64)

        # forces scatter (padded) -- vectorized
        F_eV = self._scatter_forces(F_all_eV)

        H_eV = torch.zeros((B, nmax, nmax), dtype=dtype, device=device)
        local_atom = self._atom_localidx                # (N,) local atom index per global atom

        # partial-Hessian movable subspace (None -> all atoms -> byte-identical full).
        mov = self._resolve_movable(movable_masks)      # base's resolver (inherited)
        movable_atom = self._movable_atom_mask(mov)     # (N,) bool; all-True when None
        row_scale = movable_atom[:, None].to(dtype)     # (N,1) 0/1 row gate (1.0 when None)

        # seeded block-diagonal Hessian: 3*Nmax_atoms backward passes for the WHOLE batch
        for k in range(3 * nmax_a):
            a_local, comp = k // 3, k % 3
            # one-hot seed: component `comp` set on every MOVABLE global atom whose local
            # index == a_local (== exactly the structures owning movable local atom a_local).
            # When movable_masks is None this is identical to (local_atom == a_local).
            seed = (local_atom == a_local) & movable_atom
            if not bool(seed.any()):
                continue                                # frozen/padding column stays zero
            go = torch.zeros((N, 3), dtype=dtype, device=device)
            go[:, comp] = seed.to(dtype)
            col = torch.autograd.grad(
                F_all_eV, coord_leaf, grad_outputs=go,
                retain_graph=True, create_graph=False)[0]   # (N,3); Hessian col = -col (H=-dF/dx)
            col = col * row_scale                        # zero frozen response rows (no-op when None)
            # scatter -col (N,3) into padded column k for ALL structures at once.
            # invalid structures (k >= dof) write 0 into a padding column -> harmless.
            H_eV[:, :, k] = -self._scatter_forces(col)

        H_eV = 0.5 * (H_eV + H_eV.transpose(1, 2))      # symmetrize per structure

        return (E_eV * EV2HARTREE,
                F_eV * EV2HARTREE,
                H_eV * EV2HARTREE,
                P)

    def _get_efh_gpu_global_ref(self):
        """Reference (pre-D15) global-matrix Hessian: 3*N_total serial backward +
        dense (3*N_total)^2 stack. Kept ONLY for parity validation of get_efh_gpu;
        OOMs at large B (do not use in production)."""
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return (torch.zeros((0,), dtype=dtype, device=device),
                    torch.zeros((0, 0), dtype=dtype, device=device),
                    torch.zeros((0, 0, 0), dtype=dtype, device=device),
                    torch.zeros((0,), dtype=torch.int64, device=device))

        E_eV, F_all_eV, coord_leaf = self._forward(self.coord, need_graph=True)

        f_flat = F_all_eV.reshape(-1)
        cols = []
        for k in range(f_flat.numel()):
            g2 = torch.autograd.grad(f_flat[k], coord_leaf,
                                     retain_graph=True, create_graph=False)[0]
            cols.append(g2.reshape(-1))
        H_global_eV = -torch.stack(cols, dim=1)
        H_global_eV = 0.5 * (H_global_eV + H_global_eV.transpose(0, 1))

        nmax = self.nmax_dof
        F_eV = torch.zeros((B, nmax), dtype=dtype, device=device)
        H_eV = torch.zeros((B, nmax, nmax), dtype=dtype, device=device)
        P    = torch.empty((B,), dtype=torch.int64, device=device)

        s = self._ptr[:-1]; t = self._ptr[1:]
        for i in range(B):
            ni  = int((t[i] - s[i]).item())
            dof = 3 * ni
            P[i] = self.Nmax_atoms - ni
            if dof > 0:
                F_eV[i, :dof]       = F_all_eV[s[i]:t[i], :].reshape(-1)
                H_eV[i, :dof, :dof] = H_global_eV[3*s[i]:3*t[i], 3*s[i]:3*t[i]]

        return (E_eV * EV2HARTREE,
                F_eV * EV2HARTREE,
                H_eV * EV2HARTREE,
                P)
