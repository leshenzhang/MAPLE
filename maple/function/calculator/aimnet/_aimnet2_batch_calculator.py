# -*- coding: utf-8 -*-
import torch
from typing import List
from ase import Atoms
import numpy as np

EH2EV = 27.211386245988


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


def _ptr_from_atoms(atoms_list: List[Atoms], device) -> torch.Tensor:
    ptr = [0]
    for at in atoms_list:
        ptr.append(ptr[-1] + len(at))
    return torch.tensor(ptr, dtype=torch.long, device=device)


class AIMNet2BatchCalc:
    """
    AIMNet2 batch calculator.
    """

    def __init__(self, model_path: str, device: str = "cuda", cutoff: float = 5.0, dtype: torch.dtype = torch.float64):
        self.device = torch.device(device)
        self.dtype  = dtype
        self.model  = torch.jit.load(model_path, map_location=self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.cutoff = float(cutoff)

        # prepare-related internal buffers
        self._prepared    = False
        self._atoms_B     = 0
        self._ptr         = None
        self.numbers      = None
        self.mol_idx      = None
        self.coord        = None
        self.N_atoms      = 0
        self.Nmax_atoms   = 0
        self.nmax_dof     = 0   # <<< will be overridden if fixed_nmax is provided
        self.sentinel_mol = 0
        self.charge       = None

        self._coord_backup = None

        # precomputed (set in prepare): block-nblist metadata + flat scatter maps
        self._blk_ptr        = None   # (B+1,)
        self._blk_nmax       = 1
        self._blk_valid      = None   # (B, nmax_b) bool
        self._blk_local2g    = None   # (B, nmax_b) int64
        self._cart_idx_flat  = None   # (3*N,) flat map: real atom DOF -> (B, nmax_dof) flat
        self._atom_localidx  = None   # (N,) local atom index of each global atom

    # -------------------------------------------------------------------------
    # prepare() modified to accept fixed_nmax
    # -------------------------------------------------------------------------
    def prepare(self, atoms_list: List[Atoms], fixed_nmax: int = None):
        """
        Prepare topology & initial coordinates.
        If fixed_nmax is provided, we override the internal nmax_dof so that
        PRFO and calculator share the same padded DOF size.

        This is crucial for compatibility with batch-PRFO, where PRFO wants
        a fixed padded size (self._nmax) across all iterations.
        """
        device, dtype = self.device, self.dtype
        self._atoms_B = len(atoms_list)
        self._ptr     = _ptr_from_atoms(atoms_list, device)

        nums, mids = [], []
        for i, at in enumerate(atoms_list):
            Z = torch.tensor(at.get_atomic_numbers(), dtype=torch.int64, device=device)
            n = Z.shape[0]
            nums.append(Z)
            mids.append(torch.full((n,), i, dtype=torch.int64, device=device))

        self.numbers = torch.cat(nums, dim=0) if nums else torch.zeros((0,), dtype=torch.int64, device=device)
        self.mol_idx = torch.cat(mids, dim=0) if mids else torch.zeros((0,), dtype=torch.int64, device=device)

        self.N_atoms     = int(self.numbers.numel())
        self.Nmax_atoms  = int(max((len(at) for at in atoms_list), default=0))

        # ---------------------------- MODIFICATION ----------------------------
        # If PRFO supplies a fixed_nmax (padded 3*Nmax from first iteration),
        # we MUST adopt that dimension for the calculator too.
        #
        # Otherwise step_cart_() will fail with shape mismatch: PRFO passes
        # (B, fixed_nmax) but calculator expects (B, 3*Nmax_atoms_current).
        # ----------------------------------------------------------------------
        if fixed_nmax is None:
            # normal behavior (first prepare call)
            self.nmax_dof = 3 * self.Nmax_atoms
        else:
            # PRFO-defined padded dimension
            self.nmax_dof = int(fixed_nmax)
        # ----------------------------------------------------------------------

        if self.N_atoms > 0:
            pos_list = [torch.tensor(at.get_positions(), dtype=dtype) for at in atoms_list]
            coord0   = torch.cat(pos_list, dim=0)
        else:
            coord0   = torch.zeros((0, 3), dtype=dtype)

        self.coord = coord0.to(device, non_blocking=True).contiguous()

        self.sentinel_mol = (int(self.mol_idx.max().item()) + 1) if self.N_atoms > 0 else 0
        self.charge       = torch.zeros(self._atoms_B + 1, dtype=dtype, device=device)

        # -------------------- precompute static index maps --------------------
        # (topology fixed across step/get_* calls; build ONCE here to remove the
        #  per-b python .item() loops & host-device syncs from the hot path)
        self._build_static_maps()

        self._coord_backup = None
        self._prepared     = True

    def _build_static_maps(self):
        """Build all coordinate/force/nblist index maps that depend only on the
        (fixed) topology. Called once per prepare()."""
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
        # a (B, nmax_dof) padded buffer flattened. Same map drives step_cart_ gather
        # AND the get_ef_/get_efh_ force scatter (reverse direction).
        nmax_dof = self.nmax_dof
        base = self.mol_idx.to(torch.int64) * nmax_dof + 3 * a_local    # (N,) start col in flat buf
        cart_idx = base[:, None] + torch.arange(3, device=device)[None, :]   # (N, 3)
        self._cart_idx_flat = cart_idx.reshape(-1)                      # (3*N,)

    # -------------------------------------------------------------------------
    # coordinate update
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def step_cart_(self, s_cart: torch.Tensor):
        """
        s_cart: (B, nmax_dof). MUST match self.nmax_dof (PRFO padded).
        """
        assert self._prepared, "call prepare() first"

        B = self._atoms_B

        # ---------------------------- MODIFICATION ----------------------------
        # PRFO enforces a fixed padded DOF; here we enforce the same.
        # ----------------------------------------------------------------------
        assert s_cart.shape == (B, self.nmax_dof), \
            f"step_cart_ expects (B,{self.nmax_dof}), got {tuple(s_cart.shape)}"
        # ----------------------------------------------------------------------

        if self.N_atoms == 0:
            return
        s_cart = s_cart.to(self.device, dtype=self.dtype)
        # vectorized: gather each real atom's 3 DOFs from the padded buffer -> (N,3)
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
        using the precomputed flat index map. Vectorized (no per-b loop)."""
        B    = self._atoms_B
        nmax = self.nmax_dof
        F_flat = torch.zeros(B * nmax, dtype=self.dtype, device=self.device)
        if self.N_atoms > 0:
            F_flat[self._cart_idx_flat] = F_all_eV.reshape(-1)
        return F_flat.view(B, nmax)

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    def _forward_energy_forces_(self, c: torch.Tensor, need_graph: bool):
        assert self._prepared, "call prepare() first"
        device, dtype = self.device, self.dtype

        coord_leaf = c.detach().to(device=device, dtype=dtype).requires_grad_(True)
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

        return E_eV, F_all_eV, coord_leaf

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
        per-structure energy as get_ef_gpu(); for SP / inner trial loops."""
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return torch.zeros((0,), dtype=dtype, device=device)
        E_eV = self._forward_energy_(self.coord)
        return E_eV / EH2EV

    def get_ef_gpu(self):
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return (torch.zeros((0,), dtype=dtype, device=device),
                    torch.zeros((0, 0), dtype=dtype, device=device))

        E_eV, F_all_eV, _ = self._forward_energy_forces_(self.coord, need_graph=False)

        F_eV = self._scatter_forces(F_all_eV)
        return E_eV / EH2EV, F_eV / EH2EV

    def get_efh_gpu(self):
        """Energy + forces + per-structure Hessian (batched, padded to nmax).

        OPTIMIZED (D-15): seeded block-diagonal analytic Hessian. Column k (local DOF)
        of ALL B per-structure Hessians is obtained from ONE backward by seeding a
        grad_outputs one-hot at local DOF k of every structure simultaneously; the
        inter-molecular blocks are zero (block-diagonal nblist + per-mol energy pooling),
        so each structure recovers its own column. Backward passes = 3*Nmax_atoms (NOT
        3*N_total) -> ~B-fold fewer, and NO global (3*N_total)^2 matrix is ever formed.
        Reference (slow, global) kept as _get_efh_gpu_global_ref for parity validation.

        OPT (D-16): the per-k python scatter loop (.nonzero().tolist()) and the per-b
        force scatter are replaced by sync-free vectorized scatters; the seed one-hot is
        built from the precomputed local-atom-index map. Results stay byte-identical to
        _get_efh_gpu_global_ref.
        """
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return (torch.zeros((0,), dtype=dtype, device=device),
                    torch.zeros((0, 0), dtype=dtype, device=device),
                    torch.zeros((0, 0, 0), dtype=dtype, device=device),
                    torch.zeros((0,), dtype=torch.int64, device=device))

        E_eV, F_all_eV, coord_leaf = self._forward_energy_forces_(self.coord, need_graph=True)

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

        # seeded block-diagonal Hessian: 3*Nmax_atoms backward passes for the WHOLE batch
        for k in range(3 * nmax_a):
            a_local, comp = k // 3, k % 3
            # one-hot seed: component `comp` set on every global atom whose local
            # index == a_local (== exactly the structures owning local atom a_local)
            go = torch.zeros((N, 3), dtype=dtype, device=device)
            go[:, comp] = (local_atom == a_local).to(dtype)
            col = torch.autograd.grad(
                F_all_eV, coord_leaf, grad_outputs=go,
                retain_graph=True, create_graph=False)[0]   # (N,3); Hessian col = -col (H=-dF/dx)
            # scatter -col (N,3) into padded column k for ALL structures at once.
            # invalid structures (k >= dof) write 0 into a padding column -> harmless.
            H_eV[:, :, k] = -self._scatter_forces(col)

        H_eV = 0.5 * (H_eV + H_eV.transpose(1, 2))      # symmetrize per structure

        return (E_eV / EH2EV,
                F_eV / EH2EV,
                H_eV / EH2EV,
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

        E_eV, F_all_eV, coord_leaf = self._forward_energy_forces_(self.coord, need_graph=True)

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

        return (E_eV / EH2EV,
                F_eV / EH2EV,
                H_eV / EH2EV,
                P)
