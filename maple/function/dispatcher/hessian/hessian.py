import torch
from torch import Tensor
from typing import Optional

def _get_derivatives_not_none(x: Tensor, y: Tensor, retain_graph: Optional[bool] = None, create_graph: bool = False) -> Tensor:
    ret = torch.autograd.grad([y.sum()], [x], retain_graph=retain_graph, create_graph=create_graph)[0]
    assert ret is not None
    return ret


def hessian(coordinates: Tensor, energies: Optional[Tensor] = None, forces: Optional[Tensor] = None) -> Tensor:
    """Compute analytical hessian from the energy graph or force graph.

    Arguments:
        coordinates (:class:`torch.Tensor`): Tensor of shape `(molecules, atoms, 3)`
        energies (:class:`torch.Tensor`): Tensor of shape `(molecules,)`, if specified,
            then `forces` must be `None`. This energies must be computed from
            `coordinates` in a graph.
        forces (:class:`torch.Tensor`): Tensor of shape `(molecules, atoms, 3)`, if specified,
            then `energies` must be `None`. This forces must be computed from
            `coordinates` in a graph.

    Returns:
        :class:`torch.Tensor`: Tensor of shape `(molecules, 3A, 3A)` where A is the number of
        atoms in each molecule
    """
    if energies is None and forces is None:
        raise ValueError('Energies or forces must be specified')
    if energies is not None and forces is not None:
        raise ValueError('Energies or forces can not be specified at the same time')
    if forces is None:
        assert energies is not None
        forces = -_get_derivatives_not_none(coordinates, energies, create_graph=True)
    flattened_force = forces.flatten(start_dim=1)
    force_components = flattened_force.unbind(dim=1)
    return -torch.stack([
        _get_derivatives_not_none(coordinates, f, retain_graph=True).flatten(start_dim=1)
        for f in force_components
    ], dim=1)


# =============================================================================
# Batched + partial NUMERICAL (finite-difference) Hessian
# =============================================================================
# Central-difference Cartesian Hessian for MANY structures concurrently on GPU,
# and over only the MOVABLE (flexible) atom subset for large/enzyme systems.
#
#   H = -(F(x+delta) - F(x-delta)) / (2*delta)     (Ha/A^2, symmetrized)
#
# Honest framing: the movable-subspace Hessian is the EXACT Hessian of a
# constrained (ASE FixAtoms) optimization over the movable DOFs -- frozen atoms
# remain at their base positions in every perturbed replica and still exert
# forces on the movable atoms. It is NOT an approximation of the full-system
# Hessian; it is the exact second-derivative block of the constrained PES.
#
# Strategy (vs. the serial 2*3*n_movable force-eval loop in
# mace/_macepol_calculator.py::_get_hessian_numerical and
# uma/_uma_calculator.py::get_hessian):
#   * enumerate every (structure b, movable atom j, axis, sign) perturbation as
#     an INDEPENDENT replica molecule (unique batch entry / mol_idx -- never fuse
#     perturbed copies into one molecule; the calculator's block-diagonal nblist
#     keeps molecules byte-isolated).
#   * STACK replicas across structures into chunked batches under a memory budget
#     and evaluate forces in a FEW big calc.get_ef_gpu() calls.
#   * assemble per-structure (3*n_movable, 3*n_movable) Hessians; all FD
#     subtraction/division done in float64 even if the model forward is f32.
#
# delta ~ 2e-3 A << cutoff (5 A) so the base connectivity is reused. (The
# AIMNet2 batch calculator rebuilds its block-diagonal neighbour list per forward
# from positions; there is no cross-forward nblist cache to bypass, and adding
# one would require editing the calculator -- out of scope. The block-diagonal
# build is cheap, and serial reference + batched use the SAME forward path, so
# parity holds.)
#
# The calculator force buffer get_ef_gpu() -> (B, nmax_dof) is per-molecule
# padded: row b = molecule b, atom a (local index) -> columns 3*a, 3*a+1, 3*a+2.
# Each replica is a single molecule, so a movable atom's global index == local
# index and its forces sit at columns 3*atom + {0,1,2}.
# =============================================================================

import numpy as np
from typing import List, Optional, Sequence, Union


def movable_from_atoms(atoms) -> List[int]:
    """Movable atom indices = atoms NOT frozen by any ASE FixAtoms constraint.

    Identical rule to the serial references in mace/_macepol_calculator.py and
    uma/_uma_calculator.py.
    """
    from ase.constraints import FixAtoms
    n = len(atoms)
    fixed = set()
    for c in getattr(atoms, "constraints", None) or []:
        if isinstance(c, FixAtoms):
            fixed.update(int(i) for i in c.get_indices())
    return [i for i in range(n) if i not in fixed]


def _resolve_movable(atoms_list, movable_masks) -> List[List[int]]:
    """Per-structure movable index lists.

    movable_masks: None (derive from FixAtoms per structure), or a sequence with
    one entry per structure where each entry is None (all atoms movable), a bool
    mask of length n_atoms, or an explicit list/array of movable atom indices.
    """
    out = []
    for b, at in enumerate(atoms_list):
        if movable_masks is None:
            out.append(movable_from_atoms(at))
            continue
        m = movable_masks[b]
        if m is None:
            out.append(list(range(len(at))))
            continue
        m_arr = np.asarray(m)
        if m_arr.dtype == bool:
            out.append([int(i) for i in np.nonzero(m_arr)[0]])
        else:
            out.append([int(i) for i in m_arr.reshape(-1)])
    return out


def _movable_cols(movable: Sequence[int]) -> np.ndarray:
    """Flat column indices into a per-molecule (nmax_dof,) force row for the
    movable DOFs, in movable order: [3a0,3a0+1,3a0+2, 3a1,...]."""
    if len(movable) == 0:
        return np.zeros((0,), dtype=np.int64)
    a = np.asarray(movable, dtype=np.int64)
    return (3 * a[:, None] + np.arange(3, dtype=np.int64)[None, :]).reshape(-1)


def serial_fd_hessian(calc, atoms, movable: Optional[Sequence[int]] = None,
                      delta: float = 2e-3) -> "torch.Tensor":
    """Serial central-FD Hessian over the movable subspace -- ONE perturbation per
    calc.get_ef_gpu() call (single-molecule prepare). Reference for batched parity:
    uses the SAME calculator forward path, so any diff isolates the batching effect.

    Returns (3*m, 3*m) float64 torch tensor in Ha/A^2 (m = len(movable)).
    Frozen atoms stay at base positions in every perturbed replica (exert forces).
    """
    from ase import Atoms
    if movable is None:
        movable = movable_from_atoms(atoms)
    m = len(movable)
    dof = 3 * m
    device = calc.device
    H = torch.zeros((dof, dof), dtype=torch.float64, device=device)
    if m == 0:
        return H

    Z = atoms.get_atomic_numbers()
    pos0 = atoms.get_positions().astype(np.float64)
    info = dict(getattr(atoms, "info", {}) or {})
    cols = _movable_cols(movable)

    def force_movable(pos):
        rep = Atoms(numbers=Z, positions=pos, info=dict(info))
        calc.prepare([rep])
        _, F = calc.get_ef_gpu()                       # (1, nmax_dof) Ha/A
        Fr = F[0].detach().to(torch.float64).cpu().numpy()
        return Fr[cols]                                # (dof,)

    for j, a in enumerate(movable):
        for k in range(3):
            row = 3 * j + k
            pp = pos0.copy(); pp[a, k] += delta
            Fp = force_movable(pp)
            pm = pos0.copy(); pm[a, k] -= delta
            Fm = force_movable(pm)
            H[row, :] = torch.from_numpy(-(Fp - Fm) / (2.0 * delta)).to(device)

    return 0.5 * (H + H.t())


def batched_fd_hessian(calc, atoms_list, movable_masks=None, delta: float = 2e-3,
                       atom_budget: int = 20000, max_replicas: int = 512,
                       return_padded: bool = False) -> dict:
    """Batched central-FD Hessian for B structures over their movable subspaces.

    Parameters
    ----------
    calc : AIMNet2BatchCalc-like
        Must expose prepare(atoms_list) and get_ef_gpu() -> (E (B,), F (B,nmax_dof))
        in Hartree / (Ha/A), per-molecule padded force layout (atom a -> cols 3a..3a+2).
    atoms_list : list[ase.Atoms]
    movable_masks : None | sequence  (see _resolve_movable)
    delta : float, FD step in Angstrom (default 2e-3 << 5 A cutoff).
    atom_budget : max total atoms per chunked forward (memory cap, enzyme-safe).
    max_replicas : max replica molecules per chunk.
    return_padded : also emit a (B, dof_max, dof_max) padded buffer + movable mask
        + padding counts (P-RFO drop-in shape).

    Returns dict with keys:
        hessians : list of (3*m_b, 3*m_b) float64 torch tensors (Ha/A^2)
        movable  : list[list[int]] movable atom indices per structure
        n_chunks, n_replicas, max_atoms_in_chunk : chunk/memory diagnostics
        (+ H_pad, movable_mask, pad if return_padded)
    """
    from ase import Atoms
    B = len(atoms_list)
    device = calc.device
    movable = _resolve_movable(atoms_list, movable_masks)

    Zs    = [at.get_atomic_numbers() for at in atoms_list]
    pos0  = [at.get_positions().astype(np.float64) for at in atoms_list]
    infos = [dict(getattr(at, "info", {}) or {}) for at in atoms_list]
    m_b   = [len(mv) for mv in movable]
    dof_b = [3 * m for m in m_b]
    cols_b = [_movable_cols(mv) for mv in movable]

    # plus/minus force accumulators per structure: row r = perturbed movable DOF,
    # column = movable force vector at that perturbation.
    Fplus  = [np.zeros((d, d), dtype=np.float64) for d in dof_b]
    Fminus = [np.zeros((d, d), dtype=np.float64) for d in dof_b]

    # enumerate perturbation jobs: (b, atom, axis, row, sign)
    jobs = []
    for b in range(B):
        mv = movable[b]
        for j in range(m_b[b]):
            a = mv[j]
            for axis in range(3):
                row = 3 * j + axis
                jobs.append((b, a, axis, row, +1))
                jobs.append((b, a, axis, row, -1))
    R = len(jobs)

    n_chunks = 0
    max_atoms_in_chunk = 0
    idx = 0
    while idx < R:
        chunk = []
        atoms_count = 0
        while idx < R and len(chunk) < max_replicas:
            b = jobs[idx][0]
            nb = len(Zs[b])
            if chunk and (atoms_count + nb > atom_budget):
                break
            chunk.append(jobs[idx])
            atoms_count += nb
            idx += 1

        rep_list = []
        for (b, a, axis, row, sign) in chunk:
            p = pos0[b].copy()
            p[a, axis] += sign * delta
            rep_list.append(Atoms(numbers=Zs[b], positions=p, info=dict(infos[b])))

        calc.prepare(rep_list)
        _, F = calc.get_ef_gpu()                        # (chunk, nmax_dof) Ha/A
        F = F.detach().to(torch.float64).cpu().numpy()

        for slot, (b, a, axis, row, sign) in enumerate(chunk):
            Fmv = F[slot][cols_b[b]]                     # (dof_b,) movable-order forces
            if sign > 0:
                Fplus[b][row, :] = Fmv
            else:
                Fminus[b][row, :] = Fmv

        n_chunks += 1
        max_atoms_in_chunk = max(max_atoms_in_chunk, atoms_count)

    hessians = []
    for b in range(B):
        d = dof_b[b]
        if d == 0:
            hessians.append(torch.zeros((0, 0), dtype=torch.float64, device=device))
            continue
        Hb = -(Fplus[b] - Fminus[b]) / (2.0 * delta)    # float64 FD
        Hb = 0.5 * (Hb + Hb.T)                          # symmetrize per structure
        hessians.append(torch.from_numpy(Hb).to(device))

    result = dict(hessians=hessians, movable=movable, n_chunks=n_chunks,
                  n_replicas=R, max_atoms_in_chunk=max_atoms_in_chunk)

    if return_padded:
        dof_max = max(dof_b) if dof_b else 0
        H_pad = torch.zeros((B, dof_max, dof_max), dtype=torch.float64, device=device)
        movable_mask = torch.zeros((B, dof_max), dtype=torch.bool, device=device)
        pad = torch.zeros((B,), dtype=torch.int64, device=device)
        for b in range(B):
            d = dof_b[b]
            if d > 0:
                H_pad[b, :d, :d] = hessians[b]
                movable_mask[b, :d] = True
            pad[b] = dof_max - d
        result.update(H_pad=H_pad, movable_mask=movable_mask, pad=pad)

    return result
