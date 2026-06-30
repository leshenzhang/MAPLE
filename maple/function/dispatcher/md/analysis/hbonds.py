"""
Geometric hydrogen-bond analysis (topology-free).

MAPLE trajectories carry NO bond graph, so donors/acceptors are inferred from
element identity + geometry:

* Acceptors: electronegative atoms (default N, O, F).
* Donors:    the same electronegative elements that are covalently bonded to an
             H (a heavy atom within ``dh_cutoff`` of an H is its donor).
* An H-bond D-H...A exists when  d(D,A) <= ``da_cutoff``  AND the
  D-H...A angle >= ``angle_cutoff`` (degrees), with A != D.

Defaults (da_cutoff 3.5 A, angle_cutoff 150 deg, dh_cutoff 1.2 A) follow the
common cpptraj/MDAnalysis geometric criterion. NOTE: this is a GEOMETRIC count
and will NOT exactly reproduce a topology-aware tool's donor/acceptor lists —
it is framed as geometric H-bonds with stated cutoffs, not a topology match.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from .pbc import mic_displacements, mic_distance_matrix

_DEFAULT_ACCEPTORS = ("N", "O", "F")


def _angle_deg(v1, v2):
    """Angle between vector batches v1, v2 (..., 3) in degrees."""
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos = np.einsum("...c,...c->...", v1, v2) / (n1 * n2 + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def count_hbonds(
    atoms: Atoms,
    donors=_DEFAULT_ACCEPTORS,
    acceptors=_DEFAULT_ACCEPTORS,
    da_cutoff=3.5,
    angle_cutoff=150.0,
    dh_cutoff=1.2,
    return_pairs=False,
):
    """
    Count geometric H-bonds in a single frame.

    Returns the integer count, or (count, list[(D,H,A)]) if ``return_pairs``.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    pos = atoms.get_positions()
    periodic = bool(np.any(atoms.pbc))
    cell = atoms.cell[:] if periodic else None

    h_idx = np.where(symbols == "H")[0]
    donor_set = set(donors)
    accept_set = set(acceptors)
    heavy_donor_idx = np.array(
        [i for i, s in enumerate(symbols) if s in donor_set], dtype=int
    )
    acceptor_idx = np.array(
        [i for i, s in enumerate(symbols) if s in accept_set], dtype=int
    )
    if len(h_idx) == 0 or len(heavy_donor_idx) == 0 or len(acceptor_idx) == 0:
        return (0, []) if return_pairs else 0

    # 1) assign each H to its donor heavy atom (nearest heavy donor within cutoff)
    dh = mic_distance_matrix(pos[h_idx], pos[heavy_donor_idx], cell=cell, pbc=atoms.pbc)
    nearest = np.argmin(dh, axis=1)
    nearest_d = dh[np.arange(len(h_idx)), nearest]
    bonded = nearest_d <= dh_cutoff
    # H -> donor atom index
    h_to_donor = {int(h_idx[k]): int(heavy_donor_idx[nearest[k]])
                  for k in range(len(h_idx)) if bonded[k]}

    if not h_to_donor:
        return (0, []) if return_pairs else 0

    count = 0
    pairs = []
    for h, d in h_to_donor.items():
        # acceptors within da_cutoff of donor, excluding the donor itself
        a_candidates = [a for a in acceptor_idx if a != d]
        if not a_candidates:
            continue
        a_arr = np.array(a_candidates, dtype=int)
        da = mic_distance_matrix(pos[[d]], pos[a_arr], cell=cell, pbc=atoms.pbc)[0]
        within = a_arr[da <= da_cutoff]
        for a in within:
            # angle D-H...A  (vectors H->D and H->A, minimum image)
            v_hd = mic_displacements(pos[[d]], pos[[h]], cell=cell, pbc=atoms.pbc)[0, 0]
            v_ha = mic_displacements(pos[[a]], pos[[h]], cell=cell, pbc=atoms.pbc)[0, 0]
            ang = _angle_deg(v_hd, v_ha)
            if ang >= angle_cutoff:
                count += 1
                if return_pairs:
                    pairs.append((int(d), int(h), int(a)))
    return (count, pairs) if return_pairs else count


def hbond_timeseries(frames, **kwargs):
    """Per-frame geometric H-bond count over a trajectory -> np.ndarray."""
    return_pairs = kwargs.pop("return_pairs", False)
    counts = []
    for atoms in frames:
        counts.append(count_hbonds(atoms, return_pairs=False, **kwargs))
    return np.array(counts)
