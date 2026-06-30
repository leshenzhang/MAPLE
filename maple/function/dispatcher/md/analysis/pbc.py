"""
Minimum-image-convention (MIC) distance helpers for general triclinic cells.

MAPLE's MD code carries NO PBC/MIC math of its own (it is a force-field-free
velocity map plus writers); this module is the *only* place the analysis
package needs the minimum-image geometry, kept deliberately self-contained
(numpy only; ASE used as the reference oracle in tests).

Convention
----------
``cell`` is a (3, 3) row-vector matrix: ``cell[i]`` is lattice vector ``a_i``
(ASE convention, ``atoms.cell[:]``). A cartesian displacement ``d`` maps to
fractional coordinates by ``f = d @ inv(cell)``.

For a *general* triclinic cell, rounding the fractional displacement to the
nearest image is NOT always the true minimum image (skewed cells can have a
shorter vector one cell further out). We therefore (a) wrap fractional to
[-0.5, 0.5), then (b) brute-force search the 3x3x3 block of neighbouring
images and keep the shortest. This reproduces ``ase.geometry.get_distances``
to ~1e-10 A for the moderately-skewed cells MD produces. Pathologically
sheared cells could need a wider search; documented, not handled.
"""

from __future__ import annotations

import numpy as np

# Neighbour-image shifts (3x3x3 = 27 cells), used to fix the triclinic MIC.
_SHIFTS = np.array(
    [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=float,
)


def _periodic_mask(pbc):
    if pbc is None:
        return np.array([True, True, True])
    pbc = np.asarray(pbc, dtype=bool)
    if pbc.ndim == 0:
        return np.array([bool(pbc)] * 3)
    return pbc


def minimum_image(disp, cell, pbc=True):
    """
    Map cartesian displacement vectors to their minimum image.

    Parameters
    ----------
    disp : (..., 3) array
        Cartesian displacement vectors (e.g. r_i - r_j).
    cell : (3, 3) array
        Row-vector lattice matrix.
    pbc : bool or (3,) bool, default True
        Which directions are periodic.

    Returns
    -------
    (..., 3) array
        Minimum-image displacement vectors.
    """
    disp = np.asarray(disp, dtype=float)
    cell = np.asarray(cell, dtype=float)
    pbc = _periodic_mask(pbc)
    if not pbc.any():
        return disp

    inv = np.linalg.inv(cell)
    frac = disp @ inv                      # (..., 3)
    # Only wrap the periodic directions.
    shift = np.where(pbc, np.round(frac), 0.0)
    frac = frac - shift
    base = frac @ cell                     # nearest-image cartesian (pre search)

    # Brute-force the 27 neighbouring images to catch skewed-cell cases.
    shifts = _SHIFTS.copy()
    shifts[:, ~pbc] = 0.0                  # never shift a non-periodic axis
    shifts = np.unique(shifts, axis=0)
    cand = base[..., None, :] + (shifts @ cell)        # (..., S, 3)
    d2 = np.einsum("...sc,...sc->...s", cand, cand)    # (..., S)
    best = np.argmin(d2, axis=-1)                      # (...)
    out = np.take_along_axis(cand, best[..., None, None], axis=-2)[..., 0, :]
    return out


def mic_displacements(pos_a, pos_b, cell=None, pbc=True):
    """
    All pairwise minimum-image displacement vectors r_a - r_b.

    Returns (Na, Nb, 3). If ``cell`` is None (or pbc all False) plain
    open-boundary displacements are returned.
    """
    pos_a = np.asarray(pos_a, dtype=float)
    pos_b = np.asarray(pos_b, dtype=float)
    disp = pos_a[:, None, :] - pos_b[None, :, :]       # (Na, Nb, 3)
    if cell is None or not _periodic_mask(pbc).any():
        return disp
    return minimum_image(disp, cell, pbc)


def mic_distance_matrix(pos_a, pos_b, cell=None, pbc=True):
    """
    Pairwise minimum-image distances, shape (Na, Nb).

    With ``cell=None`` this is the plain euclidean distance matrix (used for
    the isolated/non-periodic batched replicas, which carry no cell).
    """
    disp = mic_displacements(pos_a, pos_b, cell=cell, pbc=pbc)
    return np.linalg.norm(disp, axis=-1)
