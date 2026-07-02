"""
Radial distribution function g(r).

Histogram of minimum-image pairwise distances normalised by the ideal-gas
shell density, following the MDAnalysis ``InterRDF`` convention so that
element-pair RDFs match it to numerical precision:

    g(r) = count(shell) / ( density * shell_volume * n_frames )
    density = (n_A * n_B) / <V>          (ordered pairs / mean cell volume)
    shell_volume = 4/3 pi (r_out^3 - r_in^3)

For periodic frames distances use the MIC; for non-periodic frames (isolated
batched replicas) plain euclidean distances are used and a ``volume`` must be
supplied for normalisation.
"""

from __future__ import annotations

import numpy as np

from .pbc import mic_distance_matrix


def _indices_for(symbols, sel):
    if sel is None:
        return np.arange(len(symbols))
    sel = {sel} if isinstance(sel, str) else set(sel)
    return np.array([i for i, s in enumerate(symbols) if s in sel], dtype=int)


def compute_rdf(
    frames,
    r_max,
    nbins=200,
    type_a=None,
    type_b=None,
    r_min=0.0,
    volume=None,
):
    """
    Compute g(r) over a trajectory.

    Parameters
    ----------
    frames : iterable of ase.Atoms
    r_max : float
        Maximum radius (Angstrom). Must be <= half the shortest box length for
        a strictly correct MIC histogram of periodic systems.
    nbins : int, default 200
    type_a, type_b : str or sequence of str, optional
        Element selections for the two groups. None = all atoms. Distinct
        element groups map 1:1 onto ``MDAnalysis.InterRDF(g_A, g_B)``.
    r_min : float, default 0.0
    volume : float, optional
        Cell volume for non-periodic frames (ignored if frames are periodic).

    Returns
    -------
    r : (nbins,) bin centres
    g : (nbins,) g(r)
    """
    edges = np.linspace(r_min, r_max, nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    shell = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)

    count = np.zeros(nbins)
    vol_acc = 0.0
    n_frames = 0
    nA = nB = None

    for atoms in frames:
        symbols = atoms.get_chemical_symbols()
        idx_a = _indices_for(symbols, type_a)
        idx_b = _indices_for(symbols, type_b)
        if nA is None:
            nA, nB = len(idx_a), len(idx_b)
        if len(idx_a) == 0 or len(idx_b) == 0:
            continue
        pos = atoms.get_positions()
        periodic = bool(np.any(atoms.pbc))
        cell = atoms.cell[:] if periodic else None
        D = mic_distance_matrix(pos[idx_a], pos[idx_b], cell=cell, pbc=atoms.pbc)

        same_group = np.array_equal(idx_a, idx_b)
        if same_group:
            np.fill_diagonal(D, np.inf)            # exclude self pairs
        h, _ = np.histogram(D.ravel(), bins=edges)
        count += h

        if periodic:
            vol_acc += atoms.get_volume()
        elif volume is not None:
            vol_acc += volume
        else:
            raise ValueError(
                "Non-periodic frame and no `volume=` given: ideal-gas RDF "
                "normalisation is undefined without a reference volume."
            )
        n_frames += 1

    if n_frames == 0:
        raise ValueError("No frames with both selections present.")

    mean_vol = vol_acc / n_frames
    density = (nA * nB) / mean_vol
    g = count / (density * shell * n_frames)
    return centres, g
