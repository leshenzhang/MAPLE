"""
Density observables.

* ``total_density``   — total mass / number density from the cell volume.
* ``density_profile`` — 1D binned profile (mass or number) along a cell axis,
                        e.g. a water slab profile along z.

Masses come from ASE (``atoms.get_masses()``), so element identity must be
correct (DCD users: supply the topology to the reader).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

# g/mol per A^3  ->  g/cm^3
_AMU_PER_A3_TO_G_CM3 = 1.66053906660


def total_density(atoms: Atoms):
    """
    Return dict with ``mass_density`` (g/cm^3), ``number_density`` (1/A^3),
    ``volume`` (A^3), ``mass`` (amu) for one periodic frame.
    """
    if not np.any(atoms.pbc):
        raise ValueError("total_density requires a periodic frame (needs cell volume).")
    vol = atoms.get_volume()
    mass = atoms.get_masses().sum()
    return {
        "mass_density": mass / vol * _AMU_PER_A3_TO_G_CM3,
        "number_density": len(atoms) / vol,
        "volume": vol,
        "mass": mass,
    }


def density_profile(frames, axis=2, nbins=100, mode="mass", indices=None, length=None):
    """
    1D density profile along a cell ``axis`` (0/1/2), averaged over frames.

    Parameters
    ----------
    frames : iterable of ase.Atoms
    axis : int, default 2 (z)
    nbins : int, default 100
    mode : {"mass", "number"}
    indices : atom subset (e.g. only water O).
    length : box length along axis; default = cell length of the first frame.

    Returns
    -------
    centres : (nbins,) bin centres (Angstrom)
    profile : (nbins,) mass density (g/cm^3) or number density (1/A^3)
    """
    frames = list(frames)
    first = frames[0]
    if length is None:
        if np.any(first.pbc):
            length = first.cell.lengths()[axis]
        else:
            pos0 = first.get_positions()[:, axis]
            length = pos0.max() - pos0.min()
    edges = np.linspace(0.0, length, nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    hist = np.zeros(nbins)
    cross_area_acc = 0.0
    nf = 0
    for atoms in frames:
        pos = atoms.get_positions()
        coord = pos[:, axis]
        if np.any(atoms.pbc):
            coord = coord % atoms.cell.lengths()[axis]
        masses = atoms.get_masses()
        sel = indices if indices is not None else slice(None)
        if mode == "mass":
            w = masses[sel]
        elif mode == "number":
            w = np.ones(np.atleast_1d(coord[sel]).shape[0])
        else:
            raise ValueError("mode must be 'mass' or 'number'")
        h, _ = np.histogram(coord[sel], bins=edges, weights=w)
        hist += h
        if np.any(atoms.pbc):
            cross_area_acc += atoms.get_volume() / atoms.cell.lengths()[axis]
        nf += 1

    bin_width = length / nbins
    if cross_area_acc > 0:
        cross_area = cross_area_acc / nf
        bin_vol = cross_area * bin_width
    else:
        bin_vol = bin_width                         # per-length profile (no cell)
    profile = hist / nf / bin_vol
    if mode == "mass" and cross_area_acc > 0:
        profile *= _AMU_PER_A3_TO_G_CM3
    return centres, profile
