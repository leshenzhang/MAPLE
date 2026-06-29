# -*- coding: utf-8 -*-
"""
Dependency-light readers for AMBER topology (prmtop) + NetCDF trajectory.

Why hand-rolled instead of parmed/pytraj?  The binding workflow only needs a
handful of prmtop FLAG sections (atomic numbers, per-residue ranges, partial
charges) plus the ``coordinates`` array from an AMBER NetCDF file.  Pulling in
parmed/pytraj just for that would violate the "no new heavy dep / reuse stdlib"
rule -- prmtop is plain text and the NetCDF read falls back through netCDF4 ->
scipy -> a tiny built-in NetCDF-3 reader, whichever is present.

prmtop FLAG layout (AMBER spec):
    %FLAG <NAME>
    %COMMENT ...        (optional, may repeat)
    %FORMAT(...)
    <whitespace-delimited values, wrapped over many lines>
    %FLAG <NEXT> ...
"""

import os
import numpy as np

# ASE atomic numbers <-> symbols (avoid a hard ase import path here; ase is a
# MAPLE core dep so this is always available, but keep the surface minimal).
from ase.data import chemical_symbols

KCAL_PER_HARTREE = 627.5094740631  # CODATA-ish; matches MAPLE/Amber convention
AMBER_CHARGE_UNIT = 18.2223        # prmtop CHARGE is stored as q[e] * 18.2223


# --------------------------------------------------------------------------- #
# prmtop                                                                       #
# --------------------------------------------------------------------------- #
def _read_flag(lines, flag):
    """Return the whitespace-split token list of one prmtop %FLAG section."""
    out = []
    i, n = 0, len(lines)
    while i < n:
        s = lines[i]
        if s.startswith("%FLAG"):
            parts = s.split()
            if len(parts) >= 2 and parts[1] == flag:
                i += 1
                # skip %COMMENT / %FORMAT header lines
                while i < n and lines[i].startswith("%"):
                    i += 1
                while i < n and not lines[i].startswith("%FLAG"):
                    if not lines[i].startswith("%"):
                        out.extend(lines[i].split())
                    i += 1
                return out
        i += 1
    return out


class Prmtop:
    """Minimal AMBER topology view: elements, residues, charges."""

    def __init__(self, path):
        self.path = path
        lines = open(path, "r", encoding="utf-8", errors="replace").read().splitlines()
        ptr = _read_flag(lines, "POINTERS")
        if not ptr:
            raise ValueError(f"{path}: no POINTERS flag -- not a valid prmtop")
        self.natom = int(ptr[0])
        self.nres = int(ptr[11])

        z = _read_flag(lines, "ATOMIC_NUMBER")
        if len(z) != self.natom:
            raise ValueError(
                f"{path}: ATOMIC_NUMBER missing/short ({len(z)} vs natom "
                f"{self.natom}). Rebuild prmtop with a modern tleap/parmed."
            )
        self.atomic_numbers = np.asarray([int(x) for x in z], dtype=int)

        self.residue_labels = _read_flag(lines, "RESIDUE_LABEL")
        rp = [int(x) for x in _read_flag(lines, "RESIDUE_POINTER")]  # 1-based
        # 0-based [start, end) slice per residue
        self._res_start = np.asarray([p - 1 for p in rp], dtype=int)
        self._res_end = np.append(self._res_start[1:], self.natom)

        chg = _read_flag(lines, "CHARGE")
        self.charges = (np.asarray([float(x) for x in chg], dtype=float)
                        / AMBER_CHARGE_UNIT) if len(chg) == self.natom else None

    @property
    def symbols(self):
        return [chemical_symbols[z] for z in self.atomic_numbers]

    def residue_ranges(self):
        """List of (label, start0, end0) over all residues."""
        return [(self.residue_labels[i], int(self._res_start[i]), int(self._res_end[i]))
                for i in range(self.nres)]

    def select(self, resnames):
        """0-based atom indices whose residue label is in ``resnames`` (set/list)."""
        rs = set(resnames)
        idx = []
        for lbl, a, b in self.residue_ranges():
            if lbl in rs:
                idx.extend(range(a, b))
        return np.asarray(idx, dtype=int)

    def ligand_receptor_masks(self, ligand_resname="LIG", solvent_resnames=None):
        """
        Return (ligand_idx, receptor_idx) as 0-based int arrays.

        receptor = everything that is NOT the ligand and NOT a solvent/ion
        residue (the latter is empty for a stripped complex).  Raises if the
        ligand residue is absent or appears more than once (ambiguous mask).
        """
        solvent = set(solvent_resnames or
                      ("WAT", "HOH", "Na+", "Cl-", "K+", "NA", "CL", "MG", "ZN"))
        lig = self.select([ligand_resname])
        if lig.size == 0:
            labels = sorted({l for l, _, _ in self.residue_ranges()})
            raise ValueError(
                f"ligand resname '{ligand_resname}' not found. "
                f"Residue labels present: {labels}")
        n_lig_res = sum(1 for l, _, _ in self.residue_ranges() if l == ligand_resname)
        if n_lig_res != 1:
            raise ValueError(
                f"ligand resname '{ligand_resname}' matches {n_lig_res} residues "
                "(expected exactly 1). Pass a unique resname.")
        lig_set = set(lig.tolist())
        rec = []
        for lbl, a, b in self.residue_ranges():
            if lbl == ligand_resname or lbl in solvent:
                continue
            rec.extend(range(a, b))
        return lig, np.asarray(rec, dtype=int)


# --------------------------------------------------------------------------- #
# NetCDF trajectory (AMBER convention: var 'coordinates' (frame, atom, 3), Ang)#
# --------------------------------------------------------------------------- #
def read_nc_coords(path, frames=None):
    """
    Read AMBER NetCDF coordinates -> ndarray (n_sel, natom, 3) in Angstrom.

    ``frames`` may be None (all), an int (first N), or an explicit index list.
    Tries netCDF4, then scipy.io.netcdf_file (NetCDF-3, the AMBER default).
    """
    coords = _load_coordinates(path)
    if frames is None:
        sel = range(coords.shape[0])
    elif isinstance(frames, int):
        sel = range(min(frames, coords.shape[0]))
    else:
        sel = list(frames)
    return np.asarray(coords[list(sel)], dtype=float)


def _load_coordinates(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # 1) netCDF4 (handles both NetCDF-3 and NetCDF-4/HDF5)
    try:
        import netCDF4  # noqa
        ds = netCDF4.Dataset(path, "r")
        try:
            return np.array(ds.variables["coordinates"][:], dtype=float)
        finally:
            ds.close()
    except ImportError:
        pass
    # 2) scipy NetCDF-3 reader (AMBER trajectories are NetCDF-3 64-bit offset)
    try:
        from scipy.io import netcdf_file
        with netcdf_file(path, "r", mmap=False) as ds:
            return np.array(ds.variables["coordinates"][:], dtype=float)
    except ImportError:
        pass
    raise ImportError(
        "Reading AMBER NetCDF needs netCDF4 or scipy. Neither is importable in "
        "this environment.")


if __name__ == "__main__":
    # Self-check: parse a prmtop given as argv[1], print masks summary.
    import sys
    if len(sys.argv) > 1:
        p = Prmtop(sys.argv[1])
        lig, rec = p.ligand_receptor_masks(
            sys.argv[2] if len(sys.argv) > 2 else "LIG")
        print(f"natom={p.natom} nres={p.nres} "
              f"ligand_atoms={lig.size} receptor_atoms={rec.size} "
              f"charges={'yes' if p.charges is not None else 'no'}")
        assert lig.size + rec.size <= p.natom
        print("OK")
