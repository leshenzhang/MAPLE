"""
Trajectory readers for MAPLE MD output.

Three backends, all yielding ``ase.Atoms`` (positions + cell + pbc):

* ``DCDTrajReader``   — exact inverse of ``dcd_writer.py``'s byte layout.
* ``MapleXYZReader``  — parses MAPLE's NON-standard XYZ comment
                        ``Frame <step>  Energy = <E> Hartree  Cell = a b c
                        alpha beta gamma  PBC = T T T`` (a naive extxyz reader
                        silently loses the cell -> wrong PBC -> wrong RDF/MSD).
* ``MultiReplicaReader`` — globs ``{stem}.rep*.xyz`` from a batched run.

#1 FOOTGUN (DCD): the DCD format stores ONLY coordinates + cell, NO element
symbols and NO topology. ``DCDTrajReader`` therefore makes a topology source
MANDATORY (``symbols=`` / ``top=`` / ``rst=``) and raises loudly if absent.
"""

from __future__ import annotations

import re
import struct
import glob
import importlib.util
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np
from ase import Atoms
from ase.geometry import cellpar_to_cell


# ----------------------------------------------------------------------------
# DCD byte layout — these MUST stay the exact inverse of dcd_writer.py.
# header : marker(4) + 21*int32(84) + marker(4)          = 92 bytes
# title  : marker(4) + 160 + marker(4)                   = 168 bytes
# natom  : marker(4) + int32(4) + marker(4)              = 12 bytes
# frame  : cell[ marker(4)+6*f64(48)+marker(4) ]         = 56 bytes
#          + X/Y/Z each [ marker(4)+natom*f32+marker(4) ]
# DCD cell order written by the writer: A, gamma, B, beta, alpha, C
# (CHARMM historical order), all as plain lengths(A) / angles(deg).
# ----------------------------------------------------------------------------
_DCD_HEADER_BYTES = 92
_DCD_TITLE_BYTES = 168
_DCD_NATOM_BYTES = 12


class DCDTrajReader:
    """
    Read a CHARMM/NAMD DCD trajectory written by ``DCDWriter``.

    Parameters
    ----------
    path : str or Path
        DCD file.
    symbols : sequence of str, optional
        Element symbols (length = natoms). MANDATORY unless ``top`` or ``rst``
        is given. DCD carries no element identity, so without this every
        downstream observable is meaningless.
    top : str or Path, optional
        Topology sidecar: an ASE-readable structure (first frame's symbols are
        used) OR a plain text file of whitespace/newline-separated symbols.
    rst : str or Path, optional
        MAPLE ``.rst`` checkpoint; symbols are taken from ``read_rst``.

    Yields ``ase.Atoms`` per frame.
    """

    def __init__(self, path, symbols=None, top=None, rst=None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"DCD file not found: {self.path}")
        self.symbols = resolve_symbols(symbols=symbols, top=top, rst=rst)
        self._read_header()
        if self.symbols is None:
            raise ValueError(
                "DCDTrajReader: DCD files store NO element symbols — a topology "
                "is MANDATORY. Pass symbols=[...] or top='<file>' or rst='<file.rst>'. "
                f"(file has natoms={self.natoms})"
            )
        if len(self.symbols) != self.natoms:
            raise ValueError(
                f"Topology/symbols length {len(self.symbols)} != DCD natoms "
                f"{self.natoms} for {self.path}"
            )

    # -- header -------------------------------------------------------------
    def _read_header(self):
        with open(self.path, "rb") as f:
            if struct.unpack("<i", f.read(4))[0] != 84:
                raise ValueError(f"Not a DCD file (bad header marker): {self.path}")
            hdr = np.frombuffer(f.read(84), dtype=np.int32)
            if struct.unpack("<i", f.read(4))[0] != 84:
                raise ValueError("DCD header end marker mismatch")
            if hdr[0] != 84 or len(hdr) != 21:
                raise ValueError("Invalid DCD header block")
            self._nset_field = int(hdr[2])           # backpatched frame count
            self.natoms = int(hdr[6])
            self.timestep = float(hdr[9]) * 1000.0   # ps -> fs
            # skip title + natom blocks
            f.read(4); f.read(160); f.read(4)
            f.read(4)
            natom_check = struct.unpack("<i", f.read(4))[0]
            f.read(4)
            if natom_check != self.natoms:
                raise ValueError(
                    f"DCD natoms inconsistency: {self.natoms} != {natom_check}"
                )
        coord_block = 4 + self.natoms * 4 + 4
        self._frame_bytes = 56 + 3 * coord_block
        file_size = self.path.stat().st_size
        traj_bytes = file_size - (_DCD_HEADER_BYTES + _DCD_TITLE_BYTES + _DCD_NATOM_BYTES)
        # Trust file-size arithmetic over the (possibly stale) NSET field.
        self.nframes = max(0, traj_bytes // self._frame_bytes)

    def __len__(self):
        return self.nframes

    # -- frame iteration ----------------------------------------------------
    def _read_record(self, f, expect_bytes):
        m0 = f.read(4)
        if len(m0) < 4:
            return None
        n = struct.unpack("<i", m0)[0]
        if n != expect_bytes:
            raise ValueError(
                f"DCD record marker mismatch: expected {expect_bytes}, got {n}"
            )
        data = f.read(n)
        m1 = struct.unpack("<i", f.read(4))[0]
        if m1 != n:
            raise ValueError("DCD record end marker mismatch")
        return data

    def __iter__(self) -> Iterator[Atoms]:
        nat = self.natoms
        with open(self.path, "rb") as f:
            f.seek(_DCD_HEADER_BYTES + _DCD_TITLE_BYTES + _DCD_NATOM_BYTES)
            while True:
                cell_data = self._read_record(f, 48)
                if cell_data is None:
                    break
                dcd_cell = np.frombuffer(cell_data, dtype=np.float64)
                x = np.frombuffer(self._read_record(f, nat * 4), dtype=np.float32)
                y = np.frombuffer(self._read_record(f, nat * 4), dtype=np.float32)
                z = np.frombuffer(self._read_record(f, nat * 4), dtype=np.float32)
                positions = np.column_stack([x, y, z]).astype(float)
                atoms = Atoms(symbols=list(self.symbols), positions=positions)
                # Invert writer cell order: A, gamma, B, beta, alpha, C
                if np.any(dcd_cell != 0.0):
                    a, gamma, b, beta, alpha, c = dcd_cell
                    cellpar = [a, b, c, alpha, beta, gamma]
                    atoms.set_cell(cellpar_to_cell(cellpar))
                    atoms.set_pbc(True)
                yield atoms

    def read_all(self) -> List[Atoms]:
        return list(self)


# ----------------------------------------------------------------------------
# MAPLE-flavoured XYZ
# ----------------------------------------------------------------------------
# Matches the writer in utils.py:write_xyz_frame, e.g.
#   Frame 1000  Energy = -123.4567890000 Hartree  Cell = 10.0 10.0 10.0 90.0 90.0 90.0  PBC = T T T
_FRAME_RE = re.compile(r"Frame\s+(-?\d+)")
_ENERGY_RE = re.compile(r"Energy\s*=\s*(-?[\d.eE+]+)")
_CELL_RE = re.compile(
    r"Cell\s*=\s*"
    r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+"
    r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)"
)
_PBC_RE = re.compile(r"PBC\s*=\s*([TF])\s+([TF])\s+([TF])")


def parse_maple_xyz_comment(comment: str):
    """
    Parse a MAPLE XYZ comment line.

    Returns dict with keys: ``frame`` (int|None), ``energy`` (float|None),
    ``cellpar`` ([a,b,c,al,be,ga] or None), ``pbc`` ([bool]*3 or None).
    """
    out = {"frame": None, "energy": None, "cellpar": None, "pbc": None}
    m = _FRAME_RE.search(comment)
    if m:
        out["frame"] = int(m.group(1))
    m = _ENERGY_RE.search(comment)
    if m:
        try:
            out["energy"] = float(m.group(1))
        except ValueError:
            pass
    m = _CELL_RE.search(comment)
    if m:
        out["cellpar"] = [float(v) for v in m.groups()]
    m = _PBC_RE.search(comment)
    if m:
        out["pbc"] = [t == "T" for t in m.groups()]
    return out


class MapleXYZReader:
    """
    Read a MAPLE-flavoured XYZ trajectory, correctly recovering the cell from
    the non-standard comment line. Symbols come from the file itself.

    Parameters
    ----------
    path : str or Path
    symbols : optional
        If given, used to cross-check the per-frame symbol column.
    """

    def __init__(self, path, symbols=None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"XYZ file not found: {self.path}")
        self.expected_symbols = list(symbols) if symbols is not None else None
        self.nframes = self._count_frames()

    def _count_frames(self):
        n = 0
        with open(self.path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                nat = int(line)
                f.readline()                       # comment
                for _ in range(nat):
                    f.readline()
                n += 1
        return n

    def __len__(self):
        return self.nframes

    def __iter__(self) -> Iterator[Atoms]:
        with open(self.path) as f:
            while True:
                header = f.readline()
                if not header:
                    break
                header = header.strip()
                if not header:
                    continue
                nat = int(header)
                comment = f.readline()
                meta = parse_maple_xyz_comment(comment)
                syms, pos = [], []
                for _ in range(nat):
                    parts = f.readline().split()
                    syms.append(parts[0])
                    pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if self.expected_symbols is not None and syms != self.expected_symbols:
                    raise ValueError(
                        f"XYZ symbol column disagrees with supplied symbols in {self.path}"
                    )
                atoms = Atoms(symbols=syms, positions=np.asarray(pos, dtype=float))
                if meta["cellpar"] is not None and any(v > 0 for v in meta["cellpar"][:3]):
                    atoms.set_cell(cellpar_to_cell(meta["cellpar"]))
                    atoms.set_pbc(meta["pbc"] if meta["pbc"] is not None else True)
                if meta["energy"] is not None:
                    atoms.info["energy"] = meta["energy"]
                if meta["frame"] is not None:
                    atoms.info["step"] = meta["frame"]
                yield atoms

    def read_all(self) -> List[Atoms]:
        return list(self)


# ----------------------------------------------------------------------------
# Multi-replica loader (batched runs write {stem}.rep{b}.xyz)
# ----------------------------------------------------------------------------
class MultiReplicaReader:
    """
    Load a batched run's per-replica XYZ files ``{stem}.rep{b}.xyz``.

    Parameters
    ----------
    stem : str or Path
        Either the run stem (``run`` -> globs ``run.rep*.xyz``) or a path with
        a ``.rep*`` glob already; the ``.xyz`` suffix is optional.
    """

    def __init__(self, stem):
        stem = str(stem)
        for suf in (".rep*.xyz", ".xyz", ""):
            if stem.endswith(suf) and suf:
                stem = stem[: -len(suf)]
                break
        pattern = f"{stem}.rep*.xyz"
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No replica files match {pattern!r}")

        def _bidx(p):
            m = re.search(r"\.rep(\d+)\.xyz$", p)
            return int(m.group(1)) if m else 0

        self.files = sorted(files, key=_bidx)
        self.B = len(self.files)
        self.readers = [MapleXYZReader(f) for f in self.files]

    def __len__(self):
        return self.B

    def replica(self, b) -> MapleXYZReader:
        return self.readers[b]

    def iter_replicas(self):
        """Yield (b, list[Atoms]) for each replica."""
        for b, r in enumerate(self.readers):
            yield b, r.read_all()

    def stack_positions(self):
        """
        Return positions as (B, nframes, N, 3). Requires every replica to have
        the same nframes and N (raises otherwise). Batched runs are isolated /
        non-periodic by construction, so no cell is returned.
        """
        per = [r.read_all() for r in self.readers]
        nframes = len(per[0])
        natoms = len(per[0][0])
        for b, frames in enumerate(per):
            if len(frames) != nframes:
                raise ValueError(
                    f"Replica {b} has {len(frames)} frames != {nframes}; "
                    "cannot stack — analyse per replica instead."
                )
            if len(frames[0]) != natoms:
                raise ValueError(f"Replica {b} has {len(frames[0])} atoms != {natoms}")
        arr = np.empty((self.B, nframes, natoms, 3), dtype=float)
        for b, frames in enumerate(per):
            for t, at in enumerate(frames):
                arr[b, t] = at.get_positions()
        return arr


# ----------------------------------------------------------------------------
# Topology / symbol resolution
# ----------------------------------------------------------------------------
def _load_read_rst(rst_path):
    """Import md/rst_io.py BY FILE PATH so we never trigger the torch-heavy
    md package __init__ — keeps this analysis package usable without the engine."""
    here = Path(__file__).resolve().parent          # .../md/analysis
    rst_io_path = here.parent / "rst_io.py"          # .../md/rst_io.py
    spec = importlib.util.spec_from_file_location("_maple_rst_io", rst_io_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.read_rst(rst_path)


def resolve_symbols(symbols=None, top=None, rst=None) -> Optional[List[str]]:
    """
    Resolve element symbols for a topology-less trajectory (DCD).

    Priority: explicit ``symbols`` > ``rst`` (read_rst) > ``top`` sidecar.
    A ``top`` file is first tried as an ASE-readable structure; on failure it
    is parsed as plain whitespace/newline-separated element tokens.
    Returns None if nothing is supplied (caller decides whether that is fatal).
    """
    if symbols is not None:
        return [str(s) for s in symbols]
    if rst is not None:
        return list(_load_read_rst(rst)["symbols"])
    if top is not None:
        top = Path(top)
        if not top.exists():
            raise FileNotFoundError(f"Topology file not found: {top}")
        try:
            from ase.io import read as ase_read
            atoms = ase_read(str(top), index=0)
            return atoms.get_chemical_symbols()
        except Exception:
            tokens = top.read_text().split()
            if not tokens:
                raise ValueError(f"Topology sidecar {top} contained no symbols")
            return tokens
    return None


def read_trajectory(path, fmt=None, symbols=None, top=None, rst=None):
    """
    Convenience dispatch by extension (or explicit ``fmt`` in {dcd,xyz,replica}).
    Returns a reader object (iterable of ase.Atoms).
    """
    path = str(path)
    if fmt is None:
        if path.endswith(".dcd"):
            fmt = "dcd"
        elif ".rep" in path and path.endswith(".xyz") and "*" in path:
            fmt = "replica"
        elif path.endswith(".xyz"):
            fmt = "xyz"
        else:
            raise ValueError(f"Cannot infer trajectory format from {path!r}; pass fmt=")
    if fmt == "dcd":
        return DCDTrajReader(path, symbols=symbols, top=top, rst=rst)
    if fmt == "xyz":
        return MapleXYZReader(path, symbols=symbols)
    if fmt == "replica":
        return MultiReplicaReader(path)
    raise ValueError(f"Unknown trajectory format {fmt!r}")
