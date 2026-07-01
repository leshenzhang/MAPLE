# -*- coding: utf-8 -*-
"""
Docking -> MAPLE bridge.

Turns a docked pose (AutoDock Vina / AutoDock / any common structure file) plus
a receptor into a single MAPLE-ready ``ase.Atoms`` complex and a starter
EM -> MD config, so an externally docked pose can be refined / simulated with a
pure-MLIP MAPLE run (EM pre-stage clears docking clashes, then NVT/NPT).

Why a bridge and not "just read a file": docking outputs are messy.  Vina emits
PDBQT (PDB plus AutoDock atom types + Gasteiger charges in trailing columns,
which ASE cannot parse); receptors come as PDB.  This module reads both, maps
AutoDock types back to elements, merges them, and emits a config whose ``em``
key is ON by default because docked poses routinely have bad contacts that would
blow up an MLIP integrator on step 1 (same rationale as the MD EM pre-stage).

Optional: ``run_vina`` shells out to a ``vina`` binary when present; it is fully
gated -- absent binary -> clear error, never a hard import dependency.
"""

import os
import shutil
import subprocess
import numpy as np
from ase import Atoms
from ase.io import read as ase_read

# AutoDock atom type -> element (types not listed fall back to the leading
# alpha chars, which covers C, N, O, H, S, P and the halogens).
_AD_TYPE_TO_ELEM = {
    "A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N",
    "O": "O", "OA": "O", "OS": "O", "S": "S", "SA": "S",
    "H": "H", "HD": "H", "HS": "H", "P": "P",
    "F": "F", "Cl": "Cl", "CL": "Cl", "Br": "Br", "BR": "Br", "I": "I",
    "MG": "Mg", "ZN": "Zn", "CA": "Ca", "FE": "Fe", "MN": "Mn",
}


def _elem_from_ad_type(tok):
    """Map an AutoDock type token to an element symbol."""
    if tok in _AD_TYPE_TO_ELEM:
        return _AD_TYPE_TO_ELEM[tok]
    # strip trailing digits, try 2- then 1-char element
    base = "".join(ch for ch in tok if ch.isalpha())
    if base[:2] in _AD_TYPE_TO_ELEM:
        return _AD_TYPE_TO_ELEM[base[:2]]
    return base[:1].upper() if base else "X"


def read_pdbqt(path):
    """Parse a Vina/AutoDock PDBQT (single model) -> ase.Atoms."""
    syms, pos = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                tok = line[77:].split()  # trailing AutoDock type (and charge)
                ad_type = tok[-1] if tok else line[12:16].strip()
                syms.append(_elem_from_ad_type(ad_type))
                pos.append((x, y, z))
            elif line.startswith("ENDMDL"):
                break  # first pose only
    if not syms:
        raise ValueError(f"{path}: no ATOM/HETATM records")
    return Atoms(symbols=syms, positions=np.array(pos))


def read_structure(path):
    """Read PDBQT / PDB / SDF / MOL / XYZ / etc. -> ase.Atoms (first model)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdbqt":
        return read_pdbqt(path)
    atoms = ase_read(path)            # ASE handles pdb/sdf/mol/xyz/...
    if isinstance(atoms, list):
        atoms = atoms[0]
    return atoms


def merge_complex(receptor, ligand):
    """Concatenate receptor + ligand into one Atoms (receptor atoms first)."""
    if isinstance(receptor, str):
        receptor = read_structure(receptor)
    if isinstance(ligand, str):
        ligand = read_structure(ligand)
    complex_atoms = receptor + ligand          # ASE Atoms concatenation
    # record the ligand atom indices so downstream binding analysis knows the mask
    n_rec = len(receptor)
    complex_atoms.info["ligand_indices"] = list(range(n_rec, n_rec + len(ligand)))
    complex_atoms.info["receptor_indices"] = list(range(n_rec))
    return complex_atoms


def default_md_config(ensemble="nvt", em="lbfgs", **overrides):
    """
    Starter EM -> MD config for a docked complex (gas/implicit, non-periodic by
    default).  ``em`` defaults ON because docked poses have clashes.  Override
    any key (e.g. add a box + ``ensemble='npt'`` for a solvated complex).
    """
    cfg = {
        # EM pre-stage (Eh/Ang force tol; clears docking clashes)
        "em": em, "emtol": 0.05, "emstep": 0.1, "em_maxsteps": 300,
        # MD
        "ensemble": ensemble, "timestep": 0.5, "steps": 2000,
        "temperature": 300.0, "thermostat": "v-rescale", "tau_t": 100.0,
        "constraints": "h-bonds", "init_velocities": True,
    }
    cfg.update(overrides)
    return cfg


def dock_to_maple(receptor, ligand, ensemble="nvt", em="lbfgs", **cfg_overrides):
    """
    One-call bridge: (receptor, docked ligand pose) -> (complex Atoms, MD config).

    The returned Atoms is ready to attach a MAPLE calculator to and hand to the
    EM pre-stage + NVT/NPT ensemble; the config carries an EM clash-clearing
    stage by default.
    """
    complex_atoms = merge_complex(receptor, ligand)
    cfg = default_md_config(ensemble=ensemble, em=em, **cfg_overrides)
    return complex_atoms, cfg


def run_vina(receptor_pdbqt, ligand_pdbqt, center, box_size,
             out_pdbqt="vina_out.pdbqt", exhaustiveness=8, vina_bin="vina"):
    """
    Optional: run AutoDock Vina to GENERATE a pose, then it can feed read_pdbqt.

    Gated on a ``vina`` binary being on PATH (or an explicit path).  center and
    box_size are (x,y,z) tuples in Angstrom.  Returns ``out_pdbqt`` on success.
    """
    exe = shutil.which(vina_bin) or (vina_bin if os.path.exists(vina_bin) else None)
    if exe is None:
        raise FileNotFoundError(
            f"vina binary '{vina_bin}' not found on PATH. Install AutoDock Vina, "
            "or pass an already-docked pose to read_pdbqt/dock_to_maple directly.")
    cmd = [exe, "--receptor", receptor_pdbqt, "--ligand", ligand_pdbqt,
           "--center_x", str(center[0]), "--center_y", str(center[1]),
           "--center_z", str(center[2]),
           "--size_x", str(box_size[0]), "--size_y", str(box_size[1]),
           "--size_z", str(box_size[2]),
           "--exhaustiveness", str(exhaustiveness), "--out", out_pdbqt]
    subprocess.run(cmd, check=True)
    return out_pdbqt


if __name__ == "__main__":
    # Self-test: write a tiny receptor + ligand PDB, bridge them, check the merge
    # mask and config without needing any external binary or MLIP.
    import tempfile
    d = tempfile.mkdtemp()
    rec = Atoms("OHH", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    lig = Atoms("CO", positions=[[5, 0, 0], [6.2, 0, 0]])
    rp, lp = os.path.join(d, "rec.pdb"), os.path.join(d, "lig.pdb")
    rec.write(rp); lig.write(lp)
    cx, cfg = dock_to_maple(rp, lp, ensemble="nvt")
    assert len(cx) == 5, len(cx)
    assert cx.info["ligand_indices"] == [3, 4], cx.info
    assert cfg["em"] == "lbfgs" and cfg["ensemble"] == "nvt"
    print(f"merged complex n={len(cx)} ligand_idx={cx.info['ligand_indices']} "
          f"em={cfg['em']} ensemble={cfg['ensemble']}")
    print("docking bridge self-test OK")
