# -*- coding: utf-8 -*-
"""Shared utilities for OPT algorithms."""
import os
from typing import List, Optional

import numpy as np
from ase import Atoms


def write_xyz(filename: str, atoms_list: List[Atoms],
              energies: Optional[List[float]] = None,
              mode: str = "w",
              start_index: int = 0) -> None:
    """Write one or more structures in XYZ format."""
    blocks = []
    for i, at in enumerate(atoms_list):
        pos = at.get_positions()
        symbols = at.get_chemical_symbols()
        image_index = start_index + i

        lines = [f"{len(symbols)}\n"]
        if energies is not None:
            lines.append(f"Image {image_index}  Energy = {energies[i]:.10f}\n")
        else:
            lines.append(f"Image {image_index}\n")
        lines.extend(
            f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n"
            for s, (x, y, z) in zip(symbols, pos)
        )
        blocks.append("".join(lines))

    with open(filename, mode) as f:
        f.writelines(blocks)
        f.flush()
        os.fsync(f.fileno())


def to_numpy_f64(x):
    """Convert input to float64 numpy array or float."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
            return arr.astype(np.float64, copy=False)
    except Exception:
        pass
    if np.isscalar(x):
        return float(x)
    return np.asarray(x, dtype=np.float64)


def vec1d(x, n_expected: Optional[int] = None) -> np.ndarray:
    """Convert to float64 1D vector and optionally check length."""
    v = to_numpy_f64(x).reshape(-1)
    if n_expected is not None and v.size != n_expected:
        raise ValueError(f"Expected size {n_expected}, got {v.size}")
    return v


def compute_metrics(atoms, step_cart, forces) -> None:
    """Set max_dp, rms_dp, max_f, rms_f on atoms from step and force arrays."""
    atoms.max_dp = np.abs(step_cart).max()
    atoms.rms_dp = np.sqrt((step_cart ** 2).sum() / step_cart.size)
    atoms.max_f = np.abs(forces).max()
    atoms.rms_f = np.sqrt((forces ** 2).sum() / forces.size)


def is_converged(atoms) -> bool:
    """Test geometry-optimization convergence against atoms.*_th thresholds."""
    return (
        atoms.max_f <= atoms.f_max_th
        and atoms.rms_f <= atoms.f_rms_th
        and atoms.max_dp <= atoms.dp_max_th
        and atoms.rms_dp <= atoms.dp_rms_th
    )
