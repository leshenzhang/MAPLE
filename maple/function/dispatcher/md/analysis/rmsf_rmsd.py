"""
RMSD (vs a reference frame) and per-atom RMSF, via Kabsch superposition.

* ``kabsch_rotate``  — optimal rigid rotation aligning mobile onto reference.
* ``compute_rmsd``   — RMSD(frame, ref) after optional Kabsch superposition.
* ``compute_rmsf``   — per-atom root-mean-square fluctuation about the mean of
                       the superposed ensemble.

Analytic guarantees used in the Gate-A tests: self-RMSD = 0, RMSD of a rigidly
rotated/translated copy = 0 (validates Kabsch), and parity with
``MDAnalysis.analysis.rms``.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms


def _coords(frame, indices=None):
    pos = frame.get_positions() if isinstance(frame, Atoms) else np.asarray(frame, float)
    if indices is not None:
        pos = pos[indices]
    return pos


def kabsch_rotate(mobile, ref, weights=None):
    """
    Rotate (and translate) ``mobile`` to best overlap ``ref`` (Kabsch/SVD).

    Both (N, 3). Optional per-atom ``weights`` (e.g. masses). Returns the
    superposed mobile coordinates (centred on the weighted ref centroid).
    """
    mobile = np.asarray(mobile, float)
    ref = np.asarray(ref, float)
    if weights is None:
        w = np.ones(len(mobile))
    else:
        w = np.asarray(weights, float)
    w = w / w.sum()

    mu_m = (w[:, None] * mobile).sum(0)
    mu_r = (w[:, None] * ref).sum(0)
    P = mobile - mu_m
    Q = ref - mu_r

    H = (P * w[:, None]).T @ Q
    V, S, Wt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(V @ Wt))
    Dm = np.diag([1.0, 1.0, d])
    R = V @ Dm @ Wt                       # rotates P onto Q
    return P @ R + mu_r


def compute_rmsd(frames, ref=None, indices=None, superpose=True, weights=None):
    """
    RMSD of each frame against a reference.

    Parameters
    ----------
    frames : iterable of ase.Atoms (or (N,3) arrays)
    ref : ase.Atoms or (N,3), optional. Default = first frame.
    indices : atom subset (e.g. heavy atoms).
    superpose : bool, remove rigid rotation/translation (Kabsch) first.
    weights : per-atom weights for both alignment and RMSD (e.g. masses).

    Returns
    -------
    (nframes,) RMSD array.
    """
    frames = list(frames)
    if ref is None:
        ref = frames[0]
    ref_xyz = _coords(ref, indices)
    if weights is not None:
        weights = np.asarray(weights, float)
        w = weights / weights.sum()
    else:
        w = np.full(len(ref_xyz), 1.0 / len(ref_xyz))

    out = np.empty(len(frames))
    for k, fr in enumerate(frames):
        xyz = _coords(fr, indices)
        if superpose:
            xyz = kabsch_rotate(xyz, ref_xyz, weights=weights)
        diff = xyz - ref_xyz
        out[k] = np.sqrt((w * np.square(diff).sum(1)).sum())
    return out


def compute_rmsf(frames, ref=None, indices=None, superpose=True, weights=None):
    """
    Per-atom RMSF over the (optionally Kabsch-superposed) ensemble.

    Returns (n_selected_atoms,) array of fluctuations about the mean structure.
    """
    frames = list(frames)
    ref_xyz = _coords(ref if ref is not None else frames[0], indices)

    aligned = []
    for fr in frames:
        xyz = _coords(fr, indices)
        if superpose:
            xyz = kabsch_rotate(xyz, ref_xyz, weights=weights)
        aligned.append(xyz)
    aligned = np.asarray(aligned)                       # (nf, N, 3)
    mean_xyz = aligned.mean(0)
    msf = np.square(aligned - mean_xyz).sum(2).mean(0)  # mean over frames of |dr|^2
    return np.sqrt(msf)
