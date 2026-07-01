"""
Mean-squared displacement (MSD) and the self-diffusion coefficient.

MSD(t) is computed with the FFT-based windowed estimator (Calandrini et al.,
2011) — averaging over ALL time origins and atoms — which is what
MDAnalysis ``msd.EinsteinMSD(fft=True)`` does. The Einstein relation gives

    D = slope( MSD vs t ) / (2 * dim)      (dim = 3 -> /6)

UNWRAPPING (periodic runs only). MAPLE writers never wrap coordinates and
never store image flags, so for a *periodic* run continuous trajectories must
be reconstructed before the Einstein fit. We infer image jumps from the
inter-frame displacement: a jump > L/2 in any periodic direction is treated as
a wrap and undone (fractional round). LIMITATION: this fails if an atom truly
moves more than half a box length between two *saved* frames — i.e. at large
``traj_every`` with fast species. Keep traj_every small for diffusion, or use
unwrap=False if MAPLE already wrote unwrapped coordinates.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms


def _to_array(frames):
    """Accept (nframes, natoms, 3) array or an iterable of ase.Atoms."""
    if isinstance(frames, np.ndarray):
        return frames, None
    frames = list(frames)
    pos = np.array([a.get_positions() for a in frames])
    cell = None
    if len(frames) and np.any(frames[0].pbc):
        cell = frames[0].cell[:]
    return pos, cell


def unwrap_positions(positions, cell, pbc=True):
    """
    Unwrap a wrapped periodic trajectory by undoing < L/2 image jumps.

    positions : (nframes, natoms, 3)
    cell      : (3, 3) row-vector lattice (assumed ~constant; first frame used)
    Returns unwrapped positions, same shape.
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    pbc = np.asarray(pbc, dtype=bool)
    if pbc.ndim == 0:
        pbc = np.array([bool(pbc)] * 3)
    inv = np.linalg.inv(cell)

    delta = np.diff(positions, axis=0)                 # (nf-1, nat, 3)
    frac = delta @ inv
    jump = np.where(pbc, np.round(frac), 0.0)
    frac_corr = frac - jump
    delta_corr = frac_corr @ cell
    unwrapped = np.empty_like(positions)
    unwrapped[0] = positions[0]
    unwrapped[1:] = positions[0] + np.cumsum(delta_corr, axis=0)
    return unwrapped


def _autocorr_fft(x):
    """FFT autocorrelation of x (N,) -> (N,), normalised by (N - m)."""
    N = len(x)
    F = np.fft.fft(x, n=2 * N)
    psd = F * F.conjugate()
    res = np.fft.ifft(psd)[:N].real
    norm = N - np.arange(N)
    return res / norm


def _msd_single(r):
    """FFT windowed MSD for one atom, r: (N, 3) -> (N,)."""
    N = len(r)
    sq = np.square(r).sum(axis=1)                       # (N,)
    D = np.append(sq, 0.0)
    S2 = sum(_autocorr_fft(r[:, d]) for d in range(r.shape[1]))
    Q = 2.0 * D.sum()
    S1 = np.empty(N)
    for m in range(N):
        Q -= D[m - 1] + D[N - m]
        S1[m] = Q / (N - m)
    return S1 - 2.0 * S2


def compute_msd(
    frames,
    dt=1.0,
    unwrap=True,
    cell=None,
    pbc=True,
    fit_range=None,
    dim=3,
):
    """
    Compute MSD(t) and the diffusion coefficient.

    Parameters
    ----------
    frames : (nframes, natoms, 3) array OR iterable of ase.Atoms
    dt : float
        Time between saved frames (fs * traj_every, or any consistent unit).
        D is returned in [length^2 / time] of these units.
    unwrap : bool, default True
        Undo periodic image jumps before the Einstein fit.
    cell, pbc : optional
        Cell (3,3) + pbc; taken from the Atoms frames if not given.
    fit_range : (lo, hi) fractional window of the MSD curve for the linear fit,
        default (0.1, 0.5) to avoid the ballistic head and the noisy tail.
    dim : int, default 3
        Spatial dimensionality for D = slope / (2*dim).

    Returns
    -------
    dict with keys ``t``, ``msd``, ``D``, ``slope``, ``intercept``.
    """
    pos, frame_cell = _to_array(frames)
    nframes, natoms, _ = pos.shape
    if cell is None:
        cell = frame_cell

    if unwrap:
        if cell is None:
            # nothing to unwrap (isolated / non-periodic run)
            pass
        else:
            pos = unwrap_positions(pos, cell, pbc)

    # MSD averaged over atoms.
    msd = np.zeros(nframes)
    for i in range(natoms):
        msd += _msd_single(pos[:, i, :])
    msd /= natoms

    t = np.arange(nframes) * dt

    if fit_range is None:
        fit_range = (0.1, 0.5)
    lo = max(1, int(fit_range[0] * nframes))
    hi = max(lo + 2, int(fit_range[1] * nframes))
    hi = min(hi, nframes)
    slope, intercept = np.polyfit(t[lo:hi], msd[lo:hi], 1)
    D = slope / (2.0 * dim)
    return {"t": t, "msd": msd, "D": D, "slope": slope, "intercept": intercept}
