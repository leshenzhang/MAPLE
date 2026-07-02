"""Two-CV (2D) umbrella-sampling recombination for MAPLE MD: 2D-window input
generation + 2D-WHAM PMF reconstruction.

Companion to :mod:`umbrella` (1D WHAM/MBAR). Same biased-dynamics workflow — one
MAPLE MD job per 2D window, each with a ``plumed.dat`` holding a *two-argument*
``RESTRAINT`` on a pair of CVs — but the recombination solves the binned WHAM
equations on a 2D bin grid (Kumar 1992 generalised to two reaction coordinates).
The 2D iteration is algebraically identical to :func:`umbrella.wham_1d` with the
bin index flattened: ``bias_i(bx,by)=½[κ_ix(x_bx−c_ix)²+κ_iy(y_by−c_iy)²]``.

Units convention (mirrors :mod:`umbrella`): restraint centres ``AT`` and force
constants ``KAPPA`` are in **Å** / **kcal/mol/Å²** (PLUMED ``UNITS LENGTH=A
ENERGY=kcal/mol``) and the COLVAR holds the two CVs in Å. The PMF is returned in
**kcal/mol**, global minimum shifted to 0.

Portability scope (pure-MLIP MD — every step is a single monolithic
``atoms.get_forces()`` with no force-field / solute–solvent energy split)
------------------------------------------------------------------------------
2D-WHAM is a **pure post-processing** recombination of biased CV time-series: it
only ever touches the *collective-variable* histograms and the analytic harmonic
restraint energy, never the potential-energy decomposition. It is therefore
fully portable to any MLIP, exactly like the 1D umbrella recombiner.

By contrast the following are **EXCLUDED** as non-portable to a pure MLIP, for
the same reason REST2 is (see :mod:`steered` / GaMD notes): they need a
force-field potential-energy *decomposition* that one monolithic MLIP forward
pass does not expose —
  * **alchemical FEP / TI** — require a λ-coupled (or solute/solvent) split of
    the potential energy to form ``∂U/∂λ`` / ``ΔU(λ→λ')``; an MLIP returns only
    the total energy/forces, so the coupling derivative is unavailable.
  * **REST2** — needs the solute–solvent energy partition to scale only the
    solute terms.
A geometric-CV **blue-moon / constrained-mean-force** estimator *is* portable in
principle (it needs only a holonomic CV constraint plus its measured
Lagrange-multiplier force + the Fixman metric correction), but is **deferred**
here: it is an MD-integration-loop change — the constraint solver must surface
the per-step multiplier and the ``|Z|^{-1/2}`` Jacobian — not an offline
post-processing module, so it is out of scope for this recombiner.

Refs: Kumar et al. J. Comput. Chem. 13, 1011 (1992) (WHAM,
doi:10.1002/jcc.540130812); Roux, Comput. Phys. Commun. 91, 275 (1995)
(multidimensional WHAM, doi:10.1016/0010-4655(95)00053-I); Souaille & Roux,
Comput. Phys. Commun. 135, 40 (2001) (doi:10.1016/S0010-4655(00)00215-0).
"""

import os
import numpy as np

from .umbrella import KCAL_PER_MOL_K


# --------------------------------------------------------------------------
# 1) 2D window-lattice generation
# --------------------------------------------------------------------------
def make_grid_2d(x_min, x_max, nx, y_min, y_max, ny):
    """Rectangular lattice of 2D restraint centres.

    Returns ``(centers, gx, gy)`` where ``centers`` is ``(nx*ny, 2)`` row-major
    over ``(x, y)`` (x varies slowest) and ``gx``/``gy`` are the 1-D axes (Å).
    """
    if nx < 2 or ny < 2:
        raise ValueError("need >= 2 windows per dimension")
    gx = np.linspace(float(x_min), float(x_max), int(nx))
    gy = np.linspace(float(y_min), float(y_max), int(ny))
    XX, YY = np.meshgrid(gx, gy, indexing="ij")
    centers = np.column_stack([XX.ravel(), YY.ravel()])
    return centers, gx, gy


def window_plumed_lines_2d(cv_defs, center, kappas, *, stride=10,
                           colvar_file="colvar.dat",
                           cv_labels=("cvx", "cvy"), units=True):
    """PLUMED input lines for one 2D umbrella window (two-argument RESTRAINT).

    Parameters
    ----------
    cv_defs : sequence[str]
        The two full PLUMED CV definition lines, e.g.
        ``("cvx: DISTANCE ATOMS=15,42", "cvy: DISTANCE ATOMS=15,90")`` — each
        must define the corresponding label in ``cv_labels``.
    center : (2,) restraint centre ``(AT_x, AT_y)`` in Å.
    kappas : scalar or (2,) force constant(s) in kcal/mol/Å².
    """
    cx, cy = float(center[0]), float(center[1])
    if np.isscalar(kappas):
        kx = ky = float(kappas)
    else:
        kx, ky = float(kappas[0]), float(kappas[1])
    lx, ly = cv_labels
    lines = []
    if units:
        lines.append("UNITS LENGTH=A ENERGY=kcal/mol TIME=fs")
    lines.extend(list(cv_defs))
    lines.append(f"RESTRAINT ARG={lx},{ly} AT={cx:.6f},{cy:.6f} "
                 f"KAPPA={kx:.6f},{ky:.6f} LABEL=restraint")
    lines.append(f"PRINT ARG={lx},{ly},restraint.bias STRIDE={stride} "
                 f"FILE={colvar_file}")
    return lines


def generate_umbrella_inputs_2d(out_dir, cv_defs, x_range, y_range, n_windows,
                                kappas, *, stride=10, cv_labels=("cvx", "cvy")):
    """Write ``window_NNN/plumed.dat`` for every 2D window + a manifest.

    ``x_range``/``y_range`` are ``(min, max)`` (Å); ``n_windows`` is ``(nx, ny)``;
    ``kappas`` is scalar or ``(2,)`` (kcal/mol/Å²). Returns the list of
    ``(index, (cx, cy), window_dir, plumed_path)``.
    """
    nx, ny = int(n_windows[0]), int(n_windows[1])
    centers, _, _ = make_grid_2d(x_range[0], x_range[1], nx,
                                 y_range[0], y_range[1], ny)
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for i, c in enumerate(centers):
        wdir = os.path.join(out_dir, f"window_{i:03d}")
        os.makedirs(wdir, exist_ok=True)
        ppath = os.path.join(wdir, "plumed.dat")
        lines = window_plumed_lines_2d(cv_defs, c, kappas, stride=stride,
                                       colvar_file="colvar.dat",
                                       cv_labels=cv_labels)
        with open(ppath, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        manifest.append((i, (float(c[0]), float(c[1])), wdir, ppath))
    with open(os.path.join(out_dir, "windows.manifest"),
              "w", encoding="utf-8") as fh:
        fh.write(f"# n_windows=({nx},{ny}) kappas={kappas} cvs={cv_defs!r}\n")
        for i, (cx, cy), wdir, _ in manifest:
            fh.write(f"{i:03d}\t{cx:.6f}\t{cy:.6f}\t{wdir}\n")
    return manifest


# --------------------------------------------------------------------------
# 2) COLVAR parsing (two CV columns)
# --------------------------------------------------------------------------
def read_colvar_2d(path, columns=(0, 1), *, discard_frac=0.1):
    """Read two CV columns from a PLUMED COLVAR file. Returns an ``(n, 2)`` array.

    ``columns`` indexes the ARG columns *after* the leading time column (so the
    default ``(0, 1)`` reads file columns 1 and 2). ``discard_frac`` drops the
    leading fraction as equilibration.
    """
    cx, cy = columns
    rows = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            p = ln.split()
            rows.append((float(p[1 + cx]), float(p[1 + cy])))
    arr = np.asarray(rows, dtype=np.float64).reshape(-1, 2)
    if discard_frac > 0 and arr.shape[0]:
        arr = arr[int(discard_frac * arr.shape[0]):]
    return arr


# --------------------------------------------------------------------------
# 3) 2D PMF: iterative binned WHAM (Kumar 1992, 2-CV generalisation)
# --------------------------------------------------------------------------
def wham_2d(centers, kappas, samples_list, temperature, *,
            nbins=(60, 60), tol=1e-3, max_iter=100000, cv_range=None):
    """Iterative binned 2D-WHAM. PMF in kcal/mol, global min shifted to 0.

    Parameters
    ----------
    centers : (K, 2) per-window restraint centres (Å).
    kappas  : scalar, (2,), or (K, 2) force constants (kcal/mol/Å²).
    samples_list : list of ``(n_i, 2)`` CV-sample arrays, one per window.
    nbins   : ``(nx, ny)`` bin counts for the recovered PMF grid.
    cv_range : optional ``((xlo, xhi), (ylo, yhi))``; defaults to sample extent.

    Returns
    -------
    (xb, yb, pmf) : 1-D bin-centre axes (Å) and the ``(nx, ny)`` PMF (kcal/mol),
    with ``pmf[i, j]`` at ``(xb[i], yb[j])``. Unvisited bins are ``nan``.
    """
    kbt = KCAL_PER_MOL_K * float(temperature)
    centers = np.asarray(centers, float).reshape(-1, 2)
    K = centers.shape[0]
    kappas = np.asarray(kappas, float)
    if kappas.ndim == 0:
        kappas = np.full((K, 2), float(kappas))
    elif kappas.shape == (2,):
        kappas = np.broadcast_to(kappas, (K, 2)).copy()
    else:
        kappas = kappas.reshape(K, 2)

    nx, ny = int(nbins[0]), int(nbins[1])
    alls = np.concatenate(samples_list, axis=0)
    if cv_range is None:
        xlo, xhi = float(alls[:, 0].min()), float(alls[:, 0].max())
        ylo, yhi = float(alls[:, 1].min()), float(alls[:, 1].max())
    else:
        (xlo, xhi), (ylo, yhi) = cv_range
    ex = np.linspace(xlo, xhi, nx + 1)
    ey = np.linspace(ylo, yhi, ny + 1)
    xb = 0.5 * (ex[:-1] + ex[1:])
    yb = 0.5 * (ey[:-1] + ey[1:])
    XB, YB = np.meshgrid(xb, yb, indexing="ij")          # (nx, ny)
    xbf, ybf = XB.ravel(), YB.ravel()                    # flattened bins (M,)
    M = nx * ny

    H = np.zeros(M)                                       # total counts per bin
    N = np.zeros(K)                                       # samples per window
    for i, s in enumerate(samples_list):
        s = np.asarray(s, float).reshape(-1, 2)
        N[i] = s.shape[0]
        h2d, _, _ = np.histogram2d(s[:, 0], s[:, 1], bins=[ex, ey])
        H += h2d.ravel()

    # bias_i at flattened bin m: 0.5[κ_ix (x_m-c_ix)^2 + κ_iy (y_m-c_iy)^2]
    bias = 0.5 * (kappas[:, 0:1] * (xbf[None, :] - centers[:, 0:1]) ** 2
                  + kappas[:, 1:2] * (ybf[None, :] - centers[:, 1:2]) ** 2)
    expo = np.exp(-bias / kbt)                            # (K, M)

    f = np.zeros(K)                                       # window free energies
    P = np.zeros(M)
    for _ in range(max_iter):
        denom = (N[:, None] * np.exp(f[:, None] / kbt) * expo).sum(axis=0)
        nz = denom > 0
        P[:] = 0.0
        P[nz] = H[nz] / denom[nz]
        f_new = -kbt * np.log((expo * P[None, :]).sum(axis=1) + 1e-300)
        f_new -= f_new[0]
        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new

    with np.errstate(divide="ignore"):
        pmf = -kbt * np.log(np.where(P > 0, P, np.nan))
    pmf -= np.nanmin(pmf)
    return xb, yb, pmf.reshape(nx, ny)


if __name__ == "__main__":
    # Runnable check: synthesise biased samples from a K-window 2D lattice on a
    # *known* separable-harmonic true PMF and require 2D-WHAM to recover it.
    #
    # For a true PMF F(x,y) = ½ kx_true (x-xc)² + ½ ky_true (y-yc)² plus a
    # harmonic window restraint ½ κ[(x-cx)² + (y-cy)²], the biased equilibrium
    # distribution is an exact product Gaussian (sum of quadratics): per axis
    #   x ~ N( (kx_true*xc + κ*cx)/(kx_true+κ),  sqrt(kBT/(kx_true+κ)) ).
    # WHAM reconstructs P_unbiased ∝ exp(-F/kBT) ⇒ PMF = F + const, so the
    # recovered surface must match F up to an additive gauge.
    rng = np.random.RandomState(0)
    T = 300.0
    kbt = KCAL_PER_MOL_K * T

    def run_case(kx_true, ky_true, label):
        xc, yc = 3.0, 3.0
        span, step = 0.8, 0.2                            # Å lattice ±0.8 / 0.2
        gx = np.round(np.arange(xc - span, xc + span + 1e-9, step), 6)
        gy = np.round(np.arange(yc - span, yc + span + 1e-9, step), 6)
        kappa = 60.0                                     # kcal/mol/Å²
        centers = np.array([(cx, cy) for cx in gx for cy in gy])
        sx = np.sqrt(kbt / (kx_true + kappa))
        sy = np.sqrt(kbt / (ky_true + kappa))
        n = 3000
        samples = []
        for (cx, cy) in centers:
            mx = (kx_true * xc + kappa * cx) / (kx_true + kappa)
            my = (ky_true * yc + kappa * cy) / (ky_true + kappa)
            samples.append(np.column_stack([mx + sx * rng.randn(n),
                                            my + sy * rng.randn(n)]))
        xb, yb, pmf = wham_2d(centers, kappa, samples, T, nbins=(40, 40))
        XB, YB = np.meshgrid(xb, yb, indexing="ij")
        Ftrue = 0.5 * kx_true * (XB - xc) ** 2 + 0.5 * ky_true * (YB - yc) ** 2
        core = ((np.abs(XB - xc) < 0.6) & (np.abs(YB - yc) < 0.6)
                & np.isfinite(pmf))
        # remove the additive gauge by mean-centring both over the core region
        a = pmf - np.nanmean(pmf[core])
        b = Ftrue - np.nanmean(Ftrue[core])
        dev = float(np.nanmax(np.abs((a - b)[core])))
        nb = int(np.count_nonzero(core))
        print(f"OK 2D-WHAM [{label:>18}] kx={kx_true:>4} ky={ky_true:>4}: "
              f"max|PMF-Ftrue| over {nb} core bins = {dev:.3f} kcal/mol "
              f"(expect <0.5)")
        assert dev < 0.5, f"2D-WHAM failed to recover {label} surface: {dev}"
        return dev

    d_flat = run_case(0.0, 0.0, "flat")
    d_harm = run_case(10.0, 6.0, "separable-harmonic")
    print(f"OK wham2d self-test PASS: flat dev={d_flat:.3f}, "
          f"separable-harmonic dev={d_harm:.3f} kcal/mol  (both <0.5)")
