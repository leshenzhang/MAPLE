"""Umbrella sampling helper for MAPLE MD: window generation + PMF (WHAM/MBAR).

Pairs with :mod:`plumed_calc`. The biased dynamics are produced by running one
MAPLE MD job per window with a per-window ``plumed.dat`` that holds a
``RESTRAINT`` on the chosen CV; this module both *writes* those window inputs
and *recombines* the resulting COLVAR time-series into an unbiased PMF.

Units convention
----------------
The generated ``plumed.dat`` starts with ``UNITS LENGTH=A ENERGY=kcal/mol``, so
the restraint centre ``AT`` and force constant ``KAPPA`` are given in **Å** and
**kcal/mol/Å²** and the COLVAR is written in Å — the units a chemist reasons in.
This is orthogonal to the Hartree/Å/fs MD-data exchange handled by
``PlumedCalculator`` via ``setMD*Units`` (PLUMED converts the bias force back to
Ha/Å regardless of the ``UNITS`` directive, which only governs input parsing and
file output). The PMF is returned in **kcal/mol**.

Refs (per skill ``umbrella-sampling-wham``):
Kumar et al. J. Comput. Chem. 13, 1011 (1992) (WHAM, doi:10.1002/jcc.540130812);
Shirts & Chodera J. Chem. Phys. 129, 124105 (2008) (MBAR, doi:10.1063/1.2978177);
Kästner WIREs CMS 1, 932 (2011) (US review). Defaults (20–40 windows,
κ=100–300 kcal/mol/Å², spacing 0.1–0.2 Å, tol 1e-3 kcal/mol, T=300 K, target
10–30 % histogram overlap) follow that skill's method-critical table.
"""

import os
import numpy as np

KCAL_PER_MOL_K = 0.0019872041   # k_B in kcal/(mol·K)


# --------------------------------------------------------------------------
# 1) Window generation
# --------------------------------------------------------------------------
def make_windows(cv_min, cv_max, n_windows):
    """Evenly spaced restraint centres on [cv_min, cv_max] (inclusive)."""
    if n_windows < 2:
        raise ValueError("need >= 2 windows")
    return np.linspace(float(cv_min), float(cv_max), int(n_windows))


def window_plumed_lines(cv_def, center, kappa, *, stride=10,
                        colvar_file="colvar.dat", cv_label="cv", units=True):
    """PLUMED input lines for one umbrella window.

    Parameters
    ----------
    cv_def : str
        Full PLUMED CV definition line, e.g. ``"cv: DISTANCE ATOMS=15,42"`` or
        ``"cv: DISTANCE ATOMS=15,42 ... "`` — must define a label ``cv_label``.
    center, kappa : float
        Restraint centre (Å) and force constant (kcal/mol/Å²).
    """
    lines = []
    if units:
        lines.append("UNITS LENGTH=A ENERGY=kcal/mol TIME=fs")
    lines.append(cv_def)
    lines.append(f"RESTRAINT ARG={cv_label} AT={center:.6f} KAPPA={kappa:.6f} "
                 f"LABEL=restraint")
    lines.append(f"PRINT ARG={cv_label},restraint.bias STRIDE={stride} "
                 f"FILE={colvar_file}")
    return lines


def generate_umbrella_inputs(out_dir, cv_def, cv_min, cv_max, n_windows,
                             kappa, *, stride=10, cv_label="cv"):
    """Write ``window_NNN/plumed.dat`` for every window + a manifest.

    Returns the list of (index, center, window_dir, plumed_path).
    """
    centers = make_windows(cv_min, cv_max, n_windows)
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for i, c in enumerate(centers):
        wdir = os.path.join(out_dir, f"window_{i:03d}")
        os.makedirs(wdir, exist_ok=True)
        ppath = os.path.join(wdir, "plumed.dat")
        lines = window_plumed_lines(cv_def, c, kappa, stride=stride,
                                    colvar_file="colvar.dat", cv_label=cv_label)
        with open(ppath, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        manifest.append((i, float(c), wdir, ppath))
    with open(os.path.join(out_dir, "windows.manifest"),
              "w", encoding="utf-8") as fh:
        fh.write(f"# n_windows={n_windows} kappa={kappa} cv={cv_def!r}\n")
        for i, c, wdir, ppath in manifest:
            fh.write(f"{i:03d}\t{c:.6f}\t{wdir}\n")
    return manifest


# --------------------------------------------------------------------------
# 2) COLVAR parsing + overlap diagnostic
# --------------------------------------------------------------------------
def read_colvar(path, column=0, *, discard_frac=0.1):
    """Read the CV time-series (first ARG column) from a PLUMED COLVAR file.

    ``discard_frac`` drops the leading fraction as equilibration (skill default
    100–500 ps). Returns a 1-D float array of CV samples.
    """
    rows = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            # column 0 is time; CV is column 1 (+column offset).
            rows.append(float(parts[1 + column]))
    arr = np.asarray(rows, dtype=np.float64)
    if discard_frac > 0 and arr.size:
        arr = arr[int(discard_frac * arr.size):]
    return arr


def histogram_overlap(samples_list, nbins=100):
    """Mean nearest-neighbour histogram overlap fraction (skill: want 10–30%).

    Returns (overlaps, edges) where ``overlaps[i]`` is the overlap between
    window i and i+1 (Bhattacharyya-style min-area). Warn if any < 0.1.
    """
    allx = np.concatenate(samples_list)
    edges = np.linspace(allx.min(), allx.max(), nbins + 1)
    hists = [np.histogram(s, bins=edges, density=True)[0] for s in samples_list]
    w = edges[1] - edges[0]
    overlaps = []
    for a, b in zip(hists[:-1], hists[1:]):
        overlaps.append(float(np.minimum(a, b).sum() * w))
    return np.asarray(overlaps), edges


# --------------------------------------------------------------------------
# 3) PMF: WHAM (always available) + MBAR (preferred if pymbar present)
# --------------------------------------------------------------------------
def wham_1d(centers, kappas, samples_list, temperature, *, nbins=100,
            tol=1e-3, max_iter=100000, cv_range=None):
    """Iterative binned WHAM (Kumar 1992). PMF in kcal/mol, min shifted to 0.

    centers, kappas : per-window restraint centre (Å) and κ (kcal/mol/Å²).
    samples_list    : list of 1-D CV-sample arrays, one per window.
    """
    kbt = KCAL_PER_MOL_K * float(temperature)
    centers = np.asarray(centers, float)
    kappas = np.broadcast_to(np.asarray(kappas, float), centers.shape)
    K = len(centers)

    allx = np.concatenate(samples_list)
    lo, hi = (allx.min(), allx.max()) if cv_range is None else cv_range
    edges = np.linspace(lo, hi, nbins + 1)
    xb = 0.5 * (edges[:-1] + edges[1:])                  # bin centres

    H = np.zeros(nbins)                                  # total counts per bin
    N = np.zeros(K)                                      # samples per window
    for i, s in enumerate(samples_list):
        N[i] = s.size
        H += np.histogram(s, bins=edges)[0]

    # bias energy of window i at bin b: 0.5 κ_i (x_b - c_i)^2  (kcal/mol)
    bias = 0.5 * kappas[:, None] * (xb[None, :] - centers[:, None]) ** 2
    expo = np.exp(-bias / kbt)                           # (K, nbins)

    f = np.zeros(K)                                      # window free energies
    P = np.zeros(nbins)
    for _ in range(max_iter):
        denom = (N[:, None] * np.exp(f[:, None] / kbt) * expo).sum(axis=0)
        nz = denom > 0
        P[:] = 0.0
        P[nz] = H[nz] / denom[nz]
        f_new = -kbt * np.log(np.where(P > 0, (expo * P[None, :]),
                                       0.0).sum(axis=1) + 1e-300)
        f_new -= f_new[0]
        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new

    with np.errstate(divide="ignore"):
        pmf = -kbt * np.log(np.where(P > 0, P, np.nan))
    pmf -= np.nanmin(pmf)
    return xb, pmf


def mbar_1d(centers, kappas, samples_list, temperature, *, nbins=100):
    """MBAR PMF via pymbar (preferred). Falls back to WHAM if pymbar absent.

    Returns (bin_centres, pmf_kcal). Uses pymbar's FES/PMF on the harmonic
    restraint energy matrix.
    """
    try:
        from pymbar import MBAR
    except Exception:
        xb, pmf = wham_1d(centers, kappas, samples_list, temperature,
                          nbins=nbins)
        return xb, pmf, "wham-fallback"

    kbt = KCAL_PER_MOL_K * float(temperature)
    centers = np.asarray(centers, float)
    kappas = np.broadcast_to(np.asarray(kappas, float), centers.shape)
    K = len(centers)
    N_k = np.array([s.size for s in samples_list])
    x_n = np.concatenate(samples_list)                          # all samples
    # reduced potential u_kn = bias_k(x_n) / kBT
    u_kn = 0.5 * kappas[:, None] * (x_n[None, :] - centers[:, None]) ** 2 / kbt
    mbar = MBAR(u_kn, N_k)
    edges = np.linspace(x_n.min(), x_n.max(), nbins + 1)
    xb = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(x_n, edges) - 1, 0, nbins - 1)
    # unbiased expectation per bin → -kT ln P
    fes = np.full(nbins, np.nan)
    w = mbar.weights()[:, 0] if hasattr(mbar, "weights") else None
    # Use the simple histogram of unbiased weights (compat across pymbar vers.)
    try:
        from pymbar import FES
        fobj = FES(u_kn, N_k)
        fobj.generate_fes(u_kn, x_n, fes_type="histogram",
                          histogram_parameters={"bin_edges": edges})
        res = fobj.get_fes(xb)
        pmf = res["f_i"] * kbt
    except Exception:
        # weight-histogram fallback
        if w is None:
            return wham_1d(centers, kappas, samples_list, temperature,
                           nbins=nbins) + ("wham-fallback",)
        P = np.zeros(nbins)
        for b in range(nbins):
            P[b] = w[bin_idx == b].sum()
        with np.errstate(divide="ignore"):
            pmf = -kbt * np.log(np.where(P > 0, P, np.nan))
    pmf = pmf - np.nanmin(pmf)
    return xb, pmf, "mbar"


if __name__ == "__main__":
    # Runnable check: synthesise samples from K harmonic windows on a FLAT true
    # PMF. WHAM must recover an (approximately) flat PMF — max deviation small.
    rng = np.random.RandomState(0)
    T = 300.0
    kbt = KCAL_PER_MOL_K * T
    centers = np.linspace(2.0, 4.0, 21)            # Å
    kappa = 200.0                                  # kcal/mol/Å²
    # flat underlying PMF ⇒ window i samples ~ N(center_i, sqrt(kBT/κ))
    sigma = np.sqrt(kbt / kappa)
    samples = [centers[i] + sigma * rng.randn(4000) for i in range(len(centers))]

    ov, _ = histogram_overlap(samples, nbins=120)
    xb, pmf = wham_1d(centers, kappa, samples, T, nbins=80)
    core = (xb > 2.2) & (xb < 3.8)                 # ignore low-stat edges
    dev = np.nanmax(pmf[core]) - np.nanmin(pmf[core])
    print(f"OK umbrella WHAM self-test: flat-PMF residual={dev:.3f} kcal/mol "
          f"(expect <~0.3); mean window overlap={ov.mean():.2f}")
    assert dev < 0.5, f"WHAM did not recover flat PMF: {dev}"
    xb2, pmf2, how = mbar_1d(centers, kappa, samples, T, nbins=80)
    print(f"OK mbar_1d path = {how}")
