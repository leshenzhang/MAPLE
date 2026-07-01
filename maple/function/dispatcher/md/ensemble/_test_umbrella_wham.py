"""
In-tree correctness gate for umbrella sampling + WHAM 1-D PMF reconstruction.

Validation axis = ALGORITHM CORRECTNESS of the PMF path used by
ensemble/umbrella_batched.py (which reuses bias/umbrella.py: make_windows,
wham_1d, histogram_overlap).

  A) ANALYTIC WHAM RECOVERY (calc-free, deterministic).  For a KNOWN true PMF
     F(x), each umbrella window i (centre c_i, force constant kappa) samples the
     biased equilibrium density
         rho_i(x) ~ exp(-beta[F(x) + 1/2 kappa (x - c_i)^2]),
     which for a polynomial F is an exact Gaussian we can draw i.i.d. samples
     from.  Feeding those samples to the SAME wham_1d must reconstruct F(x)
     within tolerance (up to an additive constant).  Two references:
        * FLAT   F(x)=0            -> WHAM must return a flat PMF (null test:
          catches window free-energy self-consistency / offset bugs), and
        * HARMONIC F(x)=1/2 K0 (x-x0)^2 -> WHAM must recover the curvature K0.
     Window overlap is checked (sane 5-60 %).

  B) BATCHED UMBRELLA PIPELINE (a100, smoke).  Actually run BatchedUmbrella on a
     small molecule so the per-window restraint + CV logging + wham_1d integrate
     end to end: assert a finite PMF and that the window mean-CVs track the
     restraint centres (the restraint holds each window where it should).

Deterministic (fixed seeds).  Run as a script (never on import).
"""
import os, sys, tempfile
import numpy as np
from scipy import stats as _st

from maple.function.dispatcher.md.bias.umbrella import (
    make_windows, wham_1d, histogram_overlap, KCAL_PER_MOL_K,
)

if __name__ != "__main__":
    raise SystemExit("run _test_umbrella_wham.py as a script, not an import")


# ---------------------------------------------------------------- Gate A
def _sample_windows(centers, kappa, F_K0, F_x0, T, M, seed):
    """Draw M i.i.d. biased samples per window for true PMF F(x)=1/2 K0 (x-x0)^2.

    rho_i(x) ~ exp(-beta[1/2 K0 (x-x0)^2 + 1/2 kappa (x-c_i)^2]) is Gaussian with
    variance kbt/(K0+kappa) and mean (K0 x0 + kappa c_i)/(K0+kappa)."""
    kbt = KCAL_PER_MOL_K * T
    rng = np.random.default_rng(seed)
    prec = (F_K0 + kappa)
    var = kbt / prec
    sd = np.sqrt(var)
    out = []
    for c in centers:
        mu = (F_K0 * F_x0 + kappa * c) / prec
        out.append(rng.normal(mu, sd, size=M))
    return out


def _recover(label, centers, kappa, F_K0, F_x0, T, M, seed, nbins, tol):
    samples = _sample_windows(centers, kappa, F_K0, F_x0, T, M, seed)
    xb, pmf = wham_1d(centers, kappa, samples, T, nbins=nbins)
    overlaps, _ = histogram_overlap(samples, nbins=nbins)
    mean_ov = float(np.mean(overlaps)) if overlaps.size else float("nan")

    # interior comparison window [c[1], c[-2]] where every bin is well sampled
    lo, hi = centers[1], centers[-2]
    m = (xb >= lo) & (xb <= hi) & np.isfinite(pmf)
    F_true = 0.5 * F_K0 * (xb - F_x0) ** 2
    dev = pmf[m] - F_true[m]
    dev = dev - dev.mean()                        # remove the additive constant
    max_dev = float(np.max(np.abs(dev)))
    print(f"[US-WHAM/{label}] windows={len(centers)} kappa={kappa} K0={F_K0} "
          f"mean_overlap={mean_ov:.3f}  max|PMF-F|(interior)={max_dev:.4f} kcal/mol (tol {tol})")
    ok = (max_dev < tol) and (0.05 < mean_ov < 0.60)
    return ok, max_dev, mean_ov


def gate_analytic_wham(T=300.0):
    centers = make_windows(2.5, 4.5, 10)
    kappa = 40.0
    okA, devA, ovA = _recover("FLAT", centers, kappa, F_K0=0.0, F_x0=3.5,
                              T=T, M=6000, seed=0, nbins=60, tol=0.20)
    okB, devB, ovB = _recover("HARMONIC", centers, kappa, F_K0=5.0, F_x0=3.5,
                              T=T, M=6000, seed=1, nbins=60, tol=0.30)
    ok = okA and okB
    print(f"[US-WHAM/ANALYTIC] {'PASS' if ok else 'FAIL'}")
    return ok, devA, devB


# ---------------------------------------------------------------- Gate B
def gate_batched_umbrella_pipeline(steps=400, dt=0.5, seed=5):
    import torch
    torch.set_default_dtype(torch.float64)
    from ase.build import molecule
    from maple.function.calculator.mace._maceoff_batch_calculator import MaceOffBatchCalc
    from maple.function.dispatcher.md.ensemble.umbrella_batched import BatchedUmbrella

    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
    et = molecule("CH3CH2OH")                      # C0 C1 O2 H...  (COM-COM CV: C0..O2)
    calc = MaceOffBatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64)
    paras = dict(us_nwindows=4, us_group1="0", us_group2="2",
                 us_kappa=250.0, us_cv_min=2.3, us_cv_max=3.1, us_nbins=40,
                 steps=steps, timestep=dt, temperature=300.0,
                 thermostat="langevin", tau_t=20.0, friction=0.02,
                 random_seed=seed, remove_com_every=100, verbose=0)
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        out = f.name
    us = BatchedUmbrella(out, et, calc=calc, paras=paras).run()

    centers = np.asarray(us._centers)
    wmean = np.asarray(us.window_mean_cv)
    pmf = np.asarray(us.pmf)
    finite_pmf = bool(np.any(np.isfinite(pmf)))
    rho, _ = _st.spearmanr(centers, wmean)          # windows must track their centres
    print(f"[US-WHAM/PIPELINE] centres={np.round(centers,3).tolist()}  "
          f"<CV>window={np.round(wmean,3).tolist()}")
    print(f"[US-WHAM/PIPELINE] finite_PMF={finite_pmf}  spearman(centres,<CV>)={rho:.3f}  "
          f"PMF_range={float(np.nanmax(pmf)-np.nanmin(pmf)):.3f} kcal/mol")
    ok = finite_pmf and (rho >= 0.8)
    print(f"[US-WHAM/PIPELINE] {'PASS' if ok else 'FAIL'}")
    return ok, float(rho)


if __name__ == "__main__":
    okA, devA, devB = gate_analytic_wham()
    okB, rho = gate_batched_umbrella_pipeline()
    allok = okA and okB
    print(f"\n[US-WHAM RESULT] analytic(flat_dev={devA:.4f} harm_dev={devB:.4f}) "
          f"pipeline(spearman={rho:.3f})")
    print(f"[US-WHAM RESULT] {'ALL UMBRELLA/WHAM GATES PASS' if allok else 'SOME FAILED'}")
    sys.exit(0 if allok else 1)
