"""Cross-method free-energy consistency gate (publication cross-validation).

Validation axis = ALGORITHM self-consistency (memory: MLP terminal value vs lit is
irrelevant; the axis is estimator-vs-estimator + vs analytic). On ONE shared 1-D
double well U(x) = H*((x/a)^2 - 1)^2 (kT=1, a=1, so F(x)=U(x) exactly and the analytic
barrier is EXACTLY H kT), three independent enhanced-sampling estimators must recover
the SAME barrier within sampling error:

  * metadynamics (well-tempered) -- FES = -gamma/(gamma-1) * V_bias
  * umbrella sampling + WHAM      -- windowed harmonic restraints -> wham_1d PMF
  * eABF + CZAR                   -- extended-Lagrangian -> czar_pmf

Estimators are reused verbatim from the production modules (bias/batched_metad.py,
bias/umbrella.py, bias/eabf.py). Pure numpy (no MLIP forward) -> runs anywhere. The
shared overdamped-Langevin propagator is the same toy driver the per-method gates use.

Acceptance: each recovered barrier within TOL_ABS kT of the analytic H, AND the three
estimates mutually within TOL_CROSS kT. This is the methods-paper cross-validation
figure (one system, orthogonal estimators agree).
"""
import numpy as np

from maple.function.dispatcher.md.bias.batched_metad import WTMetadEngine
from maple.function.dispatcher.md.bias.umbrella import make_windows, wham_1d
from maple.function.dispatcher.md.bias.eabf import eabf_toy_1d

H = 5.0        # barrier height in kT (analytic answer)
A = 1.0        # minima at x = +-A
KT = 1.0
TOL_ABS = 1.0      # each method within 1.0 kT of analytic H (finite-sampling)
TOL_CROSS = 1.0    # three estimates mutually within 1.0 kT

def U(x):  return H * ((x / A) ** 2 - 1.0) ** 2
def dU(x): return H * 2.0 * ((x / A) ** 2 - 1.0) * (2.0 * x / A ** 2)


def _barrier_top_minus_wells(xgrid, fes):
    """Physical double-well barrier = F(transition top x~0) - mean F(minima near +-A).
    Robust to edge/noise contamination a global max-min catches (top at x=0, minima are
    the basin floors near +-A, NOT the grid edges)."""
    xgrid = np.asarray(xgrid); fes = np.asarray(fes)
    fin = np.isfinite(fes)
    ctr = fin & (np.abs(xgrid) < 0.25)                       # transition region x~0
    left = fin & (xgrid < -0.6) & (xgrid > -1.4)             # left basin near -A
    right = fin & (xgrid > 0.6) & (xgrid < 1.4)              # right basin near +A
    if not (ctr.any() and left.any() and right.any()):
        return float(np.nanmax(fes) - np.nanmin(fes))        # fallback
    f_top = float(np.nanmax(fes[ctr]))
    f_well = 0.5 * (float(np.nanmin(fes[left])) + float(np.nanmin(fes[right])))
    return f_top - f_well


def _ol_step(x, extra_force, dt, D, rng):
    """One overdamped-Langevin step: dx = -D/kT (U' - extra_force) dt + sqrt(2 D dt) xi.
    extra_force is the (bias/restraint) force ADDED to the physical -U'."""
    f = -dU(x) + extra_force
    return x + (D / KT) * f * dt + np.sqrt(2.0 * D * dt) * rng.standard_normal(np.shape(x))


# ---------------------------------------------------------------- metadynamics
def barrier_metad(nsteps=400000, dt=5e-3, D=1.0, pace=200, seed=1):
    eng = WTMetadEngine(KT, sigma=0.1, height=0.3, biasfactor=8.0,
                        cv_min=-2.0, cv_max=2.0, nbins=400)
    rng = np.random.default_rng(seed)
    x = -A
    for k in range(nsteps):
        _, dV = eng.eval_grid(np.array([x]))
        x = float(_ol_step(np.array([x]), -float(np.ravel(dV)[0]), dt, D, rng)[0])
        x = min(max(x, -1.9), 1.9)
        if (k + 1) % pace == 0:
            eng.deposit(np.array([x]), use_grid=True)
    gamma = 8.0
    fes = -(gamma / (gamma - 1.0)) * eng.grid_V           # WT relation
    return eng.grid_x, fes


# ------------------------------------------------------------ umbrella + WHAM
def barrier_umbrella(n_windows=24, kappa=40.0, M=4000, dt=5e-3, D=1.0, seed=2):
    """Sampler runs in reduced units (kT=1). wham_1d's kbt = KCAL_PER_MOL_K*T, so we
    pass T = 1/KCAL_PER_MOL_K -> kbt == 1 kT and the returned PMF is already in kT
    (same units as the bias 0.5*kappa*(x-c)^2 the sampler used) -> no conversion."""
    from maple.function.dispatcher.md.bias.umbrella import KCAL_PER_MOL_K
    T_kT = 1.0 / KCAL_PER_MOL_K                     # -> kbt = 1 kT inside wham_1d
    centers = make_windows(-1.4, 1.4, n_windows)
    rng = np.random.default_rng(seed)
    samples_list = []
    for c in centers:
        x = c
        xs = []
        for k in range(M + 500):
            x = float(_ol_step(np.array([x]), -kappa * (x - c), dt, D, rng)[0])
            if k >= 500:
                xs.append(x)
        samples_list.append(np.asarray(xs))
    xb, pmf = wham_1d(centers, kappa, samples_list, T_kT, nbins=80)   # PMF in kT
    return xb, pmf


# -------------------------------------------------------------------- eABF/CZAR
def barrier_eabf(nsteps=200000, seed=3):
    res = eabf_toy_1d(U, dU, x0=-A, m_x=1.0, m_lam=1.0, k=50.0, kT=KT,
                      dt=0.0015, gamma_x=2.0, gamma_lam=2.0, cv_min=-1.6,
                      cv_max=1.6, nbins=60, nsteps=nsteps, full_samples=100, seed=seed)
    return np.asarray(res["czar_x"]), np.asarray(res["czar_pmf"])


def convergence_metad(lengths=(50000, 100000, 200000, 400000), seeds=(1, 2, 3, 4)):
    """metaD FES barrier vs simulation length, mean+-std over independent seeds -- the
    'FES plateaus in time with block/replica error bars' convergence convention
    (Bussi-Laio 2020; block/bootstrap error is the accepted metaD diagnostic). A correct
    estimator converges to the analytic H and the spread shrinks with length."""
    print("\n[convergence] metaD barrier vs length (mean +- std over %d seeds):" % len(seeds))
    rows = []
    for n in lengths:
        bs = [_barrier_top_minus_wells(*barrier_metad(nsteps=n, seed=s)) for s in seeds]
        bs = np.array(bs)
        rows.append((n, float(bs.mean()), float(bs.std())))
        print(f"    n={n:>7}  barrier = {bs.mean():.3f} +- {bs.std():.3f} kT   (analytic {H})")
    # convergence: longest-length estimate within 0.6 kT of analytic + spread not growing
    n_hi, m_hi, s_hi = rows[-1]
    ok = abs(m_hi - H) < 0.6 and s_hi < 0.6
    print("[convergence]", "PASS (plateau near analytic, error bars bounded)" if ok
          else "WARN (not converged -- report as-is)")
    return rows, ok


if __name__ == "__main__":
    print(f"cross-method PMF consistency  U(x)=H((x/a)^2-1)^2  H={H} kT  analytic barrier={H} kT")
    b_meta = _barrier_top_minus_wells(*barrier_metad())
    b_eabf = _barrier_top_minus_wells(*barrier_eabf())
    b_umb = _barrier_top_minus_wells(*barrier_umbrella())     # PMF already in kT
    print(f"  [metaD]     barrier = {b_meta:.3f} kT")
    print(f"  [umbrella]  barrier = {b_umb:.3f} kT")
    print(f"  [eABF-CZAR] barrier = {b_eabf:.3f} kT")
    est = np.array([b_meta, b_umb, b_eabf])
    dev_analytic = float(np.max(np.abs(est - H)))
    dev_cross = float(est.max() - est.min())
    print(f"  max|barrier - H|     = {dev_analytic:.3f} kT  (tol {TOL_ABS})")
    print(f"  cross-method spread  = {dev_cross:.3f} kT  (tol {TOL_CROSS})")
    ok = dev_analytic < TOL_ABS and dev_cross < TOL_CROSS
    print("[CROSS-METHOD-PMF]", "ALL PASS" if ok else "FAIL")
    _, conv_ok = convergence_metad()
    print("\n[CROSS-METHOD-PMF+CONVERGENCE]", "ALL PASS" if (ok and conv_ok) else "SEE ABOVE")
    if not ok:
        raise SystemExit("cross-method PMF consistency FAIL")
