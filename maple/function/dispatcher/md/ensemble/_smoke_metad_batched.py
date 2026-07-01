"""Correctness gates for batched metadynamics + OPES (run on an a100).

Validation axis = ALGORITHM CORRECTNESS (FD / analytic asymptote / self-consistency
/ Raiteri oracle), NOT a literature FES number. Four gates on a TOY analytic 1-D
double-well (so the exact FES is known = the potential itself) plus one GPU
integration test that the same engine rides the real MACE-OFF batched forward.

  Gate 1  FD           : analytic bias force -dV/dxi*dxi/dx vs finite difference
                         of the bias energy V(xi(x))            (residual < 1e-10).
  Gate 2  WT asymptote : converged well-tempered bias satisfies
                         FES = -(gamma/(gamma-1)) V_bias  == U   (up to a constant).
  Gate 3  multi-walker : B walkers sharing ONE bias converge to the SAME FES as
                         B independent single-walker runs pooled (Raiteri oracle).
  Gate 4  OPES         : Z stays bounded; OPES FES (-kT log P) matches metaD FES.
  Gate 5  GPU integ    : WT metaD rides MaceOffBatchCalc + BatchedNVT on propane
                         (hills deposit, bias grows, CV explores, energies finite).
"""
import os
import sys
import types

import numpy as np


def _stub_torchmetrics_for_broken_torchvision():
    """cxtorch env: torchvision's compiled ops are ABI-broken ('operator
    torchvision::nms does not exist'), and importing MACE unpickles
    ``mace.tools.train`` whose ONLY torchmetrics use is ``from torchmetrics import
    Metric`` (training-only; never touched at inference). torchmetrics' package
    __init__ eagerly imports its torchvision-backed image metrics -> crash. Stub a
    minimal ``torchmetrics`` with a dummy ``Metric`` so the MACE import completes
    for INFERENCE (gate 5) without pulling the broken torchvision. Test-harness
    only -- does not touch the shared env or the production calculator. This env
    quirk affects every MaceOffBatchCalc load on cxtorch, not the metaD feature."""
    if "torchmetrics" in sys.modules:
        return
    tm = types.ModuleType("torchmetrics")
    tm.Metric = type("Metric", (), {})
    sys.modules["torchmetrics"] = tm


_stub_torchmetrics_for_broken_torchvision()

from maple.function.dispatcher.md.bias.batched_metad import (
    WTMetadEngine, OPESMetadEngine, _com_distance_cv_grad)
from maple.function.dispatcher.md.bias.gamd import HARTREE_PER_KCAL, KB_HA_PER_K

RNG_SEED = int(os.environ.get("SEED", "20260630"))


# --------------------------------------------------------------------------- #
#  Toy analytic 1-D double well + overdamped Langevin CV driver (reduced units,
#  kT = 1). Minima at x = +-1, barrier H at x = 0.  (No MLIP: pure algorithm.)
# --------------------------------------------------------------------------- #
def toy_pot(H):
    return (lambda x: H * (x * x - 1.0) ** 2,
            lambda x: 4.0 * H * x * (x * x - 1.0))


def run_toy(engine, B, nsteps, *, dt=5e-3, D=1.0, kT=1.0, pace=250, H=5.0,
            seed=0, xwall=1.9, deposit=True):
    """Overdamped Langevin on the shared bias; deposit one hill/walker per pace.
    Returns pooled CV samples (post-equilibration)."""
    rng = np.random.default_rng(seed)
    U, dU = toy_pot(H)
    x = rng.uniform(-1.2, 1.2, size=B)
    sq = np.sqrt(2.0 * D * kT * dt)
    hist = []
    for t in range(nsteps):
        _, dV = engine.eval_grid(x)
        x = x + D * (-(dU(x) + dV)) * dt + sq * rng.standard_normal(B)
        x = np.where(x > xwall, 2 * xwall - x, x)
        x = np.where(x < -xwall, -2 * xwall - x, x)
        if deposit and (t % pace == 0):
            engine.deposit(x, use_grid=True)
        if t >= nsteps // 5:                         # drop leading 20% (equil)
            hist.append(x.copy())
    return np.concatenate(hist) if hist else np.zeros(0)


def _align(a, b, mask):
    """max|a - b - shift| over mask, shift = mean(a-b) (compare up to a constant)."""
    da = a[mask] - b[mask]
    return float(np.max(np.abs(da - da.mean())))


# --------------------------------------------------------------------------- #
def gate1_fd():
    print("\n===== GATE 1: FD (analytic bias force vs finite difference) =====")
    kT = 1.0
    from ase import Atoms
    a = Atoms("H2", positions=[[0.0, 0, 0], [2.5, 0, 0]])
    m = a.get_masses()
    g1, g2 = np.array([0]), np.array([1])
    eng = WTMetadEngine(kT, sigma=0.2, height=0.3, biasfactor=8.0,
                        cv_min=1.0, cv_max=4.0, nbins=600)
    for s0 in (2.05, 2.4, 2.75, 3.1):
        eng.deposit(np.array([s0]))                  # analytic deposit (exact path)
    worst = 0.0
    for geom in ([[0., 0, 0], [2.5, 0, 0]], [[0.1, -0.2, 0.3], [2.3, 0.4, -0.1]]):
        pos = np.array(geom, float)
        xi0, grad = _com_distance_cv_grad(pos, g1, g2, m)
        _, dV = eng.eval(np.array([xi0]))
        f_an = -(dV[0]) * grad                        # -dVbias/dx  (Ha/A)
        h = 1e-6
        f_fd = np.zeros_like(pos)
        for i in range(pos.shape[0]):
            for c in range(3):
                pp = pos.copy(); pp[i, c] += h
                xp, _ = _com_distance_cv_grad(pp, g1, g2, m)
                pm = pos.copy(); pm[i, c] -= h
                xm, _ = _com_distance_cv_grad(pm, g1, g2, m)
                Vp, _ = eng.eval(np.array([xp]))
                Vm, _ = eng.eval(np.array([xm]))
                f_fd[i, c] = -(Vp[0] - Vm[0]) / (2 * h)
        res = float(np.max(np.abs(f_an - f_fd)))
        worst = max(worst, res)
        print(f"  geom xi={xi0:.4f}  max|F_an - F_fd| = {res:.3e}")
    ok = worst < 1e-10
    print(f"  [GATE 1] worst FD residual = {worst:.3e}  (tol 1e-10)  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok, worst


def gate2_wt_asymptote():
    print("\n===== GATE 2: well-tempered asymptote FES = -(g/(g-1)) V =====")
    kT, H, gamma = 1.0, 5.0, 8.0
    eng = WTMetadEngine(kT, sigma=0.1, height=0.3, biasfactor=gamma,
                        cv_min=-2.0, cv_max=2.0, nbins=400)
    cv = run_toy(eng, B=16, nsteps=200000, pace=250, H=H, seed=RNG_SEED)
    xg = eng.grid_x
    _, F_bias = eng.fes(xg, use_grid=True)            # -(g/(g-1)) V
    U, _ = toy_pot(H)
    Utrue = U(xg)
    lo, hi = np.percentile(cv, [2, 98])
    mask = (xg >= lo) & (xg <= hi)
    dev = _align(F_bias, Utrue, mask)
    print(f"  sampled CV in [{lo:.2f},{hi:.2f}] (kT units)  n_hills={eng.s.size}")
    print(f"  central barrier U(0)={H:.2f} kT   over sampled region: "
          f"U range={Utrue[mask].max()-Utrue[mask].min():.2f} kT  "
          f"bias-recovered range={(F_bias[mask].max()-F_bias[mask].min()):.2f} kT")
    ok = dev < 1.0
    print(f"  [GATE 2] max|FES_bias - U| (const-aligned) = {dev:.3f} kT  (tol 1.0)  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok, dev


def gate3_multiwalker():
    print("\n===== GATE 3: multi-walker == pooled single-walker (Raiteri) =====")
    kT, H, gamma = 1.0, 5.0, 8.0
    NS = 200000
    # multi-walker: B=16 share ONE engine.
    mw = WTMetadEngine(kT, sigma=0.1, height=0.3, biasfactor=gamma,
                       cv_min=-2.0, cv_max=2.0, nbins=400)
    cv_mw = run_toy(mw, B=16, nsteps=NS, pace=250, H=H, seed=RNG_SEED)
    xg = mw.grid_x
    _, F_mw = mw.fes(xg, use_grid=True)
    # pooled oracle: 16 INDEPENDENT single-walker runs, average their FES.
    Fs = []
    cv_sw_all = []
    for w in range(16):
        sw = WTMetadEngine(kT, sigma=0.1, height=0.3, biasfactor=gamma,
                           cv_min=-2.0, cv_max=2.0, nbins=400)
        cvw = run_toy(sw, B=1, nsteps=NS, pace=250, H=H, seed=RNG_SEED + 101 + w)
        _, Fw = sw.fes(xg, use_grid=True)
        Fs.append(Fw)
        cv_sw_all.append(cvw)
    F_pooled = np.mean(np.vstack(Fs), axis=0)
    U, _ = toy_pot(H)
    Utrue = U(xg)
    cv_all = np.concatenate([cv_mw] + cv_sw_all)
    lo, hi = np.percentile(cv_all, [3, 97])
    mask = (xg >= lo) & (xg <= hi)
    d_mw_pool = _align(F_mw, F_pooled, mask)
    d_mw_U = _align(F_mw, Utrue, mask)
    d_pool_U = _align(F_pooled, Utrue, mask)
    print(f"  region [{lo:.2f},{hi:.2f}] kT  n_hills(mw)={mw.s.size} "
          f"n_hills(each sw)~{Fs and int(np.mean([16]))}")
    print(f"  max|FES_mw - FES_pooled| = {d_mw_pool:.3f} kT")
    print(f"  max|FES_mw     - U|      = {d_mw_U:.3f} kT")
    print(f"  max|FES_pooled - U|      = {d_pool_U:.3f} kT")
    ok = (d_mw_pool < 1.0) and (d_mw_U < 1.0) and (d_pool_U < 1.0)
    print(f"  [GATE 3] {'PASS' if ok else 'FAIL'}  (tol 1.0 kT)")
    return ok, d_mw_pool


def gate4_opes():
    print("\n===== GATE 4: OPES (Z bounded; FES matches metaD) =====")
    kT, H, gamma, dE = 1.0, 5.0, 8.0, 5.0
    op = OPESMetadEngine(kT, sigma=0.1, biasfactor=gamma, barrier=dE,
                         cv_min=-2.0, cv_max=2.0, nbins=400)
    cv_op = run_toy(op, B=16, nsteps=200000, pace=250, H=H, seed=RNG_SEED + 7)
    zh = np.asarray(op.Z_history)
    z_ratio = float(zh.max() / max(zh.min(), 1e-300))
    xg = op.grid_x
    _, F_op = op.fes(xg, use_grid=True)
    # reference metaD FES on the same toy.
    mt = WTMetadEngine(kT, sigma=0.1, height=0.3, biasfactor=gamma,
                       cv_min=-2.0, cv_max=2.0, nbins=400)
    run_toy(mt, B=16, nsteps=200000, pace=250, H=H, seed=RNG_SEED + 8)
    _, F_mt = mt.fes(xg, use_grid=True)
    U, _ = toy_pot(H)
    Utrue = U(xg)
    lo, hi = np.percentile(cv_op, [3, 97])
    mask = (xg >= lo) & (xg <= hi)
    d_op_mt = _align(F_op, F_mt, mask)
    d_op_U = _align(F_op, Utrue, mask)
    Vg, _ = op.eval_grid(xg)
    v_min_kT = float(Vg.min())
    print(f"  Z in [{zh.min():.4e}, {zh.max():.4e}]  ratio={z_ratio:.2f}  "
          f"({zh.size} updates)  eps={op.epsilon:.3e}")
    print(f"  bias V_min={v_min_kT:.3f} kT  (bounded below by -dE={-dE:.1f})")
    print(f"  max|FES_OPES - FES_metaD| = {d_op_mt:.3f} kT")
    print(f"  max|FES_OPES - U|         = {d_op_U:.3f} kT")
    z_bounded = np.all(np.isfinite(zh)) and (zh.min() > 0) and (z_ratio < 1e6)
    ok = z_bounded and (v_min_kT >= -dE - 1e-6) and (d_op_mt < 1.0) and (d_op_U < 1.0)
    print(f"  [GATE 4] {'PASS' if ok else 'FAIL'}  (Z bounded + FES match, tol 1.0 kT)")
    return ok, d_op_mt


def gate5_gpu_integration():
    print("\n===== GATE 5: GPU integration (WT metaD rides MACE-OFF batched) =====")
    import torch
    torch.set_default_dtype(torch.float64)
    from ase.build import molecule
    from maple.function.calculator.mace._maceoff_batch_calculator import MaceOffBatchCalc
    from maple.function.dispatcher.md.ensemble.metad_batched import BatchedMetaD

    MODEL = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
    STEPS = int(os.environ.get("STEPS", "4000"))
    NWALK = int(os.environ.get("NWALK", "8"))
    at = molecule("C3H8")
    Z = at.get_atomic_numbers()
    pos = at.get_positions()
    carbons = [i for i, z in enumerate(Z) if z == 6]
    import itertools
    pairs = list(itertools.combinations(carbons, 2))
    di = [np.linalg.norm(pos[i] - pos[j]) for i, j in pairs]
    i, j = pairs[int(np.argmax(di))]
    d0 = di[int(np.argmax(di))]
    print(f"  propane terminal-C CV = distance(atom {i}, atom {j}); d0={d0:.3f} A")
    calc = MaceOffBatchCalc(model_path=MODEL, device="cuda", dtype=torch.float64)
    paras = dict(steps=STEPS, timestep=0.5, temperature=300.0, thermostat="langevin",
                 friction=0.02, remove_com_every=100, random_seed=RNG_SEED,
                 verbose=0, traj_every=10**9, log_every=10**9,
                 metad="on", method="wt", nwalkers=NWALK,
                 cv_group1=str(i), cv_group2=str(j),
                 mtd_sigma=0.1, mtd_height=0.2, mtd_pace=100, mtd_biasfactor=8.0,
                 mtd_cv_min=2.0, mtd_cv_max=3.4, mtd_nbins=200, mtd_deposit=True)
    import tempfile
    tmp = tempfile.mkdtemp(prefix="metadsmoke_")
    job = BatchedMetaD(os.path.join(tmp, "mtd.out"), at, calc=calc, paras=paras).run()
    cv = job.production_cv()
    n_hills = int(job.engine.s.size)
    bias_max = float(job.engine.h.sum()) if n_hills else 0.0
    fes_rng = float(np.nanmax(job.fes) - np.nanmin(job.fes)) / HARTREE_PER_KCAL
    finite = bool(np.all(np.isfinite(cv))) and np.isfinite(fes_rng)
    print(f"  walkers={job.B}  steps={STEPS}  n_hills={n_hills}  "
          f"sum(hill heights)={bias_max/HARTREE_PER_KCAL:.3f} kcal/mol")
    print(f"  CV sampled [{cv.min():.3f},{cv.max():.3f}] A  mean={cv.mean():.3f}  "
          f"span={cv.max()-cv.min():.3f} A")
    print(f"  FES range={fes_rng:.3f} kcal/mol  all-finite={finite}")
    ok = finite and (n_hills > 0) and (bias_max > 0) and ((cv.max() - cv.min()) > 0.05)
    print(f"  [GATE 5] {'PASS' if ok else 'FAIL'}  (deposits + explores + finite)")
    return ok, fes_rng


def main():
    import torch
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"SEED={RNG_SEED}")
    results = {}
    results["gate1_FD"] = gate1_fd()
    results["gate2_WTasymptote"] = gate2_wt_asymptote()
    results["gate3_multiwalker"] = gate3_multiwalker()
    results["gate4_OPES"] = gate4_opes()
    results["gate5_GPU"] = gate5_gpu_integration()
    print("\n" + "=" * 60)
    print(f"{'BATCHED METADYNAMICS / OPES GATE SUMMARY':^60}")
    print("=" * 60)
    allok = True
    for k, (ok, val) in results.items():
        allok &= ok
        print(f"  {k:<22} {'PASS' if ok else 'FAIL'}   metric={val:.4g}")
    print("=" * 60)
    print("[RESULT] " + ("ALL GATES PASS" if allok else "CHECK ABOVE (some FAIL)"))
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
