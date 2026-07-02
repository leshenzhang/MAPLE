"""
In-tree correctness gate for extended-system ABF (eABF) + the CZAR estimator
(``bias/eabf.py``).

Validation axis = ALGORITHM CORRECTNESS of the eABF machinery (spring forces,
extended-Lagrangian equilibrium, mean-force flattening, CZAR PMF reconstruction),
NOT a literature free-energy number -- exactly as ``_test_umbrella_wham.py``
validates the WHAM path. Gates 1-4 are CALC-FREE and deterministic (fixed seeds);
they run the SAME production code (``ExtendedABF`` spring/CV kernel, ``ABFGrid``,
``czar_pmf``) via the numpy toy driver ``eabf_toy_1d`` -- no torch/MLIP/GPU needed.
Gate B smoke-tests the real batched pipeline on a MACE-OFF molecule (a100; auto
SKIP when torch / the model are unavailable).

  1. SPRING-FORCE CORRECTNESS.  Analytic spring force ``-k(xi-lambda) dxi/dx`` vs a
     central finite difference of ``U_ext = 1/2 k (xi-lambda)^2`` on the COM-COM
     distance CV (lambda fixed) -- must agree to < 1e-6 (biasing force is exact).

  2. STIFF-SPRING LIMIT.  On the extended Lagrangian at equilibrium (ABF off,
     free physical coord), ``<(xi-lambda)^2>`` must (a) decrease monotonically as
     ``k`` grows and (b) track the equipartition value ``kT/k`` (var*k ~ O(1)) --
     i.e. lambda tracks xi ever more tightly. AND the eABF (ABF-on) CZAR PMF must
     recover the physical PMF (checked jointly with gate 3).

  3. CZAR PMF PARITY.  On a 1-D double well, the CZAR-estimated PMF must match an
     INDEPENDENT umbrella-sampling + WHAM reference (reusing the existing
     ``bias.umbrella.wham_1d``) AND the analytic true PMF, both to kT-level.

  4. FLATTENING.  With ABF on, the sampling along lambda becomes ~uniform: the
     lambda histogram is flatter (higher entropy, lower coefficient of variation,
     fuller coverage) than the ABF-off run trapped in the wells -- the ABF force
     is doing its job.

Deterministic (fixed seeds). Run as a script (never on import).
"""
import os
import sys

import numpy as np
from ase import Atoms

from maple.function.dispatcher.md.bias.eabf import (
    ExtendedABF, ABFGrid, czar_pmf, eabf_toy_1d, umbrella_wham_reference,
    attach_eabf, _com_distance_cv_grad,
)

if __name__ != "__main__":
    raise SystemExit("run _test_eabf.py as a script, not an import")


# --------------------------------------------------------------- toy double well
KT = 1.0                     # reduced units (kT = 1)
H_BAR = 4.0                  # barrier height ~ 4 kT
A_W = 1.0                    # well positions +/- A_W
CV_MIN, CV_MAX, NB = -1.5, 1.5, 40


def U(x):
    return H_BAR * ((x / A_W) ** 2 - 1.0) ** 2


def dU(x):
    return H_BAR * 2.0 * ((x / A_W) ** 2 - 1.0) * (2.0 * x / A_W ** 2)


def _interior_maxdev(p, q, x, lo=-1.2, hi=1.2):
    """max|p-q| over the well-sampled interior, after removing the additive const."""
    m = np.isfinite(p) & np.isfinite(q) & (x >= lo) & (x <= hi)
    d = (p - q)[m]
    if d.size < 2:
        return float("inf"), int(m.sum())
    d = d - d.mean()
    return float(np.max(np.abs(d))), int(m.sum())


# ---------------------------------------------------------------- Gate 1
def gate_spring_force_fd():
    """Analytic spring force vs FD of U_ext on the COM-COM distance CV."""
    a = Atoms("H2", positions=[[0.0, 0, 0], [2.5, 0, 0]])
    m = a.get_masses()
    g1, g2 = np.array([0]), np.array([1])
    k, lam = 1.5, 2.0                                     # Ha/A^2, fixed lambda
    pos = a.get_positions()
    xi0, grad = _com_distance_cv_grad(pos, g1, g2, m)
    f_analytic = -(k * (xi0 - lam)) * grad               # (n,3) Ha/A
    h = 1e-6
    f_fd = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for c in range(3):
            pp = pos.copy(); pp[i, c] += h
            xp, _ = _com_distance_cv_grad(pp, g1, g2, m)
            pm = pos.copy(); pm[i, c] -= h
            xm, _ = _com_distance_cv_grad(pm, g1, g2, m)
            f_fd[i, c] = -(0.5 * k * (xp - lam) ** 2
                           - 0.5 * k * (xm - lam) ** 2) / (2 * h)
    res = float(np.max(np.abs(f_analytic - f_fd)))
    ok = res < 1e-6
    print(f"[eABF/SPRING-FD] max|F_analytic - F_fd| = {res:.2e} (tol 1e-6)  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok, res


# ---------------------------------------------------------------- Gate 2
def gate_stiff_spring():
    """Extended-Lagrangian equilibrium (ABF off): <(xi-lambda)^2> -> kT/k, and
    monotonically smaller as k grows (lambda tracks xi more tightly)."""
    Uz = lambda x: 0.0 * x                               # noqa: E731 (free coord)
    dUz = lambda x: 0.0 * x                              # noqa: E731
    ks = (40.0, 160.0, 640.0)
    var = []
    for k in ks:
        r = eabf_toy_1d(Uz, dUz, x0=0.0, m_x=1.0, m_lam=1.0, k=k, kT=KT,
                        dt=0.001, gamma_x=2.0, gamma_lam=2.0, cv_min=-4.0,
                        cv_max=4.0, nbins=40, nsteps=150000, apply_abf=False,
                        seed=1, equil_frac=0.2)
        var.append(float((r["xi"] - r["lam"]).var()))
    var = np.asarray(var)
    monotone = bool(np.all(np.diff(var) < 0.0))
    vk = var * np.asarray(ks)                            # var*k ~ O(1) (equipartition)
    equipart = bool(np.all((vk > 0.7) & (vk < 1.8)))
    ok = monotone and equipart
    print(f"[eABF/STIFF-SPRING] k={list(ks)}  var(xi-lambda)="
          f"{np.round(var, 5).tolist()}  var*k={np.round(vk, 3).tolist()}  "
          f"(exp ~ kT/k, monotone down)")
    print(f"[eABF/STIFF-SPRING] monotone={monotone} equipartition={equipart}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok, var


# ---------------------------------------------------------------- Gate 3
def gate_czar_parity():
    """CZAR PMF (double well) vs umbrella+WHAM reference AND analytic true PMF."""
    res = eabf_toy_1d(U, dU, x0=-A_W, m_x=1.0, m_lam=1.0, k=50.0, kT=KT,
                      dt=0.0015, gamma_x=2.0, gamma_lam=2.0, cv_min=CV_MIN,
                      cv_max=CV_MAX, nbins=NB, nsteps=250000, full_samples=100,
                      apply_abf=True, seed=0)
    xc, czar = res["czar_x"], res["czar_pmf"]
    xw, wham = umbrella_wham_reference(U, dU, cv_min=CV_MIN, cv_max=CV_MAX,
                                       nwin=13, kappa=80.0, m_x=1.0, kT=KT,
                                       dt=0.0015, gamma=2.0, nsteps=20000,
                                       nbins=NB, seed=100)
    Utrue = U(xc)
    Utrue = Utrue - np.nanmin(Utrue)
    d_cw, n = _interior_maxdev(czar, wham, xc)           # CZAR vs WHAM
    d_cu, _ = _interior_maxdev(czar, Utrue, xc)          # CZAR vs analytic
    d_wu, _ = _interior_maxdev(wham, Utrue, xw)          # WHAM vs analytic (sanity)
    b_czar = float(np.nanmax(czar) - np.nanmin(czar))
    b_wham = float(np.nanmax(wham) - np.nanmin(wham))
    ok = (d_cw < 1.5 * KT) and (d_cu < 1.5 * KT) and (d_wu < 1.2 * KT)
    print(f"[eABF/CZAR] barrier CZAR={b_czar:.2f} WHAM={b_wham:.2f} (true~4.0) kT")
    print(f"[eABF/CZAR] max|CZAR-WHAM|={d_cw:.3f}  max|CZAR-U|={d_cu:.3f}  "
          f"max|WHAM-U|={d_wu:.3f} kT (interior n={n}, tol ~1.5 kT)")
    print(f"[eABF/CZAR] {'PASS' if ok else 'FAIL'}")
    return ok, d_cw


# ---------------------------------------------------------------- Gate 4
def gate_flattening():
    """ABF on -> lambda sampling ~uniform (flatter than the ABF-off trapped run)."""
    edges = np.linspace(CV_MIN, CV_MAX, NB + 1)

    def flat(apply_abf):
        r = eabf_toy_1d(U, dU, x0=-A_W, m_x=1.0, m_lam=1.0, k=50.0, kT=KT,
                        dt=0.0015, gamma_x=2.0, gamma_lam=2.0, cv_min=CV_MIN,
                        cv_max=CV_MAX, nbins=NB, nsteps=250000, full_samples=100,
                        apply_abf=apply_abf, seed=0)
        hist, _ = np.histogram(r["lam"], bins=edges)
        p = hist / max(hist.sum(), 1)
        nz = p[p > 0]
        entropy = float(-(nz * np.log(nz)).sum() / np.log(NB))    # 1.0 = uniform
        cov = int((hist > 0).sum())
        cvar = float(hist.std() / max(hist.mean(), 1e-30))
        return entropy, cov, cvar

    ent_on, cov_on, cv_on = flat(True)
    ent_off, cov_off, cv_off = flat(False)
    ok = (ent_on > ent_off) and (cov_on >= cov_off) and (cv_on < cv_off)
    print(f"[eABF/FLATTEN] ABF ON : entropy={ent_on:.3f} coverage={cov_on}/{NB} "
          f"hist_CV={cv_on:.3f}")
    print(f"[eABF/FLATTEN] ABF OFF: entropy={ent_off:.3f} coverage={cov_off}/{NB} "
          f"hist_CV={cv_off:.3f}")
    print(f"[eABF/FLATTEN] flatter-with-ABF={ok}  {'PASS' if ok else 'FAIL'}")
    return ok, (ent_on, ent_off)


# ---------------------------------------------------------------- Gate B (a100)
def gate_batched_pipeline(steps=400, dt=0.5, seed=5):
    """Smoke: ExtendedABF ridden on a real BatchedNVT + MACE-OFF molecule.
    Assert a finite CZAR PMF and that each replica's lambda tracks its xi (the
    spring holds). SKIP (return None) when torch / the model are unavailable."""
    try:
        import torch
        torch.set_default_dtype(torch.float64)
        from ase.build import molecule
        from maple.function.calculator.mace._maceoff_batch_calculator import (
            MaceOffBatchCalc)
        from maple.function.dispatcher.md.ensemble.nvt_batched import BatchedNVT
    except Exception as exc:                             # torch / mace absent locally
        print(f"[eABF/PIPELINE] SKIP (deps unavailable: {type(exc).__name__}: {exc})")
        return None, None

    MODEL = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
    if not os.path.exists(MODEL):
        print(f"[eABF/PIPELINE] SKIP (model not found: {MODEL})")
        return None, None

    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    import tempfile
    et = molecule("CH3CH2OH")                            # COM-COM CV: atom0 (C) .. atom2 (O)
    B = 4
    calc = MaceOffBatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64)
    replicas = [et.copy() for _ in range(B)]
    paras = dict(steps=steps, timestep=dt, temperature=300.0,
                 thermostat="langevin", friction=0.02, random_seed=seed,
                 remove_com_every=100, verbose=0)
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        out = f.name
    job = BatchedNVT(out, replicas, calc=calc, paras=paras)
    bias = attach_eabf(job, "0", "2", k=2.0, cv_min=1.9, cv_max=3.3, nbins=40,
                       full_samples=100, lam_tau_fs=50.0, lam_friction=0.02,
                       shared_grid=True, seed=seed)
    job.run()

    xc, pmf = bias.czar()
    finite = bool(np.any(np.isfinite(pmf)))
    # each replica's <lambda> should track its <xi> (spring holds them together).
    dtrack = []
    for b in range(B):
        xi = np.asarray(bias.xi_history[b][len(bias.xi_history[b]) // 5:])
        lam = np.asarray(bias.lam_history[b][len(bias.lam_history[b]) // 5:])
        if xi.size:
            dtrack.append(abs(float(xi.mean()) - float(lam.mean())))
    maxtrack = max(dtrack) if dtrack else float("inf")
    ok = finite and (maxtrack < 0.3)                     # <lambda> within 0.3 A of <xi>
    print(f"[eABF/PIPELINE] finite_CZAR_PMF={finite}  "
          f"max|<xi>-<lambda>|={maxtrack:.3f} A  PMF_range="
          f"{float(np.nanmax(pmf) - np.nanmin(pmf)):.4f} Ha")
    print(f"[eABF/PIPELINE] {'PASS' if ok else 'FAIL'}")
    return ok, maxtrack


if __name__ == "__main__":
    ok1, _ = gate_spring_force_fd()
    ok2, _ = gate_stiff_spring()
    ok3, _ = gate_czar_parity()
    ok4, _ = gate_flattening()
    okB, _ = gate_batched_pipeline()

    numpy_ok = ok1 and ok2 and ok3 and ok4
    print(f"\n[eABF RESULT] calc-free gates: spring-fd={ok1} stiff-spring={ok2} "
          f"czar={ok3} flatten={ok4}")
    if okB is None:
        print("[eABF RESULT] pipeline gate SKIPPED (verify on GPU: "
              "python -m maple.function.dispatcher.md.bias._test_eabf on an a100)")
        allok = numpy_ok
    else:
        print(f"[eABF RESULT] pipeline gate (a100): {okB}")
        allok = numpy_ok and okB
    print(f"[eABF RESULT] {'ALL eABF/CZAR GATES PASS' if allok else 'SOME FAILED'}")
    sys.exit(0 if allok else 1)
