"""
In-tree conformance + smoke gate for the first-class batched eABF ensemble
wrapper (``ensemble/eabf_batched.py``: ``BatchedEABF`` / ``ExtendedABFParams``).

The eABF ALGORITHM (spring forces, extended-Lagrangian equilibrium, CZAR PMF)
is already validated by ``bias/_test_eabf.py`` (spring-FD 9.1e-11, CZAR-vs-WHAM
0.84 kT, flatten). This test does NOT re-validate that math; it validates the
WRAPPER: (a) the CZAR PMF math that ``run()`` delegates to is finite on the toy
driver, (b) the param -> ``ExtendedABF`` plumbing maps 1:1 and reduces sanely
(k->0 spring vanishes), (c) ``BatchedEABF`` conforms to the ensemble ABC
(subclass + Params + defaults), and (d) a real ``BatchedEABF(...).run()``
produces a finite CZAR PMF with lambda tracking xi.

Gates A1/A2 are CALC-FREE and torch-free (import only ``bias.eabf``, which has no
module-level torch), so they ALWAYS run. Gate S (structural: importing the
wrapper) and Gate B (real MACE-OFF pipeline) auto-SKIP when torch / the model are
unavailable, mirroring ``bias/_test_eabf.py``'s Gate B and ``_smoke_gamd_batched``.

Deterministic (fixed seeds). Run as a script (never on import).
"""
import os
import sys

import numpy as np
from ase import Atoms

# bias.eabf is torch-free at module level -> the calc-free gates always import.
from maple.function.dispatcher.md.bias.eabf import (
    ExtendedABF, eabf_toy_1d, czar_pmf,
)
from maple.function.dispatcher.md.bias.gamd import KB_HA_PER_K

if __name__ != "__main__":
    raise SystemExit("run _test_eabf_batched.py as a script, not an import")


# ---------------------------------------------------------------- Gate A1
def gate_run_delegated_czar():
    """The CZAR PMF that BatchedEABF.run() delegates to (``czar_pmf`` via the toy
    driver) must return a FINITE PMF with a positive barrier on a double well."""
    kT = 1.0
    a_w = 1.0
    h_b = 4.0 * kT
    U = lambda x: h_b * ((x / a_w) ** 2 - 1.0) ** 2                    # noqa: E731
    dU = lambda x: h_b * 2.0 * ((x / a_w) ** 2 - 1.0) * (2.0 * x / a_w ** 2)  # noqa: E731
    res = eabf_toy_1d(U, dU, x0=-a_w, m_x=1.0, m_lam=1.0, k=50.0, kT=kT,
                      dt=0.0015, gamma_x=2.0, gamma_lam=2.0, cv_min=-1.5,
                      cv_max=1.5, nbins=40, nsteps=80000, full_samples=100,
                      apply_abf=True, seed=0)
    pmf = res["czar_pmf"]
    finite = np.isfinite(pmf)
    barrier = float(np.nanmax(pmf) - np.nanmin(pmf)) if finite.any() else 0.0
    ok = bool(finite.any()) and (barrier > 0.5 * kT)
    print(f"[eABF-BATCHED/CZAR] finite bins={int(finite.sum())}/{pmf.size}  "
          f"barrier={barrier:.3f} kT  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- Gate A2
def gate_param_plumbing_and_reduction():
    """The wrapper builds ExtendedABF from ExtendedABFParams via attach_eabf; verify
    the param->bias mapping (k/nbins/shared_grid/m_lam) and a k->0 reduction (spring
    force vanishes -> passive logger). CALC-FREE: ExtendedABF.__init__ is pure numpy."""
    at = [Atoms("H2", positions=[[0.0, 0, 0], [2.5, 0, 0]]) for _ in range(3)]
    kT = KB_HA_PER_K * 300.0
    # shared-grid multi-walker eABF: one grid across all replicas.
    bias = ExtendedABF(at, "0", "1", k=2.0, kT=kT, dt_fs=0.5, cv_min=1.9,
                       cv_max=3.3, nbins=40, full_samples=100, lam_tau_fs=50.0,
                       lam_friction=0.02, shared_grid=True, apply_abf=True)
    map_ok = (bias.B == 3 and bias.k == 2.0 and bias.nbins == 40
              and bias.shared_grid and bias.m_lam > 0.0
              and (bias._grids[0] is bias._grids[1] is bias._grids[2]))
    # spring force: with k>0 the extended-var force = k(xi-lambda) (+ABF; off here).
    b_spring = ExtendedABF(at, "0", "1", k=2.0, kT=kT, dt_fs=0.5, cv_min=1.9,
                           cv_max=3.3, nbins=40, apply_abf=False)
    f_spring = b_spring._lambda_force(0, xi=3.0, lam=2.0)             # expect 2.0*(3-2)=2.0
    # k->0 reduction: spring vanishes -> zero force (needs explicit lam_mass since
    # m_lam = k(tau/2pi)^2 would be 0 otherwise). apply_abf off -> pure logger.
    b_zero = ExtendedABF(at, "0", "1", k=0.0, kT=kT, dt_fs=0.5, cv_min=1.9,
                         cv_max=3.3, nbins=40, apply_abf=False, lam_mass=1.0)
    f_zero = b_zero._lambda_force(0, xi=3.0, lam=2.0)                 # expect 0.0
    red_ok = (abs(f_spring - 2.0) < 1e-12) and (abs(f_zero) < 1e-12)
    ok = map_ok and red_ok
    print(f"[eABF-BATCHED/PLUMBING] param->bias map={map_ok}  "
          f"f_spring={f_spring:.4f}(exp 2.0) f_k0={f_zero:.4f}(exp 0.0)  "
          f"reduction={red_ok}  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- Gate S
def gate_structural_conformance():
    """BatchedEABF conforms to the ensemble ABC: subclass of BatchedNVT, Params
    subclass of BatchedNVTParams, defaults track ExtendedABF. SKIP if torch absent
    (importing the wrapper pulls the batched NVT kernel -> torch)."""
    try:
        from maple.function.dispatcher.md.ensemble.eabf_batched import (
            BatchedEABF, ExtendedABFParams)
        from maple.function.dispatcher.md.ensemble.nvt_batched import (
            BatchedNVT, BatchedNVTParams)
    except Exception as exc:                             # torch absent locally
        print(f"[eABF-BATCHED/STRUCT] SKIP (deps unavailable: "
              f"{type(exc).__name__}: {exc})")
        return None
    sub_ok = (issubclass(BatchedEABF, BatchedNVT)
              and issubclass(ExtendedABFParams, BatchedNVTParams))
    d = ExtendedABFParams()                              # defaults track ExtendedABF
    def_ok = (d.cv_type == "distance" and d.k == 2.0 and d.lam_tau_fs == 100.0
              and d.lam_friction == 0.01 and d.n_bins == 100
              and d.full_samples == 200 and d.shared_grid is False
              and d.apply_abf is True and d.lam0 is None and d.lam_mass is None)
    exported = ("eabf" in BatchedEABF._ALIASES)
    ok = sub_ok and def_ok and exported
    print(f"[eABF-BATCHED/STRUCT] subclass={sub_ok} defaults={def_ok} "
          f"alias={exported}  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- Gate B (a100)
def gate_batched_pipeline(steps=400, dt=0.5, seed=5):
    """Real BatchedEABF on a MACE-OFF molecule: finite CZAR PMF + each walker's
    <lambda> tracks its <xi> (the spring holds). SKIP when torch / model absent."""
    try:
        import tempfile
        import torch
        torch.set_default_dtype(torch.float64)
        from ase.build import molecule
        from maple.function.calculator.mace._maceoff_batch_calculator import (
            MaceOffBatchCalc)
        from maple.function.dispatcher.md.ensemble.eabf_batched import BatchedEABF
    except Exception as exc:                             # torch / mace absent locally
        print(f"[eABF-BATCHED/PIPELINE] SKIP (deps unavailable: "
              f"{type(exc).__name__}: {exc})")
        return None

    MODEL = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
    if not os.path.exists(MODEL):
        print(f"[eABF-BATCHED/PIPELINE] SKIP (model not found: {MODEL})")
        return None

    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    et = molecule("CH3CH2OH")                            # COM-COM CV: atom0 (C) .. atom2 (O)
    calc = MaceOffBatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64)
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        out = f.name
    paras = dict(
        # NVT block
        steps=steps, timestep=dt, temperature=300.0, thermostat="langevin",
        friction=0.02, random_seed=seed, remove_com_every=100, verbose=0,
        traj_every=10 ** 9, log_every=10 ** 9,
        # eABF block (goes through the SAME ensemble entry)
        cv_type="distance", cv_group1="0", cv_group2="2", n_walkers=4,
        k=2.0, cv_min=1.9, cv_max=3.3, n_bins=40, full_samples=100,
        lam_tau_fs=50.0, lam_friction=0.02, shared_grid=True)
    job = BatchedEABF(out, et, calc=calc, paras=paras).run()

    pmf = job.pmf
    finite = bool(pmf is not None and np.any(np.isfinite(pmf)))
    maxtrack = float(np.nanmax(np.abs(job.walker_mean_xi - job.walker_mean_lambda)))
    ok = finite and (maxtrack < 0.3)
    print(f"[eABF-BATCHED/PIPELINE] B={job.B} finite_CZAR_PMF={finite}  "
          f"max|<xi>-<lambda>|={maxtrack:.3f} A  PMF_range="
          f"{float(np.nanmax(pmf) - np.nanmin(pmf)):.6f} Ha")
    print(f"[eABF-BATCHED/PIPELINE] {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    okA1 = gate_run_delegated_czar()
    okA2 = gate_param_plumbing_and_reduction()
    okS = gate_structural_conformance()
    okB = gate_batched_pipeline()

    calcfree_ok = okA1 and okA2
    print(f"\n[eABF-BATCHED RESULT] calc-free gates: czar={okA1} plumbing={okA2}")
    allok = calcfree_ok
    for name, g in (("struct", okS), ("pipeline", okB)):
        if g is None:
            print(f"[eABF-BATCHED RESULT] {name} gate SKIPPED (torch/model absent; "
                  "verify on an a100)")
        else:
            print(f"[eABF-BATCHED RESULT] {name} gate: {g}")
            allok = allok and g
    print(f"[eABF-BATCHED RESULT] "
          f"{'ALL RUNNABLE GATES PASS' if allok else 'SOME FAILED'}")
    sys.exit(0 if allok else 1)
