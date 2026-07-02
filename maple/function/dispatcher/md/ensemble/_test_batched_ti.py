"""
In-tree conformance + smoke gate for the first-class batched Thermodynamic
Integration + MBAR ensemble wrapper (``ensemble/ti_batched.py``: ``BatchedTI`` /
``BatchedTIParams``) and its estimator/bias layer (``bias/ti_mbar.py``,
``bias/lambda_mix.py``).

Four gates (task spec):
  1. **TI == MBAR** on the SAME collected data (calc-free, numpy): the trapezoid TI
     and the MBAR fixed-point agree within statistical tolerance.
  2. **Analytic match** (calc-free, numpy): two shifted harmonics with a KNOWN
     dF = 1/2 kT ln(kB/kA); TI and MBAR both match it.
  3. **One-forward-over-lambda** (real BatchedTI on MACE-OFF): ALL N_lambda windows
     advance in ONE get_ef_gpu per step (the batch axis) -- verified by counting the
     forwards (== steps + 1, independent of B) and that every window logged one
     dU/dlambda per forward. Per-window lambda is applied (the grid is on the bias).
  4. **Endpoints** (calc-free, numpy): the linear coupling gives lambda=0 -> E_A,
     lambda=1 -> E_B (energy AND force).

Gates 1/2/4 are CALC-FREE and torch-free (they import only ``bias.ti_mbar`` /
``bias.lambda_mix``, which carry no module-level torch), so they ALWAYS run. Gate S
(structural: importing the wrapper) and Gate 3 (real MACE-OFF pipeline) auto-SKIP
when torch / the model are unavailable, mirroring ``_test_eabf_batched.py``.

Deterministic (fixed seeds). Run as a script (never on import).
"""
import os
import sys

import numpy as np

# torch-free at module level -> the calc-free gates always import.
from maple.function.dispatcher.md.bias.ti_mbar import (
    ti_integrate, mbar_delta_f, bar_2state, linear_mix_energy, linear_mix_force,
)
from maple.function.dispatcher.md.bias.lambda_mix import HarmonicEndState

if __name__ != "__main__":
    raise SystemExit("run _test_batched_ti.py as a script, not an import")


# --------------------------------------------------------- shared synthetic data
def _make_harmonic_samples(kA=1.0, kB=4.0, a=0.0, b=1.0, kT=1.0, nlam=11,
                           M=40000, seed=0):
    """Draw equilibrium samples for the linear mix of two shifted 1-D harmonics.

    Returns (lambdas, EA_list, EB_list, dudl_means, dF_analytic). The mixed
    potential is harmonic with spring K_l=(1-l)kA+l*kB, so each window's
    equilibrium is exactly N(mu_l, sqrt(kT/K_l))."""
    rng = np.random.RandomState(seed)
    lambdas = np.linspace(0.0, 1.0, nlam)
    EA_list, EB_list, dudl_means = [], [], []
    for lam in lambdas:
        K_l = (1.0 - lam) * kA + lam * kB
        mu = ((1.0 - lam) * kA * a + lam * kB * b) / K_l
        sig = np.sqrt(kT / K_l)
        x = mu + sig * rng.randn(M)
        EA = 0.5 * kA * (x - a) ** 2
        EB = 0.5 * kB * (x - b) ** 2
        EA_list.append(EA)
        EB_list.append(EB)
        dudl_means.append(float(np.mean(EB - EA)))
    dF_analytic = 0.5 * kT * np.log(kB / kA)
    return lambdas, EA_list, EB_list, np.array(dudl_means), dF_analytic


# ---------------------------------------------------------------- Gate 1
def gate_ti_equals_mbar():
    """TI and MBAR on the SAME data agree within statistical tolerance."""
    kT = 1.0
    lam, EA, EB, dudl_m, _ = _make_harmonic_samples(kT=kT, seed=1)
    dF_ti = ti_integrate(lam, dudl_m)
    dF_mbar, _f = mbar_delta_f(lam, EA, EB, kT)
    agree = abs(dF_ti - dF_mbar)
    ok = agree < 0.03
    print(f"[TI-BATCHED/TI==MBAR] TI={dF_ti:.5f}  MBAR={dF_mbar:.5f}  "
          f"|TI-MBAR|={agree:.5f} (tol 0.03)  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- Gate 2
def gate_analytic_match():
    """Toy with analytic dF=1/2 kT ln(kB/kA): TI and MBAR both match it.
    Also cross-checks BAR (K=2, endpoints) as the MBAR->BAR reduction."""
    kT = 1.0
    lam, EA, EB, dudl_m, dF_an = _make_harmonic_samples(kT=kT, seed=2)
    dF_ti = ti_integrate(lam, dudl_m)
    dF_mbar, _f = mbar_delta_f(lam, EA, EB, kT)
    dF_bar = bar_2state(EB[0] - EA[0], EA[-1] - EB[-1], kT)   # endpoints only
    e_ti, e_mbar, e_bar = (abs(dF_ti - dF_an), abs(dF_mbar - dF_an),
                           abs(dF_bar - dF_an))
    ok = (e_ti < 0.03) and (e_mbar < 0.03) and (e_bar < 0.05)
    print(f"[TI-BATCHED/ANALYTIC] analytic={dF_an:.5f}  TI={dF_ti:.5f}(e={e_ti:.4f})"
          f"  MBAR={dF_mbar:.5f}(e={e_mbar:.4f})  BAR={dF_bar:.5f}(e={e_bar:.4f})  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- Gate 4
def gate_endpoints():
    """Linear coupling: lambda=0 -> E_A/F_A, lambda=1 -> E_B/F_B (energy + force).
    E_A/F_A are arbitrary; E_B/F_B come from the toy HarmonicEndState so the whole
    coupling path (end-state B evaluator + mix) is exercised, CALC-FREE."""
    ref = [np.array([[0.0, 0, 0], [1.0, 0, 0]])]
    es = HarmonicEndState(ref, kappa=3.0)
    coord = np.array([[0.0, 0, 0], [1.4, 0, 0]])            # atom1 displaced +0.4x
    E_B, fB = es.energy_forces(coord, np.array([0, 2]))
    E_A = np.array([2.5])                                    # arbitrary base energy
    F_A = np.array([[0.7, 0.0, 0.0], [-0.7, 0.0, 0.0]])     # arbitrary base force
    # endpoints
    e0 = linear_mix_energy(E_A[0], E_B[0], 0.0)
    e1 = linear_mix_energy(E_A[0], E_B[0], 1.0)
    f0 = linear_mix_force(F_A, fB[0], 0.0)
    f1 = linear_mix_force(F_A, fB[0], 1.0)
    # E_B analytic: 1/2 * 3 * 0.4^2 = 0.24 ; F_B on atom1 x: -3*0.4 = -1.2
    eB_ok = abs(E_B[0] - 0.24) < 1e-12 and abs(fB[0][1, 0] + 1.2) < 1e-12
    end_ok = (abs(e0 - E_A[0]) < 1e-12 and abs(e1 - E_B[0]) < 1e-12
              and np.allclose(f0, F_A) and np.allclose(f1, fB[0]))
    ok = eB_ok and end_ok
    print(f"[TI-BATCHED/ENDPOINTS] E_B={E_B[0]:.4f}(exp 0.24) lambda0 E={e0:.4f}"
          f"(exp {E_A[0]:.2f}) lambda1 E={e1:.4f}(exp {E_B[0]:.4f}) forces_ok="
          f"{end_ok}  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- Gate S
def gate_structural_conformance():
    """BatchedTI conforms to the ensemble ABC: subclass of BatchedNVT, Params
    subclass of BatchedNVTParams, sane defaults. SKIP if torch absent (importing the
    wrapper pulls the batched NVT kernel -> torch)."""
    try:
        from maple.function.dispatcher.md.ensemble.ti_batched import (
            BatchedTI, BatchedTIParams)
        from maple.function.dispatcher.md.ensemble.nvt_batched import (
            BatchedNVT, BatchedNVTParams)
    except Exception as exc:
        print(f"[TI-BATCHED/STRUCT] SKIP (deps unavailable: "
              f"{type(exc).__name__}: {exc})")
        return None
    sub_ok = (issubclass(BatchedTI, BatchedNVT)
              and issubclass(BatchedTIParams, BatchedNVTParams))
    d = BatchedTIParams()
    def_ok = (d.ti_nwindows == 8 and d.ti_lambda_min == 0.0
              and d.ti_lambda_max == 1.0 and d.ti_kappa == 2.0
              and d.ti_discard_frac == 0.2)
    exported = ("ti" in BatchedTI._ALIASES)
    ok = sub_ok and def_ok and exported
    print(f"[TI-BATCHED/STRUCT] subclass={sub_ok} defaults={def_ok} "
          f"alias={exported}  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- Gate 3 (a100)
def gate_one_forward_pipeline(steps=200, dt=0.5, seed=5, nwin=6):
    """Real BatchedTI on a MACE-OFF molecule: ALL N_lambda windows advance in ONE
    get_ef_gpu per step. Verify (a) #forwards == steps+1 (independent of B -> the
    lambda windows share ONE batched forward), (b) B == N_lambda, (c) every window
    logged one dU/dlambda per forward, (d) finite TI and MBAR dF. SKIP when torch /
    model absent."""
    try:
        import tempfile
        import torch
        torch.set_default_dtype(torch.float64)
        from ase.build import molecule
        from maple.function.calculator.mace._maceoff_batch_calculator import (
            MaceOffBatchCalc)
        from maple.function.dispatcher.md.ensemble.ti_batched import BatchedTI
    except Exception as exc:
        print(f"[TI-BATCHED/PIPELINE] SKIP (deps unavailable: "
              f"{type(exc).__name__}: {exc})")
        return None

    MODEL = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
    if not os.path.exists(MODEL):
        print(f"[TI-BATCHED/PIPELINE] SKIP (model not found: {MODEL})")
        return None

    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    et = molecule("CH3CH2OH")
    calc = MaceOffBatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64)
    # count get_ef_gpu calls: each call advances ALL B lambda windows at once.
    n_forward = {"c": 0}
    _orig = calc.get_ef_gpu
    def _counting():
        n_forward["c"] += 1
        return _orig()
    calc.get_ef_gpu = _counting

    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        out = f.name
    paras = dict(
        steps=steps, timestep=dt, temperature=300.0, thermostat="langevin",
        friction=0.02, random_seed=seed, remove_com_every=100, verbose=0,
        traj_every=10 ** 9, log_every=10 ** 9,
        ti_nwindows=nwin, ti_lambda_min=0.0, ti_lambda_max=1.0,
        ti_kappa=1.0, ti_discard_frac=0.2)
    job = BatchedTI(out, et, calc=calc, paras=paras).run()

    # (a) one batched forward per step (+1 cached forward at t=0), regardless of B.
    one_forward = (n_forward["c"] == steps + 1)
    # (b) B == N_lambda ; (c) every window advanced once per forward.
    bwin_ok = (job.B == nwin)
    per_win_len = [len(job._bias.dudl_history[b]) for b in range(job.B)]
    all_advanced = all(L == n_forward["c"] for L in per_win_len)
    # (d) finite dF both ways.
    finite = bool(np.isfinite(job.dF_ti) and np.isfinite(job.dF_mbar))
    ok = one_forward and bwin_ok and all_advanced and finite
    print(f"[TI-BATCHED/PIPELINE] B={job.B} forwards={n_forward['c']} "
          f"(exp {steps+1}) one_forward_over_lambda={one_forward} "
          f"per_win_logs={per_win_len[:3]}... all_advanced={all_advanced}")
    print(f"[TI-BATCHED/PIPELINE] dF_TI={job.dF_ti:.6f} Ha  dF_MBAR={job.dF_mbar:.6f}"
          f" Ha  |TI-MBAR|={abs(job.dF_ti-job.dF_mbar):.6f}  finite={finite}")
    print(f"[TI-BATCHED/PIPELINE] {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ok1 = gate_ti_equals_mbar()
    ok2 = gate_analytic_match()
    ok4 = gate_endpoints()
    okS = gate_structural_conformance()
    ok3 = gate_one_forward_pipeline()

    calcfree_ok = ok1 and ok2 and ok4
    print(f"\n[TI-BATCHED RESULT] calc-free gates: TI==MBAR={ok1} analytic={ok2} "
          f"endpoints={ok4}")
    allok = calcfree_ok
    for name, g in (("struct", okS), ("one-forward-pipeline", ok3)):
        if g is None:
            print(f"[TI-BATCHED RESULT] {name} gate SKIPPED (torch/model absent; "
                  "verify on an a100)")
        else:
            print(f"[TI-BATCHED RESULT] {name} gate: {g}")
            allok = allok and g
    print(f"[TI-BATCHED RESULT] "
          f"{'ALL RUNNABLE GATES PASS' if allok else 'SOME FAILED'}")
    sys.exit(0 if allok else 1)
