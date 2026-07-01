# -*- coding: utf-8 -*-
"""Parity + benefit test for the GP-surrogate saddle search (``gp_saddle.py``).

TWO independent checks
----------------------
1. ``analytic_self_test()`` -- TORCH-FREE, no MLIP. Drives the pure-numpy GP
   inner loop + acquisition + EXACT-at-convergence gate on a deterministic 2D
   (embedded 3D) quadratic saddle. Proves the surrogate min-mode + acquisition +
   true-force gate work bitwise-deterministically, and that the GP-surrogate
   search reaches the saddle in FAR fewer oracle calls than a plain dimer on the
   SAME analytic PES. Runs anywhere numpy is importable (loads ``gp_saddle.py``
   standalone, bypassing the torch-importing package ``__init__``).

2. ``run_parity(make_calc)`` -- on a real small saddle (HCN <-> HNC, the same
   system as the sibling GSM test; reactant/product build a midpoint TS guess),
   runs ``BatchGPSaddle`` vs the plain ``BatchDimer`` min-mode oracle over the
   SAME batched calculator and asserts (a) the SAME saddle is found (per-structure
   Kabsch-RMSD <= ~1e-2 A, barrier within ~1e-3 Eh) and (b) ``BatchGPSaddle`` used
   FEWER true ``get_ef_gpu`` forwards than the oracle (reports the ratio). Needs a
   GPU batch calc (UMA-omol / MACE-OFF / ANI2x / AIMNet2-decoupled) -- pass a
   ``make_calc`` factory; the ``__main__`` block builds one from the environment
   and skips gracefully if none is available (mirrors the sibling tests).

    from tests_campaign.test_gp_saddle import run_parity, analytic_self_test
    analytic_self_test()                       # always runnable (numpy only)
    run_parity(lambda: ANIBatchCalc(model_path=None, device="cuda"))

Tolerances (real MLIP path): both searches stop at |F| < fmax, so the TS
geometries agree to the force-convergence floor (~1e-2 A) and the barrier to
~1e-3 Eh (fp32 forward + GP-steering noise). Parity here is "same physical
first-order saddle + fewer forwards", NOT bit-exact trajectories (the GP takes a
different path to the same saddle -- that IS the speedup).
"""

from __future__ import annotations

import os
import sys
import math
import importlib.util

import numpy as np
from ase import Atoms

# --- repo importable when run as a bare script --------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ============================================================================ #
#  Load gp_saddle.py STANDALONE (torch-free): bypass the package __init__ chain #
#  (maple/function/dispatcher/ts/__init__.py imports torch). The module's       #
#  guarded relative imports fall back to a shim when loaded this way.           #
# ============================================================================ #
def _load_gp_saddle_standalone():
    path = os.path.join(_ROOT, "maple", "function", "dispatcher", "ts",
                        "algorithm", "gp_saddle.py")
    spec = importlib.util.spec_from_file_location("gp_saddle_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod           # required for dataclass inheritance
    spec.loader.exec_module(mod)
    return mod


# ============================================================================ #
#  Analytic PES: 3D quadratic saddle at the origin, one negative mode (y).      #
#    E = 1/2 (A x^2 - B y^2 + C z^2),  saddle at (0,0,0), neg curvature along y. #
#  Deterministic; the true force is exact -> bitwise-deterministic search.      #
# ============================================================================ #
_A, _B, _C = 1.5, 2.5, 1.0


def _quad_oracle(x):
    """x (1,3) -> (E, F (1,3)) for the quadratic saddle. Counts nothing here."""
    xx, yy, zz = x.reshape(-1)
    E = 0.5 * (_A * xx * xx - _B * yy * yy + _C * zz * zz)
    F = -np.array([[_A * xx, -_B * yy, _C * zz]], dtype=np.float64)
    return float(E), F


def _plain_dimer_analytic(x0, seed, delta=5e-3, rot_iter=10, max_it=400,
                          fmax=1e-3):
    """Reference plain dimer on the analytic oracle (counts oracle calls).
    Rotation curvature via forward-FD HVP of the true gradient (2 grad/rot-iter),
    trust-clipped steepest-descent translation -- the classic per-step cost the GP
    surrogate eliminates."""
    x = x0.reshape(-1).astype(np.float64).copy()
    calls = [0]
    rng = np.random.default_rng(seed)
    N = rng.normal(size=3); N /= np.linalg.norm(N)

    def grad(xf):
        _E, F = _quad_oracle(xf.reshape(1, 3)); calls[0] += 1
        return -F.reshape(-1)

    alpha, C = 0.15, 1.0
    for it in range(max_it):
        for _ in range(rot_iter):
            g0 = grad(x); gd = grad(x + delta * N); Hn = (gd - g0) / delta
            C = float(N @ Hn); Frot = -(Hn - C * N); fr = np.linalg.norm(Frot)
            if fr < 1e-3:
                break
            Th = Frot / fr
            N = N * math.cos(0.5) + Th * math.sin(0.5); N /= np.linalg.norm(N)
        g = grad(x); F = -g
        Fpar = (F @ N) * N; Fperp = F - Fpar
        Ftrans = (Fperp - Fpar) if C < 0 else Fperp
        step = alpha * Ftrans; sm = float(np.max(np.abs(step.reshape(-1, 3))))
        if sm > 0.15:
            step *= 0.15 / sm
        x = x + step
        if float(np.max(np.abs(F.reshape(-1, 3)))) < fmax and C < 0:
            return True, calls[0], x
    return False, calls[0], x


def analytic_self_test(verbose=True):
    """TORCH-FREE proof: the GP surrogate min-mode + acquisition + exact-force
    gate converge to the analytic saddle in far fewer oracle calls than a plain
    dimer, bitwise-deterministically. Raises AssertionError on failure; returns a
    metrics dict."""
    gp_mod = _load_gp_saddle_standalone()
    assert gp_mod.torch is None or True   # module imports with or without torch
    GPSaddle = gp_mod.GPSaddle
    GPSaddleParams = gp_mod.GPSaddleParams
    saddle_converged = gp_mod.saddle_converged

    # ---- gate unit check: exact-at-convergence reads the TRUE force ----
    assert saddle_converged(np.zeros((1, 3)), curvature=-1.0,
                            f_max_th=5e-3, f_rms_th=1e-3) is True
    assert saddle_converged(np.array([[0.1, 0, 0]]), curvature=-1.0) is False, \
        "gate must reject a large true force even at negative curvature"
    assert saddle_converged(np.zeros((1, 3)), curvature=+1.0) is False, \
        "gate must reject a non-saddle (no negative curvature)"

    def _run_gp(x0, seed):
        p = GPSaddleParams(descriptor="cartesian", length_scale=1.0, sigma_f=1.0,
                           noise_e=1e-7, noise_g=1e-6, m_max=16,
                           remove_rigid=False, n_init="random", seed=seed,
                           f_max_th=1e-3, f_rms_th=1e-3, max_acq_step=0.4,
                           step_max=0.15, max_inner=50, rot_max_iter=10)
        E0, F0 = _quad_oracle(x0)
        s = GPSaddle(x0, E0, F0, p)
        calls = 1                                    # the seed forward
        for it in range(100):
            xp, C = s.propose()
            E, F = _quad_oracle(xp); calls += 1
            s.accept(xp, E, F)
            if s.check_converged(F):
                return True, it + 1, calls, xp.reshape(-1), C
        return False, it + 1, calls, xp.reshape(-1), C

    x0 = np.array([[0.35, 0.30, 0.15]])
    seeds = list(range(5))
    ratios, gp_calls_all, results = [], [], []
    for seed in seeds:
        ok, rounds, gp_calls, xf, C = _run_gp(x0, seed)
        assert ok, f"[seed {seed}] GP surrogate search did NOT converge"
        # exact saddle recovery (origin) + genuine first-order saddle
        assert float(np.max(np.abs(xf))) <= 1e-2, \
            f"[seed {seed}] converged geometry {xf} not at the analytic saddle"
        assert C < 0.0, f"[seed {seed}] final curvature {C} is not negative"
        p_ok, plain_calls, _xp = _plain_dimer_analytic(x0, seed)
        assert p_ok, f"[seed {seed}] plain-dimer reference failed to converge"
        ratio = plain_calls / gp_calls
        assert gp_calls < plain_calls, (
            f"[seed {seed}] GP used {gp_calls} >= plain {plain_calls} calls")
        ratios.append(ratio); gp_calls_all.append(gp_calls)
        results.append((seed, rounds, gp_calls, plain_calls, ratio, xf, C))

    # ---- bitwise determinism: identical seed -> identical trajectory ----
    r1 = _run_gp(x0, 0); r2 = _run_gp(x0, 0)
    assert r1[2] == r2[2] and np.array_equal(r1[3], r2[3]), \
        "GP surrogate search is NOT bitwise-deterministic for a fixed seed"

    mean_ratio = float(np.mean(ratios))
    assert mean_ratio > 3.0, (
        f"expected a meaningful true-call reduction; got mean ratio "
        f"{mean_ratio:.1f}x")

    if verbose:
        print("\n=== GP-saddle ANALYTIC self-test (torch-free) ===")
        for seed, rounds, gpc, plc, ratio, xf, C in results:
            print(f"  seed {seed}: GP {gpc:2d} true calls ({rounds} rounds) vs "
                  f"plain-dimer {plc:3d}  -> {ratio:5.1f}x   "
                  f"saddle |x|={np.max(np.abs(xf)):.1e} curv={C:+.3f}")
        print(f"  mean true-force-call reduction : {mean_ratio:.1f}x")
        print(f"  GP true calls (mean)           : {np.mean(gp_calls_all):.1f}")
        print("  bitwise-deterministic          : yes")
        print("  exact-at-convergence gate      : verified (true-force gated)")
        print("  ANALYTIC SELF-TEST OK")
    return {"mean_ratio": mean_ratio, "gp_calls_mean": float(np.mean(gp_calls_all)),
            "per_seed": results}


# ============================================================================ #
#  Molecular regression self-test (TORCH-FREE): an HCN-like anharmonic 3-atom   #
#  distance PES with a REAL barrier, exercised through the INVDIST descriptor.   #
#  This is the case that reproduced the real-UMA ``max_iter`` failure before the #
#  fix (fixed length_scale over-smoothed the invdist GP -> garbage surrogate     #
#  curvature -> the search wandered and never reached low force). Guards the fix: #
#  data-driven ``length_scale="auto"`` + exact-true-force gate with the          #
#  curv_patience fallback -> converge to the TRUE saddle.                        #
# ============================================================================ #
# saddle: rCH=rNH (a=0), rCH+rNH=L, rCN=dCN ; neg curvature along a=rCH-rNH.
_dCN, _w, _L, _kCN, _km, _g = 1.15, 0.35, 2.2, 4.0, 2.5, 6.0


def _hcnlike_energy(x):
    x = x.reshape(3, 3)                                # atoms: H(0), C(1), N(2)
    rCH = np.linalg.norm(x[1] - x[0]); rNH = np.linalg.norm(x[2] - x[0])
    rCN = np.linalg.norm(x[1] - x[2]); a = rCH - rNH
    return (0.5 * _kCN * (rCN - _dCN) ** 2 + 0.25 * _g * (a * a - _w * _w) ** 2
            + 0.5 * _km * (rCH + rNH - _L) ** 2)


def _hcnlike_oracle(x):
    """Black-box like a real MLIP: force = -central-FD gradient of the energy."""
    x = x.reshape(3, 3); h = 1e-6; G = np.zeros((3, 3))
    for i in range(3):
        for k in range(3):
            xp = x.copy(); xp[i, k] += h; xm = x.copy(); xm[i, k] -= h
            G[i, k] = (_hcnlike_energy(xp) - _hcnlike_energy(xm)) / (2 * h)
    return float(_hcnlike_energy(x)), -G


def _true_hessian_index(oracle, x, h=1e-4):
    """# negative modes of the TRUE Hessian at x (FD of the true force) -- the
    ground-truth saddle index used to assert the gate never false-converges."""
    x = np.asarray(x, float).reshape(-1); n = x.size
    H = np.zeros((n, n))
    for j in range(n):
        dp = x.copy(); dp[j] += h; dm = x.copy(); dm[j] -= h
        H[:, j] = ((-oracle(dp)[1].reshape(-1)) - (-oracle(dm)[1].reshape(-1))) / (2 * h)
    w = np.linalg.eigvalsh(0.5 * (H + H.T))
    sc = max(float(np.max(np.abs(w))), 1e-9)
    return int(np.sum(w < -1e-2 * sc))


def _drive_gpsaddle_numpy(gp_mod, oracle, guess, seed, max_outer=100):
    """Faithful TORCH-FREE mirror of BatchGPSaddle.run for ONE structure: warm-up
    seeding + surrogate min-mode propose within a MODEL-QUALITY trust radius +
    accept-always + EXACT-force gate with the surrogate-Hessian INDEX==1
    first-order-saddle test + the true FD-HVP curvature confirmation. Returns
    (converged, rounds, true_calls, xp)."""
    GPSaddle = gp_mod.GPSaddle
    GPSaddleParams = gp_mod.GPSaddleParams
    warmup_displacements = gp_mod.warmup_displacements
    model_quality_trust = gp_mod.model_quality_trust
    _maxatom = gp_mod._maxatom; _rmsatom = gp_mod._rmsatom

    p = GPSaddleParams(); p.seed = seed
    E0, F0 = oracle(guess)
    s = GPSaddle(guess.copy(), E0, F0, p)
    calls = 1
    trust = p.max_acq_step
    rng = np.random.default_rng(seed + 777)
    for d in warmup_displacements(guess, p.n_warmup, p.warmup_delta,
                                  p.remove_rigid, rng):
        gm = guess + d; Ew, Fw = oracle(gm); calls += 1
        s.add_observation(gm, Ew, Fw)
    for it in range(1, max_outer + 1):
        xp, C = s.propose(trust=trust)
        _Ep, gpred = s.gp.predict(xp); Fpred = -gpred.reshape(-1)
        E, F = oracle(xp); calls += 1
        Ff = F.reshape(-1)
        force_ok = (_maxatom(Ff) <= p.f_max_th) and (_rmsatom(Ff) <= p.f_rms_th)
        n_neg = C_true = None
        if force_ok:
            n_neg = s.n_negative_surrogate_modes(xp)
            Np = s.N.reshape(s.n, 3)
            F2 = oracle(xp + p.delta * Np)[1]; calls += 1
            C_true = float(np.sum(-(F2 - F) / p.delta * Np))
        s.add_observation(xp, E, F)
        if s.check_converged(F, n_neg=n_neg, curvature=C_true):
            return True, it, calls, xp
        ferr = (float(np.linalg.norm(Fpred - Ff))
                / (float(np.linalg.norm(Ff)) + 1e-6))
        trust = model_quality_trust(ferr, trust, p.trust_grow, p.trust_shrink,
                                    p.trust_min, p.max_acq_step,
                                    p.trust_err_lo, p.trust_err_hi)
        s.set_center(xp)
    return False, max_outer, calls, None


def molecular_self_test(verbose=True):
    """TORCH-FREE regression guard mirroring the FULL BatchGPSaddle driver on an
    anharmonic HCN-like invdist PES (the case that reproduced the real-UMA
    max_iter). Guards: auto length_scale + warm-up + model-quality trust +
    EXACT-force gate with the surrogate-Hessian INDEX==1 first-order-saddle test.
    Asserts (a) most seeds reach the TRUE bridged saddle (rCH=rNH=1.100, rCN=1.150)
    and (b) ZERO false convergences -- every ``converged`` result is a genuine
    first-order saddle of the TRUE PES (true Hessian index == 1)."""
    gp_mod = _load_gp_saddle_standalone()

    R, P = _build_reaction()
    guess = 0.5 * (R.get_positions() + P.get_positions())
    guess[0, 1] += 0.6                                 # lift H off the C-N axis

    n_true_saddle = 0
    n_false_pos = 0
    results = []
    seeds = list(range(6))
    for seed in seeds:
        ok, rounds, calls, xp = _drive_gpsaddle_numpy(
            gp_mod, _hcnlike_oracle, guess, seed)
        if ok:
            xx = xp.reshape(3, 3)
            r = (float(np.linalg.norm(xx[1] - xx[0])),
                 float(np.linalg.norm(xx[2] - xx[0])),
                 float(np.linalg.norm(xx[1] - xx[2])))
            is_hcn_ts = (abs(r[0] - 1.10) < 2e-2 and abs(r[1] - 1.10) < 2e-2
                         and abs(r[2] - 1.15) < 2e-2)
            true_idx = _true_hessian_index(_hcnlike_oracle, xp)
            n_true_saddle += int(is_hcn_ts)
            n_false_pos += int(true_idx != 1)          # accepted a non-first-order pt?
            results.append((seed, "CONV", rounds, calls, r, true_idx, is_hcn_ts))
        else:
            results.append((seed, "MAXITER", rounds, calls, None, None, False))

    # (a) EXACTNESS: the gate must NEVER declare convergence at a non-first-order
    #     point (a |F|<fmax minimum / higher-order saddle). This is the correctness
    #     invariant (a false saddle is worse than an honest max_iter).
    assert n_false_pos == 0, (
        f"{n_false_pos} FALSE convergence(s): the gate accepted a point whose TRUE "
        f"Hessian index != 1 (not a first-order saddle).")
    # (b) ROBUSTNESS: most seeds reach the intended HCN saddle. A min-mode dimer can
    #     wander to a different stationary point from some starts (inherent); we
    #     require a strong majority, not all.
    assert n_true_saddle >= 4, (
        f"only {n_true_saddle}/{len(seeds)} seeds reached the true HCN saddle "
        f"(expected >=4); the invdist GP acquisition regressed.")

    if verbose:
        print("\n=== GP-saddle MOLECULAR self-test (torch-free, invdist, full driver) ===")
        for seed, st, rounds, calls, r, ti, is_ts in results:
            if st == "CONV":
                print(f"  seed {seed}: CONV in {calls:3d} true calls ({rounds} rounds) "
                      f"rCH={r[0]:.3f} rNH={r[1]:.3f} rCN={r[2]:.3f} "
                      f"true_index={ti} {'HCN-TS' if is_ts else 'other-saddle'}")
            else:
                print(f"  seed {seed}: MAXITER (honest -- min-mode did not reach a "
                      f"first-order saddle)")
        print(f"  true-HCN-saddle: {n_true_saddle}/{len(seeds)}   "
              f"false-convergences: {n_false_pos}   (exact-at-convergence: index==1)")
        print("  MOLECULAR SELF-TEST OK")
    return {"n_true_saddle": n_true_saddle, "n_false_pos": n_false_pos,
            "per_seed": results}


# ============================================================================ #
#  HCN <-> HNC endpoints (same as the sibling GSM test). Midpoint = TS guess.   #
# ============================================================================ #
def _build_reaction():
    R = Atoms("HCN", positions=[[-1.066, 0.0, 0.0],
                                [0.000, 0.0, 0.0],
                                [1.153, 0.0, 0.0]])
    P = Atoms("HCN", positions=[[2.160, 0.0, 0.0],
                                [0.000, 0.0, 0.0],
                                [1.169, 0.0, 0.0]])
    return R, P


def _midpoint_guess():
    """0.5 (R + P) TS guess with a small out-of-line kick so the H is off the
    C-N axis (the HCN<->HNC TS has the H bridging above the C-N bond)."""
    R, P = _build_reaction()
    g = 0.5 * (R.get_positions() + P.get_positions())
    g[0, 1] += 0.6                                   # lift H off the axis
    at = Atoms("HCN", positions=g)
    at.info.update(charge=0, spin=1, mult=1)
    return at


def _kabsch_rmsd(P, Q):
    """RMSD after optimal rotation+translation (Kabsch) of P onto Q."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    Rrot = Vt.T @ D @ U.T
    Pr = Pc @ Rrot.T
    return float(np.sqrt(np.mean(np.sum((Pr - Qc) ** 2, axis=1))))


class _Mols:
    """Trivial mols container (BatchDimer / BatchGPSaddle read .multiatoms/.calc)."""
    def __init__(self, atoms_list, calc):
        self.multiatoms = atoms_list
        self.calc = calc


def run_parity(make_calc, B=4, tol_rmsd=1.0e-2, tol_barrier=1.0e-3,
               device="cuda", verbose=True, workdir=None):
    """Assert BatchGPSaddle finds the SAME HCN<->HNC saddle as the BatchDimer
    oracle with FEWER true forwards. ``make_calc`` is a zero-arg factory (called
    twice: one calc for the oracle, one for the GP run). Returns a metrics dict;
    raises AssertionError on parity failure."""
    import tempfile
    from maple.function.dispatcher.ts.algorithm.dimer import BatchDimer
    from maple.function.dispatcher.ts.algorithm.gp_saddle import BatchGPSaddle

    wd = workdir or tempfile.mkdtemp(prefix="gp_saddle_parity_")

    # -------- oracle: plain BatchDimer min-mode over the SAME calc --------
    oracle_atoms = [_midpoint_guess() for _ in range(B)]
    bd = BatchDimer(output=os.path.join(wd, "oracle_dimer.out"), device=device,
                    paras={"dimer": {"save_traj": False, "shrink_on_converge": False}})
    bd.run(_Mols(oracle_atoms, make_calc()))
    oracle_fe = int(bd.total_force_evals)
    oracle_ts = [a.get_positions().copy() for a in oracle_atoms]
    oracle_E = np.asarray(bd.final_energy, dtype=float)

    # -------- GP-surrogate: BatchGPSaddle over a fresh calc --------
    gp_atoms = [_midpoint_guess() for _ in range(B)]
    gp = BatchGPSaddle(output=os.path.join(wd, "gp_saddle.out"), device=device,
                       paras={"gp": {"save_traj": False,
                                     "confirm_curv_true": True,   # exact FD-HVP saddle confirm
                                     "max_outer": 80,             # real PES needs more rounds
                                     "verbose": True}})           # per-round trace -> job log
    gp.run(_Mols(gp_atoms, make_calc()))
    gp_fe = int(gp.true_forward_calls)
    gp_ts = [a.get_positions().copy() for a in gp_atoms]
    gp_E = np.asarray(gp.final_energy, dtype=float)

    # -------- assertions --------
    assert gp_fe > 0 and oracle_fe > 0
    worst_rmsd = 0.0
    worst_dE = 0.0
    for k in range(B):
        assert gp.final_status[k] == "converged", (
            f"structure {k}: BatchGPSaddle did not converge "
            f"(status={gp.final_status[k]})")
        rmsd = _kabsch_rmsd(gp_ts[k], oracle_ts[k])
        worst_rmsd = max(worst_rmsd, rmsd)
        if np.isfinite(gp_E[k]) and np.isfinite(oracle_E[k]):
            dE = abs(float(gp_E[k]) - float(oracle_E[k]))
            worst_dE = max(worst_dE, dE)
            assert dE <= tol_barrier, (
                f"structure {k}: TS energy dev {dE:.3e} Eh > {tol_barrier:.1e} "
                f"(GP found a different saddle than the oracle)")
        assert rmsd <= tol_rmsd, (
            f"structure {k}: TS Kabsch-RMSD {rmsd:.3e} A > {tol_rmsd:.1e} "
            f"(GP found a different saddle than the oracle)")

    assert gp_fe < oracle_fe, (
        f"BatchGPSaddle used {gp_fe} true forwards >= BatchDimer oracle "
        f"{oracle_fe}; expected a meaningful reduction")
    ratio = oracle_fe / gp_fe

    if verbose:
        print("\n=== GP-saddle HCN<->HNC parity (real MLIP) ===")
        print(f"  workdir                 : {wd}")
        print(f"  oracle (BatchDimer) true forwards : {oracle_fe}")
        print(f"  BatchGPSaddle true forwards       : {gp_fe}")
        print(f"  true-force-call reduction         : {ratio:.1f}x")
        print(f"  worst TS Kabsch-RMSD    : {worst_rmsd:.3e} A (tol {tol_rmsd:.1e})")
        print(f"  worst TS barrier dev    : {worst_dE:.3e} Eh (tol {tol_barrier:.1e})")
        print("  PARITY OK  (same saddle, fewer forwards)")
    return {"workdir": wd, "oracle_fe": oracle_fe, "gp_fe": gp_fe,
            "ratio": ratio, "worst_rmsd": worst_rmsd, "worst_dE": worst_dE}


# ============================================================================ #
#  __main__: always run the torch-free analytic self-test; then best-effort     #
#  build a calc from the environment for the real-MLIP parity (skip if none).   #
#    MAPLE_TEST_CALC  : 'ani'(default) | 'uma' | 'mace' | 'aimnet'              #
#    MAPLE_TEST_MODEL : model path (required for uma/mace)                       #
#    MAPLE_TEST_DEVICE: 'cuda'(default) | 'cpu'                                  #
#    MAPLE_TEST_TASK  : UMA task (default 'omol')                               #
#    MAPLE_TEST_B     : number of concurrent searches (default 4)               #
# ============================================================================ #
def _make_calc_from_env():
    import torch
    kind = os.environ.get("MAPLE_TEST_CALC", "ani").lower()
    dev = os.environ.get("MAPLE_TEST_DEVICE", "cuda")
    model = os.environ.get("MAPLE_TEST_MODEL")
    dt = torch.float64
    if kind == "ani":
        from maple.function.calculator.ani._ani_batch_calculator import ANIBatchCalc
        return lambda: ANIBatchCalc(model_path=model, device=dev, dtype=dt)
    if kind == "uma":
        from maple.function.calculator.uma._uma_batch_calculator import UMABatchCalc
        task = os.environ.get("MAPLE_TEST_TASK", "omol")
        assert model, "MAPLE_TEST_MODEL required for UMA"
        return lambda: UMABatchCalc(model, device=dev, dtype=dt, task=task)
    if kind == "mace":
        from maple.function.calculator.mace._mace_batch_calculator import MACEBatchCalc
        assert model, "MAPLE_TEST_MODEL required for MACE"
        return lambda: MACEBatchCalc(model, device=dev, dtype=dt)
    if kind == "aimnet":
        from maple.function.calculator.aimnet._aimnet2_decoupled_batch_calculator import (
            AIMNet2DecoupledBatchCalc)
        return lambda: AIMNet2DecoupledBatchCalc(model or "aimnet2", device=dev, dtype=dt)
    raise SystemExit(f"unknown MAPLE_TEST_CALC={kind!r}")


if __name__ == "__main__":
    # 1) torch-free self-tests (always run): deterministic quadratic + the
    #    invdist HCN-like PES that guards the real-UMA max_iter fix.
    analytic_self_test()
    molecular_self_test()

    # 2) real-MLIP parity (optional; skips cleanly if no calc/model available)
    try:
        make_calc = _make_calc_from_env()
    except Exception as e:
        print(f"\n[skip] real-MLIP parity: could not build a Batch calc "
              f"from the environment: {e!r}")
        print("       set MAPLE_TEST_CALC / MAPLE_TEST_MODEL and rerun, or call "
              "run_parity(make_calc) with your own factory.")
        raise SystemExit(0)
    B = int(os.environ.get("MAPLE_TEST_B", "4"))
    run_parity(make_calc, B=B,
               device=os.environ.get("MAPLE_TEST_DEVICE", "cuda"))
    print("\nOK")
