"""
Forward Flux Sampling (FFS) gates. Mirrors _test_tps.py / _test_we.py methodology.

  GATE 1  INTERFACE-PROBABILITIES : every P(lambda_{i+1}|lambda_i) in [0,1] and the
          product gives a FINITE, positive rate k_AB = Phi_A * prod P. Checked both
          on the pure-numpy 1D FFS and (on GPU) through the real ForwardFluxSampling
          driver + the batched double-well calc.
  GATE 2  RATE CROSS-CHECK        : on a double-well toy, FFS k_AB (MFPT = 1/k_AB)
          agrees within a small factor with a brute-force MD first-passage MFPT on
          the SAME toy + dynamics -- and the rate is INVARIANT to interface
          placement (the FFS guarantee). The numpy gate cross-checks against a
          vectorized 1D first-passage; the driver gate reuses _test_we's
          _DoubleWellDimer + _brute_force_mfpt so the numbers are directly
          comparable to the (already validated) WE MFPT.
  GATE 3  BATCH ISOLATION         : the N parallel shots at an interface are
          independent -- perturbing ONE shot's initial velocity does NOT change any
          other shot's trajectory (like TPS gate 2). (Toy batched calc, friction=0.)
  GATE 4  FLUX MEASUREMENT        : the effective-positive-flux crossing counter
          (count_interface_crossings, the canonical Phi_A definition) is EXACT on
          synthetic lambda series with a KNOWN number of crossings (arming/gating,
          jitter, mid-series start), and the driver flux run reports Phi_A > 0.

Gates 4(counter) / 1(numpy) / 2(numpy) are PURE NUMPY (no torch/GPU) and run
anywhere numpy+ase are present. Gates 3 / 1(driver) / 2(driver) / 4(driver) need
torch -- they drive the REAL ForwardFluxSampling through the batched kernel and
run on GPU/CPU-with-torch (like the other _test_*). Run with ``--numpy-only`` to
skip the torch gates.
"""
import os
import sys
import tempfile

import numpy as np

from maple.function.dispatcher.md.ensemble.ffs import (
    ForwardFluxSampling, FFSParams, count_interface_crossings, ffs_rate)
from maple.function.dispatcher.md.ensemble.tps import OrderParameter


def _tmp():
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        return f.name


# =============================================================================
# Pure-numpy 1D overdamped double well (no torch): V = h*((x/w0)^2-1)^2, wells at
# +/- w0, barrier h at 0. lambda = x. A = left well; lambda_A = lambda_0. Brownian
# dynamics x' = x - (D/kT) V'(x) dt + sqrt(2 D dt) N(0,1). Reuses the library's
# count_interface_crossings (flux) + ffs_rate (k_AB product); OrderParameter is
# reused for the basin/threshold consistency check.
# =============================================================================
_H, _W0, _KT, _D, _DT = 3.0, 1.0, 1.0, 1.0, 0.001     # barrier = 3 kT
_XA_MIN, _XB_MIN = -1.0, 1.0
_LAM0, _LAMB = -0.9, 0.9                                # lambda_A=lambda_0 ; lambda_n


def _dVdx(x):
    return 4.0 * _H * x * ((x / _W0) ** 2 - 1.0) / (_W0 ** 2)


def _bd_step(x, rng):
    return x - (_D / _KT) * _dVdx(x) * _DT + np.sqrt(2.0 * _D * _DT) * rng.standard_normal(x.shape)


def _brute_mfpt_1d(M=400, max_steps=200000, seed=1):
    rng = np.random.default_rng(seed)
    x = np.full(M, _XA_MIN)
    fpt = np.full(M, -1.0)
    for s in range(1, max_steps + 1):
        x = _bd_step(x, rng)
        newly = (fpt < 0) & (x >= _LAMB)
        fpt[newly] = s
        if np.all(fpt > 0):
            break
    crossed = fpt[fpt > 0]
    mfpt = float(np.mean(crossed)) * _DT if crossed.size else float("inf")
    return mfpt, crossed.size / M


def _ffs_1d(interfaces, n_shots=2000, flux_walkers=200, flux_steps=20000,
            shot_max=20000, seed=2):
    """Vectorized numpy direct FFS. Flux uses the LIBRARY crossing rule (mirrored
    online for config capture); rate uses the LIBRARY ffs_rate. Returns
    (Phi_A, P_list, k_AB, n_cross)."""
    rng = np.random.default_rng(seed)
    lamA = lam0 = float(interfaces[0])
    # --- flux: walkers in A, count effective positive crossings of lambda_0 ------
    x = np.full(flux_walkers, _XA_MIN)
    prev = x.copy()
    armed = (x <= lamA)
    n_cross = 0
    pool = []
    for _ in range(flux_steps):
        x = _bd_step(x, rng)
        cr = armed & (prev < lam0) & (x >= lam0)      # same rule as count_interface_crossings
        if cr.any():
            n_cross += int(cr.sum())
            pool.extend(x[cr].tolist())
            armed[cr] = False
        armed[x <= lamA] = True
        prev = x.copy()
    total_time = flux_walkers * flux_steps * _DT
    phi = n_cross / total_time if total_time > 0 else 0.0
    # --- interfaces: batched shots i -> i+1 --------------------------------------
    P = []
    cur = np.asarray(pool, dtype=float)
    for i in range(len(interfaces) - 1):
        lam_next = float(interfaces[i + 1])
        if cur.size == 0:
            P.append(0.0)
            cur = np.asarray([], dtype=float)
            continue
        xs = cur[rng.integers(cur.size, size=n_shots)].copy()
        outcome = np.zeros(n_shots, dtype=int)        # 0 live, 1 success, -1 fail
        succ_x = []
        for _ in range(shot_max):
            live = outcome == 0
            if not live.any():
                break
            xs[live] = _bd_step(xs[live], rng)
            succ = live & (xs >= lam_next)
            fail = live & (xs <= lamA)
            if succ.any():
                outcome[succ] = 1
                succ_x.extend(xs[succ].tolist())
            outcome[fail] = -1
        P.append(int((outcome == 1).sum()) / n_shots)
        cur = np.asarray(succ_x, dtype=float)
    return phi, P, ffs_rate(phi, P), n_cross


# ============================================================ GATE 4 (numpy: counter)
def run_gate4_flux_counter():
    """count_interface_crossings EXACT on synthetic lambda series with known counts."""
    cases = [
        # (series, lam0, lamA, expected_n)
        ([-2, -2, 0.5, 0.5, -2, 0.6, 0.6, -2, 0.7], 0.0, -1.0, 3),   # 3 clean crossings
        ([-2, 0.5, 0.4, 0.6, 0.55], 0.0, -1.0, 1),                    # jitter above, no return -> 1
        ([0.5, 0.5, -2, 0.6], 0.0, -1.0, 1),                          # start OUTSIDE A -> not armed
        ([-2, -2, -2], 0.0, -1.0, 0),                                 # never crosses -> 0
        ([-2, 1, -2, 1, -2, 1], 0.0, -1.0, 3),                        # 3 up-down cycles
        ([-2, -0.5, 1, -2, 1], 0.0, -1.0, 2),                         # hysteresis: lamA<lam0 gap
    ]
    for series, lam0, lamA, exp in cases:
        n, idxs = count_interface_crossings(series, lam0, lamA)
        assert n == exp, f"counter {series} lam0={lam0} lamA={lamA}: got {n}, expect {exp}"
        assert len(idxs) == n
    # known analytic flux: N sawtooth up-crossings over a fixed time -> Phi = N / T.
    N = 25
    series = []
    for _ in range(N):
        series += [-2.0, 1.0]        # A -> above lam0 (one crossing), then back to A next cycle
    n, _ = count_interface_crossings(series, 0.0, -1.0)
    assert n == N, f"sawtooth flux count {n} != {N}"
    print(f"[GATE4 flux-counter] {len(cases)} gated cases + sawtooth(N={N}) all EXACT")
    print("[GATE4-numpy] PASS  (effective-positive-flux counter exact on known series)")
    return True


# ============================================================ GATE 1 (numpy: wellformed)
def run_gate1_wellformed_numpy():
    """Every P in [0,1]; product -> a finite positive rate. Reuses OrderParameter
    to confirm the basin thresholds match the interface endpoints."""
    op = OrderParameter(kind="custom", func=lambda P: float(P[1, 0] - P[0, 0]),
                        a_max=_LAM0, b_min=_LAMB)
    assert op.basin(_LAM0 - 0.1) == "A" and op.basin(_LAMB + 0.1) == "B"
    interfaces = np.linspace(_LAM0, _LAMB, 10)
    phi, P, k, nc = _ffs_1d(interfaces, seed=17)
    print(f"[GATE1 wellformed-numpy] Phi_A={phi:.4e}/t  crossings={nc}  "
          f"P={[f'{p:.3f}' for p in P]}")
    print(f"    prod P={np.prod(P):.4e}  k_AB={k:.4e}/t  MFPT={1/k:.3f}")
    assert all(0.0 <= p <= 1.0 for p in P), "some P(i+1|i) outside [0,1]"
    assert phi > 0 and np.isfinite(k) and k > 0, "flux/rate not finite-positive"
    print("[GATE1-numpy] PASS  (all P in [0,1]; k_AB finite and positive)")
    return phi, P, k


# ============================================================ GATE 2 (numpy: rate)
def run_gate2_rate_numpy():
    """FFS k_AB (MFPT=1/k) vs a brute-force 1D first-passage MFPT (SAME dynamics),
    AND interface-placement invariance of the rate."""
    mfpt_bf, frac = _brute_mfpt_1d(seed=1)
    rows = []
    for nif in (6, 10, 19):
        interfaces = np.linspace(_LAM0, _LAMB, nif)
        phi, P, k, nc = _ffs_1d(interfaces, n_shots=3000, seed=nif + 40)
        mfpt = (1.0 / k) if k > 0 else float("inf")
        rows.append((nif, k, mfpt, mfpt / mfpt_bf))
        print(f"[GATE2 rate-numpy] n_if={nif:2d}  k_AB={k:.4e}/t  MFPT_ffs={mfpt:.3f}  "
              f"MFPT_bf={mfpt_bf:.3f}  ratio={mfpt / mfpt_bf:.3f}")
    ratios = [r[3] for r in rows]
    mfpts = [r[2] for r in rows]
    assert frac > 0.5, "brute-force reference saw too few crossings"
    for _nif, _k, _m, ratio in rows:
        assert 0.3 < ratio < 3.0, f"FFS MFPT off by >3x from brute-force (n_if={_nif})"
    # interface-placement invariance: all placements give the SAME rate up to the
    # finite-shot statistical error (the FFS guarantee) -> tight max/min spread.
    invar = max(mfpts) / min(mfpts)
    assert invar < 1.5, f"FFS rate not interface-invariant (max/min={invar:.2f})"
    print(f"[GATE2-numpy] PASS  (FFS MFPT within {max(ratios):.2f}x of brute-force; "
          f"interface-invariant max/min={invar:.3f})")
    return rows


# =============================================================================
# Torch driver gates -- REAL ForwardFluxSampling through the batched kernel, on the
# SAME double-well fixture as _test_we.py (directly comparable to the WE MFPT).
# =============================================================================
def _we_fixture():
    """Import the _test_we double-well batch calc + constants + brute-force ref."""
    from maple.function.dispatcher.md.ensemble._test_we import (
        _DoubleWellDimer, _dimer, T_K, R1, R2, RMID, WHALF, H_BARRIER,
        _brute_force_mfpt)
    return (_DoubleWellDimer, _dimer, T_K, R1, R2, RMID, WHALF, H_BARRIER,
            _brute_force_mfpt)


# ---------------------------------------------------------------- GATE 3 (driver)
def run_gate3_isolation(steps=60, seed=23, N=4, perturb=0.37):
    """N parallel shots independent: perturb shot 0's initial velocity -> others
    unchanged. friction=0 (NVE) so the propagation is deterministic given the
    initial velocities (RNG-independent), isolating the batch-independence check."""
    import torch
    torch.set_default_dtype(torch.float64)
    (_DoubleWellDimer, _dimer, T_K, R1, R2, RMID, WHALF, H_BARRIER, _bf) = _we_fixture()
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    calc = _DoubleWellDimer(RMID, WHALF, H_BARRIER, device=DEV, dtype=torch.float64)
    # unreachable thresholds so _propagate_shots runs the FULL `steps` (no S/F).
    op = OrderParameter(kind="distance", indices="0,1", a_max=-1e9, b_min=1e9)
    ffs = ForwardFluxSampling(
        _tmp(), _dimer(R1), calc=calc, order_parameter=op,
        paras=dict(n_shots=N, flux_steps=1, shot_max_steps=steps, timestep=0.5,
                   temperature=T_K, friction=0.0, remove_com_every=0,
                   random_seed=seed, verbose=0))
    ffs._ensure_prepared()

    x0 = _dimer(R1).get_positions()
    v0 = [ffs._draw_shoot_vel(b) for b in range(N)]

    # run A: propagate all shots from x0 with v0 (no shot terminates).
    ffs._propagate_shots([x0] * N, [v.copy() for v in v0], lam_next=1e18, max_steps=steps)
    posA = ffs._positions_per_replica()

    # run B: perturb ONLY shot 0's initial velocity; others identical to run A.
    v0b = [v.copy() for v in v0]
    v0b[0] = v0b[0] + perturb
    ffs._propagate_shots([x0] * N, v0b, lam_next=1e18, max_steps=steps)
    posB = ffs._positions_per_replica()

    d_shot0 = float(np.max(np.abs(posB[0] - posA[0])))
    d_others = max(float(np.max(np.abs(posB[b] - posA[b]))) for b in range(1, N))
    print(f"[GATE3 ISOLATION] N={N} steps={steps}  perturbed shot0 moved={d_shot0:.3e} A  "
          f"other shots max|d|={d_others:.3e} A")
    assert d_shot0 > 1e-8, "shot 0 velocity perturbation had no effect (test broken)"
    assert d_others < 1e-9, "ISOLATION FAIL (perturbing shot 0 changed another shot)"
    print("[GATE3] PASS  (N shots independent: perturb one -> others unchanged)")
    return d_shot0, d_others


# ------------------------------------------------------- GATE 1/2/4 (driver)
def run_gate124_driver(seed=3, n_shots=200):
    """Full ForwardFluxSampling on the _test_we double well. Checks P in [0,1] +
    finite k (GATE1), Phi_A>0 (GATE4-driver), and MFPT vs brute-force (GATE2)."""
    import torch
    torch.set_default_dtype(torch.float64)
    (_DoubleWellDimer, _dimer, T_K, R1, R2, RMID, WHALF, H_BARRIER, _bf) = _we_fixture()
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    calc = _DoubleWellDimer(RMID, WHALF, H_BARRIER, device=DEV, dtype=torch.float64)

    # lambda = bond length; lambda_0 just outside A (R1=1.3), lambda_B past barrier
    # (rmid=1.7); target r>=2.0 == the _test_we brute-force target -> comparable.
    lam0, lamB, r_target = 1.45, 2.0, 2.0
    op = OrderParameter(kind="distance", indices="0,1", a_max=lam0, b_min=lamB)
    ffs = ForwardFluxSampling(
        _tmp(), _dimer(R1), calc=calc, order_parameter=op,
        paras=dict(n_shots=n_shots, n_interfaces=7, flux_steps=4000,
                   shot_max_steps=4000, op_every=5, timestep=0.5, temperature=T_K,
                   thermostat="langevin", friction=0.02, remove_com_every=0,
                   remove_com=True, random_seed=seed, ffs_seed=seed + 11,
                   verbose=0)).run()

    k = ffs.k_AB
    mfpt_ffs = ffs.mfpt_fs
    paras = dict(timestep=0.5, temperature=T_K, thermostat="langevin", friction=0.02,
                 remove_com_every=0, remove_com=True, verbose=0)
    mfpt_bf, frac = _bf(calc, r_target, paras)
    ratio = mfpt_ffs / mfpt_bf if (np.isfinite(mfpt_ffs) and mfpt_bf > 0) else float("inf")

    print(f"\n[GATE1/4 driver] Phi_A={ffs.flux:.4e}/fs ({ffs.n_crossings} crossings)  "
          f"P={[f'{p:.3f}' for p in ffs.P_interfaces]}")
    print(f"[GATE2 rate-driver] k_AB={k:.4e}/fs  MFPT_ffs={mfpt_ffs:.3e} fs "
          f"({mfpt_ffs / 1000:.3f} ps)")
    print(f"  brute-force ref:  crossed_frac={frac:.2f}  MFPT_bf={mfpt_bf:.3e} fs "
          f"({mfpt_bf / 1000:.3f} ps)   ratio={ratio:.3f}")
    # GATE 1
    assert all(0.0 <= p <= 1.0 for p in ffs.P_interfaces), "P(i+1|i) outside [0,1]"
    assert np.isfinite(k) and k > 0, "k_AB not finite-positive"
    # GATE 4 (driver)
    assert ffs.flux > 0 and ffs.n_crossings > 0, "no lambda_0 flux measured"
    # GATE 2 (order-of-magnitude / statistical, like WE gate4)
    assert frac > 0.3, "brute-force reference saw too few crossings"
    assert 0.1 < ratio < 10.0, "FFS MFPT not within an order of magnitude of brute-force"
    print("[GATE1/2/4-driver] PASS  (P well-formed; flux>0; MFPT ~ brute-force)")
    return k, mfpt_ffs, mfpt_bf, ratio


if __name__ == "__main__":
    # numpy gates run with NO torch/GPU (always).
    run_gate4_flux_counter()
    run_gate1_wellformed_numpy()
    run_gate2_rate_numpy()

    if "--numpy-only" in sys.argv:
        print("\n[FFS RESULT] numpy gates PASS (torch driver gates skipped)")
        raise SystemExit(0)

    import torch
    print("\ntorch", torch.__version__, "cuda", torch.cuda.is_available(),
          "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    d0, dov = run_gate3_isolation()
    k, mfpt_ffs, mfpt_bf, ratio = run_gate124_driver()
    print(f"\n[FFS RESULT] gate3 isolation shot0_moved={d0:.3e} others_delta={dov:.3e} A")
    print(f"[FFS RESULT] gate2 k_AB={k:.4e}/fs MFPT_ffs={mfpt_ffs:.3e} fs "
          f"MFPT_bf={mfpt_bf:.3e} fs ratio={ratio:.3f}")
    print("[FFS RESULT] ALL GATES PASS")
