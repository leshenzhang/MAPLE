"""
Population Annealing MD (PA-MD) gates. Mirrors _test_we.py / _test_remd.py.

  GATE 1  FREE-ENERGY LADDER : on an analytic toy (3D isotropic harmonic on a
          dimer's relative vector, F(T) known in closed form) the accumulated
          ``lnZ_ratio`` = ln[ Z(T_target)/Z(T_0) ] matches the analytic value
          (3/2) ln(T_target/T_0) within tolerance. Tested (a) numpy-only on EXACT
          per-node Boltzmann samples (``run_gate1_estimator_numpy``, the estimator
          math), and (b) end-to-end through the MD driver on the torch fixture
          (``run_gate1_gate4_md``).
  GATE 2  DEGENERATE-REDUCTION : a single-temperature ladder (temp_start ==
          temp_target => every Delta_beta == 0 => uniform weights => systematic
          resampling is the identity NO-OP) === a plain ``BatchedNVT`` run of the
          same total steps + seed, positions BIT-IDENTICAL to fp64 machine
          precision; weights stay uniform 1/M and lnZ_ratio == 0.
  GATE 3  RESAMPLING INVARIANTS : population size stays M; the reweight weights
          sum to 1 to 1e-12; systematic resampling of uniform weights is exactly
          the identity; resampling preserves the weighted mean of an observable IN
          EXPECTATION (Monte-Carlo mean over many resamples converges to the
          pre-resample weighted mean). Pure numpy.
  GATE 4  ANNEALED-WEIGHT UNBIASEDNESS : the annealed importance weights recover
          the Boltzmann average of an observable at the target temperature. Tested
          (a) numpy-only by reweighting exact T_src samples to T_target and
          checking <|d|^2> == 3 kB T_target / k (``run_gate4_reweight_numpy``), and
          (b) end-to-end via the final PA population (``run_gate1_gate4_md``).

Gates 1(numpy), 3, 4(numpy) are NUMPY-ONLY and run without torch/MACE (the
estimator + resampling helpers are pure numpy). Gate 2 and the MD legs of gates
1/4 use a self-contained analytic ``_HarmonicDimer`` batch calc (a TEST FIXTURE
mirroring the ``_DoubleWellDimer`` in _test_we.py / the ``_HarmonicBatch`` in
batch_calculator_base.py's __main__) so they need only torch (CPU ok), no MACE
model file. torch imported lazily INSIDE the torch gates so the numpy gates run
even where torch is absent.
"""
import os, sys, tempfile
import numpy as np

from maple.function.dispatcher.md.utils import KELVIN_TO_HARTREE
from maple.function.dispatcher.md.ensemble.population_annealing import (
    systematic_resample, residual_resample, resample_indices,
    reduced_free_energy_increment)

# ---- analytic toy: 3D isotropic harmonic E = 1/2 k |d|^2 on the relative vector
K_HARM = 0.05                       # Ha / Angstrom^2 (force constant)
N_HARM = 3                          # harmonic dofs (relative vector d in R^3)
kB = KELVIN_TO_HARTREE             # Hartree / K


def analytic_lnZ_ratio(T0, Tt):
    """ln[ Z(T_target)/Z(T_0) ] = (3/2) ln(T_target/T_0) for the 3D harmonic
    (Z(beta) ~ beta^{-3/2}, beta ~ 1/T)."""
    return 0.5 * N_HARM * np.log(Tt / T0)


def analytic_d2(T):
    """<|d|^2>_T = 3 kB T / k (equipartition over the 3 harmonic dofs)."""
    return N_HARM * kB * T / K_HARM


def sample_E(T, M, rng):
    """M EXACT Boltzmann samples at temperature T: d ~ N(0, kB T / k) per axis,
    E = 1/2 k |d|^2. Returns (E (M,), d (M,3))."""
    sigma = np.sqrt(kB * T / K_HARM)
    d = rng.normal(0.0, sigma, size=(M, 3))
    E = 0.5 * K_HARM * np.sum(d * d, axis=1)
    return E, d


def _tmp():
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        return f.name


# ============================================================ GATE 3 (numpy only)
def run_gate3_resampling(seed=5):
    rng = np.random.default_rng(seed)

    # (a) reweight weights sum to 1 to 1e-12 over many random (E, dbeta).
    worst_w = 0.0
    for _ in range(3000):
        M = int(rng.integers(2, 40))
        E = rng.normal(0.0, 1e-3, M)
        db = rng.uniform(-600.0, 600.0)
        _dlnz, w = reduced_free_energy_increment(E, db)
        worst_w = max(worst_w, abs(float(w.sum()) - 1.0))
    print(f"[GATE3 weights] max |sum w - 1| over 3000 reweights = {worst_w:.3e}")
    assert worst_w < 1e-12, "reweight weights not normalized to 1e-12"

    # (b) population size stays M; indices in range; both methods.
    for method in ("systematic", "residual"):
        for _ in range(2000):
            M = int(rng.integers(1, 40))
            w = rng.random(M)
            idx = resample_indices(w, rng, method)
            assert idx.size == M, (method, M, idx.size)
            assert idx.min() >= 0 and idx.max() < M, (method, M, idx.min(), idx.max())
    print("[GATE3 size]    population size stays M and indices in [0,M) (both methods)")

    # (c) systematic resampling of UNIFORM weights == identity (the NO-OP path).
    for M in (1, 2, 3, 5, 8, 16, 37, 64):
        idx = systematic_resample(np.full(M, 1.0 / M), rng)
        assert np.array_equal(idx, np.arange(M)), (M, idx)
    print("[GATE3 identity] systematic(uniform) == arange(M) for M in {1..64}")

    # (d) resampling preserves the weighted mean IN EXPECTATION (Monte-Carlo).
    for method in ("systematic", "residual"):
        M = 12
        w = rng.random(M); w = w / w.sum()
        O = rng.normal(0.0, 1.0, M)
        target = float(np.dot(w, O))
        Kdraw = 200000
        acc = 0.0
        for _ in range(Kdraw):
            idx = resample_indices(w, rng, method)
            acc += float(O[idx].mean())
        mc = acc / Kdraw
        print(f"[GATE3 unbiased/{method}] MC mean over {Kdraw} = {mc:.6f}  "
              f"target=<O>_w={target:.6f}  err={abs(mc-target):.2e}")
        assert abs(mc - target) < 5e-3, (method, mc, target)
    print("[GATE3] PASS  (size M, sum w=1, identity NO-OP, weighted-mean-preserving)")
    return worst_w


# ============================================================ GATE 1 (numpy only)
def run_gate1_estimator_numpy(seed=1, M=300000, K=20, T0=600.0, Tt=300.0):
    """Free-energy ladder estimator on EXACT per-node Boltzmann samples: each
    interval's ln<exp(-dbeta E)>_{beta_k} == ln[Z_{k+1}/Z_k]; the sum telescopes to
    ln[Z(T_target)/Z(T_0)]. Validates reduced_free_energy_increment + accumulation."""
    rng = np.random.default_rng(seed)
    betas = np.linspace(1.0 / (kB * T0), 1.0 / (kB * Tt), K + 1)
    lnZ = 0.0
    for k in range(K):
        Tk = 1.0 / (kB * betas[k])
        E, _ = sample_E(Tk, M, rng)
        dlnz, _w = reduced_free_energy_increment(E, betas[k + 1] - betas[k])
        lnZ += dlnz
    a = analytic_lnZ_ratio(T0, Tt)
    print(f"[GATE1 estimator-numpy] K={K} M={M}  lnZ_ratio={lnZ:.5f}  "
          f"analytic={a:.5f}  |d|={abs(lnZ-a):.3e}")
    assert abs(lnZ - a) < 0.02, "free-energy ladder estimator off analytic"
    print("[GATE1-numpy] PASS  (ladder log-Z ratio matches analytic)")
    return lnZ, a


# ============================================================ GATE 4 (numpy only)
def run_gate4_reweight_numpy(seed=11, M=500000, T_src=500.0, T_tgt=300.0):
    """Annealed importance reweighting: reweight EXACT T_src samples to T_tgt with
    w ~ exp(-dbeta E); the weighted <|d|^2> recovers the analytic Boltzmann average
    at T_tgt (and the unweighted mean recovers T_src -> the reweight actually moved
    it)."""
    rng = np.random.default_rng(seed)
    E, d = sample_E(T_src, M, rng)
    d2 = np.sum(d * d, axis=1)
    unweighted = float(d2.mean())
    dbeta = 1.0 / (kB * T_tgt) - 1.0 / (kB * T_src)
    _dlnz, w = reduced_free_energy_increment(E, dbeta)
    reweighted = float(np.dot(w, d2))
    a_src, a_tgt = analytic_d2(T_src), analytic_d2(T_tgt)
    print(f"[GATE4 reweight-numpy] <|d|^2> unweighted@{T_src:.0f}K={unweighted:.5f} "
          f"(analytic {a_src:.5f});  reweighted->{T_tgt:.0f}K={reweighted:.5f} "
          f"(analytic {a_tgt:.5f})")
    assert abs(unweighted - a_src) / a_src < 0.02, "T_src sampling off analytic"
    assert abs(reweighted - a_tgt) / a_tgt < 0.03, "annealed reweight off analytic"
    print("[GATE4-numpy] PASS  (annealed weights recover the T_target Boltzmann average)")
    return reweighted, a_tgt


# ================================================= torch fixture (lazy: no torch here)
def _make_harmonic_calc(device, k=K_HARM):
    """3D isotropic harmonic on each 2-atom molecule's relative vector d = r0 - r1:
    E = 1/2 k |d|^2 (Hartree), F0 = -k d, F1 = +k d. Block-diagonal (each molecule's
    E depends only on its own d) -> batch-ISOLATED, MLIP-shaped test fixture.
    NOT a production calculator (mirrors _DoubleWellDimer / _HarmonicBatch)."""
    import torch
    from maple.function.calculator.batch_calculator_base import BatchCalcABC

    class _HarmonicDimer(BatchCalcABC):
        MODEL_NAMES = ("_pa_harmonic_test",)
        MODEL_ENERGY_UNIT = "hartree"
        MODEL_DTYPE = torch.float64
        SUPPORTED_HESSIAN_MODES = ("numerical",)

        def __init__(self, kk, device="cpu", dtype=torch.float64):
            super().__init__(device=device, dtype=dtype)
            self.k = float(kk)

        def _forward(self, coord, need_graph):
            P0, P1 = coord[0::2], coord[1::2]           # (B,3) atom0 / atom1
            d = P0 - P1
            E = 0.5 * self.k * (d * d).sum(dim=1)       # (B,) Hartree
            F0 = -self.k * d                            # (B,3) Ha/A
            F1 = self.k * d
            F_all = torch.empty((2 * self._atoms_B, 3), dtype=coord.dtype, device=coord.device)
            F_all[0::2] = F0
            F_all[1::2] = F1
            return E, F_all, None

    return _HarmonicDimer(k, device=device, dtype=torch.float64)


def _dimer(r):
    """A 2-atom C-C 'molecule' with atom separation r (Angstrom) along x."""
    from ase import Atoms
    return Atoms("CC", positions=[[0.0, 0.0, 0.0], [float(r), 0.0, 0.0]])


# ============================================================ GATE 2 (torch fixture)
def run_gate2_degenerate(M=6, K=8, n_sweep=25, T=350.0, seed=42):
    import torch
    torch.set_default_dtype(torch.float64)
    from maple.function.dispatcher.md.ensemble.nvt_batched import BatchedNVT
    from maple.function.dispatcher.md.ensemble.population_annealing import PopulationAnnealing
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    calc = _make_harmonic_calc(dev)
    starts = [_dimer(0.30 + 0.02 * b) for b in range(M)]
    paras = dict(timestep=0.5, temperature=T, thermostat="langevin", friction=0.03,
                 remove_com_every=50, random_seed=seed, verbose=0)
    total = (K + 1) * n_sweep

    # (1) plain BatchedNVT over the SAME M replicas, steps = (K+1)*n_sweep.
    BatchedNVT(_tmp(), [a.copy() for a in starts], calc=calc,
               paras=dict(paras, steps=total)).run()
    pos_ref = calc.coord.detach().to("cpu").numpy().copy()

    # (2) PA with a single-temperature ladder -> every resample is the identity NO-OP.
    pa = PopulationAnnealing(
        _tmp(), [a.copy() for a in starts], calc=calc,
        paras=dict(paras, n_replicas=M, temp_start=T, temp_target=T,
                   n_anneal_steps=K, n_sweep=n_sweep, resample_method="systematic",
                   resample_seed=seed + 7)).run()
    pos_pa = calc.coord.detach().to("cpu").numpy().copy()

    dpos = float(np.max(np.abs(pos_ref - pos_pa)))
    w_dev = float(np.max(np.abs(pa.weights - 1.0 / M)))
    print(f"[GATE2 degenerate] M={M} steps={total} seed={seed}  max|dpos|={dpos:.3e} A  "
          f"max|w-1/M|={w_dev:.3e}  lnZ_ratio={pa.lnZ_ratio:.3e}  dF={pa.delta_betaF:.3e}")
    assert dpos < 1e-10, "DEGENERATE FAIL (PA single-T != plain BatchedNVT positions)"
    assert w_dev < 1e-12, "DEGENERATE FAIL (weights not uniform 1/M)"
    assert abs(pa.lnZ_ratio) < 1e-12 and abs(pa.delta_betaF) < 1e-12, \
        "DEGENERATE FAIL (single-T ladder must give lnZ_ratio == 0)"
    print("[GATE2] PASS  (single-T PA === plain BatchedNVT; weights uniform; dF==0)")
    return dpos, w_dev


# ================================================== GATE 1 + GATE 4 (torch fixture MD)
def run_gate1_gate4_md(M=4000, K=20, n_sweep=60, T0=600.0, Tt=300.0, seed=3):
    import torch
    torch.set_default_dtype(torch.float64)
    from maple.function.dispatcher.md.ensemble.population_annealing import PopulationAnnealing
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    calc = _make_harmonic_calc(dev)
    starts = [_dimer(0.30) for _ in range(M)]
    paras = dict(timestep=0.5, thermostat="langevin", friction=0.05,
                 remove_com_every=50, random_seed=seed, verbose=0)
    pa = PopulationAnnealing(
        _tmp(), starts, calc=calc,
        paras=dict(paras, n_replicas=M, temp_start=T0, temp_target=Tt,
                   n_anneal_steps=K, n_sweep=n_sweep, resample_method="systematic",
                   resample_seed=seed + 101)).run()

    # GATE 1: free-energy ladder vs analytic (3/2) ln(T_target/T_0).
    a_lnZ = analytic_lnZ_ratio(T0, Tt)
    print(f"\n[GATE1 ladder-MD] M={M} K={K} n_sweep={n_sweep}  T:{T0:.0f}->{Tt:.0f} K")
    print(f"  PA lnZ_ratio = {pa.lnZ_ratio:.4f}   analytic = {a_lnZ:.4f}   "
          f"|d| = {abs(pa.lnZ_ratio - a_lnZ):.4f}")
    assert abs(pa.lnZ_ratio - a_lnZ) < 0.10, "PA free-energy ladder off analytic"

    # GATE 4: annealed observable on the final T_target population vs analytic.
    a_d2 = analytic_d2(Tt)
    d2_pa = pa.observable_mean(lambda p: float(np.sum((p[0] - p[1]) ** 2)))
    rel = abs(d2_pa - a_d2) / a_d2
    print(f"[GATE4 observable-MD] <|d|^2>@{Tt:.0f}K  PA={d2_pa:.5f}  analytic={a_d2:.5f}  "
          f"rel={rel:.3f}")
    assert rel < 0.12, "PA annealed observable off the analytic Boltzmann average"
    print("[GATE1+4-MD] PASS  (ladder free energy + T_target observable match analytic)")
    return pa.lnZ_ratio, a_lnZ, d2_pa, a_d2


if __name__ == "__main__":
    print("PA-MD gates")
    # numpy-only gates (run everywhere, no torch/MACE needed)
    run_gate3_resampling()
    run_gate1_estimator_numpy()
    run_gate4_reweight_numpy()
    print("[NUMPY GATES] PASS  (gate3 + gate1-estimator + gate4-reweight)")
    # torch/MD gates (fixture calc; CPU ok, GPU faster) -- skip cleanly if no torch
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as e:
        print(f"[torch gates SKIPPED — torch not installed: {e}]")
        print("  (GATE2 degenerate + GATE1/GATE4 MD legs must be verified on GPU)")
        sys.exit(0)
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    run_gate2_degenerate()
    run_gate1_gate4_md()
    print("[PA-MD RESULT] ALL GATES PASS")
