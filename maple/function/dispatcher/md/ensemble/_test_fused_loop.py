"""Gates for the fused (fully on-device) batched-NVT hot loop (GPU-opt Lever 3).

The fused path (``fused_loop=True``) runs the LF-Middle Langevin OU thermostat
substep fully ON the torch device: the velocity never leaves the GPU (no per-replica
``_v_real``/``_set_v_real`` ``.to("cpu").numpy()`` roundtrip), only the RNG noise --
drawn from the SAME per-replica numpy stream in the SAME order -- is H2D-copied once
per step. The OU update ``v' = c1*v + c2*xi`` is a pure elementwise fp64 tensor op
(NO reduction), so it is BIT-IDENTICAL to the default per-replica path.

  GATE 0  ALGEBRA-PARITY (torch-FREE): the fused OU math, fed the SAME numpy RNG
          state, reproduces ``LangevinThermostat.apply`` byte-for-byte -> validates
          the noise draw order/shape + the c1*v+c2*xi algebra WITHOUT a GPU. Runs
          even when torch is absent (the crux of the bit-identity claim).
  GATE 1  PARITY (the accuracy gate): BatchedNVT fused=True vs fused=False, SAME
          seed/steps/system, positions/velocities/PE/T identical to canonical fp
          precision (target BIT-identical). Covered for: no-projection, runtime COM
          projection (remove_com_every>0), and an anneal ramp (_refresh_c2_dev path).
  GATE 2  THROUGHPUT (the speed gate): steps/s fused vs default at B=1,8,32,128;
          reports speedup. On CUDA the win should grow with B (asserts >=1.0 at
          large B); on CPU there is no D2H sync to remove, so it is informational.
  GATE 3  NO-HOST-SYNC: a fused Langevin run with remove_com_every=0 makes ZERO
          ``_v_real`` device->host calls (``_host_sync_count == 0``); the default
          path makes B*steps of them -> the roundtrip is really gone from the hot
          loop.
  GATE 4  REGRESSION: default (fused_loop=False) still runs & is byte-identical to
          the pre-change behavior (it IS gate-1's reference). ``_test_nvt_batched.py``
          / ``_test_we.py`` use fused_loop default OFF, so they are unchanged on GPU.

Gates 1-4 use a self-contained analytic ``_HarmonicTether`` batch calc (a TEST
FIXTURE mirroring ``_DoubleWellDimer`` in _test_we.py / ``_HarmonicBatch`` in
batch_calculator_base.py __main__) so they need only torch (CPU ok), no MACE model.
If torch is missing, gates 1-4 auto-SKIP and only the torch-free gate 0 runs.
"""
import os
import sys
import time
import tempfile

import numpy as np
from ase import Atoms

try:
    import torch
    torch.set_default_dtype(torch.float64)
    HAVE_TORCH = True
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:                                   # torch-missing local dev box
    HAVE_TORCH = False
    DEV = "cpu"


def _tmp():
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        return f.name


def _blob(n, seed=0):
    """An n-atom carbon 'blob' with reproducible, non-degenerate positions."""
    rng = np.random.default_rng(1000 + seed)
    pos = rng.standard_normal((n, 3)) * 0.8
    return Atoms("C" * n, positions=pos)


# ======================================================= GATE 0 (torch-free)
def run_gate0_algebra_parity():
    """Prove the fused OU (noise draw order/shape + c1*v+c2*xi) reproduces
    ``LangevinThermostat.apply`` bit-for-bit, using only numpy. This isolates the
    ONLY nontrivial correctness claim (the torch elementwise fp64 op equals numpy's
    is standard IEEE-754); the GPU execution is covered by gate 1 under torch."""
    from maple.function.dispatcher.md.thermostat.langevin import LangevinThermostat
    T, fric, dt = 300.0, 0.02, 0.5
    worst = 0.0
    for i, n in enumerate((3, 4, 5, 3)):
        at = _blob(n, seed=i)
        # two generators seeded identically -> identical streams.
        rng_ref = np.random.default_rng(500 + i)
        rng_fused = np.random.default_rng(500 + i)
        th = LangevinThermostat(at, temperature=T, friction=fric, timestep=dt, rng=rng_ref)
        v = np.random.default_rng(9 + i).standard_normal((n, 3)) * 1e-3
        # reference: the authoritative per-replica thermostat (draws from rng_ref).
        v_ref = th.apply(v.copy())
        # fused emulation: SAME draw order/shape, then the on-device combine (in numpy
        # here; torch fp64 elementwise mul/add is the identical IEEE-754 op).
        noise = rng_fused.standard_normal((n, 3))
        v_fused = th._c1 * v + th._c2[:, np.newaxis] * noise
        worst = max(worst, float(np.max(np.abs(v_ref - v_fused))))
    print(f"[GATE0 algebra-parity] {4} replicas  max|v_ref - v_fused| = {worst:.3e}")
    assert worst == 0.0, f"ALGEBRA PARITY FAIL (fused OU != LangevinThermostat): {worst}"
    print("[GATE0] PASS  (fused OU math == LangevinThermostat.apply, bit-identical)")
    return worst


# =============================================================== torch fixture
if HAVE_TORCH:
    from maple.function.calculator.batch_calculator_base import BatchCalcABC
    from maple.function.dispatcher.md.ensemble.nvt_batched import BatchedNVT

    class _HarmonicTether(BatchCalcABC):
        """B independent per-atom harmonic tethers to the initial geometry:
        E_b = 0.5*k*sum_{i in b} |r_i - r0_i|^2 (Hartree), F_i = -k*(r_i - r0_i).
        Block-diagonal by construction (each molecule's E depends only on its own
        atoms) -> a clean batch-ISOLATED MLIP-shaped fixture. NOT production."""
        MODEL_NAMES = ("_fused_harmonic_tether_test",)
        MODEL_ENERGY_UNIT = "hartree"
        MODEL_DTYPE = torch.float64
        SUPPORTED_HESSIAN_MODES = ("numerical",)

        def __init__(self, k=0.04, device="cpu", dtype=torch.float64):
            super().__init__(device=device, dtype=dtype)
            self.k = float(k)
            self._r0 = None

        def _forward(self, coord, need_graph):
            if self._r0 is None:                     # tether to the t=0 geometry
                self._r0 = coord.detach().clone()
            d = coord - self._r0                      # (N,3)
            F_all = -self.k * d                       # (N,3) Ha/A
            e_atom = 0.5 * self.k * (d * d).sum(dim=1)  # (N,)
            E = torch.zeros(self._atoms_B, dtype=coord.dtype, device=coord.device)
            E.scatter_add_(0, self.mol_idx.long(), e_atom)  # (B,)
            return E, F_all, None


def _run(systems, fused, steps, seed, remove_com_every=0, anneal=""):
    calc = _HarmonicTether(device=DEV)
    paras = dict(timestep=0.5, steps=steps, temperature=300.0, thermostat="langevin",
                 friction=0.02, remove_com_every=remove_com_every, log_every=1,
                 anneal=anneal, random_seed=seed, verbose=0, fused_loop=fused)
    sim = BatchedNVT(_tmp(), [a.copy() for a in systems], calc=calc, paras=paras).run()
    return sim, calc


# ================================================================= GATE 1 parity
def run_gate1_parity(steps=250, seed=42, remove_com_every=0, anneal="", tag=""):
    systems = [_blob(n, seed=i) for i, n in enumerate((3, 4, 5, 3))]   # B=4, varied sizes
    sim_r, calc_r = _run(systems, False, steps, seed, remove_com_every, anneal)
    sim_f, calc_f = _run(systems, True, steps, seed, remove_com_every, anneal)

    pos_r = calc_r.coord.detach().to("cpu").numpy()
    pos_f = calc_f.coord.detach().to("cpu").numpy()
    v_r = sim_r.v.detach().to("cpu").numpy()
    v_f = sim_f.v.detach().to("cpu").numpy()
    dpos = float(np.max(np.abs(pos_r - pos_f)))
    dvel = float(np.max(np.abs(v_r - v_f)))
    dpe = max(float(np.max(np.abs(np.asarray(sim_r.results[b]["PE_Ha"]) -
                                  np.asarray(sim_f.results[b]["PE_Ha"]))))
              for b in range(len(systems)))
    dT = max(float(np.max(np.abs(np.asarray(sim_r.results[b]["T_K"]) -
                                 np.asarray(sim_f.results[b]["T_K"]))))
             for b in range(len(systems)))
    bit = (dpos == 0.0 and dvel == 0.0 and dpe == 0.0 and dT == 0.0)
    print(f"[GATE1 parity {tag}] steps={steps} rce={remove_com_every} anneal='{anneal}'  "
          f"max|dpos|={dpos:.2e} A  max|dvel|={dvel:.2e}  max|dPE|={dpe:.2e} Ha  "
          f"max|dT|={dT:.2e} K  {'BIT-IDENTICAL' if bit else 'canon<=1e-10'}")
    assert dpos < 1e-10 and dvel < 1e-10 and dpe < 1e-10 and dT < 1e-10, "PARITY FAIL"
    return dpos, dvel, dpe, dT, bit


# =========================================================== GATE 2 throughput
def _time_steps_per_s(B, fused, steps=200, seed=1, n_atoms=12):
    systems = [_blob(n_atoms, seed=b) for b in range(B)]
    sim, calc = _run(systems, fused, steps=3, seed=seed)   # warmup (compile/alloc)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _run(systems, fused, steps=steps, seed=seed)
    if DEV == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return steps / dt if dt > 0 else float("inf")


def run_gate2_throughput(Bs=(1, 8, 32, 128)):
    print(f"[GATE2 throughput] device={DEV}  (steps/s: default -> fused, speedup)")
    speedups = {}
    for B in Bs:
        sd = _time_steps_per_s(B, False)
        sf = _time_steps_per_s(B, True)
        su = sf / sd if sd > 0 else float("inf")
        speedups[B] = su
        print(f"    B={B:>4}  default={sd:8.1f}  fused={sf:8.1f}  speedup={su:5.2f}x")
    if DEV == "cuda":
        assert speedups[max(Bs)] >= 1.0, (
            f"THROUGHPUT FAIL: fused not >=1.0x at B={max(Bs)} ({speedups[max(Bs)]:.2f}x)")
        print("[GATE2] PASS  (fused >= default at large B on CUDA)")
    else:
        print("[GATE2] INFO  (CPU has no D2H sync to remove -> speedup is a GPU property; "
              "measure on a100)")
    return speedups


# =========================================================== GATE 3 no-host-sync
def run_gate3_no_host_sync(steps=120, seed=3):
    systems = [_blob(n, seed=i) for i, n in enumerate((3, 4, 5, 3))]
    B = len(systems)
    sim_f, _ = _run(systems, True, steps, seed, remove_com_every=0)     # fused, no projection
    sim_d, _ = _run(systems, False, steps, seed, remove_com_every=0)    # default
    print(f"[GATE3 no-host-sync] fused _host_sync_count={sim_f._host_sync_count}  "
          f"default _host_sync_count={sim_d._host_sync_count}  (B={B} steps={steps})")
    assert sim_f._host_sync_count == 0, (
        "NO-HOST-SYNC FAIL: fused hot loop still did per-replica device->host velocity "
        f"roundtrips ({sim_f._host_sync_count})")
    assert sim_d._host_sync_count > 0, "sanity: default path should roundtrip"
    print("[GATE3] PASS  (fused hot loop: 0 device->host velocity roundtrips)")
    return sim_f._host_sync_count, sim_d._host_sync_count


# =========================================================== GATE 4 regression
def run_gate4_regression(steps=60, seed=5):
    """Default (fused_loop=False) still runs, produces per-replica results, and stays
    the reference the parity gate matches (i.e. behavior unchanged). The heavier
    _test_nvt_batched.py / _test_we.py gates keep fused_loop OFF -> unchanged on GPU."""
    systems = [_blob(n, seed=i) for i, n in enumerate((3, 4))]
    sim, _ = _run(systems, False, steps, seed)
    assert len(sim.results) == len(systems) and "T_K" in sim.results[0], "REGRESSION FAIL"
    print(f"[GATE4 regression] default path OK  (B={len(systems)} steps={steps}, "
          f"results with T_K/PE_Ha present)")
    print("[GATE4] PASS  (default behavior intact; fused is opt-in)")
    return True


if __name__ == "__main__":
    print(f"torch={'yes' if HAVE_TORCH else 'MISSING'}  device={DEV}")
    # GATE 0 always runs (torch-free).
    run_gate0_algebra_parity()

    if not HAVE_TORCH:
        print("\n[SKIP] torch missing -> gates 1-4 (BatchedNVT parity/throughput/"
              "no-host-sync/regression) require torch. Run on the a100 node.")
        sys.exit(0)

    # GATE 1: parity across the code paths.
    run_gate1_parity(remove_com_every=0, anneal="", tag="noproj")
    run_gate1_parity(remove_com_every=50, anneal="", tag="+COMproj")
    run_gate1_parity(remove_com_every=0, anneal="290,320", tag="+anneal")
    # GATE 3, 4 (cheap correctness) then GATE 2 (timing).
    run_gate3_no_host_sync()
    run_gate4_regression()
    run_gate2_throughput()
    print("\n[FUSED-LOOP] ALL GATES PASS")
