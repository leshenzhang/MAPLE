"""
TPS aimless-shooting gates. Mirrors _test_remd.py methodology.

  GATE 1  TIME-REVERSAL : integrate NVE forward then backward (negated velocities)
          returns to the start to MD tolerance -- validates the backward leg of
          two-way shooting. (MACE-OFF ethanol, the real BatchedNVT VV integrator.)
  GATE 2  ISOLATION     : the N parallel shots are independent -- perturbing one
          shot's initial velocities does NOT change any other shot's trajectory
          (batch isolation, like the other batched drivers). (MACE-OFF ethanol.)
  GATE 3  COMMITTOR     : on a double-well toy with an ANALYTIC committor, aimless
          shooting from the barrier top gives pB ~ 0.5, from near A ~ 0, near B
          ~ 1 -- proves shooting + state detection. Run through the real TPS
          driver + BatchedNVT (torch toy calc), PLUS a pure-numpy local check
          (no torch/GPU) that reuses the driver's OrderParameter + committor.
  GATE 4  REACTIVE-PATH : accepted paths actually connect A<->B (endpoint states
          satisfy the lambda thresholds); rejected ones do not. (Toy double well.)

GPU note: gates 1/2/3(driver)/4 need torch (run on GPU/CPU-with-torch later, like
the other _test_*). Gate 3's numpy check runs anywhere numpy+ase are present.
"""
import os
import sys
import tempfile

import numpy as np

from maple.function.dispatcher.md.ensemble.tps import (
    TPS, OrderParameter, committor_fraction)

MODEL = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
DEV = "cuda"


def _tmp():
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        return f.name


# =============================================================================
# Toy 1D double well as a batch calculator (torch; drives the REAL TPS + BatchedNVT).
# Diatomic per replica; order parameter = signed x-separation rx = x1 - x0.
#   V(rx) = h * ((rx/w0)^2 - 1)^2   [Hartree]   -> wells at rx = +/- w0, barrier at 0.
# Purely local (block-diagonal) => batch-isolated. Forces along x only (Ha/Angstrom).
# =============================================================================
class ToyDoubleWellCalc:
    SUPPORTS_COUPLING = False          # pure-local; block-diagonal batch is isolated
    SUPPORTS_PBC = False
    batch_isolated = True
    implemented_properties = ("energy", "forces")

    def __init__(self, h=0.02, w0=1.0, device="cpu", dtype=None):
        import torch
        self._torch = torch
        self.device = torch.device(device)
        self.dtype = dtype if dtype is not None else torch.float64
        self.h = float(h)
        self.w0 = float(w0)
        self._prepared = False
        self._backup = None

    def prepare(self, atoms_list, fixed_nmax=None):
        torch = self._torch
        self.B = len(atoms_list)
        self.n_b = np.array([len(a) for a in atoms_list], dtype=int)
        if not np.all(self.n_b == 2):
            raise ValueError("ToyDoubleWellCalc expects diatomic replicas (2 atoms).")
        self.N_atoms = int(self.n_b.sum())
        self.Nmax = int(self.n_b.max())
        self.nmax_dof = (3 * self.Nmax) if fixed_nmax is None else int(fixed_nmax)
        pos = np.concatenate([a.get_positions() for a in atoms_list], axis=0)
        self.coord = torch.tensor(pos, dtype=self.dtype, device=self.device)
        self._ptr = np.concatenate([[0], np.cumsum(self.n_b)]).astype(int)
        self._backup = None
        self._prepared = True

    def step_cart_(self, s_cart):
        disp = s_cart.reshape(self.B, -1, 3)[:, :2, :].reshape(-1, 3).to(
            self.device, self.dtype)
        self.coord.add_(disp)

    def set_coords_(self, coord):
        self.coord.copy_(coord.to(self.device, self.dtype))

    def backup_coords(self):
        if self._prepared:
            self._backup = self.coord.clone()

    def restore_coords(self):
        if self._backup is not None:
            self.coord.copy_(self._backup)

    def get_ef_gpu(self):
        torch = self._torch
        p = self.coord.view(self.B, 2, 3)
        rx = p[:, 1, 0] - p[:, 0, 0]                      # (B,) signed x-separation
        s = (rx / self.w0) ** 2 - 1.0                     # (B,)
        E = self.h * s * s                                # (B,) Hartree
        dEdrx = 4.0 * self.h * s * rx / (self.w0 ** 2)    # dE/drx  (Ha/Angstrom)
        F = torch.zeros(self.B, self.nmax_dof, dtype=self.dtype, device=self.device)
        F[:, 0] = dEdrx                                   # F0x = -dE/dx0 = +dEdrx
        F[:, 3] = -dEdrx                                  # F1x = -dE/dx1 = -dEdrx
        return E, F

    def isolation_check(self, perturb=0.05):
        E0, _ = self.get_ef_gpu()
        self.backup_coords()
        self.coord[0, 0] += perturb
        E1, _ = self.get_ef_gpu()
        self.restore_coords()
        return float((E1[1:] - E0[1:]).abs().max().item())


def _diatomic(rx):
    from ase import Atoms
    return Atoms("N2", positions=[[0.0, 0.0, 0.0], [float(rx), 0.0, 0.0]])


# order parameter for the toy: lambda = signed x-separation; A = rx<=-0.5, B = rx>=0.5.
def _toy_op():
    return OrderParameter(kind="custom",
                          func=lambda P: float(P[1, 0] - P[0, 0]),
                          a_max=-0.5, b_min=0.5)


# =============================================================================
def ethanol(shift=(0.0, 0.0, 0.0)):
    from ase.build import molecule
    at = molecule("CH3CH2OH")
    at.positions = at.positions + np.asarray(shift)
    return at


def _maceoff_calc():
    import torch
    from maple.function.calculator.mace._maceoff_batch_calculator import MaceOffBatchCalc
    return MaceOffBatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64)


# ---------------------------------------------------------------- GATE 1
def run_time_reversal(steps=50, seed=11, N=4, tol=1e-6):
    import torch
    torch.set_default_dtype(torch.float64)
    bc = _maceoff_calc()
    a = ethanol()
    op = OrderParameter(kind="distance", indices="0,1", a_max=0.5, b_min=5.0)  # dummy
    tps = TPS(_tmp(), a, calc=bc, order_parameter=op,
              paras=dict(n_shots=N, segment_steps=steps, timestep=0.5,
                         temperature=300.0, random_seed=seed, verbose=0))
    dpos, dvel = tps.time_reversal_residual(steps)
    print(f"[GATE1 TIME-REVERSAL] N={N} steps={steps}  max|dpos|={dpos:.3e} A  "
          f"max|dvel|={dvel:.3e} au  (tol {tol:.0e})")
    assert dpos < tol and dvel < tol, "TIME-REVERSAL FAIL (backward != negated forward)"
    print("[GATE1] PASS  (forward -> negate v -> backward returns to start; VV reversible)")
    return dpos, dvel


# ---------------------------------------------------------------- GATE 2
def run_isolation(steps=40, seed=23, N=4, perturb=0.1):
    import torch
    torch.set_default_dtype(torch.float64)
    bc = _maceoff_calc()
    a = ethanol()
    op = OrderParameter(kind="distance", indices="0,1", a_max=0.5, b_min=5.0)
    tps = TPS(_tmp(), a, calc=bc, order_parameter=op,
              paras=dict(n_shots=N, segment_steps=steps, timestep=0.5,
                         temperature=300.0, random_seed=seed, verbose=0))
    tps._ensure_prepared()
    leak = bc.isolation_check(perturb=perturb)               # cross-replica energy leak

    x0 = a.get_positions()
    v0 = [tps._draw_shoot_vel(b) for b in range(N)]

    # run A: propagate all shots from x0 with v0.
    tps._set_coords([x0] * N)
    tps._load_v_std([v.copy() for v in v0])
    tps._propagate(steps, detect=False)
    posA = tps._positions_per_replica()

    # run B: perturb ONLY shot 0's initial velocities; others identical to run A.
    v0b = [v.copy() for v in v0]
    v0b[0] = v0b[0] + 0.37                                    # arbitrary kick to shot 0
    tps._set_coords([x0] * N)
    tps._load_v_std(v0b)
    tps._propagate(steps, detect=False)
    posB = tps._positions_per_replica()

    d_shot0 = float(np.max(np.abs(posB[0] - posA[0])))       # SHOULD change
    d_others = max(float(np.max(np.abs(posB[b] - posA[b]))) for b in range(1, N))
    print(f"[GATE2 ISOLATION] N={N}  calc isolation_dE={leak:.3e} Ha   "
          f"perturbed shot0 moved={d_shot0:.3e} A   other shots max|d|={d_others:.3e} A")
    assert leak < 1e-6, "ISOLATION FAIL (calc leaks across the batch)"
    assert d_shot0 > 1e-8, "shot 0 velocity perturbation had no effect (test broken)"
    # < 1e-9: real cross-shot leakage would be O(0.1 A) (a 0.37-au kick to a bonded
    # atom); the residual here is only GPU scatter-atomics run-to-run noise (~1e-12).
    assert d_others < 1e-9, "ISOLATION FAIL (perturbing shot 0 changed another shot)"
    print("[GATE2] PASS  (N shots independent: perturb one -> others unchanged)")
    return leak, d_shot0, d_others


# ---------------------------------------------------------------- GATE 3 (driver)
def run_committor_toy(seg=1200, n_shots=200, seed=7, tol=0.15):
    import torch
    torch.set_default_dtype(torch.float64)
    op = _toy_op()
    results = {}
    for name, rx0, lo, hi in (("barrier(TS)", 0.0, 0.5 - tol, 0.5 + tol),
                              ("near A", -0.4, -1.0, tol),
                              ("near B", 0.4, 1.0 - tol, 2.0)):
        calc = ToyDoubleWellCalc(h=0.02, w0=1.0, device="cpu", dtype=torch.float64)
        at = _diatomic(rx0)
        tps = TPS(_tmp(), at, calc=calc, order_parameter=op,
                  paras=dict(n_shots=n_shots, segment_steps=seg, timestep=0.5,
                             temperature=300.0, op_every=5, random_seed=seed, verbose=0))
        pB, first = tps.committor(at)
        ncommit = sum(1 for s in first if s is not None)
        results[name] = pB
        print(f"[GATE3 COMMITTOR-driver] {name:>11}: rx0={rx0:+.2f}  pB={pB:.3f}  "
              f"committed={ncommit}/{n_shots}  (expect {lo:.2f}..{hi:.2f})")
        assert not np.isnan(pB) and lo <= pB <= hi, f"committor {name} out of range"
    assert results["near A"] < 0.5 < results["near B"], "committor not monotone A<TS<B"
    print("[GATE3-driver] PASS  (committor ~0.5 at TS, ~0 near A, ~1 near B)")
    return results


# ---------------------------------------------------------------- GATE 3 (numpy-only)
def run_committor_numpy(n_shots=400, seed=3, tol=0.15):
    """Pure-numpy (no torch/GPU) analytic double-well committor. Reuses the DRIVER's
    OrderParameter (state detection) + committor_fraction (shooting count). Natural
    units: V(rx)=h*(rx^2-1)^2, wells +/-1, barrier at 0, mu=1, kT<<h so off-barrier
    shots are trapped. NVE leapfrog + first-hitting -> pB(0)~0.5, pB(-)~0, pB(+)~1."""
    op = _toy_op()               # A: rx<=-0.5, B: rx>=0.5 (same code the driver uses)
    h, mu, kT, dt, nsteps = 1.0, 1.0, 0.05, 0.005, 4000

    def dVdrx(rx):
        return 4.0 * h * rx * (rx * rx - 1.0)

    def committor_1d(rx0, rng):
        firsts = []
        for _ in range(n_shots):
            rx = float(rx0)
            v = rng.normal(0.0, np.sqrt(kT / mu))         # Maxwell-Boltzmann (1 DOF)
            a = -dVdrx(rx) / mu
            first = None
            for _s in range(nsteps):
                v += 0.5 * a * dt
                rx += v * dt
                a = -dVdrx(rx) / mu
                v += 0.5 * a * dt
                b = op.basin(op.value(np.array([[0.0, 0.0, 0.0], [rx, 0.0, 0.0]])))
                if b is not None:
                    first = b
                    break
            firsts.append(first)
        return committor_fraction(firsts), sum(1 for s in firsts if s is not None)

    rng = np.random.default_rng(seed)
    pB0, c0 = committor_1d(0.0, rng)
    pBa, ca = committor_1d(-0.4, rng)
    pBb, cb = committor_1d(0.4, rng)
    print(f"[GATE3 COMMITTOR-numpy] barrier pB={pB0:.3f} (committed {c0}/{n_shots}), "
          f"near A pB={pBa:.3f} ({ca}), near B pB={pBb:.3f} ({cb})")
    assert abs(pB0 - 0.5) <= tol, "numpy committor at barrier != ~0.5"
    assert pBa <= tol, "numpy committor near A != ~0"
    assert pBb >= 1.0 - tol, "numpy committor near B != ~1"
    print("[GATE3-numpy] PASS  (analytic double-well committor via driver OP + counter)")
    return pB0, pBa, pBb


# ---------------------------------------------------------------- GATE 4
def run_reactive_paths(seg=1000, n_shots=32, iters=12, seed=5):
    import torch
    torch.set_default_dtype(torch.float64)
    op = _toy_op()
    calc = ToyDoubleWellCalc(h=0.02, w0=1.0, device="cpu", dtype=torch.float64)
    at = _diatomic(0.0)          # start shooting at the barrier top (TS)
    tps = TPS(_tmp(), at, calc=calc, order_parameter=op,
              paras=dict(n_shots=n_shots, segment_steps=seg, timestep=0.5,
                         temperature=300.0, op_every=5, n_iterations=iters,
                         random_seed=seed, verbose=0)).run()

    n_acc = len(tps.paths)
    # (a) every ACCEPTED path connects A<->B (endpoint basins differ and cover {A,B}).
    acc_ok = all(r["reactive"] and {r["fwd_basin"], r["bwd_basin"]} == {"A", "B"}
                 for r in tps.paths)
    # (b) every REJECTED shot does NOT connect A<->B (same basin or one half uncommitted).
    rej_ok = all((not r["reactive"]) and {r["fwd_basin"], r["bwd_basin"]} != {"A", "B"}
                 for r in tps.all_shots if not r["reactive"])

    # (c) DETERMINISTIC rejected population: two-way shots from DEEP INSIDE basin A
    #     (rx=-1.0) -- both halves first-hit A -> not reactive -> must all be rejected.
    recs_A, pB_A, _ = tps._shoot_iteration(_diatomic(-1.0).get_positions())
    basinA_all_rejected = all((not r["reactive"])
                              and {r["fwd_basin"], r["bwd_basin"]} != {"A", "B"}
                              for r in recs_A)
    n_rej = sum(1 for r in tps.all_shots if not r["reactive"]) + \
        sum(1 for r in recs_A if not r["reactive"])

    print(f"[GATE4 REACTIVE-PATH] iters={iters} shots/iter={n_shots} trials={tps.n_trials}  "
          f"accepted(reactive)={n_acc}  rejected(total)={n_rej}  "
          f"acc_frac={tps.acceptance_rate:.3f}")
    print(f"    accepted-all-connect-A<->B={acc_ok}   rejected-none-connect-A<->B={rej_ok}   "
          f"shots-from-inside-A-all-rejected={basinA_all_rejected}   "
          f"TS-ensemble(|pB-0.5|<={tps.params.ts_committor_tol})={len(tps.ts_ensemble)}")
    assert n_acc > 0, "no reactive path accepted (test needs a shootable TS)"
    assert acc_ok, "GATE4 FAIL: an accepted path does not connect A<->B"
    assert rej_ok, "GATE4 FAIL: a rejected shot connects A<->B"
    assert basinA_all_rejected, "GATE4 FAIL: shots from inside basin A were accepted"
    print("[GATE4] PASS  (accepted paths connect A->B via thresholds; non-reactive rejected)")
    return n_acc, n_rej, tps.acceptance_rate


if __name__ == "__main__":
    # GATE 3 numpy check runs with NO torch/GPU (always).
    run_committor_numpy()

    if "--numpy-only" in sys.argv:
        print("\n[TPS RESULT] numpy committor gate PASS (torch gates skipped)")
        raise SystemExit(0)

    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    dpos, dvel = run_time_reversal()
    leak, d0, dov = run_isolation()
    ces = run_committor_toy()
    nacc, nrej, af = run_reactive_paths()
    print(f"\n[TPS RESULT] gate1 time-reversal max|dpos|={dpos:.3e} A max|dvel|={dvel:.3e} au")
    print(f"[TPS RESULT] gate2 isolation leak={leak:.3e} Ha  others_delta={dov:.3e} A")
    print(f"[TPS RESULT] gate3 committor TS/A/B="
          f"{ces['barrier(TS)']:.2f}/{ces['near A']:.2f}/{ces['near B']:.2f}")
    print(f"[TPS RESULT] gate4 accepted={nacc} rejected={nrej} reactive_frac={af:.3f}")
    print("[TPS RESULT] ALL GATES PASS")
