"""
Weighted-Ensemble (WE) gates. Mirrors _test_remd.py / _test_nvt_batched.py.

  GATE 1  WEIGHT-CONSERVATION : Sum(weights) == 1 to 1e-12 after every split/merge
          (unit-tested on the pure-numpy ``we_split_merge`` over many random bins,
          and asserted at RUNTIME inside every WE._resample -- gate 2/4 exercising
          the driver without an AssertionError is the runtime proof).
  GATE 2  DEGENERATE-REDUCTION : 1 bin + walkers_per_bin == N (no split/merge, no
          recycling) === plain BatchedNVT of the same total steps + seed, to fp64
          machine precision (positions BIT-IDENTICAL; weights stay uniform 1/N).
  GATE 3  UNBIASED-ESTIMATOR  : the weighted ensemble average of an observable xi is
          invariant under resampling -- EXACTLY under a split (clones inherit xi),
          and in EXPECTATION under a merge (survivor picked prop.-to-weight); the
          closed-form expectation equals the pre-merge weighted sum exactly, and a
          Monte-Carlo mean over many merges converges to it.
  GATE 4  TOY-RATE            : on a bond-length double well, WE steady-state flux
          gives an MFPT consistent (order of magnitude) with a long plain-MD
          first-passage reference (same toy potential + thermostat).

Gates 2 & 4 use a self-contained analytic ``_DoubleWellDimer`` batch calc (a TEST
FIXTURE mirroring the ``_HarmonicBatch`` in batch_calculator_base.py's __main__) so
they need only torch (CPU ok), no MACE model file. A MACE-OFF dihedral variant is
noted at the bottom for the on-GPU physical smoke.
"""
import os, sys, tempfile
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from ase import Atoms

from maple.function.calculator.batch_calculator_base import BatchCalcABC
from maple.function.dispatcher.md.ensemble.nvt_batched import BatchedNVT
from maple.function.dispatcher.md.ensemble.weighted_ensemble import (
    WeightedEnsemble, we_split_merge)
from maple.function.dispatcher.md.utils import KELVIN_TO_HARTREE

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _tmp():
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        return f.name


# ======================================================= toy double-well fixture
class _DoubleWellDimer(BatchCalcABC):
    """B independent 2-atom 'molecules'; the bond length r = |r0 - r1| sits in a
    symmetric double well  E(r) = h * (((r - rmid)/w)^2 - 1)^2  (Hartree), with
    minima at r = rmid +/- w and a barrier of height h at r = rmid. Block-diagonal
    by construction (each molecule's E depends only on its own bond) -> a clean,
    batch-ISOLATED MLIP-shaped fixture for the WE rate gate. NOT a production
    calculator (mirrors _HarmonicBatch)."""
    MODEL_NAMES = ("_we_doublewell_test",)
    MODEL_ENERGY_UNIT = "hartree"           # E,F handed back already in Ha, Ha/A
    MODEL_DTYPE = torch.float64
    SUPPORTED_HESSIAN_MODES = ("numerical",)

    def __init__(self, rmid, w, h, device="cpu", dtype=torch.float64):
        super().__init__(device=device, dtype=dtype)
        self.rmid, self.w, self.h = float(rmid), float(w), float(h)

    def _forward(self, coord, need_graph):
        # homogeneous 2-atom molecules -> atom0 = even rows, atom1 = odd rows.
        P0, P1 = coord[0::2], coord[1::2]               # (B,3)
        d = P0 - P1
        r = torch.linalg.norm(d, dim=1).clamp_min(1e-8)  # (B,)
        s = ((r - self.rmid) / self.w) ** 2 - 1.0
        E = self.h * s * s                               # (B,) Hartree
        dEdr = 4.0 * self.h / (self.w ** 2) * s * (r - self.rmid)   # (B,) Ha/A
        u = d / r[:, None]
        F0 = -dEdr[:, None] * u                          # F on atom0 (Ha/A)
        F1 = dEdr[:, None] * u
        F_all = torch.empty((2 * self._atoms_B, 3), dtype=coord.dtype, device=coord.device)
        F_all[0::2] = F0
        F_all[1::2] = F1
        return E, F_all, None


def _dimer(r):
    """A 2-atom C-C 'molecule' with bond length r (Angstrom), placed along x."""
    return Atoms("CC", positions=[[0.0, 0.0, 0.0], [float(r), 0.0, 0.0]])


# toy double well: minima at r1, r2; barrier h ~ n_kT * kT(300 K).
T_K = 300.0
R1, R2 = 1.3, 2.1
RMID, WHALF = 0.5 * (R1 + R2), 0.5 * (R2 - R1)
N_KT = 3.5
H_BARRIER = N_KT * KELVIN_TO_HARTREE * T_K              # Hartree
BOND_CV = {"type": "distance", "atoms": [0, 1]}


# ============================================================ GATE 1 (numpy only)
def run_gate1_weights(trials=4000, seed=1):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        c = int(rng.integers(1, 12))                    # walkers currently in the bin
        m = int(rng.integers(1, 12))                    # target per bin
        w = rng.random(c)
        w = w / w.sum() * rng.random()                  # arbitrary (bin) sub-total
        sub = float(w.sum())
        out = we_split_merge(list(range(c)), w.tolist(), m, rng)
        assert len(out) == m, (c, m, len(out))
        got = float(sum(x[1] for x in out))
        worst = max(worst, abs(got - sub))
    print(f"[GATE1 weight-conservation] {trials} random split/merge bins  "
          f"max |Sum w_out - Sum w_in| = {worst:.3e}")
    assert worst < 1e-12, "WEIGHT NOT CONSERVED across split/merge"
    print("[GATE1] PASS  (weight conserved to < 1e-12 across every split/merge)")
    return worst


# ============================================================ GATE 3 (numpy only)
def run_gate3_unbiased(seed=7):
    rng = np.random.default_rng(seed)

    # --- SPLIT: clones inherit the parent xi -> weighted mean EXACTLY invariant ---
    ids = [0, 1]
    w = np.array([0.3, 0.7])
    xi = {0: -1.2, 1: 2.4}
    pre = float(sum(w[i] * xi[ids[i]] for i in range(len(ids))))
    out = we_split_merge(ids, w.tolist(), 5, rng)        # c=2 -> m=5 (all splits)
    post = float(sum(wt * xi[sid] for sid, wt in out))
    print(f"[GATE3 split] pre=<xi>w={pre:.15f}  post={post:.15f}  |d|={abs(post-pre):.2e}")
    assert abs(post - pre) < 1e-14 and len(out) == 5, "SPLIT changed the weighted sum"

    # --- MERGE: survivor prop.-to-weight -> weighted mean invariant IN EXPECTATION.
    # closed-form expectation over the survivor choice equals the pre-merge sum
    # EXACTLY; verify analytically + by Monte-Carlo convergence.
    wa, wb, wc = 0.5, 0.3, 0.2                            # 3 walkers, merge -> 2
    xa, xb, xc = 0.4, 1.9, -0.7
    pre = wa * xa + wb * xb + wc * xc
    # we_split_merge merges the TWO LIGHTEST (b,c): survivor b w.p. wb/(wb+wc).
    W = wb + wc
    exp_pair = (wb / W) * (W * xb) + (wc / W) * (W * xc)  # == wb*xb + wc*xc (exact)
    exp_total = wa * xa + exp_pair
    print(f"[GATE3 merge] closed-form E[<xi>w_post]={exp_total:.15f}  pre={pre:.15f}  "
          f"|d|={abs(exp_total-pre):.2e}")
    assert abs(exp_total - pre) < 1e-14, "MERGE expectation != pre-merge weighted sum"

    # Monte-Carlo: average the post-merge weighted sum over many random survivors.
    xmap = {0: xa, 1: xb, 2: xc}
    K = 200000
    acc = 0.0
    for _ in range(K):
        out = we_split_merge([0, 1, 2], [wa, wb, wc], 2, rng)
        acc += sum(wt * xmap[sid] for sid, wt in out)
    mc = acc / K
    print(f"[GATE3 merge] Monte-Carlo mean over {K} merges = {mc:.6f}  (pre={pre:.6f}, "
          f"err={abs(mc-pre):.2e})")
    assert abs(mc - pre) < 5e-3, "MERGE Monte-Carlo mean drifted from the pre-merge sum"
    print("[GATE3] PASS  (split exact; merge unbiased in expectation)")
    return exp_total, mc


# ============================================================ GATE 2 (torch fixture)
def run_gate2_degenerate(N=6, tau=25, niter=8, seed=42):
    calc = _DoubleWellDimer(RMID, WHALF, H_BARRIER, device=DEV, dtype=torch.float64)
    starts = [_dimer(R1 + 0.03 * b) for b in range(N)]    # near-well, per-walker offset
    paras = dict(timestep=0.5, temperature=T_K, thermostat="langevin", friction=0.02,
                 remove_com_every=100, random_seed=seed, verbose=0)

    # (1) plain BatchedNVT over the SAME N walkers, steps = tau*niter.
    bn = BatchedNVT(_tmp(), [a.copy() for a in starts], calc=calc,
                    paras=dict(paras, steps=tau * niter)).run()
    pos_ref = calc.coord.detach().to("cpu").numpy().copy()

    # (2) WE: 1 bin, walkers_per_bin == N, no target -> resample is a pure NO-OP.
    we = WeightedEnsemble(
        _tmp(), [a.copy() for a in starts], calc=calc, xi=BOND_CV,
        paras=dict(paras, n_walkers=N, tau_steps=tau, n_iterations=niter,
                   walkers_per_bin=N, n_bins=1, xi_min=0.0, xi_max=10.0,
                   target_bin=None)).run()
    pos_we = calc.coord.detach().to("cpu").numpy().copy()

    dpos = float(np.max(np.abs(pos_ref - pos_we)))
    w_dev = float(np.max(np.abs(we.weights - 1.0 / N)))
    n_final = int(we.n_walkers_hist[-1]) if len(we.n_walkers_hist) else N
    print(f"[GATE2 degenerate] N={N} steps={tau*niter} seed={seed}  "
          f"max|dpos|={dpos:.3e} A  max|w-1/N|={w_dev:.3e}  final walkers={n_final}")
    assert dpos < 1e-10, "DEGENERATE FAIL (WE no-op != plain BatchedNVT positions)"
    assert w_dev < 1e-12 and n_final == N, "DEGENERATE FAIL (weights not uniform 1/N)"
    print("[GATE2] PASS  (1-bin/m=N WE === plain BatchedNVT; weights uniform 1/N)")
    return dpos, w_dev


# ============================================================ GATE 4 (torch fixture)
def _brute_force_mfpt(calc, r_target, paras, Bref=64, max_steps=40000, seed=11):
    """Long plain-MD first-passage reference: Bref dimers started in the near well;
    mean first time the bond reaches r_target (far basin). Reuses BatchedNVT's step
    primitive (no reimplemented integrator)."""
    starts = [_dimer(R1) for _ in range(Bref)]
    bn = BatchedNVT(_tmp(), starts, calc=calc,
                    paras=dict(paras, steps=max_steps, random_seed=seed))
    bn._log_parameters(); bn._prepare_buffers()
    v = bn.v
    E, F = bn._forces_au()
    v = v - 0.5 * F / bn.mass * bn.dt_au                  # langevin: std -> carried
    fpt = np.full(Bref, -1.0)
    for step in range(1, max_steps + 1):
        v, E, F = bn._step_langevin(v, F, step)
        c = calc.coord.detach().to("cpu").numpy()
        r = np.linalg.norm(c[0::2] - c[1::2], axis=1)     # (Bref,) bond lengths
        newly = (fpt < 0) & (r >= r_target)
        fpt[newly] = step
    crossed = fpt[fpt > 0]
    dt_fs = float(paras["timestep"])
    mfpt = float(np.mean(crossed) * dt_fs) if crossed.size else float("inf")
    return mfpt, crossed.size / Bref


def run_gate4_rate(seed=3):
    calc = _DoubleWellDimer(RMID, WHALF, H_BARRIER, device=DEV, dtype=torch.float64)
    paras = dict(timestep=0.5, temperature=T_K, thermostat="langevin", friction=0.02,
                 remove_com_every=0, remove_com=True, verbose=0)

    # fixed bins spanning the two wells; the LAST bin is a wide catch-all [2.0, 5.0)
    # so ANY crossing r >= r_target lands in the single target bin (incl. the far
    # minimum r2=2.1). source = near-well bin, target = last (far-basin) bin.
    r_target = 2.0                                        # past the barrier (rmid=1.7)
    edges = np.concatenate([np.linspace(1.0, r_target, 11), [5.0]])  # 11 x 0.1 A + catch-all
    interior = edges[1:-1]
    source_bin = int(np.clip(np.digitize([R1], interior)[0], 0, len(edges) - 2))
    target_bin = len(edges) - 2                           # last bin = [2.0, 5.0)

    N = 24
    we = WeightedEnsemble(
        _tmp(), [_dimer(R1) for _ in range(N)], calc=calc, xi=BOND_CV,
        paras=dict(paras, n_walkers=N, tau_steps=20, n_iterations=200,
                   walkers_per_bin=4, bin_edges=edges.tolist(),
                   source_bin=source_bin, target_bin=target_bin, recycle=True,
                   random_seed=seed, resample_seed=seed + 101)).run()
    mfpt_we = we.mfpt_fs

    mfpt_bf, frac = _brute_force_mfpt(calc, r_target, paras)
    ratio = mfpt_we / mfpt_bf if (np.isfinite(mfpt_we) and mfpt_bf > 0) else float("inf")
    print(f"\n[GATE4 rate] barrier ~ {N_KT:.1f} kT   target r>= {r_target} A "
          f"(source bin {source_bin} -> target bin {target_bin})")
    print(f"  WE steady-state:  flux_total={we.flux_total:.4e}  MFPT_we ={mfpt_we:.3e} fs "
          f"({mfpt_we/1000:.3f} ps)")
    print(f"  brute-force ref:  crossed_frac={frac:.2f}      MFPT_bf ={mfpt_bf:.3e} fs "
          f"({mfpt_bf/1000:.3f} ps)")
    print(f"  MFPT_we / MFPT_bf = {ratio:.3f}  (order-of-magnitude agreement expected)")
    # algorithm-correctness gate: statistical / order-of-magnitude, NOT bit-level.
    assert frac > 0.3, "brute-force reference saw too few crossings (raise max_steps / lower barrier)"
    assert 0.2 < ratio < 5.0, "WE MFPT not within an order of magnitude of the MD reference"
    print("[GATE4] PASS  (WE flux MFPT consistent with the plain-MD reference)")
    return mfpt_we, mfpt_bf, ratio


if __name__ == "__main__":
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "dev", DEV)
    run_gate1_weights()
    run_gate3_unbiased()
    dpos, wdev = run_gate2_degenerate()
    mfpt_we, mfpt_bf, ratio = run_gate4_rate()
    print(f"\n[WE RESULT] gate2 max|dpos|={dpos:.3e} A  max|w-1/N|={wdev:.3e}")
    print(f"[WE RESULT] gate4 MFPT_we={mfpt_we:.3e} fs  MFPT_bf={mfpt_bf:.3e} fs  "
          f"ratio={ratio:.3f}")
    print("[WE RESULT] ALL GATES PASS")
