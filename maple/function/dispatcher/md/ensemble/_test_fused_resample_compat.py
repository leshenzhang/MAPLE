"""Fused-loop <-> resampling/temperature-swap compatibility gate.

The GPU-opt Lever-3 fused (fully on-device) Langevin loop pre-builds the OU device
buffers ``_c1_dev``/``_c2_dev``/``_noise_np`` sized ``(B, nmax)`` at ``_prepare_buffers``
time.  Any ensemble that later CHANGES B (walker split/merge/recycle) or CHANGES a
replica's target temperature (annealing rung, T-ladder relabel swap) must refresh those
buffers, else the fused thermostat either shape-crashes (B changed) or SILENTLY injects
the wrong noise amplitude (T changed, no crash -> biased free energy).  Regressions:

  * WE   -- ``_rebuild`` on a resample changed B -> ``_c1_dev*v`` shape-crashed (B-153).
  * PA   -- ``_set_node_temperature`` retuned ._c2 each rung, _c2_dev stale -> biased lnZ.
  * REMD -- ``_apply_ladder`` / accepted swaps retuned per-replica ._c2, _c2_dev stale.

Because the fused OU substep is a BIT-EXACT drop-in for the per-replica LangevinThermostat
(fused-loop GATE0), a CORRECT fused run must reproduce the non-fused run BYTE-FOR-BYTE
through every resample / swap.  Each gate asserts fused-ON == fused-OFF bit-identity.

Self-contained toy ``_DoubleWellDimer`` fixture (torch, CPU ok, no MACE model).
"""
import numpy as np
import torch

from maple.function.dispatcher.md.ensemble._test_we import (
    _DoubleWellDimer, _dimer, _tmp, RMID, WHALF, H_BARRIER, R1, T_K, BOND_CV)
from maple.function.dispatcher.md.ensemble.weighted_ensemble import WeightedEnsemble
from maple.function.dispatcher.md.ensemble.population_annealing import PopulationAnnealing
from maple.function.dispatcher.md.ensemble.remd import REMD

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_BASE = dict(timestep=0.5, thermostat="langevin", friction=0.02,
             remove_com_every=0, remove_com=True, verbose=0)


def _calc():
    return _DoubleWellDimer(RMID, WHALF, H_BARRIER, device=DEV, dtype=torch.float64)


def _we(fused, seed=3):
    calc = _calc()
    r_target = 2.0
    edges = np.concatenate([np.linspace(1.0, r_target, 11), [5.0]])
    interior = edges[1:-1]
    source_bin = int(np.clip(np.digitize([R1], interior)[0], 0, len(edges) - 2))
    N = 24
    we = WeightedEnsemble(
        _tmp(), [_dimer(R1) for _ in range(N)], calc=calc, xi=BOND_CV,
        paras=dict(_BASE, temperature=T_K, fused_loop=fused, n_walkers=N, tau_steps=20,
                   n_iterations=200, walkers_per_bin=4, bin_edges=edges.tolist(),
                   source_bin=source_bin, target_bin=len(edges) - 2, recycle=True,
                   random_seed=seed, resample_seed=seed + 101)).run()
    return we.flux_total, we.mfpt_fs, calc.coord.detach().to("cpu").numpy().copy()


def _pa(fused, M=300, K=8, n_sweep=15, T0=600.0, Tt=300.0, seed=3):
    calc = _calc()
    pa = PopulationAnnealing(
        _tmp(), [_dimer(0.60) for _ in range(M)], calc=calc,
        paras=dict(_BASE, friction=0.05, remove_com_every=50, fused_loop=fused,
                   n_replicas=M, temp_start=T0, temp_target=Tt, n_anneal_steps=K,
                   n_sweep=n_sweep, resample_method="systematic",
                   random_seed=seed, resample_seed=seed + 101)).run()
    d2 = pa.observable_mean(lambda p: float(np.sum((p[0] - p[1]) ** 2)))
    return pa.lnZ_ratio, d2, calc.coord.detach().to("cpu").numpy().copy()


def _remd(fused, N=6, steps=1500, seed=7):
    calc = _calc()
    sim = REMD(_tmp(), _dimer(R1), calc=calc,
               paras=dict(_BASE, fused_loop=fused, n_replicas=N, temp_min=300.0,
                          temp_max=600.0, mode="temperature", exchange_every=50,
                          steps=steps, random_seed=seed, swap_seed=seed + 11)).run()
    return (int(np.sum(sim._n_accept)), calc.coord.detach().to("cpu").numpy().copy())


def run_gate_we():
    f_off, m_off, c_off = _we(False)
    f_on, m_on, c_on = _we(True)
    dc = float(np.max(np.abs(c_on - c_off)))
    print(f"[GATE WE]   fused ON vs OFF  max|dcoord|={dc:.2e}  d(flux)={abs(f_on-f_off):.2e}  "
          f"d(mfpt)={abs(m_on-m_off):.2e}")
    assert dc == 0.0 and f_on == f_off and m_on == m_off, "WE fused not bit-identical (resample rebuild)"
    print("[GATE WE]   PASS  (resample _rebuild refreshes fused buffers)")


def run_gate_pa():
    l_off, d_off, c_off = _pa(False)
    l_on, d_on, c_on = _pa(True)
    dc = float(np.max(np.abs(c_on - c_off)))
    print(f"[GATE PA]   fused ON vs OFF  max|dcoord|={dc:.2e}  d(lnZ)={abs(l_on-l_off):.2e}  "
          f"d(<d2>)={abs(d_on-d_off):.2e}")
    assert dc == 0.0 and l_on == l_off and d_on == d_off, "PA fused not bit-identical (rung-T _c2_dev stale)"
    print("[GATE PA]   PASS  (annealing rung refreshes fused _c2_dev)")


def run_gate_remd():
    acc_off, c_off = _remd(False)
    acc_on, c_on = _remd(True)
    dc = float(np.max(np.abs(c_on - c_off)))
    print(f"[GATE REMD] fused ON vs OFF  max|dcoord|={dc:.2e}  swaps_accepted OFF={acc_off} ON={acc_on}")
    assert acc_off > 0, "REMD gate saw no swaps -- relabel path not exercised (raise steps)"
    assert dc == 0.0 and acc_on == acc_off, "REMD fused not bit-identical (swap-relabel _c2_dev stale)"
    print("[GATE REMD] PASS  (ladder + swap relabel refresh fused _c2_dev)")


if __name__ == "__main__":
    print("fused<->resample compat gates  dev=", DEV)
    run_gate_we()
    run_gate_pa()
    run_gate_remd()
    print("\n[FUSED-RESAMPLE-COMPAT] ALL GATES PASS")
