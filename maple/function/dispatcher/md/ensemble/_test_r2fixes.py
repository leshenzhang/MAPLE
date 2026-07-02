"""
B-81 R2 correctness-fix gates for the batched NVT kernel (a100; MACE-OFF fp64).

  DETERMINISM : dump final coords/vel/last-PE for the DEFAULT path (langevin B=2 +
                v-rescale B=1, fixed seed, remove_angular_every=0, default
                log_every) to an .npz. Running this on base b75ec55 (integ
                worktree) and on feat/md-r2fixes must give BIT-IDENTICAL dumps ->
                the three fixes do NOT perturb the default trajectory.
  FIX1 (angular): with remove_angular_every=1, the angular momentum L of each
                replica measured in the CURRENT frame (current positions from the
                calc + final velocities) must be ~0. The pre-fix code projected
                against the FROZEN t=0 geometry, so current-frame L was NOT zero.
  FIX3 (cadence): with steps S and log_every L, the recorded history length must
                be S/L (final step always recorded); PE_Ha[-1] == final-step E.
  FIX2 (FixAtoms): constructing BatchedNVT with an ASE FixAtoms constraint must
                emit a clear RuntimeWarning (the batched path does not honor it).

Usage:
  python _test_r2fixes.py dump  <out.npz>   # determinism dump (run on base + fix)
  python _test_r2fixes.py checks            # fix1/fix2/fix3 targeted gates
"""
import os, sys, warnings
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from ase.build import molecule
from ase.constraints import FixAtoms

from maple.function.calculator.mace._maceoff_batch_calculator import MaceOffBatchCalc
from maple.function.dispatcher.md.ensemble.nvt_batched import BatchedNVT

MODEL = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
DEV = "cuda"
AMU_TO_AU = 1822.888486209
ANG_TO_BOHR = 1.0 / 0.52917721067


def ethanol(shift=(0.0, 0.0, 0.0)):
    at = molecule("CH3CH2OH")
    at.positions = at.positions + np.asarray(shift)
    return at


def _calc():
    return MaceOffBatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64)


def dump_default(out_npz):
    """DEFAULT-path determinism dump: langevin B=2 + v-rescale B=1, fixed seed."""
    # langevin B=2
    a0, a1 = ethanol(), ethanol(shift=(60.0, 0, 0))
    bc = _calc()
    p_lan = dict(timestep=0.5, steps=120, temperature=300.0, thermostat="langevin",
                 friction=0.02, remove_com_every=100, random_seed=7, verbose=0)
    sim = BatchedNVT("/tmp/_r2_lan.out", [a0, a1], calc=bc, paras=p_lan).run()
    n0 = len(a0)
    lan_pos = bc.coord.detach().cpu().numpy().copy()
    lan_vel = sim.v.detach().cpu().numpy().copy()
    lan_pe = np.array([sim.results[b]["PE_Ha"][-1] for b in range(2)])

    # v-rescale B=1
    av = ethanol()
    bc2 = _calc()
    p_vr = dict(timestep=0.5, steps=120, temperature=300.0, thermostat="v-rescale",
                tau_t=100.0, remove_com_every=100, random_seed=42, verbose=0)
    sim2 = BatchedNVT("/tmp/_r2_vr.out", [av], calc=bc2, paras=p_vr).run()
    vr_pos = bc2.coord.detach().cpu().numpy().copy()
    vr_vel = sim2.v.detach().cpu().numpy().copy()
    vr_pe = np.array([sim2.results[0]["PE_Ha"][-1]])

    np.savez(out_npz, lan_pos=lan_pos, lan_vel=lan_vel, lan_pe=lan_pe,
             vr_pos=vr_pos, vr_vel=vr_vel, vr_pe=vr_pe)
    print(f"[DUMP] wrote {out_npz}  lan_pe={lan_pe}  vr_pe={vr_pe}")


def check_fix1_angular():
    """remove_angular_every=1 -> CURRENT-frame angular momentum ~0 after run."""
    a0 = ethanol(); a1 = ethanol(shift=(40.0, 0, 0))
    bc = _calc()
    p = dict(timestep=0.5, steps=60, temperature=400.0, thermostat="v-rescale",
             tau_t=50.0, remove_com_every=1, remove_angular_every=1,
             random_seed=11, verbose=0)
    sim = BatchedNVT("/tmp/_r2_ang.out", [a0, a1], calc=bc, paras=p).run()
    cur = bc.coord.detach().cpu().numpy()
    cum = np.concatenate(([0], np.cumsum([len(a0), len(a1)]))).astype(int)
    t0_pos = [a0.get_positions(), a1.get_positions()]   # FROZEN t=0 geometry
    ok = True
    for b, at in enumerate([a0, a1]):
        n = len(at)
        m = at.get_masses() * AMU_TO_AU
        v = sim.v[b, :3 * n].detach().cpu().numpy().reshape(n, 3)  # au
        # CURRENT frame
        rc = (cur[cum[b]:cum[b + 1]] * ANG_TO_BOHR)
        comc = (m[:, None] * rc).sum(0) / m.sum()
        Lc = (m[:, None] * np.cross(rc - comc, v)).sum(0)
        # FROZEN t=0 frame (what the buggy code zeroed instead)
        rf = t0_pos[b] * ANG_TO_BOHR
        comf = (m[:, None] * rf).sum(0) / m.sum()
        Lf = (m[:, None] * np.cross(rf - comf, v)).sum(0)
        nLc, nLf = float(np.linalg.norm(Lc)), float(np.linalg.norm(Lf))
        print(f"[FIX1] rep{b}: |L|_current={nLc:.3e}  |L|_frozen-t0={nLf:.3e} (au)")
        ok = ok and (nLc < 1e-8)
    assert ok, "FIX1 FAIL: current-frame angular momentum not removed"
    print("[FIX1] PASS (current-frame L ~ 0)")


def check_fix3_cadence():
    """history length == steps/log_every; PE_Ha[-1] == final-step energy."""
    for steps, le in [(200, 50), (100, 25), (90, 30)]:
        a0 = ethanol()
        bc = _calc()
        p = dict(timestep=0.5, steps=steps, temperature=300.0, thermostat="langevin",
                 friction=0.02, remove_com_every=100, log_every=le,
                 random_seed=3, verbose=0)
        sim = BatchedNVT("/tmp/_r2_cad.out", [a0], calc=bc, paras=p).run()
        L = len(sim.results[0]["T_K"])
        expect = steps // le
        print(f"[FIX3] steps={steps} log_every={le}: history_len={L} expect={expect}")
        assert L == expect, f"FIX3 FAIL: len {L} != {expect}"
        # final-step energy consistency: recompute E at the final geometry
        E_Ha, _ = bc.get_ef_gpu()
        assert abs(float(sim.results[0]["PE_Ha"][-1]) - float(E_Ha[0])) < 1e-9, \
            "FIX3 FAIL: PE_Ha[-1] != final-step energy"
    print("[FIX3] PASS (cadence-gated, final recorded, values consistent)")


def check_fix2_fixatoms():
    """ASE FixAtoms on an input replica -> RuntimeWarning emitted."""
    a0 = ethanol()
    a0.set_constraint(FixAtoms(indices=[0, 1]))
    a1 = ethanol(shift=(40, 0, 0))
    bc = _calc()
    p = dict(timestep=0.5, steps=2, temperature=300.0, thermostat="langevin",
             random_seed=1, verbose=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BatchedNVT("/tmp/_r2_fix.out", [a0, a1], calc=bc, paras=p)
        msgs = [str(x.message) for x in w]
    hit = [m for m in msgs if "frozen" in m.lower() or "fixatoms" in m.lower()]
    print(f"[FIX2] warnings captured={len(msgs)}  frozen-warning={bool(hit)}")
    assert hit, "FIX2 FAIL: no FixAtoms warning emitted"
    print("[FIX2] PASS:", hit[0][:90], "...")


if __name__ == "__main__":
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    if len(sys.argv) >= 2 and sys.argv[1] == "dump":
        dump_default(sys.argv[2])
    else:
        check_fix2_fixatoms()
        check_fix3_cadence()
        check_fix1_angular()
        print("\n[R2 P1 CHECKS] ALL PASS")
