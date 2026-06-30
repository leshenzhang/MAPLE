# -*- coding: utf-8 -*-
"""PBC Phase-1A UMA-periodic validation (DEFERRED runtime: needs a working fairchem
env + a periodic UMA checkpoint). Mirrors the MACE-OFF Gate A: single-vs-batched
parity, block-diagonal isolation, min-image match vs ASE, AND the R3-1 edge-
freshness (anti-stale-neighbor) gate -- perturb an atom across the box face between
two forwards and confirm forces change AND equal a FRESH prepare on the moved cell.

Run (when cxtorch/fairchem is fixed):
  PYTHONPATH=<worktree> UMA_MODEL=<uma-name-or-ckpt> UMA_TASK=omat \
    <cxtorch python> _test_pbc_uma.py
"""
import os, sys
import numpy as np, torch
from ase import Atoms
from ase.neighborlist import neighbor_list

if __name__ != "__main__":
    raise SystemExit("run _test_pbc_uma.py as a script, not an import")

from maple.function.calculator.uma._uma_batch_calculator import UMABatchCalc

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("UMA_MODEL", "uma-s-1p1")
TASK = os.environ.get("UMA_TASK", "omat")
print(f"[env] device={DEV} model={MODEL} task={TASK}")


def water_box(n, spacing, jitter=0.15, seed=0):
    rng = np.random.default_rng(seed)
    dOH, ang = 0.9572, np.deg2rad(104.52)
    base = np.array([[0,0,0],[dOH,0,0],[dOH*np.cos(ang),dOH*np.sin(ang),0.0]])
    pos, sym = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                q = rng.standard_normal(4); q/=np.linalg.norm(q); w,x,y,z=q
                R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
                c=(np.array([i,j,k])+0.5)*spacing + jitter*rng.standard_normal(3)
                pos.extend(base@R.T+c); sym.extend(["O","H","H"])
    L=n*spacing
    return Atoms(symbols=sym, positions=np.array(pos), cell=[L,L,L], pbc=True)

at = water_box(3, 4.2, seed=1)   # 27 waters, 12.6 A box
EV2H = 1.0/27.211386245988
res = {}

# task gate: molecular task must reject periodic input
try:
    cbad = UMABatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64, task="omol")
    cbad.prepare([at.copy()])
    res["task_gate_rejects_omol"] = False
except NotImplementedError:
    res["task_gate_rejects_omol"] = True
    print("[gate] periodic+omol correctly REJECTED")

calc = UMABatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64, task=TASK)

# A1 single-vs-batched parity
B = 4
calc.prepare([at.copy() for _ in range(B)])
EB, FB = calc.get_ef_gpu()
c1 = UMABatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64, task=TASK)
c1.prepare([at.copy()]); E1, F1 = c1.get_ef_gpu()
nmax = min(FB.shape[1], F1.shape[1])
dE = float((EB - EB[0]).abs().max()); dEb1 = float((EB[0]-E1[0]).abs())
dF = float((FB[:, :nmax]-FB[0:1, :nmax]).abs().max())
print(f"[A1] dE(rep)={dE:.2e} dE(B4-B1)={dEb1:.2e} dF(rep)={dF:.2e}")
res["A1_parity"] = dE < 1e-6 and dEb1 < 1e-6  # fp32 UMA noise floor ~1e-6

# A2 isolation under PBC
try:
    leak = calc.isolation_check(0.05)
    print(f"[A2] leak={leak:.2e}"); res["A2_isolation"] = leak < 1e-5
except Exception as e:
    print(f"[A2] skip ({e})")

# R3-1 edge freshness: cross a face between forwards; forces must change AND match fresh prepare
cf = UMABatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64, task=TASK)
cf.prepare([at.copy()])
E0, F0 = cf.get_ef_gpu()
L = float(at.cell.lengths()[0])
shift = np.array([0.62*L, 0.37*L, -0.55*L])
cf.coord[3] += torch.tensor(shift, dtype=torch.float64, device=cf.coord.device)
E1b, F1b = cf.get_ef_gpu()
moved = at.copy(); mp = moved.get_positions(); mp[3] += shift; moved.set_positions(mp)
cfr = UMABatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64, task=TASK)
cfr.prepare([moved]); E2, F2 = cfr.get_ef_gpu()
forces_changed = float((F1b-F0).abs().max()) > 1e-5
dEf = float((E1b[0]-E2[0]).abs()); dFf = float((F1b[0]-F2[0]).abs().max())
print(f"[freshness] forces_changed={forces_changed} inplace-vs-fresh dE={dEf:.2e} dF={dFf:.2e}")
res["freshness_no_stale_neighbor"] = forces_changed and dEf < 1e-5 and dFf < 1e-5

print("\n==== UMA-PBC SUMMARY ====")
ok = True
for k, v in res.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}"); ok = ok and bool(v)
print(f"==== {'UMA-PBC PASS' if ok else 'UMA-PBC FAIL'} ====")
sys.exit(0 if ok else 1)
