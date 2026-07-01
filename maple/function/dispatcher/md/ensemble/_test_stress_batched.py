# -*- coding: utf-8 -*-
"""Phase-2C Gate G1 (batched-stress parity) + G2 (analytic-virial vs finite-diff)
for the MACE-OFF batched calculator's configurational stress path.
Self-contained: builds periodic water boxes with ASE, no external data. Prints
PASS/FAIL lines the validation report harvests.

G1  batched stress[b] == single-system (B=1) stress per replica, fp64 (<1e-12);
    plus an independent cross-check vs ase MACECalculator.get_stress(voigt=True).
G2  analytic stress == central-difference of the SAME potential's energy under a
    symmetric strain (ase calculate_numerical_stress algorithm), per Voigt (<1e-5).
    ASE/MACE convention: stress = +(1/V) dE/deps (eV/Ang^3, Voigt [xx,yy,zz,yz,xz,xy]).
"""
import sys
import numpy as np
import torch
from ase import Atoms

from maple.function.calculator.mace._maceoff_batch_calculator import MaceOffBatchCalc

if __name__ != "__main__":
    raise SystemExit("run _test_stress_batched.py as a script, not an import")

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[env] device={DEV} torch={torch.__version__}")


def water_box(n, spacing, jitter=0.15, seed=0, pbc=True):
    rng = np.random.default_rng(seed)
    dOH, ang = 0.9572, np.deg2rad(104.52)
    base = np.array([[0, 0, 0], [dOH, 0, 0],
                     [dOH*np.cos(ang), dOH*np.sin(ang), 0.0]])
    sym = ["O", "H", "H"]
    pos, symbols = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                q = rng.standard_normal(4); q /= np.linalg.norm(q)
                w, x, y, z = q
                R = np.array([
                    [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
                c = (np.array([i, j, k]) + 0.5)*spacing + jitter*rng.standard_normal(3)
                mol = base @ R.T + c
                pos.extend(mol); symbols.extend(sym)
    L = n*spacing
    return Atoms(symbols=symbols, positions=np.array(pos), cell=[L, L, L], pbc=pbc)


calc = MaceOffBatchCalc(device=DEV, dtype=torch.float64)
RMAX = calc.r_max
print(f"[env] MACE-OFF r_max={RMAX:.4f} A; box side must be >= {2*RMAX:.3f} A")

side_n = 3
spacing = max(3.4, (2*RMAX + 1.0)/side_n)
# 3 DISTINCT replicas (different seeds) so parity is a real per-replica test, not
# 3 copies of one geometry.
ats = [water_box(side_n, spacing, seed=s) for s in (1, 7, 23)]
print(f"[sys] {len(ats)} periodic water boxes: {len(ats[0])} atoms each, "
      f"L={ats[0].cell.lengths()[0]:.3f} A (>= 2*r_max={2*RMAX:.3f})")

results = {}

# ===================================================================== G1
# batched B=3 (distinct replicas) stress vs per-replica single-system (B=1) stress.
calcB = MaceOffBatchCalc(device=DEV, dtype=torch.float64)
calcB.prepare([a.copy() for a in ats])
SB = calcB.get_stress_gpu().cpu().numpy()                 # (3,6) eV/Ang^3
S1 = np.zeros_like(SB)
for b, a in enumerate(ats):
    c1 = MaceOffBatchCalc(device=DEV, dtype=torch.float64)
    c1.prepare([a.copy()])
    S1[b] = c1.get_stress_gpu().cpu().numpy()[0]
d_batch_single = float(np.abs(SB - S1).max())
print(f"[G1] batched(B=3) vs single(B=1) per-replica stress: max|dstress|={d_batch_single:.3e} eV/Ang^3")
for b in range(len(ats)):
    print(f"      rep{b} stress_voigt(batched)={np.array2string(SB[b], precision=6)}")
# fp64 GPU scatter-add is non-deterministic at ULP; floor is ~machine-eps not 0.
results["G1_batched_single_parity"] = d_batch_single < 1e-12

# independent cross-check vs ase MACECalculator (the reference the single-system NPT
# actually reads through get_stress). Same checkpoint, matscipy neighbour list.
try:
    from mace.calculators import MACECalculator
    import os
    ref = MACECalculator(model_paths=os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model"),
                         device=DEV, default_dtype="float64")
    d_ase = 0.0
    for b, a in enumerate(ats):
        aa = a.copy(); aa.calc = ref
        s_ase = aa.get_stress(voigt=True)                 # eV/Ang^3 ASE Voigt
        d_ase = max(d_ase, float(np.abs(SB[b] - s_ase).max()))
    print(f"[G1b] batched vs ase MACECalculator.get_stress: max|dstress|={d_ase:.3e} eV/Ang^3")
    results["G1b_vs_ase_maceoff"] = d_ase < 1e-9
except Exception as e:
    print(f"[G1b] ase MACECalculator cross-check unavailable: {type(e).__name__}: {e}")
    results["G1b_vs_ase_maceoff"] = None

# ===================================================================== G2
# analytic stress == central-difference of the batched calc's OWN energy under a
# symmetric strain (ase.calculate_numerical_stress algorithm, energy_fn = this calc).
_fd = MaceOffBatchCalc(device=DEV, dtype=torch.float64)

def energy_ev(atoms):
    _fd.prepare([atoms])
    return float(_fd._batched_ef_eV()[0][0].item())

def numerical_stress_voigt(atoms, d=1e-4):
    stress = np.zeros((3, 3))
    cell0 = atoms.cell.copy()
    V = atoms.get_volume()
    a = atoms.copy()
    for i in range(3):
        x = np.eye(3); x[i, i] += d
        a.set_cell(np.dot(cell0, x), scale_atoms=True); ep = energy_ev(a)
        x[i, i] -= 2*d
        a.set_cell(np.dot(cell0, x), scale_atoms=True); em = energy_ev(a)
        stress[i, i] = (ep - em) / (2*d*V)
        x[i, i] = 1.0
        j = i - 2
        x[i, j] = x[j, i] = d
        a.set_cell(np.dot(cell0, x), scale_atoms=True); ep = energy_ev(a)
        x[i, j] = x[j, i] = -d
        a.set_cell(np.dot(cell0, x), scale_atoms=True); em = energy_ev(a)
        stress[i, j] = stress[j, i] = (ep - em) / (4*d*V)
        x[i, j] = x[j, i] = 0.0
    return stress.flat[[0, 4, 8, 5, 2, 1]].copy()          # [xx,yy,zz,yz,xz,xy]

g2_max = 0.0
for b, a in enumerate(ats):
    ca = MaceOffBatchCalc(device=DEV, dtype=torch.float64)
    ca.prepare([a.copy()])
    s_ana = ca.get_stress_gpu().cpu().numpy()[0]           # (6,) eV/Ang^3
    s_num = numerical_stress_voigt(a.copy(), d=1e-4)
    dmax = float(np.abs(s_ana - s_num).max())
    g2_max = max(g2_max, dmax)
    print(f"[G2] rep{b} analytic vs FD stress: max|d|={dmax:.3e} eV/Ang^3")
    print(f"      analytic={np.array2string(s_ana, precision=6)}")
    print(f"      numeric ={np.array2string(s_num, precision=6)}")
print(f"[G2] worst analytic-vs-FD per-Voigt discrepancy = {g2_max:.3e} eV/Ang^3 (tol 1e-5)")
results["G2_virial_vs_fd"] = g2_max < 1e-5

# ===================================================================== summary
print("\n==== STRESS GATE SUMMARY (G1,G2) ====")
allpass = True
for k, v in results.items():
    if isinstance(v, bool):
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
        allpass = allpass and v
    else:
        print(f"  ----  {k} = {v}")
print(f"==== {'STRESS GATES G1/G2 PASS' if allpass else 'SOME STRESS CHECKS FAILED'} ====")
sys.exit(0 if allpass else 1)
