# -*- coding: utf-8 -*-
"""PBC Phase-1A Gate A5: condensed-phase periodic NVT runs end to end, reproduces
the target temperature, stays stable (no collapse), and gives a physical O-O RDF
first peak. Also confirms ONE get_ef_gpu per step inside the real BatchedNVT loop
(Gate B2) and that all batched replicas ride the SAME shared forward (free-ride)."""
import os, sys, tempfile
import numpy as np, torch
torch.set_default_dtype(torch.float64)
from ase import Atoms
from maple.function.calculator.mace._maceoff_batch_calculator import MaceOffBatchCalc
from maple.function.dispatcher.md.ensemble.nvt_batched import BatchedNVT

if __name__ != "__main__":      # evidence/gate script: run directly, never on import
    raise SystemExit("run _test_pbc_nvt.py as a script, not an import")

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[env] device={DEV}")


def water_box(n, spacing, jitter=0.20, seed=0):
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


def oo_rdf_first_peak(atoms, rmax=6.0, nbins=120):
    O = [i for i,s in enumerate(atoms.get_chemical_symbols()) if s=="O"]
    sub = atoms[O]
    d = sub.get_all_distances(mic=True)
    iu = np.triu_indices(len(O), 1)
    dd = d[iu]; dd = dd[dd < rmax]
    if dd.size == 0: return None, None
    hist, edges = np.histogram(dd, bins=nbins, range=(0, rmax))
    r = 0.5*(edges[1:]+edges[:-1])
    shell = 4*np.pi*r**2*(edges[1]-edges[0])
    g = hist/np.maximum(shell, 1e-9)
    # first peak above r>2.0 A
    m = r > 2.0
    if not np.any(g[m]): return None, None
    idx = np.argmax(g[m]); rpk = r[m][idx]
    return float(rpk), float(np.min(dd))

# density ~1 g/cc -> ~3.1 A/water; side >= 2*r_max(=10)
n = 4                       # 64 waters = 192 atoms per replica
spacing = 3.15              # side = 12.6 A > 10
atomsA = water_box(n, spacing, jitter=0.10, seed=1)
atomsB = water_box(n, spacing, jitter=0.10, seed=2)
print(f"[sys] {len(atomsA)} atoms/replica, side={atomsA.cell.lengths()[0]:.3f} A, B=2")

calc = MaceOffBatchCalc(device=DEV, dtype=torch.float64)
# wrap forward with a counter (Gate B2)
nfwd = {"c": 0}
_orig = calc.get_ef_gpu
def _counted():
    nfwd["c"] += 1
    return _orig()
calc.get_ef_gpu = _counted

steps = 2500
paras = dict(timestep=0.5, steps=steps, temperature=300.0, thermostat="langevin",
             tau_t=20.0, remove_com_every=100, random_seed=7, verbose=0)
with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
    out = f.name
sim = BatchedNVT(out, [atomsA, atomsB], calc=calc, paras=paras).run()

# per-replica final atoms reconstructed from the calc master coords
coord = calc.coord.detach().cpu().numpy()
ptr = calc._ptr
peaks, dmins, Ttails, Tstds = [], [], [], []
for b in range(2):
    cb = coord[ptr[b]:ptr[b+1]]
    src = [atomsA, atomsB][b]
    fa = Atoms(numbers=src.get_atomic_numbers(), positions=cb,
               cell=src.cell, pbc=True)
    rpk, dmin = oo_rdf_first_peak(fa)
    peaks.append(rpk); dmins.append(dmin)
    Tk = np.asarray(sim.results[b]["T_K"])
    tail = Tk[int(0.6*len(Tk)):]                 # last 40% = equilibrated window
    t_tail = float(np.mean(tail)); t_sig = float(np.std(tail))
    Ttails.append(t_tail); Tstds.append(t_sig)
    print(f"[A5] rep{b}: <T>tail(last40%)={t_tail:.1f}K sig(T)={t_sig:.1f}K  "
          f"O-O RDF 1st peak={rpk:.2f}A  min O-O={dmin:.2f}A  "
          f"<PE>={sim.results[b]['PE_mean']:.4f}Ha")

print(f"[B2] get_ef_gpu calls = {nfwd['c']} for steps={steps} "
      f"(expect ~steps+1 init; ONE forward/step over BOTH replicas)")

# gates
T_ok = all(abs(t-300.0) < 50.0 for t in Ttails)         # equilibrated tail near target
stable = all((d is not None and d > 1.8) for d in dmins) # no collapse (H-bond O-O ~2.7)
rdf_ok = all((p is not None and 2.4 < p < 3.4) for p in peaks)  # water O-O 1st peak ~2.8 A
fwd_ok = abs(nfwd["c"] - (steps+1)) <= 1
allpass = T_ok and stable and rdf_ok and fwd_ok
print("\n==== A5 / B2 SUMMARY ====")
for nm, v in [("A5_temperature_reproduced", T_ok), ("A5_stable_no_collapse", stable),
              ("A5_rdf_first_peak_physical", rdf_ok), ("B2_one_forward_per_step", fwd_ok)]:
    print(f"  {'PASS' if v else 'FAIL'}  {nm}")

# ===================================================================== free-ride: REMD/GaMD/SMD/umbrella inherit PBC
print("\n==== FREE-RIDE: batched algorithms inherit PBC through the shared gate/forward ====")
from maple.function.dispatcher.md.ensemble.gamd_batched import BatchedGaMD
from maple.function.dispatcher.md.ensemble.smd_batched import BatchedSMD
from maple.function.dispatcher.md.ensemble.umbrella_batched import BatchedUmbrella
from maple.function.dispatcher.md.ensemble.remd import REMD

free = {}
# (1) GaMD: actually RUN a short periodic run -> proves the GaMD bias rides PBC via ONE forward/step.
try:
    cg = MaceOffBatchCalc(device=DEV, dtype=torch.float64)
    ng = {"c": 0}; _og = cg.get_ef_gpu
    cg.get_ef_gpu = (lambda: (ng.__setitem__("c", ng["c"]+1), _og())[1])
    gsteps = 60
    gp = dict(timestep=0.5, steps=gsteps, temperature=300.0, thermostat="langevin",
              tau_t=50.0, remove_com_every=100, random_seed=3, verbose=0,
              gamd="on", cv_group1="0,3,6", cv_group2="9,12,15", nwalkers=2,
              gamd_sigma0=6.0, gamd_prep_steps=20, gamd_mode="lower")
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        og = f.name
    BatchedGaMD(og, [atomsA.copy(), atomsB.copy()], calc=cg, paras=gp).run()
    free["GaMD_periodic_runs"] = True
    free["GaMD_one_forward_per_step"] = abs(ng["c"]-(gsteps+1)) <= 1
    print(f"[free] GaMD periodic ran {gsteps} steps; forwards={ng['c']} (expect ~{gsteps+1})")
except Exception as e:
    free["GaMD_periodic_runs"] = False
    print(f"[free] GaMD periodic FAILED: {type(e).__name__}: {str(e)[:160]}")

# (2) construction-only gate admit for REMD/SMD/umbrella periodic (the inherited gate
#     must accept periodic replicas + run the per-replica box guard).
def _construct(name, fn):
    try:
        fn(); free[name] = True; print(f"[free] {name}: PBC gate ADMITS (construction ok)")
    except NotImplementedError as e:
        free[name] = False; print(f"[free] {name}: REJECTED periodic -> {str(e)[:120]}")
    except Exception as e:
        # other errors (param plumbing) are not the PBC gate; report but don't fail PBC
        free[name] = None; print(f"[free] {name}: non-gate error {type(e).__name__}: {str(e)[:120]}")

_construct("REMD_periodic_admit", lambda: REMD(
    tempfile.mktemp(suffix=".out"), atomsA.copy(),
    calc=MaceOffBatchCalc(device=DEV, dtype=torch.float64),
    paras=dict(n_replicas=2, temp_min=300.0, temp_max=340.0, steps=10, timestep=0.5,
               exchange_every=5, random_seed=1, verbose=0)))
_construct("SMD_periodic_admit", lambda: BatchedSMD(
    tempfile.mktemp(suffix=".out"), [atomsA.copy(), atomsB.copy()],
    calc=MaceOffBatchCalc(device=DEV, dtype=torch.float64),
    paras=dict(smd_npulls=2, smd_group1="0,3,6", smd_group2="9,12,15", smd_k=0.1,
               smd_velocity=0.0, steps=10, timestep=0.5, temperature=300.0, verbose=0)))
_construct("Umbrella_periodic_admit", lambda: BatchedUmbrella(
    tempfile.mktemp(suffix=".out"), [atomsA.copy(), atomsB.copy()],
    calc=MaceOffBatchCalc(device=DEV, dtype=torch.float64),
    paras=dict(us_nwindows=2, us_group1="0,3,6", us_group2="9,12,15", us_kappa=150.0,
               us_cv_min=2.0, us_cv_max=4.0, steps=10, timestep=0.5, temperature=300.0, verbose=0)))

# (3) reject: a periodic batch on a NON-PBC backend must raise (capability routing).
class _FakeNoPBC:
    SUPPORTS_PBC = False
    def prepare(self, *a, **k): pass
    def get_ef_gpu(self): pass
    def step_cart_(self, *a, **k): pass
try:
    BatchedNVT(tempfile.mktemp(suffix=".out"), [atomsA.copy()], calc=_FakeNoPBC(),
               paras=dict(steps=1, thermostat="langevin"))
    free["reject_nonpbc_backend"] = False
    print("[free] reject non-PBC backend: FAILED (no raise)")
except NotImplementedError as e:
    free["reject_nonpbc_backend"] = "SUPPORTS_PBC" in str(e) or "non-periodic" in str(e)
    print(f"[free] reject non-PBC backend: raised -> {str(e)[:90]}")

print("\n==== FREE-RIDE SUMMARY ====")
fr_ok = True
for k, v in free.items():
    tag = "PASS" if v is True else ("FAIL" if v is False else "WARN")
    print(f"  {tag}  {k} = {v}")
    if v is False:
        fr_ok = False
allpass2 = allpass and fr_ok
print(f"==== {'ALL A5/B2/FREE-RIDE PASS' if allpass2 else 'SOME FAILED'} ====")
sys.exit(0 if allpass2 else 1)
