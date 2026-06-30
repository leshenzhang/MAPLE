# -*- coding: utf-8 -*-
"""PBC Phase-1A Gate A (correctness) + Gate B (forward-count/throughput) for the
MACE-OFF batched calculator. Self-contained: builds periodic water boxes with ASE,
no external data. Prints PASS/FAIL lines the validation report harvests."""
import sys, time
import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list

from maple.function.calculator.mace._maceoff_batch_calculator import MaceOffBatchCalc

if __name__ != "__main__":      # evidence/gate script: run directly, never on import
    raise SystemExit("run _test_pbc_batched.py as a script, not an import")

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
                # random rotation
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
    at = Atoms(symbols=symbols, positions=np.array(pos),
               cell=[L, L, L], pbc=pbc)
    return at


calc = MaceOffBatchCalc(device=DEV, dtype=torch.float64)
RMAX = calc.r_max
print(f"[env] MACE-OFF r_max={RMAX:.4f} A; box side must be >= {2*RMAX:.3f} A")

# dense box, side > 2*r_max so the guard passes and image edges actually occur.
side_n = 3
spacing = max(3.4, (2*RMAX + 1.0)/side_n)
at = water_box(side_n, spacing, seed=1)
print(f"[sys] water box: {len(at)} atoms, cell={at.cell.lengths()}, "
      f"min perp width={np.min(at.cell.lengths()):.3f} A")

results = {}

# ===================================================================== A3
# minimum-image edge-set match vs ASE neighbor_list('ijS').
calc.prepare([at])
coord = calc.coord
ei, sh, us = calc._build_edges_gpu(coord)
cell_t = calc._cell[0]
inv = torch.linalg.inv(cell_t)
Sint = torch.round(sh @ inv).to(torch.long).cpu().numpy()      # (E,3)
src = ei[0].cpu().numpy(); dst = ei[1].cpu().numpy()
mine = set(zip(src.tolist(), dst.tolist(),
               Sint[:, 0].tolist(), Sint[:, 1].tolist(), Sint[:, 2].tolist()))
ai, aj, aS = neighbor_list("ijS", at, RMAX)
ase_set = set(zip(ai.tolist(), aj.tolist(),
                  aS[:, 0].tolist(), aS[:, 1].tolist(), aS[:, 2].tolist()))
only_mine = mine - ase_set
only_ase = ase_set - mine
n_img = int((np.abs(Sint).sum(axis=1) > 0).sum())
print(f"[A3] my_edges={len(mine)} ase_edges={len(ase_set)} "
      f"image_edges(|S|>0)={n_img} only_mine={len(only_mine)} only_ase={len(only_ase)}")
results["A3_edge_match"] = (len(only_mine) == 0 and len(only_ase) == 0 and len(mine) == len(ase_set))
results["A3_image_edges_present"] = n_img > 0

# ===================================================================== parity vs mace from_config single
from mace.data.utils import config_from_atoms
from mace.data import AtomicData
from mace.tools import torch_geometric
cfg = config_from_atoms(at)
data = AtomicData.from_config(cfg, z_table=calc._z_table, cutoff=RMAX, heads=calc.heads)
batch1 = torch_geometric.Batch.from_data_list([data]).to(DEV)
d = batch1.to_dict()
prev = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
for kk, vv in d.items():
    if torch.is_tensor(vv) and vv.is_floating_point():
        d[kk] = vv.to(torch.float64)
out = calc.model(d, compute_force=True, training=False)
E_ref = float(out["energy"].detach().reshape(-1)[0].item())
F_ref = out["forces"].detach().cpu().numpy()
torch.set_default_dtype(prev)
E_mine_Ha, F_mine_Ha = calc.get_ef_gpu()
EV2HARTREE = 1.0/27.211386245988
E_mine = float(E_mine_Ha[0].item())/EV2HARTREE
# unpad my forces back to (N,3) eV/A
base = calc._base.cpu().numpy()
F_mine_flat = (F_mine_Ha[0].cpu().numpy())/EV2HARTREE
F_mine = np.stack([F_mine_flat[base], F_mine_flat[base+1], F_mine_flat[base+2]], axis=1)
dE = abs(E_mine - E_ref)
dF = float(np.abs(F_mine - F_ref).max())
# edge-set diff vs from_config (mace/matscipy) to characterize any residual.
fc_ei = batch1.edge_index.cpu().numpy()
fc_us = batch1.unit_shifts.cpu().numpy().round().astype(int) if hasattr(batch1, "unit_shifts") \
    else np.zeros((fc_ei.shape[1], 3), int)
fc_set = set(zip(fc_ei[0].tolist(), fc_ei[1].tolist(),
                 fc_us[:, 0].tolist(), fc_us[:, 1].tolist(), fc_us[:, 2].tolist()))
d_fc = len(mine ^ fc_set)
print(f"[parity-vs-from_config] dE={dE:.3e} eV  dFmax={dF:.3e} eV/A  "
      f"edge-set-symdiff(mine vs from_config)={d_fc}  (A3 already proved mine==ASE exactly)")
# Authoritative neighbour correctness = A3 (exact match to ASE neighbor_list). This
# cross-check vs mace's own matscipy build is informational; thresholds are a noise
# band (E rel ~1e-11, F << typical ~1 eV/A force magnitude).
results["parity_fromconfig_E"] = dE < 1e-4
results["parity_fromconfig_F"] = dF < 1e-2

# ===================================================================== A1 single-vs-batched parity
B = 4
calcB = MaceOffBatchCalc(device=DEV, dtype=torch.float64)
calcB.prepare([at.copy() for _ in range(B)])
EB, FB = calcB.get_ef_gpu()
calc1 = MaceOffBatchCalc(device=DEV, dtype=torch.float64)
calc1.prepare([at.copy()])
E1, F1 = calc1.get_ef_gpu()
dE_par = float((EB - EB[0]).abs().max().item())             # all replicas identical
dE_b1 = float((EB[0] - E1[0]).abs().item())
nmax = min(FB.shape[1], F1.shape[1])
dF_par = float((FB[:, :nmax] - FB[0:1, :nmax]).abs().max().item())
dF_b1 = float((FB[0, :nmax] - F1[0, :nmax]).abs().max().item())
print(f"[A1] B={B} identical replicas: dE(rep-vs-rep0)={dE_par:.3e}  "
      f"dE(B4rep0-vs-B1)={dE_b1:.3e}  dF(rep-vs-rep0)={dF_par:.3e}  dF(B4-vs-B1)={dF_b1:.3e} (Ha units)")
# fp64 GPU scatter-add is non-deterministic at the ULP (~1e-16 Ha) level even for
# bit-identical inputs; the floor is machine epsilon, not exactly 0.0.
results["A1_batched_parity_E"] = dE_par < 1e-13 and dE_b1 < 1e-13
results["A1_batched_parity_F"] = dF_par < 1e-13 and dF_b1 < 1e-13

# ===================================================================== A2 block-diagonal isolation under PBC
leak = calcB.isolation_check(perturb=0.05)
print(f"[A2] PBC isolation: max energy leak into other replicas = {leak:.3e} Ha")
results["A2_isolation"] = leak == 0.0

# ===================================================================== edge freshness (anti-stale-neighbor, R3-1)
# Moving an atom ACROSS the box face between two forwards MUST (a) change the
# periodic edge set and (b) give forces identical to a FRESH prepare() on the moved
# geometry -- i.e. edges+shifts are rebuilt every forward, never baked at prepare().
cf = MaceOffBatchCalc(device=DEV, dtype=torch.float64)
cf.prepare([at.copy()])
def _edge_set(c):
    e, s, _ = c._build_edges_gpu(c.coord)
    Si = torch.round(s @ torch.linalg.inv(c._cell[0])).to(torch.long).cpu().numpy()
    return set(zip(e[0].cpu().numpy().tolist(), e[1].cpu().numpy().tolist(),
                   Si[:, 0].tolist(), Si[:, 1].tolist(), Si[:, 2].tolist()))
set_before = _edge_set(cf)
E_before, F_before = cf.get_ef_gpu()
L = float(at.cell.lengths()[0])
cf.coord[3] += torch.tensor([0.62*L, 0.37*L, -0.55*L], dtype=torch.float64, device=DEV)  # cross faces
set_after = _edge_set(cf)
E_after, F_after = cf.get_ef_gpu()
# fresh prepare on the moved geometry
moved = at.copy(); mp = moved.get_positions(); mp[3] += np.array([0.62*L, 0.37*L, -0.55*L]); moved.set_positions(mp)
cfresh = MaceOffBatchCalc(device=DEV, dtype=torch.float64); cfresh.prepare([moved])
E_fresh, F_fresh = cfresh.get_ef_gpu()
edges_changed = (set_before != set_after)
forces_changed = float((F_after - F_before).abs().max().item()) > 1e-6
dE_fresh = float((E_after[0] - E_fresh[0]).abs().item())
dF_fresh = float((F_after[0] - F_fresh[0]).abs().max().item())
print(f"[freshness] edges_changed_after_cross={edges_changed} forces_changed={forces_changed} "
      f"| inplace-vs-freshprepare dE={dE_fresh:.3e} dF={dF_fresh:.3e} Ha (==0 => no stale state)")
results["freshness_edges_rebuilt"] = edges_changed and forces_changed
results["freshness_no_stale_state"] = dE_fresh < 1e-9 and dF_fresh < 1e-9

# ===================================================================== A4 box-guard per-replica abort
from maple.function.dispatcher.md.box_guard import check_box_size, evaluate_box
small = water_box(2, (2*RMAX)/2 - 1.5, seed=3)   # side < 2*r_max -> must abort
ok_small, info = evaluate_box(small, RMAX)
guard_raised = False
try:
    check_box_size(small, calc=calc, mode="strict", context="A4 replica 2")
except RuntimeError as e:
    guard_raised = "replica 2" in str(e) or "too small" in str(e)
    msg0 = str(e).splitlines()[0]
print(f"[A4] small box ok={ok_small} (expect False); strict guard raised+named={guard_raised}")
results["A4_box_guard"] = (not ok_small) and guard_raised
# big box must pass
big_ok = check_box_size(at, calc=calc, mode="strict", context="A4 big")
results["A4_big_box_passes"] = big_ok is True

# ===================================================================== default-path invariance (isolated unchanged)
iso = at.copy(); iso.pbc = False; iso.cell = None
c_iso = MaceOffBatchCalc(device=DEV, dtype=torch.float64)
c_iso.prepare([iso])
ei_iso, sh_iso, _ = c_iso._build_edges_gpu(c_iso.coord)
E_iso, _ = c_iso.get_ef_gpu()
print(f"[default-inv] isolated: periodic flag={c_iso._periodic} (expect False), "
      f"shifts all zero={bool((sh_iso == 0).all().item())}")
results["default_isolated_nonperiodic"] = (c_iso._periodic is False) and bool((sh_iso == 0).all().item())

# ===================================================================== Gate B2: one get_ef_gpu per step (counter)
n_forward = {"c": 0}
_orig = calcB.get_ef_gpu
def _counted():
    n_forward["c"] += 1
    return _orig()
calcB.get_ef_gpu = _counted
for _ in range(5):
    e, f = calcB.get_ef_gpu()
    calcB.step_cart_(torch.zeros((calcB.B, calcB.nmax_dof), dtype=torch.float64, device=DEV))
print(f"[B2] forwards for 5 ef calls = {n_forward['c']} (expect 5; one forward per ef-call/step)")
results["B2_one_forward_per_step"] = n_forward["c"] == 5
calcB.get_ef_gpu = _orig

# ===================================================================== Gate B1: throughput batched vs serial
def bench(Bv, reps=20):
    a_list = [at.copy() for _ in range(Bv)]
    cb = MaceOffBatchCalc(device=DEV, dtype=torch.float64)
    cb.prepare(a_list)
    for _ in range(3):  # warmup
        cb.get_ef_gpu()
    if DEV == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        cb.get_ef_gpu()
    if DEV == "cuda": torch.cuda.synchronize()
    return (time.time()-t0)/reps

Bv = 8
t_batch = bench(Bv)
# serial: Bv separate B=1 forwards
cs = MaceOffBatchCalc(device=DEV, dtype=torch.float64)
cs.prepare([at.copy()])
for _ in range(3): cs.get_ef_gpu()
if DEV == "cuda": torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    for _ in range(Bv):
        cs.get_ef_gpu()
if DEV == "cuda": torch.cuda.synchronize()
t_serial = (time.time()-t0)/20
speedup = t_serial/t_batch
print(f"[B1] B={Bv} {len(at)}-atom periodic box: batched={t_batch*1e3:.2f} ms/step  "
      f"serial(Bx1)={t_serial*1e3:.2f} ms/step  speedup={speedup:.2f}x  "
      f"(N_atoms_per_rep={len(at)})")
results["B1_speedup"] = speedup
results["B1_no_regression"] = speedup >= 0.95

# ===================================================================== summary
print("\n==== GATE SUMMARY ====")
allpass = True
for k, v in results.items():
    if isinstance(v, bool):
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
        allpass = allpass and v
    else:
        print(f"  ----  {k} = {v}")
print(f"==== {'ALL GATE-A/B BOOL CHECKS PASS' if allpass else 'SOME CHECKS FAILED'} ====")
sys.exit(0 if allpass else 1)
