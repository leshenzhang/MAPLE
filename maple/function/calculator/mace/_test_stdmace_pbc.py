# -*- coding: utf-8 -*-
"""std-MACE / mace-mp batched PERIODIC edge-builder gate (project ai-maple-md, Phase B).

Validates the ``radius_graph_pbc`` just wired into ``MACEBatchCalc`` (the std-MACE / mace-mp
BATCHED calc, ``_mace_batch_calculator.py``), which B-88 explicitly DEFERRED ("std-MACE
periodic edge builder — DEFERRED ... needs a block-diagonal radius_graph_pbc + a traced
model that accepts nonzero shifts at B>1"). The block-diagonal edge builder is now
implemented; NO traced 6-arg std-MACE `.pt` asset exists in the tree (the exact blocker
B-88 named), so the gates split into two tiers, BOTH rigorous:

  Tier-1 (MACEBatchCalc's OWN new code, model-free, direct):
    G1  pbc=False bit-identical      -- non-periodic branch unchanged (zero shifts)
    G2  edge-set vs ASE              -- (i,j,S) symmetric-difference vs neighbor_list('ijS')=0
    G4  min-image / short-contact    -- edge distances == ASE, no spurious <1.5A, box guard
    XID cross-calc identity          -- MACEBatchCalc edges == validated MaceOffBatchCalc edges

  Tier-2 (real mace-mp-0 forward via the runnable mace-package batched backend
          MaceOffBatchCalc(model_path=mace-mp-0), which runs the byte-identical
          _build_edges_pbc algorithm certified equal by XID):
    G3  single-vs-batched dF         -- B periodic replicas -> per-replica F == single (fp64)
    G5  mace-mp NVT-PBC bulk water   -- stable <T>, physical O-O RDF first peak

  Bonus: G3 is ALSO attempted directly on MACEBatchCalc's real 6-arg forward via a
  hand-wired EAGER mace-mp-0 wrapper (a stand-in for the not-yet-existent traced model),
  giving direct evidence of the wired calc's forward + PBC edges when the wrapper loads.

Self-contained (ASE water boxes, no external data). Prints PASS/FAIL + numbers.
"""
import os, sys, tempfile
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from ase import Atoms
from ase.neighborlist import neighbor_list

if __name__ != "__main__":
    raise SystemExit("run _test_stdmace_pbc.py as a script, not an import")

from maple.function.calculator.mace._mace_batch_calculator import MACEBatchCalc, _one_hot_node_attrs
from maple.function.calculator.mace._maceoff_batch_calculator import MaceOffBatchCalc

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
EV2H = 1.0 / 27.211386245988
MODEL_MP = os.path.expanduser("~/.cache/mace/20231203mace128L1_epoch199model")
print(f"[env] device={DEV} torch={torch.__version__} model=mace-mp-0({os.path.basename(MODEL_MP)})")

# ---- load mace-mp-0 (mace-package checkpoint) once: r_max / atomic table / real forward ----
_mace = torch.load(MODEL_MP, map_location=DEV, weights_only=False).to(DEV).to(DT).eval()
for p in _mace.parameters():
    p.requires_grad_(False)
R_MAX = float(_mace.r_max)
ATN = [int(z) for z in _mace.atomic_numbers]
HEADS = list(getattr(_mace, "heads", ["Default"]))
print(f"[env] mace-mp-0 r_max={R_MAX:.4f} A  n_elements={len(ATN)}  heads={HEADS}  "
      f"box side must be >= {2*R_MAX:.2f} A")

results = {}


def water_box(n, spacing, jitter=0.15, seed=0, pbc=True):
    rng = np.random.default_rng(seed)
    dOH, ang = 0.9572, np.deg2rad(104.52)
    base = np.array([[0, 0, 0], [dOH, 0, 0],
                     [dOH*np.cos(ang), dOH*np.sin(ang), 0.0]])
    pos, sym = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                q = rng.standard_normal(4); q /= np.linalg.norm(q); w, x, y, z = q
                R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                              [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                              [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
                c = (np.array([i, j, k]) + 0.5)*spacing + jitter*rng.standard_normal(3)
                pos.extend(base @ R.T + c); sym.extend(["O", "H", "H"])
    L = n*spacing
    return Atoms(symbols=sym, positions=np.array(pos), cell=[L, L, L], pbc=pbc)


def make_edgeonly_calc():
    """MACEBatchCalc built WITHOUT a jit model (edge builder is model-free)."""
    c = MACEBatchCalc.__new__(MACEBatchCalc)
    c.device = torch.device(DEV); c.dtype = DT; c.mdtype = DT
    c._model_name = "mace-mp-edgeonly"; c._model_path = MODEL_MP; c.model = None
    c.r_max = R_MAX; c.atomic_numbers = ATN
    c._batch_native = True; c._hess_mode = None
    c._prepared = False; c._atoms_B = 0
    c._ptr = None; c.numbers = None; c.node_attrs = None; c.mol_idx = None
    c.coord = None; c.N_atoms = 0; c.Nmax_atoms = 0; c.nmax_dof = 0
    c.cand_i = c.cand_j = c.cand_rep = c._base = c._n_b = c._coord_backup = None
    c._periodic = False; c._cell = c._cell_inv = c._shift_combos = None
    return c


# ===================================================================== box for edge gates
side_n = 3
spacing = max(3.2, (2*R_MAX + 1.5)/side_n)
at = water_box(side_n, spacing, seed=1)
print(f"[sys] edge-gate water box: {len(at)} atoms, side={at.cell.lengths()[0]:.3f} A, "
      f"min perp width={np.min(at.cell.lengths()):.3f} A")


# ===================================================================== G1 pbc=False bit-identical
# Isolated batch -> _periodic must be False and _build_edges must equal the pre-PBC formula
# (rij; d2<= (r_max+1e-12)^2; symmetric i<->j; ZERO shifts) bit-for-bit.
iso = at.copy(); iso.pbc = False; iso.cell = None
c_iso = make_edgeonly_calc(); c_iso.prepare([iso])
ei_iso, sh_iso = c_iso._build_edges(c_iso.coord)
# reference: the exact non-periodic branch, recomputed independently.
coord = c_iso.coord
rij = coord[c_iso.cand_i] - coord[c_iso.cand_j]
d2 = (rij*rij).sum(dim=-1)
keep = d2 <= (R_MAX + 1e-12)**2
bci = c_iso.cand_i[keep]; bcj = c_iso.cand_j[keep]
base_ei = torch.stack([torch.cat([bci, bcj]), torch.cat([bcj, bci])], dim=0)
base_sh = torch.zeros((base_ei.size(1), 3), dtype=DT, device=c_iso.device)
g1_edges = bool(torch.equal(ei_iso, base_ei))
g1_shifts_zero = bool((sh_iso == 0).all().item()) and bool(torch.equal(sh_iso, base_sh))
g1_flag = (c_iso._periodic is False)
print(f"[G1] _periodic={c_iso._periodic} (expect False)  edge_index bit-identical={g1_edges}  "
      f"shifts all-zero & identical={g1_shifts_zero}  n_edges={ei_iso.size(1)}")
results["G1_pbc_false_bit_identical"] = g1_edges and g1_shifts_zero and g1_flag


# ===================================================================== G2 edge-set vs ASE
c2 = make_edgeonly_calc(); c2.prepare([at])
ei, sh = c2._build_edges(c2.coord)
cell_t = c2._cell[0]; inv = torch.linalg.inv(cell_t)
Sint = torch.round(sh @ inv).to(torch.long).cpu().numpy()
src = ei[0].cpu().numpy(); dst = ei[1].cpu().numpy()
mine = set(zip(src.tolist(), dst.tolist(),
               Sint[:, 0].tolist(), Sint[:, 1].tolist(), Sint[:, 2].tolist()))
ai, aj, aS = neighbor_list("ijS", at, R_MAX)
ase_set = set(zip(ai.tolist(), aj.tolist(),
                  aS[:, 0].tolist(), aS[:, 1].tolist(), aS[:, 2].tolist()))
only_mine = mine - ase_set; only_ase = ase_set - mine
n_img = int((np.abs(Sint).sum(axis=1) > 0).sum())
print(f"[G2] my_edges={len(mine)} ase_edges={len(ase_set)} image_edges(|S|>0)={n_img} "
      f"only_mine={len(only_mine)} only_ase={len(only_ase)} symdiff={len(mine ^ ase_set)}")
results["G2_edge_set_vs_ASE"] = (len(only_mine) == 0 and len(only_ase) == 0
                                 and len(mine) == len(ase_set) and n_img > 0)


# ===================================================================== G4 min-image / short-contact
# (a) every edge's minimum-image distance == ASE's distance for the same (i,j,S)
ai2, aj2, aS2, aD2 = neighbor_list("ijSd", at, R_MAX)
ase_d = {}
for a, b, s0, s1, s2, dd in zip(ai2, aj2, aS2[:, 0], aS2[:, 1], aS2[:, 2], aD2):
    ase_d[(int(a), int(b), int(s0), int(s1), int(s2))] = float(dd)
pos = c2.coord
Scart = sh
vec = pos[ei[1]] - pos[ei[0]] + Scart          # mace convention pos[recv]-pos[send]+shift
dist = vec.norm(dim=1).cpu().numpy()
max_dist_err = 0.0
for k, (a, b, s0, s1, s2) in enumerate(zip(src, dst, Sint[:, 0], Sint[:, 1], Sint[:, 2])):
    key = (int(a), int(b), int(s0), int(s1), int(s2))
    if key in ase_d:
        max_dist_err = max(max_dist_err, abs(dist[k] - ase_d[key]))
all_within = bool((dist < R_MAX + 1e-9).all())
min_edge = float(dist.min()) if dist.size else 999.0
# spurious short contact = any INTER-molecular (different water) edge below 1.5 A. Intra-water
# O-H (~0.96) and H-H (~1.5) are legitimate; a wrong-image collapse shows as a cross-mol
# pair << 1.5 A. Flag cross-molecule edges only.
mol_of = np.repeat(np.arange(len(at)//3), 3)    # water i -> atoms [3i,3i+1,3i+2]
cross = mol_of[src] != mol_of[dst]
cross_min = float(dist[cross].min()) if cross.any() else 999.0
no_spurious = cross_min > 1.5
# (b) box guard aborts a too-small box (side < 2*r_max)
guard_raised = False
try:
    from maple.function.dispatcher.md.box_guard import check_box_size, evaluate_box
    small = water_box(2, R_MAX - 0.5, seed=3)   # side ~ 2*r_max/2 < 2*r_max
    ok_small, _info = evaluate_box(small, R_MAX)
    try:
        check_box_size(small, calc=c2, mode="strict", context="G4 small box")
    except RuntimeError as e:
        guard_raised = ("too small" in str(e).lower()) or ("replica" in str(e).lower())
    big_ok = check_box_size(at, calc=c2, mode="strict", context="G4 big") is True
except Exception as e:
    ok_small, big_ok = None, None
    print(f"[G4] box_guard import/exec issue: {type(e).__name__}: {str(e)[:120]}")
print(f"[G4] max|edge_dist-ASE_dist|={max_dist_err:.2e} A  all_within_rmax={all_within}  "
      f"min_edge={min_edge:.3f}A  cross_mol_min={cross_min:.3f}A no_spurious(<1.5)={no_spurious}  "
      f"small_box_ok={ok_small}(exp False) guard_raised={guard_raised} big_box_passes={big_ok}")
results["G4_min_image_short_contact"] = (max_dist_err < 1e-9 and all_within
                                         and no_spurious and (ok_small is False)
                                         and guard_raised and (big_ok is True))


# ===================================================================== XID cross-calc identity
# MACEBatchCalc._build_edges_pbc must produce the SAME edge set (i,j,S) AND the same cartesian
# shifts as the B-88-validated MaceOffBatchCalc._build_edges_pbc on the identical geometry.
cm = MaceOffBatchCalc(device=DEV, dtype=DT, model_path=MODEL_MP)
cm.prepare([at.copy()])
ei_m, sh_m, _us = cm._build_edges_gpu(cm.coord)
Sint_m = torch.round(sh_m @ torch.linalg.inv(cm._cell[0])).to(torch.long).cpu().numpy()
src_m = ei_m[0].cpu().numpy(); dst_m = ei_m[1].cpu().numpy()
maceoff_set = set(zip(src_m.tolist(), dst_m.tolist(),
                      Sint_m[:, 0].tolist(), Sint_m[:, 1].tolist(), Sint_m[:, 2].tolist()))
xid_symdiff = len(mine ^ maceoff_set)
# cartesian-shift agreement per shared directed edge (dict keyed by (i,j,S))
def _shiftmap(srcv, dstv, Sintv, shv):
    m = {}
    shn = shv.detach().cpu().numpy()
    for k in range(len(srcv)):
        m[(int(srcv[k]), int(dstv[k]), int(Sintv[k, 0]), int(Sintv[k, 1]), int(Sintv[k, 2]))] = shn[k]
    return m
sm_a = _shiftmap(src, dst, Sint, sh)
sm_b = _shiftmap(src_m, dst_m, Sint_m, sh_m)
shift_max_diff = 0.0
for key in (set(sm_a) & set(sm_b)):
    shift_max_diff = max(shift_max_diff, float(np.abs(sm_a[key] - sm_b[key]).max()))
print(f"[XID] MACEBatchCalc edges={len(mine)} MaceOffBatchCalc edges={len(maceoff_set)} "
      f"symdiff={xid_symdiff}  max|cartesian shift diff|={shift_max_diff:.2e} A")
results["XID_cross_calc_identity"] = (xid_symdiff == 0 and shift_max_diff < 1e-12
                                      and len(mine) == len(maceoff_set))


# ===================================================================== G3 single-vs-batched dF
# Real mace-mp-0 forward via MaceOffBatchCalc (runs the XID-certified identical edge algorithm).
B = 4
cB = MaceOffBatchCalc(device=DEV, dtype=DT, model_path=MODEL_MP)
cB.prepare([at.copy() for _ in range(B)])
EB, FB = cB.get_ef_gpu()
c1 = MaceOffBatchCalc(device=DEV, dtype=DT, model_path=MODEL_MP)
c1.prepare([at.copy()])
E1, F1 = c1.get_ef_gpu()
dE_par = float((EB - EB[0]).abs().max().item())
dE_b1 = float((EB[0] - E1[0]).abs().item())
nmax = min(FB.shape[1], F1.shape[1])
dF_par = float((FB[:, :nmax] - FB[0:1, :nmax]).abs().max().item())
dF_b1 = float((FB[0, :nmax] - F1[0, :nmax]).abs().max().item())
leak = cB.isolation_check(perturb=0.05)
print(f"[G3] (mace-mp-0, real forward) B={B} identical periodic replicas: "
      f"dE(rep-rep0)={dE_par:.3e} dE(B{B}-B1)={dE_b1:.3e} Ha | "
      f"dF(rep-rep0)={dF_par:.3e} dF(B{B}-B1)={dF_b1:.3e} Ha/A | PBC isolation leak={leak:.2e} Ha")
results["G3_single_vs_batched_dF"] = (dE_par < 1e-12 and dE_b1 < 1e-12
                                      and dF_par < 1e-12 and dF_b1 < 1e-12 and leak == 0.0)


# ===================================================================== G3-bonus MACEBatchCalc real forward
# Direct evidence on MACEBatchCalc's OWN 6-arg forward via an EAGER mace-mp-0 wrapper
# (stand-in for the not-yet-existent traced 6-arg model). Bonus: not a required gate.
class Eager6(torch.nn.Module):
    def __init__(self, mace):
        super().__init__(); self.mace = mace
    def forward(self, positions, node_attrs, edge_index, shifts, batch, ptr):
        Bn = int(ptr.numel() - 1); dev, dt = positions.device, positions.dtype
        data = {"positions": positions, "node_attrs": node_attrs,
                "edge_index": edge_index, "shifts": shifts,
                "unit_shifts": torch.zeros_like(shifts),
                "batch": batch, "ptr": ptr,
                "cell": torch.zeros((Bn, 3, 3), dtype=dt, device=dev),
                "head": torch.zeros(Bn, dtype=torch.long, device=dev)}
        out = self.mace(data, compute_force=False, training=False)
        return out["energy"].reshape(-1)

try:
    def make_fwd_calc():
        c = make_edgeonly_calc(); c.model = Eager6(_mace); return c
    cBb = make_fwd_calc(); cBb.prepare([at.copy() for _ in range(B)])
    EBb, FBb = cBb.get_ef_gpu()
    c1b = make_fwd_calc(); c1b.prepare([at.copy()])
    E1b, F1b = c1b.get_ef_gpu()
    dEp = float((EBb - EBb[0]).abs().max().item())
    dEb = float((EBb[0] - E1b[0]).abs().item())
    nm = min(FBb.shape[1], F1b.shape[1])
    dFp = float((FBb[:, :nm] - FBb[0:1, :nm]).abs().max().item())
    dFb = float((FBb[0, :nm] - F1b[0, :nm]).abs().max().item())
    leakb = cBb.isolation_check(perturb=0.05)
    print(f"[G3-bonus] (MACEBatchCalc OWN 6-arg forward, eager mace-mp-0) B={B}: "
          f"dE(rep-rep0)={dEp:.3e} dE(B{B}-B1)={dEb:.3e} | dF(rep-rep0)={dFp:.3e} "
          f"dF(B{B}-B1)={dFb:.3e} Ha/A | isolation leak={leakb:.2e} Ha")
    results["G3bonus_MACEBatchCalc_own_forward"] = (dEp < 1e-12 and dEb < 1e-12
                                                    and dFp < 1e-12 and dFb < 1e-12 and leakb == 0.0)
except Exception as e:
    import traceback
    print(f"[G3-bonus] eager MACEBatchCalc forward unavailable: {type(e).__name__}: {str(e)[:200]}")
    traceback.print_exc()
    results["G3bonus_MACEBatchCalc_own_forward"] = None


# ===================================================================== G5 mace-mp NVT-PBC bulk water
def oo_rdf_first_peak(atoms, rmax=6.0, nbins=120):
    O = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "O"]
    sub = atoms[O]; d = sub.get_all_distances(mic=True)
    iu = np.triu_indices(len(O), 1); dd = d[iu]; dd = dd[dd < rmax]
    if dd.size == 0:
        return None, None
    hist, edges = np.histogram(dd, bins=nbins, range=(0, rmax))
    r = 0.5*(edges[1:]+edges[:-1]); shell = 4*np.pi*r**2*(edges[1]-edges[0])
    g = hist/np.maximum(shell, 1e-9); m = r > 2.0
    if not np.any(g[m]):
        return None, None
    idx = np.argmax(g[m]); return float(r[m][idx]), float(np.min(dd))

try:
    from maple.function.dispatcher.md.ensemble.nvt_batched import BatchedNVT
    nb = 4
    sp = max(3.15, (2*R_MAX + 1.0)/nb)      # side >= 2*r_max
    wa = water_box(nb, sp, jitter=0.10, seed=1)
    wb = water_box(nb, sp, jitter=0.10, seed=2)
    print(f"[G5] bulk water: {len(wa)} atoms/replica, side={wa.cell.lengths()[0]:.2f} A, B=2, mace-mp-0")
    calcN = MaceOffBatchCalc(device=DEV, dtype=DT, model_path=MODEL_MP)
    steps = 1500
    paras = dict(timestep=0.5, steps=steps, temperature=300.0, thermostat="langevin",
                 tau_t=20.0, remove_com_every=100, random_seed=7, verbose=0)
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        outp = f.name
    sim = BatchedNVT(outp, [wa, wb], calc=calcN, paras=paras).run()
    coordN = calcN.coord.detach().cpu().numpy(); ptrN = calcN._ptr
    peaks, dmins, Ttails = [], [], []
    for b in range(2):
        cb = coordN[ptrN[b]:ptrN[b+1]]; srcw = [wa, wb][b]
        fa = Atoms(numbers=srcw.get_atomic_numbers(), positions=cb, cell=srcw.cell, pbc=True)
        rpk, dmin = oo_rdf_first_peak(fa); peaks.append(rpk); dmins.append(dmin)
        Tk = np.asarray(sim.results[b]["T_K"]); tail = Tk[int(0.6*len(Tk)):]
        Ttails.append(float(np.mean(tail)))
        print(f"[G5] rep{b}: <T>tail={np.mean(tail):.1f}K sig={np.std(tail):.1f}K  "
              f"O-O 1st peak={rpk}A  min O-O={dmin:.2f}A  <PE>={sim.results[b]['PE_mean']:.4f}Ha")
    T_ok = all(abs(t-300.0) < 60.0 for t in Ttails)
    stable = all((d is not None and d > 1.8) for d in dmins)
    rdf_ok = all((p is not None and 2.3 < p < 3.4) for p in peaks)
    results["G5_nvt_pbc_stable_rdf"] = T_ok and stable and rdf_ok
    print(f"[G5] T_ok={T_ok} stable={stable} rdf_ok={rdf_ok}")
except Exception as e:
    import traceback
    print(f"[G5] NVT-PBC run FAILED: {type(e).__name__}: {str(e)[:200]}")
    traceback.print_exc()
    results["G5_nvt_pbc_stable_rdf"] = None


# ===================================================================== SUMMARY
print("\n==================== std-MACE PBC GATE SUMMARY ====================")
order = ["G1_pbc_false_bit_identical", "G2_edge_set_vs_ASE", "G4_min_image_short_contact",
         "XID_cross_calc_identity", "G3_single_vs_batched_dF",
         "G3bonus_MACEBatchCalc_own_forward", "G5_nvt_pbc_stable_rdf"]
n_fail = 0
for k in order:
    v = results.get(k, None)
    tag = "PASS" if v is True else ("FAIL" if v is False else "N/A ")
    if v is False:
        n_fail += 1
    print(f"  {tag}  {k}")
# required gates (bonus excluded); N/A on a required gate counts as failure.
required = [k for k in order if k != "G3bonus_MACEBatchCalc_own_forward"]
req_fail = sum(1 for k in required if results.get(k) is not True)
print(f"\nREQUIRED_GATES_FAIL={req_fail}  (0 => ALL REQUIRED PASS)")
print("STDMACE_PBC_ALL_PASS" if req_fail == 0 else "STDMACE_PBC_SOME_FAIL")
sys.exit(0 if req_fail == 0 else 1)
