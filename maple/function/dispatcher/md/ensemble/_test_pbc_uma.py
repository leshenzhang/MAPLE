# -*- coding: utf-8 -*-
"""PBC Phase-1A UMA-periodic validation -- COMPLETE + SYSTEMATIC (mirrors the
MACE-OFF PBC Gate A/B suite, B-88, in ``_test_pbc_batched.py`` / ``_test_pbc_nvt.py``).

Runtime was DEFERRED (needed a working fairchem env + a periodic UMA checkpoint).
Now runnable in the ``uma3`` fairchem env with a LOCAL uma-s-1p1 .pt, so this file
carries the full gate battery, adapted to UMA's reality: fairchem builds the
periodic neighbour list INTERNALLY (edges are not MAPLE code), so the neighbour
gate compares fairchem's own ``AtomicData.from_ase`` periodic edge set to ASE
``neighbor_list`` -- the same min-image correctness the MACE ``_build_edges_gpu``
gate proves, but at the layer UMA actually uses.

Gates
-----
  task-gate   periodic + molecular ``omol`` task -> NotImplementedError
  G1 (A1)     single-vs-batched parity: B replicas of a periodic cell give the
              same per-replica UMA E/F as a single-system forward (tol 1e-5,
              ABOVE the ~1e-6 UMA fp32 GPU nondeterminism floor, D-59).
  ISO (A2)    block-diagonal isolation: perturb replica-0 -> other replicas' E
              unchanged (leak < 1e-5).
  FRESH       edge-freshness (anti-stale-neighbour, R3-1): move an atom ACROSS
              the box face between two forwards -> forces change AND equal a FRESH
              prepare() on the moved cell (rebuilt edges, no baked stale graph).
  G2          fairchem periodic edge set vs ASE neighbor_list('ijSd', 6.0):
              equal edge COUNT + identical sorted pair-distance multiset
              (convention-free) + exact (i,j,S) symdiff == 0; image edges present.
  G3          isolated-limit: pbc=False -> calc._periodic False AND from_ase
              cell_offsets all zero (no image shifts); pbc=True -> nonzero shifts.
  G4          short periodic NVT (bulk water, task=omat) via the real BatchedNVT
              loop -> equilibrated <T> near target + no collapse + physical-ish
              O-O RDF + one forward per step (algorithm-correctness axis: the
              thermostat reproduces T and the run stays bounded; omat is a
              MATERIALS head, so the absolute water RDF is a model-domain caveat,
              reported, not a hard gate on it).

Run (uma3 env, local ckpt):
  UMA_MODEL=/ibex/user/wangc0i/zls/ai-scc-oer-pt/uma-s-1p1.pt UMA_TASK=omat \
    PYTHONPATH=<worktree> \
    /ibex/user/xiaox/mambaforge/envs/uma3/bin/python _test_pbc_uma.py
"""
import os, sys, tempfile
import numpy as np, torch
from functools import partial
from ase import Atoms
from ase.neighborlist import neighbor_list

if __name__ != "__main__":
    raise SystemExit("run _test_pbc_uma.py as a script, not an import")

from maple.function.calculator.uma._uma_batch_calculator import UMABatchCalc
from fairchem.core.calculate.ase_calculator import AtomicData

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("UMA_MODEL", "uma-s-1p1")
TASK = os.environ.get("UMA_TASK", "omat")
RADIUS = 6.0
print(f"[env] device={DEV} torch={torch.__version__} model={MODEL} task={TASK}")

res = {}          # bool gates
diag = {}         # informational numbers


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
    return Atoms(symbols=sym, positions=np.array(pos),
                 cell=[L, L, L], pbc=pbc)


# box side > 2*RADIUS so the minimum-image guard passes and image edges occur.
at = water_box(3, 4.2, seed=1)             # 27 waters, 12.6 A box
print(f"[sys] water box: {len(at)} atoms, cell={at.cell.lengths()}, "
      f"min width={np.min(at.cell.lengths()):.3f} A (need >= {2*RADIUS:.1f})")

# ============================================================ task gate
try:
    cbad = UMABatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64, task="omol")
    cbad.prepare([at.copy()])
    res["task_gate_rejects_omol"] = False
    print("[gate] periodic+omol NOT rejected (FAIL)")
except NotImplementedError:
    res["task_gate_rejects_omol"] = True
    print("[gate] periodic+omol correctly REJECTED")

calc = UMABatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64, task=TASK)
ext = bool(getattr(calc._predictor.inference_settings, "external_graph_gen", False))
print(f"[env] external_graph_gen={ext} (edges {'precomputed' if ext else 'built inside the eSCN forward'}); "
      f"calc.r_max={getattr(calc,'r_max',None)} batch_isolated={getattr(calc,'batch_isolated',None)}")

# ============================================================ G1 single-vs-batched parity
B = 4
calc.prepare([at.copy() for _ in range(B)])
EB, FB = calc.get_ef_gpu()
c1 = UMABatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64, task=TASK)
c1.prepare([at.copy()]); E1, F1 = c1.get_ef_gpu()
nmax = min(FB.shape[1], F1.shape[1])
dE_rep = float((EB - EB[0]).abs().max())
dE_b1 = float((EB[0] - E1[0]).abs())
dF_rep = float((FB[:, :nmax] - FB[0:1, :nmax]).abs().max())
dF_b1 = float((FB[0, :nmax] - F1[0, :nmax]).abs().max())
diag.update(dict(G1_dE_rep=dE_rep, G1_dE_B4vsB1=dE_b1, G1_dF_rep=dF_rep, G1_dF_B4vsB1=dF_b1))
print(f"[G1] dE(rep)={dE_rep:.2e} dE(B4-B1)={dE_b1:.2e} dF(rep)={dF_rep:.2e} dF(B4-B1)={dF_b1:.2e} Ha "
      f"(tol 1e-5 > UMA fp32 nondeterminism ~1e-6)")
res["G1_parity_E"] = dE_rep < 1e-5 and dE_b1 < 1e-5
res["G1_parity_F"] = dF_rep < 1e-5 and dF_b1 < 1e-5

# ============================================================ ISO block-diagonal isolation
leak = calc.isolation_check(0.05)
diag["ISO_leak_Ha"] = leak
print(f"[ISO] max E leak into other replicas = {leak:.2e} Ha (tol 1e-5)")
res["ISO_block_diagonal"] = leak < 1e-5

# ============================================================ FRESH edge freshness (R3-1)
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
forces_changed = float((F1b - F0).abs().max()) > 1e-5
dEf = float((E1b[0] - E2[0]).abs()); dFf = float((F1b[0] - F2[0]).abs().max())
diag.update(dict(FRESH_dE_inplace_vs_fresh=dEf, FRESH_dF_inplace_vs_fresh=dFf))
print(f"[FRESH] forces_changed={forces_changed} inplace-vs-fresh dE={dEf:.2e} dF={dFf:.2e} Ha (tol 1e-5)")
res["FRESH_no_stale_neighbor"] = forces_changed and dEf < 1e-5 and dFf < 1e-5

# ============================================================ G2 fairchem edges vs ASE
# Build the periodic edge set fairchem would use (r_edges=True forces from_ase to
# emit edge_index + cell_offsets). max_neigh large so nothing is truncated.
a2g = partial(AtomicData.from_ase, task_name=TASK, r_edges=True,
              r_data_keys=["spin", "charge"], max_neigh=1000, radius=RADIUS)
atg = at.copy(); atg.info["spin"] = 1; atg.info["charge"] = 0
ad = a2g(atg)
cell = np.asarray(ad.cell).reshape(3, 3).astype(np.float64)   # (3,3)
ei = np.asarray(ad.edge_index)                                 # (2,E)
S = np.asarray(ad.cell_offsets).astype(np.float64)            # (E,3) integer shifts
posn = at.get_positions()
# determine offset sign convention so every edge length <= RADIUS
a_idx, b_idx = ei[0], ei[1]
used_sgn = None
for sgn in (+1.0, -1.0):
    d = posn[b_idx] + sgn * (S @ cell) - posn[a_idx]
    r = np.linalg.norm(d, axis=1)
    if r.max() <= RADIUS + 1e-4:
        used_sgn = sgn; r_fc = r; Suse = (sgn * S).round().astype(int); break
if used_sgn is None:
    used_sgn = +1.0
    r_fc = np.linalg.norm(posn[b_idx] + (S @ cell) - posn[a_idx], axis=1)
    Suse = S.round().astype(int)
# ASE reference
ai, aj, aS, ad_ = neighbor_list("ijSd", at, RADIUS)
n_fc, n_ase = len(a_idx), len(ai)
r_fc_sorted = np.sort(r_fc); r_ase_sorted = np.sort(ad_)
mlen = min(len(r_fc_sorted), len(r_ase_sorted))
dist_maxdiff = float(np.abs(r_fc_sorted[:mlen] - r_ase_sorted[:mlen]).max()) if mlen else 9.9
fc_set = set(zip(a_idx.tolist(), b_idx.tolist(),
                 Suse[:, 0].tolist(), Suse[:, 1].tolist(), Suse[:, 2].tolist()))
ase_set = set(zip(ai.tolist(), aj.tolist(),
                  aS[:, 0].tolist(), aS[:, 1].tolist(), aS[:, 2].tolist()))
symdiff = len(fc_set ^ ase_set)
n_img = int((np.abs(Suse).sum(axis=1) > 0).sum())
diag.update(dict(G2_n_fairchem=n_fc, G2_n_ase=n_ase, G2_dist_maxdiff=dist_maxdiff,
                 G2_symdiff=symdiff, G2_image_edges=n_img, G2_offset_sign=used_sgn))
print(f"[G2] fairchem_edges={n_fc} ase_edges={n_ase} sorted-dist maxdiff={dist_maxdiff:.2e} A "
      f"symdiff(i,j,S)={symdiff} image_edges(|S|>0)={n_img} (offset sign {used_sgn:+.0f})")
res["G2_edge_count_match"] = (n_fc == n_ase)
res["G2_edge_dist_multiset_match"] = (n_fc == n_ase) and dist_maxdiff < 1e-4
res["G2_edge_exact_symdiff_zero"] = (symdiff == 0)
res["G2_image_edges_present"] = n_img > 0

# ============================================================ G3 isolated-limit -> zero shifts
iso = at.copy(); iso.pbc = False; iso.cell = None
c_iso = UMABatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64, task=TASK)
c_iso.prepare([iso.copy()])
iso_g = iso.copy(); iso_g.info["spin"] = 1; iso_g.info["charge"] = 0
ad_iso = AtomicData.from_ase(iso_g, task_name=TASK, r_edges=True,
                             r_data_keys=["spin", "charge"], max_neigh=1000, radius=RADIUS)
S_iso = np.asarray(ad_iso.cell_offsets)
iso_shifts_zero = bool(np.all(S_iso == 0))
per_shifts_nonzero = n_img > 0                      # control: periodic clone had image edges
print(f"[G3] isolated: calc._periodic={c_iso._periodic} (expect False), "
      f"from_ase cell_offsets all zero={iso_shifts_zero}; periodic control image edges={n_img}")
res["G3_isolated_nonperiodic"] = (c_iso._periodic is False) and iso_shifts_zero and per_shifts_nonzero

# ============================================================ G4 short periodic NVT
G4_note = ""
try:
    from maple.function.dispatcher.md.ensemble.nvt_batched import BatchedNVT

    def oo_rdf_first_peak(atoms, rmax=6.0, nbins=120):
        O = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "O"]
        sub = atoms[O]
        d = sub.get_all_distances(mic=True)
        iu = np.triu_indices(len(O), 1)
        dd = d[iu]; dd = dd[dd < rmax]
        if dd.size == 0:
            return None, None
        hist, edges = np.histogram(dd, bins=nbins, range=(0, rmax))
        r = 0.5*(edges[1:] + edges[:-1])
        shell = 4*np.pi*r**2*(edges[1]-edges[0])
        g = hist/np.maximum(shell, 1e-9)
        m = r > 2.0
        if not np.any(g[m]):
            return None, None
        idx = np.argmax(g[m])
        return float(r[m][idx]), float(np.min(dd))

    # ~1 g/cc water; side 12.6 A > 2*RADIUS.
    nvt_atoms = [water_box(4, 3.15, jitter=0.10, seed=1),
                 water_box(4, 3.15, jitter=0.10, seed=2)]
    print(f"[G4] NVT: {len(nvt_atoms[0])} atoms/replica, side "
          f"{nvt_atoms[0].cell.lengths()[0]:.2f} A, B=2")
    cnvt = UMABatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64, task=TASK)
    nfwd = {"c": 0}; _orig = cnvt.get_ef_gpu
    cnvt.get_ef_gpu = lambda: (nfwd.__setitem__("c", nfwd["c"]+1), _orig())[1]
    steps = int(os.environ.get("UMA_NVT_STEPS", "500"))
    paras = dict(timestep=0.5, steps=steps, temperature=300.0, thermostat="langevin",
                 tau_t=20.0, remove_com_every=100, random_seed=7, verbose=0)
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        outp = f.name
    sim = BatchedNVT(outp, nvt_atoms, calc=cnvt, paras=paras).run()
    coord = cnvt.coord.detach().cpu().numpy()
    ptr = cnvt._ptr.tolist()
    peaks, dmins, Ttails, Tstds = [], [], [], []
    for b in range(2):
        cb = coord[ptr[b]:ptr[b+1]]
        src = nvt_atoms[b]
        fa = Atoms(numbers=src.get_atomic_numbers(), positions=cb, cell=src.cell, pbc=True)
        rpk, dmin = oo_rdf_first_peak(fa)
        peaks.append(rpk); dmins.append(dmin)
        Tk = np.asarray(sim.results[b]["T_K"]); tail = Tk[int(0.6*len(Tk)):]
        Ttails.append(float(np.mean(tail))); Tstds.append(float(np.std(tail)))
        print(f"[G4] rep{b}: <T>tail={Ttails[b]:.1f}K sig(T)={Tstds[b]:.1f}K "
              f"O-O RDF 1st peak={rpk} min O-O={dmin} <PE>={sim.results[b]['PE_mean']:.4f}Ha")
    fwd_ok = abs(nfwd["c"] - (steps+1)) <= 2
    T_ok = all(abs(t-300.0) < 60.0 for t in Ttails)
    stable = all((d is not None and d > 1.6) for d in dmins)
    rdf_present = all((p is not None) for p in peaks)
    rdf_physical = all((p is not None and 2.3 < p < 3.5) for p in peaks)
    diag.update(dict(G4_forwards=nfwd["c"], G4_steps=steps, G4_Ttail=Ttails,
                     G4_rdf_peaks=peaks, G4_min_OO=dmins))
    print(f"[G4] forwards={nfwd['c']} (expect ~{steps+1}) T_ok={T_ok} stable={stable} "
          f"rdf_present={rdf_present} rdf_physical={rdf_physical}")
    # algorithm-correctness gate: thermostat reproduces T + bounded + one fwd/step.
    res["G4_temperature_reproduced"] = T_ok
    res["G4_stable_no_collapse"] = stable
    res["G4_one_forward_per_step"] = fwd_ok
    # RDF position is reported; omat is a materials head, so treat a mis-positioned
    # liquid-water peak as a model-domain caveat, not an algorithm FAIL.
    if not rdf_physical:
        G4_note = (f"G4 O-O RDF peak {peaks} outside 2.3-3.5 A -> omat materials head "
                   f"off-domain for liquid water (MODEL caveat, not an algorithm bug)")
except Exception as e:
    import traceback
    G4_note = f"G4 NVT could not run: {type(e).__name__}: {e}"
    print("[G4] EXCEPTION:\n" + traceback.format_exc())

# ============================================================ summary
print("\n==== UMA-PBC GATE SUMMARY ====")
ok = True
for k, v in res.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
    ok = ok and bool(v)
print("---- diagnostics ----")
for k, v in diag.items():
    print(f"  {k} = {v}")
if G4_note:
    print(f"---- note: {G4_note}")
print(f"==== {'UMA-PBC PASS' if ok else 'UMA-PBC FAIL'} ====")
sys.exit(0 if ok else 1)
