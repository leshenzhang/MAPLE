#!/usr/bin/env python
"""A3-fwd re-run of the two gates that errored in fwd_micro.py (G3 stale-edge
evidence, G4 edge-vs-ASE), with the root cause fixed.

Root cause of the first attempt's failure (BOTH gates, same line):
``AtomicData.from_ase(..., r_edges=True)`` routes through
``get_neighbors_pymatgen`` -> ``AseAtomsAdaptor.get_structure`` -> lattice
inverse, which raises ``numpy.linalg.LinAlgError: Singular matrix`` on a
molecule with a zero cell. So the precomputed-edge builder cannot even RUN on a
non-periodic molecule in this fairchem build. Fix: wrap each molecule in a large
cubic cell (BOX A, >= 2*radius + molecule extent) so the minimum-image edge set
is IDENTICAL to the molecular one, and use the same celled atoms for the ASE
reference.

Gates:
  G4 edge reference : fairchem r_edges edge set vs ase.neighborlist at radius 6.0
                      -> symmetric difference must be 0 (10 structures)
  G3a guard         : UMABatchCalc with _r_edges forced True must RAISE at prepare()
  G3b stale evidence: raw predictor built with external_graph_gen=True --
                      pos-overwrite on a prebuilt AtomicData vs a fresh rebuild at
                      the SAME displaced geometry. A nonzero energy difference is
                      the silent wrongness the guard blocks.
  G3c molecular-path note: from_ase(r_edges=True) on a CELL-LESS molecule raises
                      LinAlgError -> record it (the guard is still the right
                      behavior, but the molecular failure mode is a crash, not a
                      silent wrong number; the silent mode needs a cell).

env: MODEL PKL OUT
"""
import json
import os
import pickle
import traceback
from functools import partial

import numpy as np
import torch
from ase import Atoms

MODEL = os.environ["MODEL"]
PKL = os.environ["PKL"]
OUT = os.environ["OUT"]
DEV = "cuda"
BOX = 60.0
RADIUS = 6.0

from maple.function.calculator.uma._uma_batch_calculator import UMABatchCalc  # noqa: E402

RES = {"gate": "R3-1_edges", "box_A": BOX, "radius_A": RADIUS,
       "gpu": torch.cuda.get_device_name(0)}


def load_atoms(n=12, cell=True):
    recs = pickle.load(open(PKL, "rb"))[:n]
    out = []
    for r in recs:
        a = Atoms(numbers=np.asarray(r["atomic_numbers"]),
                  positions=np.asarray(r["transition_state"]["positions"], float))
        a.info["spin"] = 1
        a.info["charge"] = 0
        if cell:
            a.set_cell([BOX, BOX, BOX])
            a.set_pbc(True)
            a.center()
        out.append(a)
    return out


def main():
    from fairchem.core.calculate.ase_calculator import AtomicData
    from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
    from ase.neighborlist import neighbor_list

    a2g = partial(AtomicData.from_ase, task_name="omol", r_edges=True,
                  r_data_keys=["spin", "charge"], max_neigh=300, radius=RADIUS)

    # ---- G4: edge set vs ASE ------------------------------------------------
    try:
        worst, per = 0, []
        for a in load_atoms(10):
            ad = a2g(a)
            ei = ad.edge_index.cpu().numpy()
            mine = set(zip(ei[0].tolist(), ei[1].tolist()))
            i, j = neighbor_list("ij", a, RADIUS)
            ref = set(zip(i.tolist(), j.tolist()))
            sd = len(mine ^ ref)
            per.append({"n_atoms": len(a), "n_edges_fairchem": len(mine),
                        "n_edges_ase": len(ref), "symdiff": sd})
            worst = max(worst, sd)
        RES["G4_edge_vs_ase"] = {"n_structs": 10, "max_symdiff": worst,
                                 "pass": worst == 0, "per_struct": per}
    except Exception:
        RES["G4_edge_vs_ase"] = {"error": traceback.format_exc(limit=4)}
    print(f"[G4] {json.dumps({k: v for k, v in RES['G4_edge_vs_ase'].items() if k != 'per_struct'})}",
          flush=True)

    # ---- G3c: molecular (cell-less) r_edges build ---------------------------
    try:
        amol = load_atoms(1, cell=False)[0]
        a2g(amol)
        RES["G3c_cellless_r_edges"] = {"raises": False,
                                       "note": "built successfully (no crash)"}
    except Exception as exc:
        RES["G3c_cellless_r_edges"] = {
            "raises": True, "exc": f"{type(exc).__name__}: {str(exc)[:160]}",
            "note": "from_ase(r_edges=True) cannot build a cell-less molecule in this "
                    "fairchem build -> on the MOLECULAR path an external_graph_gen=True "
                    "predictor crashes rather than silently going stale; the silent-stale "
                    "mode needs a cell (see G3b)"}
    print(f"[G3c] {json.dumps(RES['G3c_cellless_r_edges'])}", flush=True)

    # ---- G3a: the guard fires ----------------------------------------------
    g3a = {}
    try:
        c = UMABatchCalc(MODEL, device=DEV, dtype=torch.float64, task="omol",
                         fast_inference=False)
        g3a["r_edges_as_built"] = bool(c._r_edges)
        g3a["external_graph_gen_resolved"] = bool(
            getattr(c._predictor.inference_settings, "external_graph_gen", False))
        c._r_edges = True
        c._a2g = a2g
        try:
            c.prepare([a.copy() for a in load_atoms(2, cell=False)])
            g3a["guard_raises_molecular"] = False
        except NotImplementedError as exc:
            g3a["guard_raises_molecular"] = True
            g3a["msg"] = str(exc)[:220]
        # periodic batch: prepare() exempts it, but get_efh_gpu/hvp must still raise
        c2 = UMABatchCalc(MODEL, device=DEV, dtype=torch.float64, task="omat",
                          fast_inference=False)
        c2._r_edges = True
        c2._a2g = partial(AtomicData.from_ase, task_name="omat", r_edges=True,
                          r_data_keys=["spin", "charge"], max_neigh=300, radius=RADIUS)
        c2.prepare([a.copy() for a in load_atoms(2, cell=True)])
        g3a["periodic_prepare_ok"] = True
        for meth in ("get_efh_gpu", "hvp"):
            try:
                if meth == "hvp":
                    c2.hvp(torch.zeros((2, c2.nmax_dof), dtype=torch.float64, device=DEV))
                else:
                    c2.get_efh_gpu()
                g3a[f"{meth}_raises"] = False
            except NotImplementedError:
                g3a[f"{meth}_raises"] = True
            except Exception as exc:
                g3a[f"{meth}_raises"] = f"other: {type(exc).__name__}"
        del c, c2
        torch.cuda.empty_cache()
    except Exception:
        g3a["error"] = traceback.format_exc(limit=4)
    RES["G3a_guard"] = g3a
    print(f"[G3a] {json.dumps(g3a)[:600]}", flush=True)

    # ---- G3b: stale-edge wrongness on a raw external_graph_gen=True predictor
    g3b = {}
    try:
        from fairchem.core.units.mlip_unit import load_predict_unit
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
        iset = InferenceSettings(tf32=False, activation_checkpointing=False,
                                 merge_mole=False, compile=False,
                                 external_graph_gen=True,
                                 internal_graph_gen_version=2)
        pu = load_predict_unit(MODEL, inference_settings=iset, device=DEV)
        a0 = load_atoms(1, cell=True)[0]
        pu.predict(atomicdata_list_to_batch([a2g(a0)]).to(DEV))     # lazy init
        E0 = float(pu.predict(atomicdata_list_to_batch([a2g(a0)]).to(DEV))["energy"]
                   .detach().cpu())
        pd = a0.get_positions()
        pd[0] += np.array([2.5, 0.0, 0.0])
        ad_stale = atomicdata_list_to_batch([a2g(a0)]).to(DEV)
        ad_stale.pos = torch.tensor(pd, dtype=torch.float32, device=DEV)
        E_stale = float(pu.predict(ad_stale)["energy"].detach().cpu())
        a0d = a0.copy()
        a0d.set_positions(pd)
        E_fresh = float(pu.predict(atomicdata_list_to_batch([a2g(a0d)]).to(DEV))
                        ["energy"].detach().cpu())
        n_stale = int(atomicdata_list_to_batch([a2g(a0)]).edge_index.shape[1])
        n_fresh = int(atomicdata_list_to_batch([a2g(a0d)]).edge_index.shape[1])
        g3b = {"E0_eV": E0, "E_stale_eV": E_stale, "E_fresh_eV": E_fresh,
               "stale_minus_fresh_eV": E_stale - E_fresh,
               "n_edges_stale": n_stale, "n_edges_fresh": n_fresh,
               "silent_error_demonstrated": abs(E_stale - E_fresh) > 1e-4}
    except Exception:
        g3b = {"error": traceback.format_exc(limit=4)}
    RES["G3b_stale_evidence"] = g3b
    print(f"[G3b] {json.dumps(g3b)[:600]}", flush=True)

    json.dump(RES, open(OUT, "w"), indent=1)
    ok = (RES.get("G4_edge_vs_ase", {}).get("pass") is True
          and RES.get("G3a_guard", {}).get("guard_raises_molecular") is True)
    print("R31_GATE_" + ("PASS" if ok else "CHECK"), flush=True)
    print("R31_EDGES_OK", flush=True)


if __name__ == "__main__":
    main()
