#!/usr/bin/env python
"""A3-fwd: NON-VACUOUS gate for MACE-POL edge_mode='full'.

Why this exists: the first parity attempt (macepol_micro.py P_parity) reported
PASS with ``edges_full == edges_radius_kept == 1646`` and ``edge_ratio == 1.0``.
On the ts1x corpus every molecule is smaller than r_max=6.0 A, so the radius
filter keeps 100% of the intramolecular candidate pairs -- the two edge sets are
IDENTICAL and the gate tested NOTHING about the actual question ("do
out-of-range edges contribute exactly zero?"). A gate whose two arms are the
same computation cannot fail. This script builds geometries where the arms
genuinely differ.

Construction: take a ts1x molecule and PULL IT APART -- translate half its atoms
by +dx along x, for dx in {0, 2, 4, 6, 10} A. At dx>0 some intramolecular pairs
exceed r_max, so 'radius' drops them while 'full' keeps them. Reported per dx:
  n_edges_radius vs n_edges_full     (must diverge for the gate to be meaningful)
  |dE|, |dF| full-vs-radius          vs the radius-arm self-spread
A PASS at dx where edge_ratio > 1 is real evidence that the cutoff envelope
zeroes out-of-range edges; a FAIL is the honest answer that 'full' is NOT exact.

NOTE: MACE-POL is used in coupling_mode='approx' (its only batched path). The
documented cross-molecule coupling error is irrelevant here because both arms
share the identical single-graph packed forward -- only the edge set differs.

env: MODEL_DIR PKL OUT
"""
import json
import os
import pickle
import traceback
import warnings

import numpy as np
import torch
from ase import Atoms

MODEL_DIR = os.environ["MODEL_DIR"]
PKL = os.environ["PKL"]
OUT = os.environ["OUT"]
DEV = "cuda"
DX = [0.0, 2.0, 4.0, 6.0, 10.0]

from maple.function.calculator.mace._macepol_batch_calculator import (  # noqa: E402
    MACEPolBatchCalc,
)

RES = {"gate": "edge_mode_full_nonvacuous", "dx_A": DX,
       "gpu": torch.cuda.get_device_name(0)}


def make(edge_mode):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MACEPolBatchCalc(device=DEV, model="macepols",
                                model_path=os.path.join(MODEL_DIR, "macepols.pt"),
                                coupling_mode="approx", edge_mode=edge_mode)


def stretched(rec, dx):
    Z = np.asarray(rec["atomic_numbers"])
    p = np.asarray(rec["transition_state"]["positions"], float).copy()
    half = len(Z) // 2
    order = np.argsort(p[:, 0])          # split along x so the pull is clean
    p[order[half:], 0] += dx
    a = Atoms(numbers=Z, positions=p)
    a.info["charge"] = 0
    a.info["mult"] = 1
    return a


def ef(calc, atoms_list):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc.prepare([a.copy() for a in atoms_list])
    E, F = calc.get_ef_gpu()
    return E.detach().cpu().numpy().copy(), F.detach().cpu().numpy().copy()


def main():
    recs = pickle.load(open(PKL, "rb"))[:8]
    cr = make("radius")
    cf = make("full")
    rows = []
    try:
        for dx in DX:
            al = [stretched(r, dx) for r in recs]
            E0a, F0a = ef(cr, al)
            E0b, F0b = ef(cr, al)                    # same arm, 2nd pass -> self-spread
            sE = float(np.abs(E0a - E0b).max())
            sF = float(np.abs(F0a - F0b).max())
            # edge counts at THIS geometry
            rij = cr.coord[cr.cand_i] - cr.coord[cr.cand_j]
            keep = ((rij * rij).sum(-1) <= (cr.r_max + 1e-12) ** 2)
            n_rad = 2 * int(keep.sum().item())
            E1, F1 = ef(cf, al)
            n_full = int(cf._full_ei.size(1))
            dE = float(np.abs(E1 - E0a).max())
            dF = float(np.abs(F1 - F0a).max())
            thrE = max(3.0 * sE, 1e-9)
            thrF = max(3.0 * sF, 1e-9)
            rows.append({
                "dx_A": dx, "n_edges_radius": n_rad, "n_edges_full": n_full,
                "edge_ratio": n_full / max(1, n_rad),
                "vacuous": n_full == n_rad,
                "radius_self_spread_dE_Ha": sE, "radius_self_spread_dF_HaA": sF,
                "full_vs_radius_maxdE_Ha": dE, "full_vs_radius_maxdF_HaA": dF,
                "thrE": thrE, "thrF": thrF,
                "pass": bool(dE <= thrE and dF <= thrF)})
            print(f"[edgemode] dx={dx:>5.1f} A  edges radius={n_rad} full={n_full} "
                  f"ratio={n_full / max(1, n_rad):.2f} vacuous={n_full == n_rad} | "
                  f"dE={dE:.3e} (thr {thrE:.3e})  dF={dF:.3e} (thr {thrF:.3e})  "
                  f"pass={rows[-1]['pass']}", flush=True)
            json.dump({**RES, "rows": rows}, open(OUT, "w"), indent=1)
    except Exception:
        RES["error"] = traceback.format_exc(limit=4)
        print(RES["error"], flush=True)

    RES["rows"] = rows
    informative = [r for r in rows if not r["vacuous"]]
    RES["n_informative_dx"] = len(informative)
    RES["verdict"] = (
        "NO_EVIDENCE_gate_vacuous" if not informative
        else ("EXACT_on_tested_range" if all(r["pass"] for r in informative)
              else "NOT_EXACT"))
    json.dump(RES, open(OUT, "w"), indent=1)
    print(f"\n[verdict] informative dx points = {len(informative)} -> {RES['verdict']}",
          flush=True)
    print("EDGEMODE_GATE_OK", flush=True)


if __name__ == "__main__":
    main()
