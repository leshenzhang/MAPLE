#!/usr/bin/env python
"""Merge the A2-phess production JSONs into the cost / same-saddle tables.

Usage: analyze_phess.py <dir-with-*.json> [--baseline base_r8]

Cost table columns
  wall_s        mean over replicates (+ per-replicate spread, the noise floor)
  GE            gradient-equivalents; 1 GE = one per-structure force evaluation
  s/GE          the normalized main metric (a wall speedup that burned more
                gradients is not a speedup)
Same-saddle gate (vs the baseline's replicate 1 final geometry)
  dX            max per-atom displacement (Angstrom)
  dE            |E_variant - E_baseline| (Hartree)
  nimag         both index-1 saddles?
"""
import argparse
import glob
import json
import os

import numpy as np


def load(d):
    runs = {}
    for p in sorted(glob.glob(os.path.join(d, "*_r*.json"))):
        try:
            j = json.load(open(p))
        except Exception:
            continue
        if "tag" not in j:
            continue
        runs.setdefault(j["tag"], {})[j["replicate"]] = j
    return runs


def saddle_map(run):
    return {s["i"]: s for s in run.get("saddle", [])}


def gate(var, base, dx_tol=5e-2, de_tol=2e-4):
    gv = var["final_geoms"]
    gb = base["final_geoms"]
    sv, sb = saddle_map(var), saddle_map(base)
    n_pair = n_same = n_dx = 0
    dxs, des = [], []
    for i in range(len(gb)):
        if gv[i] is None or gb[i] is None or i not in sv or i not in sb:
            continue
        if sb[i]["nimag"] != 1:
            continue                     # baseline itself is not an index-1 saddle
        n_pair += 1
        dx = float(np.abs(np.asarray(gv[i]) - np.asarray(gb[i])).max())
        de = abs(sv[i]["E"] - sb[i]["E"])
        dxs.append(dx)
        des.append(de)
        if dx <= dx_tol:
            n_dx += 1
        if dx <= dx_tol and de <= de_tol and sv[i]["nimag"] == 1:
            n_same += 1
    return dict(n_pair=n_pair, n_same=n_same, n_dx_ok=n_dx,
                dx_med=float(np.median(dxs)) if dxs else float("nan"),
                dx_max=float(np.max(dxs)) if dxs else float("nan"),
                de_med=float(np.median(des)) if des else float("nan"),
                de_max=float(np.max(des)) if des else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--baseline", default="base_r8")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    runs = load(args.dir)
    if args.baseline not in runs:
        raise SystemExit(f"baseline {args.baseline} not found in {sorted(runs)}")
    base = runs[args.baseline][sorted(runs[args.baseline])[0]]

    print("## cost table (N=%d)" % len(base["natoms"]))
    hdr = (f"{'config':>16} {'rep':>4} {'wall_s':>9} {'GE_total':>9} {'GE_hess':>9} "
           f"{'GE_iter':>8} {'GE_hvp':>8} {'s/GE(ms)':>9} {'conv':>5} {'evic':>5} "
           f"{'maxit':>5} {'peakMB':>7}")
    print(hdr)
    summary = {}
    for tag in sorted(runs):
        walls, ges = [], []
        for rep in sorted(runs[tag]):
            r = runs[tag][rep]
            lg = r["log"]
            walls.append(r["wall_s"])
            ges.append(lg.get("ge_total", 0))
            spg = (r["wall_s"] / lg["ge_total"] * 1e3) if lg.get("ge_total") else float("nan")
            print(f"{tag:>16} {rep:>4} {r['wall_s']:>9.1f} {lg.get('ge_total', 0):>9} "
                  f"{lg.get('ge_hess', 0):>9} {lg.get('ge_iter', 0):>8} "
                  f"{lg.get('ge_hvp', 0):>8} {spg:>9.3f} {lg.get('n_converged', -1):>5} "
                  f"{lg.get('n_evicted', -1):>5} {lg.get('n_maxiter', -1):>5} "
                  f"{r['peak_MB']:>7.0f}")
        summary[tag] = dict(
            wall_mean=float(np.mean(walls)),
            wall_spread=float((max(walls) - min(walls)) / max(np.mean(walls), 1e-9)),
            ge_mean=float(np.mean(ges)),
            ge_spread=float((max(ges) - min(ges)) / max(np.mean(ges), 1e-9)),
            s_per_ge=float(np.mean(walls) / max(np.mean(ges), 1e-9)),
        )

    b = summary[args.baseline]
    print(f"\n## normalized vs {args.baseline} "
          f"(baseline replicate spread: wall {b['wall_spread'] * 100:.1f}%, "
          f"GE {b['ge_spread'] * 100:.1f}%)")
    print(f"{'config':>16} {'wall_x':>8} {'GE_x':>8} {'s/GE_x':>8} {'wall_spread%':>13} "
          f"{'GE_spread%':>11}")
    for tag in sorted(summary):
        s = summary[tag]
        print(f"{tag:>16} {b['wall_mean'] / s['wall_mean']:>8.3f} "
              f"{b['ge_mean'] / max(s['ge_mean'], 1e-9):>8.3f} "
              f"{b['s_per_ge'] / max(s['s_per_ge'], 1e-12):>8.3f} "
              f"{s['wall_spread'] * 100:>13.1f} {s['ge_spread'] * 100:>11.1f}")

    print(f"\n## same-saddle gate vs {args.baseline} r1 "
          f"(dX<=5e-2 A, |dE|<=2e-4 Ha, both n_imag==1)")
    print(f"{'config':>16} {'rep':>4} {'pairs':>6} {'same':>6} {'dX_ok':>6} "
          f"{'dX_med':>9} {'dX_max':>9} {'dE_med':>10} {'dE_max':>10} {'nimag1':>7}")
    gates = {}
    for tag in sorted(runs):
        for rep in sorted(runs[tag]):
            r = runs[tag][rep]
            g = gate(r, base)
            n1 = sum(s["nimag"] == 1 for s in r.get("saddle", []))
            g["nimag1"] = n1
            gates[f"{tag}_r{rep}"] = g
            print(f"{tag:>16} {rep:>4} {g['n_pair']:>6} {g['n_same']:>6} "
                  f"{g['n_dx_ok']:>6} {g['dx_med']:>9.2e} {g['dx_max']:>9.2e} "
                  f"{g['de_med']:>10.2e} {g['de_max']:>10.2e} {n1:>7}")

    if args.json_out:
        json.dump(dict(summary=summary, gates=gates), open(args.json_out, "w"), indent=1)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
