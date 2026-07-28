#!/usr/bin/env python
"""Aggregate maple-bench-v1 run JSONs -> summary CSV (+ optional BASELINE json,
+ delta-vs-baseline verdicts, + B3 autoneb serial/batched pairing).

Usage
-----
  aggregate.py <rundir> [--baseline BASELINE.json] [--write-baseline OUT.json]
               [--csv OUT.csv]

Group key: (bench, dispatcher, backend, B, mode/arm). Per group: mean + full-range
spread (%) over replicates for wall_s, s_per_grad_equiv, grad_equiv_total, util,
VRAM, science. With --baseline: Delta% vs baseline mean, verdict
  WITHIN_SPREAD   |Delta| <= baseline spread  (== "no regression", NOT a speedup)
  FASTER / SLOWER outside the spread band.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np


def kabsch_rmsd(P, Q):
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return float(np.sqrt(((Pc @ (U @ np.diag([1, 1, d]) @ Vt) - Qc) ** 2).sum(1).mean()))


def spread_pct(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) < 2 or v.mean() == 0:
        return None
    return float(100.0 * (v.max() - v.min()) / v.mean())


def gkey(r):
    mode = r.get("params", {}).get("mode") or r.get("params", {}).get("arm") or ""
    return (r["bench"], r["dispatcher"], r["backend"], r["B"], mode)


def collect(rundir):
    runs = []
    for p in sorted(glob.glob(os.path.join(rundir, "*.json"))):
        if p.endswith(".partial.json") or os.path.basename(p).startswith(("BASELINE", "SUMMARY")):
            continue
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r.get("schema") == "maple-bench-v1":
            r["_file"] = os.path.basename(p)
            runs.append(r)
    return runs


METRICS = ["wall_s", "s_per_grad_equiv", "grad_equiv_total", "gpu_util_mean",
           "gpu_util_peak", "vram_reserved_MB"]
SCI = ["success_rate", "barrier_MAE_eV", "median_TS_RMSD_A", "pct_1imag"]


def summarize(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[gkey(r)].append(r)
    rows = []
    for k in sorted(groups, key=str):
        g = groups[k]
        row = dict(bench=k[0], dispatcher=k[1], backend=k[2], B=k[3], mode=k[4],
                   n_reps=len(g), reps=[r["rep"] for r in g],
                   nodes=[r["env"].get("node") for r in g],
                   files=[r["_file"] for r in g])
        for m in METRICS:
            vals = [r.get(m) for r in g if r.get(m) is not None]
            row[m + "_mean"] = float(np.mean(vals)) if vals else None
            row[m + "_per_rep"] = vals
            row[m + "_spread_pct"] = spread_pct(vals)
        for m in SCI:
            vals = [r.get("science", {}).get(m) for r in g
                    if r.get("science", {}).get(m) is not None]
            row["sci_" + m + "_mean"] = float(np.mean(vals)) if vals else None
            row["sci_" + m + "_per_rep"] = vals
        rows.append(row)
    return rows


def pair_autoneb(runs):
    """B3: pair serial vs batched per-reaction on the non-censored intersection."""
    ser, bat = {}, {}
    for r in runs:
        if r["bench"] != "autoneb_B3":
            continue
        d = ser if r["params"]["arm"] == "serial" else bat
        for row in r["extra"]["rows"]:
            if not row["censored"]:
                d.setdefault(row["idx"], []).append(row)
    common = sorted(set(ser) & set(bat))
    if not common:
        return None
    def agg(d, idxs):
        w = [np.mean([x["wall_s"] for x in d[i]]) for i in idxs]
        g = [np.mean([x["grad_equiv"] for x in d[i]]) for i in idxs]
        return float(np.sum(w)), float(np.sum(g))
    ws, gs = agg(ser, common)
    wb, gb = agg(bat, common)
    dbar, rmsd = [], []
    for i in common:
        b_s = np.mean([x["barrier_Eh"] for x in ser[i]])
        b_b = np.mean([x["barrier_Eh"] for x in bat[i]])
        dbar.append(abs(b_s - b_b))
        rmsd.append(kabsch_rmsd(np.asarray(ser[i][0]["ts_pos"]),
                                np.asarray(bat[i][0]["ts_pos"])))
    n_cens = len(set(ser) ^ set(bat))
    return dict(
        n_paired=len(common), censored_or_missing=n_cens,
        wall_serial_s=ws, wall_batched_s=wb,
        wall_speedup=ws / wb if wb else None,
        grad_equiv_serial=gs, grad_equiv_batched=gb,
        grad_equiv_ratio_batched_over_serial=gb / gs if gs else None,
        s_per_gE_serial=ws / gs if gs else None,
        s_per_gE_batched=wb / gb if gb else None,
        s_per_gE_speedup=(ws / gs) / (wb / gb) if (gs and gb) else None,
        d_barrier_median_Eh=float(np.median(dbar)),
        d_barrier_max_Eh=float(np.max(dbar)),
        ts_rmsd_median_A=float(np.median(rmsd)),
        ts_rmsd_max_A=float(np.max(rmsd)),
        note="paired on non-censored intersection; wall paired with grad-equiv (iron rule)")


def compare(rows, baseline):
    bmap = {(r["bench"], r["dispatcher"], r["backend"], r["B"], r["mode"]): r
            for r in baseline["groups"]}
    out = []
    for r in rows:
        k = (r["bench"], r["dispatcher"], r["backend"], r["B"], r["mode"])
        b = bmap.get(k)
        if not b:
            out.append(dict(key=str(k), verdict="NO_BASELINE"))
            continue
        rec = dict(key=str(k))
        for m in ("wall_s", "s_per_grad_equiv"):
            cm, bm = r.get(m + "_mean"), b.get(m + "_mean")
            sp = b.get(m + "_spread_pct") or 0.0
            if cm is None or bm is None or bm == 0:
                rec[m] = dict(verdict="SKIP")
                continue
            dpct = 100.0 * (cm - bm) / bm
            v = ("WITHIN_SPREAD" if abs(dpct) <= max(sp, 1e-9)
                 else ("FASTER" if dpct < 0 else "SLOWER"))
            rec[m] = dict(delta_pct=round(dpct, 2), baseline_spread_pct=sp, verdict=v)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--baseline")
    ap.add_argument("--write-baseline")
    ap.add_argument("--csv")
    a = ap.parse_args()
    runs = collect(a.rundir)
    rows = summarize(runs)
    b3 = pair_autoneb(runs)
    print(f"[aggregate] {len(runs)} runs -> {len(rows)} groups")
    for r in rows:
        print(f"  {r['bench']:<15} {r['dispatcher']:<9} {r['backend']:<6} B={r['B']:<4} "
              f"{r['mode']:<10} reps={r['n_reps']} wall={r['wall_s_mean']} "
              f"(spread {r['wall_s_spread_pct']}%) s/gE={r['s_per_grad_equiv_mean']}")
    if b3:
        print(f"  [B3 paired] {json.dumps({k: v for k, v in b3.items() if k != 'note'})}")
    if a.csv:
        cols = ["bench", "dispatcher", "backend", "B", "mode", "n_reps",
                "wall_s_mean", "wall_s_spread_pct", "s_per_grad_equiv_mean",
                "s_per_grad_equiv_spread_pct", "grad_equiv_total_mean",
                "gpu_util_mean_mean", "gpu_util_peak_mean", "vram_reserved_MB_mean",
                "sci_success_rate_mean", "sci_barrier_MAE_eV_mean",
                "sci_median_TS_RMSD_A_mean", "sci_pct_1imag_mean"]
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"[aggregate] csv -> {a.csv}")
    if a.write_baseline:
        payload = dict(schema="maple-bench-baseline-v1",
                       commit=os.environ.get("MAPLE_COMMIT", "bc3335e"),
                       groups=rows, b3_paired=b3,
                       n_runs=len(runs))
        json.dump(payload, open(a.write_baseline, "w"), indent=1, default=float)
        print(f"[aggregate] baseline -> {a.write_baseline}")
    if a.baseline:
        base = json.load(open(a.baseline))
        verdicts = compare(rows, base)
        print(json.dumps(verdicts, indent=1))
    print("AGGREGATE_OK")


if __name__ == "__main__":
    main()
