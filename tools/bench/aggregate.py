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


def split_spread(vals, nodes):
    """Separate run-to-run noise from node-to-node variation.

    A spread computed over replicates that landed on DIFFERENT nodes is the SUM of
    both effects. Reporting it as "the baseline noise" lets a candidate arm that
    happens to win a fast node read as a real speedup -- fatal for this campaign,
    where several conclusions live below 10% (B7 retrofit 3.2%, B6 allocator ~3%,
    MACE-POL edge_mode 1-2%, torch.compile 'no gain').

    Returns dict with
      within_node : max spread among replicates SHARING a node (None if no node
                    has >=2 replicates) -- the honest run-to-run noise
      cross_node  : spread over all replicates (run-to-run + node-to-node)
      node_groups : per-node values
      dominant    : 'node' when cross_node >= 2x within_node
    """
    pairs = [(n, v) for n, v in zip(nodes, vals) if v is not None]
    if len(pairs) < 2:
        return dict(within_node=None, cross_node=None, node_groups={},
                    n_nodes=len({n for n, _ in pairs}), dominant=None)
    by = {}
    for n, v in pairs:
        by.setdefault(n or "unknown", []).append(float(v))
    within = [spread_pct(v) for v in by.values() if len(v) >= 2]
    within = max([w for w in within if w is not None], default=None)
    cross = spread_pct([v for _, v in pairs])
    dom = None
    if within is not None and cross is not None:
        dom = "node" if cross >= 2.0 * max(within, 1e-9) else "run"
    return dict(within_node=within, cross_node=cross, node_groups=by,
                n_nodes=len(by), dominant=dom)


def gkey(r):
    mode = r.get("params", {}).get("mode") or r.get("params", {}).get("arm") or ""
    # FIX-1: hessian_mode is part of the identity of a group. grad-equivalents
    # are not commensurable across Hessian modes, so runs that differ in it must
    # never land in the same group (and never be ratioed against each other).
    return (r["bench"], r["dispatcher"], r["backend"], r["B"], mode,
            r.get("hessian_mode"))


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


METRICS = ["wall_s", "s_per_grad_equiv", "grad_equiv_total", "forward_calls",
           "gpu_util_mean", "gpu_util_peak", "vram_reserved_MB"]
SCI = ["success_rate", "barrier_MAE_eV", "median_TS_RMSD_A", "pct_1imag"]


def summarize(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[gkey(r)].append(r)
    rows = []
    for k in sorted(groups, key=str):
        g = groups[k]
        row = dict(bench=k[0], dispatcher=k[1], backend=k[2], B=k[3], mode=k[4],
                   hessian_mode=k[5],
                   n_reps=len(g), reps=[r["rep"] for r in g],
                   nodes=[r["env"].get("node") for r in g],
                   n_distinct_nodes=len({r["env"].get("node") for r in g}),
                   same_node_comparison=(len({r["env"].get("node") for r in g}) == 1),
                   files=[r["_file"] for r in g])
        for m in METRICS:
            pairs = [(r.get("env", {}).get("node"), r.get(m)) for r in g]
            vals = [v for _, v in pairs if v is not None]
            row[m + "_mean"] = float(np.mean(vals)) if vals else None
            row[m + "_per_rep"] = vals
            row[m + "_spread_pct"] = spread_pct(vals)          # == cross-node
            sp = split_spread([v for _, v in pairs], [n for n, _ in pairs])
            row[m + "_within_node_spread_pct"] = sp["within_node"]
            row[m + "_cross_node_spread_pct"] = sp["cross_node"]
            row[m + "_spread_dominant"] = sp["dominant"]
            row[m + "_by_node"] = sp["node_groups"]
        # a broken instrument must never look like a missing cell -- but a record
        # written BEFORE inline certification existed is neither: it is LEGACY, and
        # its s/grad-equiv is kept and flagged rather than destroyed (the counter
        # for that revision is certified by its companion `counter` run instead).
        row["counter_status"] = sorted({r.get("counter_status", "LEGACY(no inline certification)")
                                        for r in g})
        row["grad_equiv_status"] = sorted({r.get("grad_equiv_status", "LEGACY") for r in g})
        if any(st not in ("OK", "LEGACY") for st in row["grad_equiv_status"]):
            row["s_per_grad_equiv_mean"] = None
            row["s_per_grad_equiv_spread_pct"] = None
            row["grad_equiv_total_mean"] = None
            row["s_per_grad_equiv_note"] = (
                "UNAVAILABLE -- the GradCounter is not verified for this backend "
                "(see counter_status). This is a BROKEN INSTRUMENT, not a missing "
                "measurement: wall_s and forward_calls in this row are still valid.")
        row["sampler_source"] = sorted({(r.get("gpu_sampler_source")
                                         or r.get("extra", {}).get("sampler_source")
                                         or "unknown") for r in g})
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
    # match on everything EXCEPT hessian_mode first, so a mode change is reported
    # as INCOMPARABLE rather than silently NO_BASELINE
    bmap = {}
    for r in baseline["groups"]:
        bmap.setdefault((r["bench"], r["dispatcher"], r["backend"], r["B"],
                         r["mode"]), []).append(r)
    out = []
    for r in rows:
        k = (r["bench"], r["dispatcher"], r["backend"], r["B"], r["mode"])
        cands = bmap.get(k, [])
        if not cands:
            out.append(dict(key=str(k), verdict="NO_BASELINE"))
            continue
        same_mode = [b for b in cands if b.get("hessian_mode") == r.get("hessian_mode")]
        b = same_mode[0] if same_mode else cands[0]
        mode_match = bool(same_mode)
        rec = dict(key=str(k), hessian_mode=r.get("hessian_mode"),
                   baseline_hessian_mode=b.get("hessian_mode"),
                   hessian_mode_match=mode_match)
        # FIX-1: wall_s and forward_calls stay valid across Hessian modes;
        # s_per_grad_equiv does NOT (different denominator definition).
        for m in ("wall_s", "forward_calls", "s_per_grad_equiv"):
            if m == "s_per_grad_equiv" and (
                    any(st not in ("OK", "LEGACY")
                        for st in r.get("grad_equiv_status", ["LEGACY"]))
                    or any(st not in ("OK", "LEGACY")
                           for st in b.get("grad_equiv_status", ["LEGACY"]))):
                rec[m] = dict(verdict="UNAVAILABLE",
                              reason="GradCounter unverified on one side -- broken "
                                     "instrument, not a missing measurement")
                continue
            if m == "s_per_grad_equiv" and not mode_match:
                rec[m] = dict(verdict="INCOMPARABLE",
                              reason=("hessian_mode differs (%s vs baseline %s): a "
                                      "numerical FD Hessian spends 2*3N counted "
                                      "forwards per structure while an autograd "
                                      "Hessian spends 1 forward + an uncounted "
                                      "double-backward -> the denominators are not "
                                      "the same quantity. Compare wall_s / "
                                      "forward_calls instead."
                                      % (r.get("hessian_mode"), b.get("hessian_mode"))))
                continue
            cm, bm = r.get(m + "_mean"), b.get(m + "_mean")
            sp = b.get(m + "_spread_pct") or 0.0
            if cm is None or bm is None or bm == 0:
                rec[m] = dict(verdict="SKIP")
                continue
            dpct = 100.0 * (cm - bm) / bm
            within = b.get(m + "_within_node_spread_pct")
            cross = b.get(m + "_cross_node_spread_pct") or sp
            same_node = (r.get("nodes") and b.get("nodes")
                         and set(r["nodes"]) == set(b["nodes"])
                         and len(set(r["nodes"])) == 1)
            band = (within if (same_node and within is not None) else cross)
            band_kind = ("within_node" if (same_node and within is not None)
                         else "cross_node")
            v = ("WITHIN_SPREAD" if abs(dpct) <= max(band or 0.0, 1e-9)
                 else ("FASTER" if dpct < 0 else "SLOWER"))
            rec[m] = dict(delta_pct=round(dpct, 2), band_pct=band, band_kind=band_kind,
                          baseline_within_node_spread_pct=within,
                          baseline_cross_node_spread_pct=cross,
                          candidate_nodes=r.get("nodes"), baseline_nodes=b.get("nodes"),
                          same_node_comparison=bool(same_node), verdict=v,
                          caveat=(None if band_kind == "within_node" else
                                  "CROSS-NODE comparison: the band includes "
                                  "node-to-node variation, so a verdict outside it "
                                  "may still be a node effect. Prefer same-node "
                                  "(or same-job, back-to-back) arms."))
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
              f"{r['mode']:<10} hess={str(r['hessian_mode']):<10} reps={r['n_reps']} "
              f"wall={r['wall_s_mean']} (within-node {r['wall_s_within_node_spread_pct']}% / "
              f"cross-node {r['wall_s_cross_node_spread_pct']}%, {r['n_distinct_nodes']} node(s)) "
              f"s/gE={r['s_per_grad_equiv_mean'] if r['s_per_grad_equiv_mean'] is not None else r.get('grad_equiv_status')}")
    if b3:
        print(f"  [B3 paired] {json.dumps({k: v for k, v in b3.items() if k != 'note'})}")
    if a.csv:
        cols = ["bench", "dispatcher", "backend", "B", "mode", "hessian_mode", "n_reps",
                "wall_s_mean", "wall_s_spread_pct", "s_per_grad_equiv_mean",
                "s_per_grad_equiv_spread_pct", "grad_equiv_total_mean",
                "forward_calls_mean", "wall_s_within_node_spread_pct",
                "wall_s_cross_node_spread_pct", "wall_s_spread_dominant",
                "nodes", "n_distinct_nodes", "same_node_comparison",
                "counter_status", "grad_equiv_status", "sampler_source",
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
