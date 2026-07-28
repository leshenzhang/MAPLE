#!/usr/bin/env python
"""A3-fwd aggregator: JSON -> markdown tables (run LOCALLY on the pulled JSONs).

usage: python aggregate.py <dir-with-jsons>
"""
import glob
import json
import os
import sys

D = sys.argv[1]


def fmt(x, n=2):
    return "n/a" if x is None else f"{x:.{n}f}"


def spread(v):
    return (max(v) - min(v)) / (sum(v) / len(v)) * 100.0 if v and min(v) > 0 else float("nan")


def micro_table(js, key="configs", label="config"):
    cfgs = list(js[key].keys())
    Bs = js["B_sweep"]
    lines = [f"| B | " + " | ".join(f"{c} ms/fwd (r1,r2)" for c in cfgs) + " | " +
             " | ".join(f"{c} vs base" for c in cfgs if c != "base") + " |",
             "|" + "---|" * (1 + len(cfgs) + len([c for c in cfgs if c != "base"]))]
    for B in Bs:
        cells, ratios = [], []
        base_ms = None
        for c in cfgs:
            e = js[key][c].get("sweep", {}).get(str(B), {})
            ms = e.get("ms_per_forward")
            if ms is None:
                cells.append("FAIL" if "error" in e else "-")
                continue
            m = sum(ms) / len(ms)
            cells.append(f"{ms[0]:.2f}, {ms[1]:.2f} (sp {spread(ms):.1f}%)"
                         if len(ms) > 1 else f"{ms[0]:.2f}")
            if c == "base":
                base_ms = m
        for c in cfgs:
            if c == "base":
                continue
            e = js[key][c].get("sweep", {}).get(str(B), {})
            ms = e.get("ms_per_forward")
            if ms is None or base_ms is None:
                ratios.append("-")
            else:
                ratios.append(f"{base_ms / (sum(ms) / len(ms)):.3f}x")
        lines.append(f"| {B} | " + " | ".join(cells) + " | " + " | ".join(ratios) + " |")
    return "\n".join(lines)


def main():
    f = os.path.join(D, "fwd_micro_uma.json")
    if os.path.exists(f):
        js = json.load(open(f))
        print("### UMA micro (ms/forward, 2 reps)\n")
        print(micro_table(js))
        print("\n### UMA gates\n")
        for k in ("G1_parity", "G2_staleness_probe", "G3_r31_guard", "G4_edge_vs_ase",
                  "cueq_probe"):
            v = js.get(k)
            if v is not None:
                v2 = {kk: vv for kk, vv in v.items() if "error" not in kk} if isinstance(v, dict) else v
                print(f"- **{k}**: `{json.dumps(v2)[:900]}`")
        for c, e in js.get("configs", {}).items():
            if "inference_settings" in e:
                print(f"- settings[{c}]: `{json.dumps(e['inference_settings'])}` r_edges={e.get('r_edges')}")
    f = os.path.join(D, "macepol_micro.json")
    if os.path.exists(f):
        js = json.load(open(f))
        print("\n### MACE-POL micro (ms/forward, 2 reps)\n")
        print(micro_table(js, key="arms"))
        print("\n### MACE-POL gates\n")
        print(f"- P_parity: `{json.dumps(js.get('P_parity'))}`")
        cg = js.get("cuda_graph", {})
        for B, r in cg.items():
            r2 = {k: v for k, v in r.items() if k != "error"}
            print(f"- cuda_graph B={B}: `{json.dumps(r2)}`")
            if "error" in r:
                print(f"  - error head: `{r['error'].splitlines()[-1][:300]}`")
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "e2e_*.json"))):
        js = json.load(open(f))
        rows.append(js)
    if rows:
        print("\n### end-to-end (N=100, B=16)\n")
        print("| tag | fast_inf | wall s (NEB/PRFO/freq) | struct/s | grad-equiv | "
              "s/grad-equiv | util% | VRAM MB | succ% | MAE eV | TS-RMSD Å |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for js in rows:
            print(f"| {js['tag']} | {js.get('fast_inference')} | "
                  f"{js['wall_total']:.0f} ({js['wall_neb']:.0f}/{js['wall_prfo']:.0f}/"
                  f"{js['wall_freq']:.0f}) | {js['struct_per_s']:.4f} | "
                  f"{js['grad_equiv_total']} | "
                  f"{js['wall_total'] / js['grad_equiv_total'] * 1e3:.3f} ms | "
                  f"{js['util_mean']:.0f} | {js['peak_vram_smi_MB']:.0f} | "
                  f"{js['success_pct']:.1f} | {js['barrier_MAE_eV']:.3f} | "
                  f"{js['median_TS_RMSD_A']:.3f} |")


if __name__ == "__main__":
    main()
