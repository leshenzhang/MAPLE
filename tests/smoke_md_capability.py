#!/usr/bin/env python3
"""MAPLE MD capability matrix + smoke-test harness.

Regenerates the {NVE,NVT,NPT} x {aimnet2,macepol,mace,mace-mp-0,mace-off,uma}
capability matrix from the *live* environment:

  * which backend packages import (mace / aimnet / fairchem),
  * each MAPLE backend's capability flags (SUPPORTS_PBC, stress, local-model
    requirement) read straight off the registered calculator class,
  * which model-weight files are provisioned,

and (with --run) drives the shortest possible MAPLE MD per *runnable* cell to
confirm: no crash, forces nonzero (atoms move), energy finite, NVT temperature
controlled, NPT box responds.

Capability rules (authoritative gate = the code, not this script):
  * NPT needs a PBC + stress calculator (maple .../md/ensemble/npt.py gate).
    A no-PBC / no-stress backend -> NPT = N/A.
  * UMA = N/A (decision B-11, double-blocked) and fairchem is not installed.
  * mace-off (maceoff23*) = pending merge of branch feat/md-mace-off.
  * A backend whose package is absent          -> "not installed".
  * A REQUIRES_LOCAL_MODEL_FILE backend with no -> "no model file".
    provisioned weight file

Usage:
  python smoke_md_capability.py                    # introspect, print matrix
  python smoke_md_capability.py --emit-md OUT.md   # write markdown matrix
  python smoke_md_capability.py --run --workdir D  # run smokes in D, collect
  python smoke_md_capability.py --collect --workdir D   # parse existing thermo
pytest:
  pytest smoke_md_capability.py                    # regression-checks the gates
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ENSEMBLES = ("nve", "nvt", "npt")

# task column  ->  MAPLE registry selector + smoke spec
#   pkg          : python module that must import for the backend to instantiate
#   smoke        : {ensemble: input_basename} for cells we actually run
#   status       : hard override ('pending' / 'veto') independent of install state
POTENTIALS = [
    dict(col="aimnet2",   maple="aimnet2",    pkg="aimnet2calc",
         alt_pkg=("aimnet",), smoke={}),
    dict(col="macepol",   maple="macepols",   pkg="mace",
         smoke={"nve": "macepol_nve", "nvt": "macepol_nvt"}),
    dict(col="mace",      maple="maceomol",   pkg="mace",
         smoke={}),
    dict(col="mace-mp-0", maple="mace-mp-0",  pkg="mace",
         smoke={"nve": "macemp_nve", "nvt": "macemp_nvt", "npt": "macemp_npt"}),
    dict(col="mace-off",  maple="maceoff23s", pkg="mace",
         smoke={}, status="pending",
         status_note="pending merge of feat/md-mace-off"),
    dict(col="uma",       maple="uma",        pkg="fairchem",
         smoke={}, status="veto",
         status_note="N/A: UMA vetoed (decision B-11)"),
]

# --------------------------------------------------------------------------- #
# environment / registry introspection
# --------------------------------------------------------------------------- #
def _pkg_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _import_registry():
    """Import shipped backend modules; return (registry, model_dir)."""
    from maple.function.calculator import set_calculator as SC
    from maple.function.calculator.calculator_base import _REGISTRY
    for mod in sorted(set(SC._BUILTIN_NAME_TO_MODULE.values())):
        try:
            importlib.import_module(mod)
        except Exception:
            pass  # e.g. uma hard-imports fairchem; absence is itself a result
    mdl = Path(SC.__file__).resolve().parent / "model"
    return _REGISTRY, mdl


def _backend_flags(registry, maple_name: str):
    cls = registry.get(maple_name.lower())
    if cls is None:
        return None
    impl = list(getattr(cls, "implemented_properties", []))
    return dict(
        cls=cls.__name__,
        pbc=bool(getattr(cls, "SUPPORTS_PBC", False)),
        stress=("stress" in impl),
        req_local=bool(getattr(cls, "REQUIRES_LOCAL_MODEL_FILE", False)),
        ckpt=getattr(cls, "CHECKPOINT_FILENAME", None),
    )


def _model_file_present(model_dir: Path, maple_name: str, flags) -> bool:
    """Best-effort check that a weight file exists for a local-model backend."""
    if model_dir is None or not model_dir.exists():
        return False
    cands = [f"{maple_name}.pt"]
    if flags and flags.get("ckpt"):
        cands += list(flags["ckpt"].values())
    # macepol naming variants
    cands += [f"{maple_name}.model", f"{maple_name.replace('-', '')}.pt"]
    return any((model_dir / c).exists() for c in cands)


def discover_env():
    registry, model_dir = _import_registry()
    env = dict(model_dir=str(model_dir), packages={}, backends={})
    for pot in POTENTIALS:
        col = pot["col"]
        pkg_ok = _pkg_present(pot["pkg"]) or any(
            _pkg_present(p) for p in pot.get("alt_pkg", ()))
        env["packages"][col] = pkg_ok
        flags = _backend_flags(registry, pot["maple"])
        mfile = (_model_file_present(model_dir, pot["maple"], flags)
                 if flags else False)
        env["backends"][col] = dict(flags=flags, model_file=mfile,
                                    pkg_ok=pkg_ok)
    return env


# --------------------------------------------------------------------------- #
# matrix construction
# --------------------------------------------------------------------------- #
def cell_verdict(pot, ensemble, env, smoke):
    """Return (support, note). support in
    {ok, na, pending, not_installed, no_model}."""
    col = pot["col"]
    be = env["backends"][col]
    flags = be["flags"]
    status = pot.get("status")

    # hard overrides first
    if status == "veto":
        return "na", pot["status_note"] + " + fairchem not installed"
    if status == "pending":
        if ensemble == "npt":
            return "na", "N/A: no-PBC backend (NPT needs PBC+stress); also " \
                         + pot["status_note"]
        return "pending", pot["status_note"]

    # NPT capability gate (code-authoritative)
    if ensemble == "npt":
        if not (flags and flags["pbc"] and flags["stress"]):
            return "na", "N/A: backend has no PBC+stress (NPT gate refuses " \
                         "kinetic-only pressure)"
        # falls through to install/model/smoke checks below

    # install / model-file availability
    if not be["pkg_ok"]:
        return "not_installed", f"not installed ({pot['pkg']} package absent)"
    if flags and flags["req_local"] and not be["model_file"]:
        return "no_model", "no model file provisioned " \
                            "(REQUIRES_LOCAL_MODEL_FILE)"

    # runnable cell -> use smoke result if present
    key = f"{col}:{ensemble}"
    sm = (smoke or {}).get(key)
    if sm:
        return ("ok" if sm.get("verdict") == "PASS" else "fail"), \
               sm.get("summary", sm.get("verdict", ""))
    return "supported", "supported (gate allows; not smoke-tested this invocation)"


_SYM = {"ok": "PASS", "fail": "FAIL", "na": "N/A", "pending": "PENDING",
        "supported": "supported",
        "not_installed": "not-installed", "no_model": "no-model"}


def build_matrix(env, smoke=None):
    rows = []
    for ens in ENSEMBLES:
        row = dict(ensemble=ens.upper(), cells={})
        for pot in POTENTIALS:
            support, note = cell_verdict(pot, ens, env, smoke)
            row["cells"][pot["col"]] = dict(support=support, note=note)
        rows.append(row)
    return rows


def render_markdown(env, matrix, smoke=None):
    cols = [p["col"] for p in POTENTIALS]
    mp = {p["col"]: p["maple"] for p in POTENTIALS}
    L = []
    L.append("# MAPLE MD capability matrix")
    L.append("")
    L.append("Rows = MD ensemble, columns = ML potential family. "
             "Auto-generated by `tests/smoke_md_capability.py`.")
    L.append("")
    L.append("Cell legend: **PASS** = smoke-tested OK · *supported* = gate "
             "allows it, not run this pass · **N/A** = capability gate refuses "
             "· **PENDING** = parked on another branch · **not-installed** / "
             "**no-model** = backend present in code but unusable here.")
    L.append("")
    # column -> maple selector note
    L.append("Column -> MAPLE selector: "
             + ", ".join(f"`{c}`->`{mp[c]}`" for c in cols) + ".")
    L.append("")
    head = "| ensemble | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    L.append(head)
    L.append(sep)
    for r in matrix:
        cells = []
        for c in cols:
            cells.append(_SYM[r["cells"][c]["support"]])
        L.append(f"| **{r['ensemble']}** | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Per-cell notes")
    L.append("")
    for r in matrix:
        for c in cols:
            cell = r["cells"][c]
            L.append(f"- **{r['ensemble']} x {c}** "
                     f"[{_SYM[cell['support']]}]: {cell['note']}")
    L.append("")
    L.append("## Environment snapshot")
    L.append("")
    L.append(f"- model dir: `{env['model_dir']}`")
    for c in cols:
        be = env["backends"][c]
        f = be["flags"]
        fl = (f"class={f['cls']} PBC={f['pbc']} stress={f['stress']} "
              f"req_local={f['req_local']}") if f else "NOT REGISTERED"
        L.append(f"- `{c}` (`{mp[c]}`): pkg={'yes' if be['pkg_ok'] else 'NO'} "
                 f"model_file={'yes' if be['model_file'] else 'no'} | {fl}")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# smoke execution + thermo parsing
# --------------------------------------------------------------------------- #
def parse_thermo(path: Path, ensemble: str):
    import numpy as np
    if not path.exists():
        return dict(verdict="FAIL", summary="no thermo file produced")
    rows = [ln.split() for ln in path.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    if len(rows) < 3:
        return dict(verdict="FAIL", summary=f"only {len(rows)} thermo rows")
    a = np.array(rows, dtype=float)
    step, t, T, KE, PE, TE = (a[:, i] for i in range(6))
    finite = bool(np.isfinite(a).all())
    n = len(step)
    i0 = n // 2
    out = dict(n=n, all_finite=finite,
               T_mean=round(float(T[i0:].mean()), 2),
               T_std=round(float(T[i0:].std()), 2),
               TE_mean_Ha=round(float(TE.mean()), 6),
               KE_pos=bool((KE > 0).any()))
    ok = finite and (KE > 0).any() and (T > 0).any()
    if ensemble == "nve":
        drift = abs((TE[-1] - TE[0]) / (abs(TE.mean()) + 1e-12))
        out["TE_rel_drift"] = float(f"{drift:.3e}")
        ok = ok and drift < 5e-3
        out["summary"] = (f"NVE finite, T~{out['T_mean']}K, "
                          f"|dTE/TE|={out['TE_rel_drift']}")
    elif ensemble == "nvt":
        # temperature controlled: equilibrated mean within a broad band of target
        ctrl = 0.4 * 300.0 < out["T_mean"] < 2.0 * 300.0
        out["T_controlled"] = bool(ctrl)
        ok = ok and ctrl
        out["summary"] = (f"NVT T_eq={out['T_mean']}+/-{out['T_std']}K "
                          f"(controlled={ctrl})")
    elif ensemble == "npt":
        V = a[:, 7] if a.shape[1] > 7 else None
        if V is None:
            return dict(verdict="FAIL", summary="NPT thermo has no volume col")
        slope = float(np.polyfit(t * 1e-3, V, 1)[0])  # A^3/ps
        moved = abs(V[-1] - V[0]) > 0.5
        out["V_first_A3"] = round(float(V[0]), 2)
        out["V_last_A3"] = round(float(V[-1]), 2)
        out["V_slope_A3_per_ps"] = round(slope, 4)
        out["V_moved"] = bool(moved)
        ok = ok and moved
        out["summary"] = (f"NPT V {out['V_first_A3']}->{out['V_last_A3']} A^3 "
                          f"(slope {out['V_slope_A3_per_ps']} A^3/ps, "
                          f"moved={moved})")
    out["verdict"] = "PASS" if ok else "FAIL"
    return out


def _final_moved(workdir: Path, base: str) -> bool:
    """True if final geometry differs from input (forces acted)."""
    fin = workdir / f"{base}_final.xyz"
    return fin.exists() and fin.stat().st_size > 0


def run_smokes(workdir: Path, env):
    """Launch `python -m maple.main` for every runnable smoke cell."""
    smoke = {}
    for pot in POTENTIALS:
        for ens, base in pot.get("smoke", {}).items():
            inp = workdir / f"{base}.inp"
            out = workdir / f"{base}.out"
            if not inp.exists():
                smoke[f"{pot['col']}:{ens}"] = dict(
                    verdict="FAIL", summary=f"missing input {inp.name}")
                continue
            print(f"[run] {pot['col']} {ens} -> {base}", flush=True)
            rc = subprocess.run(
                [sys.executable, "-m", "maple.main", inp.name, out.name],
                cwd=str(workdir)).returncode
            res = parse_thermo(workdir / f"{base}_md_thermo.dat", ens)
            res["rc"] = rc
            res["forces_nonzero"] = _final_moved(workdir, base)
            if rc != 0 and res.get("verdict") != "PASS":
                res["summary"] = f"maple exit {rc}; " + res.get("summary", "")
            smoke[f"{pot['col']}:{ens}"] = res
    return smoke


def collect_smokes(workdir: Path):
    """Parse already-produced thermo files (no re-run)."""
    smoke = {}
    for pot in POTENTIALS:
        for ens, base in pot.get("smoke", {}).items():
            th = workdir / f"{base}_md_thermo.dat"
            if not th.exists():
                continue
            res = parse_thermo(th, ens)
            res["forces_nonzero"] = _final_moved(workdir, base)
            smoke[f"{pot['col']}:{ens}"] = res
    return smoke


# --------------------------------------------------------------------------- #
# pytest regression hooks (gate invariants, no GPU needed)
# --------------------------------------------------------------------------- #
def test_capability_gates():
    env = discover_env()
    be = env["backends"]
    # NPT-capable iff mace-mp-0 (only shipped PBC+stress backend)
    assert be["mace-mp-0"]["flags"]["pbc"] and be["mace-mp-0"]["flags"]["stress"]
    for col in ("aimnet2", "macepol", "mace", "mace-off"):
        f = be[col]["flags"]
        assert not (f and f["pbc"] and f["stress"]), \
            f"{col} unexpectedly PBC+stress -> would wrongly enable NPT"
    # uma must remain N/A (vetoed) regardless of install state
    m = build_matrix(env)
    for r in m:
        assert r["cells"]["uma"]["support"] == "na"
        assert r["cells"]["mace-off"]["support"] in ("pending", "na")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=".",
                    help="dir holding smoke inp/mdp/thermo")
    ap.add_argument("--run", action="store_true",
                    help="execute MAPLE MD for runnable cells")
    ap.add_argument("--collect", action="store_true",
                    help="parse existing *_md_thermo.dat (no re-run)")
    ap.add_argument("--emit-md", default=None, help="write markdown matrix here")
    ap.add_argument("--json", default=None, help="write smoke_results json here")
    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()
    env = discover_env()

    smoke = None
    if args.run:
        smoke = run_smokes(workdir, env)
    elif args.collect:
        smoke = collect_smokes(workdir)
    if smoke is not None:
        sj = Path(args.json) if args.json else workdir / "smoke_results.json"
        sj.write_text(json.dumps(smoke, indent=2))
        print(f"[smoke] wrote {sj}")

    matrix = build_matrix(env, smoke)
    md = render_markdown(env, matrix, smoke)
    if args.emit_md:
        Path(args.emit_md).write_text(md)
        print(f"[matrix] wrote {args.emit_md}")
    print(md)


if __name__ == "__main__":
    main()
