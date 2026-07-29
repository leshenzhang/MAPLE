#!/usr/bin/env python
"""A3-fwd micro-benchmark + parity/edge gates (opt campaign 2026-07-28).

UMA batched raw forward:
  configs: base (fast_inference=False, the D-174/D-227 baseline)
           fi   (fast_inference=True: activation_checkpointing OFF, tf32 OFF,
                 merge_mole OFF -- parity-safe subset)
           fic  (fi + compile_model=True, torch.compile)
           tf32 (base forward under torch.backends allow_tf32=True -- POLICY probe,
                 numerics MAY change; reported, never a default)
  B sweep {1,8,16,32,64,128} on ts1x_seed42_100.pkl TS geometries, 2 in-job
  replicates per point, per-forward jitter to defeat caching.

Gates (all reported in the JSON):
  G1 parity: E/F of fi/fic/tf32 vs base, thresholded against the base SELF-SPREAD
     (two independent base instances, same geometry -- canon-vs-canon).
  G2 R3-1 functional staleness probe: displace one atom 2.5 A through the batched
     path, compare vs a freshly-prepared calc at the same geometry.
  G3 R3-1 guard: a calc mutated to external_graph_gen/r_edges=True must RAISE at
     prepare() (the new fail-fast), and a raw predictor built with
     external_graph_gen=True must show the stale-edge wrongness the guard blocks.
  G4 edge reference: AtomicData(r_edges=True) edge set vs ase.neighborlist at
     radius 6.0 -- symmetric difference must be 0.

Also: torch.profiler breakdown of one base and one fi forward at B=32
(top kernels + host share), and a cuequivariance availability probe.

env: MODEL PKL OUT (json path)
"""
import json
import os
import pickle
import sys
import time
import traceback

import numpy as np
import torch
from ase import Atoms

MODEL = os.environ["MODEL"]
PKL = os.environ["PKL"]
OUT = os.environ["OUT"]
DEV = "cuda"
B_SWEEP = [1, 8, 16, 32, 64, 128]
T_TIMED = 20      # timed forwards per replicate
W_WARM = 3        # warmup forwards
HA = 27.211386245988

from maple.function.calculator.uma._uma_batch_calculator import UMABatchCalc  # noqa: E402

RES = {"bench": "fwd_micro_uma", "model": MODEL, "pkl": PKL,
       "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0),
       "B_sweep": B_SWEEP, "T_timed": T_TIMED, "configs": {}}


def load_atoms(n=100):
    recs = pickle.load(open(PKL, "rb"))[:n]
    out = []
    for r in recs:
        Z = np.asarray(r["atomic_numbers"])
        out.append(Atoms(numbers=Z,
                         positions=np.asarray(r["transition_state"]["positions"], float)))
    return out


ATOMS = load_atoms()


def make_calc(cfg):
    if cfg == "base" or cfg == "tf32":
        return UMABatchCalc(MODEL, device=DEV, dtype=torch.float64, task="omol",
                            fast_inference=False)
    if cfg == "fi":
        return UMABatchCalc(MODEL, device=DEV, dtype=torch.float64, task="omol",
                            fast_inference=True)
    if cfg == "fic":
        return UMABatchCalc(MODEL, device=DEV, dtype=torch.float64, task="omol",
                            fast_inference=True, compile_model=True)
    raise ValueError(cfg)


def chunk_atoms(B):
    src = ATOMS * ((B // len(ATOMS)) + 1) if B > len(ATOMS) else ATOMS
    return [a.copy() for a in src[:B]]


def timed_forwards(calc, B, tf32=False, reps=2):
    """Prepare once; per rep: W warmups + T timed forwards (jitter outside timing).
    Returns per-rep [ms/forward], first-call ms (compile/lazy-init cost)."""
    al = chunk_atoms(B)
    calc.prepare(al)
    g = torch.Generator(device="cpu").manual_seed(1234 + B)
    jit = 0.01 * torch.randn((B, calc.nmax_dof), generator=g).to(DEV)
    prev = torch.backends.cuda.matmul.allow_tf32
    prevc = torch.backends.cudnn.allow_tf32
    try:
        if tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        t0 = time.perf_counter()
        calc.get_ef_gpu()
        torch.cuda.synchronize()
        first_ms = (time.perf_counter() - t0) * 1e3
        out = []
        for _ in range(reps):
            for _ in range(W_WARM):
                calc.step_cart_(jit)
                calc.get_ef_gpu()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(T_TIMED):
                calc.step_cart_(jit)          # tiny; keeps geometry moving
                calc.get_ef_gpu()
            torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) * 1e3 / T_TIMED)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
        torch.backends.cudnn.allow_tf32 = prevc
    return out, first_ms


def ef_once(calc, al, tf32=False):
    calc.prepare([a.copy() for a in al])
    prev = torch.backends.cuda.matmul.allow_tf32
    prevc = torch.backends.cudnn.allow_tf32
    try:
        if tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        E, F = calc.get_ef_gpu()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
        torch.backends.cudnn.allow_tf32 = prevc
    return E.detach().cpu().numpy(), F.detach().cpu().numpy()


def main():
    # ---------------- timing sweep ----------------
    calcs = {}
    for cfg in ["base", "fi", "fic"]:
        try:
            calcs[cfg] = make_calc(cfg)
            iset = calcs[cfg]._predictor.inference_settings
            RES["configs"][cfg] = {
                "inference_settings": {k: getattr(iset, k, None) for k in
                                       ("tf32", "activation_checkpointing", "merge_mole",
                                        "compile", "external_graph_gen",
                                        "internal_graph_gen_version")},
                "r_edges": bool(calcs[cfg]._r_edges),
                "sweep": {}}
        except Exception:
            RES["configs"][cfg] = {"error": traceback.format_exc()}
    RES["configs"]["tf32"] = {"note": "base calc + torch.backends allow_tf32=True scoped",
                              "sweep": {}}

    for B in B_SWEEP:
        for cfg in ["base", "fi", "fic", "tf32"]:
            if "error" in RES["configs"].get(cfg, {}):
                continue
            calc = calcs["base"] if cfg == "tf32" else calcs[cfg]
            try:
                per_fwd, first = timed_forwards(calc, B, tf32=(cfg == "tf32"))
                RES["configs"][cfg]["sweep"][str(B)] = {
                    "ms_per_forward": per_fwd,
                    "ms_per_structure": [x / B for x in per_fwd],
                    "first_call_ms": first}
                print(f"[micro] {cfg} B={B}: {['%.2f' % x for x in per_fwd]} ms/fwd "
                      f"(first {first:.0f} ms)", flush=True)
            except Exception:
                RES["configs"][cfg]["sweep"][str(B)] = {"error": traceback.format_exc()}
                print(f"[micro] {cfg} B={B}: FAILED", flush=True)
        json.dump(RES, open(OUT, "w"), indent=1)   # checkpoint-as-you-go

    # ---------------- G1 parity (B=16, structures 0..15) ----------------
    al16 = ATOMS[:16]
    base2 = make_calc("base")                     # independent instance -> self-spread
    E0a, F0a = ef_once(calcs["base"], al16)
    E0b, F0b = ef_once(base2, al16)
    spread_E = float(np.abs(E0a - E0b).max())
    spread_F = float(np.abs(F0a - F0b).max())
    gate = {"base_self_spread_maxdE_Ha": spread_E,
            "base_self_spread_maxdF_HaA": spread_F}
    for cfg in ["fi", "fic", "tf32"]:
        try:
            if cfg == "tf32":
                E1, F1 = ef_once(base2, al16, tf32=True)
            else:
                if "error" in RES["configs"].get(cfg, {}):
                    continue
                E1, F1 = ef_once(calcs[cfg], al16)
            dE = float(np.abs(E1 - E0a).max())
            dF = float(np.abs(F1 - F0a).max())
            thr_E = max(3.0 * spread_E, 1e-8)
            thr_F = max(3.0 * spread_F, 1e-8)
            gate[cfg] = {"maxdE_Ha": dE, "maxdF_HaA": dF,
                         "thr_E": thr_E, "thr_F": thr_F,
                         "pass": bool(dE <= thr_E and dF <= thr_F)}
        except Exception:
            gate[cfg] = {"error": traceback.format_exc()}
    RES["G1_parity"] = gate
    print(f"[G1] spread dE={spread_E:.2e} dF={spread_F:.2e}; " +
          " ".join(f"{c}:{gate.get(c, {}).get('pass')}" for c in ("fi", "fic", "tf32")),
          flush=True)
    json.dump(RES, open(OUT, "w"), indent=1)

    # ---------------- G2 functional staleness probe (base + fi) ----------------
    g2 = {}
    for cfg in ["base", "fi"]:
        if "error" in RES["configs"].get(cfg, {}):
            continue
        calc = calcs[cfg]
        al4 = [a.copy() for a in ATOMS[:4]]
        calc.prepare(al4)
        s = torch.zeros((4, calc.nmax_dof), dtype=torch.float64, device=DEV)
        s[0, 0] = 2.5                                # move mol0 atom0 x by 2.5 A
        calc.step_cart_(s)
        E1, F1 = calc.get_ef_gpu()
        E1 = E1.detach().cpu().numpy(); F1 = F1.detach().cpu().numpy()
        al4d = [a.copy() for a in ATOMS[:4]]
        p = al4d[0].get_positions(); p[0, 0] += 2.5; al4d[0].set_positions(p)
        E2, F2 = ef_once(base2 if cfg == "base" else calc, al4d)
        g2[cfg] = {"maxdE_Ha": float(np.abs(E1 - E2).max()),
                   "maxdF_HaA": float(np.abs(F1 - F2).max())}
    RES["G2_staleness_probe"] = g2
    print(f"[G2] {g2}", flush=True)
    json.dump(RES, open(OUT, "w"), indent=1)

    # ---------------- G3 R3-1 guard + stale-edge evidence ----------------
    g3 = {}
    try:
        from functools import partial
        from fairchem.core.calculate.ase_calculator import AtomicData
        guard_calc = make_calc("base")
        guard_calc._r_edges = True
        guard_calc._a2g = partial(AtomicData.from_ase, task_name="omol",
                                  r_edges=True, r_data_keys=["spin", "charge"],
                                  max_neigh=300, radius=6.0)
        try:
            guard_calc.prepare([a.copy() for a in ATOMS[:2]])
            g3["guard_raises"] = False
        except NotImplementedError as exc:
            g3["guard_raises"] = True
            g3["guard_msg"] = str(exc)[:200]
    except Exception:
        g3["guard_error"] = traceback.format_exc()
    # stale-edge WRONGNESS evidence: predictor with external_graph_gen=True
    try:
        from fairchem.core.units.mlip_unit import load_predict_unit
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
        from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
        from fairchem.core.calculate.ase_calculator import AtomicData
        from functools import partial
        iset = InferenceSettings(tf32=False, activation_checkpointing=False,
                                 merge_mole=False, compile=False,
                                 external_graph_gen=True,
                                 internal_graph_gen_version=2)
        pu = load_predict_unit(MODEL, inference_settings=iset, device="cuda")
        a2g = partial(AtomicData.from_ase, task_name="omol", r_edges=True,
                      r_data_keys=["spin", "charge"], max_neigh=300, radius=6.0)
        a0 = ATOMS[0].copy()
        a0.info["spin"] = 1; a0.info["charge"] = 0
        ad0 = atomicdata_list_to_batch([a2g(a0)]).to("cuda")
        pu.predict(ad0)                                  # lazy init
        ad = atomicdata_list_to_batch([a2g(a0)]).to("cuda")
        E0 = float(pu.predict(ad)["energy"].detach().cpu())
        # displaced geometry: rotate a dihedral-ish large move (atom0 +2.5 A)
        ad_stale = atomicdata_list_to_batch([a2g(a0)]).to("cuda")
        pd = a0.get_positions(); pd[0, 0] += 2.5
        ad_stale.pos = torch.tensor(pd, dtype=torch.float32, device="cuda")
        E_stale = float(pu.predict(ad_stale)["energy"].detach().cpu())
        a0d = a0.copy(); a0d.set_positions(pd)
        ad_fresh = atomicdata_list_to_batch([a2g(a0d)]).to("cuda")
        E_fresh = float(pu.predict(ad_fresh)["energy"].detach().cpu())
        g3["stale_evidence"] = {
            "E0_eV": E0, "E_stale_eV": E_stale, "E_fresh_eV": E_fresh,
            "stale_minus_fresh_eV": E_stale - E_fresh,
            "note": "pos-overwrite on r_edges=True AtomicData vs fresh rebuild; "
                    "nonzero difference = the silent wrongness the guard blocks"}
        del pu
        torch.cuda.empty_cache()
    except Exception:
        g3["stale_evidence_error"] = traceback.format_exc()
    RES["G3_r31_guard"] = g3
    print(f"[G3] {json.dumps(g3)[:400]}", flush=True)
    json.dump(RES, open(OUT, "w"), indent=1)

    # ---------------- G4 edge reference vs ASE ----------------
    g4 = {}
    try:
        from functools import partial
        from fairchem.core.calculate.ase_calculator import AtomicData
        from ase.neighborlist import neighbor_list
        a2g = partial(AtomicData.from_ase, task_name="omol", r_edges=True,
                      r_data_keys=["spin", "charge"], max_neigh=300, radius=6.0)
        worst = 0
        for a in ATOMS[:10]:
            aa = a.copy(); aa.info["spin"] = 1; aa.info["charge"] = 0
            ad = a2g(aa)
            ei = ad.edge_index.cpu().numpy()
            fset = set(zip(ei[0].tolist(), ei[1].tolist()))
            i, j = neighbor_list("ij", aa, 6.0)
            aset = set(zip(i.tolist(), j.tolist()))
            sd = len(fset ^ aset)
            worst = max(worst, sd)
        g4 = {"n_structs": 10, "max_symdiff": worst, "pass": worst == 0}
    except Exception:
        g4 = {"error": traceback.format_exc()}
    RES["G4_edge_vs_ase"] = g4
    print(f"[G4] {g4}", flush=True)
    json.dump(RES, open(OUT, "w"), indent=1)

    # ---------------- profiler breakdown (B=32) ----------------
    prof = {}
    for cfg in ["base", "fi"]:
        if "error" in RES["configs"].get(cfg, {}):
            continue
        try:
            calc = calcs[cfg]
            calc.prepare(chunk_atoms(32))
            for _ in range(3):
                calc.get_ef_gpu()
            torch.cuda.synchronize()
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
                for _ in range(5):
                    calc.get_ef_gpu()
                torch.cuda.synchronize()
            prof[cfg] = pr.key_averages().table(sort_by="cuda_time_total", row_limit=15)
        except Exception:
            prof[cfg] = traceback.format_exc()
    RES["profiler_B32"] = prof
    for k, v in prof.items():
        print(f"\n===== profiler {k} (B=32) =====\n{v}\n", flush=True)

    # ---------------- cuEq availability probe ----------------
    cue = {}
    try:
        import cuequivariance  # noqa: F401
        cue["cuequivariance"] = "importable"
    except Exception as exc:
        cue["cuequivariance"] = f"MISSING: {exc!r}"
    cue["maceoff_ckpt"] = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
    cue["maceoff_ckpt_exists"] = os.path.exists(cue["maceoff_ckpt"])
    RES["cueq_probe"] = cue
    print(f"[cueq] {cue}", flush=True)

    json.dump(RES, open(OUT, "w"), indent=1)
    print("FWD_MICRO_OK", flush=True)


if __name__ == "__main__":
    main()
