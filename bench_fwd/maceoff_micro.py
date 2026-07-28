#!/usr/bin/env python
"""A3-fwd: MACE (mace-package native multi-graph) forward micro-benchmark +
dtype/TF32 policy probe (candidate 6) on a REAL MACE checkpoint.

Backend: MaceOffBatchCalc (the runnable mace-package batched calc). Checkpoint =
MACE_CKPT (mace-mp L0 found on Ibex; MACE-OFF23 is absent from this sandbox).

Arms (B ∈ {1,8,16,32,64,128}, 2 replicates):
  fp64        -- default (MODEL_DTYPE float64), the oracle
  fp32        -- set_precision(float32, allow_tf32=False)
  fp32_tf32   -- set_precision(float32, allow_tf32=True)   [R3-6 scoped flag]

Gates:
  S  fp64 self-spread (two independent instances, same geometry) -> canon threshold
  D1 fp32   vs fp64 : |dE|, |dF| (POLICY number, NOT expected to pass a parity gate)
  D2 fp32_tf32 vs fp32 : the marginal TF32 cost on top of fp32
  E1 non-periodic edge set vs ase.neighborlist(strict <) symdiff == 0
  E2 R3-6 check: the process-global allow_tf32 flag is RESTORED after a scoped
     fp32_tf32 forward (no leak into other calculators)

env: MACE_CKPT PKL OUT
"""
import json
import os
import pickle
import time
import traceback

import numpy as np
import torch
from ase import Atoms

CKPT = os.environ["MACE_CKPT"]
PKL = os.environ["PKL"]
OUT = os.environ["OUT"]
DEV = "cuda"
B_SWEEP = [1, 8, 16, 32, 64, 128]
T_TIMED = 20
W_WARM = 3

from maple.function.calculator.mace._maceoff_batch_calculator import (  # noqa: E402
    MaceOffBatchCalc,
)

RES = {"bench": "maceoff_micro", "ckpt": CKPT, "gpu": torch.cuda.get_device_name(0),
       "B_sweep": B_SWEEP, "T_timed": T_TIMED, "arms": {}}


def load_atoms(n=100):
    recs = pickle.load(open(PKL, "rb"))[:n]
    return [Atoms(numbers=np.asarray(r["atomic_numbers"]),
                  positions=np.asarray(r["transition_state"]["positions"], float))
            for r in recs]


ATOMS = load_atoms()


def chunk_atoms(B):
    src = ATOMS * ((B // len(ATOMS)) + 1) if B > len(ATOMS) else ATOMS
    return [a.copy() for a in src[:B]]


def make(arm):
    c = MaceOffBatchCalc(model_path=CKPT, device=DEV, dtype=torch.float64)
    if arm == "fp32":
        c.set_precision(torch.float32, allow_tf32=False)
    elif arm == "fp32_tf32":
        c.set_precision(torch.float32, allow_tf32=True)
    return c


def timed(calc, B, reps=2):
    calc.prepare(chunk_atoms(B))
    g = torch.Generator(device="cpu").manual_seed(31 + B)
    jit = (0.01 * torch.randn((B, calc.nmax_dof), generator=g)).to(DEV, calc.dtype)
    calc.get_ef_gpu(); torch.cuda.synchronize()
    ms = []
    for _ in range(reps):
        for _ in range(W_WARM):
            calc.step_cart_(jit); calc.get_ef_gpu()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(T_TIMED):
            calc.step_cart_(jit); calc.get_ef_gpu()
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1e3 / T_TIMED)
    return ms


def ef(calc, al):
    calc.prepare([a.copy() for a in al])
    E, F = calc.get_ef_gpu()
    return (E.detach().double().cpu().numpy(), F.detach().double().cpu().numpy())


def main():
    calcs = {}
    for arm in ["fp64", "fp32", "fp32_tf32"]:
        try:
            calcs[arm] = make(arm)
            RES["arms"][arm] = {"sweep": {}, "dtype": str(calcs[arm].dtype),
                                "allow_tf32": bool(calcs[arm].allow_tf32),
                                "r_max": float(calcs[arm].r_max)}
        except Exception:
            RES["arms"][arm] = {"error": traceback.format_exc(limit=3)}
    for B in B_SWEEP:
        for arm, c in calcs.items():
            try:
                ms = timed(c, B)
                RES["arms"][arm]["sweep"][str(B)] = {
                    "ms_per_forward": ms, "ms_per_structure": [x / B for x in ms]}
                print(f"[maceoff] {arm} B={B}: {['%.2f' % x for x in ms]} ms/fwd",
                      flush=True)
            except Exception:
                RES["arms"][arm]["sweep"][str(B)] = {"error": traceback.format_exc(limit=3)}
                print(f"[maceoff] {arm} B={B}: FAILED", flush=True)
        json.dump(RES, open(OUT, "w"), indent=1)

    # ---- dtype/TF32 policy numbers (B=16) ----
    al = ATOMS[:16]
    try:
        E64a, F64a = ef(calcs["fp64"], al)
        c64b = make("fp64")
        E64b, F64b = ef(c64b, al)
        sE = float(np.abs(E64a - E64b).max()); sF = float(np.abs(F64a - F64b).max())
        pol = {"fp64_self_spread_maxdE_Ha": sE, "fp64_self_spread_maxdF_HaA": sF}
        E32, F32 = ef(calcs["fp32"], al)
        pol["fp32_vs_fp64"] = {"maxdE_Ha": float(np.abs(E32 - E64a).max()),
                               "maxdF_HaA": float(np.abs(F32 - F64a).max())}
        Et, Ft = ef(calcs["fp32_tf32"], al)
        pol["fp32tf32_vs_fp64"] = {"maxdE_Ha": float(np.abs(Et - E64a).max()),
                                   "maxdF_HaA": float(np.abs(Ft - F64a).max())}
        pol["fp32tf32_vs_fp32"] = {"maxdE_Ha": float(np.abs(Et - E32).max()),
                                   "maxdF_HaA": float(np.abs(Ft - F32).max())}
        RES["dtype_policy"] = pol
        print(f"[policy] {json.dumps(pol)}", flush=True)
    except Exception:
        RES["dtype_policy"] = {"error": traceback.format_exc(limit=3)}

    # ---- E2: R3-6 global TF32 flag restored after a scoped forward ----
    try:
        before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        calcs["fp32_tf32"].prepare(chunk_atoms(4))
        calcs["fp32_tf32"].get_ef_gpu()
        after = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        RES["E2_tf32_flag_restored"] = {"before": list(before), "after": list(after),
                                        "pass": before == after}
        print(f"[E2] tf32 flag before={before} after={after}", flush=True)
    except Exception:
        RES["E2_tf32_flag_restored"] = {"error": traceback.format_exc(limit=3)}

    # ---- E1: non-periodic edge set vs ASE ----
    try:
        from ase.neighborlist import neighbor_list
        c = calcs["fp64"]
        worst = 0
        for a in ATOMS[:10]:
            c.prepare([a.copy()])
            ei, _sh, _us = c._build_edges_gpu(c.coord)
            e = ei.cpu().numpy()
            mine = set(zip(e[0].tolist(), e[1].tolist()))
            i, j = neighbor_list("ij", a, c.r_max)
            ref = set(zip(i.tolist(), j.tolist()))
            worst = max(worst, len(mine ^ ref))
        RES["E1_edge_vs_ase"] = {"n_structs": 10, "max_symdiff": worst,
                                 "pass": worst == 0, "r_max": float(c.r_max)}
        print(f"[E1] {RES['E1_edge_vs_ase']}", flush=True)
    except Exception:
        RES["E1_edge_vs_ase"] = {"error": traceback.format_exc(limit=3)}

    json.dump(RES, open(OUT, "w"), indent=1)
    print("MACEOFF_MICRO_OK", flush=True)


if __name__ == "__main__":
    main()
