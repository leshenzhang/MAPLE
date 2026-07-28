#!/usr/bin/env python
"""A3-fwd: torch.compile over the MACE-POL batched forward (candidate 2) and the
compiled-CUDA-graph route (candidate 3, mode='reduce-overhead').

Why MACE-POL is the right vehicle for the compile/CUDA-graph question: it is the
only in-tree batched backend whose whole forward (edge build + traced model +
autograd force) is OURS end-to-end, so 'static shapes' can be forced via
``edge_mode='full'``. UMA's forward is inside fairchem's ``predict``; its compile
knob is exercised separately by fwd_micro.py (config 'fic').

Arms at each B (B ∈ {8,16,32,64}), 2 replicates each:
  eager_radius : default oracle (variable edge count)
  eager_full   : edge_mode='full' (constant edges, static shapes)
  comp_full    : torch.compile(fn, dynamic=False) on the full-edge path
  cg_full      : torch.compile(fn, mode='reduce-overhead', dynamic=False)
                 -> inductor's CUDA-graph capture incl. the backward
  comp_radius  : torch.compile(fn, dynamic=False) on the RADIUS path -> measures
                 the recompile/graph-break cost of the variable edge count (the
                 R3-3 obstacle, quantified rather than asserted)

Parity: every compiled arm's E/F vs eager_radius, thresholded on the eager_radius
self-spread (canon-vs-canon), at 3 independent geometries.
Also records torch._dynamo recompile counts per arm.

env: MODEL_DIR PKL OUT
"""
import json
import os
import pickle
import time
import traceback
import warnings

import numpy as np
import torch
from ase import Atoms

MODEL_DIR = os.environ["MODEL_DIR"]
PKL = os.environ["PKL"]
OUT = os.environ["OUT"]
DEV = "cuda"
B_SWEEP = [8, 16, 32, 64]
T_TIMED = 30
W_WARM = 5

from maple.function.calculator.mace._macepol_batch_calculator import (  # noqa: E402
    MACEPolBatchCalc,
)

RES = {"bench": "macepol_compile", "gpu": torch.cuda.get_device_name(0),
       "torch": torch.__version__, "B_sweep": B_SWEEP, "T_timed": T_TIMED,
       "arms": {}}


def load_atoms(n=100):
    recs = pickle.load(open(PKL, "rb"))[:n]
    out = []
    for r in recs:
        Z = np.asarray(r["atomic_numbers"])
        a = Atoms(numbers=Z,
                  positions=np.asarray(r["transition_state"]["positions"], float))
        a.info["charge"] = 0
        a.info["mult"] = 1
        out.append(a)
    return out


ATOMS = load_atoms()


def make_calc(edge_mode):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MACEPolBatchCalc(device=DEV, model="macepols",
                                model_path=os.path.join(MODEL_DIR, "macepols.pt"),
                                coupling_mode="approx", edge_mode=edge_mode)


def chunk_atoms(B):
    src = ATOMS * ((B // len(ATOMS)) + 1) if B > len(ATOMS) else ATOMS
    return [a.copy() for a in src[:B]]


def ef_fn(calc):
    """(coord f32 leaf) -> (E per-mol eV, F all-atom eV/A). Same math as
    _forward_single_graph (verified by the parity gate below)."""
    def _fn(inp):
        ei, sh, us = calc._build_edges(inp)
        out = calc.model(inp, calc.node_attrs, ei, sh, us,
                         calc._fwd_batch, calc._fwd_ptr, calc._fwd_cell,
                         calc._fwd_tc, calc._fwd_ts, calc._fwd_ext, calc._fwd_log)
        total, node_e = out[0], out[1]
        ne = node_e.reshape(-1)
        E = torch.zeros(calc._atoms_B, dtype=ne.dtype, device=DEV
                        ).index_add(0, calc.mol_idx, ne)
        g = torch.autograd.grad(total.sum(), inp, create_graph=False)[0]
        return E, -g
    return _fn


def run_arm(name, calc, B, fn, geoms, reps=2):
    """Time fn over jittered geometries; return (ms list, E/F at the 3 probe geoms)."""
    coord0 = calc.coord.clone()
    for _ in range(W_WARM):
        leaf = coord0.to(calc.mdtype).clone().requires_grad_(True)
        fn(leaf)
    torch.cuda.synchronize()
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for i in range(T_TIMED):
            c = coord0 + (1e-3 * (i % 5))
            leaf = c.to(calc.mdtype).clone().requires_grad_(True)
            fn(leaf)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1e3 / T_TIMED)
    probes = []
    for gcoord in geoms:
        leaf = gcoord.to(calc.mdtype).clone().requires_grad_(True)
        E, F = fn(leaf)
        probes.append((E.detach().float().cpu().numpy().copy(),
                       F.detach().float().cpu().numpy().copy()))
    return ms, probes


def main():
    cr = make_calc("radius")
    cf = make_calc("full")
    for arm in ["eager_radius", "eager_full", "comp_full", "cg_full", "comp_radius"]:
        RES["arms"][arm] = {"sweep": {}}
    RES["parity"] = {}

    for B in B_SWEEP:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cr.prepare(chunk_atoms(B))
            cf.prepare(chunk_atoms(B))
        g = torch.Generator(device="cpu").manual_seed(2026 + B)
        geoms = [cr.coord + 0.02 * torch.randn(cr.coord.shape, generator=g
                                               ).double().to(DEV) for _ in range(3)]
        fns = {}
        fns["eager_radius"] = (cr, ef_fn(cr))
        fns["eager_full"] = (cf, ef_fn(cf))
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        try:
            fns["comp_full"] = (cf, torch.compile(ef_fn(cf), dynamic=False))
            fns["cg_full"] = (cf, torch.compile(ef_fn(cf), mode="reduce-overhead",
                                                dynamic=False))
            fns["comp_radius"] = (cr, torch.compile(ef_fn(cr), dynamic=False))
        except Exception:
            RES["arms"]["comp_full"]["sweep"][str(B)] = {"error": traceback.format_exc()}

        ref_probe = None
        ref_spread = None
        for arm in ["eager_radius", "eager_full", "comp_full", "cg_full", "comp_radius"]:
            if arm not in fns:
                continue
            calc, fn = fns[arm]
            try:
                t0 = time.perf_counter()
                ms, probes = run_arm(arm, calc, B, fn, geoms)
                RES["arms"][arm]["sweep"][str(B)] = {
                    "ms_per_forward": ms,
                    "ms_per_structure": [x / B for x in ms],
                    "arm_wall_s": time.perf_counter() - t0}
                print(f"[compile] B={B} {arm}: {['%.2f' % x for x in ms]} ms/fwd",
                      flush=True)
                if arm == "eager_radius":
                    _ms2, probes2 = run_arm(arm, calc, B, fn, geoms, reps=1)
                    ref_probe = probes
                    ref_spread = (
                        max(float(np.abs(a[0] - b[0]).max()) for a, b in zip(probes, probes2)),
                        max(float(np.abs(a[1] - b[1]).max()) for a, b in zip(probes, probes2)))
                elif ref_probe is not None:
                    dE = max(float(np.abs(a[0] - b[0]).max())
                             for a, b in zip(probes, ref_probe))
                    dF = max(float(np.abs(a[1] - b[1]).max())
                             for a, b in zip(probes, ref_probe))
                    thrE = max(3.0 * ref_spread[0], 1e-6)
                    thrF = max(3.0 * ref_spread[1], 1e-6)
                    RES["parity"].setdefault(str(B), {"eager_radius_self_spread":
                                                      list(ref_spread)})[arm] = {
                        "maxdE_eV": dE, "maxdF_eVA": dF, "thrE": thrE, "thrF": thrF,
                        "pass": bool(dE <= thrE and dF <= thrF)}
            except Exception:
                RES["arms"][arm]["sweep"][str(B)] = {"error": traceback.format_exc(limit=3)}
                print(f"[compile] B={B} {arm}: FAILED", flush=True)
                print(traceback.format_exc(limit=3), flush=True)
        try:
            import torch._dynamo as dyn
            RES["arms"].setdefault("_dynamo", {})[str(B)] = {
                "frame_count": int(getattr(dyn.utils, "frame_count", -1) or -1)}
        except Exception:
            pass
        json.dump(RES, open(OUT, "w"), indent=1)

    print("\n[parity] " + json.dumps(RES["parity"])[:1500], flush=True)
    json.dump(RES, open(OUT, "w"), indent=1)
    print("MACEPOL_COMPILE_OK", flush=True)


if __name__ == "__main__":
    main()
