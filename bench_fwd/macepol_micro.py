#!/usr/bin/env python
"""A3-fwd MACE-POL micro-benchmark: edge_mode radius vs full + CUDA-graph attempt
(opt campaign 2026-07-28; R3-3 verdict experiment).

Arms (macepols.pt, coupling_mode='approx', f32 traced model, gas-phase ts1x TS
geometries -- 'approx' is the ONLY batched MACE-POL path; its documented
cross-molecule coupling error is orthogonal to what is measured here, since ALL
arms share the identical single-graph packed forward):
  radius : per-forward radius filter (baseline oracle, 2 replicates)
  full   : constant full candidate edge set (edge_mode='full', static shapes)
  graph  : edge_mode='full' + manual torch.cuda.CUDAGraph capture of the
           (forward + autograd.grad) E/F computation, replayed with new coords.
           Capture failure (exception) is itself the R3-3 verdict evidence.

Gates:
  P1 radius self-spread (same arm, 2 independent passes) -- the canon threshold.
  P2 full vs radius E/F (must sit within ~3x self-spread for 'full' to be exact:
     the cutoff-envelope-zeroes-out-of-range-edges argument, VERIFIED not assumed).
  P3 graph-replay vs eager-full E/F at 5 jittered geometries.
  Edge counts: full vs radius kept (extra-compute ratio).

env: MODEL_DIR PKL OUT
"""
import json
import os
import pickle
import time
import traceback

import numpy as np
import torch
from ase import Atoms

MODEL_DIR = os.environ["MODEL_DIR"]          # dir containing macepols.pt
PKL = os.environ["PKL"]
OUT = os.environ["OUT"]
DEV = "cuda"
B_SWEEP = [1, 8, 16, 32, 64, 128]
T_TIMED = 30
W_WARM = 3

from maple.function.calculator.mace._macepol_batch_calculator import (  # noqa: E402
    MACEPolBatchCalc,
)

RES = {"bench": "macepol_micro", "model": os.path.join(MODEL_DIR, "macepols.pt"),
       "pkl": PKL, "gpu": torch.cuda.get_device_name(0),
       "B_sweep": B_SWEEP, "T_timed": T_TIMED, "arms": {}}


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
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MACEPolBatchCalc(device=DEV, model="macepols",
                                model_path=os.path.join(MODEL_DIR, "macepols.pt"),
                                coupling_mode="approx", edge_mode=edge_mode)


def chunk_atoms(B):
    src = ATOMS * ((B // len(ATOMS)) + 1) if B > len(ATOMS) else ATOMS
    return [a.copy() for a in src[:B]]


def timed(calc, B, reps=2):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc.prepare(chunk_atoms(B))
    g = torch.Generator(device="cpu").manual_seed(77 + B)
    jit = 0.01 * torch.randn((B, calc.nmax_dof), generator=g).double().to(DEV)
    calc.get_ef_gpu(); torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        for _ in range(W_WARM):
            calc.step_cart_(jit); calc.get_ef_gpu()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(T_TIMED):
            calc.step_cart_(jit); calc.get_ef_gpu()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3 / T_TIMED)
    return out


def ef_now(calc):
    E, F = calc.get_ef_gpu()
    return E.detach().cpu().numpy(), F.detach().cpu().numpy()


def main():
    import warnings
    cr = make_calc("radius")
    cf = make_calc("full")

    # ---------------- timing sweep ----------------
    for arm, calc in (("radius", cr), ("full", cf)):
        RES["arms"][arm] = {"sweep": {}}
        for B in B_SWEEP:
            try:
                ms = timed(calc, B)
                RES["arms"][arm]["sweep"][str(B)] = {
                    "ms_per_forward": ms, "ms_per_structure": [x / B for x in ms]}
                print(f"[macepol] {arm} B={B}: {['%.2f' % x for x in ms]} ms/fwd",
                      flush=True)
            except Exception:
                RES["arms"][arm]["sweep"][str(B)] = {"error": traceback.format_exc()}
                print(f"[macepol] {arm} B={B}: FAILED", flush=True)
        json.dump(RES, open(OUT, "w"), indent=1)

    # ---------------- parity gates (B=16) ----------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cr.prepare(chunk_atoms(16))
        E0a, F0a = ef_now(cr)
        E0b, F0b = ef_now(cr)                       # same geometry, 2nd pass
        cr2 = make_calc("radius")                   # independent instance
        cr2.prepare(chunk_atoms(16))
        E0c, F0c = ef_now(cr2)
        spread_E = float(max(np.abs(E0a - E0b).max(), np.abs(E0a - E0c).max()))
        spread_F = float(max(np.abs(F0a - F0b).max(), np.abs(F0a - F0c).max()))
        cf.prepare(chunk_atoms(16))
        E1, F1 = ef_now(cf)
    dE = float(np.abs(E1 - E0a).max()); dF = float(np.abs(F1 - F0a).max())
    thr_E = max(3.0 * spread_E, 1e-8); thr_F = max(3.0 * spread_F, 1e-8)
    n_full = int(cf._full_ei.size(1))
    rij = cr.coord[cr.cand_i] - cr.coord[cr.cand_j]
    keep = ((rij * rij).sum(-1) <= (cr.r_max + 1e-12) ** 2)
    n_kept = 2 * int(keep.sum().item())
    RES["P_parity"] = {
        "radius_self_spread_maxdE_Ha": spread_E,
        "radius_self_spread_maxdF_HaA": spread_F,
        "full_vs_radius_maxdE_Ha": dE, "full_vs_radius_maxdF_HaA": dF,
        "thr_E": thr_E, "thr_F": thr_F,
        "pass": bool(dE <= thr_E and dF <= thr_F),
        "edges_full": n_full, "edges_radius_kept": n_kept,
        "edge_ratio": n_full / max(1, n_kept), "r_max_A": float(cr.r_max)}
    print(f"[P1/P2] spread dE={spread_E:.2e} dF={spread_F:.2e}; full-vs-radius "
          f"dE={dE:.2e} dF={dF:.2e} pass={RES['P_parity']['pass']} "
          f"edges {n_full}/{n_kept}", flush=True)
    json.dump(RES, open(OUT, "w"), indent=1)

    # ---------------- CUDA-graph attempt (R3-3) ----------------
    graph_res = {}
    for B in [16, 64]:
        r = {}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cf.prepare(chunk_atoms(B))
            coord0 = cf.coord.clone()

            static_in = coord0.detach().to(dtype=cf.mdtype).clone().requires_grad_(True)

            def _ef(inp):
                ei, sh, us = cf._build_edges(inp)
                out = cf.model(inp, cf.node_attrs, ei, sh, us,
                               cf._fwd_batch, cf._fwd_ptr, cf._fwd_cell,
                               cf._fwd_tc, cf._fwd_ts, cf._fwd_ext, cf._fwd_log)
                total, node_e = out[0], out[1]
                ne = node_e.reshape(-1)
                E = torch.zeros(cf._atoms_B, dtype=ne.dtype, device=DEV
                                ).index_add(0, cf.mol_idx, ne)
                g = torch.autograd.grad(total.sum(), inp)[0]
                return E, -g

            # eager reference at capture geometry
            E_ref, F_ref = _ef(static_in)
            torch.cuda.synchronize()

            # warmup on side stream (torch cuda-graphs protocol)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    _E, _F = _ef(static_in)
            torch.cuda.current_stream().wait_stream(s)

            gph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gph):
                static_E, static_F = _ef(static_in)
            r["capture"] = "OK"

            # parity over 5 jittered geometries: replay vs eager
            gges = torch.Generator(device="cpu").manual_seed(9 + B)
            maxdE = maxdF = 0.0
            for _ in range(5):
                newc = (coord0 + 0.02 * torch.randn(coord0.shape, generator=gges
                                                    ).double().to(DEV)).to(cf.mdtype)
                with torch.no_grad():
                    static_in.copy_(newc)
                gph.replay()
                torch.cuda.synchronize()
                Eg = static_E.detach().clone(); Fg = static_F.detach().clone()
                leaf = newc.detach().clone().requires_grad_(True)
                Ee, Fe = _ef(leaf)
                maxdE = max(maxdE, float((Eg - Ee).abs().max()))
                maxdF = max(maxdF, float((Fg - Fe).abs().max()))
            r["replay_vs_eager_maxdE_eV"] = maxdE
            r["replay_vs_eager_maxdF_eVA"] = maxdF

            # timing: eager-full vs graph replay
            jitc = 0.01 * torch.randn(coord0.shape, generator=gges).double().to(DEV)
            for _ in range(W_WARM):
                leaf = (coord0 + jitc).to(cf.mdtype).requires_grad_(True)
                _ef(leaf)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(T_TIMED):
                leaf = (coord0 + jitc).to(cf.mdtype).requires_grad_(True)
                _ef(leaf)
            torch.cuda.synchronize()
            eager_ms = (time.perf_counter() - t0) * 1e3 / T_TIMED
            with torch.no_grad():
                static_in.copy_((coord0 + jitc).to(cf.mdtype))
            for _ in range(W_WARM):
                gph.replay()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(T_TIMED):
                gph.replay()
            torch.cuda.synchronize()
            r["eager_full_ms"] = eager_ms
            r["graph_replay_ms"] = (time.perf_counter() - t0) * 1e3 / T_TIMED
        except Exception:
            r["capture"] = "FAILED"
            r["error"] = traceback.format_exc(limit=4)
        graph_res[str(B)] = r
        print(f"[cudagraph] B={B}: {json.dumps({k: v for k, v in r.items() if k != 'error'})}",
              flush=True)
        if r["capture"] == "FAILED":
            print(r["error"], flush=True)
    RES["cuda_graph"] = graph_res
    json.dump(RES, open(OUT, "w"), indent=1)
    print("MACEPOL_MICRO_OK", flush=True)


if __name__ == "__main__":
    main()
