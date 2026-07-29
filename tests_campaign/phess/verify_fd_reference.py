#!/usr/bin/env python
"""Is the shipped 'numerical' Hessian really a finite difference of forces?

Motivated by the counter self-test failure reported 2026-07-28: a hook on
``calc._forward`` measures only 2 gradient-equivalents for a B=2 numerical
Hessian, i.e. the Hessian work is INVISIBLE to that hook. Two candidate causes:
  (a) the FD replica forwards do not go through the hooked entry point;
  (b) 'numerical' silently runs the (known-broken for UMA eSCN-MoE)
      double-backward, in which case it is not a valid accuracy reference.

This decides it WITHOUT trusting any mode name:
  1. H_ship  = calc.get_efh_gpu()                       (default mode)
  2. H_mine  = central difference built HERE, by displacing coordinates and
               calling get_ef_gpu() in an explicit python loop (6n forwards)
  3. H_auto  = calc.get_efh_gpu(mode='autograd')        (explicit double backward)
  4. counters: how many times _forward vs _predict_forces is entered, and how
     many replica-atoms the Hessian path actually pushes through the model.

Expected if (a): H_ship == H_mine to FD round-off, H_auto differs by ~1e-2
(D-56), _forward count == 1 while _predict_forces count == n_chunks >> 0.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from ase import Atoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--delta", type=float, default=2e-3)
    args = ap.parse_args()
    sys.path.insert(0, args.fork)
    from maple.function.calculator.uma._uma_batch_calculator import UMABatchCalc

    import pickle
    d = pickle.load(open(args.pkl, "rb"))[args.case]
    Z = np.asarray(d["atomic_numbers"], dtype=int)
    pos = np.asarray(d["transition_state"]["positions"], dtype=float)
    n, dof = len(Z), 3 * len(Z)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[verify] case natoms={n} dof={dof} delta={args.delta} dev={dev}", flush=True)

    calc = UMABatchCalc(args.model, device=dev, dtype=torch.float64, task="omol",
                        fd_mode="central", hessian_delta=args.delta)

    # ---- instrument BOTH entry points -------------------------------------
    cnt = dict(_forward=0, _predict_forces=0, replica_atoms=0)
    raw_fwd = calc._forward
    raw_pred = calc._predict_forces

    def fwd(*a, **k):
        cnt["_forward"] += 1
        return raw_fwd(*a, **k)

    def pred(batch_ad, *a, **k):
        cnt["_predict_forces"] += 1
        try:
            cnt["replica_atoms"] += int(batch_ad.pos.shape[0])
        except Exception:
            pass
        return raw_pred(batch_ad, *a, **k)

    calc._forward = fwd
    calc._predict_forces = pred

    at = Atoms(numbers=Z, positions=pos)
    calc.prepare([at])
    t0 = time.time()
    E, F, H, P = calc.get_efh_gpu()
    t_ship = time.time() - t0
    H_ship = H[0, :dof, :dof].detach().cpu().numpy().astype(float)
    H_ship = 0.5 * (H_ship + H_ship.T)
    c_ship = dict(cnt)
    print(f"[verify] shipped get_efh_gpu: {t_ship:.2f}s  counters={c_ship}", flush=True)
    print(f"[verify]   -> _forward sees {c_ship['_forward']} call(s) "
          f"(= what a _forward-based GE hook would report, times B)", flush=True)
    print(f"[verify]   -> _predict_forces sees {c_ship['_predict_forces']} call(s), "
          f"{c_ship['replica_atoms']} replica-atoms "
          f"(expected ~ 6*n*n = {6 * n * n} for central FD)", flush=True)

    # ---- 2. hand-rolled central FD, explicit displacement loop -------------
    calc._forward = raw_fwd
    calc._predict_forces = raw_pred
    dl = args.delta
    Hm = np.zeros((dof, dof))
    t0 = time.time()
    for a in range(n):
        for c in range(3):
            row = 3 * a + c
            Fpm = []
            for s in (+1, -1):
                p = pos.copy()
                p[a, c] += s * dl
                calc.prepare([Atoms(numbers=Z, positions=p)])
                _, Fx = calc.get_ef_gpu()
                Fpm.append(Fx[0, :dof].detach().cpu().numpy().astype(float))
            Hm[row, :] = -(Fpm[0] - Fpm[1]) / (2.0 * dl)
    t_mine = time.time() - t0
    Hm = 0.5 * (Hm + Hm.T)
    print(f"[verify] hand-rolled central FD ({6 * n} single forwards): {t_mine:.1f}s",
          flush=True)

    # ---- 3. explicit autograd (double backward) ---------------------------
    H_auto = None
    try:
        calc.prepare([at])
        _, _, Ha, _ = calc.get_efh_gpu(mode="autograd")
        H_auto = Ha[0, :dof, :dof].detach().cpu().numpy().astype(float)
        H_auto = 0.5 * (H_auto + H_auto.T)
    except Exception as e:
        print(f"[verify] autograd mode raised: {e!r}", flush=True)

    def cmp(A, B):
        return dict(max_abs=float(np.abs(A - B).max()),
                    fro_rel=float(np.linalg.norm(A - B) / max(np.linalg.norm(B), 1e-30)),
                    d_lam0=float(np.linalg.eigvalsh(A)[0] - np.linalg.eigvalsh(B)[0]))

    res = dict(natoms=n, delta=dl, counters_shipped=c_ship,
               wall_shipped=t_ship, wall_handrolled=t_mine,
               ship_vs_mine=cmp(H_ship, Hm))
    print(f"\n[verify] SHIPPED vs HAND-ROLLED FD: {res['ship_vs_mine']}", flush=True)
    if H_auto is not None:
        res["auto_vs_mine"] = cmp(H_auto, Hm)
        res["auto_vs_ship"] = cmp(H_auto, H_ship)
        print(f"[verify] AUTOGRAD vs HAND-ROLLED FD: {res['auto_vs_mine']}", flush=True)
        print(f"[verify] AUTOGRAD vs SHIPPED:        {res['auto_vs_ship']}", flush=True)

    # ---- verdict ----------------------------------------------------------
    same = res["ship_vs_mine"]["max_abs"] < 1e-6
    res["verdict"] = ("shipped_is_true_FD" if same else "shipped_is_NOT_plain_FD")
    print(f"\n[verify] VERDICT: {res['verdict']} "
          f"(max|dH| ship-vs-mine = {res['ship_vs_mine']['max_abs']:.3e} Ha/A^2)",
          flush=True)
    print(f"[verify] GE-hook blind spot: _forward={c_ship['_forward']} vs "
          f"_predict_forces={c_ship['_predict_forces']} "
          f"-> a hook on _forward misses "
          f"{100 * (1 - c_ship['_forward'] / max(c_ship['_predict_forces'] + 1, 1)):.0f}% "
          f"of the Hessian model invocations", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[verify] wrote {args.out}", flush=True)
    if not same:
        print("VERIFY_FD_FAIL", flush=True)
        sys.exit(2)
    print("VERIFY_FD_DONE", flush=True)


if __name__ == "__main__":
    main()
