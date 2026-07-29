#!/usr/bin/env python
"""A2-phess end-to-end campaign: batched P-RFO saddle search on the N=100 ts1x
production set under different partial/approximate-Hessian policies.

Reports, per config and replicate:
  wall_s                 total BatchPRFO.run wall time
  ge_total               gradient-equivalents (1 GE = one per-structure force
                         evaluation), split into ge_iter / ge_hess / ge_hvp
  s_per_ge               the normalized main metric (wall / GE)
  converged / evicted / max_iter
  per-structure final geometry, TS energy, lowest mass-weighted eigenvalue and
  n_imag from an EXACT central-FD Hessian  -> the same-saddle gate

The start geometries are a fixed-seed Gaussian perturbation of the reference TS,
so every config sees byte-identical inputs.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from ase import Atoms

NEG_CUT = 1e-3


def load_cases(pkl, k):
    import pickle
    data = pickle.load(open(pkl, "rb"))
    out = []
    for d in data[:k]:
        out.append(dict(idx=int(d.get("index", -1)),
                        Z=np.asarray(d["atomic_numbers"], dtype=int),
                        ts=np.asarray(d["transition_state"]["positions"], dtype=float)))
    return out


def read_traj_last(path):
    if not os.path.exists(path):
        return None
    lines = open(path).read().split("\n")
    frames, i = [], 0
    while i < len(lines):
        if lines[i].strip() == "":
            i += 1
            continue
        try:
            nat = int(lines[i].strip())
        except ValueError:
            break
        if i + 2 + nat > len(lines):
            break
        frames.append(np.asarray(
            [[float(x) for x in lines[i + 2 + k].split()[1:4]] for k in range(nat)]))
        i += 2 + nat
    return frames[-1] if frames else None


def parse_log(path):
    txt = open(path).read()
    out = {}
    m = re.search(r"# grad_equiv: ge_iter=(\d+) ge_hess=(\d+) ge_hvp=(\d+) ge_total=(\d+)", txt)
    if m:
        out.update(ge_iter=int(m.group(1)), ge_hess=int(m.group(2)),
                   ge_hvp=int(m.group(3)), ge_total=int(m.group(4)))
    m = re.search(r"# Final status: converged=(\d+) evicted_straggler=(\d+) max_iter=(\d+)", txt)
    if m:
        out.update(n_converged=int(m.group(1)), n_evicted=int(m.group(2)),
                   n_maxiter=int(m.group(3)))
    m = re.search(r"# forward_reuse: reuse_forward=\S+ fwd_calls=(\d+) fwd_reused=(\d+)", txt)
    if m:
        out.update(fwd_calls=int(m.group(1)), fwd_reused=int(m.group(2)))
    m = re.search(r"# iterative: hvp_kind=(\S+) hvp_calls=(\d+) lanczos_calls=(\d+) "
                  r"lanczos_steps=(\d+) unconverged_ritz=(\d+)", txt)
    if m:
        out.update(hvp_kind=m.group(1), hvp_calls=int(m.group(2)),
                   lanczos_calls=int(m.group(3)), lanczos_steps=int(m.group(4)),
                   unconverged_ritz=int(m.group(5)))
    # per-structure convergence step
    step, cyc = {}, 0
    for ln in txt.split("\n"):
        mc = re.search(r"RS-PRFO Cycle\s+(\d+)", ln)
        if mc:
            cyc = int(mc.group(1))
            continue
        mm = re.match(r"^\[\s*(\d+)\]", ln)
        if mm and ln.strip().endswith("YES"):
            b = int(mm.group(1))
            step.setdefault(b, cyc)
    out["conv_step"] = step
    return out


def saddle_metrics(calc, Zs, geoms, chunk=25):
    """Exact central-FD Hessian at each final geometry -> (lam0_mw, n_imag, E)."""
    res = []
    for s in range(0, len(geoms), chunk):
        sl = slice(s, min(s + chunk, len(geoms)))
        al = [Atoms(numbers=Zs[i], positions=geoms[i])
              for i in range(sl.start, sl.stop) if geoms[i] is not None]
        idx = [i for i in range(sl.start, sl.stop) if geoms[i] is not None]
        if not al:
            continue
        calc.prepare(al)
        E, F, H, P = calc.get_efh_gpu()
        for j, i in enumerate(idx):
            n = 3 * len(al[j])
            Hb = H[j, :n, :n].detach().cpu().numpy().astype(float)
            Hb = 0.5 * (Hb + Hb.T)
            D = 1.0 / np.sqrt(np.repeat(al[j].get_masses(), 3))
            Hm = Hb * np.outer(D, D)
            ev = np.linalg.eigvalsh(0.5 * (Hm + Hm.T))
            res.append(dict(i=i, lam0=float(ev[0]), nimag=int((ev < -NEG_CUT).sum()),
                            E=float(E[j].item()),
                            fmax=float(np.abs(F[j, :n].detach().cpu().numpy()).max())))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--max-iter", type=int, default=120)
    ap.add_argument("--recalc", type=int, default=8)
    ap.add_argument("--hessian-update", default="bofill")
    ap.add_argument("--hessian-mode", default="full")
    ap.add_argument("--fd-mode", default="central")
    ap.add_argument("--warm-start", action="store_true")
    ap.add_argument("--lanczos-m", type=int, default=8)
    ap.add_argument("--iter-solver", default="lanczos", choices=["lanczos", "lobpcg"])
    ap.add_argument("--lobpcg-max", type=int, default=6)
    ap.add_argument("--precond", default="model", choices=["model", "none"])
    ap.add_argument("--reorth", action="store_true")
    ap.add_argument("--adapt", action="store_true")
    ap.add_argument("--quality", action="store_true")
    ap.add_argument("--quality-tol", type=float, default=0.5)
    ap.add_argument("--initial-hessian", default="identity")
    ap.add_argument("--ts-inject", action="store_true")
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--skip-saddle", action="store_true")
    # integrated-stack knobs (A3 forward accel + A1 streaming pool)
    ap.add_argument("--fast-inference", action="store_true")
    ap.add_argument("--pool", type=int, default=0, help="0=off; >0 = pool target batch")
    ap.add_argument("--refill-min", type=int, default=1)
    ap.add_argument("--no-refill-partial", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.fork)
    from maple.function.calculator.uma._uma_batch_calculator import UMABatchCalc
    from maple.function.dispatcher.ts.algorithm import BatchPRFO
    from maple.function.utility.molecules import Molecules

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cases = load_cases(args.pkl, args.n)
    rng = np.random.RandomState(args.seed)
    starts = [c["ts"] + rng.randn(*c["ts"].shape) * args.sigma for c in cases]
    print(f"[prod] tag={args.tag} rep={args.replicate} dev={dev} N={len(cases)} "
          f"recalc={args.recalc} upd={args.hessian_update} mode={args.hessian_mode} "
          f"fd={args.fd_mode} warm={args.warm_start}", flush=True)

    calc = UMABatchCalc(args.model, device=dev, dtype=torch.float64, task="omol",
                        fd_mode=args.fd_mode, hessian_delta=2e-3,
                        fast_inference=args.fast_inference)
    odir = os.path.join(args.out_dir, f"{args.tag}_r{args.replicate}")
    os.makedirs(odir, exist_ok=True)
    atoms = [Atoms(numbers=c["Z"], positions=starts[b].copy()) for b, c in enumerate(cases)]
    pool_queue = None
    if args.pool > 0:
        # streaming pool: keep the first --pool structures active, queue the rest.
        # The largest molecule must sit in the first batch (fixed nmax padding
        # silently drops 3n > nmax queue entries -- see OPT_A1 report section 3).
        order = sorted(range(len(atoms)), key=lambda i: -len(atoms[i]))
        atoms = [atoms[i] for i in order]
        cases = [cases[i] for i in order]
        pool_queue, atoms = atoms[args.pool:], atoms[:args.pool]
    mols = Molecules(atoms)
    mols.calc = calc

    kw = dict(output=os.path.join(odir, "run.out"), device=dev,
              max_outer_iter=args.max_iter, recalc=args.recalc,
              hessian_update=args.hessian_update, hessian_mode=args.hessian_mode,
              initial_hessian=args.initial_hessian,
              ts_hessian_inject=args.ts_inject,
              hessian_recalc_adapt=args.adapt,
              hessian_recalc_quality=args.quality,
              recalc_quality_tol=args.quality_tol,
              refill_min=args.refill_min,
              refill_partial_hessian=(not args.no_refill_partial))
    if args.hessian_mode == "iterative":
        kw.update(iter_lanczos_m=args.lanczos_m, iter_warm_start=args.warm_start,
                  iter_solver=args.iter_solver, iter_lobpcg_max=args.lobpcg_max,
                  iter_precond=args.precond, iter_reorth=args.reorth)
    if args.pool > 0:
        kw.update(pool_queue=pool_queue, B_target=args.pool)
    bp = BatchPRFO(**kw)

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    bp.run(mols)
    if dev == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else 0.0
    print(f"[prod] run {wall:.1f}s peakMB={peak:.0f}", flush=True)

    log = parse_log(os.path.join(odir, "run.out"))
    geoms = [read_traj_last(os.path.join(odir, f"ts_batch{b + 1}.xyz"))
             for b in range(len(cases))]

    sad = []
    t_sad = 0.0
    if not args.skip_saddle:
        # exact-Hessian same-saddle gate always uses CENTRAL FD (the oracle),
        # independent of the FD mode the run itself used.
        calc_ref = UMABatchCalc(args.model, device=dev, dtype=torch.float64,
                                task="omol", fd_mode="central", hessian_delta=2e-3)
        t0 = time.time()
        sad = saddle_metrics(calc_ref, [c["Z"] for c in cases], geoms)
        t_sad = time.time() - t0
        print(f"[prod] saddle gate {t_sad:.1f}s: "
              f"nimag==1: {sum(s['nimag'] == 1 for s in sad)}/{len(sad)}", flush=True)

    ge = log.get("ge_total", 0)
    res = dict(
        tag=args.tag, replicate=args.replicate, wall_s=wall, peak_MB=peak,
        s_per_ge=(wall / ge if ge else None), saddle_gate_s=t_sad,
        config=dict(n=args.n, seed=args.seed, sigma=args.sigma, max_iter=args.max_iter,
                    recalc=args.recalc, hessian_update=args.hessian_update,
                    hessian_mode=args.hessian_mode, fd_mode=args.fd_mode,
                    warm_start=args.warm_start, lanczos_m=args.lanczos_m,
                    iter_solver=args.iter_solver, lobpcg_max=args.lobpcg_max,
                    precond=args.precond, reorth=args.reorth,
                    adapt=args.adapt, quality=args.quality,
                    quality_tol=args.quality_tol,
                    initial_hessian=args.initial_hessian,
                    ts_inject=args.ts_inject),
        env=dict(device=dev, torch=torch.__version__,
                 gpu=(torch.cuda.get_device_name(0) if dev == "cuda" else "cpu")),
        log=log, saddle=sad,
        natoms=[int(len(c["Z"])) for c in cases],
        final_geoms=[(None if g is None else g.tolist()) for g in geoms],
    )
    path = os.path.join(args.out_dir, f"{args.tag}_r{args.replicate}.json")
    json.dump(res, open(path, "w"))
    print(f"[prod] ge_total={ge} ge_iter={log.get('ge_iter')} ge_hess={log.get('ge_hess')} "
          f"ge_hvp={log.get('ge_hvp')} s_per_ge={res['s_per_ge']}", flush=True)
    print(f"[prod] wrote {path}", flush=True)
    # ARTIFACT GATE: exit code 0 is not evidence of a run. Demand a non-zero
    # gradient budget, at least one final geometry, and (unless explicitly
    # skipped) at least one saddle measurement -- otherwise FAIL loudly.
    n_geom = sum(1 for g in geoms if g is not None)
    bad = []
    if not ge:
        bad.append("ge_total==0")
    if n_geom == 0:
        bad.append("no final geometries")
    if not args.skip_saddle and not sad:
        bad.append("saddle gate produced 0 cases")
    if bad:
        print(f"PHESS_PROD_FAIL {args.tag}_r{args.replicate}: {'; '.join(bad)}", flush=True)
        sys.exit(2)
    print(f"[prod] geoms={n_geom}/{len(cases)} saddle_cases={len(sad)}", flush=True)
    print("PHESS_PROD_DONE", flush=True)


if __name__ == "__main__":
    main()
