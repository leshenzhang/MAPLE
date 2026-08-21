#!/usr/bin/env python
"""maple-bench matrix runner: one command runs {dispatcher} x {B} x {backend} with
replicates built in, unified maple-bench-v1 JSON per run.

Dispatchers
-----------
pipeline  full batched TS pipeline, chunked at B: CI-NEB (n_images/maxiter/DyNEB)
          -> BatchPRFO(recalc, lindh, ts_hessian_inject) -> batched frequency.
          Stage params/metrics identical to bench_2026-07-14 b67_pipeline_chunked.py
          (the D-227/B7 harness) so numbers stay comparable.
forward   raw get_ef_gpu micro-benchmark (prepare+forward per timed iteration,
          same convention as the July B5/a3 scaling loop).
hessian   get_efh_gpu segment timing, per mode (numerical FD / autograd).
autoneb   B3 arms: --arm serial (single-reaction AutoNEB oracle, per-rxn budget
          via SIGALRM, incremental flush) | --arm batched (AutoNEBBatch).
parity    canon-vs-canon floors + gate negative-control self-test. Two-tier:
          PASS / NUMERICALLY_DIFFERENT (escalates to same-saddle+science) /
          REGRESSION / SKIP.
counter   EMISSION self-test: pins GradCounter against known answers (B=1 -> 1,
          B=16 -> 16, FD Hessian -> the analytic 2*3N form) for every backend in
          the registry. A backend that FAILs here may not publish s/grad-equiv.

Examples
--------
  run_matrix.py --dispatcher pipeline --backend uma --B 1,16,64 --N 100 --reps 2 \
      --model $MODEL --pkl $PKL --outdir $OUT --tag base
  run_matrix.py --dispatcher forward --B 1,8,16,32,64,128 --reps 2 ...
  run_matrix.py --dispatcher hessian --B 1,16,64 --hess-modes numerical,autograd ...
  run_matrix.py --dispatcher autoneb --arm serial --rxn-start 0 --rxn-count 8 \
      --serial-budget-s 720 ...
  run_matrix.py --dispatcher parity ...
"""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.core import (HA2EV, GPUSampler, GradCounter, PathProbe, build_backend,
                        counter_self_test, dump, kabsch_rmsd, load_ts1x,
                        n_imag_batch, run_record, stage_share_gate,
                        verify_counter_inline)


def _certify(args, needs_hessian, probe_mode=None, **calc_kw):
    """Per-run counter self-certification -> (status, detail). Cheap; every record
    carries it so a broken instrument can never be mistaken for a missing cell.

    calc_kw is the SAME calculator configuration the run itself uses (D-273: a
    counter case pinned to a different mode than the real case reports a fake
    FAIL/PASS)."""
    data = load_ts1x(args.pkl, 4)
    base = [d["TS"] for d in data]
    mols_fn = lambda B: [base[i % len(base)].copy() for i in range(B)]
    build = lambda **kw: build_backend(args.backend, args.model, **dict(calc_kw, **kw))
    st, det = verify_counter_inline(build, mols_fn, needs_hessian=needs_hessian,
                                    hessian_mode=probe_mode)
    print(f"[certify] backend={args.backend} needs_hessian={needs_hessian} "
          f"counter={st} {det}", flush=True)
    return st, det


def _cuda_sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak():
    """Reset the peak counters AND release the cached pool.

    Without empty_cache() the allocator pool from an earlier run in the SAME
    process is still reserved, so max_memory_reserved() reports that high-water
    mark instead of this run's own footprint (observed: every rep-2 forward run
    reporting an identical 3740 MB inherited from rep-1's B=128).
    """
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peaks():
    import torch
    if torch.cuda.is_available():
        return (torch.cuda.max_memory_reserved() / 2**20,
                torch.cuda.max_memory_allocated() / 2**20)
    return (0.0, 0.0)


# --------------------------------------------------------------------- pipeline
def run_pipeline(args, B, rep):
    import torch
    from maple.function.dispatcher.ts.algorithm import BatchPRFO
    from maple.function.dispatcher.ts.algorithm.neb import NEB
    from maple.function.utility.molecules import Molecules

    if args.backend != "uma" and (args.fd_mode != "central" or args.fast_inference):
        raise SystemExit(f"--fd-mode/--fast-inference are UMA-only knobs; "
                         f"backend={args.backend} would silently ignore them")
    cert_st, cert_det = _certify(args, needs_hessian=True, fd_mode=args.fd_mode,
                                 fast_inference=args.fast_inference)
    # --start makes concurrent processes take DISJOINT slices. Without it every
    # process reads [0:N] and a "same total work" comparison silently becomes
    # "one process does N reactions, K processes each redo the same N".
    data = load_ts1x(args.pkl, args.N, start=args.start, nat_min=args.natoms_min,
                     nat_max=args.natoms_max, tier_sample=args.tier_sample)
    # (e) end-to-end arm knobs: the same three factors the P-RFO+freq campaign
    # swept (D-275), now applied to the WHOLE pipeline so the CI-NEB segment is
    # inside the measured wall. fd_mode/fast_inference live on the calculator,
    # recalc on BatchPRFO (already exposed).
    calc = build_backend(args.backend, args.model,
                         fd_mode=args.fd_mode, fast_inference=args.fast_inference)
    ctr = GradCounter(calc)
    probe = PathProbe(calc)          # which Hessian implementation actually runs
    tag = f"{args.tag}_pipeline_{args.backend}_B{B}_r{rep}"
    nchunk = int(np.ceil(len(data) / B))
    neb = NEB(output=f"/dev/shm/{tag}_neb.out",
              atoms_or_molecules=Molecules([data[0]["R"].copy(), data[0]["P"].copy()]),
              paras={"n_images": args.n_images, "max_iter": args.neb_maxiter})
    neb._mol_calc = calc
    neb._use_batch = neb._is_batch_calc(calc)

    # warmup (weights/graphs; excluded)
    neb.run_multiband([[data[0]["R"].copy(), data[0]["P"].copy()]], climbing=True,
                      dyneb=args.dyneb, spring_mode="dynamic", verbose=False)
    _cuda_sync()
    _reset_peak()
    ctr.take()

    smp = GPUSampler()
    smp.start()
    t_all0 = time.time()
    t_st = dict(neb=0.0, prfo=0.0, freq=0.0)
    ge_st = dict(neb=0, prfo=0, freq=0)
    calls = 0
    neb_conv = prfo_conv = 0
    all_ts, all_nim, all_ereact, chunk_log = [], [], [], []

    for ci in range(nchunk):
        chunk = data[ci * B:(ci + 1) * B]
        t0 = time.time()
        bands = [[d["R"].copy(), d["P"].copy()] for d in chunk]
        res, st = neb.run_multiband(bands, climbing=True, dyneb=args.dyneb,
                                    spring_mode="dynamic", verbose=False)
        t_st["neb"] += time.time() - t0
        g, c = ctr.take(); ge_st["neb"] += g; calls += c
        heis = [r["images"][r["hei"]] for r in res]
        all_ereact.extend(float(r["energies"][0]) for r in res)
        neb_conv += sum(1 for r in res if r["converged"])

        t0 = time.time()
        m = Molecules([h.copy() for h in heis]); m.calc = calc
        pr = BatchPRFO(output=f"/dev/shm/{tag}_prfo.out", device="cuda",
                       max_outer_iter=200, recalc=args.recalc, initial_hessian="lindh",
                       ts_hessian_inject=True, trust_mode="nw", mode_follow_guard=True)
        pr.run(m)
        t_st["prfo"] += time.time() - t0
        g, c = ctr.take(); ge_st["prfo"] += g; calls += c
        ts_chunk = list(m.multiatoms)
        prfo_conv += sum(1 for s in pr._final_status if s == "converged")

        t0 = time.time()
        # D-280: the n_imag CLASSIFICATION must not inherit the search's FD mode.
        # forward-FD leaves the geometry untouched (median |dRMSD| vs central =
        # 1e-4 A) but its O(delta) error pushes the near-zero translational/
        # rotational modes negative -> 86/87 index-1 saddles get re-labelled as
        # 2-4 imaginary. The freq stage is ~8% of wall, so validating it with
        # central FD costs almost nothing. --freq-fd-mode same reproduces the
        # pre-D-280 records byte-for-byte.
        _fd_saved = getattr(calc, "_fd_mode", None)
        if args.freq_fd_mode != "same" and _fd_saved is not None:
            calc._fd_mode = args.freq_fd_mode
        try:
            nim = n_imag_batch(calc, ts_chunk, [d["Z"] for d in chunk])
        finally:
            if _fd_saved is not None:
                calc._fd_mode = _fd_saved
        calc.prepare(ts_chunk)
        E_ts = calc.get_ef_gpu()[0].detach().cpu().numpy().astype(float)
        t_st["freq"] += time.time() - t0
        g, c = ctr.take(); ge_st["freq"] += g; calls += c

        all_ts.extend(zip(ts_chunk, list(E_ts)))
        all_nim.extend(nim)
        chunk_log.append(dict(chunk=ci, n=len(chunk)))
        print(f"[{tag}] chunk {ci+1}/{nchunk} done "
              f"(NEB {t_st['neb']:.0f}s | P-RFO {t_st['prfo']:.0f}s | freq {t_st['freq']:.0f}s)",
              flush=True)

    wall = time.time() - t_all0
    gpu = smp.stop()
    pk_resv, pk_alloc = _peaks()

    rows = []
    for d, (ts, ets), ne, nm in zip(data, all_ts, all_ereact, all_nim):
        bar = (float(ets) - ne) * HA2EV
        rows.append(dict(idx=d["idx"], ok=(nm == 1 and bar > 0), bar=bar,
                         err=abs(bar - (d["e_ts"] - d["e_react"])),
                         rmsd=kabsch_rmsd(ts.get_positions(), d["ts"]), nim=nm))
    good = [r for r in rows if r["ok"]]
    nims = [r["nim"] for r in rows]
    science = dict(
        success_rate=len(good) / len(rows),
        barrier_MAE_eV=float(np.mean([r["err"] for r in good])) if good else None,
        median_TS_RMSD_A=float(np.median([r["rmsd"] for r in good])) if good else None,
        pct_1imag=100.0 * sum(1 for x in nims if x == 1) / len(nims),
        n_imag_hist={str(k): int(sum(1 for x in nims if x == k)) for k in sorted(set(nims))},
        neb_converged=f"{neb_conv}/{len(data)}", prfo_converged=f"{prfo_conv}/{len(data)}")
    ge_tot = sum(ge_st.values())
    share_gate = stage_share_gate(t_st, ge_st)
    print(f"[{tag}] stage_share_gate = {share_gate['status']} "
          f"{[(r['stage'], round(r['wall_share'],3), round(r['ge_share'],4), r['status']) for r in share_gate.get('stages',[])]}",
          flush=True)
    print(f"[{tag}] hessian path probe = {probe.report()}", flush=True)
    rec = run_record(
        bench="pipeline_ts", dispatcher="pipeline", backend=args.backend, B=B,
        N=len(data), rep=rep, tag=tag,
        hessian_mode=getattr(calc, "hessian_mode", "n/a"),
        counter_status=cert_st, counter_detail=cert_det,
        params=dict(model=os.path.basename(args.model), dtype="float64", task="omol",
                    n_images=args.n_images, neb_maxiter=args.neb_maxiter,
                    dyneb=args.dyneb, recalc=args.recalc, n_chunks=nchunk,
                    fd_mode=args.fd_mode, freq_fd_mode=args.freq_fd_mode,
                    fast_inference=int(args.fast_inference),
                    start=args.start,
                    natoms_tier=[args.natoms_min, args.natoms_max],
                    tier_sample=args.tier_sample,
                    natoms_mean=float(np.mean([d["natoms"] for d in data])),
                    natoms_min=min(d["natoms"] for d in data),
                    natoms_max=max(d["natoms"] for d in data)),
        wall_s=wall, grad_equiv_total=ge_tot, forward_calls=calls,
        wall_stages=t_st, ge_stages=ge_st, gpu=gpu,
        vram_reserved_MB=pk_resv, vram_alloc_MB=pk_alloc, science=science,
        extra=dict(struct_per_s=len(data) / wall,
                   grad_equiv_per_rxn=ge_tot / len(data),
                   oom_halve_retries=int(getattr(calc, "_auto_chunk_retries", 0)),
                   stage_share_gate=share_gate,
                   hessian_path_probe=probe.report(),
                   rows=rows))
    print(f"[{tag}] counter={cert_st} wall={wall:.1f}s gE={ge_tot} "
          f"s/gE={(wall/ge_tot if cert_st == 'VERIFIED' and ge_tot else float('nan')):.5f} "
          f"succ={science['success_rate']*100:.1f}% MAE={science['barrier_MAE_eV']} "
          f"medRMSD={science['median_TS_RMSD_A']} util={gpu['util_mean']:.0f}%", flush=True)
    return dump(rec, args.outdir, tag)


# --------------------------------------------------------------------- forward
def run_forward(args, B, rep):
    cert_st, cert_det = _certify(args, needs_hessian=False)
    data = load_ts1x(args.pkl, max(B, 16))
    mols = [d["TS"] for d in data][:B] if B <= len(data) else None
    if mols is None or len(mols) < B:
        base = [d["TS"] for d in data]
        mols = [base[i % len(base)].copy() for i in range(B)]
    calc = build_backend(args.backend, args.model)
    ctr = GradCounter(calc)
    tag = f"{args.tag}_forward_{args.backend}_B{B}_r{rep}"
    # warmup
    calc.prepare([m.copy() for m in mols]); calc.get_ef_gpu()
    _cuda_sync(); _reset_peak(); ctr.take()
    smp = GPUSampler(dt=0.25); smp.start()
    t0 = time.time()
    for _ in range(args.iters):
        calc.prepare([m.copy() for m in mols])
        calc.get_ef_gpu()
    _cuda_sync()
    wall = time.time() - t0
    gpu = smp.stop()
    ge, calls = ctr.take()
    pk_resv, pk_alloc = _peaks()
    rec = run_record(
        bench="raw_forward", dispatcher="forward", backend=args.backend, B=B,
        N=B, rep=rep, tag=tag, hessian_mode="none",
        counter_status=cert_st, counter_detail=cert_det,
        params=dict(model=os.path.basename(args.model), dtype="float64", task="omol",
                    iters=args.iters, loop="prepare+get_ef_gpu (a3/B5 convention)",
                    geometry="ts1x DFT TS structures"),
        wall_s=wall, grad_equiv_total=ge, forward_calls=calls, gpu=gpu,
        vram_reserved_MB=pk_resv, vram_alloc_MB=pk_alloc,
        extra=dict(ms_per_iter=1e3 * wall / args.iters,
                   struct_per_s=B * args.iters / wall))
    print(f"[{tag}] {1e3*wall/args.iters:.2f} ms/iter  {B*args.iters/wall:.1f} struct/s  "
          f"s/gE={wall/ge:.6f} util={gpu['util_mean']:.0f}%", flush=True)
    return dump(rec, args.outdir, tag)


# --------------------------------------------------------------------- hessian
def run_hessian(args, B, rep, mode):
    # probe_mode pins what get_efh_gpu the PROBE calls; calc_kw configures the
    # calculator itself. They must stay separate names -- passing both as
    # `hessian_mode` collides ("got multiple values for keyword argument").
    cert_st, cert_det = _certify(args, needs_hessian=(mode not in ("autograd", "analytic")),
                                 probe_mode=mode,
                                 **(dict(hessian_mode=mode) if args.backend == "uma" else {}))
    data = load_ts1x(args.pkl, max(B, 16))
    base = [d["TS"] for d in data]
    mols = [base[i % len(base)].copy() for i in range(B)]
    kw = dict(hessian_mode=mode) if args.backend == "uma" else {}
    tag = f"{args.tag}_hessian-{mode}_{args.backend}_B{B}_r{rep}"
    try:
        calc = build_backend(args.backend, args.model, **kw)
        ctr = GradCounter(calc)
        calc.prepare([m.copy() for m in mols])
        calc.get_efh_gpu(mode=mode)                     # warmup
        _cuda_sync(); _reset_peak(); ctr.take()
        smp = GPUSampler(dt=0.25); smp.start()
        t0 = time.time()
        for _ in range(args.hess_iters):
            calc.prepare([m.copy() for m in mols])
            calc.get_efh_gpu(mode=mode)
        _cuda_sync()
        wall = time.time() - t0
        gpu = smp.stop()
        ge, calls = ctr.take()
        pk_resv, pk_alloc = _peaks()
    except Exception as e:
        rec = run_record(bench="hessian_segment", dispatcher="hessian",
                         backend=args.backend, B=B, N=B, rep=rep, tag=tag,
                         hessian_mode=mode, counter_status=cert_st,
                         counter_detail=cert_det,
                         params=dict(mode=mode, error=f"{type(e).__name__}: {str(e)[:200]}"),
                         wall_s=0.0, grad_equiv_total=0, forward_calls=0,
                         extra=dict(status="FAILED"))
        print(f"[{tag}] FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)
        return dump(rec, args.outdir, tag)
    rec = run_record(
        bench="hessian_segment", dispatcher="hessian", backend=args.backend, B=B,
        N=B, rep=rep, tag=tag, hessian_mode=mode,
        counter_status=cert_st, counter_detail=cert_det,
        params=dict(model=os.path.basename(args.model), dtype="float64", task="omol",
                    mode=mode, iters=args.hess_iters,
                    note=("grad-equiv counts calc._forward only; the autograd Hessian's "
                          "double-backward work is NOT a forward, so compare autograd-vs-FD "
                          "on wall_s, not on s_per_grad_equiv")),
        wall_s=wall, grad_equiv_total=ge, forward_calls=calls, gpu=gpu,
        vram_reserved_MB=pk_resv, vram_alloc_MB=pk_alloc,
        extra=dict(s_per_hessian_call=wall / args.hess_iters,
                   s_per_structure_hessian=wall / (args.hess_iters * B)))
    print(f"[{tag}] {wall/args.hess_iters:.3f} s/call ({wall/(args.hess_iters*B)*1e3:.1f} ms/struct) "
          f"gE={ge} util={gpu['util_mean']:.0f}%", flush=True)
    return dump(rec, args.outdir, tag)


# --------------------------------------------------------------------- autoneb
class _Budget(Exception):
    pass


def _alarm(sig, frm):
    raise _Budget()


def run_autoneb(args, rep):
    import torch
    from ase.calculators.calculator import Calculator, all_changes
    from maple.function.dispatcher.ts.algorithm.autoneb import AutoNEB, AutoNEBBatch

    class Shim(Calculator):
        implemented_properties = ["energy", "free_energy", "forces"]

        def __init__(self, bc, **kw):
            super().__init__(**kw)
            self._bc = bc

        def calculate(self, atoms=None, properties=("energy",),
                      system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self._bc.prepare([self.atoms], fixed_nmax=None)
            E, F = self._bc.get_ef_gpu()
            n = len(self.atoms)
            e = float(E.detach().cpu().numpy()[0])
            self.results["energy"] = e
            self.results["free_energy"] = e
            self.results["forces"] = (F.detach().cpu().numpy()[0, :3 * n]
                                      .reshape(n, 3).astype(np.float64).copy())

    cert_st, cert_det = _certify(args, needs_hessian=False)
    data = load_ts1x(args.pkl, args.rxn_count, start=args.rxn_start)
    calc = build_backend(args.backend, args.model)
    ctr = GradCounter(calc)
    paras = {"autoneb": {
        "n_images": args.aneb_nimages, "ifidpp": 1, "ang_max": 0.3, "ang_iter": 20,
        "path_iter": 9999, "ep_iter": 9999, "max_iter": args.aneb_maxiter,
        "do_final_refine": True, "final_refine_max_iter": 60, "verbose": 0}}
    tag = f"{args.tag}_autoneb-{args.arm}_{args.backend}_rx{args.rxn_start}-{args.rxn_start+args.rxn_count-1}_r{rep}"

    # warmup
    w = load_ts1x(args.pkl, 1)[0]
    Rw, Pw = w["R"].copy(), w["P"].copy()
    Rw.calc = Pw.calc = Shim(calc)
    AutoNEBBatch("/dev/shm/aneb_warm.out", [[Rw, Pw]], calc=calc,
                 paras={"autoneb": dict(paras["autoneb"], max_iter=3,
                                        do_final_refine=False)}).run()
    _cuda_sync(); _reset_peak(); ctr.take()

    rows = []
    smp = GPUSampler(); smp.start()
    t0 = time.time()
    if args.arm == "serial":
        signal.signal(signal.SIGALRM, _alarm)
        for d in data:
            R, P = d["R"].copy(), d["P"].copy()
            R.calc = P.calc = Shim(calc)
            g0, c0 = ctr.n, ctr.calls
            t1 = time.time()
            censored = False
            try:
                signal.alarm(int(args.serial_budget_s))
                job = AutoNEB("/dev/shm/aneb_serial.out", [R, P], paras)
                job.run()
                signal.alarm(0)
                E = np.asarray(job.final_energies, float)
                hei = 1 + int(np.argmax(E[1:-1])) if len(E) >= 3 else int(np.argmax(E))
                mx = max(float(np.abs(im.get_positions()).max())
                         for im in job.final_images)
                rows.append(dict(idx=d["idx"], natoms=d["natoms"],
                                 wall_s=time.time() - t1,
                                 grad_equiv=ctr.n - g0, forward_calls=ctr.calls - c0,
                                 barrier_Eh=float(E[hei] - E[0]),
                                 ts_pos=np.asarray(job.final_images[hei].get_positions()).tolist(),
                                 n_images=len(E), diverged=bool(mx > 100.0),
                                 censored=False))
            except _Budget:
                censored = True
                rows.append(dict(idx=d["idx"], natoms=d["natoms"],
                                 wall_s=time.time() - t1,
                                 grad_equiv=ctr.n - g0, forward_calls=ctr.calls - c0,
                                 barrier_Eh=None, ts_pos=None, n_images=None,
                                 diverged=None, censored=True))
            finally:
                signal.alarm(0)
            print(f"[{tag}] rxn idx={d['idx']} wall={rows[-1]['wall_s']:.0f}s "
                  f"gE={rows[-1]['grad_equiv']} censored={censored}", flush=True)
            # incremental flush so a walltime kill still leaves data
            json.dump(rows, open(os.path.join(args.outdir, tag + "_rows.partial.json"), "w"),
                      default=float)
    else:  # batched
        for i in range(0, len(data), args.B_single):
            chunk = data[i:i + args.B_single]
            reactions = []
            for d in chunk:
                R, P = d["R"].copy(), d["P"].copy()
                R.calc = P.calc = Shim(calc)
                reactions.append([R, P])
            g0, c0 = ctr.n, ctr.calls
            t1 = time.time()
            # A diverged band (|x| > 100 A) explodes the neighbour-list grid and
            # OOMs the whole job (observed: radius_graph_pbc_v2 asking for 115.73
            # GiB, job 49538528). Isolate it to its chunk instead of losing the run.
            try:
                job = AutoNEBBatch("/dev/shm/aneb_batch.out", reactions, calc=calc,
                                   paras=paras)
                res = job.run()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                dt = time.time() - t1
                torch.cuda.empty_cache()
                for d in chunk:
                    rows.append(dict(idx=d["idx"], natoms=d["natoms"],
                                     wall_s=dt / len(chunk), chunk_wall_s=dt,
                                     grad_equiv=(ctr.n - g0) / len(chunk),
                                     forward_calls=(ctr.calls - c0) / len(chunk),
                                     barrier_Eh=None, ts_pos=None, n_images=None,
                                     diverged=True, censored=True,
                                     failure="OOM (diverged band blew up the "
                                             "neighbour-list grid)"))
                print(f"[{tag}] chunk {i//args.B_single}: OOM -> whole chunk censored "
                      f"({len(chunk)} rxns), continuing", flush=True)
                json.dump(rows, open(os.path.join(args.outdir,
                                                  tag + "_rows.partial.json"), "w"),
                          default=float)
                continue
            dt = time.time() - t1
            for d, r in zip(chunk, res):
                rows.append(dict(idx=d["idx"], natoms=d["natoms"],
                                 wall_s=dt / len(chunk),      # chunk wall, amortized
                                 chunk_wall_s=dt,
                                 grad_equiv=(ctr.n - g0) / len(chunk),
                                 forward_calls=(ctr.calls - c0) / len(chunk),
                                 barrier_Eh=float(r["barrier_Eh"]),
                                 ts_pos=np.asarray(r["ts_image"].get_positions()).tolist(),
                                 n_images=int(r["n_images"]),
                                 diverged=bool(np.abs(np.asarray(
                                     r["ts_image"].get_positions())).max() > 100.0),
                                 censored=False))
            print(f"[{tag}] chunk {i//args.B_single}: wall={dt:.0f}s "
                  f"gE={ctr.n-g0}", flush=True)
            json.dump(rows, open(os.path.join(args.outdir, tag + "_rows.partial.json"), "w"),
                      default=float)
    wall = time.time() - t0
    gpu = smp.stop()
    pk_resv, pk_alloc = _peaks()
    ge_tot = sum(r["grad_equiv"] for r in rows)
    n_cens = sum(1 for r in rows if r["censored"])
    rec = run_record(
        bench="autoneb_B3", dispatcher="autoneb", backend=args.backend,
        B=(1 if args.arm == "serial" else args.B_single), N=len(data), rep=rep, tag=tag,
        hessian_mode="none", counter_status=cert_st, counter_detail=cert_det,
        params=dict(model=os.path.basename(args.model), dtype="float64", task="omol",
                    arm=args.arm, rxn_start=args.rxn_start, rxn_count=args.rxn_count,
                    serial_budget_s=(args.serial_budget_s if args.arm == "serial" else None),
                    paras=paras["autoneb"]),
        wall_s=wall, grad_equiv_total=ge_tot,
        forward_calls=sum(r["forward_calls"] for r in rows),
        gpu=gpu, vram_reserved_MB=pk_resv, vram_alloc_MB=pk_alloc,
        science=dict(n_censored=n_cens,
                     n_diverged=sum(1 for r in rows if r.get("diverged"))),
        extra=dict(rows=rows))
    print(f"[{tag}] wall={wall:.1f}s gE={ge_tot} censored={n_cens}/{len(rows)}", flush=True)
    return dump(rec, args.outdir, tag)


# --------------------------------------------------------------------- parity
def run_parity(args, rep):
    from bench import parity_gate as pg
    data = load_ts1x(args.pkl, 4)
    mols = [d["TS"] for d in data]
    build = lambda: build_backend(args.backend, args.model)
    tag = f"{args.tag}_parity_{args.backend}_r{rep}"
    t0 = time.time()
    st = pg.self_test(build, mols)
    wall = time.time() - t0
    rec = run_record(bench="parity_gate", dispatcher="parity", backend=args.backend,
                     B=4, N=4, rep=rep, tag=tag, hessian_mode="numerical",
                     params=dict(model=os.path.basename(args.model)),
                     wall_s=wall, grad_equiv_total=0, forward_calls=0,
                     extra=dict(self_test=st))
    print(f"[{tag}] gate self-test verdict = {st['verdict']} "
          f"(clean_pass={st['clean_pass']}, injected_fired={st['injected_1e-3A_fired']}"
          f"->{st['injected_tier']}, ss_clean={st['same_saddle_clean']}, "
          f"ss_inj={st['same_saddle_injected_3e-2A']}, "
          f"tiers={st['tier_classifier']['verdict']})", flush=True)
    return dump(rec, args.outdir, tag)


# --------------------------------------------------------------------- counter
def run_counter(args, rep):
    """FIX-2 EMISSION self-test: does GradCounter see EVERY forward path?

    Runs known-answer cases per backend in the registry. A backend whose count
    does not match an analytic form is reported FAIL and its s/grad-equiv numbers
    must not be published.
    """
    data = load_ts1x(args.pkl, 16)
    base = [d["TS"] for d in data]
    mols_fn = lambda B: [base[i % len(base)].copy() for i in range(B)]
    backends = [b.strip() for b in args.counter_backends.split(",") if b.strip()]
    results = {}
    for name in backends:
        model = args.model
        if name == "mace_traced":
            model = os.environ.get("TOY_MACE", "")
        elif name == "mace_autograd":
            model = os.environ.get("MACEOFF_RAW", "")
        if not model or not os.path.exists(model):
            results[name] = dict(name="counter_self_test", backend=name,
                                 verdict="SKIP",
                                 note=f"no model file for backend '{name}' "
                                      f"(path='{model}')")
            print(f"[counter] {name}: SKIP (no model)", flush=True)
            continue
        modes = (("numerical", "autograd") if name == "uma" else ("numerical",))
        build = (lambda m=model, n=name: (lambda **kw: build_backend(n, m, **kw)))()
        r = counter_self_test(build, mols_fn, name, hessian_modes=modes)
        results[name] = r
        print(f"[counter] {name}: {r['verdict']}", flush=True)
        for c in r["cases"]:
            print(f"    {c['name']:<24} {c['status']:<6} "
                  f"measured={c.get('measured')} expected={c.get('expected') or c.get('analytic_forms')}",
                  flush=True)
    verdicts = [v["verdict"] for v in results.values()]
    overall = ("FAIL" if "FAIL" in verdicts
               else ("SKIP" if all(v == "SKIP" for v in verdicts) else "OK"))
    tag = f"{args.tag}_counter_r{rep}"
    rec = run_record(bench="counter_self_test", dispatcher="counter",
                     backend=",".join(backends), B=0, N=0, rep=rep, tag=tag,
                     hessian_mode="mixed(by design)",
                     params=dict(backends=backends),
                     wall_s=0.0, grad_equiv_total=0, forward_calls=0,
                     extra=dict(overall=overall, per_backend=results))
    print(f"[{tag}] counter self-test overall = {overall}", flush=True)
    return dump(rec, args.outdir, tag)


# ------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dispatcher", required=True,
                   choices=["pipeline", "forward", "hessian", "autoneb", "parity",
                            "counter"])
    p.add_argument("--backend", default="uma")
    p.add_argument("--model", default=os.environ.get("MODEL"))
    p.add_argument("--pkl", default=os.environ.get("PKL"))
    p.add_argument("--outdir", default=os.environ.get("OUTDIR", "."))
    p.add_argument("--tag", default="bench")
    p.add_argument("--B", default="16", help="comma list of batch widths")
    p.add_argument("--N", type=int, default=100)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--rep-offset", type=int, default=1,
                   help="first replicate id (so separate jobs can be r1 / r2)")
    # pipeline knobs (b67/D-227 defaults)
    p.add_argument("--n-images", type=int, default=10)
    p.add_argument("--neb-maxiter", type=int, default=150)
    p.add_argument("--dyneb", type=int, default=1)
    p.add_argument("--recalc", type=int, default=8)
    p.add_argument("--start", type=int, default=0,
                   help="pipeline: first record of the slice (disjoint work for concurrent runs)")
    p.add_argument("--natoms-min", type=int, default=None,
                   help="pipeline: system-size tier lower bound (dimension (d))")
    p.add_argument("--natoms-max", type=int, default=None)
    p.add_argument("--tier-sample", default="head", choices=["head", "spread"],
                   help="pipeline: how to draw N records from a natoms tier (D-283: "
                        "'head' returns the tier's lower edge because ts1x is size-ordered)")
    p.add_argument("--fd-mode", default="central", choices=["central", "forward"],
                   help="pipeline: numerical-Hessian finite-difference mode (F factor)")
    p.add_argument("--freq-fd-mode", default="same", choices=["same", "central", "forward"],
                   help="pipeline: FD mode for the freq/n_imag stage only (D-280: "
                        "'central' keeps TS validation honest while the search runs forward-FD)")
    p.add_argument("--fast-inference", type=int, default=0,
                   help="pipeline: UMA activation-checkpointing off (I factor); VRAM ~2.96x")
    # forward/hessian knobs
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--hess-iters", type=int, default=3)
    p.add_argument("--hess-modes", default="numerical,autograd")
    # autoneb knobs
    p.add_argument("--arm", default="batched", choices=["serial", "batched"])
    p.add_argument("--rxn-start", type=int, default=0)
    p.add_argument("--rxn-count", type=int, default=16)
    p.add_argument("--serial-budget-s", type=int, default=720)
    p.add_argument("--B-single", type=int, default=8, help="autoneb batched chunk width")
    p.add_argument("--aneb-nimages", type=int, default=7)
    p.add_argument("--aneb-maxiter", type=int, default=100)
    # counter self-test knobs
    p.add_argument("--counter-backends", default="uma,mace_traced,mace_autograd",
                   help="registry backends to pin the GradCounter against")
    args = p.parse_args()
    args.dyneb = bool(args.dyneb)
    os.makedirs(args.outdir, exist_ok=True)
    Bs = [int(x) for x in str(args.B).split(",") if x]

    for rep in range(args.rep_offset, args.rep_offset + args.reps):
        if args.dispatcher == "pipeline":
            for B in Bs:
                run_pipeline(args, B, rep)
        elif args.dispatcher == "forward":
            for B in Bs:
                run_forward(args, B, rep)
        elif args.dispatcher == "hessian":
            for B in Bs:
                for mode in args.hess_modes.split(","):
                    run_hessian(args, B, rep, mode)
        elif args.dispatcher == "autoneb":
            run_autoneb(args, rep)
        elif args.dispatcher == "parity":
            run_parity(args, rep)
        elif args.dispatcher == "counter":
            run_counter(args, rep)
    print("RUN_MATRIX_OK", flush=True)


if __name__ == "__main__":
    main()
