#!/usr/bin/env python
"""A3-fwd e2e (opt campaign 2026-07-28; verbatim copy of the validated
b67_pipeline_chunked.py harness + FAST_INFERENCE/COMPILE_MODEL env knobs): the a7_pipeline.py end-to-end batched
TS-search pipeline (batched CI-NEB -> batched DS-P-RFO -> batched frequency),
CHUNKED at batch size B and fully instrumented, so it can serve as the common
A/B harness for two different questions:

  B6  vary PYTORCH_CUDA_ALLOC_CONF (expandable_segments True vs False), same tree
      -> does the F2 allocator env-var side-effect in maple/__init__.py actually
         do anything?  metrics: OOM halve-retry count, peak VRAM, wall.
  B7  vary FORK (baseline_96eff9f  vs  refactor/batch-calc-unify), same env
      -> did the 7-backend BatchCalcABC retrofit regress throughput?
         gate: |dwall| <= 3%.  metrics: wall, struct/s, grad-equivalents.

Same stages / same params as bench/a7_pipeline.py; the ONLY differences are
(a) the pipeline is chunked at B reactions end-to-end (NEB + P-RFO + freq all run
    at width B, i.e. a real production run at batch size B), and
(b) instrumentation: GradCounter (exact DFT-gradient-equivalents via calc._forward),
    torch peak VRAM, nvidia-smi util/VRAM sampler, calc._auto_chunk_retries
    (the OOM halve-and-retry event counter that lives in UMABatchCalc), JSON out.

Both trees expose every symbol used here, so the SAME file runs verbatim on both
-> a clean A/B.

Science sanity (must be identical across arms, else the arms did different work):
success% (1 imag & E>0), barrier MAE vs DFT, median TS-RMSD.

env: FORK MODEL PKL N B N_IMAGES NEB_MAXITER DYNEB RECALC OUTDIR TAG
"""
import os, sys, time, json, pickle, threading, subprocess
import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_masses

FORK = os.environ["FORK"]; sys.path.insert(0, FORK)
MODEL = os.environ["MODEL"]; PKL = os.environ["PKL"]; OUTDIR = os.environ["OUTDIR"]
N = int(os.environ.get("N", "100"))
B = int(os.environ.get("B", "16"))
N_IMAGES = int(os.environ.get("N_IMAGES", "10"))
NEB_MAXITER = int(os.environ.get("NEB_MAXITER", "150"))
DYNEB = os.environ.get("DYNEB", "1") == "1"
RECALC = int(os.environ.get("RECALC", "8"))
TAG = os.environ.get("TAG", "b67")
FAST_INFERENCE = os.environ.get("FAST_INFERENCE", "0") == "1"
COMPILE_MODEL = os.environ.get("COMPILE_MODEL", "0") == "1"
ALLOC_ENV_AT_START = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>")
HA2EV = 27.211386245988

from maple.function.dispatcher.ts.algorithm.neb import NEB
from maple.function.dispatcher.ts.algorithm import BatchPRFO
from maple.function.utility.molecules import Molecules
from maple.function.calculator.uma._uma_batch_calculator import UMABatchCalc

ALLOC_ENV_AFTER_MAPLE = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>")


class GradCounter:
    """Exact DFT-gradient-equivalents: one MLIP forward over B structures == B
    single-gradient-equivalents. Wraps calc._forward and sums the batch dim, so
    it honestly includes DyNEB image freezing and the FD-Hessian forwards."""
    def __init__(self, calc):
        self.orig = calc._forward; self.n = 0; self.calls = 0
        calc._forward = self._w

    def _w(self, *a, **k):
        out = self.orig(*a, **k)
        E = out[0] if isinstance(out, (tuple, list)) else out
        self.calls += 1
        try:
            self.n += int(E.shape[0])
        except Exception:
            self.n += 1
        return out

    def take(self):
        n, c = self.n, self.calls
        self.n = 0; self.calls = 0
        return n, c


class GPUSampler(threading.Thread):
    def __init__(self, dt=0.5):
        super().__init__(daemon=True); self.dt = dt; self.run_flag = True
        self.util = []; self.mem = []

    def run(self):
        while self.run_flag:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2)
                u, m = out.stdout.strip().split("\n")[0].split(",")
                self.util.append(float(u)); self.mem.append(float(m))
            except Exception:
                pass
            time.sleep(self.dt)

    def stop(self):
        self.run_flag = False; self.join(timeout=3)
        u = np.array(self.util) if self.util else np.array([0.0])
        m = np.array(self.mem) if self.mem else np.array([0.0])
        return float(u.mean()), float(u.max()), float(m.max()), len(self.util)


def kabsch_rmsd(P, Q):
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc); d = np.sign(np.linalg.det(Vt.T @ U.T))
    return float(np.sqrt(((Pc @ (U @ np.diag([1, 1, d]) @ Vt) - Qc) ** 2).sum(1).mean()))


def load(pkl, n):
    recs = pickle.load(open(pkl, "rb"))[:n]
    out = []
    for r in recs:
        Z = np.asarray(r["atomic_numbers"])
        out.append(dict(idx=r.get("index"), formula=r.get("formula"), Z=Z,
                        natoms=int(r["natoms"]),
                        R=Atoms(numbers=Z, positions=np.asarray(r["reactant"]["positions"], float)),
                        P=Atoms(numbers=Z, positions=np.asarray(r["product"]["positions"], float)),
                        ts=np.asarray(r["transition_state"]["positions"], float),
                        e_react=float(r["reactant"]["energy"]),
                        e_ts=float(r["transition_state"]["energy"])))
    return out


def n_imag(calc, atoms_list, Z_list):
    calc.prepare(atoms_list)
    H = calc.get_efh_gpu()[2]
    res = []
    for b, (a, Z) in enumerate(zip(atoms_list, Z_list)):
        dof = 3 * len(a)
        Hb = H[b, :dof, :dof].double().cpu().numpy()
        msr = np.repeat(np.sqrt(atomic_masses[Z]), 3)
        Hmw = Hb / np.outer(msr, msr)
        ev = np.linalg.eigvalsh(0.5 * (Hmw + Hmw.T))
        res.append(int((ev < -1e-5).sum()))
    return res


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTDIR, exist_ok=True)
    data = load(PKL, N)
    nchunk = int(np.ceil(len(data) / B))
    print(f"[{TAG}] FORK={FORK} fast_inference={FAST_INFERENCE} compile_model={COMPILE_MODEL}", flush=True)
    print(f"[{TAG}] PYTORCH_CUDA_ALLOC_CONF at start='{ALLOC_ENV_AT_START}' "
          f"after maple import='{ALLOC_ENV_AFTER_MAPLE}'", flush=True)
    print(f"[{TAG}] N={len(data)} B={B} ({nchunk} chunks) n_images={N_IMAGES} "
          f"neb_maxiter={NEB_MAXITER} dyneb={DYNEB} prfo_recalc={RECALC} dev={dev}", flush=True)
    print(f"[{TAG}] natoms {min(d['natoms'] for d in data)}..{max(d['natoms'] for d in data)} "
          f"median {sorted(d['natoms'] for d in data)[len(data)//2]}", flush=True)

    calc = UMABatchCalc(MODEL, device=dev, dtype=torch.float64, task="omol",
                        fast_inference=FAST_INFERENCE, compile_model=COMPILE_MODEL)
    ctr = GradCounter(calc)
    neb = NEB(output="/dev/shm/%s_neb.out" % TAG,
              atoms_or_molecules=Molecules([data[0]["R"].copy(), data[0]["P"].copy()]),
              paras={"n_images": N_IMAGES, "max_iter": NEB_MAXITER})
    neb._mol_calc = calc
    neb._use_batch = neb._is_batch_calc(calc)

    # ---- warmup (weights/graphs; excluded from timings) ----
    neb.run_multiband([[data[0]["R"].copy(), data[0]["P"].copy()]], climbing=True,
                      dyneb=DYNEB, spring_mode="dynamic", verbose=False)
    if dev == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ctr.take()
    retries0 = int(getattr(calc, "_auto_chunk_retries", 0))

    smp = GPUSampler(); smp.start()
    t_all0 = time.time()
    t_neb = t_prfo = t_freq = 0.0
    ge_neb = ge_prfo = ge_freq = 0
    cl_neb = cl_prfo = cl_freq = 0
    neb_conv = prfo_conv = 0
    all_ts, all_nim, all_ereact, chunk_log = [], [], [], []

    for ci in range(nchunk):
        chunk = data[ci * B:(ci + 1) * B]
        # ---- Stage 1: batched CI-NEB over the chunk ----
        t0 = time.time()
        bands = [[d["R"].copy(), d["P"].copy()] for d in chunk]
        res, st = neb.run_multiband(bands, climbing=True, dyneb=DYNEB,
                                    spring_mode="dynamic", verbose=False)
        dt = time.time() - t0; t_neb += dt
        g, c = ctr.take(); ge_neb += g; cl_neb += c
        heis = [r["images"][r["hei"]] for r in res]
        all_ereact.extend(float(r["energies"][0]) for r in res)
        nc = sum(1 for r in res if r["converged"]); neb_conv += nc

        # ---- Stage 2: batched DS-P-RFO refine of the HEIs ----
        t0 = time.time()
        m = Molecules([h.copy() for h in heis]); m.calc = calc
        pr = BatchPRFO(output="/dev/shm/%s_prfo.out" % TAG, device=dev,
                       max_outer_iter=200, recalc=RECALC, initial_hessian="lindh",
                       ts_hessian_inject=True, trust_mode="nw", mode_follow_guard=True)
        pr.run(m)
        dt2 = time.time() - t0; t_prfo += dt2
        g, c = ctr.take(); ge_prfo += g; cl_prfo += c
        ts_chunk = list(m.multiatoms)
        pc = sum(1 for s in pr._final_status if s == "converged"); prfo_conv += pc

        # ---- Stage 3: batched frequency (MW Hessian -> n_imag) + E_ts ----
        t0 = time.time()
        nim = n_imag(calc, ts_chunk, [d["Z"] for d in chunk])
        calc.prepare(ts_chunk)
        E_ts = calc.get_ef_gpu()[0].detach().cpu().numpy().astype(float)
        dt3 = time.time() - t0; t_freq += dt3
        g, c = ctr.take(); ge_freq += g; cl_freq += c

        all_ts.extend(zip(ts_chunk, list(E_ts)))
        all_nim.extend(nim)
        chunk_log.append(dict(chunk=ci, n=len(chunk), neb_s=dt, prfo_s=dt2, freq_s=dt3,
                              neb_conv=nc, prfo_conv=pc))
        print(f"[{TAG}] chunk {ci+1}/{nchunk} (n={len(chunk)}): NEB={dt:.1f}s({nc}/{len(chunk)}) "
              f"P-RFO={dt2:.1f}s({pc}/{len(chunk)}) freq={dt3:.1f}s", flush=True)

    wall_total = time.time() - t_all0
    u_mean, u_max, vram_smi, nsamp = smp.stop()
    retries = int(getattr(calc, "_auto_chunk_retries", 0)) - retries0
    pk_alloc = (torch.cuda.max_memory_allocated() / 2**20) if dev == "cuda" else 0.0
    pk_resv = (torch.cuda.max_memory_reserved() / 2**20) if dev == "cuda" else 0.0

    # ---- science sanity ----
    rows = []
    for d, (ts, ets), ne, nm in zip(data, all_ts, all_ereact, all_nim):
        bar = (float(ets) - ne) * HA2EV
        bar_dft = d["e_ts"] - d["e_react"]
        rmsd = kabsch_rmsd(ts.get_positions(), d["ts"])
        ok = (nm == 1) and bar > 0
        rows.append(dict(idx=d["idx"], ok=ok, bar=bar, err=abs(bar - bar_dft),
                         rmsd=rmsd, nim=nm))
    good = [r for r in rows if r["ok"]]
    succ = 100.0 * len(good) / len(rows)
    mae = float(np.mean([r["err"] for r in good])) if good else float("nan")
    medr = float(np.median([r["rmsd"] for r in good])) if good else float("nan")

    ge_tot = ge_neb + ge_prfo + ge_freq
    out = dict(bench="A3_fwd_e2e", tag=TAG, fork=FORK,
               tree=("baseline_96eff9f" if "baseline" in FORK else
                     ("refactor_batchcalc" if "refactor" in FORK else FORK)),
               alloc_conf_at_start=ALLOC_ENV_AT_START,
               alloc_conf_after_maple_import=ALLOC_ENV_AFTER_MAPLE,
               model="uma-s-1p1", dtype="float64", task="omol", device=dev,
               fast_inference=FAST_INFERENCE, compile_model=COMPILE_MODEL,
               gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", n_gpu=1,
               N=len(data), B=B, n_chunks=nchunk, n_images=N_IMAGES,
               neb_maxiter=NEB_MAXITER, dyneb=DYNEB, prfo_recalc=RECALC,
               natoms_min=min(d["natoms"] for d in data),
               natoms_max=max(d["natoms"] for d in data),
               wall_total=wall_total, wall_neb=t_neb, wall_prfo=t_prfo, wall_freq=t_freq,
               struct_per_s=len(data) / wall_total,
               grad_equiv_total=ge_tot, grad_equiv_neb=ge_neb, grad_equiv_prfo=ge_prfo,
               grad_equiv_freq=ge_freq, grad_equiv_per_rxn=ge_tot / len(data),
               forward_calls_total=cl_neb + cl_prfo + cl_freq,
               forward_calls_neb=cl_neb, forward_calls_prfo=cl_prfo,
               forward_calls_freq=cl_freq,
               oom_halve_retries=retries,
               peak_vram_torch_alloc_MB=pk_alloc, peak_vram_torch_reserved_MB=pk_resv,
               peak_vram_smi_MB=vram_smi, util_mean=u_mean, util_max=u_max,
               util_samples=nsamp,
               neb_converged=f"{neb_conv}/{len(data)}",
               prfo_converged=f"{prfo_conv}/{len(data)}",
               success_pct=succ, barrier_MAE_eV=mae, median_TS_RMSD_A=medr,
               n_imag_dist=[r["nim"] for r in rows], chunks=chunk_log)

    print(f"\n[{TAG}] ===== AGGREGATE =====", flush=True)
    print(f"  tree                 : {out['tree']}", flush=True)
    print(f"  alloc conf           : {ALLOC_ENV_AFTER_MAPLE}", flush=True)
    print(f"  reactions / B        : {len(data)} / {B}  ({nchunk} chunks)", flush=True)
    print(f"  wall total           : {wall_total:.1f}s  "
          f"(NEB {t_neb:.1f} + P-RFO {t_prfo:.1f} + freq {t_freq:.1f})", flush=True)
    print(f"  structures/s         : {len(data)/wall_total:.4f}", flush=True)
    print(f"  DFT-grad-equivalents : {ge_tot} total ({ge_tot/len(data):.0f}/rxn)  "
          f"[NEB {ge_neb} | P-RFO {ge_prfo} | freq {ge_freq}]", flush=True)
    print(f"  GPU forward calls    : {cl_neb+cl_prfo+cl_freq}", flush=True)
    print(f"  OOM halve-retries    : {retries}", flush=True)
    print(f"  peak VRAM            : torch_alloc={pk_alloc:.0f}MB "
          f"torch_reserved={pk_resv:.0f}MB nvidia-smi={vram_smi:.0f}MB", flush=True)
    print(f"  GPU util             : mean={u_mean:.0f}% max={u_max:.0f}% ({nsamp} samples)", flush=True)
    print(f"  CI-NEB conv          : {neb_conv}/{len(data)}", flush=True)
    print(f"  P-RFO conv           : {prfo_conv}/{len(data)}", flush=True)
    print(f"  success%(1imag&E>0)  : {succ:.1f}", flush=True)
    print(f"  barrier MAE (eV)     : {mae:.3f}", flush=True)
    print(f"  median TS-RMSD (A)   : {medr:.3f}", flush=True)

    p = os.path.join(OUTDIR, f"{TAG}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n[{TAG}] -> {p}", flush=True)
    print("B67_PIPELINE_OK", flush=True)


if __name__ == "__main__":
    main()
