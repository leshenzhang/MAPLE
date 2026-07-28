"""Shared instrumentation + schema for maple-bench (schema maple-bench-v1).

Metric conventions copied verbatim from bench_2026-07-14 (B_CLASS_BENCH) so all
2026-07-28 numbers are directly comparable with the July baselines:
  * grad-equiv: GradCounter wraps calc._forward and sums the batch dim -- one MLIP
    forward over B structures == B single-gradient-equivalents (hardware-independent).
  * GPU util/VRAM: nvidia-smi sampler (0.5 s), reported as auxiliary only.
  * VRAM: torch max_memory_reserved (primary), max_memory_allocated + smi as extras.
  * science sanity: success%(1imag & E>0), barrier MAE vs DFT (eV), median TS-RMSD (A).
"""
import json
import os
import pickle
import subprocess
import threading
import time

import numpy as np

HA2EV = 27.211386245988
SCHEMA = "maple-bench-v1"


class GradCounter:
    """Exact DFT-gradient-equivalents via calc._forward (sums the batch dim)."""

    def __init__(self, calc):
        self.orig = calc._forward
        self.n = 0
        self.calls = 0
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
        self.n = 0
        self.calls = 0
        return n, c


class GPUSampler(threading.Thread):
    def __init__(self, dt=0.5):
        super().__init__(daemon=True)
        self.dt = dt
        self.run_flag = True
        self.util = []
        self.mem = []

    def run(self):
        while self.run_flag:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2)
                u, m = out.stdout.strip().split("\n")[0].split(",")
                self.util.append(float(u))
                self.mem.append(float(m))
            except Exception:
                pass
            time.sleep(self.dt)

    def stop(self):
        self.run_flag = False
        self.join(timeout=3)
        u = np.array(self.util) if self.util else np.array([0.0])
        m = np.array(self.mem) if self.mem else np.array([0.0])
        return dict(util_mean=float(u.mean()), util_peak=float(u.max()),
                    vram_smi_MB=float(m.max()), n_samples=len(self.util))


def kabsch_rmsd(P, Q):
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return float(np.sqrt(((Pc @ (U @ np.diag([1, 1, d]) @ Vt) - Qc) ** 2).sum(1).mean()))


def load_ts1x(pkl, n, start=0):
    """ts1x records -> list of dicts (R/P/ts ase.Atoms + DFT energies)."""
    from ase import Atoms
    recs = pickle.load(open(pkl, "rb"))[start:start + n]
    out = []
    for r in recs:
        Z = np.asarray(r["atomic_numbers"]).astype(int)
        out.append(dict(
            idx=r.get("index"), formula=r.get("formula"), Z=Z,
            natoms=int(r["natoms"]),
            R=Atoms(numbers=Z, positions=np.asarray(r["reactant"]["positions"], float)),
            P=Atoms(numbers=Z, positions=np.asarray(r["product"]["positions"], float)),
            TS=Atoms(numbers=Z, positions=np.asarray(r["transition_state"]["positions"], float)),
            ts=np.asarray(r["transition_state"]["positions"], float),
            e_react=float(r["reactant"]["energy"]),
            e_ts=float(r["transition_state"]["energy"])))
    return out


def n_imag_batch(calc, atoms_list, Z_list):
    """Mass-weighted FD Hessian -> per-structure n_imag (same as b67 pipeline freq stage)."""
    from ase.data import atomic_masses
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


def build_backend(name, model_path, device="cuda", dtype="float64", task="omol", **kw):
    """Backend registry. Unknown/unavailable backend -> raise (caller records SKIP)."""
    import torch
    dt = getattr(torch, dtype)
    if name == "uma":
        from maple.function.calculator.uma._uma_batch_calculator import UMABatchCalc
        return UMABatchCalc(model_path, device=device, dtype=dt, task=task,
                            fast_inference=False, **kw)
    if name == "mace_traced":
        from maple.function.calculator.mace._mace_batch_calculator import MACEBatchCalc
        return MACEBatchCalc(device=device, model="maceomol", model_path=model_path,
                             dtype=dt)
    if name == "mace_autograd":
        from maple.function.calculator.mace._mace_autograd_batch_calculator import (
            MACEAutogradBatchCalc)
        return MACEAutogradBatchCalc(model_path=model_path, model="maceoff23s",
                                     device=device, dtype=dt)
    raise ValueError(f"unknown backend '{name}' (registry: uma, mace_traced, mace_autograd)")


def env_info():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return dict(
        device=dev,
        gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        torch=torch.__version__,
        node=os.environ.get("SLURMD_NODENAME", os.uname().nodename),
        jobid=os.environ.get("SLURM_JOB_ID", "none"),
        alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>"),
        commit=os.environ.get("MAPLE_COMMIT", "unknown"),
    )


def run_record(*, bench, dispatcher, backend, B, N, rep, tag, params, wall_s,
               grad_equiv_total, forward_calls, wall_stages=None, ge_stages=None,
               gpu=None, vram_reserved_MB=None, vram_alloc_MB=None,
               science=None, extra=None):
    """The unified maple-bench-v1 run record. wall ALWAYS paired with grad-equiv."""
    rec = dict(schema=SCHEMA, bench=bench, dispatcher=dispatcher, backend=backend,
               B=B, N=N, rep=rep, tag=tag, params=params, env=env_info(),
               wall_s=float(wall_s),
               grad_equiv_total=int(grad_equiv_total),
               forward_calls=int(forward_calls),
               s_per_grad_equiv=(float(wall_s) / grad_equiv_total
                                 if grad_equiv_total else None),
               wall_stages=wall_stages or {}, grad_equiv_stages=ge_stages or {},
               gpu_util_mean=(gpu or {}).get("util_mean"),
               gpu_util_peak=(gpu or {}).get("util_peak"),
               vram_smi_MB=(gpu or {}).get("vram_smi_MB"),
               vram_reserved_MB=vram_reserved_MB, vram_alloc_MB=vram_alloc_MB,
               science=science or {}, extra=extra or {},
               timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return rec


def dump(rec, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name if name.endswith(".json") else name + ".json")
    json.dump(rec, open(p, "w"), indent=1, default=float)
    print(f"[bench] -> {p}", flush=True)
    return p


def spread_pct(vals):
    """Full-range spread in % of the mean (D-253 convention: 2 reps differing 3.8%)."""
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) < 2 or v.mean() == 0:
        return None
    return float(100.0 * (v.max() - v.min()) / v.mean())
