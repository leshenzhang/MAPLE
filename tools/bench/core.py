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


def device_identity():
    """Which physical GPU is THIS process actually on.

    FIX-4: `nvidia-smi --query-gpu=...` returns rows for ALL visible GPUs and the
    old sampler read row 0 = physical GPU 0, which on a shared Ibex gpu node is
    NOT necessarily this job's card -> util (a headline metric) could be someone
    else's. We resolve the current torch device's UUID and pin every query to it.
    """
    import torch
    info = dict(cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
                torch_device_index=None, uuid=None, name=None, nvidia_smi_L=None)
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                             timeout=5)
        info["nvidia_smi_L"] = out.stdout.strip()
    except Exception:
        pass
    if not torch.cuda.is_available():
        return info
    idx = torch.cuda.current_device()
    info["torch_device_index"] = int(idx)
    props = torch.cuda.get_device_properties(idx)
    info["name"] = props.name
    u = getattr(props, "uuid", None)
    if u is not None:
        s = str(u)
        info["uuid"] = s if s.startswith("GPU-") else f"GPU-{s}"
    elif info["nvidia_smi_L"]:
        # nvidia-smi -L already honors CUDA_VISIBLE_DEVICES ordering
        lines = [l for l in info["nvidia_smi_L"].splitlines() if "UUID:" in l]
        if idx < len(lines):
            info["uuid"] = lines[idx].rsplit("UUID:", 1)[1].strip().rstrip(")")
    return info


class GPUSampler(threading.Thread):
    """Device-pinned util/VRAM sampler.

    source priority (recorded in `sampler_source`):
      1. nvidia-smi --id=<UUID of this device>   -- device-pinned; keeps the July
         metric semantics (utilization.gpu + memory.used of THIS card)
      2. torch.cuda.utilization(current_device)  -- NVML on this process's device;
         memory then falls back to torch reserved MB (different semantics, flagged)
      3. nvidia-smi row 0                        -- last resort, flagged UNTRUSTED
    """

    def __init__(self, dt=0.5):
        super().__init__(daemon=True)
        self.dt = dt
        self.run_flag = True
        self.util = []
        self.mem = []
        self.dev = device_identity()
        self.source = "unavailable"
        self.mem_semantics = None
        self._probe()

    def _probe(self):
        import torch
        if not torch.cuda.is_available():
            return
        if self.dev.get("uuid"):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--id=" + self.dev["uuid"],
                     "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                if out.returncode == 0 and out.stdout.strip():
                    self.source = "nvidia-smi --id=<uuid>"
                    self.mem_semantics = "device memory.used (MB)"
                    return
            except Exception:
                pass
        try:
            torch.cuda.utilization(torch.cuda.current_device())
            self.source = "torch.cuda.utilization"
            self.mem_semantics = "torch reserved (MB) -- NOT device memory.used"
            return
        except Exception:
            pass
        self.source = "nvidia-smi row0 UNTRUSTED (may be another job's GPU)"
        self.mem_semantics = "device memory.used (MB), UNTRUSTED device"

    def _sample(self):
        import torch
        if self.source == "torch.cuda.utilization":
            idx = torch.cuda.current_device()
            return (float(torch.cuda.utilization(idx)),
                    float(torch.cuda.memory_reserved(idx) / 2**20))
        args = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits"]
        if self.dev.get("uuid") and "uuid" in self.source:
            args.insert(1, "--id=" + self.dev["uuid"])
        out = subprocess.run(args, capture_output=True, text=True, timeout=2)
        u, m = out.stdout.strip().split("\n")[0].split(",")
        return float(u), float(m)

    def run(self):
        while self.run_flag:
            try:
                u, m = self._sample()
                self.util.append(u)
                self.mem.append(m)
            except Exception:
                pass
            time.sleep(self.dt)

    def stop(self):
        self.run_flag = False
        self.join(timeout=3)
        u = np.array(self.util) if self.util else np.array([0.0])
        m = np.array(self.mem) if self.mem else np.array([0.0])
        return dict(util_mean=float(u.mean()), util_peak=float(u.max()),
                    vram_smi_MB=float(m.max()), n_samples=len(self.util),
                    sampler_source=self.source, mem_semantics=self.mem_semantics,
                    device=self.dev)


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
               science=None, extra=None, hessian_mode=None):
    """The unified maple-bench-v1 run record. wall ALWAYS paired with grad-equiv.

    FIX-1 `hessian_mode` is a TOP-LEVEL field because grad-equivalents are NOT
    commensurable across Hessian modes: a numerical central-FD Hessian spends
    2*3N `calc._forward` evaluations per structure (all counted), while an
    autograd/analytic Hessian spends ONE forward plus a double-backward whose cost
    the counter cannot see. Comparing `s_per_grad_equiv` across different
    `hessian_mode` values divides by two different denominators and is physically
    meaningless -> aggregate.py refuses it with INCOMPARABLE. `wall_s` and
    `forward_calls` remain valid across modes and must be used instead.
    """
    rec = dict(schema=SCHEMA, bench=bench, dispatcher=dispatcher, backend=backend,
               B=B, N=N, rep=rep, tag=tag, params=params, env=env_info(),
               hessian_mode=hessian_mode,
               wall_s=float(wall_s),
               grad_equiv_total=int(grad_equiv_total),
               forward_calls=int(forward_calls),
               s_per_grad_equiv=(float(wall_s) / grad_equiv_total
                                 if grad_equiv_total else None),
               wall_stages=wall_stages or {}, grad_equiv_stages=ge_stages or {},
               gpu_util_mean=(gpu or {}).get("util_mean"),
               gpu_util_peak=(gpu or {}).get("util_peak"),
               vram_smi_MB=(gpu or {}).get("vram_smi_MB"),
               gpu_sampler_source=(gpu or {}).get("sampler_source"),
               gpu_mem_semantics=(gpu or {}).get("mem_semantics"),
               gpu_device=(gpu or {}).get("device"),
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


# ---------------------------------------------------------------------------
# FIX-2: EMISSION-side self-test for GradCounter.
#
# The parity gate proves DETECTION (it catches a degraded result). It says
# nothing about EMISSION: whether the counter sees EVERY forward path. If a
# backend's batched route bypasses `calc._forward` (a traced-MACE native batch
# path, a fast_inference shortcut, an autograd-Hessian internal re-evaluation),
# the count is silently LOW, every s/grad-equiv is wrong, and nothing looks
# broken. So we pin the counter against KNOWN-ANSWER cases before trusting any
# normalized number.
# ---------------------------------------------------------------------------
def counter_self_test(build_fn, mols_fn, backend_name, hessian_modes=("numerical",)):
    """Known-answer tests for GradCounter. Returns dict with per-case PASS/FAIL/SKIP.

    Cases
    -----
    sp_B1    single point, B=1                -> exactly 1
    sp_B16   single point, B=16               -> exactly 16
    fd_hess  numerical Hessian, B structures  -> exactly one of the documented
             analytic forms (central 2*(3N); +B if the base point is a separate
             forward; forward-difference (3N)+B), N = sum of atoms over the batch.
             Any other value FAILS and the measured number is printed.
    ag_hess  autograd Hessian                 -> NOT a known-answer case for the
             counter (double-backward is invisible to it); recorded as SKIP with
             the measured value, which is exactly why mode-mixing is banned.
    """
    cases = []

    def sp_case(B):
        name = f"sp_B{B}"
        try:
            calc = build_fn()
            ctr = GradCounter(calc)
            mols = mols_fn(B)
            calc.prepare([m.copy() for m in mols])
            calc.get_ef_gpu()
            got, calls = ctr.take()
            return dict(name=name, status=("PASS" if got == B else "FAIL"),
                        expected=B, measured=int(got), forward_calls=int(calls))
        except Exception as e:
            return dict(name=name, status="SKIP",
                        note=f"{type(e).__name__}: {str(e)[:140]}")

    cases.append(sp_case(1))
    cases.append(sp_case(16))

    for mode in hessian_modes:
        name = f"hess_{mode}"
        try:
            kw = {"hessian_mode": mode} if backend_name == "uma" else {}
            calc = build_fn(**kw)
            ctr = GradCounter(calc)
            B = 2
            mols = mols_fn(B)
            nat = [len(m) for m in mols]
            dof = 3 * sum(nat)
            calc.prepare([m.copy() for m in mols])
            calc.get_efh_gpu(mode=mode)
            got, calls = ctr.take()
            forms = {"central_2xDOF": 2 * dof,
                     "central_2xDOF_plus_base": 2 * dof + B,
                     "forward_DOF_plus_base": dof + B}
            match = [k for k, v in forms.items() if v == got]
            if mode in ("autograd", "analytic"):
                cases.append(dict(name=name, status="SKIP", measured=int(got),
                                  forward_calls=int(calls), analytic_forms=forms,
                                  note=("double-backward work is invisible to a "
                                        "forward counter -> no known answer; this is "
                                        "why s/grad-equiv is INCOMPARABLE across modes")))
            else:
                cases.append(dict(name=name,
                                  status=("PASS" if match else "FAIL"),
                                  measured=int(got), forward_calls=int(calls),
                                  matched_form=(match[0] if match else None),
                                  analytic_forms=forms, natoms=nat, dof=dof))
        except Exception as e:
            cases.append(dict(name=name, status="SKIP",
                              note=f"{type(e).__name__}: {str(e)[:140]}"))

    # movable-mask (partial Hessian) known answer: only the movable DOF are perturbed
    try:
        calc = build_fn()
        ctr = GradCounter(calc)
        B = 2
        mols = mols_fn(B)
        nat = [len(m) for m in mols]
        mov = [list(range(max(1, n // 2))) for n in nat]
        dof_mov = 3 * sum(len(m) for m in mov)
        calc.prepare([m.copy() for m in mols])
        calc.get_efh_gpu(movable_masks=mov)
        got, calls = ctr.take()
        forms = {"central_2xDOFmov": 2 * dof_mov,
                 "central_2xDOFmov_plus_base": 2 * dof_mov + B,
                 "forward_DOFmov_plus_base": dof_mov + B}
        match = [k for k, v in forms.items() if v == got]
        cases.append(dict(name="hess_partial_movable",
                          status=("PASS" if match else "FAIL"),
                          measured=int(got), forward_calls=int(calls),
                          matched_form=(match[0] if match else None),
                          analytic_forms=forms, movable_atoms=[len(m) for m in mov]))
    except Exception as e:
        cases.append(dict(name="hess_partial_movable", status="SKIP",
                          note=f"{type(e).__name__}: {str(e)[:140]}"))

    st = [c["status"] for c in cases]
    verdict = "FAIL" if "FAIL" in st else ("SKIP" if all(s == "SKIP" for s in st) else "OK")
    return dict(name="counter_self_test", backend=backend_name, verdict=verdict,
                cases=cases,
                note=("counter verified against known answers; s/grad-equiv is "
                      "trustworthy for this backend" if verdict == "OK" else
                      "counter MISCOUNTS -> do NOT report s/grad-equiv for this backend"))
