"""Shared instrumentation + schema for maple-bench (schema maple-bench-v1).

Metric conventions copied verbatim from bench_2026-07-14 (B_CLASS_BENCH) so all
2026-07-28 numbers are directly comparable with the July baselines:
  * grad-equiv: GradCounter wraps EVERY model-evaluation entry point the calculator
    exposes (_predict_forces and/or _forward, re-entrancy-guarded so nesting counts
    once) and sums the batch dim -- one MLIP forward over B structures == B
    single-gradient-equivalents (hardware-independent). Hooking only `_forward`
    silently missed UMA's whole FD-Hessian segment; see GradCounter's docstring.
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


# Per-backend model-evaluation choke points. NOT all backends call them `_forward`
# -- each of these was found by reading the backend's own Hessian/FD path:
#   UMA            _predict_forces   (_uma_batch_calculator.py:895 -- FD chunks call it
#                                     directly; _forward itself calls it at :505)
#   MACE traced    _forward_ef_      (_mace_batch_calculator.py:512-513 -- _efh_fd calls
#                                     it per DOF; _forward is a thin wrapper at :326)
#   MACE autograd  _model_energy     (_mace_autograd_batch_calculator.py:443 -- get_ef_gpu
#                                     calls it directly; its own :407-409 docstring states
#                                     that get_ef_gpu/_efh_fd/_efh_analytic/hvp* do NOT
#                                     route through _forward)
#   generic base   _forward          (batch_calculator_base.py:516-518)
# The re-entrancy guard makes it safe to hook all of them at once: nested calls are
# counted once, at the outermost hooked frame.
ENTRY_POINTS = ("_predict_forces", "_forward_ef_", "_model_energy", "_forward")


class GradCounter:
    """Exact DFT-gradient-equivalents: sums the batch dim of every model evaluation.

    Hooks EVERY entry point in ENTRY_POINTS that the calculator exposes, and uses a
    re-entrancy guard so a nested call is counted once, at the OUTERMOST hooked
    frame. Rationale (the emission bug this fixes):

      * `get_ef_gpu` -> `_forward` -> `_predict_forces`  (nested; counted once = B)
      * UMA's optimized FD Hessian `_get_efh_numerical` builds the +/- displaced
        replica chunks itself and calls `_predict_forces` DIRECTLY
        (_uma_batch_calculator.py:895) -- it never passes through `_forward`.
        Hooking only `_forward` therefore counted the ENTIRE Hessian segment as 0
        (measured 2 instead of the analytic 122 for B=2, natoms 10+10).
      * Backends without a `_predict_forces` (the generic base FD at
        batch_calculator_base.py:516-518) displace through `_forward` and were
        already counted correctly -- the guard leaves them unchanged.

    Note UMA's own diagnostic `self._fwd_count` (_uma_batch_calculator.py:341, bumped
    only at :493 inside `_forward`) has the SAME blind spot and also undercounts the
    Hessian segment.
    """

    def __init__(self, calc, entry_points=ENTRY_POINTS):
        self.calc = calc
        self.n = 0
        self.calls = 0
        self._depth = 0
        self.hooked = []
        self._orig = {}
        for name in entry_points:
            fn = getattr(calc, name, None)
            if fn is None or not callable(fn):
                continue
            self._orig[name] = fn
            setattr(calc, name, self._make(name))
            self.hooked.append(name)
        if not self.hooked:
            raise RuntimeError(
                f"GradCounter: {type(calc).__name__} exposes none of {entry_points} "
                "-- refusing to report grad-equivalents for an unhooked backend")
        # kept for the record: which entry points were actually instrumented
        self.orig = self._orig.get("_forward")

    def _make(self, name):
        def wrapper(*a, **k):
            outer = (self._depth == 0)
            self._depth += 1
            try:
                out = self._orig[name](*a, **k)
            finally:
                self._depth -= 1
            if outer:
                E = out[0] if isinstance(out, (tuple, list)) else out
                self.calls += 1
                try:
                    self.n += int(E.shape[0])
                except Exception:
                    self.n += 1
            return out
        return wrapper

    def take(self):
        n, c = self.n, self.calls
        self.n = 0
        self.calls = 0
        return n, c

    def unhook(self):
        for name, fn in self._orig.items():
            setattr(self.calc, name, fn)


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


def load_ts1x(pkl, n, start=0, nat_min=None, nat_max=None, tier_sample="head"):
    """ts1x records -> list of dicts (R/P/ts ase.Atoms + DFT energies).

    nat_min/nat_max select a system-size tier BEFORE slicing (dimension (d)):
    without them the behaviour is the plain contiguous [start:start+n] slice, so
    every earlier record stays reproducible.
    """
    from ase import Atoms
    recs = pickle.load(open(pkl, "rb"))
    if nat_min is not None or nat_max is not None:
        lo = -1 if nat_min is None else int(nat_min)
        hi = 10 ** 9 if nat_max is None else int(nat_max)
        recs = [r for r in recs if lo <= int(r["natoms"]) <= hi]
        # D-283: ts1x_data_all is ordered by size, so the plain head slice of a
        # filtered tier returns only its LOWER EDGE (tier 17-23 came back as
        # natoms 17 only). 'spread' takes an evenly strided sample so the tier
        # actually spans its range. 'head' reproduces the earlier records.
        if tier_sample == "spread" and len(recs) > n > 0:
            step = len(recs) / float(n)
            recs = [recs[min(len(recs) - 1, int(i * step))] for i in range(n)]
    recs = recs[start:start + n]
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
                            fast_inference=bool(kw.pop("fast_inference", False)), **kw)
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
               science=None, extra=None, hessian_mode=None,
               counter_status="UNVERIFIED", counter_detail=None):
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
    # A broken counter must NOT emit a number, and must NOT emit a bare None
    # either: None reads as "this cell has no data", while the truth is "the
    # instrument for this cell is broken". grad_equiv_status carries that apart.
    trustworthy = (counter_status == "VERIFIED")
    ge_status = ("OK" if trustworthy else
                 ("UNAVAILABLE(counter %s)" % counter_status))
    rec = dict(schema=SCHEMA, bench=bench, dispatcher=dispatcher, backend=backend,
               B=B, N=N, rep=rep, tag=tag, params=params, env=env_info(),
               hessian_mode=hessian_mode,
               counter_status=counter_status, counter_detail=(counter_detail or {}),
               grad_equiv_status=ge_status,
               wall_s=float(wall_s),
               grad_equiv_total=(int(grad_equiv_total) if trustworthy else None),
               grad_equiv_measured_raw=int(grad_equiv_total),
               forward_calls=int(forward_calls),
               s_per_grad_equiv=((float(wall_s) / grad_equiv_total)
                                 if (trustworthy and grad_equiv_total) else None),
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


def verify_counter_inline(build_fn, mols_fn, needs_hessian=False):
    """Cheap per-run self-certification of the counter (~0.1-2 s).

    Every record must be able to say whether its own grad-equivalents are
    trustworthy. Returns (status, detail) with status in:
      VERIFIED  known answers hit (B=1 -> 1, B=2 -> 2, and the FD Hessian form
                when needs_hessian)
      BROKEN    the counter miscounts on THIS backend -> the caller must publish
                grad_equiv as UNAVAILABLE, never as 0 and never as a plain None
                (a None reads as 'no data'; this is 'the instrument is broken')
      SKIP      the probe itself could not run (no model / no device)
    """
    detail = {}
    try:
        for B in (1, 2):
            calc = build_fn()
            ctr = GradCounter(calc)
            detail.setdefault("hooked", list(ctr.hooked))
            mols = mols_fn(B)
            calc.prepare([m.copy() for m in mols])
            calc.get_ef_gpu()
            got, _ = ctr.take()
            detail[f"sp_B{B}"] = dict(expected=B, measured=int(got))
            if int(got) != B:
                return "BROKEN", detail
        if needs_hessian:
            calc = build_fn()
            ctr = GradCounter(calc)
            mols = mols_fn(2)
            nat = [len(m) for m in mols]
            dof, nmax_a, B = 3 * sum(nat), max(nat), 2
            calc.prepare([m.copy() for m in mols])
            calc.get_efh_gpu()
            got, _ = ctr.take()
            forms = {"central_2xDOF": 2 * dof, "central_2xDOF_plus_base": 2 * dof + B,
                     "forward_DOF_plus_base": dof + B,
                     "central_padded_2x3xNmaxA_xB_plus_base": 2 * 3 * nmax_a * B + B}
            match = [k for k, v in forms.items() if v == int(got)]
            detail["fd_hessian"] = dict(measured=int(got), analytic_forms=forms,
                                        matched_form=(match[0] if match else None))
            if not match:
                return "BROKEN", detail
        return "VERIFIED", detail
    except Exception as e:
        detail["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        return "SKIP", detail


UNCOUNTED_WORK_FACTOR = 5.0


def stage_share_gate(wall_stages, ge_stages, factor=UNCOUNTED_WORK_FACTOR,
                     min_wall_share=0.02):
    """STANDING EMISSION GATE: per-stage wall share vs grad-equivalent share.

    For a forward-bound stage the two shares must be the same order of magnitude.
    A stage that burns X% of the wall while reporting X/20 % of the gradient
    equivalents is doing work the counter cannot see. This gate is what SHOULD
    have caught the July freq-stage hole automatically:

        NEB   43.0% wall / 87.5% gE  -> ratio 0.49  ok
        P-RFO 48.7% wall / 12.1% gE  -> ratio 4.0   borderline
        freq   8.3% wall /  0.41% gE -> ratio 20.3  UNCOUNTED_WORK_SUSPECTED

    Stages under `min_wall_share` of the wall are SKIPped (too small to judge).
    Returns PASS / UNCOUNTED_WORK_SUSPECTED / SKIP plus the per-stage table.
    """
    if not wall_stages or not ge_stages:
        return dict(name="stage_share_gate", status="SKIP",
                    note="no per-stage accounting for this dispatcher")
    wtot = float(sum(wall_stages.values()))
    gtot = float(sum(ge_stages.values()))
    if wtot <= 0 or gtot <= 0:
        return dict(name="stage_share_gate", status="SKIP", note="empty stage totals")
    rows, flagged = [], []
    for k in wall_stages:
        ws = float(wall_stages[k]) / wtot
        gs = float(ge_stages.get(k, 0)) / gtot
        if ws < min_wall_share:
            rows.append(dict(stage=k, wall_share=ws, ge_share=gs, ratio=None,
                             status="SKIP", note="wall share below %.0f%%"
                                                 % (100 * min_wall_share)))
            continue
        ratio = (ws / gs) if gs > 0 else float("inf")
        st = "UNCOUNTED_WORK_SUSPECTED" if ratio >= factor else "PASS"
        if st != "PASS":
            flagged.append(k)
        rows.append(dict(stage=k, wall_share=ws, ge_share=gs, ratio=ratio, status=st))
    applicable = [r for r in rows if r["status"] != "SKIP"]
    status = ("SKIP" if not applicable
              else ("UNCOUNTED_WORK_SUSPECTED" if flagged else "PASS"))
    return dict(name="stage_share_gate", status=status, factor=factor,
                flagged_stages=flagged, stages=rows,
                note=("a stage whose wall share exceeds its grad-equiv share by "
                      ">=%gx is doing work the counter cannot see" % factor))


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
class PathProbe:
    """Records WHICH Hessian implementation actually executed.

    Settles 'is the counter blind' (a) vs 'is it secretly running the double
    backward' (b) by direct observation instead of arithmetic inference. Hooks the
    numerical FD body, the VRAM-adaptive FD body, the legacy FD body and the
    autograd body, and counts entries into each.
    """

    TARGETS = ("_get_efh_numerical", "_get_efh_numerical_auto",
               "_get_efh_gpu_legacy", "_efh_gpu_autograd", "_efh_analytic",
               "_efh_fd")

    def __init__(self, calc):
        self.calc = calc
        self.counts = {}
        self._orig = {}
        for name in self.TARGETS:
            fn = getattr(calc, name, None)
            if fn is None or not callable(fn):
                continue
            self.counts[name] = 0
            self._orig[name] = fn
            setattr(calc, name, self._make(name))

    def _make(self, name):
        def wrapper(*a, **k):
            self.counts[name] += 1
            return self._orig[name](*a, **k)
        return wrapper

    def report(self):
        fired = {k: v for k, v in self.counts.items() if v}
        fd = sum(v for k, v in fired.items()
                 if k in ("_get_efh_numerical", "_get_efh_numerical_auto",
                          "_get_efh_gpu_legacy", "_efh_fd"))
        ag = sum(v for k, v in fired.items()
                 if k in ("_efh_gpu_autograd", "_efh_analytic"))
        path = ("finite-difference" if fd and not ag else
                "autograd/double-backward" if ag and not fd else
                "mixed" if ag and fd else "none")
        return dict(hooked=sorted(self.counts), fired=fired,
                    fd_entries=fd, autograd_entries=ag, executed_path=path)

    def unhook(self):
        for name, fn in self._orig.items():
            setattr(self.calc, name, fn)


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
    hooked = None

    def sp_case(B):
        name = f"sp_B{B}"
        try:
            calc = build_fn()
            ctr = GradCounter(calc)
            nonlocal hooked
            hooked = list(ctr.hooked)
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
            probe = PathProbe(calc)
            B = 2
            mols = mols_fn(B)
            nat = [len(m) for m in mols]
            dof = 3 * sum(nat)
            calc.prepare([m.copy() for m in mols])
            calc.get_efh_gpu(mode=mode)
            got, calls = ctr.take()
            path = probe.report()
            nmax_a = max(nat)
            forms = {"central_2xDOF": 2 * dof,
                     "central_2xDOF_plus_base": 2 * dof + B,
                     "forward_DOF_plus_base": dof + B,
                     # MACE-style padded loop: `for k in range(3*nmax_atoms)` with
                     # both +/- forwards covering ALL B structures each iteration
                     "central_padded_2x3xNmaxA_xB_plus_base": 2 * 3 * nmax_a * B + B}
            match = [k for k, v in forms.items() if v == got]
            if mode in ("autograd", "analytic"):
                cases.append(dict(name=name, status="SKIP", measured=int(got),
                                  forward_calls=int(calls), analytic_forms=forms,
                                  executed_path=path,
                                  note=("double-backward work is invisible to a "
                                        "forward counter -> no known answer; this is "
                                        "why s/grad-equiv is INCOMPARABLE across modes")))
            else:
                cases.append(dict(name=name,
                                  status=("PASS" if match else "FAIL"),
                                  measured=int(got), forward_calls=int(calls),
                                  matched_form=(match[0] if match else None),
                                  analytic_forms=forms, natoms=nat, dof=dof,
                                  executed_path=path))
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
        # Pass the SAME mode the full-Hessian case used. With mode=None some backends
        # auto-select the seeded analytic Hessian (traced MACE does when the model can
        # double-backward), which is one forward + a double backward -- no known answer
        # for a forward counter, so comparing it against the FD forms below reports a
        # spurious FAIL. See D-269.
        _mode = hessian_modes[0] if hessian_modes else "numerical"
        calc.get_efh_gpu(movable_masks=mov, mode=_mode)
        got, calls = ctr.take()
        nmov_max = max(len(m) for m in mov)
        forms = {"central_2xDOFmov": 2 * dof_mov,
                 "central_2xDOFmov_plus_base": 2 * dof_mov + B,
                 "forward_DOFmov_plus_base": dof_mov + B,
                 "central_padded_2x3xNmovMax_xB_plus_base": 2 * 3 * nmov_max * B + B}
        match = [k for k, v in forms.items() if v == got]
        cases.append(dict(name="hess_partial_movable",
                          status=("PASS" if match else "FAIL"),
                          measured=int(got), forward_calls=int(calls),
                          matched_form=(match[0] if match else None),
                          analytic_forms=forms, movable_atoms=[len(m) for m in mov],
                          hessian_mode=_mode))
    except Exception as e:
        cases.append(dict(name="hess_partial_movable", status="SKIP",
                          note=f"{type(e).__name__}: {str(e)[:140]}"))

    st = [c["status"] for c in cases]
    verdict = "FAIL" if "FAIL" in st else ("SKIP" if all(s == "SKIP" for s in st) else "OK")
    return dict(name="counter_self_test", backend=backend_name, verdict=verdict,
                hooked_entry_points=hooked, cases=cases,
                note=("counter verified against known answers; s/grad-equiv is "
                      "trustworthy for this backend" if verdict == "OK" else
                      "counter MISCOUNTS -> do NOT report s/grad-equiv for this backend"))
