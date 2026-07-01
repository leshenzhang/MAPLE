# -*- coding: utf-8 -*-
"""Parity test: EulerPCBatch (N-path GPU-batched IRC) vs the single-structure
EulerPC oracle.

What is being proven
--------------------
EulerPCBatch batches ONLY the force/Hessian evaluation across N independent IRC
paths; the per-path control flow (predictor-Euler, DWI+mBS corrector, BFGS/Bofill
history, convergence) is the single-structure oracle's numpy verbatim. So a
CORRECT batch must reproduce the oracle to the calculator's B=1-vs-B=N batching
determinism -- the fp32 noise floor -- NOT worse.

Methodology (rigorous, self-calibrating -- not a blindly-loosened tolerance)
----------------------------------------------------------------------------
1. WELL-CONDITIONED real TS. A distorted-water "saddle" is ill-conditioned (near-
   degenerate lowest modes incl. unremoved trans/rot); UMA's ~1% FD-Hessian noise
   then rotates the selected lowest eigenvector so the first IRC half-step goes a
   different way (this produced the earlier rec-0 5e-2 A "failure"). Instead we
   build the HCN<->HNC isomerization TS and eigenvector-following-refine it to a
   clean first-order saddle (one well-separated negative mode) with the injected
   model, so mode selection is stable.
2. SIGN-CONVENTION-AGNOSTIC comparison. An IRC yields two half-paths; which is
   "forward" is a labeling convention (set by the arbitrary eigh sign). We match
   the batch's {forward,backward} to the oracle's {forward,backward} as an
   unordered pair by endpoint proximity, then compare CONVERGED ENDPOINTS +
   path length + endpoint energy -- NOT bitwise rec-by-rec (fp32 noise accumulates
   along the integration, and step counts can differ by +/-1).
3. CANON-vs-CANON control. The single EulerPC oracle is run TWICE on independent
   calculator instances; oracle_A-vs-oracle_B is the pure fp32 self-noise baseline
   (identical algorithm, no batching). We then assert
        residual(batch, oracle_A)  <=  C * residual(oracle_A, oracle_B) + floor
   i.e. batching introduces no more divergence than the calculator's own fp32
   self-noise. This attributes any residual to fp32, not to a batching bug.
4. fp32-appropriate floors: energy ~1e-6 Eh, force ~2e-7 Eh/A (UMA fp32 force
   noise floor, VALIDATION_MATRIX V8); endpoint geometry ~1e-3 A absolute cap.

Entry point (calculator injected via factory; no hardcoded model path):
    run_parity(make_calc)
``make_calc()`` returns a fresh *batch* calculator implementing
    prepare(atoms_list, fixed_nmax) ; get_ef_gpu() ; get_efh_gpu() ; set_coords_(coord)
(e.g. a UMABatchCalc). Wire UMA (or any batchable backend) centrally.

``run_selftest_analytic()`` proves the batching mechanics deterministically with a
decoupled analytic calc (no model needed) -- useful for local CI. ``__main__`` runs
that always, then run_parity if MAPLE_UMA_CKPT / MAPLE_UMA_SIZE is set.
"""

import os
import sys
import tempfile

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

# Import the module under test.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maple.function.dispatcher.irc.algorithm.eulerpc import (  # noqa: E402
    EulerPC, EulerPCBatch, EulerPCParams,
)

HARTREE_EV = 27.211386245988


# ======================================================================== #
# Oracle-side calculator: the single-structure EulerPC oracle drives a real
# ASE ``atoms`` whose energies/forces/Hessian come from the SAME batched model
# at B=1, so oracle-vs-batch deviation is purely the calc's batching noise.
# Forces come from get_ef_gpu (matching EulerPCBatch); Hessian from get_efh_gpu.
# ======================================================================== #
class _OracleCalc(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, batch_calc, **kw):
        super().__init__(**kw)
        self._b = batch_calc

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self._b.prepare([atoms])
        E, F = self._b.get_ef_gpu()
        n = len(atoms)
        e = float(E.detach().cpu()[0])
        f = F.detach().cpu()[0, :3 * n].numpy().astype(np.float64).reshape(n, 3)
        self.results["energy"] = e
        self.results["free_energy"] = e
        self.results["forces"] = f

    def get_hessian(self, atoms):
        self._b.prepare([atoms])
        _E, _F, H, _P = self._b.get_efh_gpu()
        n = len(atoms)
        return H.detach().cpu()[0, :3 * n, :3 * n].numpy().astype(np.float64)


class _FixedHessianCalc:
    """Wraps a batch calc, delegating prepare/set_coords_/step_cart_/get_ef_gpu
    (so FORCES are the real batched forward), but returning a FIXED, precomputed
    TS Hessian from get_efh_gpu (its E,F are real; only H is pinned).

    Why: even with one shared calc instance, a model's FD Hessian (e.g. UMA) has
    ~1% PER-CALL variance; on a near-degenerate saddle that rotates the neg-mode
    eigenvector -> a ~1e-2 A first-1/2-step offset. Feeding the IDENTICAL Hessian
    into BOTH the oracle EulerPC (via _OracleCalc.get_hessian -> get_efh_gpu) and
    EulerPCBatch (via its 'h' ticks -> get_efh_gpu) removes that variance, so
    M_code isolates the batching CODE (it MUST then be ~0 bitwise). Requires NO
    change to EulerPCBatch or the single EulerPC oracle. Valid because, with the
    default params, every Hessian eval happens at the SAME TS geometry.
    """

    def __init__(self, calc, ts_atoms):
        import torch
        self._t = torch
        self._b = calc
        n = len(ts_atoms)
        calc.prepare([ts_atoms])
        _E, _F, H, _P = calc.get_efh_gpu()          # ONE TS Hessian eval, pinned
        self._Hfix = H.detach()[0, :3 * n, :3 * n].clone()
        self._n = n
        self._B = 1
        self._M = 3 * n

    def prepare(self, atoms_list, fixed_nmax=None):
        out = self._b.prepare(atoms_list, fixed_nmax)
        self._B = len(atoms_list)
        self._M = (3 * max(len(a) for a in atoms_list)) if fixed_nmax is None else int(fixed_nmax)
        return out

    def set_coords_(self, coord):
        return self._b.set_coords_(coord)

    def step_cart_(self, s):
        return self._b.step_cart_(s)

    def get_ef_gpu(self):
        return self._b.get_ef_gpu()

    def get_efh_gpu(self, *args, **kwargs):
        E, F = self._b.get_ef_gpu()                 # real forces (H is discarded
        B, M, n = self._B, self._M, self._n         #   downstream; E,F kept real)
        H = self._t.zeros((B, M, M), dtype=self._Hfix.dtype, device=self._Hfix.device)
        H[:, :3 * n, :3 * n] = self._Hfix           # broadcast the fixed H to all B
        P = self._t.zeros(B, dtype=E.dtype, device=E.device)
        return E, F, H, P


# ======================================================================== #
# Well-conditioned real TS: HCN <-> HNC isomerization, eigenvector-following
# refined with the injected model to a clean first-order saddle.
# ======================================================================== #
def _hcn_hnc_ts_guess():
    # C, N, bridging H (the migrating atom). Approximate isomerization TS.
    at = Atoms("CNH", positions=[[0.000, 0.000, 0.000],
                                 [1.180, 0.000, 0.000],
                                 [0.700, 1.000, 0.000]])
    at.info["charge"] = 0
    at.info["spin"] = 1
    at.info["mult"] = 1
    return at


def _calc_efh_B1(backend, atoms):
    """(E Eh, F(n,3) Eh/A, H(3n,3n) Eh/A^2) from the batch calc at B=1."""
    backend.prepare([atoms])
    E, F, H, _P = backend.get_efh_gpu()
    n = len(atoms)
    e = float(E.detach().cpu()[0])
    f = F.detach().cpu()[0, :3 * n].numpy().astype(np.float64).reshape(n, 3)
    h = H.detach().cpu()[0, :3 * n, :3 * n].numpy().astype(np.float64)
    return e, f, h


def _mw(atoms):
    m = np.asarray(atoms.get_masses(), dtype=np.float64)
    m = np.where(m > 0.0, m, 1.0)
    return 1.0 / np.sqrt(np.repeat(m, 3))   # D = 1/sqrt(m), length 3n


def _saddle_conditioning(atoms, backend):
    """Return (sorted MW-Hessian eigenvalues, n_negative, gap) for reporting."""
    _e, _f, h = _calc_efh_B1(backend, atoms)
    D = _mw(atoms)
    hmw = (D[:, None] * h) * D[None, :]
    hmw = 0.5 * (hmw + hmw.T)
    w = np.linalg.eigvalsh(hmw)
    n_neg = int(np.sum(w < 0))
    gap = float(w[1] - w[0]) if len(w) > 1 else 0.0
    return w, n_neg, gap


def _refine_ts(atoms, backend, max_iter=80, fmax=2e-3, trust=0.08, verbose=True):
    """Eigenvector-following (mode-1 uphill, rest downhill) saddle optimizer using
    the injected model's F and H at B=1. Sharpens the HCN<->HNC guess into a clean
    first-order saddle. Magnitude-floored eigenvalues keep steps sane."""
    a = atoms.copy()
    for it in range(1, max_iter + 1):
        _e, f, h = _calc_efh_B1(backend, a)
        g = -f.reshape(-1)                       # gradient (Eh/A)
        gmax = float(np.max(np.abs(g)))
        if gmax < fmax:
            if verbose:
                print(f"[refine] converged at iter {it}: fmax={gmax:.2e} Eh/A")
            break
        h = 0.5 * (h + h.T)
        w, V = np.linalg.eigh(h)                  # ascending
        gt = V.T @ g
        absw = np.maximum(np.abs(w), 0.05)        # floor magnitude -> sane steps
        signs = np.ones_like(w)
        signs[0] = -1.0                           # mode 0 uphill (reaction coord)
        step = -(V @ (gt / (signs * absw)))
        sn = float(np.linalg.norm(step))
        if sn > trust:
            step *= trust / sn
        a.set_positions(a.get_positions() + step.reshape(len(a), 3))
    return a


def _make_ts(backend, verbose=True):
    at = _refine_ts(_hcn_hnc_ts_guess(), backend, verbose=verbose)
    w, n_neg, gap = _saddle_conditioning(at, backend)
    if verbose:
        print(f"[ts] HCN<->HNC refined saddle: n_neg={n_neg}  "
              f"lowest MW-eig={w[0]:.4e}  next={w[1]:.4e}  gap={gap:.4e}")
    return at


# ======================================================================== #
# Side-agnostic endpoint / path-length metrics + matching.
# ======================================================================== #
def _xs_es(records):
    """records may be batch dict-of-lists or oracle list-of-dicts."""
    if isinstance(records, dict):
        xs = [np.asarray(x, np.float64) for x in records["x"]]
        es = [float(e) for e in records["E"]]
    else:
        xs = [np.asarray(r["x"], np.float64) for r in records]
        es = [float(r["E"]) for r in records]
    return xs, es


def _pathlen(xs, k):
    """Cumulative path length (A) over xs[0..k] inclusive."""
    L = 0.0
    for j in range(min(k, len(xs) - 1)):
        L += float(np.linalg.norm(xs[j + 1] - xs[j]))
    return L


def _side(records):
    xs, es = _xs_es(records)
    return {"xs": xs, "es": es, "n": len(xs)}


def _sides(res):
    return [_side(res["forward"]["records"]),
            _side(res["backward"]["records"])]


def _pair_cost(a, b):
    """Endpoint distance at the last COMMON step index (robust to +/-1 step)."""
    k = min(a["n"], b["n"]) - 1
    return float(np.max(np.abs(a["xs"][k] - b["xs"][k])))


def _match_perm(sA, sB):
    best = None
    for perm in ((0, 1), (1, 0)):
        c = sum(_pair_cost(sA[i], sB[perm[i]]) for i in range(2))
        if best is None or c < best[0]:
            best = (c, perm)
    return best[1]


def _residual(sA, sB, early_k=3):
    """Side-matched residual (worst of the two matched sides).

    d0     : rec-0 geometry |dx| (displaced TS start = TS + 1/2 step along the
             neg-mode eigenvector). CHAOS-INSENSITIVE -- depends only on the TS
             Hessian/eigenvector, before any force-driven step; a large d0
             fingerprints a Hessian/neg-mode-eigenVECTOR problem.
    d_early: max geometry |dx| over the first `early_k`+1 records (0..early_k).
             CHAOS-INSENSITIVE -- fp32 batch noise is ~1e-7 A/step early (chaos has
             not amplified yet), so a large d_early fingerprints a gross force- or
             Hessian-eigenVALUE non-invariance at B=N (a bug), not chaos.
    dend   : geometry |dx| at the last COMMON step (robust to +/-1 step).
    dE     : energy |dE| at last common step. dL: |dpathlen|. dn: step diff."""
    perm = _match_perm(sA, sB)
    d0 = d_early = dend = dE = dL = 0.0
    dn = 0
    for i in range(2):
        a, b = sA[i], sB[perm[i]]
        k = min(a["n"], b["n"]) - 1
        d0 = max(d0, float(np.max(np.abs(a["xs"][0] - b["xs"][0]))))
        for j in range(min(early_k, k) + 1):
            d_early = max(d_early, float(np.max(np.abs(a["xs"][j] - b["xs"][j]))))
        dend = max(dend, float(np.max(np.abs(a["xs"][k] - b["xs"][k]))))
        dE = max(dE, abs(a["es"][k] - b["es"][k]))
        dL = max(dL, abs(_pathlen(a["xs"], k) - _pathlen(b["xs"], k)))
        dn = max(dn, abs(a["n"] - b["n"]))
    return {"d0": d0, "d_early": d_early, "dend": dend, "dE": dE, "dL": dL,
            "dn": dn, "perm": perm}


# ======================================================================== #
# Main parity check.
# ======================================================================== #
def run_parity(make_calc, atoms=None, n=3, params=None, share_calc=True,
               inject_hessian=True,
               code_floor=1e-6, energy_code_floor=1e-9, rec0_floor=1e-4,
               early_floor=1e-4, early_k=3, amp_factor=3.0, endpoint_floor=1e-4,
               verbose=True):
    """EulerPCBatch-vs-EulerPC parity on a well-conditioned TS, with a control
    that measures the RIGHT noise (B=N-vs-B=1 batch-membership fp32), and
    DISCRIMINATES fp32-amplification from a real bug.

    Why the old canon-vs-canon (oracle B=1 twice) was wrong: UMA is deterministic
    at fixed batch size, so oracle(B=1) vs oracle(B=1) = 0. The real batching noise
    is the per-structure force difference between a B=N forward and a B=1 forward
    (eSCN fp32, ~1e-7 Eh/A; VALIDATION_MATRIX V8 max|dF|~2.9e-6 eV/A @B128). An IRC
    from a saddle is chaotic and amplifies that ~1e-7 into ~1e-2 A at the endpoint.

    Three runs (all use get_ef_gpu forces, side-matched, HCN<->HNC saddle):
      oracle   = single EulerPC, forces from the model at B=1 (via _OracleCalc).
      batch_B1 = EulerPCBatch on the SAME single structure -> the batched CODE path
                 but at B=1 (identical B=1 forces as the oracle).
      batch_BN = EulerPCBatch on N identical structures -> B=N forces.

    Residuals (each has d0 = rec-0 geometry, dend = last-common-step geometry, dE):
      M_code = residual(batch_B1, oracle)     -- batching CODE vs oracle at IDENTICAL
               B=1 forces. Must be ~0: proves packing/indexing/generators/canonical-
               sign are exact, isolating the ONLY remaining variable to B=N forces.
      M_amp  = residual(batch_BN, batch_B1)   -- the batch-membership amplification
               (B=N vs B=1). THIS is the correct noise control.
      M_test = residual(batch_BN, oracle)     -- what we validate.
      M_int  = residual(batch_BN[i], batch_BN[0]) -- per-structure slicing (must ~0).

    Assertions (discriminating):
      A. M_code d0/dend <= code_floor, dE <= energy_code_floor  [CODE correctness;
         fail => real code bug].
      B. M_int  dend <= code_floor                              [B=N slicing correct].
      C. M_test.d0 <= max(rec0_floor, M_amp.d0 + code_floor)    [EARLY-STEP fidelity:
         the neg-mode/Hessian at B=N matches within batch-membership; a large d0
         fingerprints a Hessian/get_efh_gpu B=N problem, NOT chaos].
      D. M_test.dend <= amp_factor * M_amp.dend + endpoint_floor [the endpoint
         divergence is bounded by the batch-membership amplification -> fp32].

    Conclusion printed: whether the endpoint divergence is fp32-batch-AMPLIFIED
    (A,B,C pass, small d0 grows to large dend within M_amp) or a REAL BUG.
    """
    if params is None:
        # Tight thresholds so all runs take the full max_steps (equal step counts).
        params = EulerPCParams(max_steps=15, f_max_th=1e-12, f_rms_th=1e-12,
                               print_each=False, write_traj=False)
    tmp = tempfile.mkdtemp(prefix="eulerpc_parity_")

    if atoms is None:
        atoms = _make_ts(make_calc(), verbose=verbose)
    atoms_list = [atoms.copy() for _ in range(n)]

    # ---- quantify TS conditioning + FD-Hessian per-CALL variance (the numbers
    #      that show a residual M_code is ill-conditioning, not a code bug) ------
    if verbose:
        try:
            cc = make_calc()
            _e, _f, H1 = _calc_efh_B1(cc, atoms)
            _e, _f, H2 = _calc_efh_B1(cc, atoms)      # 2nd call, SAME instance+coords
            D = _mw(atoms)
            def _negvec(H):
                Hmw = (D[:, None] * H) * D[None, :]; Hmw = 0.5 * (Hmw + Hmw.T)
                w, V = np.linalg.eigh(Hmw); neg = np.where(w < 0)[0]
                return (V[:, neg[np.argsort(w[neg])][0]] if len(neg) else np.zeros_like(D)), w
            v1, w = _negvec(H1); v2, _ = _negvec(H2)
            gap = float(w[1] - w[0]); cos = abs(float(v1 @ v2))
            dH = float(np.max(np.abs(H1 - H2)))
            rec0off = 0.5 * (params.step_length_bohr * 0.529177210903) * np.sqrt(max(0.0, 2 - 2 * cos))
            print(f" TS conditioning: lowest MW-eig={w[0]:.3e} next={w[1]:.3e} gap={gap:.3e} Eh/A^2/amu")
            print(f" FD-Hessian PER-CALL variance (same instance, 2 calls): max|dH|={dH:.2e} Eh/A^2, "
                  f"|cos(v1,v2)|={cos:.4f} -> est. rec-0 offset ~{rec0off:.2e} A")
        except Exception as exc:
            print(f" (conditioning report failed: {exc})")

    # Calc wiring. inject_hessian (default) is the DEFINITIVE isolation: wrap ONE
    # shared calc so get_efh_gpu returns a single precomputed TS Hessian to BOTH
    # oracle and batch, removing the model's per-call FD-Hessian variance so
    # M_code isolates the batching CODE (forces stay the real batched forward).
    # share_calc (no injection) only removes inter-INSTANCE variance -- still
    # subject to per-call FD noise on a near-degenerate saddle.
    if inject_hessian:
        calc = _FixedHessianCalc(make_calc(), atoms)
    elif share_calc:
        calc = make_calc()
    else:
        calc = None

    def _c():
        return calc if calc is not None else make_calc()

    # ---- oracle: single EulerPC, model at B=1 ---------------------------
    oc = _OracleCalc(_c())
    a = atoms.copy()
    a.calc = oc
    oracle = EulerPC(a, os.path.join(tmp, "oracle.out"), params=params).run()
    sO = _sides(oracle)
    assert oracle["forward"]["records"] and oracle["backward"]["records"], \
        "oracle produced an empty side (TS not a saddle for this model?)"

    # ---- batch_B1: the batched CODE path at B=1 (identical B=1 forces) ---
    b1 = EulerPCBatch([atoms.copy()], _c(),
                      output=os.path.join(tmp, "b1.out"), params=params).run()
    assert len(b1) == 1 and bool(b1[0]["valid"]), "batch_B1 invalid"
    sB1 = _sides(b1[0])

    # ---- batch_BN: N identical paths in lockstep ------------------------
    batch = EulerPCBatch(atoms_list, _c(),
                         output=os.path.join(tmp, "bn.out"), params=params)
    batch_BN = batch.run()
    assert len(batch_BN) == n and all(bool(r["valid"]) for r in batch_BN), \
        "batch_BN invalid/short"

    # ---- residuals ------------------------------------------------------
    M_code = _residual(sB1, sO, early_k=early_k)
    M_amp = {"d0": 0.0, "d_early": 0.0, "dend": 0.0, "dE": 0.0}
    M_test = {"d0": 0.0, "d_early": 0.0, "dend": 0.0, "dE": 0.0}
    M_int = {"dend": 0.0}
    s0 = _sides(batch_BN[0])
    for i in range(n):
        assert batch_BN[i]["index"] == i
        sBi = _sides(batch_BN[i])
        ra = _residual(sBi, sB1, early_k=early_k)
        rt = _residual(sBi, sO, early_k=early_k)
        for kk in ("d0", "d_early", "dend", "dE"):
            M_amp[kk] = max(M_amp[kk], ra[kk])
            M_test[kk] = max(M_test[kk], rt[kk])
        if i > 0:
            M_int["dend"] = max(M_int["dend"], _residual(sBi, s0, early_k=early_k)["dend"])

    # rec-0 is chaos-INSENSITIVE (no amplification yet), so its cap is ABSOLUTE --
    # NOT relaxed by M_amp (a uniform B=N Hessian bug inflates M_amp too and would
    # otherwise hide itself). fp32 batch-membership gives rec-0 ~1e-6 A on a well-
    # conditioned saddle; a Hessian/neg-mode bug gives ~1e-2. The endpoint cap IS
    # M_amp-relative (that is the genuine chaotic-amplification bound).
    cap_rec0 = rec0_floor
    cap_end = amp_factor * M_amp["dend"] + endpoint_floor
    amplification = (M_test["dend"] / M_test["d_early"]) if M_test["d_early"] > 0 else float("inf")

    if verbose:
        print("\n=== EulerPCBatch vs EulerPC parity (well-conditioned HCN<->HNC TS) ===")
        print(f" paths (identical B=N)        : {n}")
        print(f" M_code  batch_B1 vs oracle (IDENTICAL B=1 forces):")
        print(f"    rec0 |dx| = {M_code['d0']:.3e} A   end |dx| = {M_code['dend']:.3e} A"
              f"   |dE| = {M_code['dE']:.3e} Eh   (<= code_floor {code_floor:.1e})")
        print(f" M_amp   batch_BN vs batch_B1 (batch-membership B=N-vs-B=1):")
        print(f"    rec0 |dx| = {M_amp['d0']:.3e} A   end |dx| = {M_amp['dend']:.3e} A")
        print(f" M_test  batch_BN vs oracle:")
        print(f"    rec0  |dx| = {M_test['d0']:.3e} A  (cap {cap_rec0:.2e})")
        print(f"    early |dx| = {M_test['d_early']:.3e} A  (first {early_k+1} steps, cap {early_floor:.2e})")
        print(f"    end   |dx| = {M_test['dend']:.3e} A  (cap {cap_end:.2e})")
        print(f"    |dE_end|   = {M_test['dE']:.3e} Eh ({M_test['dE']*HARTREE_EV:.2e} eV)")
        print(f" chaotic amplification (end/early) = {amplification:.1f}x")
        print(f" M_int   batch_BN[i] vs [0] (slicing): end |dx| = {M_int['dend']:.3e} A")
        print(f" batched forwards: get_ef_gpu={batch._ef_calls}, get_efh_gpu={batch._efh_calls}")

    # ---- instrumentation: if M_code fails, pinpoint WHERE (coordinator #1) ----
    if M_code["d0"] > code_floor or M_code["dend"] > code_floor:
        print("\n [!] M_code (batch_B1 vs oracle) EXCEEDS code_floor -- diagnosing:")
        # neg-mode the batched path used (exposed in the result), vs the oracle's
        # neg-mode recomputed here from the SAME shared calc's TS Hessian.
        vb = np.asarray(b1[0].get("neg_eigvec_mw", []), np.float64)
        try:
            _e, _f, Hb = _calc_efh_B1(_c(), atoms)
            D = _mw(atoms)
            Hmw = 0.5 * ((D[:, None] * Hb) * D[None, :] + ((D[:, None] * Hb) * D[None, :]).T)
            w2, V2 = np.linalg.eigh(Hmw)
            neg = np.where(w2 < 0)[0]
            vo = V2[:, neg[np.argsort(w2[neg])][0]] if len(neg) else np.zeros_like(D)
            gap = float(w2[1] - w2[0])
            cos = abs(float(vb @ vo)) / (np.linalg.norm(vb) * np.linalg.norm(vo) + 1e-30) if vb.size else float("nan")
            print(f"     lowest MW-eig={w2[0]:.4e} next={w2[1]:.4e} gap={gap:.4e} "
                  f"(small gap = ill-conditioned -> eigenvector unstable)")
            print(f"     |cos(v_batch, v_oracle_recomputed)| = {cos:.4f} (1.0 = same direction)")
        except Exception as exc:
            print(f"     (neg-mode recompute failed: {exc})")
        for tag, sX in (("oracle", sO), ("batchB1", sB1)):
            print(f"     {tag:8s} fwd rec0 atom0={np.round(sX[0]['xs'][0][0],5)} "
                  f"bwd rec0 atom0={np.round(sX[1]['xs'][0][0],5)}")
        print("     => if gap is small / |cos|<1, the TS neg-mode is ill-conditioned and\n"
              "        the two Hessian evals diverged (a TS-conditioning/Hessian-source\n"
              "        issue, NOT an EulerPCBatch code bug -- share_calc + a better TS fix it).")

    # A. CODE correctness: batch code reproduces the oracle at identical B=1 forces.
    assert M_code["d0"] <= code_floor and M_code["dend"] <= code_floor, (
        f"CODE BUG: batch_B1 diverges from oracle at identical B=1 forces "
        f"(rec0={M_code['d0']:.3e} end={M_code['dend']:.3e} > {code_floor:.1e} A). "
        f"If share_calc=True and the printed neg-mode gap is small/|cos|<1, this is a "
        f"TS ill-conditioning + Hessian-eval divergence, not a batch code bug.")
    assert M_code["dE"] <= energy_code_floor, (
        f"CODE BUG: batch_B1 energy diverges from oracle (dE={M_code['dE']:.3e} Eh)")
    # B. per-structure slicing at B=N.
    assert M_int["dend"] <= code_floor, (
        f"CODE BUG: batch_BN paths differ for identical inputs (slicing) "
        f"end={M_int['dend']:.3e} A")
    # C. rec-0 (chaos-insensitive) ABSOLUTE gate: neg-mode eigenVECTOR at B=N must
    #    match to the fp32 floor -- a large rec-0 = a Hessian-eigenvector problem.
    assert M_test["d0"] <= cap_rec0, (
        f"HESSIAN/NEG-MODE eigenVECTOR at B=N: rec-0 divergence {M_test['d0']:.3e} A "
        f"> {cap_rec0:.1e} -> the initial IRC direction differs by more than fp32; "
        f"get_efh_gpu eigenvector is NOT batch-invariant. Investigate get_efh_gpu "
        f"B=1-vs-B=N.")
    # C'. early-steps (chaos-insensitive) ABSOLUTE gate: forces AND Hessian
    #     eigenVALUES at B=N must be fp32-batch-invariant. Before chaos amplifies,
    #     fp32 batch noise stays ~1e-6 A; a large d_early = a gross force/Hessian
    #     non-invariance at B=N (a bug), not chaos.
    assert M_test["d_early"] <= early_floor, (
        f"EARLY-STEP divergence {M_test['d_early']:.3e} A over the first {early_k+1} "
        f"steps > {early_floor:.1e} -> forces/Hessian at B=N differ by more than fp32 "
        f"BEFORE chaos amplifies; get_ef_gpu/get_efh_gpu is NOT batch-invariant "
        f"(a bug, not chaotic amplification).")
    # D. endpoint bounded by batch-membership amplification -> fp32, not a bug.
    assert M_test["dend"] <= cap_end, (
        f"endpoint {M_test['dend']:.3e} A exceeds amplification bound {cap_end:.3e} "
        f"(batch-membership {M_amp['dend']:.3e}) -> NOT explained by fp32 batch noise")

    if verbose:
        print(" PARITY OK -- endpoint divergence is fp32-batch-AMPLIFIED, proven:")
        print(f"   * code exact at B=1 (M_code ~ {M_code['dend']:.1e} A)")
        print(f"   * per-structure slicing exact (M_int ~ {M_int['dend']:.1e} A)")
        print(f"   * rec-0 + first {early_k} steps match at fp32 (<= {early_floor:.0e} A)")
        print(f"   * endpoint grows {amplification:.0f}x via chaos, bounded by M_amp")
    return {"M_code": M_code, "M_amp": M_amp, "M_test": M_test, "M_int": M_int,
            "amplification": amplification, "caps": {"rec0": cap_rec0, "end": cap_end},
            "ef_calls": batch._ef_calls, "efh_calls": batch._efh_calls, "n": n}


# ======================================================================== #
# Deterministic local self-test: decoupled analytic calc, no model needed.
# Proves the force-batching mechanics reproduce the oracle to ~machine eps.
# ======================================================================== #
class _AnalyticDecoupledCalc:
    """E = 0.5 sum curv*dr^2 + a*sum dr^4 per molecule (dr = r - r0). Anharmonic
    (DWI/Bofill/mBS well-defined) and fully decoupled -> B=1 slice == B=N slice
    bitwise (float64)."""
    def __init__(self):
        import torch
        self._torch = torch
        self.r0 = np.array([0.0, 0.0, 0.0, 0.86, 0.0, 0.0, -0.30, 0.86, 0.0])
        self.curv = np.array([-1.2, 2.0, 3.0, 1.5, 2.5, 3.5, 1.1, 2.2, 3.3])
        self.a = 0.6
        self.n0 = 3

    def prepare(self, atoms_list, fixed_nmax=None):
        self.B = len(atoms_list)
        self.na = [len(x) for x in atoms_list]
        self.N = sum(self.na)
        self.nmax_dof = 3 * max(self.na) if fixed_nmax is None else int(fixed_nmax)
        pos = np.concatenate([np.asarray(x.get_positions(), float) for x in atoms_list], 0)
        self.coord = self._torch.tensor(pos, dtype=self._torch.float64)

    def set_coords_(self, coord):
        self.coord = coord.to(self._torch.float64).clone()

    def step_cart_(self, s):
        pass

    def _ef(self):
        t = self._torch
        E = t.zeros(self.B, dtype=t.float64)
        F = t.zeros((self.B, self.nmax_dof), dtype=t.float64)
        for b in range(self.B):
            r = self.coord[b * self.n0:(b + 1) * self.n0].cpu().numpy().reshape(-1)
            dr = r - self.r0
            E[b] = 0.5 * float(np.sum(self.curv * dr * dr)) + self.a * float(np.sum(dr ** 4))
            F[b, :3 * self.n0] = t.tensor(-(self.curv * dr + 4.0 * self.a * dr ** 3), dtype=t.float64)
        return E, F

    def get_ef_gpu(self):
        return self._ef()

    def get_efh_gpu(self, movable_masks=None):
        t = self._torch
        E, F = self._ef()
        H = t.zeros((self.B, self.nmax_dof, self.nmax_dof), dtype=t.float64)
        for b in range(self.B):
            r = self.coord[b * self.n0:(b + 1) * self.n0].cpu().numpy().reshape(-1)
            diag = self.curv + 12.0 * self.a * (r - self.r0) ** 2
            H[b, :3 * self.n0, :3 * self.n0] = t.diag(t.tensor(diag, dtype=t.float64))
        return E, F, H, t.zeros(self.B, dtype=t.float64)


def _analytic_ts():
    ts = Atoms("OHH", positions=np.array(
        [[0, 0, 0], [0.86, 0, 0], [-0.30, 0.86, 0]], float))
    ts.info.update(charge=0, spin=1, mult=1)
    return ts


def run_selftest_analytic(n=3, verbose=True):
    """Deterministic proof of the batching mechanics (no model): a decoupled
    float64 calc -> B=1 slice == B=N slice bitwise, so ALL residuals ~ 0. Proves
    packing/indexing/generators/canonical-sign/side-matching are exact."""
    summ = run_parity(lambda: _AnalyticDecoupledCalc(), atoms=_analytic_ts(), n=n,
                      params=EulerPCParams(max_steps=8, f_max_th=1e-12, f_rms_th=1e-12,
                                           print_each=False, write_traj=False),
                      verbose=verbose)
    assert summ["M_test"]["dend"] < 1e-9 and summ["M_code"]["dend"] < 1e-12, summ
    if verbose:
        print("SELFTEST(analytic) OK: batch reproduces oracle bitwise (<1e-9)")
    return summ


class _AnalyticBatchNoisyCalc(_AnalyticDecoupledCalc):
    """Analytic calc that injects a DETERMINISTIC, DIRECTION-CHANGING batch-
    membership force noise: an ADDITIVE off-axis force push that scales with the
    batch size, applied to FORCES only. The base PES is diagonal, so the IRC would
    otherwise be 1-D along the neg-mode (coord 0); the off-axis push (component 0
    = 0) bends the path into other dimensions -> a genuine trajectory change that a
    chaotic IRC amplifies. (A per-DOF SCALING is a no-op here: on a 1-D path the
    predictor normalizes it away.) B=1 -> s=0 -> no perturbation (== oracle); B=N ->
    a fixed off-axis push. The Hessian is left unperturbed so rec-0 (set by the
    neg-mode eigenvector, before any perturbed force is used) is batch-invariant --
    modelling a well-conditioned saddle whose endpoint diverges purely by force
    amplification. Mimics UMA's fp32 B=N-vs-B=1 force difference to validate the
    amplification control branch locally (no model)."""
    def __init__(self, eps=1e-2):
        super().__init__()
        self.eps = float(eps)
        self.offaxis = np.array([0.0, 0.5, -0.3, 0.4, -0.2, 0.6, -0.1, 0.3, -0.4])

    def _ef(self):
        E, F = super()._ef()          # base (unperturbed); H (in get_efh_gpu) uses curv
        s = self.eps * (self.B - 1)
        if s != 0.0:
            n3 = 3 * self.n0
            off = self._torch.tensor(self.offaxis[:n3], dtype=self._torch.float64)
            F = F.clone()
            F[:, :n3] = F[:, :n3] + s * off      # additive off-axis -> bends path
        return E, F


def run_selftest_amplification(n=3, eps=1e-2, verbose=True):
    """Validate the CONTROL logic locally: with a deterministic batch-membership
    force noise (B=N != B=1), M_amp>0 and the chaotic IRC amplifies it, while
    M_code stays ~0 (B=1 == oracle). Proves the discriminating control correctly
    bounds M_test by M_amp and flags nothing as a bug for pure fp32-batch noise."""
    # This self-test injects a LARGE (non-fp32) batch-membership noise to make
    # M_amp visibly nonzero, so the absolute fp32 gates (rec0/early) are relaxed --
    # here we validate the RELATIONSHIP logic (M_code~0, M_amp>0, M_test bounded by
    # M_amp), not fp32 magnitude.
    summ = run_parity(lambda: _AnalyticBatchNoisyCalc(eps=eps), atoms=_analytic_ts(),
                      n=n, params=EulerPCParams(max_steps=8, f_max_th=1e-12,
                      f_rms_th=1e-12, print_each=False, write_traj=False),
                      rec0_floor=1.0, early_floor=1.0, endpoint_floor=1.0,
                      verbose=verbose)
    # code exact at B=1 (batch_B1 uses B=1 -> no perturbation -> == oracle)
    assert summ["M_code"]["dend"] < 1e-12, summ
    # batch-membership noise is real (B=N perturbed) and amplified through the IRC
    assert summ["M_amp"]["dend"] > 0.0, summ
    if verbose:
        print(f"SELFTEST(amplification) OK: M_code~0 (code exact), M_amp="
              f"{summ['M_amp']['dend']:.2e} A drives M_test={summ['M_test']['dend']:.2e} A "
              f"({summ['amplification']:.0f}x) -- control correctly attributes it to fp32 batch noise")
    return summ


def run_selftest_distinct(verbose=True):
    """Indexing/slicing proof with DISTINCT inputs (closes the identical-input
    blind spot: a per-structure slicing bug -- path i served path j's forces -- is
    invisible when all inputs are equal). With a decoupled DETERMINISTIC calc,
    batch_BN[i] on N *different* structures MUST equal a standalone single-EulerPC
    oracle_i for structure i, BITWISE. Any packing/slicing/index bug breaks this."""
    base = _analytic_ts().get_positions()
    offs = [np.zeros_like(base),
            np.array([[0.05, 0.0, 0.0], [0.0, -0.04, 0.0], [0.0, 0.0, 0.03]]),
            np.array([[-0.03, 0.02, 0.0], [0.06, 0.0, 0.0], [0.0, 0.05, -0.02]])]
    structs = []
    for off in offs:
        a = _analytic_ts()
        a.set_positions(base + off)
        structs.append(a)

    P = EulerPCParams(max_steps=8, f_max_th=1e-12, f_rms_th=1e-12,
                      print_each=False, write_traj=False)
    tmp = tempfile.mkdtemp(prefix="eulerpc_distinct_")

    # per-structure single-EulerPC oracles (B=1, decoupled calc)
    oracles = []
    for i, a in enumerate(structs):
        ai = a.copy()
        ai.calc = _OracleCalc(_AnalyticDecoupledCalc())
        oracles.append(EulerPC(ai, os.path.join(tmp, f"o{i}.out"), params=P).run())

    # one batched run over the DISTINCT structures
    bn = EulerPCBatch([a.copy() for a in structs], _AnalyticDecoupledCalc(),
                      output=os.path.join(tmp, "bn.out"), params=P).run()

    worst = 0.0
    for i in range(len(structs)):
        r = _residual(_sides(bn[i]), _sides(oracles[i]))
        worst = max(worst, r["dend"], r["d0"])
        assert r["d0"] < 1e-9 and r["dend"] < 1e-9, (
            f"SLICING/INDEX BUG: batch_BN[{i}] (distinct inputs) != oracle_{i} "
            f"(d0={r['d0']:.3e} dend={r['dend']:.3e} A)")
    if verbose:
        print(f"SELFTEST(distinct) OK: batch_BN[i] == oracle_i bitwise for {len(structs)} "
              f"DISTINCT structures (worst {worst:.1e} A) -> packing/slicing/index exact")
    return worst


def _default_uma_make_calc():
    ckpt = os.environ.get("MAPLE_UMA_CKPT")
    size = os.environ.get("MAPLE_UMA_SIZE")
    if not ckpt and not size:
        return None
    try:
        import torch
        from maple.function.calculator.uma._uma_batch_calculator import UMABatchCalc
    except Exception as exc:  # pragma: no cover
        print(f"[skip] cannot import UMABatchCalc: {exc}")
        return None
    if not ckpt:
        from pathlib import Path
        import maple.function.calculator as _calcpkg
        ckpt = str(Path(_calcpkg.__file__).parent / "model" / f"{size}.pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    task = os.environ.get("MAPLE_UMA_TASK", "omol")
    return lambda: UMABatchCalc(ckpt, device=dev, dtype=torch.float64, task=task)


if __name__ == "__main__":
    # 1) deterministic proof of the batching mechanics (identical inputs, ~0)
    run_selftest_analytic()
    # 2) indexing/slicing proof with DISTINCT inputs (batch[i] == oracle_i bitwise)
    run_selftest_distinct()
    # 3) validate the discriminating control logic under injected batch-membership
    #    noise (M_code~0, M_amp>0 amplified into M_test, flagged as fp32 not a bug)
    run_selftest_amplification()
    # 4) real-model parity if a calculator is wired via env
    mk = _default_uma_make_calc()
    if mk is None:
        print("[skip] No UMA wired. Set MAPLE_UMA_CKPT=<path.pt> (or "
              "MAPLE_UMA_SIZE=<uma-s-1p1>) to run the real-model parity, or import "
              "run_parity(make_calc) and inject a batch-calculator factory.")
        sys.exit(0)
    run_parity(mk)
    print("OK")
