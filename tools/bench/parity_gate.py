"""Parity gate (canon-vs-canon) + same-saddle gate + negative-control self-test.

Contract (OPT_CAMPAIGN iron rules 2/3/6):
  * canon-vs-canon: the acceptance floor is the backend's OWN run-to-run self-noise
    (two fresh instances of the SAME code on the SAME structures), never an f64
    constant. threshold = max(3 * canon_floor, abs_min).
  * same-saddle: energy invariant + TS-RMSD + n_imag. Speed without same-saddle
    is not a result.
  * Not-applicable (quantity missing, backend lacks Hessian, empty input) is SKIP
    -- never PASS.
  * self_test() proves the gate can both pass a clean pair AND catch an injected
    degradation. A gate that fires 100% or 0% of the time is broken.

FIX-3 -- TWO-TIER verdicts instead of a single FAIL
---------------------------------------------------
An optimization that changes summation order (cuEq kernels, torch.compile,
constant-tensor pre-allocation, a different chunking of the same reduction) moves
E/F/H by a fp32/kernel-order amount that can exceed 3x the canon floor while being
the SAME science. A flat FAIL would report those as bugs (A3's whole workstream).

  PASS                  <= threshold (3x canon floor, or the abs floor)
  NUMERICALLY_DIFFERENT  > threshold but <= ESCALATE_FACTOR x threshold
                         -> NOT a verdict on its own: the run MUST be escalated
                            to the same-saddle gate + success% / TS-RMSD. It is a
                            request for more evidence, not an accusation.
  REGRESSION            > ESCALATE_FACTOR x threshold, OR in the escalation band
                         with the same-saddle / science evidence also failing.

`classify()` is the single place that maps (value, threshold) -> tier, and
`resolve()` folds in the escalation evidence.
"""
import numpy as np

from .core import kabsch_rmsd

ABS_MIN = dict(dE=1e-10, dF=1e-9, dH=1e-7)          # Ha / Ha A^-1 / Ha A^-2
SS_DEFAULT = dict(dE_Ha=1e-6, rmsd_A=5e-3)          # same-saddle absolute floors
ESCALATE_FACTOR = 10.0                              # PASS < 1x <= NUM_DIFF <= 10x < REGRESSION


def _npy(t):
    return t.detach().cpu().numpy().astype(np.float64)


def compute_efh(calc, mols, with_hessian=True):
    """E/F (+H if the backend supports it) as numpy. Returns dict; H=None if n/a."""
    calc.prepare([m.copy() for m in mols])
    out = dict(H=None)
    if with_hessian:
        try:
            E, F, H, _ = calc.get_efh_gpu()
            out.update(E=_npy(E), F=_npy(F), H=_npy(H))
            return out
        except Exception as e:
            out["hessian_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    E, F = calc.get_ef_gpu()
    out.update(E=_npy(E), F=_npy(F))
    return out


def canon_floor(build_fn, mols, with_hessian=True):
    """Self-noise floor: two FRESH instances of the same code, same structures."""
    a = compute_efh(build_fn(), mols, with_hessian)
    b = compute_efh(build_fn(), mols, with_hessian)
    mx = lambda x, y: float(np.abs(x - y).max())
    fl = dict(dE=mx(a["E"], b["E"]), dF=mx(a["F"], b["F"]))
    fl["dH"] = mx(a["H"], b["H"]) if (a["H"] is not None and b["H"] is not None) else None
    return fl, a


def classify(value, threshold, escalate_factor=ESCALATE_FACTOR):
    """(value, threshold) -> PASS | NUMERICALLY_DIFFERENT | REGRESSION | SKIP.

    NOTE the `_REL` slack on both band edges. Without it the boundary itself is
    decided by floating-point round-off: `10.0 * 1e-6 == 9.999999999999999e-06`,
    so a residual of exactly 10x the threshold was classified REGRESSION instead
    of NUMERICALLY_DIFFERENT. _tier_self_test() caught this; the slack makes the
    documented band edges inclusive as written.
    """
    _REL = 1e-9
    if value is None or threshold is None:
        return "SKIP"
    if value <= threshold * (1.0 + _REL):
        return "PASS"
    if value <= escalate_factor * threshold * (1.0 + _REL):
        return "NUMERICALLY_DIFFERENT"
    return "REGRESSION"


def _check(name, value, floor, abs_min):
    if value is None:
        return dict(name=name, status="SKIP", value=None, threshold=None,
                    note="quantity not available for this backend/run")
    thr = max(3.0 * (floor if floor is not None else 0.0), abs_min)
    st = classify(value, thr)
    out = dict(name=name, status=st, value=float(value), threshold=float(thr),
               ratio_to_threshold=float(value / thr) if thr else None,
               canon_floor=(None if floor is None else float(floor)))
    if st == "NUMERICALLY_DIFFERENT":
        out["note"] = ("within %gx of threshold -- kernel/summation-order class; "
                       "MUST escalate to the same-saddle + science gate before any "
                       "verdict" % ESCALATE_FACTOR)
    return out


def gate_efh(ref, cand, floors):
    """Compare two computed efh dicts under canon floors. list of check dicts."""
    mx = lambda x, y: (None if (x is None or y is None) else float(np.abs(x - y).max()))
    return [
        _check("energy", mx(ref["E"], cand["E"]), floors.get("dE"), ABS_MIN["dE"]),
        _check("forces", mx(ref["F"], cand["F"]), floors.get("dF"), ABS_MIN["dF"]),
        _check("hessian", mx(ref.get("H"), cand.get("H")), floors.get("dH"), ABS_MIN["dH"]),
    ]


def gate_same_saddle(ref_pos, cand_pos, ref_E, cand_E, ref_nimag, cand_nimag,
                     dE_floor=None, thresholds=None):
    """Same-saddle gate over a list of structures.

    ref_pos/cand_pos: list of (n,3) arrays; ref_E/cand_E: energies (Ha);
    ref_nimag/cand_nimag: list[int] or None (-> SKIP that leg).
    PASS only if EVERY structure satisfies all applicable legs.
    """
    th = dict(SS_DEFAULT)
    th.update(thresholds or {})
    dE_thr = max(3.0 * (dE_floor or 0.0), th["dE_Ha"])
    if not ref_pos:
        return dict(name="same_saddle", status="SKIP", note="no structures")
    rows, n_fail = [], 0
    for i in range(len(ref_pos)):
        dE = (None if (ref_E is None or cand_E is None)
              else abs(float(ref_E[i]) - float(cand_E[i])))
        rmsd = kabsch_rmsd(ref_pos[i], cand_pos[i])
        nim_ok = (None if (ref_nimag is None or cand_nimag is None)
                  else (int(ref_nimag[i]) == int(cand_nimag[i])))
        legs = dict(
            dE_Ha=(None if dE is None else (dE, dE <= dE_thr)),
            rmsd_A=(rmsd, rmsd <= th["rmsd_A"]),
            n_imag=(None if nim_ok is None
                    else ((int(ref_nimag[i]), int(cand_nimag[i])), nim_ok)))
        ok = all(v[1] for v in legs.values() if v is not None)
        n_fail += 0 if ok else 1
        rows.append(dict(i=i, ok=ok, legs={k: (v if v is None else v[0]) for k, v in legs.items()}))
    applicable = [k for k in ("dE_Ha", "rmsd_A", "n_imag")
                  if rows and rows[0]["legs"][k] is not None]
    return dict(name="same_saddle", status=("PASS" if n_fail == 0 else "FAIL"),
                n=len(rows), n_fail=n_fail, applicable_legs=applicable,
                skipped_legs=[k for k in ("dE_Ha", "rmsd_A", "n_imag") if k not in applicable],
                thresholds=dict(dE_Ha=dE_thr, rmsd_A=th["rmsd_A"]), rows=rows)


def overall(checks):
    """Worst tier across checks (SKIP only if nothing was applicable)."""
    st = [c["status"] for c in checks]
    for tier in ("REGRESSION", "NUMERICALLY_DIFFERENT"):
        if tier in st:
            return tier
    if "FAIL" in st:            # legacy label, treated as REGRESSION
        return "REGRESSION"
    if all(s == "SKIP" for s in st):
        return "SKIP"
    return "PASS"


def resolve(efh_checks, same_saddle=None, science=None, science_gate=None):
    """Fold escalation evidence into a final verdict (FIX-3).

    efh_checks   list from gate_efh
    same_saddle  dict from gate_same_saddle (or None -> that evidence is SKIP)
    science      dict with observed success_rate / median_TS_RMSD_A (or None)
    science_gate dict with the reference values + tolerances, e.g.
                 {"success_rate_min": 0.85, "median_TS_RMSD_A_max": 0.08}

    Rules
    -----
    * PASS stays PASS.
    * REGRESSION (>10x threshold) stays REGRESSION regardless of evidence.
    * NUMERICALLY_DIFFERENT is resolved by the escalation evidence:
        same-saddle PASS and science within gate  -> ACCEPTED_NUMERICALLY_DIFFERENT
        same-saddle FAIL or science outside gate  -> REGRESSION
        no evidence available                     -> UNRESOLVED (never PASS)
    """
    base = overall(efh_checks)
    ev = dict(same_saddle=(same_saddle or {}).get("status", "SKIP"),
              science_checked=False, science_ok=None)
    if science and science_gate:
        ok = True
        if "success_rate_min" in science_gate and science.get("success_rate") is not None:
            ok = ok and science["success_rate"] >= science_gate["success_rate_min"]
        if ("median_TS_RMSD_A_max" in science_gate
                and science.get("median_TS_RMSD_A") is not None):
            ok = ok and science["median_TS_RMSD_A"] <= science_gate["median_TS_RMSD_A_max"]
        ev["science_checked"] = True
        ev["science_ok"] = bool(ok)

    if base in ("PASS", "SKIP", "REGRESSION"):
        return dict(verdict=base, tier_from_residuals=base, evidence=ev)

    # base == NUMERICALLY_DIFFERENT -> must be resolved by evidence
    if ev["same_saddle"] == "SKIP" and not ev["science_checked"]:
        v = "UNRESOLVED"
    elif ev["same_saddle"] in ("FAIL", "REGRESSION") or ev["science_ok"] is False:
        v = "REGRESSION"
    elif ev["same_saddle"] == "PASS" or ev["science_ok"] is True:
        v = "ACCEPTED_NUMERICALLY_DIFFERENT"
    else:
        v = "UNRESOLVED"
    return dict(verdict=v, tier_from_residuals=base, evidence=ev)


def _tier_self_test():
    """Synthetic known-answer check of the two-tier classifier (FIX-3).

    Deterministic and backend-free: the escalation band is a property of
    classify(), so it is pinned directly instead of hoping a random injection
    lands in a 1x-10x window.
    """
    thr = 1e-6
    cases = [
        (0.5e-6, "PASS"), (1.0e-6, "PASS"),
        (3.0e-6, "NUMERICALLY_DIFFERENT"), (1.0e-5, "NUMERICALLY_DIFFERENT"),
        (1.1e-5, "REGRESSION"), (1.0e-3, "REGRESSION"),
        (None, "SKIP"),
    ]
    rows = [dict(value=v, expected=e, got=classify(v, thr)) for v, e in cases]
    ok = all(r["got"] == r["expected"] for r in rows)
    # resolve(): a NUMERICALLY_DIFFERENT residual must NOT become PASS by itself
    nd = [dict(name="forces", status="NUMERICALLY_DIFFERENT", value=3e-6, threshold=thr)]
    res_no_ev = resolve(nd)["verdict"]
    res_ss_ok = resolve(nd, same_saddle=dict(status="PASS"))["verdict"]
    res_ss_bad = resolve(nd, same_saddle=dict(status="FAIL"))["verdict"]
    res_sci_bad = resolve(nd, same_saddle=dict(status="PASS"),
                          science=dict(success_rate=0.40),
                          science_gate=dict(success_rate_min=0.85))["verdict"]
    ok = (ok and res_no_ev == "UNRESOLVED"
          and res_ss_ok == "ACCEPTED_NUMERICALLY_DIFFERENT"
          and res_ss_bad == "REGRESSION" and res_sci_bad == "REGRESSION")
    return dict(name="tier_classifier_self_test", verdict=("OK" if ok else "BROKEN"),
                threshold=thr, classify_cases=rows,
                resolve_no_evidence=res_no_ev, resolve_same_saddle_pass=res_ss_ok,
                resolve_same_saddle_fail=res_ss_bad,
                resolve_science_fail=res_sci_bad)


def self_test(build_fn, mols, rng_seed=0):
    """Negative control: clean canon pair must PASS; injected degradation must fire.

    Injection 1 (efh gate): 1e-3 A gaussian position noise -> dE/dF/dH must leave PASS.
    Injection 2 (same-saddle): 3e-2 A noise -> TS-RMSD leg must fire.
    Plus the synthetic two-tier classifier check (_tier_self_test).
    verdict OK only if the gate neither fires on clean (100%-trigger = broken)
    nor stays silent on injected (0%-trigger = broken), and the tiers behave.
    """
    rng = np.random.default_rng(rng_seed)
    floors, ref = canon_floor(build_fn, mols)
    clean = compute_efh(build_fn(), mols)
    clean_checks = gate_efh(ref, clean, floors)
    clean_ok = overall(clean_checks) == "PASS"

    noisy = [m.copy() for m in mols]
    for m in noisy:
        m.positions = m.positions + rng.normal(0, 1e-3, m.positions.shape)
    inj = compute_efh(build_fn(), noisy)
    inj_checks = gate_efh(ref, inj, floors)
    inj_caught = overall(inj_checks) in ("NUMERICALLY_DIFFERENT", "REGRESSION", "FAIL")

    big = [m.copy() for m in mols]
    for m in big:
        m.positions = m.positions + rng.normal(0, 3e-2, m.positions.shape)
    ss_clean = gate_same_saddle([m.positions for m in mols], [m.positions for m in mols],
                                clean["E"], clean["E"], None, None, floors.get("dE"))
    ss_inj = gate_same_saddle([m.positions for m in mols], [m.positions for m in big],
                              None, None, None, None, floors.get("dE"))
    ss_ok = ss_clean["status"] == "PASS" and ss_inj["status"] in ("FAIL", "REGRESSION")

    tier = _tier_self_test()
    verdict = ("OK" if (clean_ok and inj_caught and ss_ok
                        and tier["verdict"] == "OK") else "BROKEN")
    return {"name": "gate_self_test", "verdict": verdict,
            "canon_floors": floors,
            "clean_pass": clean_ok, "clean_checks": clean_checks,
            "injected_1e-3A_fired": inj_caught, "injected_checks": inj_checks,
            "injected_tier": overall(inj_checks),
            "same_saddle_clean": ss_clean["status"],
            "same_saddle_injected_3e-2A": ss_inj["status"],
            "tier_classifier": tier,
            "note": ("gate catches 1e-3 A degradation, passes clean canon pair, and "
                     "the PASS / NUMERICALLY_DIFFERENT / REGRESSION tiers behave"
                     if verdict == "OK" else "GATE BROKEN - see checks")}
