"""Parity gate (canon-vs-canon) + same-saddle gate + negative-control self-test.

Contract (OPT_CAMPAIGN iron rules 2/3/6):
  * canon-vs-canon: the acceptance floor is the backend's OWN run-to-run self-noise
    (two fresh instances of the SAME code on the SAME structures), never an f64
    constant. threshold = max(3 * canon_floor, abs_min).
  * same-saddle: energy invariant + TS-RMSD + n_imag. Speed without same-saddle
    is not a result.
  * Every check returns PASS / FAIL / SKIP. Not-applicable (quantity missing,
    backend lacks Hessian, empty input) is SKIP -- never PASS.
  * self_test() proves the gate can both pass a clean pair AND catch an injected
    degradation. A gate that fires 100% or 0% of the time is broken.
"""
import numpy as np

from .core import kabsch_rmsd

ABS_MIN = dict(dE=1e-10, dF=1e-9, dH=1e-7)          # Ha / Ha A^-1 / Ha A^-2
SS_DEFAULT = dict(dE_Ha=1e-6, rmsd_A=5e-3)          # same-saddle absolute floors


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


def _check(name, value, floor, abs_min):
    if value is None:
        return dict(name=name, status="SKIP", value=None, threshold=None,
                    note="quantity not available for this backend/run")
    thr = max(3.0 * (floor if floor is not None else 0.0), abs_min)
    return dict(name=name, status=("PASS" if value <= thr else "FAIL"),
                value=float(value), threshold=float(thr),
                canon_floor=(None if floor is None else float(floor)))


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
    st = [c["status"] for c in checks]
    if "FAIL" in st:
        return "FAIL"
    if all(s == "SKIP" for s in st):
        return "SKIP"
    return "PASS"


def self_test(build_fn, mols, rng_seed=0):
    """Negative control: clean canon pair must PASS; injected degradation must FAIL.

    Injection 1 (efh gate): 1e-3 A gaussian position noise -> dE/dF/dH must fire.
    Injection 2 (same-saddle): 3e-2 A noise -> TS-RMSD leg must fire.
    verdict OK only if the gate neither fires on clean (100%-trigger = broken)
    nor stays silent on injected (0%-trigger = broken).
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
    inj_caught = overall(inj_checks) == "FAIL"

    big = [m.copy() for m in mols]
    for m in big:
        m.positions = m.positions + rng.normal(0, 3e-2, m.positions.shape)
    ss_clean = gate_same_saddle([m.positions for m in mols], [m.positions for m in mols],
                                clean["E"], clean["E"], None, None, floors.get("dE"))
    ss_inj = gate_same_saddle([m.positions for m in mols], [m.positions for m in big],
                              None, None, None, None, floors.get("dE"))
    ss_ok = ss_clean["status"] == "PASS" and ss_inj["status"] == "FAIL"

    verdict = "OK" if (clean_ok and inj_caught and ss_ok) else "BROKEN"
    return {"name": "gate_self_test", "verdict": verdict,
            "canon_floors": floors,
            "clean_pass": clean_ok, "clean_checks": clean_checks,
            "injected_1e-3A_fail": inj_caught, "injected_checks": inj_checks,
            "same_saddle_clean": ss_clean["status"],
            "same_saddle_injected_3e-2A": ss_inj["status"],
            "note": ("gate catches 1e-3 A degradation and passes clean canon pair"
                     if verdict == "OK" else "GATE BROKEN - see checks")}
