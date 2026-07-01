# -*- coding: utf-8 -*-
"""
Parity test for the OPT-IN batched Growing String Method (``GSMBatch``) added to
maple/function/dispatcher/ts/algorithm/string.py.

WHAT IT CHECKS
--------------
Build N (=2-3) IDENTICAL small reactions (reactant + product) and run:
  * ORACLE   : the unchanged single ``GSM`` once per reaction, driven by a B=1
               ``_BatchToASEShim`` over the SAME Batch calculator.
  * BATCHED  : ``GSMBatch`` over all N reactions -- every still-live reaction's
               force call is packed into ONE ``calc.get_ef_gpu()`` forward per
               tick (per-reaction control flow stays per-item; finished
               reactions are masked out of the batch).
Both paths consume the SAME model / units (Hartree, Eh/A), so any discrepancy is
purely the block-diagonal batch isolation (perturb-one byte floor).

ASSERTION (two-tier, mirrors the task's "within fp32 noise floor OR same
converged TS node"):
  Tier 1 (strict)   : the final equal-arc MEP strings match image-by-image --
                      max |dx| <= ``tol_geom`` (A) and max |dE| <= ``tol_energy``
                      (Eh) across all 9 resampled nodes of every reaction.
  Tier 2 (fallback) : if the GSM growth path is chaotic and Tier 1 fails, assert
                      the CONVERGED TS node instead -- the PRFO-refined TS (or,
                      absent a TS file, the HEI node) matches after Kabsch
                      alignment within ``tol_ts_rmsd`` (A) and ``tol_ts_energy``
                      (Eh).

USAGE (no hardcoded model paths -- pass a calculator factory)
-------------------------------------------------------------
    from tests_campaign.test_gsm_batch import run_parity
    import torch
    from maple.function.calculator.ani._ani_batch_calculator import ANIBatchCalc
    run_parity(lambda: ANIBatchCalc(model_path=None, device="cuda",
                                    dtype=torch.float64))          # ANI2x
    # or any local/block-diagonal Batch calc: UMABatchCalc(task='omol'),
    # MACEBatchCalc, MACE-OFF, AIMNet2DecoupledBatchCalc.

``run_parity(make_calc)`` returns a metrics dict and raises AssertionError on a
parity failure.  The ``__main__`` block below is a best-effort smoke driver that
builds a calc from the environment (see the bottom of this file).
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
from ase import Atoms

# Ensure the package root (this file's parent's parent) is importable when run
# directly as ``python tests_campaign/test_gsm_batch.py``.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from maple.function.dispatcher.ts.algorithm.string import (  # noqa: E402
    GSM,
    GSMBatch,
    kabsch_align,
    _BatchToASEShim,
    _gsm_parse_xyz,
)


# --------------------------------------------------------------------------- #
#  Test system:  HCN  <->  HNC   (H migrates from the C side to the N side).
#  3 atoms (H, C, N) -- supported by UMA-omol / MACE-OFF / ANI2x / AIMNet2.
#  Atom ORDER is identical between reactant and product (GSM requirement).
# --------------------------------------------------------------------------- #
def _build_reaction():
    """One (reactant HCN, product HNC) endpoint pair.  Fresh copies each call."""
    R = Atoms("HCN", positions=[[-1.066, 0.0, 0.0],
                                [ 0.000, 0.0, 0.0],
                                [ 1.153, 0.0, 0.0]])
    P = Atoms("HCN", positions=[[ 2.160, 0.0, 0.0],
                                [ 0.000, 0.0, 0.0],
                                [ 1.169, 0.0, 0.0]])
    return R, P


def _ts_frame(rec_mep, rec_ts):
    """Pick the converged TS node for the Tier-2 fallback: prefer the PRFO TS
    (``_stringts_ts.xyz``), else the HEI (highest interior MEP node)."""
    if rec_ts:
        return rec_ts[-1]
    if rec_mep and len(rec_mep) > 2:
        Es = [f["energy"] for f in rec_mep]
        if all(e is not None for e in Es):
            hei = max(range(1, len(Es) - 1), key=lambda k: Es[k])
            return rec_mep[hei]
    return None


def run_parity(make_calc,
               n_reactions: int = 2,
               tol_geom: float = 1.0e-4,      # A   (strict, Tier 1)
               tol_energy: float = 1.0e-6,    # Eh  (strict, Tier 1)
               tol_ts_rmsd: float = 5.0e-2,   # A   (fallback, Tier 2)
               tol_ts_energy: float = 1.0e-3, # Eh  (fallback, Tier 2)
               workdir: str = None,
               verbose: bool = True):
    """Run the ORACLE vs BATCHED parity check.

    Parameters
    ----------
    make_calc : callable() -> Batch calculator
        Zero-arg factory returning a LOCAL / block-diagonal Batch calc
        (``prepare`` + ``get_ef_gpu``).  Called ONCE; the same instance drives
        both the oracle (via a B=1 shim) and the batched run (sequential phases,
        so ``prepare`` state is reset between them).
    n_reactions : int
        Number of identical reactions to run (2-3 recommended).

    Returns a metrics dict; raises AssertionError on parity failure.
    """
    assert 1 <= n_reactions <= 8, "n_reactions out of sane range"
    wd = workdir or tempfile.mkdtemp(prefix="gsm_batch_parity_")
    calc = make_calc()

    # ---------------- ORACLE: single GSM per reaction (B=1 shim) -------------
    oracle = []
    for i in range(n_reactions):
        R, P = _build_reaction()
        shim = _BatchToASEShim(calc)
        R.calc = shim
        P.calc = shim
        out = os.path.join(wd, f"oracle_rxn{i}.out")
        try:
            GSM(out, R, P).run()
        except BaseException as e:               # keep the MEP written pre-PRFO
            if verbose:
                print(f"[oracle {i}] GSM raised (post-MEP ok): {e!r}")
        base = os.path.splitext(out)[0]
        oracle.append({"mep": _gsm_parse_xyz(base + "_gsm_mep.xyz"),
                       "ts": _gsm_parse_xyz(base + "_stringts_ts.xyz")})

    # ---------------- BATCHED: GSMBatch over all reactions -------------------
    reactions = [_build_reaction() for _ in range(n_reactions)]
    gb = GSMBatch(reactions, calc, output=os.path.join(wd, "batch.out"))
    results = gb.run()

    # ------------------------------ compare ---------------------------------
    assert len(results) == n_reactions
    metrics = {"workdir": wd, "n_reactions": n_reactions,
               "n_forwards": gb.n_forwards, "n_node_evals": gb.n_node_evals,
               "per_reaction": []}

    strict_ok = True
    for i in range(n_reactions):
        om, ot = oracle[i]["mep"], oracle[i]["ts"]
        bm, bt = results[i]["mep"], results[i]["ts"]
        assert om, f"reaction {i}: oracle produced no MEP (growth failed)"
        assert bm, f"reaction {i}: batched produced no MEP (growth failed)"

        rxn = {"index": i, "batched_error": results[i]["error"],
               "n_mep_oracle": len(om), "n_mep_batched": len(bm)}

        tier1 = (len(om) == len(bm))
        dgeom = denergy = float("nan")
        if tier1:
            dgeom = 0.0
            denergy = 0.0
            for fo, fb in zip(om, bm):
                assert fo["symbols"] == fb["symbols"], \
                    f"reaction {i}: atom order mismatch in MEP"
                dgeom = max(dgeom, float(np.max(np.abs(fo["positions"]
                                                       - fb["positions"]))))
                if fo["energy"] is not None and fb["energy"] is not None:
                    denergy = max(denergy, abs(fo["energy"] - fb["energy"]))
            tier1 = (dgeom <= tol_geom) and (denergy <= tol_energy)
        rxn["mep_max_dgeom_A"] = dgeom
        rxn["mep_max_denergy_Eh"] = denergy
        rxn["tier1_strict_pass"] = bool(tier1)

        if not tier1:
            strict_ok = False
            # ---- Tier 2: same converged TS node within tolerance ----
            fo = _ts_frame(om, ot)
            fb = _ts_frame(bm, bt)
            assert fo is not None and fb is not None, (
                f"reaction {i}: strict MEP parity failed and no TS node available "
                f"for the fallback comparison")
            assert fo["symbols"] == fb["symbols"], \
                f"reaction {i}: TS atom order mismatch"
            _, rmsd, _, _ = kabsch_align(fo["positions"], fb["positions"])
            de_ts = (abs(fo["energy"] - fb["energy"])
                     if (fo["energy"] is not None and fb["energy"] is not None)
                     else float("nan"))
            rxn["ts_rmsd_A"] = float(rmsd)
            rxn["ts_denergy_Eh"] = de_ts
            ts_ok = (rmsd <= tol_ts_rmsd) and (np.isnan(de_ts)
                                               or de_ts <= tol_ts_energy)
            rxn["tier2_ts_pass"] = bool(ts_ok)
            assert ts_ok, (
                f"reaction {i}: PARITY FAILED both tiers -- "
                f"MEP max dgeom={dgeom:.3e} A / dE={denergy:.3e} Eh; "
                f"TS rmsd={rmsd:.3e} A (tol {tol_ts_rmsd}) / "
                f"dE_ts={de_ts:.3e} Eh (tol {tol_ts_energy})")

        metrics["per_reaction"].append(rxn)

    metrics["all_tier1_strict"] = strict_ok
    if verbose:
        print("\n=== GSMBatch parity ===")
        print(f"  workdir       : {wd}")
        print(f"  batched fwds  : {gb.n_forwards}  node-evals: {gb.n_node_evals}")
        for rxn in metrics["per_reaction"]:
            tag = ("STRICT" if rxn["tier1_strict_pass"]
                   else f"TS-node(rmsd={rxn.get('ts_rmsd_A', float('nan')):.2e}A)")
            print(f"  rxn {rxn['index']}: {tag}  "
                  f"MEP dgeom={rxn['mep_max_dgeom_A']:.2e}A "
                  f"dE={rxn['mep_max_denergy_Eh']:.2e}Eh "
                  f"err={rxn['batched_error']}")
        print(f"  ALL Tier-1 strict: {strict_ok}")
        print("  PARITY OK")
    return metrics


# --------------------------------------------------------------------------- #
#  Best-effort smoke driver (no hardcoded paths -- reads the environment).
#    MAPLE_TEST_CALC   : 'ani' (default) | 'uma' | 'mace' | 'aimnet'
#    MAPLE_TEST_MODEL  : model path (required for uma/mace; optional for ani)
#    MAPLE_TEST_DEVICE : 'cuda' (default) | 'cpu'
#    MAPLE_TEST_TASK   : UMA task name (default 'omol')
#    MAPLE_TEST_N      : number of reactions (default 2)
# --------------------------------------------------------------------------- #
def _make_calc_from_env():
    import torch
    kind = os.environ.get("MAPLE_TEST_CALC", "ani").lower()
    dev = os.environ.get("MAPLE_TEST_DEVICE", "cuda")
    model = os.environ.get("MAPLE_TEST_MODEL")
    dt = torch.float64
    if kind == "ani":
        from maple.function.calculator.ani._ani_batch_calculator import ANIBatchCalc
        return lambda: ANIBatchCalc(model_path=model, device=dev, dtype=dt)
    if kind == "uma":
        from maple.function.calculator.uma._uma_batch_calculator import UMABatchCalc
        task = os.environ.get("MAPLE_TEST_TASK", "omol")
        assert model, "MAPLE_TEST_MODEL required for UMA"
        return lambda: UMABatchCalc(model, device=dev, dtype=dt, task=task)
    if kind == "mace":
        from maple.function.calculator.mace._mace_batch_calculator import MACEBatchCalc
        assert model, "MAPLE_TEST_MODEL required for MACE"
        return lambda: MACEBatchCalc(model, device=dev, dtype=dt)
    if kind == "aimnet":
        from maple.function.calculator.aimnet._aimnet2_decoupled_batch_calculator import (
            AIMNet2DecoupledBatchCalc)
        return lambda: AIMNet2DecoupledBatchCalc(
            model or "aimnet2", device=dev, dtype=dt)
    raise SystemExit(f"unknown MAPLE_TEST_CALC={kind!r}")


if __name__ == "__main__":
    try:
        make_calc = _make_calc_from_env()
    except Exception as e:  # torch / model / calc unavailable in this env
        print(f"[skip] could not build a Batch calc from the environment: {e!r}")
        print("       set MAPLE_TEST_CALC / MAPLE_TEST_MODEL and rerun, or call "
              "run_parity(make_calc) directly with your own factory.")
        raise SystemExit(0)
    n = int(os.environ.get("MAPLE_TEST_N", "2"))
    run_parity(make_calc, n_reactions=n)
