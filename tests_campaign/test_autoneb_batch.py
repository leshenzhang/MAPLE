# -*- coding: utf-8 -*-
"""Parity test for the OPT-IN batched AutoNEB (``AutoNEBBatch``).

WHAT IT CHECKS
--------------
Build N (=2) IDENTICAL small reactions (reactant + product) and run:
  * ORACLE  : the unchanged single-reaction ``AutoNEB`` once per reaction,
              driven by a B=1 ``_BatchToASEShim`` over the SAME batch calculator
              (so a batch-only MLIP such as UMABatchCalc -- which has no per-atom
              ASE ``get_potential_energy`` -- still drives the oracle, at the SAME
              model / units [Hartree, Eh/A] as the batched path).
  * BATCHED : ``AutoNEBBatch`` over BOTH reactions -- every live image of every
              reaction packed into ONE ``calc.prepare()`` + ``calc.get_ef_gpu()``
              per optimizer half-step. Per-reaction adaptive image insertion /
              path split / endpoint / convergence stays per-item; the batch width
              changes as reactions insert/trim images and is re-packed each step;
              finished reactions are masked out.

Both paths consume the SAME model/units, so any discrepancy is purely the
block-diagonal batch isolation at the calculator's precision (fp32 for UMA), NOT
an algorithm error of the batched orchestration.

The REACTION GEOMETRY is a real molecular reaction (HCN <-> HNC, the H-migration
isomerization, 3 atoms, one clean barrier -- the same system the sibling GSM
parity test validates against real UMA) and is INDEPENDENT of any calculator
attribute, so the SAME test validates the analytic default AND a real injected
MLIP (``run_parity(make_calc)`` with UMA / MACE / ANI / decoupled AIMNet2).

ASSERTION (fp32-appropriate, attribution-controlled)
-----------------------------------------------------
For each reaction the batched-vs-oracle residual of
  (i)   the forward barrier (Eh),
  (ii)  the TS-image (internal HEI) energy (Eh),
  (iii) the TS-image geometry (Kabsch RMSD, A),
must be <= max(CANON, FLOOR), where CANON is the SAME residual measured between
TWO independent oracle runs of the identical reaction (the canonical fp32 noise
floor of this calculator on this system) and FLOOR is an absolute fp32 floor
(~1e-6 Eh energy; ~5e-3 A geometry over an optimized band). The canon-vs-canon
control attributes any residual to calculator fp32 noise, not to the batching.
(For the deterministic analytic default CANON == 0 and the batched residual is
bitwise 0, so FLOOR governs and passes trivially.)

USAGE (no hardcoded model paths -- pass a calculator factory)
-------------------------------------------------------------
    from tests_campaign.test_autoneb_batch import run_parity
    import torch
    from maple.function.calculator.uma._uma_batch_calculator import UMABatchCalc
    run_parity(lambda: UMABatchCalc(model_path, device="cuda",
                                    dtype=torch.float64, task="omol"))
    # or MACEBatchCalc / MACE-OFF / ANIBatchCalc / AIMNet2DecoupledBatchCalc.

Run offline (analytic default, no torch/GPU/MLIP):
    python3 tests_campaign/test_autoneb_batch.py
"""
from __future__ import annotations

import os
import sys
import types
import tempfile

import numpy as np
from ase import Atoms


# =============================================================================
# Import AutoNEB / AutoNEBBatch.
#
# On a full install (torch + lib2to3 present) the normal package import works.
# In a bare CPU dev env (no torch; some 3.12 builds also drop lib2to3, which
# neb.py has a stray unused import of) we bootstrap: stub the missing modules +
# the torch-heavy parent package __init__ files so ONLY the leaf modules
# (neb.py -- torch-guarded, autoneb.py, jobABC.py, logger.py, molecules.py) load.
# =============================================================================
def _import_autoneb():
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)  # .../MAPLE
    if _root not in sys.path:
        sys.path.insert(0, _root)
    try:
        from maple.function.dispatcher.ts.algorithm.autoneb import (  # noqa: F401
            AutoNEB, AutoNEBBatch, AutoNEBParams)
        return AutoNEB, AutoNEBBatch, AutoNEBParams
    except Exception:
        pass

    # stub lib2to3 (unused stray import in neb.py; absent in some 3.12+ builds)
    for nm in ("lib2to3", "lib2to3.pgen2"):
        if nm not in sys.modules:
            sys.modules[nm] = types.ModuleType(nm)
    if not hasattr(sys.modules["lib2to3.pgen2"], "driver"):
        sys.modules["lib2to3.pgen2"].driver = None

    # stub the torch-heavy parent packages with real __path__ so submodule
    # imports resolve WITHOUT running the package __init__ files (which pull
    # torch via ts/ts.py + ts/algorithm/__init__ -> PRFO/BPRFO/iso_artn).
    chain = [
        ("maple", "maple"),
        ("maple.function", "maple/function"),
        ("maple.function.dispatcher", "maple/function/dispatcher"),
        ("maple.function.dispatcher.ts", "maple/function/dispatcher/ts"),
        ("maple.function.dispatcher.ts.algorithm",
         "maple/function/dispatcher/ts/algorithm"),
    ]
    for name, rel in chain:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [os.path.join(_root, rel)]
            m.__package__ = name
            sys.modules[name] = m

    import importlib
    an = importlib.import_module(
        "maple.function.dispatcher.ts.algorithm.autoneb")
    return an.AutoNEB, an.AutoNEBBatch, an.AutoNEBParams


AutoNEB, AutoNEBBatch, AutoNEBParams = _import_autoneb()

from ase.calculators.calculator import Calculator, all_changes  # noqa: E402


def _to_np(x):
    """Coerce E/F from get_ef_gpu to float64 numpy (numpy or torch tensor)."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    if hasattr(x, "detach"):                # torch.Tensor duck-type
        return x.detach().to("cpu").numpy().astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64)


def _kabsch_rmsd(P, Q):
    """RMSD after optimal rigid (Kabsch) superposition of Q onto P."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Qc.T @ Pc
    U, _S, Vt = np.linalg.svd(H)
    dsign = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, dsign])
    R = Vt.T @ D @ U.T
    Qr = Qc @ R.T
    return float(np.sqrt(np.mean(np.sum((Qr - Pc) ** 2, axis=1))))


# =============================================================================
# Batch-calc -> ASE shim (B=1). Wraps ANY batch calc (prepare + get_ef_gpu) as
# an ASE Calculator so the single-reaction AutoNEB oracle drives the SAME batch
# calculator (same model / units) at batch size 1. Mirrors the sibling GSM
# parity test's ``_BatchToASEShim``; makes the oracle work with a batch-only
# MLIP (UMA/MACE/...) that has no native per-atom ASE energy.
# =============================================================================
class _BatchToASEShim(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, batch_calc, **kwargs):
        super().__init__(**kwargs)
        self._bc = batch_calc

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self._bc.prepare([self.atoms], fixed_nmax=None)
        E, F = self._bc.get_ef_gpu()
        E = _to_np(E)
        F = _to_np(F)
        n = len(self.atoms)
        e = float(E[0])
        self.results["energy"] = e
        self.results["free_energy"] = e
        self.results["forces"] = np.asarray(F)[0, : 3 * n].reshape(n, 3).copy()


# =============================================================================
# Default self-contained analytic calculator (batch contract only; wrapped by
# the shim for the oracle, used directly by the batch). Lab-fixed double well on
# the migrating atom (atom 0, the H) between its reactant / product x-minima,
# with the frame atoms harmonically tethered. Every molecule's E/F is
# independent of its co-batched peers (no cross terms) -> the batched forward is
# byte-identical to per-molecule reads, i.e. a deterministic parity oracle with
# CANON == 0. NOTE: physical realism is NOT required of the default -- the real
# correctness proof is ``run_parity`` with a real MLIP injected; the default only
# exercises the batched orchestration offline (no torch / GPU / MLIP needed).
# =============================================================================
class ToyDoubleWellCalc:
    def __init__(self, xR, xP, centers, A=0.02, Kperp=0.5):
        self.xR = float(xR)
        self.xP = float(xP)
        self.c = np.asarray(centers, dtype=np.float64)   # (n,3) tether centers
        self.A = float(A)
        self.Kperp = float(Kperp)
        self._batch = None
        self._nmax = None

    def _ef_single(self, pos):
        pos = np.asarray(pos, dtype=np.float64)
        n = pos.shape[0]
        c = self.c[:n]
        x = pos[0, 0]
        # double well on atom-0 x (minima at xR, xP; barrier between)
        E = self.A * (x - self.xR) ** 2 * (x - self.xP) ** 2
        F = np.zeros_like(pos)
        F[0, 0] = -(self.A * 2.0 * (x - self.xR) * (x - self.xP)
                    * ((x - self.xP) + (x - self.xR)))
        # harmonic tether on every coord EXCEPT atom-0 x
        disp = pos - c
        E += 0.5 * self.Kperp * (np.sum(disp * disp) - disp[0, 0] ** 2)
        Ftet = -self.Kperp * disp
        Ftet[0, 0] = 0.0
        F += Ftet
        return float(E), F

    # ---- MAPLE batch contract ----
    def prepare(self, atoms_list, fixed_nmax=None):
        self._batch = [a.get_positions().copy() for a in atoms_list]
        self._nmax = fixed_nmax

    def get_ef_gpu(self):
        B = len(self._batch)
        nmax = self._nmax or max(3 * p.shape[0] for p in self._batch)
        E = np.zeros(B, dtype=np.float64)
        F = np.zeros((B, nmax), dtype=np.float64)
        for i, pos in enumerate(self._batch):
            e, f = self._ef_single(pos)
            E[i] = e
            F[i, : 3 * pos.shape[0]] = f.reshape(-1)
        return E, F


# =============================================================================
# Test reaction: HCN <-> HNC (H migrates from C side to N side). 3 atoms,
# identical atom order R/P, one clean barrier. CALCULATOR-INDEPENDENT geometry
# (copied from the sibling GSM parity test), so the same test validates the
# analytic default AND any real injected MLIP.
# =============================================================================
def _build_reaction():
    """One (reactant HCN, product HNC) endpoint pair. Fresh copies each call."""
    R = Atoms("HCN", positions=[[-1.066, 0.0, 0.0],
                                [0.000, 0.0, 0.0],
                                [1.153, 0.0, 0.0]])
    P = Atoms("HCN", positions=[[2.160, 0.0, 0.0],
                                [0.000, 0.0, 0.0],
                                [1.169, 0.0, 0.0]])
    return [R, P]


def _default_make_calc():
    """Analytic offline default: a double well pinned to the HCN/HNC H-minima."""
    R, P = _build_reaction()
    Rp = R.get_positions()
    centers = Rp.copy()
    centers[0] = [0.0, Rp[0, 1], Rp[0, 2]]              # H x is the free coord
    # A / Kperp tuned so the band takes enough optimizer iterations to trigger
    # adaptive image insertion (exercising the variable-batch-width re-pack) while
    # staying bounded and fast; the numbers are irrelevant to the parity proof.
    return ToyDoubleWellCalc(xR=Rp[0, 0], xP=P.get_positions()[0, 0],
                             centers=centers, A=0.5, Kperp=3.0)


# --------------------------------------------------------------------------- #
#  harvest helpers
# --------------------------------------------------------------------------- #
def _barrier_ts(energies, images):
    """(barrier, ts_energy, ts_positions) from the internal-image HEI (the
    endpoints are the fixed reactant/product; after Kabsch alignment an endpoint
    can carry a rotation penalty, so the TS is the interior argmax)."""
    E = np.asarray(energies, dtype=np.float64)
    if len(E) >= 3:
        hei = 1 + int(np.argmax(E[1:-1]))
    else:
        hei = int(np.argmax(E))
    return float(E[hei] - E[0]), float(E[hei]), np.asarray(images[hei].get_positions())


def _run_oracle(calc, paras, workdir, tag):
    R, P = _build_reaction()
    shim = _BatchToASEShim(calc)
    R.calc = shim
    P.calc = shim
    job = AutoNEB(os.path.join(workdir, f"{tag}.out"), [R, P], paras)
    job.run()
    return _barrier_ts(job.final_energies, job.final_images)


# =============================================================================
# Parity harness
# =============================================================================
def run_parity(make_calc, n_reactions: int = 2, workdir: str = None,
               tol_barrier: float = 1e-6, tol_ts_energy: float = 1e-6,
               tol_ts_rmsd: float = 5e-3, canon_margin: float = 4.0,
               verbose: bool = True):
    """Run ``AutoNEBBatch`` vs the single-reaction ``AutoNEB`` oracle over
    ``n_reactions`` identical reactions and assert parity within the fp32 noise
    floor, with a canon-vs-canon control attributing any residual to the
    calculator (not the batching).

    ``make_calc`` : zero-arg factory returning a batch calculator implementing
                    the MAPLE batch contract (``prepare`` + ``get_ef_gpu``,
                    Hartree / Eh/A). No hardcoded model paths.
    """
    assert 1 <= n_reactions <= 8
    wd = workdir or tempfile.mkdtemp(prefix="autoneb_batch_parity_")
    os.makedirs(wd, exist_ok=True)
    calc = make_calc()

    # Count batched forwards + max batch width WITHOUT touching autoneb.py:
    # wrap the calc's own prepare(). Reset before the batch phase so the oracle
    # (B=1 shim) forwards are excluded.
    counts = {"forwards": 0, "max_B": 0}
    _orig_prepare = calc.prepare

    def _counting_prepare(atoms_list, fixed_nmax=None):
        counts["forwards"] += 1
        counts["max_B"] = max(counts["max_B"], len(atoms_list))
        return _orig_prepare(atoms_list, fixed_nmax=fixed_nmax)

    calc.prepare = _counting_prepare

    # AutoNEB params: small + fast, but exercise the full pipeline (adaptive
    # insertion -> the batch-width-divergence path -- + final refinement).
    paras = {"autoneb": {
        "n_images": 5,
        # ifidpp=0 (linear interpolation, no IDPP smoothing): HCN<->HNC sends H
        # exactly through C in the interpolation, and IDPP's 1/dij term diverges
        # at that coincidence -- a property of THIS reaction's init independent of
        # the calculator (it would blow up the oracle identically). Linear init
        # avoids it; NEB relaxes the collision perpendicularly.
        "ifidpp": 0,
        "ang_max": 0.15,       # < image spacing -> triggers adaptive insertion
        "ang_iter": 5,         # check insertion every 5 iters (band converges ~7-9)
        # path_iter / ep_iter disabled: HCN<->HNC is a single barrier (no interior
        # minimum to split on), and this keeps the tree phase off the parent
        # AutoNEB's aggressive-endpoint-trim edge case. The final global
        # refinement still exercises endpoint updates (final_refine_ep_iter) and
        # its own batched re-pack.
        "path_iter": 9999,
        "ep_iter": 9999,
        "max_iter": 120,
        "do_final_refine": True,
        "final_refine_max_iter": 40,
        "verbose": 0,
    }}

    # ---- ORACLE (run 1) + CANON control (run 2) per reaction ----
    oracle = [_run_oracle(calc, paras, wd, f"oracle_rxn{i}") for i in range(n_reactions)]
    canon = [_run_oracle(calc, paras, wd, f"canon_rxn{i}") for i in range(n_reactions)]

    # ---- BATCHED: AutoNEBBatch over all reactions at once ----
    counts["forwards"] = 0
    counts["max_B"] = 0
    reactions = []
    for _i in range(n_reactions):
        R, P = _build_reaction()
        shim = _BatchToASEShim(calc)          # for the rare control-flow ASE reads
        R.calc = shim
        P.calc = shim
        reactions.append([R, P])
    batch_job = AutoNEBBatch(os.path.join(wd, "batch.out"),
                             reactions, calc=calc, paras=paras)
    results = batch_job.run()
    assert len(results) == n_reactions
    assert batch_job._forwards > 0, "batch never issued a batched forward"

    max_final_band = max(r["n_images"] for r in results)
    metrics = {"workdir": wd, "n_reactions": n_reactions,
               "batch_forwards": batch_job._forwards,
               "batch_image_evals": batch_job._image_evals,
               "prepare_max_B": counts["max_B"],
               "max_final_band": max_final_band, "per_reaction": []}

    for i in range(n_reactions):
        o_bar, o_ets, o_ts = oracle[i]
        c_bar, c_ets, c_ts = canon[i]
        b_bar = float(results[i]["barrier_Eh"])
        b_ets = float(results[i]["energies"][results[i]["hei"]])
        b_ts = np.asarray(results[i]["ts_image"].get_positions())

        d_bar = abs(b_bar - o_bar)
        d_ets = abs(b_ets - o_ets)
        d_rmsd = _kabsch_rmsd(o_ts, b_ts)
        canon_bar = abs(c_bar - o_bar)
        canon_ets = abs(c_ets - o_ets)
        canon_rmsd = _kabsch_rmsd(o_ts, c_ts)

        lim_bar = max(canon_margin * canon_bar, tol_barrier)
        lim_ets = max(canon_margin * canon_ets, tol_ts_energy)
        lim_rmsd = max(canon_margin * canon_rmsd, tol_ts_rmsd)

        metrics["per_reaction"].append(dict(
            index=i, d_barrier=d_bar, d_ts_energy=d_ets, d_ts_rmsd=d_rmsd,
            canon_barrier=canon_bar, canon_ts_energy=canon_ets,
            canon_ts_rmsd=canon_rmsd, n_img=results[i]["n_images"]))

        assert d_bar <= lim_bar, (
            f"rxn {i}: barrier residual {d_bar:.3e} Eh > max(canon x{canon_margin}"
            f"={canon_margin*canon_bar:.3e}, floor={tol_barrier:.1e}) "
            f"(batch={b_bar:.6f} oracle={o_bar:.6f})")
        assert d_ets <= lim_ets, (
            f"rxn {i}: TS-energy residual {d_ets:.3e} Eh > "
            f"max(canon x{canon_margin}={canon_margin*canon_ets:.3e}, "
            f"floor={tol_ts_energy:.1e})")
        assert d_rmsd <= lim_rmsd, (
            f"rxn {i}: TS-geometry RMSD {d_rmsd:.3e} A > "
            f"max(canon x{canon_margin}={canon_margin*canon_rmsd:.3e}, "
            f"floor={tol_ts_rmsd:.1e})")

    # identical inputs -> identical outputs across the two batched reactions
    # within the fp32 floor (a cross-reaction coupling bug would break this even
    # though each also matched its own isolated oracle).
    if n_reactions >= 2:
        d01 = abs(float(results[0]["barrier_Eh"]) - float(results[1]["barrier_Eh"]))
        lim01 = max(canon_margin * max(m["canon_barrier"]
                                       for m in metrics["per_reaction"]), tol_barrier)
        assert d01 <= lim01, (
            f"identical batched reactions diverged: |dBarrier|={d01:.3e} Eh "
            f"(cross-reaction coupling?)")

    # cross-reaction batching actually happened: a single forward packed more
    # images than any one reaction's final band (only possible if >1 reaction was
    # co-batched in one forward).
    assert counts["max_B"] > max_final_band, (
        f"no forward spanned >1 reaction (max_B={counts['max_B']} <= "
        f"max_final_band={max_final_band}); cross-reaction batching not exercised")

    if verbose:
        print("AutoNEBBatch parity PASS")
        print(f"  reactions            : {n_reactions}  ({wd})")
        print(f"  batched forwards     : {batch_job._forwards}")
        print(f"  total image-evals    : {batch_job._image_evals}")
        print(f"  max batch width (B)  : {counts['max_B']}  "
              f"(> max single band {max_final_band} => cross-reaction batched)")
        print(f"  final image counts   : {[r['n_images'] for r in results]}")
        for m in metrics["per_reaction"]:
            print(f"  rxn {m['index']}: dBarrier={m['d_barrier']:.3e} "
                  f"(canon {m['canon_barrier']:.3e})  "
                  f"dTS_E={m['d_ts_energy']:.3e} (canon {m['canon_ts_energy']:.3e})  "
                  f"TS_RMSD={m['d_ts_rmsd']:.3e} (canon {m['canon_ts_rmsd']:.3e})")
    return metrics


# --------------------------------------------------------------------------- #
#  Best-effort smoke driver (no hardcoded paths -- reads the environment).
#    MAPLE_TEST_CALC   : '' (default analytic toy) | 'uma' | 'mace' | 'ani'
#                        | 'aimnet'
#    MAPLE_TEST_MODEL  : model path (required for uma/mace; optional for ani)
#    MAPLE_TEST_DEVICE : 'cuda' (default) | 'cpu'
#    MAPLE_TEST_TASK   : UMA task name (default 'omol')
#    MAPLE_TEST_N      : number of reactions (default 2)
# --------------------------------------------------------------------------- #
def _make_calc_from_env():
    kind = os.environ.get("MAPLE_TEST_CALC", "").lower()
    if kind in ("", "toy", "default"):
        return _default_make_calc
    import torch
    dev = os.environ.get("MAPLE_TEST_DEVICE", "cuda")
    model = os.environ.get("MAPLE_TEST_MODEL")
    dt = torch.float64
    if kind == "uma":
        from maple.function.calculator.uma._uma_batch_calculator import UMABatchCalc
        task = os.environ.get("MAPLE_TEST_TASK", "omol")
        assert model, "MAPLE_TEST_MODEL required for UMA"
        return lambda: UMABatchCalc(model, device=dev, dtype=dt, task=task)
    if kind == "mace":
        from maple.function.calculator.mace._mace_batch_calculator import MACEBatchCalc
        assert model, "MAPLE_TEST_MODEL required for MACE"
        return lambda: MACEBatchCalc(model, device=dev, dtype=dt)
    if kind == "ani":
        from maple.function.calculator.ani._ani_batch_calculator import ANIBatchCalc
        return lambda: ANIBatchCalc(model_path=model, device=dev, dtype=dt)
    if kind == "aimnet":
        from maple.function.calculator.aimnet._aimnet2_decoupled_batch_calculator import (
            AIMNet2DecoupledBatchCalc)
        return lambda: AIMNet2DecoupledBatchCalc(model or "aimnet2", device=dev, dtype=dt)
    raise SystemExit(f"unknown MAPLE_TEST_CALC={kind!r}")


if __name__ == "__main__":
    try:
        make_calc = _make_calc_from_env()
    except Exception as e:  # torch / model / calc unavailable in this env
        print(f"[skip] could not build a calc from the environment: {e!r}")
        raise SystemExit(0)
    n = int(os.environ.get("MAPLE_TEST_N", "2"))
    run_parity(make_calc, n_reactions=n)
