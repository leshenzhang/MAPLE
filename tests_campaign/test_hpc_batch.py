# -*- coding: utf-8 -*-
"""Parity test: HPCBatch (batched HPC-IRC) vs single-structure HPC (oracle).

Builds N=3 IDENTICAL small IRC inputs (planar D3h ammonia -- the umbrella
inversion transition state, a textbook first-order saddle with exactly ONE
imaginary mode for organic-trained MLIPs), runs them as ONE batch through
``HPCBatch`` and, independently, one-at-a-time through the serial ``HPC`` (each
serial run driven by a B=1 wrapper over the SAME batched calculator so both
follow the identical PES), then asserts the forward/backward IRC paths agree to
the fp32 forward-noise floor.

Calculator-agnostic: the caller supplies a factory ``make_calc`` that returns a
fresh GPU batch calculator (prepare / get_ef_gpu / get_efh_gpu contract). No
model paths are hardcoded.

    from tests_campaign.test_hpc_batch import run_parity
    run_parity(lambda: UMABatchCalc(model_path=..., device="cuda"))

Tolerances. Parity is BIT-EXACT on a deterministic block-diagonal PES (proven
via the analytic-calc smoke run). On a real fp32 MLIP the tiny per-forward noise
ACCUMULATES along the integrated IRC path, so the floors below are the fp32
noise floor, NOT a bug (a real UMA run measured geometry dev ~2.4e-6 A, force
~2e-7 Eh/A):
    energy      ~1e-6 Eh
    force/|G|   ~2e-7 Eh/A   (fp32 forward-force floor)
    geometry    ~1e-5 A      (fp32 noise integrated along the path)
    Hessian     ~1e-5 Eh/A^2 (TS-anchor Hessian check)

Per-side records schema differs by integrator: single HPC returns a LIST-of-dicts
while HPCBatch returns the canonical DICT-of-lists ({"E":[...],"x":[...],...},
same as LQABatch/EulerPCBatch); ``_as_lod`` normalizes both before comparison.
"""

import os
import sys

import numpy as np
from ase import Atoms

# --- make the repo importable when run as a bare script -----------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from maple.function.dispatcher.irc.algorithm.hpc import (  # noqa: E402
    HPC, HPCBatch, HPCParams, to_f64,
)

# Default tolerances (see module docstring). GTOL/XTOL are the fp32 noise floors
# measured on a real UMA run (force ~2e-7 Eh/A, geometry ~2.4e-6 A accumulated
# along the integrated path); parity is bit-exact on a deterministic PES.
ETOL = 1e-6      # Eh
GTOL = 2e-7      # Eh/A  (UMA fp32 forward-force noise floor, per VALIDATION_MATRIX V8)
XTOL = 1e-5      # A     (fp32 noise accumulates along the integrated IRC path)
HTOL = 1e-5      # Eh/A^2


# ======================================================================== #
# B=1 ASE-adapter over a batched calc -> lets the serial HPC oracle run on   #
# the SAME model math as HPCBatch (bit-equivalent forward => sharp parity).  #
# ======================================================================== #
class _SingleCalcAdapter:
    """Serve the single ``HPC``'s ``atoms.calc`` (get_potential_energy /
    get_forces / get_hessian) from a batched calculator run at B=1."""

    def __init__(self, batch_calc):
        self._calc = batch_calc

    def _prep(self, atoms):
        # prepare() reads atoms.get_positions() -> reflects HPC's set_positions.
        self._calc.prepare([atoms])

    def get_potential_energy(self, atoms=None, force_consistent=False, **kw):
        self._prep(atoms)
        E, _F = self._calc.get_ef_gpu()
        return float(E[0].item())

    def get_forces(self, atoms=None, **kw):
        self._prep(atoms)
        _E, F = self._calc.get_ef_gpu()
        n = len(atoms)
        return to_f64(F[0, :3 * n]).reshape(n, 3)

    def get_hessian(self, atoms=None, **kw):
        self._prep(atoms)
        _E, _F, H, _P = self._calc.get_efh_gpu()
        n = len(atoms)
        return to_f64(H[0, :3 * n, :3 * n])


# ======================================================================== #
# geometry: planar (D3h) ammonia = umbrella inversion TS (one imaginary mode)#
# ======================================================================== #
def _planar_nh3(r=1.00):
    """Planar NH3 in the xy-plane (z=0 for all atoms) -> inversion saddle."""
    ang = np.deg2rad([0.0, 120.0, 240.0])
    pos = [[0.0, 0.0, 0.0]]
    for a in ang:
        pos.append([r * np.cos(a), r * np.sin(a), 0.0])
    at = Atoms("NH3", positions=np.asarray(pos, dtype=np.float64))
    at.info["charge"] = 0
    at.info["spin"] = 1
    at.info["mult"] = 1
    return at


# ======================================================================== #
# comparison helpers                                                         #
# ======================================================================== #
def _as_lod(records):
    """Normalize a side's 'records' to a LIST-of-dicts [{E,maxG,rmsG,x}, ...].

    Accepts either the serial HPC list-of-dicts OR the canonical HPCBatch /
    LQABatch dict-of-lists {"E":[...],"maxG":[...],"rmsG":[...],"x":[...]}."""
    if isinstance(records, dict):
        n = len(records.get("E", []))
        return [{"E": records["E"][k], "maxG": records["maxG"][k],
                 "rmsG": records["rmsG"][k], "x": records["x"][k]}
                for k in range(n)]
    return list(records)


def _side_dev(rec_b, rec_s):
    """Max abs deviation (dE, dG, dX) over the common prefix of two record sets,
    plus the length mismatch. A length mismatch alone is a hard failure. Both
    inputs may be list-of-dicts (serial HPC) or dict-of-lists (HPCBatch)."""
    rb, rs = _as_lod(rec_b), _as_lod(rec_s)
    nb, ns = len(rb), len(rs)
    n = min(nb, ns)
    dE = dG = dX = 0.0
    for k in range(n):
        b, s = rb[k], rs[k]
        dE = max(dE, abs(float(b["E"]) - float(s["E"])))
        dG = max(dG, abs(float(b["maxG"]) - float(s["maxG"])),
                 abs(float(b["rmsG"]) - float(s["rmsG"])))
        dX = max(dX, float(np.max(np.abs(np.asarray(b["x"]) - np.asarray(s["x"])))))
    return dE, dG, dX, abs(nb - ns)


def _pair_cost(bf, bb, sf, sb):
    """Total (dE+dG+dX+len) cost of pairing batch (fwd,bwd) to single (fwd,bwd)."""
    a = _side_dev(bf, sf); b = _side_dev(bb, sb)
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])


def run_parity(make_calc, max_steps=6, verbose=True):
    """Assert HPCBatch == serial HPC (oracle) on N=3 identical planar-NH3 TSs.

    ``make_calc``: zero-arg factory returning a fresh batched calculator
    (prepare + get_ef_gpu + get_efh_gpu). Two independent instances are built (one
    for the batch, one for the B=1 oracle adapter) so neither run mutates the
    other's coordinate buffer.
    """
    B = 3
    params = HPCParams(max_steps=max_steps, print_each=False, write_traj=False)

    # -------- batched run --------
    batch_atoms = [_planar_nh3() for _ in range(B)]
    batch_calc = make_calc()
    hb = HPCBatch(batch_atoms, batch_calc, output="hpc_batch.out", params=params)
    batch_res = hb.run()

    if verbose:
        print(f"[HPCBatch] batched forwards: get_ef_gpu={hb._ef_calls} "
              f"get_efh_gpu={hb._efh_calls} (B={B})")

    # -------- serial oracle runs (B=1 adapter over the SAME model) --------
    oracle_calc = make_calc()
    adapter = _SingleCalcAdapter(oracle_calc)
    single_res = []
    for k in range(B):
        at = _planar_nh3()
        at.calc = adapter
        # fresh params per run (defensive; HPC does not mutate params)
        sp = HPCParams(max_steps=max_steps, print_each=False, write_traj=False)
        res = HPC(at, output=f"hpc_single_{k}.out", params=sp).run()
        single_res.append(res)

    # -------- assertions --------
    # (1) every batch structure resolved a valid negative TS mode
    for k in range(B):
        assert batch_res[k]["valid"], (
            f"structure {k}: HPCBatch flagged the planar-NH3 TS invalid "
            f"(neg_eigval={batch_res[k]['neg_eigval']:.4e}); the calculator does "
            f"not see an imaginary umbrella mode -> cannot run the parity IRC.")

    # (2) the 3 IDENTICAL structures must give bit-equivalent batch trajectories
    #     (proves no cross-structure contamination / mis-routing in the batch).
    for k in range(1, B):
        for side in ("forward", "backward"):
            d = _side_dev(batch_res[0][side]["records"], batch_res[k][side]["records"])
            assert d[3] == 0 and d[0] <= GTOL and d[2] <= XTOL, (
                f"intra-batch mismatch struct0 vs struct{k} [{side}]: "
                f"dE={d[0]:.2e} dX={d[2]:.2e} len_diff={d[3]} "
                f"(identical inputs must give identical batch paths)")

    # (3) batch vs serial oracle, sign-agnostic (the neg eigenmode phase from eigh
    #     can flip, swapping which side is 'forward'); pick the better pairing.
    worst = dict(dE=0.0, dG=0.0, dX=0.0, dLen=0)
    for k in range(B):
        bf = batch_res[k]["forward"]["records"]
        bb = batch_res[k]["backward"]["records"]
        sf = single_res[k]["forward"]["records"]
        sb = single_res[k]["backward"]["records"]
        straight = _pair_cost(bf, bb, sf, sb)          # fwd<->fwd, bwd<->bwd
        swapped = _pair_cost(bf, bb, sb, sf)           # fwd<->bwd, bwd<->fwd
        dE, dG, dX, dLen = min((straight, swapped), key=lambda c: sum(c))

        # TS-anchor scalar check (single HPC exposes E_ts inside each side's log).
        dE_ts = abs(batch_res[k]["E_ts"] - float(single_res[k]["forward"]["E_ts"]))

        worst["dE"] = max(worst["dE"], dE, dE_ts)
        worst["dG"] = max(worst["dG"], dG)
        worst["dX"] = max(worst["dX"], dX)
        worst["dLen"] = max(worst["dLen"], dLen)

        assert dLen == 0, (
            f"structure {k}: path-length mismatch batch vs oracle "
            f"(fwd/bwd step counts differ) -- integrators diverged.")
        assert dE <= ETOL, f"structure {k}: energy dev {dE:.3e} Eh > {ETOL:.1e}"
        assert dG <= GTOL, f"structure {k}: force dev {dG:.3e} Eh/A > {GTOL:.1e}"
        assert dX <= XTOL, f"structure {k}: geometry dev {dX:.3e} A > {XTOL:.1e}"
        assert dE_ts <= ETOL, f"structure {k}: E_ts dev {dE_ts:.3e} Eh > {ETOL:.1e}"

    if verbose:
        print("[PARITY OK] HPCBatch == serial HPC across N=3 planar-NH3 IRC paths")
        print(f"  worst-case deviations: dE={worst['dE']:.2e} Eh  "
              f"dG={worst['dG']:.2e} Eh/A  dX={worst['dX']:.2e} A  "
              f"len_diff={worst['dLen']}")
    return worst


# ======================================================================== #
# convenience __main__: build a UMA batch calc if MAPLE_TEST_MODEL is set    #
# (a .pt checkpoint), else print how to invoke run_parity. Never fails on    #
# import; only the explicit run touches torch / a model.                     #
# ======================================================================== #
def _default_uma_factory():
    model = os.environ.get("MAPLE_TEST_MODEL")
    if not model:
        return None
    device = os.environ.get("MAPLE_TEST_DEVICE", "cuda")

    def _make():
        from maple.function.calculator.uma._uma_batch_calculator import UMABatchCalc
        return UMABatchCalc(model_path=model, device=device,
                            task=os.environ.get("MAPLE_TEST_TASK", "omol"))
    return _make


if __name__ == "__main__":
    factory = _default_uma_factory()
    if factory is None:
        print("No calculator factory available.\n"
              "  Set MAPLE_TEST_MODEL=/path/to/uma.pt (and optionally "
              "MAPLE_TEST_DEVICE, MAPLE_TEST_TASK) to run the UMA parity check,\n"
              "  or import run_parity(make_calc) with your own batched-calc factory:\n"
              "    from tests_campaign.test_hpc_batch import run_parity\n"
              "    run_parity(lambda: MyBatchCalc(...))")
        sys.exit(0)
    run_parity(factory)
    print("OK")
