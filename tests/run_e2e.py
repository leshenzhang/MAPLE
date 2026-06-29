# -*- coding: utf-8 -*-
"""
CMET_4 (c-Met / 4R1Y) end-to-end ML-GBSA: batched vs serial EndpointBinding(mode='gb').

Standalone (fresh process -> clean GPU; the combined smoke accumulated memory across
its parity sweep). Reports dG_bind parity (batched vs serial) + wall-time speedup.
Backend = MACE-OFF23-medium via the standard-MACE autograd batch calc; serial uses the
same model through the upstream mace_off ASE calculator (Hartree-wrapped).
"""
import os
import sys
import gc
import time
import numpy as np

REPO = "/ibex/user/xiaox/zls/ai-maple-md/MAPLE/worktrees/mmgbsa-batch"
sys.path.insert(0, REPO)
CACHE = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
CMET = "/ibex/user/xiaox/zls/ai-maple-md/val_dock_pbsa/cmet_data/CMET_4"
PRMTOP = os.path.join(CMET, "stripped.prmtop")
NC = os.path.join(CMET, "snapshots.nc")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0) if DEV=='cuda' else 'cpu'}",
      flush=True)

from maple.workflow.mmpbsa import EndpointBinding
from maple.function.calculator.mace._mace_autograd_batch_calculator import (
    MACEAutogradBatchCalc, EV2HARTREE)
from mace.calculators.foundations_models import mace_off
from ase.calculators.calculator import Calculator, all_changes


class HartreeWrap(Calculator):
    """Hartree-native ASE wrapper (EndpointBinding serial path treats get_potential_energy
    as Hartree; the upstream MACE-OFF calc returns eV)."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, base):
        super().__init__()
        self.base = base

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = self.atoms.copy(); a.calc = self.base
        self.results["energy"] = a.get_potential_energy() * EV2HARTREE
        if "forces" in properties:
            self.results["forces"] = a.get_forces() * EV2HARTREE


Fe = 12

# ---- batched run (OOM-resilient ladder) ----
# A single 4331-atom-complex forward at f64 caps the batch: B=2 (~8.7k atoms) fits an
# 80GB A100; B>=3 (>=13k atoms) OOMs. TorchScript wraps CUDA-OOM as a generic
# RuntimeError, so catch both. (An energy-only get_e_gpu would ~halve memory and lift
# this cap; the autograd batch calc currently exposes only get_ef_gpu.)
bcalc = MACEAutogradBatchCalc(model_path=CACHE, device=DEV, dtype=torch.float64)
r_b, t_b, mb = None, None, None
for cand in (2, 1):
    try:
        if DEV == "cuda":
            torch.cuda.empty_cache()
        t0 = time.time()
        eb_b = EndpointBinding(PRMTOP, NC, calc=bcalc, ligand_resname="LIG",
                               frames=Fe, mode="gb", output="/tmp/mmgbsa_b", max_batch=cand)
        assert eb_b._batched is True
        r_b = eb_b.run()
        if DEV == "cuda":
            torch.cuda.synchronize()
        t_b = time.time() - t0
        mb = cand
        break
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise
        print(f"[D] batched max_batch={cand} OOM, dropping", flush=True)
        del bcalc
        gc.collect()
        if DEV == "cuda":
            torch.cuda.empty_cache()
        bcalc = MACEAutogradBatchCalc(model_path=CACHE, device=DEV, dtype=torch.float64)
        continue
assert r_b is not None, "batched run OOM at every max_batch"
print(f"[D] BATCHED  dG_bind={r_b['dG_bind_mean']:.4f} +/- {r_b['dG_bind_sem']:.4f} kcal/mol "
      f"(dE_int={r_b['dE_int_mean']:.3f}, ddG_solv={r_b['ddG_solv_mean']:.3f}, "
      f"F={r_b['n_frames']}, natom={r_b['n_atoms_complex']}, max_batch={mb}) | {t_b:.2f}s",
      flush=True)

del bcalc
gc.collect()
if DEV == "cuda":
    torch.cuda.empty_cache()

# ---- serial run (same MACE-OFF model, upstream ASE calc, Hartree-wrapped) ----
ase_calc = mace_off(model=CACHE, device=DEV, default_dtype="float64", return_raw_model=False)
t0 = time.time()
eb_s = EndpointBinding(PRMTOP, NC, calc=HartreeWrap(ase_calc), ligand_resname="LIG",
                       frames=Fe, mode="gb", output="/tmp/mmgbsa_s")
assert eb_s._batched is False
r_s = eb_s.run()
t_s = time.time() - t0
print(f"[D] SERIAL   dG_bind={r_s['dG_bind_mean']:.4f} +/- {r_s['dG_bind_sem']:.4f} kcal/mol "
      f"(dE_int={r_s['dE_int_mean']:.3f}, ddG_solv={r_s['ddG_solv_mean']:.3f}) | {t_s:.2f}s",
      flush=True)

dd = abs(r_b["dG_bind_mean"] - r_s["dG_bind_mean"])
print(f"[D] dG_bind batched-vs-serial |d| = {dd:.3e} kcal/mol (want < 1e-3)", flush=True)
print(f"[D] WALL: batched={t_b:.2f}s serial={t_s:.2f}s -> SPEEDUP {t_s / t_b:.1f}x "
      f"(F={Fe}, 3 segments, batched max_batch={mb}; serial = 3F={3*Fe} forwards)", flush=True)
assert dd < 1e-3, dd
print("E2E-DONE", flush=True)
