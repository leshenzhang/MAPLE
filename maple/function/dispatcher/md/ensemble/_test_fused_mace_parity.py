"""Real-MLIP (MACE-OFF23) fused ON==OFF parity on A100 -- the canonical validation
axis (canon-vs-canon) for the B-154 fused-buffer fix, with a REAL force field rather
than the toy analytic calc.  The fix touches only the thermostat noise buffers (force-
engine-independent), so a correct fused run must reproduce the non-fused run BIT-FOR-BIT
even under a real MLIP.

  GATE A  BatchedNVT (fixed B=4) fused ON vs OFF -- the base fused path with MACE forces.
  GATE B  REMD (T-ladder + Metropolis swaps) fused ON vs OFF -- exercises the _apply_ladder
          + accepted-swap _refresh_c2_dev fix with real forces (the case that was silently
          stale before B-154).

Run under cxtorch on an A100 (sitecustomize.py stubs the broken torchvision).
"""
import os
import numpy as np
import torch

from ase.build import molecule

from maple.function.calculator.mace._maceoff_batch_calculator import MaceOffBatchCalc
from maple.function.dispatcher.md.ensemble.nvt_batched import BatchedNVT
from maple.function.dispatcher.md.ensemble.remd import REMD

MODEL = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _sys():
    at = molecule("CH3CH2OH")
    at.rattle(0.01, seed=1)
    return at


def _coord(calc):
    return calc.coord.detach().to("cpu").numpy().copy()


def gate_a_nvt(steps=200, B=4, seed=42):
    base = dict(timestep=0.5, temperature=300.0, thermostat="langevin", friction=0.02,
                remove_com_every=0, random_seed=seed, verbose=0)
    outs = {}
    for fused in (False, True):
        bc = MaceOffBatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64)
        reps = [_sys() for _ in range(B)]
        sim = BatchedNVT("/tmp/nvtf_%d" % int(fused), reps, calc=bc,
                         paras=dict(base, steps=steps, fused_loop=fused)).run()
        outs[fused] = _coord(bc)
    dc = float(np.max(np.abs(outs[True] - outs[False])))
    print(f"[GATE A nvt-MACE] B={B} steps={steps}  fused ON vs OFF  max|dcoord|={dc:.3e} A")
    assert dc == 0.0, f"MACE BatchedNVT fused not bit-identical (dcoord={dc})"
    print("[GATE A] PASS  (real-MLIP fused nvt bit-identical)")


def gate_b_remd(steps=2000, N=6, seed=7):
    base = dict(timestep=0.5, thermostat="langevin", friction=0.02, remove_com_every=0,
                n_replicas=N, temp_min=300.0, temp_max=600.0, mode="temperature",
                exchange_every=50, random_seed=seed, swap_seed=seed + 11, verbose=0)
    outs, accs = {}, {}
    for fused in (False, True):
        bc = MaceOffBatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64)
        sim = REMD("/tmp/remdf_%d" % int(fused), _sys(), calc=bc,
                   paras=dict(base, steps=steps, fused_loop=fused)).run()
        outs[fused] = _coord(bc)
        accs[fused] = int(np.sum(sim._n_accept))
    dc = float(np.max(np.abs(outs[True] - outs[False])))
    print(f"[GATE B remd-MACE] N={N} steps={steps}  swaps OFF={accs[False]} ON={accs[True]}  "
          f"fused ON vs OFF max|dcoord|={dc:.3e} A")
    assert accs[False] > 0, "REMD gate saw no swaps -- relabel path not exercised"
    assert dc == 0.0 and accs[True] == accs[False], f"MACE REMD fused not bit-identical (dcoord={dc})"
    print("[GATE B] PASS  (real-MLIP fused REMD bit-identical through swaps)")


if __name__ == "__main__":
    print("fused MACE-parity  torch", torch.__version__, "cuda", torch.cuda.is_available(), "dev", DEV)
    gate_a_nvt()
    gate_b_remd()
    print("\n[FUSED-MACE-PARITY] ALL GATES PASS")
