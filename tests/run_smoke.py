# -*- coding: utf-8 -*-
"""
Smoke test for the batchified ML-GBSA per-frame MLIP energy (feat/md-mmgbsa-batch).

A) import maple.workflow.mmpbsa
B) GB-OBC frame-vectorized (gb_obc_polar_frames) == per-frame gb_obc_polar (<1e-9)
C) MLIP batched per-frame E == sequential per-frame E (same MACE-OFF autograd calc,
   B=F vs B=1) for ALL 3 segments (<1e-8 Ha) + independent upstream mace_off ASE
   cross-check of the absolute energy.
D) CMET_4 (c-Met / 4R1Y) end-to-end EndpointBinding(mode='gb'): batched vs serial
   dG_bind parity + wall-time speedup factor.

Backend: MACE-OFF23-medium (~/.cache/mace/MACE-OFF23_medium.model) via the
standard-MACE AUTOGRAD batch calc (pure-local MLIP -> exactly-isolated multi-graph
batch). macepol weights are NOT loadable here and MACE-POL cannot be batched anyway
(B=1-locked trace + global charge coupling) -> production accuracy would use the
MACE-POL *sequential* calc; this smoke validates the batching machinery + speedup.
"""
import os
import sys
import time
import numpy as np

REPO = "/ibex/user/xiaox/zls/ai-maple-md/MAPLE/worktrees/mmgbsa-batch"
sys.path.insert(0, REPO)
CACHE = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
CMET = "/ibex/user/xiaox/zls/ai-maple-md/val_dock_pbsa/cmet_data/CMET_4"
PRMTOP = os.path.join(CMET, "stripped.prmtop")
NC = os.path.join(CMET, "snapshots.nc")

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def synchro():
    if DEV == "cuda":
        torch.cuda.synchronize()


# ----------------------------- PART A: import ----------------------------- #
import maple.workflow.mmpbsa as M
from maple.workflow.mmpbsa import gb_obc_polar, gb_obc_polar_frames, EndpointBinding
from maple.workflow.amber_io import Prmtop, read_nc_coords
print("[A] import maple.workflow.mmpbsa OK", flush=True)

# ------------------------ PART B: GB frames parity ------------------------ #
rng = np.random.default_rng(0)
Fb, nb = 8, 30
cB = rng.normal(scale=4.0, size=(Fb, nb, 3))
zB = rng.choice([1, 6, 7, 8], size=nb)
qB = rng.normal(scale=0.3, size=nb)
qB -= qB.mean()                                   # ~neutral
seq = np.array([gb_obc_polar(cB[f], qB, zB) for f in range(Fb)])
vec = gb_obc_polar_frames(cB, qB, zB)
dgb = float(np.max(np.abs(seq - vec)))
print(f"[B] GB frames-vec vs per-frame: max|d| = {dgb:.3e} kcal/mol (want <1e-9)", flush=True)
assert dgb < 1e-9, dgb
assert gb_obc_polar_frames(cB[0], qB, zB).shape == (1,)   # F=1 degenerate
print("[B] GB frame-vectorization parity + F=1 degenerate OK", flush=True)

# ----------------------------- build batch calc --------------------------- #
from maple.function.calculator.mace._mace_autograd_batch_calculator import (
    MACEAutogradBatchCalc, EV2HARTREE)
bcalc = MACEAutogradBatchCalc(model_path=CACHE, device=DEV, dtype=torch.float64)
print(f"[init] MACEAutogradBatchCalc loaded MACE-OFF23_medium r_max={bcalc.r_max} "
      f"Z={len(bcalc.atomic_numbers)} elems", flush=True)

# --------------- PART C: MLIP batch==sequential, all 3 segments ------------ #
top = Prmtop(PRMTOP)
lig_idx, rec_idx = top.ligand_receptor_masks("LIG")
Fc = 12
coords_all = read_nc_coords(NC, frames=Fc)        # (Fc, natom, 3)
print(f"[C] natom={top.natom} frames_used={coords_all.shape[0]} "
      f"lig={lig_idx.size} rec={rec_idx.size} q_lig={top.charges[lig_idx].sum():.3f}e",
      flush=True)
from ase import Atoms
syms_all = top.symbols


def seg_atoms(idx):
    syms = [syms_all[i] for i in idx]
    return [Atoms(symbols=syms, positions=coords_all[f][idx]) for f in range(Fc)]


# independent upstream MACE-OFF ASE calculator (different code path)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from mace.calculators.foundations_models import mace_off
try:
    ase_calc = mace_off(model=CACHE, device=DEV, default_dtype="float64",
                        return_raw_model=False)
    have_ase = True
except Exception as exc:
    print(f"[C] upstream mace_off ASE calc unavailable ({exc}); cross-check skipped",
          flush=True)
    ase_calc, have_ase = None, False

worst = 0.0
for tag, idx in [("ligand", lig_idx), ("receptor", rec_idx),
                 ("complex", np.arange(top.natom))]:
    al = seg_atoms(idx)
    bcalc.prepare(al)                              # B = Fc (ONE forward)
    Eb, _ = bcalc.get_ef_gpu()
    Eb = Eb.detach().cpu().numpy()
    Es = np.empty(Fc)                              # sequential, B=1 per frame
    for i, a in enumerate(al):
        bcalc.prepare([a])
        e, _ = bcalc.get_ef_gpu()
        Es[i] = float(e.detach().cpu()[0])
    dpar = float(np.max(np.abs(Eb - Es)))
    worst = max(worst, dpar)
    line = f"[C] {tag:9s}: batch-vs-seq max|d|={dpar:.3e} Ha"
    if have_ase:
        a0 = al[0].copy(); a0.calc = ase_calc
        dref = abs(Eb[0] - a0.get_potential_energy() * EV2HARTREE)
        line += f" ; batch-vs-upstream(f0) |d|={dref:.3e} Ha"
    print(line, flush=True)
    assert dpar < 1e-8, (tag, dpar)
print(f"[C] all-3-segment batch==sequential parity OK (worst {worst:.3e} Ha < 1e-8)",
      flush=True)

# --------------------- PART D: CMET end-to-end + timing ------------------- #
from ase.calculators.calculator import Calculator, all_changes


class HartreeWrap(Calculator):
    """Hartree-native ASE wrapper so EndpointBinding's SERIAL path (which treats
    get_potential_energy() as Hartree) gets correct units from the eV MACE-OFF calc."""
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


Fe = 16
MB = 8           # cap frames/forward for the 4331-atom complex (GPU-memory safe)

t0 = time.time()
eb_b = EndpointBinding(PRMTOP, NC, calc=bcalc, ligand_resname="LIG",
                       frames=Fe, mode="gb", output="/tmp/mmgbsa_b", max_batch=MB)
assert eb_b._batched is True
r_b = eb_b.run()
synchro()
t_b = time.time() - t0

t0 = time.time()
if have_ase:
    eb_s = EndpointBinding(PRMTOP, NC, calc=HartreeWrap(ase_calc), ligand_resname="LIG",
                           frames=Fe, mode="gb", output="/tmp/mmgbsa_s")
    assert eb_s._batched is False
    r_s = eb_s.run()
    t_s = time.time() - t0
else:
    r_s, t_s = None, float("nan")

print(f"[D] batched dG_bind={r_b['dG_bind_mean']:.4f} +/- {r_b['dG_bind_sem']:.4f} kcal/mol"
      f" (dE_int={r_b['dE_int_mean']:.3f}, ddG_solv={r_b['ddG_solv_mean']:.3f}, "
      f"F={r_b['n_frames']}, natom={r_b['n_atoms_complex']})", flush=True)
if r_s is not None:
    dd = abs(r_b["dG_bind_mean"] - r_s["dG_bind_mean"])
    print(f"[D] serial  dG_bind={r_s['dG_bind_mean']:.4f} kcal/mol | "
          f"batched-vs-serial dG d={dd:.3e} kcal/mol", flush=True)
    print(f"[D] wall: batched={t_b:.2f}s serial={t_s:.2f}s "
          f"speedup={t_s / t_b:.1f}x (F={Fe}, 3 segments, max_batch={MB})", flush=True)
    assert dd < 1e-3, dd
else:
    print(f"[D] batched wall={t_b:.2f}s (serial reference unavailable)", flush=True)
print("SMOKE-DONE", flush=True)
