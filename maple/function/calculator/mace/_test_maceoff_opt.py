"""
B-81 PART 2 gates for the MACE-OFF batch-calculator rebuild optimization.

GATE A (HARD, abandon-if-fail): the OPTIMIZED MaceOffBatchCalc (cache-static +
on-GPU edge build + no per-step dtype toggle, from feat/md-r2fixes) must give E/F
EQUAL to the ORIGINAL calc (base b75ec55, loaded from the integ worktree) on the
SAME systems, both at the prepare geometry AND after a few step_cart_ moves
(coords change -> edges rebuilt). fp64 floor: energy < 1e-6 eV, force < 1e-5 eV/A.

GATE B (speedup): median per-step wall-time (get_ef_gpu + step_cart_) optimized vs
original on ethanol B=16/32 (small, launch-bound -> expect speedup) and a 192-atom
water cluster B=4 (large, compute-bound -> ~1x).

The original file has NO relative imports, so it is loaded directly by path.
"""
import os, sys, time, importlib.util
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from ase.build import molecule

EV = 27.211386245988
MODEL = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
DEV = "cuda"
ORIG_PATH = "/ibex/user/xiaox/zls/ai-maple-md/MAPLE/worktrees/integ/" \
            "maple/function/calculator/mace/_maceoff_batch_calculator.py"

# optimized (package import; r2fixes on PYTHONPATH)
from maple.function.calculator.mace._maceoff_batch_calculator import MaceOffBatchCalc as OptCalc

# original (base b75ec55) loaded by file path as a standalone module
_spec = importlib.util.spec_from_file_location("_maceoff_orig", ORIG_PATH)
_orig = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_orig)
OrigCalc = _orig.MaceOffBatchCalc


def ethanol(shift):
    at = molecule("CH3CH2OH"); at.positions = at.positions + np.asarray(shift); return at


def water_cluster(nx=4, ny=4, nz=4, a=3.2):
    """Isolated (non-periodic) cluster of nx*ny*nz water molecules (3 atoms each)."""
    from ase import Atoms
    base = molecule("H2O")
    allp, alls = [], []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                off = np.array([i * a, j * a, k * a])
                allp.append(base.get_positions() + off)
                alls += list(base.get_chemical_symbols())
    at = Atoms(symbols=alls, positions=np.concatenate(allp, 0))
    at.center(vacuum=0.0)
    return at


def _systems(kind):
    # Block-diagonal isolation comes from SEPARATE per-replica graphs, NOT spatial
    # separation -- so replicas are kept at small coordinates (no large per-molecule
    # offset). This avoids the original calc's pathological matscipy "very large cell"
    # slow path (cell ~ max|pos|*5*cutoff) while leaving the physics identical.
    if kind == "ethanol16":
        return [ethanol((0, 0, 0)) for _ in range(16)]
    if kind == "ethanol32":
        return [ethanol((0, 0, 0)) for _ in range(32)]
    if kind == "water192":
        w = water_cluster()
        assert len(w) == 192, len(w)
        return [w.copy() for _ in range(4)]
    raise ValueError(kind)


def parity(kind, nsteps=5, seed=0):
    systems = _systems(kind)
    opt = OptCalc(model_path=MODEL, device=DEV, dtype=torch.float64)
    org = OrigCalc(model_path=MODEL, device=DEV, dtype=torch.float64)
    opt.prepare(systems, fixed_nmax=None)
    org.prepare([s.copy() for s in systems], fixed_nmax=None)
    rng = np.random.default_rng(seed)
    worst_dE = worst_dF = 0.0
    for s in range(nsteps + 1):
        Eo, Fo = opt.get_ef_gpu(); Eg, Fg = org.get_ef_gpu()
        dE = float((Eo - Eg).abs().max().item()) * EV          # eV
        dF = float((Fo - Fg).abs().max().item()) * EV          # eV/A
        worst_dE = max(worst_dE, dE); worst_dF = max(worst_dF, dF)
        # identical small random displacement to both, then re-evaluate.
        B, nmax = opt.B, opt.nmax_dof
        disp = torch.tensor(rng.normal(0, 2e-3, size=(B, nmax)),
                            dtype=torch.float64, device=opt.device)
        opt.step_cart_(disp); org.step_cart_(disp)
    print(f"[PARITY {kind:10s}] over {nsteps+1} evals: max|dE|={worst_dE:.3e} eV  "
          f"max|dF|={worst_dF:.3e} eV/A")
    ok = worst_dE < 1e-6 and worst_dF < 1e-5
    return ok, worst_dE, worst_dF


def bench(kind, nsteps=60, warmup=10):
    systems = _systems(kind)
    def run(Calc):
        c = Calc(model_path=MODEL, device=DEV, dtype=torch.float64)
        c.prepare([s.copy() for s in systems], fixed_nmax=None)
        B, nmax = c.B, c.nmax_dof
        disp = torch.zeros((B, nmax), dtype=torch.float64, device=c.device)
        disp += 1e-4
        for _ in range(warmup):
            c.get_ef_gpu(); c.step_cart_(disp)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(nsteps):
            c.get_ef_gpu(); c.step_cart_(disp)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / nsteps * 1e3      # ms/step
    t_opt = run(OptCalc)
    t_org = run(OrigCalc)
    sp = t_org / t_opt if t_opt > 0 else float("nan")
    print(f"[BENCH {kind:10s}] orig={t_org:.3f} ms/step  opt={t_opt:.3f} ms/step  "
          f"speedup={sp:.2f}x")
    return t_org, t_opt, sp


if __name__ == "__main__":
    quick = (len(sys.argv) >= 2 and sys.argv[1] == "quick")
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
          "quick" if quick else "full", flush=True)
    print("\n========== GATE A: PARITY (abandon-if-fail) ==========", flush=True)
    allok = True
    res = {}
    for kind in ("ethanol16", "ethanol32", "water192"):
        ok, dE, dF = parity(kind)
        res[kind] = (ok, dE, dF); allok = allok and ok
        sys.stdout.flush()
    print("\nPARITY ALL PASS:", allok, flush=True)
    if not allok:
        print("[GATE A] FAIL -> Part 2 must be ABANDONED", flush=True)
        sys.exit(2)
    print("[GATE A] PASS", flush=True)
    print("\n========== GATE B: SPEEDUP ==========", flush=True)
    # quick mode caps the (matscipy-slow) original water192 bench cost so the run
    # flushes results well within walltime; ethanol regimes use the full count.
    bspec = {"ethanol16": (60, 10), "ethanol32": (60, 10),
             "water192": (12, 3) if quick else (60, 10)}
    for kind in ("ethanol16", "ethanol32", "water192"):
        ns, wu = bspec[kind]
        bench(kind, nsteps=ns, warmup=wu)
        sys.stdout.flush()
    print("\n[PART 2] GATES COMPLETE", flush=True)
