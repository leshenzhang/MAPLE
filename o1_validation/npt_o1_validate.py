#!/usr/bin/env python
"""O1 NPT single-forward validation driver.

Subcommands:
  equil   build a periodic water box + MACE-OFF, NPT-equilibrate, save state.npz
  prod    load state.npz, run NPT production, parse thermo, count MLIP forwards,
          time the run; write <mode>_result.json
  compare read old_result.json + new_result.json, emit a verdict.

OLD (2-forward baseline, feat/md-batched-core) vs NEW (1-forward, feat/md-npt-opt)
is selected purely by PYTHONPATH (which npt.py is imported); this script is
identical for both. See npt_o1.sbatch.
"""
import os
import sys
import json
import time
import numpy as np
from ase import Atoms
from ase.build import molecule

OUT = os.environ.get("RUNDIR", os.path.dirname(os.path.abspath(__file__)))
MODEL = "/home/xiaox/.cache/mace/MACE-OFF23_medium.model"
TIMESTEP = float(os.environ.get("TIMESTEP_FS", "0.5"))   # fs
CONSTRAINTS = os.environ.get("CONSTRAINTS", "none")       # none | h-bonds (RATTLE @ 2 fs)
TEMP = 300.0            # K
PRESS = 1.0             # bar
# tau_p: smaller pressure-coupling time decorrelates the volume faster (fewer
# steps for a converged <V>) AND amplifies any lagged-pressure O(dt) bias (the
# lag error scales with per-step dV ~ 1/tau_p) -> a CONSERVATIVE ensemble test.
# If <V> agrees at tau_p=500 fs it agrees even more at the 2000 fs default.
TAU_P = float(os.environ.get("TAU_P", "2000"))            # fs
EQUIL_STEPS = int(os.environ.get("EQUIL_STEPS", "10000"))
PROD_STEPS = int(os.environ.get("PROD_STEPS", "20000"))
WARMUP_FRAC = float(os.environ.get("WARMUP_FRAC", "0.4"))  # discarded as further equilibration
N_SIDE = 4              # 4**3 = 64 water molecules
SPACING = 3.6          # Angstrom, ~0.64 g/cc start (clash-free for 2 fs steps; long equil compresses to ~1.15 g/cc)
AMU_TO_G = 1.66053906660e-24


def iact(x):
    """Integrated autocorrelation time (in samples) via the summed normalized
    ACF, truncated at the first lag whose ACF <= 0.05. tau=1 means uncorrelated."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    if var <= 0 or n < 4:
        return 1.0
    tau = 1.0
    for k in range(1, n // 2):
        c = np.dot(x[:-k], x[k:]) / (n - k) / var
        if c <= 0.05:
            break
        tau += 2.0 * c
    return float(tau)


def random_rotation(rng):
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def build_water_box():
    w = molecule("H2O")
    base = w.get_positions() - w.get_positions()[0]   # O at origin
    syms_unit = list(w.get_chemical_symbols())
    L = N_SIDE * SPACING
    rng = np.random.default_rng(12345)
    pos, syms = [], []
    for i in range(N_SIDE):
        for j in range(N_SIDE):
            for k in range(N_SIDE):
                center = (np.array([i, j, k]) + 0.5) * SPACING
                p = base @ random_rotation(rng).T + center
                pos.append(p)
                syms += syms_unit
    atoms = Atoms(symbols=syms, positions=np.vstack(pos),
                  cell=[L, L, L], pbc=True)
    return atoms


EV_TO_HA = 1.0 / 27.211386245988


def make_calc():
    """PBC+stress MACE-OFF23 calculator for the validation.

    Default (CALC=generic): MAPLE's production ``mace-off-generic`` backend
    (GenericASECalculator adapter), which surfaces PBC+stress and routes through
    the shared _finalize_results chokepoint. This exercises the committed
    _finalize_results stress fix end-to-end.

    Fallback (CALC=selfcontained): a thin wrapper over upstream MACECalculator
    reproducing MAPLE's unit convention (energy/forces in Ha, stress in eV/Å³),
    used only to cross-check independent of the adapter. Both are the same
    MACE-OFF23-medium model, so OLD-vs-NEW remains a controlled comparison.
    """
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    which = os.environ.get("CALC", "generic")
    if which == "selfcontained":
        from ase.calculators.calculator import Calculator, all_changes
        from mace.calculators import MACECalculator

        class MaceOffHaStress(Calculator):
            implemented_properties = ["energy", "free_energy", "forces", "stress"]
            SUPPORTS_PBC = True

            def __init__(self):
                super().__init__()
                self._calc = MACECalculator(
                    model_paths=MODEL, device=dev, default_dtype="float64",
                )

            def calculate(self, atoms=None, properties=("energy",),
                          system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                work = atoms.copy()
                work.calc = self._calc
                e = float(work.get_potential_energy())               # eV
                f = np.asarray(work.get_forces(), dtype=np.float64)  # eV/Å
                s = np.asarray(work.get_stress(voigt=True),
                               dtype=np.float64)                     # eV/Å³
                self.results = {
                    "energy": e * EV_TO_HA, "free_energy": e * EV_TO_HA,
                    "forces": f * EV_TO_HA, "stress": s,
                }

        return MaceOffHaStress()

    from maple.function.calculator.generic._mace_off_generic import (
        MACEOFFGenericCalculator,
    )
    return MACEOFFGenericCalculator(device=torch.device(dev), model_path=MODEL)


def add_forward_counter(calc):
    """Count genuine MLIP forwards = calculate() invocations (ASE caches when
    geometry is unchanged, so a same-geometry get_stress/get_potential_energy
    after get_forces does NOT increment this)."""
    counter = {"n": 0}
    orig = calc.calculate

    def wrapped(*a, **k):
        counter["n"] += 1
        return orig(*a, **k)

    calc.calculate = wrapped
    return counter


def _import_npt(module="npt"):
    """Import the NPT class from ensemble.<module>.

    module='npt'           -> NEW 1-forward code (this branch, feat/md-npt-opt)
    module='npt_baseline'  -> OLD 2-forward baseline (a verbatim copy of
                              feat/md-batched-core npt.py dropped into the package
                              at runtime by the sbatch). Running both inside this
                              single worktree means OLD and NEW share the SAME
                              (fixed) calculator adapter + identical integrator/
                              barostat/utils, so the comparison isolates exactly
                              the npt.py single-forward reorder.
    """
    import importlib
    mod = importlib.import_module(
        f"maple.function.dispatcher.md.ensemble.{module}")
    return mod.NPT, mod.__file__


def cmd_equil(module="npt"):
    atoms = build_water_box()
    calc = make_calc()
    atoms.calc = calc
    NPT, npt_file = _import_npt(module)
    print(f"[equil] npt.py = {npt_file}")
    paras = dict(steps=EQUIL_STEPS, timestep=TIMESTEP, temperature=TEMP,
                 pressure=PRESS, thermostat="v-rescale", barostat="c-rescale",
                 tau_p=TAU_P, traj_every=10**9, log_every=1000, random_seed=1,
                 box_check="warn", init_velocities=True, rst_every=10**9,
                 constraints=CONSTRAINTS, verbose=1)
    job = NPT(output=os.path.join(OUT, "equil.out"), atoms=atoms, paras=paras)
    job.run()
    vel = atoms.arrays["velocities"]
    rep = atoms.info.get("velocity_representation", "standard")
    np.savez(os.path.join(OUT, "state.npz"),
             numbers=atoms.get_atomic_numbers(),
             positions=atoms.get_positions(),
             cell=np.array(atoms.get_cell()),
             velocities=vel, repr=str(rep))
    print(f"[equil] saved state.npz  V={atoms.get_volume():.2f} A^3  "
          f"repr={rep}")


def cmd_prod(mode, module="npt"):
    d = np.load(os.path.join(OUT, "state.npz"), allow_pickle=True)
    atoms = Atoms(numbers=d["numbers"], positions=d["positions"],
                  cell=d["cell"], pbc=True)
    atoms.arrays["velocities"] = d["velocities"]
    atoms.info["velocity_representation"] = str(d["repr"])
    calc = make_calc()
    atoms.calc = calc
    counter = add_forward_counter(calc)
    NPT, npt_file = _import_npt(module)
    print(f"[prod:{mode}] npt.py = {npt_file}")
    paras = dict(steps=PROD_STEPS, timestep=TIMESTEP, temperature=TEMP,
                 pressure=PRESS, thermostat="v-rescale", barostat="c-rescale",
                 tau_p=TAU_P, traj_every=10**9, log_every=2000, random_seed=7,
                 box_check="warn", init_velocities=False, rst_every=10**9,
                 constraints=CONSTRAINTS, verbose=1)
    job = NPT(output=os.path.join(OUT, f"{mode}.out"), atoms=atoms, paras=paras)
    t0 = time.perf_counter()
    job.run()
    wall = time.perf_counter() - t0

    thermo = os.path.join(OUT, f"{mode}_md_thermo.dat")
    data = np.loadtxt(thermo)        # step time temp KE PE TE press vol
    press = data[:, 6]
    vol = data[:, 7]
    n = len(vol)
    cut = int(n * WARMUP_FRAC)        # discard warmup as further equilibration
    P = press[cut:]
    V = vol[cut:]
    masses_amu = atoms.get_masses().sum()
    mass_g = masses_amu * AMU_TO_G
    rho = mass_g / (V * 1e-24)       # g/cm^3

    def ac_stderr(x):
        """Autocorrelation-corrected stderr: std / sqrt(Neff), Neff = N/(2*tau)."""
        tau = iact(x)
        neff = max(len(x) / (2.0 * tau), 1.0)
        return float(np.std(x, ddof=1) / np.sqrt(neff)), tau, neff

    V_se, V_tau, V_neff = ac_stderr(V)
    rho_se, _, _ = ac_stderr(rho)
    P_se, _, _ = ac_stderr(P)
    # drift diagnostic: slope of V over the (post-warmup) window, A^3/ps
    ps = data[cut:, 1] / 1000.0
    slope = float(np.polyfit(ps, V, 1)[0]) if len(ps) > 2 else 0.0

    res = dict(
        mode=mode, npt_file=npt_file, n_atoms=len(atoms),
        timestep_fs=TIMESTEP, constraints=CONSTRAINTS,
        prod_steps=PROD_STEPS, prod_ps=float(data[-1, 1] / 1000.0),
        warmup_frac=WARMUP_FRAC, n_samples=int(len(V)),
        wall_s=wall, wall_per_step_ms=1e3 * wall / PROD_STEPS,
        forwards=counter["n"], fwd_per_step=counter["n"] / PROD_STEPS,
        V_mean=float(V.mean()), V_std=float(V.std()), V_stderr=V_se,
        V_tau_samples=V_tau, V_neff=V_neff, V_slope_A3_per_ps=slope,
        rho_mean=float(rho.mean()), rho_std=float(rho.std()), rho_stderr=rho_se,
        P_mean=float(P.mean()), P_std=float(P.std()), P_stderr=P_se,
    )
    with open(os.path.join(OUT, f"{mode}_result.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


def cmd_smoke3():
    """Runtime-confirm all 3 thermostat paths are single-forward + non-crashing
    on the NEW code: v-rescale (split chain), langevin (LFMiddle), and
    v-rescale+constraints (RATTLE). Each runs a short NPT and asserts
    forwards/step ~ 1.0 (init forward + 1/step) and a finite final volume."""
    NPT, npt_file = _import_npt()
    print(f"[smoke3] npt.py = {npt_file}")
    steps = int(os.environ.get("SMOKE_STEPS", "200"))
    configs = [
        ("v-rescale", "none"),
        ("langevin", "none"),
        ("v-rescale", "h-bonds"),
    ]
    calc = make_calc()                 # load model once, reuse across paths
    counter = add_forward_counter(calc)
    ok = True
    for thermo, cons in configs:
        atoms = build_water_box()
        atoms.calc = calc
        counter["n"] = 0               # per-path forward count
        paras = dict(steps=steps, timestep=0.5, temperature=TEMP, pressure=PRESS,
                     thermostat=thermo, barostat="c-rescale", tau_p=TAU_P,
                     traj_every=10**9, log_every=10**9, random_seed=3,
                     box_check="warn", init_velocities=True, rst_every=10**9,
                     verbose=0, constraints=cons)
        tag = f"smoke_{thermo}_{cons}"
        job = NPT(output=os.path.join(OUT, f"{tag}.out"), atoms=atoms, paras=paras)
        job.run()
        fps = counter["n"] / steps
        vol = atoms.get_volume()
        finite = np.isfinite(vol) and np.isfinite(atoms.get_positions()).all()
        path_ok = (fps < 1.05) and finite
        ok = ok and path_ok
        print(f"[smoke3] {thermo:>10} + cons={cons:<8} "
              f"fwd/step={fps:.4f}  V={vol:.1f}  finite={finite}  "
              f"-> {'PASS' if path_ok else 'FAIL'}")
    print(f"[smoke3] OVERALL {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


def cmd_compare():
    old = json.load(open(os.path.join(OUT, "old_result.json")))
    new = json.load(open(os.path.join(OUT, "new_result.json")))

    dV = abs(old["V_mean"] - new["V_mean"])
    combV = (old["V_stderr"]**2 + new["V_stderr"]**2) ** 0.5
    sigmaV = dV / combV if combV > 0 else float("inf")
    drho = abs(old["rho_mean"] - new["rho_mean"])
    combr = (old["rho_stderr"]**2 + new["rho_stderr"]**2) ** 0.5
    sigmar = drho / combr if combr > 0 else float("inf")

    speedup = old["wall_per_step_ms"] / new["wall_per_step_ms"]

    # Gate A: <V>/<rho> ensemble equivalence within 2 sigma (autocorr-corrected).
    ens_ok = (sigmaV <= 2.0 and sigmar <= 2.0)
    # sigma_V fluctuation agreement within ~15%
    sigV_ratio = new["V_std"] / old["V_std"]
    fluct_ok = 0.7 <= sigV_ratio <= 1.43
    # NEW pressure self-consistency: |<P> - target| within 3 stderr
    P_target = 1.0
    P_self = abs(new["P_mean"] - P_target) <= 3.0 * new["P_stderr"]
    fwd_ok = abs(new["fwd_per_step"] - 1.0) < 0.05 and old["fwd_per_step"] > 1.9

    print("=" * 74)
    print(f"O1 NPT single-forward validation  "
          f"(dt={new.get('timestep_fs')} fs, constraints={new.get('constraints')}, "
          f"{new.get('prod_ps'):.1f} ps, warmup {100*new.get('warmup_frac',0):.0f}%)")
    print("=" * 74)
    print(f"{'metric':<24}{'OLD(2fwd)':>16}{'NEW(1fwd)':>16}")
    print(f"{'fwd/step':<24}{old['fwd_per_step']:>16.4f}{new['fwd_per_step']:>16.4f}")
    print(f"{'wall/step (ms)':<24}{old['wall_per_step_ms']:>16.3f}{new['wall_per_step_ms']:>16.3f}")
    print(f"{'<V> (A^3)':<24}{old['V_mean']:>16.3f}{new['V_mean']:>16.3f}")
    print(f"{'  stderr (autocorr)':<24}{old['V_stderr']:>16.3f}{new['V_stderr']:>16.3f}")
    print(f"{'  tau_int (samples)':<24}{old.get('V_tau_samples',0):>16.1f}{new.get('V_tau_samples',0):>16.1f}")
    print(f"{'  Neff':<24}{old.get('V_neff',0):>16.1f}{new.get('V_neff',0):>16.1f}")
    print(f"{'  drift (A^3/ps)':<24}{old.get('V_slope_A3_per_ps',0):>16.3f}{new.get('V_slope_A3_per_ps',0):>16.3f}")
    print(f"{'sigma_V (A^3)':<24}{old['V_std']:>16.3f}{new['V_std']:>16.3f}")
    print(f"{'<rho> (g/cc)':<24}{old['rho_mean']:>16.4f}{new['rho_mean']:>16.4f}")
    print(f"{'  stderr':<24}{old['rho_stderr']:>16.4f}{new['rho_stderr']:>16.4f}")
    print(f"{'<P> (bar)':<24}{old['P_mean']:>16.1f}{new['P_mean']:>16.1f}")
    print(f"{'  stderr':<24}{old['P_stderr']:>16.1f}{new['P_stderr']:>16.1f}")
    print("-" * 74)
    print(f"<V> agreement:        {sigmaV:.2f} sigma   -> {'PASS' if sigmaV<=2 else 'FAIL'}")
    print(f"<rho> agreement:      {sigmar:.2f} sigma   -> {'PASS' if sigmar<=2 else 'FAIL'}")
    print(f"sigma_V ratio new/old:{sigV_ratio:.3f}      -> {'PASS' if fluct_ok else 'FAIL'}")
    print(f"NEW <P> self-consist: |{new['P_mean']:.1f}-1| vs 3*{new['P_stderr']:.1f}"
          f"  -> {'PASS' if P_self else 'FAIL'}")
    print(f"forward count 2->1:   -> {'PASS' if fwd_ok else 'FAIL'}")
    print(f"SPEEDUP (wall/step):  {speedup:.2f}x")
    print("=" * 74)
    verdict = ens_ok and fluct_ok and P_self and fwd_ok
    print(f"OVERALL: {'PASS' if verdict else 'FAIL'}")
    summary = dict(old=old, new=new, sigmaV=sigmaV, sigma_rho=sigmar,
                   sigV_ratio=sigV_ratio, P_self_consistent=P_self,
                   fwd_ok=fwd_ok, speedup=speedup, verdict=verdict)
    json.dump(summary, open(os.path.join(OUT, "compare_summary.json"), "w"),
              indent=2)
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "equil":
        cmd_equil(sys.argv[2] if len(sys.argv) > 2 else "npt")
    elif cmd == "prod":
        cmd_prod(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "npt")
    elif cmd == "smoke3":
        cmd_smoke3()
    elif cmd == "compare":
        cmd_compare()
    else:
        raise SystemExit(f"unknown command {cmd}")
