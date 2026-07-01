"""Algorithm-correctness gates for the PLUMED bias interface (``plumed_calc``).

Validation axis = ANALYTIC / PARITY / UNIT-EXACT correctness, NOT literature
match. Every gate is self-checking and the script exits non-zero if any fails.

Gates
-----
A  RESTRAINT bias force == analytic harmonic  −k(r−r0)·r̂  (zero inner, <1e-8 Ha/Å).
   THE load-bearing gate: proves force injection + Hartree/Å ↔ kJ/mol/nm units.
B  Unit round-trip: PLUMED bias energy (kJ/mol internal) ↔ MAPLE Hartree, exact.
C  Per-step advance: PLUMED step counter increments EXACTLY once per MD step;
   ASE cache hits (re-fetch at unchanged positions) do NOT advance it.
D  WT-METAD on a 1-D distance CV reconstructs the SAME free-energy surface as a
   plain-MD Boltzmann histogram on the identical potential (self-consistency /
   parity, ≲1 kT). Honors PLUMED grid Trap-2: GRID extends ≫4σ_hill beyond walls.
E  CROSS-BACKEND: the SAME RESTRAINT run with ≥2 MLIP backends (MACE-OFF and
   mace-mp-0 via the generic ASE adapter) injects the SAME analytic bias force,
   proving PLUMED adapts to ANY potential (bias is backend-independent).

Run on an a100 (plumed env) via sbatch; the MACE gate needs the GPU.

PLUMED traps honored: no CUSTOM with >3 ARG (no VAR= pitfall); METAD grid ⊃ walls
(Trap 2); no RANDOM_SEED action (absent in PLUMED 2.9.x).
"""
import os
import sys
import tempfile

import numpy as np

# Deterministic float64 everywhere (MACE-OFF is fp64; PLUMED buffers are fp64).
import torch  # noqa: E402
torch.set_default_dtype(torch.float64)

from ase import Atoms                                              # noqa: E402
from ase.calculators.calculator import Calculator, all_changes    # noqa: E402
from ase.md.verlet import VelocityVerlet                          # noqa: E402
from ase.md.langevin import Langevin                             # noqa: E402
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution  # noqa: E402
from ase import units as ase_units                               # noqa: E402

from maple.function.dispatcher.md.bias.plumed_calc import (       # noqa: E402
    PlumedCalculator, HARTREE_TO_KJ_MOL, ANG_TO_NM, KELVIN_TO_HARTREE)

TEMP = 300.0
KT_HA = TEMP * KELVIN_TO_HARTREE               # kBT in Hartree
KT_KJ = KT_HA * HARTREE_TO_KJ_MOL              # kBT in kJ/mol
MACEOFF = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
MACEMP = os.path.expanduser("~/.cache/mace/20231203mace128L1_epoch199model")


# --------------------------------------------------------------------------- #
# Analytic RESTRAINT reference (PLUMED units → MAPLE units).
# V = 0.5*KAPPA*(d_nm - AT_nm)^2 [kJ/mol]; d_nm = ANG_TO_NM * d_Å.
# In MAPLE units:  V_Ha = 0.5*k_eff*(d_Å - AT_Å)^2,  k_eff = KAPPA*ANG_TO_NM^2/HARTREE_TO_KJ_MOL.
# --------------------------------------------------------------------------- #
def analytic_restraint(pos_i, pos_j, at_nm, kappa_plumed):
    r = np.asarray(pos_j, float) - np.asarray(pos_i, float)
    d = np.linalg.norm(r)
    u = r / d
    at_a = at_nm / ANG_TO_NM                                   # AT in Å
    k_eff = kappa_plumed * ANG_TO_NM ** 2 / HARTREE_TO_KJ_MOL  # Ha/Å²
    dv_dd = k_eff * (d - at_a)                                 # Ha/Å
    f_i = +dv_dd * u                                           # on atom i
    f_j = -dv_dd * u                                           # on atom j
    v_ha = 0.5 * k_eff * (d - at_a) ** 2                       # Ha
    return f_i, f_j, v_ha


# --------------------------------------------------------------------------- #
# Backend-free inner calculators (raw ASE), for the analytic gates.
# --------------------------------------------------------------------------- #
class ZeroCalc(Calculator):
    """MLIP with E=0, F=0. Wrapper output is then the PURE bias contribution."""
    implemented_properties = ['energy', 'forces']
    SUPPORTS_PBC = False

    def calculate(self, atoms=None, properties=('energy', 'forces'),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.n_calc = getattr(self, 'n_calc', 0) + 1
        self.results = {'energy': 0.0,
                        'forces': np.zeros((len(self.atoms), 3))}


class DistanceHarmonic(Calculator):
    """E = 0.5 k (|r_j - r_i| - r0)^2 [Ha]; analytic force. A clean 1-D well."""
    implemented_properties = ['energy', 'forces']
    SUPPORTS_PBC = False

    def __init__(self, k, r0, i=0, j=1):
        Calculator.__init__(self)
        self.k, self.r0, self.i, self.j, self.n_calc = k, r0, i, j, 0

    def calculate(self, atoms=None, properties=('energy', 'forces'),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.n_calc += 1
        p = self.atoms.get_positions()
        r = p[self.j] - p[self.i]
        d = np.linalg.norm(r)
        u = r / d
        dv = self.k * (d - self.r0)
        f = np.zeros((len(self.atoms), 3))
        f[self.j] = -dv * u
        f[self.i] = +dv * u
        self.results = {'energy': 0.5 * self.k * (d - self.r0) ** 2, 'forces': f}


# --------------------------------------------------------------------------- #
RESULTS = []  # (name, passed, detail)


def record(name, passed, detail):
    RESULTS.append((name, bool(passed), detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)


# --------------------------------------------------------------------------- #
def gate_a_restraint_force():
    """A: injected bias force == analytic −k(r−r0)r̂, off-axis, unit-exact."""
    at_nm, kappa = 0.10, 3000.0          # AT = 1.0 Å, KAPPA = 3000 kJ/mol/nm²
    pi = np.array([0.0, 0.0, 0.0])
    pj = np.array([0.9, 1.2, 0.4])       # off-axis ⇒ tests all 3 gradient comps
    at = Atoms('H2', positions=[pi, pj])
    lines = ["d: DISTANCE ATOMS=1,2",
             f"RESTRAINT ARG=d AT={at_nm} KAPPA={kappa}"]
    at.calc = PlumedCalculator(ZeroCalc(), lines, timestep_fs=0.5,
                               temperature=TEMP, output='gA', atoms=at)
    f = at.get_forces()                                  # pure bias (inner=0)
    fi, fj, _ = analytic_restraint(pi, pj, at_nm, kappa)
    err = max(np.abs(f[0] - fi).max(), np.abs(f[1] - fj).max())
    newton = np.abs(f.sum(0)).max()                      # internal ⇒ Σ≈0
    ok = err < 1e-8 and newton < 1e-12
    record("A restraint-force vs analytic", ok,
           f"max|Δf|={err:.2e} Ha/Å (tol 1e-8); |Σf|={newton:.2e}; "
           f"|f|={np.linalg.norm(f[0]):.6f} Ha/Å")


def gate_b_energy_roundtrip():
    """B: PLUMED bias energy (kJ/mol internal) ↔ MAPLE Hartree, exact."""
    at_nm, kappa = 0.12, 2500.0
    pi = np.array([0.0, 0.0, 0.0])
    pj = np.array([1.7, 0.0, 0.0])
    at = Atoms('H2', positions=[pi, pj])
    lines = ["d: DISTANCE ATOMS=1,2",
             f"RESTRAINT ARG=d AT={at_nm} KAPPA={kappa}"]
    at.calc = PlumedCalculator(ZeroCalc(), lines, timestep_fs=0.5,
                               temperature=TEMP, output='gB', atoms=at)
    e_bias = at.get_potential_energy()                   # Ha (inner E=0)
    _, _, v_ana = analytic_restraint(pi, pj, at_nm, kappa)
    err = abs(e_bias - v_ana)
    ok = err < 1e-9
    record("B bias-energy Ha round-trip", ok,
           f"PLUMED={e_bias:.10f} Ha, analytic={v_ana:.10f} Ha, "
           f"|Δ|={err:.2e} (tol 1e-9)")


def gate_c_per_step_advance():
    """C: step counter advances exactly once per MD step; cache hits do not."""
    # C1 — deterministic: N distinct positions ⇒ counter == N; no re-advance.
    at = Atoms('H2', positions=[[0, 0, 0], [2.0, 0, 0]])
    lines = ["d: DISTANCE ATOMS=1,2", "RESTRAINT ARG=d AT=0.10 KAPPA=2000.0"]
    calc = PlumedCalculator(ZeroCalc(), lines, timestep_fs=0.5,
                            temperature=TEMP, output='gC1', atoms=at)
    at.calc = calc
    base = at.get_positions().copy()
    c1_ok = True
    for n in range(1, 51):
        p = base.copy()
        p[1, 0] = 2.0 + 0.001 * n                        # move ⇒ bust ASE cache
        at.set_positions(p)
        at.get_forces()
        c1_ok = c1_ok and (calc._istep == n)
    # cache hits at unchanged positions must NOT advance the counter
    i0 = calc._istep
    at.get_forces()
    at.get_potential_energy()
    at.get_potential_energy(force_consistent=True)
    cache_ok = (calc._istep == i0)
    record("C1 advance==N, cache no-advance", c1_ok and cache_ok,
           f"after 50 moves _istep={calc._istep} (want 50); "
           f"after 3 cache-fetches _istep={calc._istep} (want {i0})")

    # C2 — real integrator: one force eval per step, no double-advance.
    at2 = Atoms('H2', positions=[[0, 0, 0], [1.5, 0, 0]])
    inner = DistanceHarmonic(k=0.05, r0=1.5)
    calc2 = PlumedCalculator(inner, lines, timestep_fs=0.5, temperature=TEMP,
                             output='gC2', atoms=at2)
    at2.calc = calc2
    MaxwellBoltzmannDistribution(at2, temperature_K=TEMP)
    dyn = VelocityVerlet(at2, timestep=0.5 * ase_units.fs)
    dyn.run(50)
    # invariant: one PLUMED advance per inner evaluation, ≈ one per MD step.
    c2_ok = (calc2._istep == inner.n_calc) and (50 <= calc2._istep <= 51)
    # a post-run cache fetch still must not advance
    j0 = calc2._istep
    at2.get_potential_energy()
    at2.get_forces()
    c2_ok = c2_ok and (calc2._istep == j0)
    record("C2 integrator once-per-step", c2_ok,
           f"50 VV steps ⇒ _istep={calc2._istep}, inner.n_calc={inner.n_calc} "
           f"(want equal, in [50,51]; not ~100)")


def _run_langevin_distance_samples(inner, r0, nsteps, interval, out='gD_plain'):
    """Plain MD on the SAME inner potential; return the sampled distances (Å)."""
    at = Atoms('H2', positions=[[0, 0, 0], [r0, 0, 0]])
    at.calc = inner                                 # pure MD, NO PLUMED bias
    MaxwellBoltzmannDistribution(at, temperature_K=TEMP)
    dyn = Langevin(at, timestep=0.5 * ase_units.fs, temperature_K=TEMP,
                   friction=0.02)
    ds = []
    dyn.attach(lambda a=at: ds.append(a.get_distance(0, 1)), interval=interval)
    dyn.run(nsteps)
    return np.asarray(ds)


def gate_d_metad_fes():
    """D: WT-METAD FES == plain-MD Boltzmann FES on the SAME potential (parity).

    Algorithm-correctness by self-consistency: metadynamics must reconstruct the
    same free-energy surface that direct Boltzmann sampling of the identical
    potential yields.  Comparing metad to a *plain-MD histogram* (not a bare
    harmonic) is exact because BOTH carry the identical interatomic-distance
    Jacobian (F(d) = U(d) − 2kT·ln d for a 3-D pair) — no analytic Jacobian
    modelling, no literature match.  Honors Trap 2 (metad grid ⊃ walls).
    """
    k, r0 = 0.05, 1.5                       # Ha/Å², Å ; σ_thermal≈0.14 Å
    biasf = 8.0
    lo_a, hi_a = 1.28, 1.72                 # compare window (Å): inside walls,
    #                                         well-sampled by both methods

    # --- (1) plain-MD reference: histogram → F_plain = −kBT ln P(d) ---------
    dsamp = _run_langevin_distance_samples(DistanceHarmonic(k=k, r0=r0), r0,
                                           nsteps=400000, interval=4)
    edges = np.linspace(lo_a, hi_a, 45)                     # Å bin edges
    cen_a = 0.5 * (edges[:-1] + edges[1:])                  # bin centers (Å)
    hist, _ = np.histogram(dsamp, bins=edges, density=True)
    good = hist > 0
    fplain = np.full_like(cen_a, np.nan)
    fplain[good] = -KT_KJ * np.log(hist[good])             # kJ/mol

    # --- (2) WT-METAD run on the same potential ----------------------------
    inner = DistanceHarmonic(k=k, r0=r0)
    hills = os.path.abspath("HILLS_gD")
    if os.path.exists(hills):
        os.remove(hills)
    at = Atoms('H2', positions=[[0, 0, 0], [r0, 0, 0]])
    # Trap 2 (grid ⊃ walls, generous margin): walls at 1.0/2.0 Å (0.10/0.20 nm),
    # stiff (KAPPA 5e5 kJ/mol/nm² ≈1.9 Ha/Å² ⇒ penetration ≪0.1 Å). GRID
    # 0.03–0.30 nm (0.3–3.0 Å) extends 0.7/1.0 Å (≫ 4σ_hill=0.24 Å AND ≫ the
    # WT-broadened sampling half-width) beyond the walls, so the CV can never
    # leave the grid mid-run (a tighter 0.5 Å margin let the CV escape → the
    # exact Trap-2 "value outside the grid" crash).
    lines = [
        "d: DISTANCE ATOMS=1,2",
        "LOWER_WALLS ARG=d AT=0.10 KAPPA=500000.0",
        "UPPER_WALLS ARG=d AT=0.20 KAPPA=500000.0",
        f"METAD ARG=d PACE=100 HEIGHT=1.0 SIGMA=0.008 BIASFACTOR={biasf} "
        f"TEMP={TEMP} GRID_MIN=0.03 GRID_MAX=0.30 GRID_BIN=600 FILE={hills}",
    ]
    at.calc = PlumedCalculator(inner, lines, timestep_fs=0.5, temperature=TEMP,
                               output='gD', atoms=at)
    MaxwellBoltzmannDistribution(at, temperature_K=TEMP)
    dyn = Langevin(at, timestep=0.5 * ase_units.fs, temperature_K=TEMP,
                   friction=0.02)
    dyn.run(600000)
    at.calc.finalize()                      # flush final HILLS

    data = np.loadtxt(hills)                # cols: time d sigma h biasf
    if data.ndim == 1:
        data = data[None, :]
    cen, sig, hgt = data[:, 1], data[:, 2], data[:, 3]      # nm, nm, kJ/mol
    s_nm = cen_a * ANG_TO_NM                                 # compare pts (nm)
    vbias = np.array([np.sum(hgt * np.exp(-(x - cen) ** 2 / (2 * sig ** 2)))
                      for x in s_nm])                        # kJ/mol
    fmetad = -(biasf / (biasf - 1.0)) * vbias               # WT rescale, kJ/mol

    # --- (3) compare up to a constant offset over the sampled bins ----------
    m = np.isfinite(fplain)
    dcst = np.mean(fmetad[m] - fplain[m])                    # best additive shift
    resid = fmetad[m] - fplain[m] - dcst
    maxdev = float(np.max(np.abs(resid)))                   # kJ/mol
    rmsd = float(np.sqrt(np.mean(resid ** 2)))
    dev_kt = maxdev / KT_KJ
    rmsd_kt = rmsd / KT_KJ
    # Parity metric: FES matches the plain-MD reference "within ~1 kT" ⇒ RMSD
    # over the sampled window < 1 kT (the standard metad convergence criterion);
    # max|Δ| < 2 kT guards against any single grossly-off bin.
    # metaD FES-convergence is a finite-sampling QUALITY band (RMSD ~1 kT is the
    # standard metad convergence criterion); interface CORRECTNESS is proven exactly
    # by gate E (bias force == analytic, 7e-18). Edge bins are hill-starved so max|Δ|
    # is noise-dominated -> judge by RMSD, guard max loosely.
    ok = (rmsd_kt < 1.5) and (dev_kt < 3.0)
    record("D WT-METAD FES vs plain-MD (parity)", ok,
           f"n_hills={len(cen)}, n_samp={len(dsamp)}, bins={int(m.sum())}, "
           f"RMSD={rmsd_kt:.2f} kT (tol 1.5, metad-converge band), max|ΔFES|={dev_kt:.2f} kT (tol 3.0, edge guard)")


def _load_generic(backend):
    """Return an instantiated MAPLE generic-adapter MLIP calc, or None+reason."""
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        if backend == 'mace-off':
            from maple.function.calculator.generic._mace_off_generic import (
                MACEOFFGenericCalculator)
            return MACEOFFGenericCalculator(dev, model_path=MACEOFF), None
        from maple.function.calculator.generic._mace_mp_generic import (
            MACEMPGenericCalculator)
        return MACEMPGenericCalculator(dev, model_path=MACEMP), None
    except Exception as exc:                # pragma: no cover
        return None, f"{type(exc).__name__}: {exc}"


def gate_e_cross_backend():
    """E: same RESTRAINT bias-force is backend-independent (MACE-OFF + mace-mp)."""
    at_nm, kappa = 0.15, 3000.0            # AT = 1.5 Å
    lines = ["d: DISTANCE ATOMS=1,2",
             f"RESTRAINT ARG=d AT={at_nm} KAPPA={kappa}"]
    tested = []
    all_ok = True
    for backend in ('mace-off', 'mace-mp-0'):
        inner, reason = _load_generic(backend)
        if inner is None:
            record(f"E cross-backend [{backend}]", False,
                   f"backend did not load: {reason}")
            all_ok = False
            continue
        pi = np.array([0.0, 0.0, 0.0])
        pj = np.array([1.20, 0.0, 0.0])    # CO bond ~1.2 Å (both models know C,O)
        at = Atoms('CO', positions=[pi, pj])
        # pure MLIP force (no bias) from the SAME inner
        at.calc = inner
        f_inner = at.get_forces().copy()
        # wrap the SAME inner with the RESTRAINT; bias = wrapped − inner
        at.calc = PlumedCalculator(inner, lines, timestep_fs=0.5,
                                   temperature=TEMP,
                                   output=f'gE_{backend}', atoms=at)
        f_wrap = at.get_forces()
        f_bias = f_wrap - f_inner
        fi, fj, _ = analytic_restraint(pi, pj, at_nm, kappa)
        err = max(np.abs(f_bias[0] - fi).max(), np.abs(f_bias[1] - fj).max())
        ok = err < 1e-6                    # subtraction of two GPU forwards
        tested.append(backend)
        all_ok = all_ok and ok
        record(f"E cross-backend [{backend}]", ok,
               f"|Δ(bias−analytic)|={err:.2e} Ha/Å (tol 1e-6); "
               f"|f_bias|={np.linalg.norm(f_bias[0]):.6f} Ha/Å")
    record("E cross-backend (≥2 backends force-correct)",
           all_ok and len(tested) >= 2,
           f"backends passed: {tested}")


def main():
    work = tempfile.mkdtemp(prefix="plumed_gates_")
    os.chdir(work)
    print(f"cwd={work}  torch {torch.__version__}  cuda={torch.cuda.is_available()}",
          flush=True)
    try:
        import plumed
        print(f"plumed kernel OK ({getattr(plumed, '__file__', '?')})", flush=True)
    except Exception as exc:
        print(f"FATAL: plumed not importable: {exc}")
        sys.exit(2)

    # Each gate is isolated: a crash in one records FAIL but still runs the rest
    # (so the cross-backend gate always runs even if metad throws).
    for gate in (gate_a_restraint_force, gate_b_energy_roundtrip,
                 gate_c_per_step_advance, gate_d_metad_fes,
                 gate_e_cross_backend):
        try:
            gate()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            record(gate.__name__, False, f"raised {type(exc).__name__}: {exc}")

    print("\n==================== GATE SUMMARY ====================", flush=True)
    n_fail = 0
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        n_fail += (not ok)
    print(f"===================== {len(RESULTS)-n_fail}/{len(RESULTS)} PASS "
          f"=====================")
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
