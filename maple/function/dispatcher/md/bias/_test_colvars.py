"""Validation gate for the MAPLE Colvars bias seam -- now driving the REAL
Colvars CV/bias engine through the pybind11 ``colvars`` binding
(``colvars_ext/colvarsmodule.cpp``, a wrapper around ``colvarproxy_stub``).

Prior to the binding, ``colvars`` was uninstallable on Ibex and this file could
only validate MAPLE's half of the seam against a *fake* analytic module. That
lib-free check is retained here as the ``seam`` gate (it still runs with no
compiled lib, e.g. on a login node). The new gates below drive the REAL binding:

  restraint_s / restraint_c : the actual Colvars harmonic-restraint force on a
      distance CV, compared to the closed form -k(xi-xi0) d(xi)/dx (unit-correct,
      Ha/A at the seam) at a stretched and a compressed geometry (sign reversal).
  abf                       : a real Colvars **ABF** biased run on MACE-OFF (GPU)
      -- Colvars' distinctive feature -- checking that the ABF free-energy
      gradient / sample histogram ACCUMULATES across the 1D CV grid.
  cross_off / cross_mp      : cross-backend -- the SAME Colvars restraint folded
      onto MACE-OFF and mace-mp-0 (GPU); the bias force (total - inner) must
      equal the analytic form for BOTH, proving the position-space bias is
      backend-independent.

Process isolation
-----------------
``colvarmodule`` keeps a *static* proxy pointer, so only ONE ``colvars.Colvars``
may live per process (a second construction raises "trying to allocate the
collective variable module twice"). Each gate therefore runs in its OWN
subprocess (this file re-invoked with the gate name); the parent aggregates.

Run:  python _test_colvars.py            # parent: run all applicable gates
      python _test_colvars.py <gate>     # child : run one gate, print result
"""
import os
import subprocess
import sys

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.md.bias.colvars_calc import (
    ColvarsCalculator, HA_TO_KCAL)

MACEOFF = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
MACEMP = os.path.expanduser("~/.cache/mace/20231203mace128L1_epoch199model")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _ConstCalc(Calculator):
    """Minimal inner MLIP stand-in: returns a fixed force field (default 0)."""
    implemented_properties = ['energy', 'forces']
    SUPPORTS_PBC = False

    def __init__(self, base_forces=None):
        Calculator.__init__(self)
        self._base = base_forces

    def calculate(self, atoms=None, properties=('energy', 'forces'),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        n = len(self.atoms)
        f = (np.zeros((n, 3)) if self._base is None
             else np.asarray(self._base, float).copy())
        self.results = {'energy': 0.0, 'forces': f}


def _restraint_cfg(xi0, k_kcal, a1=1, a2=2):
    # Colvars config is newline-delimited: a keyword's value runs to end-of-line,
    # so top-level statements MUST be on separate lines. A single space-joined
    # line makes `name`'s value swallow the rest of the block -> "multiple values
    # not allowed for keyword name" (read_config_string code 5).
    return ("colvar {\n"
            "  name d\n"
            "  distance {\n"
            "    group1 { atomNumbers %d }\n"
            "    group2 { atomNumbers %d }\n"
            "  }\n"
            "}\n"
            "harmonic {\n"
            "  colvars d\n"
            "  centers %g\n"
            "  forceConstant %g\n"
            "}\n"
            % (a1, a2, xi0, k_kcal))


def _emit(name, ok, detail):
    # Colvars' C++ stdout is buffered independently of Python's and can leave the
    # cursor mid-line; lead with a flushed newline so the GATE record is isolated
    # on its own line, and flush so it lands before any C++ teardown output.
    sys.stdout.flush()
    print("\nGATE|%s|%s|%s" % (name, "PASS" if ok else "FAIL", detail), flush=True)
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# child gates
# --------------------------------------------------------------------------- #
def gate_seam():
    """Lib-free seam validation against a FAKE analytic colvars module.
    Validates MAPLE's half of the contract with no compiled lib: Ha<->kcal/mol
    conversion, once-per-step drive, bias-force sign/fold, and the actionable
    guard when the real lib is absent."""
    import re
    import types
    import builtins

    def _install_fake_colvars():
        mod = types.ModuleType("colvars")

        class Colvars:
            def __init__(self):
                self._k = self._c = self._pos = None
                self._E = 0.0
                self._F = None

            def set_unit_system(self, name):
                assert name == "real", name

            def read_config_string(self, cfg):
                self._c = float(re.search(r"centers\s+([-\d.]+)", cfg).group(1))
                self._k = float(re.search(r"forceConstant\s+([-\d.]+)", cfg).group(1))

            def set_step(self, s):
                pass

            def set_positions(self, p):
                self._pos = np.asarray(p, float)

            def calc(self):
                r = self._pos[1] - self._pos[0]
                xi = np.linalg.norm(r)
                u = r / xi
                dxi = xi - self._c
                self._E = 0.5 * self._k * dxi ** 2
                f = np.zeros_like(self._pos)
                f[0] = +self._k * dxi * u
                f[1] = -self._k * dxi * u
                self._F = f

            def get_energy(self):
                return float(self._E)

            def get_forces(self):
                return self._F

        mod.Colvars = Colvars
        sys.modules["colvars"] = mod

    k_kcal, xi0, d0 = 200.0, 1.0, 2.0
    cfg = _restraint_cfg(xi0, k_kcal)
    tol = 1e-8
    fa = k_kcal * (d0 - xi0) / HA_TO_KCAL
    ok = True

    _install_fake_colvars()
    at = Atoms('H2', positions=[[0, 0, 0], [d0, 0, 0]])
    at.calc = ColvarsCalculator(_ConstCalc(), cfg, 0.5, 300.0, atoms=at)
    f = at.get_forces()
    ok &= (abs(f[0, 0] - fa) < tol and abs(f[1, 0] + fa) < tol
           and abs(f[0, 0] + f[1, 0]) < tol and f[0, 0] > 0 and f[1, 0] < 0)
    e = at.get_potential_energy()
    ok &= abs(e - 0.5 * k_kcal * (d0 - xi0) ** 2 / HA_TO_KCAL) < tol

    base = np.array([[0.3, -0.1, 0.05], [-0.2, 0.4, -0.02]])
    at2 = Atoms('H2', positions=[[0, 0, 0], [d0, 0, 0]])
    at2.calc = ColvarsCalculator(_ConstCalc(base), cfg, 0.5, 300.0, atoms=at2)
    f2 = at2.get_forces()
    exp = base.copy(); exp[0, 0] += fa; exp[1, 0] += -fa
    ok &= float(np.max(np.abs(f2 - exp))) < tol
    ok &= (at2.calc._istep == 1)
    ok &= abs(HA_TO_KCAL - 627.5094740631) < 1e-6

    sys.modules.pop("colvars", None)
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "colvars":
            raise ModuleNotFoundError("No module named 'colvars'")
        return real_import(name, *a, **k)

    builtins.__import__ = blocked
    try:
        at3 = Atoms('H2', positions=[[0, 0, 0], [d0, 0, 0]])
        at3.calc = ColvarsCalculator(_ConstCalc(), cfg, 0.5, 300.0, atoms=at3)
        try:
            at3.get_forces()
            ok = False
        except RuntimeError as exc:
            s = str(exc)
            ok &= ("PLUMED" in s and "colvars" in s.lower())
    finally:
        builtins.__import__ = real_import

    return _emit("seam(lib-free fake)", ok, "MAPLE-half: conv+drive+fold+guard")


def _restraint_real(d0, k_kcal=200.0, xi0=1.0):
    """Drive the REAL Colvars binding: harmonic restraint on an H-H distance."""
    cfg = _restraint_cfg(xi0, k_kcal)
    fa = k_kcal * (d0 - xi0) / HA_TO_KCAL            # analytic |force| on x, Ha/A
    at = Atoms('H2', positions=[[0, 0, 0], [d0, 0, 0]])
    at.calc = ColvarsCalculator(_ConstCalc(), cfg, 0.5, 300.0, atoms=at)
    f = at.get_forces()
    e = at.get_potential_energy()
    ea = 0.5 * k_kcal * (d0 - xi0) ** 2 / HA_TO_KCAL
    tol = 1e-8
    ok = (abs(f[0, 0] - fa) < tol and abs(f[1, 0] + fa) < tol
          and abs(f[0, 0] + f[1, 0]) < tol and abs(e - ea) < tol
          and np.max(np.abs(f[:, 1:])) < tol)
    sign = "+" if (d0 - xi0) > 0 else "-"
    return ok, ("d0=%.2f f0x=%.10f exp=%.10f E=%.3e expE=%.3e sign(%s)"
                % (d0, f[0, 0], fa, e, ea, sign))


def gate_restraint_s():
    ok, det = _restraint_real(2.0)     # stretched: force pulls atoms together (+x on atom0)
    return _emit("restraint force vs analytic (stretched)", ok, det)


def gate_restraint_c():
    ok, det = _restraint_real(0.7)     # compressed (<center): force reverses sign
    return _emit("restraint force vs analytic (compressed, sign flip)", ok, det)


def _load_backend(which):
    import torch
    torch.set_default_dtype(torch.float64)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if which == 'off':
        from maple.function.calculator.generic._mace_off_generic import (
            MACEOFFGenericCalculator)
        return MACEOFFGenericCalculator(dev, model_path=MACEOFF), dev
    from maple.function.calculator.generic._mace_mp_generic import (
        MACEMPGenericCalculator)
    return MACEMPGenericCalculator(dev, model_path=MACEMP), dev


def _water_dimer(sep=2.9):
    from ase.build import molecule
    w1 = molecule('H2O')
    w2 = molecule('H2O')
    w2.positions = w2.positions + np.array([sep, 0.0, 0.0])
    return w1 + w2                                   # O0 H1 H2  O3 H4 H5


def gate_abf():
    """Real Colvars ABF on MACE-OFF (GPU): biased O-O distance run; check the
    ABF sample histogram / free-energy gradient ACCUMULATES across the grid."""
    import tempfile
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units

    inner, dev = _load_backend('off')
    at = _water_dimer(2.9)
    lo, hi, w = 2.4, 3.8, 0.1
    cfg = ("colvar {\n"
           "  name doo\n"
           "  width %g\n"
           "  lowerBoundary %g\n"
           "  upperBoundary %g\n"
           "  distance {\n"
           "    group1 { atomNumbers 1 }\n"
           "    group2 { atomNumbers 4 }\n"
           "  }\n"
           "}\n"
           "abf {\n"
           "  name myabf\n"
           "  colvars doo\n"
           "  fullSamples 40\n"
           "}\n" % (w, lo, hi))
    at.calc = ColvarsCalculator(inner, cfg, 1.0, 400.0, atoms=at)

    tmp = tempfile.mkdtemp(prefix="colvars_abf_")
    prefix = os.path.join(tmp, "run")
    _ = at.get_forces()                              # lazily builds at.calc._cv
    at.calc._cv.set_output_prefix(prefix)

    MaxwellBoltzmannDistribution(at, temperature_K=400.0)
    dyn = Langevin(at, 1.0 * units.fs, temperature_K=400.0, friction=0.02)
    nsteps = 1500
    max_abf_f = 0.0
    for _i in range(nsteps):
        dyn.run(1)
        max_abf_f = max(max_abf_f, float(np.max(np.abs(at.get_forces()))))
    at.calc._cv.write_output_files()

    cnt = os.path.join(tmp, "run.myabf.count")
    grad = os.path.join(tmp, "run.myabf.grad")
    counts = []
    if os.path.exists(cnt):
        with open(cnt) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln or ln.startswith('#'):
                    continue
                counts.append(float(ln.split()[-1]))
    counts = np.asarray(counts) if counts else np.zeros(1)
    total = float(counts.sum())
    populated = int((counts > 0).sum())
    grad_written = os.path.exists(grad)
    finite_forces = np.isfinite(at.get_forces()).all()
    ok = (grad_written and finite_forces and populated >= 2
          and total > 0.4 * nsteps)
    return _emit("ABF smoke (MACE-OFF GPU): histogram accumulates",
                 ok, ("dev=%s bins_populated=%d total_samples=%.0f/%d "
                      "max|F|=%.4f grad_file=%s"
                      % (dev, populated, total, nsteps, max_abf_f, grad_written)))


def _cross_backend(which):
    """Same Colvars harmonic restraint on H2O; bias = total - inner must equal
    the analytic -k(xi-xi0)u for the given backend (backend-independent)."""
    from ase.build import molecule
    inner, dev = _load_backend(which)
    at = molecule('H2O')                             # O0 H1 H2
    k, c = 100.0, 1.2
    cfg = _restraint_cfg(c, k, a1=1, a2=2)           # distance O(1)-H(2)
    at.calc = ColvarsCalculator(inner, cfg, 1.0, 300.0, atoms=at)
    f_total = at.get_forces()
    f_inner = np.asarray(at.calc.inner.results['forces'], float)
    bias = f_total - f_inner                         # Ha/A

    pos = at.get_positions()
    r = pos[1] - pos[0]
    xi = np.linalg.norm(r)
    u = r / xi
    dxi = xi - c
    exp = np.zeros_like(pos)
    exp[0] = +k * dxi * u / HA_TO_KCAL
    exp[1] = -k * dxi * u / HA_TO_KCAL
    d = float(np.max(np.abs(bias - exp)))
    finite = np.isfinite(f_total).all()
    ok = finite and d < 1e-7
    return ok, ("dev=%s xi=%.4f maxdiff(bias-analytic)=%.2e finite=%s"
                % (dev, xi, d, finite))


def gate_cross_off():
    ok, det = _cross_backend('off')
    return _emit("cross-backend bias vs analytic (MACE-OFF)", ok, det)


def gate_cross_mp():
    ok, det = _cross_backend('mp')
    return _emit("cross-backend bias vs analytic (mace-mp-0)", ok, det)


GATES = {
    'seam': gate_seam,
    'restraint_s': gate_restraint_s,
    'restraint_c': gate_restraint_c,
    'abf': gate_abf,
    'cross_off': gate_cross_off,
    'cross_mp': gate_cross_mp,
}


# --------------------------------------------------------------------------- #
# parent
# --------------------------------------------------------------------------- #
def _real_colvars_available():
    try:
        import colvars  # noqa: F401
        return hasattr(colvars, 'Colvars')
    except Exception:
        return False


def _cuda_and_mace():
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        import mace  # noqa: F401
        return os.path.exists(MACEOFF) and os.path.exists(MACEMP)
    except Exception:
        return False


def main():
    real = _real_colvars_available()
    gpu = _cuda_and_mace()

    plan = ['seam']
    if real:
        plan += ['restraint_s', 'restraint_c']
        if gpu:
            plan += ['abf', 'cross_off', 'cross_mp']
    skipped = [g for g in GATES if g not in plan]

    print("=" * 78)
    print("MAPLE Colvars REAL-binding gate  (real colvars=%s, gpu+mace=%s)"
          % (real, gpu))
    print("=" * 78)

    results = []
    for g in plan:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), g],
                           capture_output=True, text=True)
        line = ""
        for ln in p.stdout.splitlines():
            idx = ln.find("GATE|")           # tolerate Colvars C++ stdout merged
            if idx >= 0:                      # onto the same line as the record
                line = ln[idx:]
        if not line:
            results.append((g, False, "no GATE line; rc=%d\n%s"
                            % (p.returncode, (p.stderr or p.stdout)[-800:])))
            continue
        _, name, verdict, detail = line.split("|", 3)
        results.append((name, verdict == "PASS", detail))

    allok = True
    for name, ok, detail in results:
        allok = allok and ok
        print("[%s] %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
    for g in skipped:
        why = "no real colvars binding" if not real else "no GPU/MACE"
        print("[SKIP] %-46s %s" % (g, why))
    print("=" * 78)
    print("OVERALL:", "PASS" if allok else "FAIL",
          "(gates run: %d, skipped: %d)" % (len(plan), len(skipped)))
    if not real:
        print("NOTE: real colvars binding not importable -> only lib-free seam "
              "ran. Build colvars_ext (see colvars_ext/README.md) to run the "
              "REAL ABF/eABF + cross-backend gates.")
    return 0 if allok else 1


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in GATES:
        raise SystemExit(GATES[sys.argv[1]]())
    raise SystemExit(main())
