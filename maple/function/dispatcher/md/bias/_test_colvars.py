"""Lib-free validation of the MAPLE side of the Colvars bias seam.

The real Colvars Python bindings are NOT installable on Ibex: `colvars` is not
on PyPI (`pip download colvars` -> "No matching distribution found ... versions:
none") or conda (`PackagesNotFoundError`), there is no Ibex `module` for it, and
upstream github.com/Colvars/colvars ships NO importable package exposing the
``Colvars()`` object API this wrapper drives -- only `misc_interfaces/python/
colvars.py`, a ctypes shim that loads scripting symbols from a host NAMD/VMD
executable and uses a string ``.run(cmd)`` API. So the real ABF/eABF CV engine
cannot be exercised here, and a genuine cross-backend GPU gate (MACE-OFF + a
second MLIP through the real lib) is BLOCKED until a pybind wrapper around
``colvarproxy_stub`` exists.

What CAN be validated without the lib is the half of the contract MAPLE OWNS at
the seam: the Ha<->kcal/mol unit conversion (HA_TO_KCAL), the once-per-step
drive, the bias-force SIGN, and the folding of the bias force into the inner
MLIP force. We inject a *fake* ``colvars`` module implementing an ANALYTIC
harmonic restraint on a distance CV in Colvars "real" units (kcal/mol, A), run
the actual, unmodified ``ColvarsCalculator``, and check its Ha/A output against
the closed form -k(xi-xi0)*dxi/dx. This mirrors plumed_calc.py's RESTRAINT
self-test but at the wrapper seam.

This is EXPLICITLY NOT a claim that real Colvars ABF/eABF works -- it validates
MAPLE's half of the contract only, and (via test 6) that the guard raises a
clean, PLUMED-pointing error when the real lib is absent. Pure numpy/ASE, no
MLIP, no GPU -> safe to run on a login node.
"""
import re
import sys
import types
import builtins
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.md.bias.colvars_calc import (
    ColvarsCalculator, HA_TO_KCAL)


def _install_fake_colvars():
    """Register a fake ``colvars`` module: an analytic harmonic restraint on the
    distance between atoms 1,2 (1-indexed per Colvars atomNumbers) in "real"
    units (kcal/mol, A). Exposes exactly the method names ColvarsCalculator
    probes, so the wrapper's real code path (probe -> drive -> convert -> fold)
    is exercised end to end."""
    mod = types.ModuleType("colvars")

    class Colvars:
        def __init__(self):
            self._k = None      # forceConstant, kcal/mol/A^2
            self._c = None      # centers xi0, A
            self._pos = None
            self._step = 0
            self._E = 0.0
            self._F = None

        def set_unit_system(self, name):
            assert name == "real", name

        def read_config_string(self, cfg):
            self._c = float(re.search(r"centers\s+([-\d.]+)", cfg).group(1))
            self._k = float(re.search(r"forceConstant\s+([-\d.]+)", cfg).group(1))

        def set_step(self, s):
            self._step = int(s)

        def set_positions(self, p):
            self._pos = np.asarray(p, float)

        def calc(self):
            r = self._pos[1] - self._pos[0]
            xi = np.linalg.norm(r)
            u = r / xi
            dxi = xi - self._c
            self._E = 0.5 * self._k * dxi ** 2          # kcal/mol
            f = np.zeros_like(self._pos)
            f[0] = +self._k * dxi * u                   # -dU/dr0, kcal/mol/A
            f[1] = -self._k * dxi * u                   # -dU/dr1
            self._F = f

        def get_energy(self):
            return float(self._E)

        def get_forces(self):
            return self._F

    mod.Colvars = Colvars
    sys.modules["colvars"] = mod


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


def main():
    k_kcal = 200.0     # forceConstant, kcal/mol/A^2
    xi0 = 1.0          # centers, A
    d0 = 2.0           # initial H-H distance, A
    cfg = ("colvar { name d "
           "distance { group1 { atomNumbers 1 } group2 { atomNumbers 2 } } } "
           "harmonic { colvars d centers %g forceConstant %g }" % (xi0, k_kcal))
    tol = 1e-8
    fa = k_kcal * (d0 - xi0) / HA_TO_KCAL   # analytic |bias force| on x, Ha/A
    results = []

    _install_fake_colvars()

    # Test 1 -- restraint bias-force vs analytic (zero inner force).
    at = Atoms('H2', positions=[[0, 0, 0], [d0, 0, 0]])
    at.calc = ColvarsCalculator(_ConstCalc(), cfg, timestep_fs=0.5,
                                temperature=300.0, atoms=at)
    f = at.get_forces()
    ok1 = (abs(f[0, 0] - fa) < tol and abs(f[1, 0] + fa) < tol
           and abs(f[0, 0] + f[1, 0]) < tol and f[0, 0] > 0 and f[1, 0] < 0)
    results.append(("restraint force vs analytic (Ha/A)",
                    "f0x=%.10f exp=%.10f f1x=%.10f sum=%.1e"
                    % (f[0, 0], fa, f[1, 0], f[0, 0] + f[1, 0]), ok1))

    # Test 2 -- bias energy conversion kcal/mol -> Ha.
    e = at.get_potential_energy()
    ea = 0.5 * k_kcal * (d0 - xi0) ** 2 / HA_TO_KCAL
    ok2 = abs(e - ea) < tol
    results.append(("restraint energy (Ha)", "E=%.10f exp=%.10f" % (e, ea), ok2))

    # Test 3 -- bias folds ADDITIVELY onto a nonzero inner MLIP force.
    base = np.array([[0.3, -0.1, 0.05], [-0.2, 0.4, -0.02]])
    at2 = Atoms('H2', positions=[[0, 0, 0], [d0, 0, 0]])
    at2.calc = ColvarsCalculator(_ConstCalc(base), cfg, timestep_fs=0.5,
                                 temperature=300.0, atoms=at2)
    f2 = at2.get_forces()
    exp = base.copy()
    exp[0, 0] += fa
    exp[1, 0] += -fa
    d3 = float(np.max(np.abs(f2 - exp)))
    ok3 = d3 < tol
    results.append(("bias folds onto inner force", "maxdiff=%.1e" % d3, ok3))

    # Test 4 -- once-per-step drive: istep advances exactly once per calculate.
    ok4 = (at2.calc._istep == 1)
    results.append(("per-step drive (istep==1 after 1 calc)",
                    "istep=%d" % at2.calc._istep, ok4))

    # Test 5 -- unit constant is the CODATA Ha->kcal/mol value.
    ok5 = abs(HA_TO_KCAL - 627.5094740631) < 1e-6
    results.append(("HA_TO_KCAL constant", "%r" % HA_TO_KCAL, ok5))

    # Test 6 -- guard raises a clean, PLUMED-pointing error when lib is absent.
    sys.modules.pop("colvars", None)
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "colvars":
            raise ModuleNotFoundError("No module named 'colvars'")
        return real_import(name, *a, **k)

    builtins.__import__ = blocked
    try:
        at3 = Atoms('H2', positions=[[0, 0, 0], [d0, 0, 0]])
        at3.calc = ColvarsCalculator(_ConstCalc(), cfg, timestep_fs=0.5,
                                     temperature=300.0, atoms=at3)
        try:
            at3.get_forces()
            msg, ok6 = "no error raised", False
        except RuntimeError as exc:
            s = str(exc)
            ok6 = ("PLUMED" in s and "colvars" in s.lower())
            msg = ("RuntimeError -> PLUMED backend" if ok6 else s[:60])
    finally:
        builtins.__import__ = real_import
    results.append(("guard raises actionable PLUMED-pointing error", msg, ok6))

    allok = True
    print("=" * 78)
    print("MAPLE Colvars-seam lib-free gate  (real Colvars bindings UNINSTALLABLE)")
    print("=" * 78)
    for name, detail, ok in results:
        allok = allok and ok
        print("[%s] %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
    print("=" * 78)
    print("OVERALL:", "PASS" if allok else "FAIL")
    print("(validates MAPLE's half of the seam only; real ABF/eABF CV engine "
          "unexercised -- BLOCKED on bindings)")
    return 0 if allok else 1


if __name__ == '__main__':
    raise SystemExit(main())
