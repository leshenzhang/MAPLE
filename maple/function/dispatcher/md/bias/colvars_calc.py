"""Colvars bias interface for MAPLE MD (second mandated enhanced-sampling lib).

Same single-injection-point architecture as :mod:`plumed_calc`: a calculator
wrapper drives the Colvars module once per MD step at ``atoms.get_forces`` and
folds the bias force into the MLIP force, so NVE/NVT/NPT all gain Colvars biases
(harmonic restraint / ABF / eABF / metadynamics) with no integrator change.

PLUMED is the *primary, validated* backend (broadest CV set, ASE-tested unit
contract). Colvars is provided for its distinctive **eABF / multiple-walker
ABF** support (roadmap P1.3). It requires a ``colvars`` Python module exposing
the generic (host-less) proxy object API used below; upstream currently ships
no such pip/conda package (only a host-linked ctypes scripting shim), so a
pybind wrapper around ``colvarproxy_stub`` must be built before this backend is
usable. Absent that, ``_init_colvars`` raises an actionable error pointing back
to the PLUMED path.

Units
-----
MAPLE works in Hartree / Å / fs. Colvars has no Hartree+Å unit system (its
"electron" system is Hartree+Bohr), so we run Colvars in its **"real"** system
(kcal/mol, Å) and convert only energy/force at the seam — positions are already
Å. HA_TO_KCAL converts only the bias energy/force Colvars returns (kcal/mol,
kcal/mol/Å) back to Hartree / Ha/Å. The inner MLIP energy/force stay in Hartree
and are never pushed to Colvars: a position-space bias (harmonic restraint /
metadynamics / eABF) needs only the atomic coordinates. (An ABF scheme that
derives the CV total force from the system forces would additionally require
pushing ``forces`` into Colvars — outside this backend's restraint/metaD/eABF
target.)
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

HA_TO_KCAL = 627.5094740631       # 1 Hartree → kcal/mol  (length stays Å)


def _load_colvars_lines(colvars_input):
    import os
    if isinstance(colvars_input, (list, tuple)):
        return "\n".join(str(x) for x in colvars_input)
    text = str(colvars_input)
    if os.path.exists(text):
        with open(text, encoding="utf-8") as fh:
            return fh.read()
    return text


class ColvarsCalculator(Calculator):
    """Wrap a MAPLE MLIP calculator with a Colvars bias (eABF/ABF focus)."""

    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

    def __init__(self, inner, colvars_input, timestep_fs, temperature,
                 *, atoms=None, output='run', restart_step=0):
        Calculator.__init__(self)
        self.inner = inner
        self._config = _load_colvars_lines(colvars_input)
        self.timestep_fs = float(timestep_fs)
        self.temperature = float(temperature)
        self._istep = int(restart_step)
        self._output = output
        self._cv = None
        self.SUPPORTS_PBC = getattr(inner, 'SUPPORTS_PBC', False)
        if atoms is not None:
            self.atoms = atoms.copy()

    def _init_colvars(self, atoms):
        try:
            import colvars
        except Exception as exc:
            raise RuntimeError(
                "Colvars bias requested but the `colvars` Python bindings are "
                "not importable (%s). Upstream github.com/Colvars/colvars ships "
                "no pip/conda `colvars` package exposing this Colvars() object "
                "API (only a ctypes scripting shim linked into a host NAMD/VMD "
                "executable); a pybind wrapper around colvarproxy_stub must be "
                "built first. Until then use the PLUMED backend (`plumed=...`), "
                "the primary validated enhanced-sampling path in MAPLE." % exc)
        # Generic ("Tcl-less") Colvars proxy. Method names are probed so a minor
        # binding-version drift surfaces as a clear error instead of silent zero.
        proxy = colvars.Colvars() if hasattr(colvars, "Colvars") else colvars
        for setter, arg in (("set_unit_system", "real"),
                            ("units", "real")):
            if hasattr(proxy, setter):
                getattr(proxy, setter)(arg)
                break
        proxy.read_config_string(self._config) if hasattr(
            proxy, "read_config_string") else proxy.read_config(self._config)
        self._cv = proxy

    def calculate(self, atoms=None, properties=('energy', 'forces'),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        atoms = self.atoms

        self.inner.calculate(atoms, list(properties), system_changes)
        energy = float(self.inner.results['energy'])               # Ha
        forces = np.asarray(self.inner.results['forces'], np.float64)  # Ha/Å

        if self._cv is None:
            self._init_colvars(atoms)
        cv = self._cv

        pos = np.ascontiguousarray(atoms.get_positions(), np.float64)  # Å
        # push state (probe common method names across binding versions)
        _call(cv, ("set_step", "setStep"), int(self._istep))
        _call(cv, ("set_positions", "setPositions"), pos)
        if bool(np.any(atoms.pbc)):
            _maybe(cv, ("set_cell", "setBox"),
                   np.ascontiguousarray(atoms.get_cell()[:], np.float64))
        _call(cv, ("calc", "compute", "update"))
        bias_e_kcal = float(_get(cv, ("get_energy", "getEnergy"), 0.0))
        bias_f_kcal = np.asarray(
            _get(cv, ("get_forces", "getForces"),
                 np.zeros_like(pos)), np.float64)                   # kcal/mol/Å
        self._istep += 1

        total_f = forces + bias_f_kcal / HA_TO_KCAL                 # → Ha/Å
        self.results = {
            'energy': energy + bias_e_kcal / HA_TO_KCAL,
            'free_energy': energy + bias_e_kcal / HA_TO_KCAL,
            'forces': total_f,
        }
        if 'stress' in self.inner.results:
            self.results['stress'] = np.asarray(self.inner.results['stress'])


def _call(obj, names, *args):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)(*args)
    raise RuntimeError(f"Colvars binding exposes none of {names}; "
                       "use the PLUMED backend instead.")


def _maybe(obj, names, *args):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)(*args)
    return None


def _get(obj, names, default):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)()
    return default


if __name__ == '__main__':
    try:
        import colvars  # noqa: F401
        print("colvars importable — ColvarsCalculator can drive eABF/ABF.")
    except Exception as exc:
        print(f"SKIP (colvars not importable: {exc}). Wrapper import OK; "
              "use the PLUMED backend for enhanced sampling.")
