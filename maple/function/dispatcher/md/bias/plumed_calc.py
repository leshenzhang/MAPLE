"""PLUMED bias interface for MAPLE MD (enhanced sampling).

A single calculator wrapper injects a PLUMED bias at the one shared
force-evaluation point (``atoms.get_forces``), so **every** ensemble
(NVE / NVT / NPT) gains metadynamics / OPES / umbrella-sampling (RESTRAINT) /
ABF / funnel / committor for free — no change to any integrator loop.

Unit convention
---------------
MAPLE backends return energy in **Hartree** and forces in **Ha/Å** (stress in
eV/Å³, ASE Voigt). PLUMED is told the host MD units via ``setMDEnergyUnits`` /
``setMDLengthUnits`` / ``setMDTimeUnits``, so every array is handed over in
MAPLE units and PLUMED converts to its internal kJ/mol·nm·ps itself; the bias
forces it adds come back in **Ha/Å** and the bias energy from ``getBias`` is in
**Hartree**. No manual force/energy conversion here.

Driving contract
----------------
PLUMED must be advanced exactly once per MD step. ASE caches
``calculate()`` on unchanged positions, and each ensemble step moves the atoms
once and pulls forces once, so the internal step counter tracks the MD step.

Refs: Tribello et al. Comput. Phys. Commun. 185, 604 (2014) (PLUMED 2);
the host-code patching contract (setMD*Units / setStep / setForces in place)
follows the PLUMED developer manual "Using PLUMED in your code".
"""

import os
import atexit
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

# Host (MAPLE) → PLUMED internal unit conversion factors.
HARTREE_TO_KJ_MOL = 2625.499639479      # 1 Ha → kJ/mol   (PLUMED energy unit)
ANG_TO_NM = 0.1                          # 1 Å  → nm       (PLUMED length unit)
FS_TO_PS = 0.001                         # 1 fs → ps       (PLUMED time unit)
KELVIN_TO_HARTREE = 3.1668114e-6         # k_B in Hartree/K (matches md/utils.py)


def _read_plumed_input(plumed_input):
    """Return cleaned PLUMED action lines from a path or an iterable/str.

    Strips ``#`` comments (full-line and trailing) and blank lines: the
    single-line ``readInputLine`` API does NOT skip comments the way PLUMED's
    own file parser does, so a bare ``# ...`` line raises ``Action "#" is not
    known``. ``#`` is always a comment delimiter in PLUMED, so truncating each
    line at the first ``#`` matches PLUMED file semantics.
    """
    if isinstance(plumed_input, (list, tuple)):
        raw = [str(line) for line in plumed_input]
    else:
        text = str(plumed_input)
        if os.path.exists(text):
            with open(text) as fh:
                raw = [ln.rstrip("\n") for ln in fh]
        else:
            raw = text.splitlines()
    out = []
    for ln in raw:
        s = ln.split("#", 1)[0].strip()
        if s:
            out.append(s)
    return out


class PlumedCalculator(Calculator):
    """Wrap a MAPLE MLIP calculator with a PLUMED bias.

    Parameters
    ----------
    inner : ase Calculator
        The MAPLE MLIP backend (energy Ha, forces Ha/Å, optional stress eV/Å³).
    plumed_input : str | list[str]
        Path to a ``plumed.dat`` file, or the input lines directly.
    timestep_fs : float
        MD timestep in fs (PLUMED needs it for time-dependent biases / PACE).
    temperature : float
        Reference temperature in K (→ kBT for well-tempered metaD / OPES).
    output : str
        Prefix for the PLUMED log file.
    restart_step : int
        Initial PLUMED step counter (set on MD restart so HILLS/bias continue).
    """

    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

    def __init__(self, inner, plumed_input, timestep_fs, temperature,
                 *, atoms=None, output='run', restart_step=0):
        Calculator.__init__(self)
        self.inner = inner
        self._lines = _read_plumed_input(plumed_input)
        self.timestep_fs = float(timestep_fs)
        self.kbt_ha = float(temperature) * KELVIN_TO_HARTREE
        self._istep = int(restart_step)
        self._logpath = f"{output}.plumed.log"
        self._plumed = None
        # Mirror the inner backend's PBC / stress capability so the NPT gate and
        # any periodic CV see the wrapper as equivalent to the raw backend.
        self.SUPPORTS_PBC = getattr(inner, 'SUPPORTS_PBC', False)
        # Advertise `stress` only when the inner backend actually produces it, so
        # the NPT capability gate reads the truth (mirrors GenericASECalculator).
        # Also drives the "always publish the full property set" contract below.
        self._inner_has_stress = 'stress' in tuple(
            getattr(inner, 'implemented_properties', ()) or ())
        props = ['energy', 'free_energy', 'forces']
        if self._inner_has_stress:
            props.append('stress')
        self.implemented_properties = props
        if atoms is not None:
            self.atoms = atoms.copy()
        atexit.register(self.finalize)

    # -- PLUMED lifecycle ------------------------------------------------
    def _init_plumed(self, atoms):
        import plumed
        p = plumed.Plumed()
        p.cmd("setMDEngine", "MAPLE")
        # Unit setters MUST precede init().
        p.cmd("setMDEnergyUnits", HARTREE_TO_KJ_MOL)
        p.cmd("setMDLengthUnits", ANG_TO_NM)
        p.cmd("setMDTimeUnits", FS_TO_PS)
        p.cmd("setMDChargeUnits", 1.0)
        p.cmd("setMDMassUnits", 1.0)
        p.cmd("setNatoms", len(atoms))
        p.cmd("setTimestep", self.timestep_fs)
        p.cmd("setKbT", self.kbt_ha)
        if self._istep > 0:
            p.cmd("setRestart", 1)
        p.cmd("setLogFile", self._logpath)
        p.cmd("init")
        for line in self._lines:
            p.cmd("readInputLine", line)
        self._plumed = p

    def finalize(self):
        """Flush PLUMED (write final HILLS / grids). Idempotent; atexit-safe."""
        p, self._plumed = self._plumed, None
        if p is not None:
            try:
                p.cmd("runFinalJobs")
            except Exception:
                pass

    # -- ASE Calculator protocol -----------------------------------------
    def calculate(self, atoms=None, properties=('energy', 'forces'),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        atoms = self.atoms                      # the copy ASE just stored

        # 1) Pure MLIP energy + forces (+ stress) from the inner backend.
        #    PLUMED needs `forces` on EVERY step (setForces buffer), so always
        #    request energy+forces regardless of what the caller asked — else an
        #    `('energy',)`-only call would miss forces and KeyError here.
        inner_props = ['energy', 'forces']
        if self._inner_has_stress:
            inner_props.append('stress')
        self.inner.calculate(atoms, inner_props, system_changes)
        energy = float(self.inner.results['energy'])                       # Ha
        forces = np.ascontiguousarray(
            self.inner.results['forces'], dtype=np.float64)                # Ha/Å

        if self._plumed is None:
            self._init_plumed(atoms)
        p = self._plumed

        pos = np.ascontiguousarray(atoms.get_positions(), dtype=np.float64)
        masses = np.ascontiguousarray(atoms.get_masses(), dtype=np.float64)
        charges = np.zeros(len(atoms), dtype=np.float64)   # MLIP has no charges
        virial = np.zeros((3, 3), dtype=np.float64)

        p.cmd("setStep", int(self._istep))
        if bool(np.any(atoms.pbc)):
            box = np.ascontiguousarray(atoms.get_cell()[:], dtype=np.float64)
            p.cmd("setBox", box)
        p.cmd("setMasses", masses)
        p.cmd("setCharges", charges)
        p.cmd("setPositions", pos)
        p.cmd("setEnergy", energy)
        p.cmd("setForces", forces)          # PLUMED adds the bias force in place
        p.cmd("setVirial", virial)          # bias virial (unused for NVE/NVT)
        p.cmd("calc")
        bias = np.zeros(1, dtype=np.float64)
        p.cmd("getBias", bias)
        self._istep += 1

        # 2) Publish MLIP + bias. forces was modified in place by PLUMED.
        #    ALWAYS publish the full property set (energy/free_energy/forces, and
        #    stress when the inner backend has it), independent of `properties`.
        #    This keeps `self.results` complete so ASE never re-enters calculate()
        #    for a "missing" property at unchanged positions — which would advance
        #    the PLUMED step counter a SECOND time within one MD step.
        self.results = {
            'energy': energy + float(bias[0]),          # Ha
            'free_energy': energy + float(bias[0]),
            'forces': forces,                           # Ha/Å (MLIP + bias)
        }
        if self._inner_has_stress and 'stress' in self.inner.results:
            # ponytail: bias contribution to the stress is NOT folded in, so an
            # NPT barostat run under a bias couples to the bare MLIP stress only.
            # Fine for the usual NVT umbrella/metaD use; upgrade by reading the
            # in-place `virial` back into a Voigt stress when NPT+bias is needed.
            self.results['stress'] = np.asarray(self.inner.results['stress'])


if __name__ == '__main__':
    # Runnable check: a 2-atom system with a DISTANCE CV under a RESTRAINT must
    # feel a bias force that pushes the distance toward AT. Skips cleanly if the
    # PLUMED kernel is not installed.
    try:
        import plumed  # noqa: F401
    except Exception as exc:
        print(f"SKIP (plumed not importable: {exc}). Wrapper import OK.")
        raise SystemExit(0)

    from ase import Atoms

    class _ConstCalc(Calculator):
        implemented_properties = ['energy', 'forces']
        SUPPORTS_PBC = False
        def calculate(self, atoms=None, properties=('energy', 'forces'),
                      system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            self.results = {'energy': 0.0,
                            'forces': np.zeros((len(self.atoms), 3))}

    d0 = 2.0
    at = Atoms('H2', positions=[[0, 0, 0], [d0, 0, 0]])
    # RESTRAINT at 1.0 Å (in nm for the .dat: PLUMED length unit is nm, but our
    # setMDLengthUnits maps Å→nm, so AT is given in PLUMED nm = 0.1 Å). Use the
    # ATOMS distance CV; AT=0.10 nm = 1.0 Å.
    lines = ["d: DISTANCE ATOMS=1,2",
             "RESTRAINT ARG=d AT=0.10 KAPPA=2000.0"]
    at.calc = PlumedCalculator(_ConstCalc(), lines, timestep_fs=0.5,
                               temperature=300.0, output='selftest', atoms=at)
    f = at.get_forces()
    # Bias = 0.5 k (d-AT)^2 with d=2 Å > 1 Å ⇒ restoring force pulls atoms
    # together: atom 1 (at x=0) pushed +x, atom 2 (at x=2) pushed −x.
    assert f[0, 0] > 0 and f[1, 0] < 0, f
    assert abs(f[0, 0] + f[1, 0]) < 1e-8, "bias force must be internal (sum≈0)"
    print(f"OK plumed bias self-test: f[0,x]={f[0,0]:.4f} f[1,x]={f[1,0]:.4f} Ha/Å")
