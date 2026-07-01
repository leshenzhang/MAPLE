"""
In-tree correctness gate for the NVE (microcanonical) ensemble.

Validation axis = ALGORITHM CORRECTNESS of the symplectic velocity-Verlet
integrator that NVE (ensemble/nve.py) wraps.  Two independent properties of a
correct microcanonical integrator are asserted (fixed seed, small system,
MACE-OFF23 fp64 on a100):

  ENERGY-CONSERVATION : run the real NVE class (thermostat off, no runtime COM
        projection => pure Hamiltonian flow) and read the per-step total energy
        H = KE + PE from the logger.  A symplectic integrator has NO secular
        drift; H oscillates around a constant.  Gate:
          * secular drift  |slope(H vs t)|  small  (<~1e-2 meV/atom/ps; the
            physical aspiration for MACE at 0.5 fs is ~1e-3 meV/atom/ps),
          * bounded fluctuation std(H)/atom small.
        (H is the microcanonical conserved quantity; there is no thermostat
        H-tilde bookkeeping in NVE.)

  TIME-REVERSIBILITY : drive the VelocityVerlet integrator NVE uses directly;
        integrate N steps, flip every velocity, integrate N steps.  A
        time-reversible integrator retraces its path exactly (to fp64 roundoff),
        recovering the initial positions and the negated initial velocities.
        Gate: max|dr| back to start << the forward displacement D.

Run as a script on a GPU node (never on import).
"""
import os, sys, tempfile
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator.mace._maceoff_batch_calculator import MaceOffBatchCalc
from maple.function.dispatcher.md.ensemble.nve import NVE
from maple.function.dispatcher.md.integrator.velocity_verlet import VelocityVerlet
from maple.function.dispatcher.md.utils import HA_PER_ANG_TO_AU, HARTREE_TO_EV

if __name__ != "__main__":
    raise SystemExit("run _test_nve.py as a script, not an import")

MODEL = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEV_PER_HA = HARTREE_TO_EV * 1000.0     # Hartree -> meV


def ethanol():
    return molecule("CH3CH2OH")


# Single-system ASE bridge: Hartree-returning MACE-OFF via MaceOffBatchCalc(B=1)
# (identical engine to the batched kernels; drives the real NVE class + the raw
#  velocity-Verlet integrator, so the gate isolates the integrator math).
class _MaceOffSingle(Calculator):
    implemented_properties = ["energy", "forces", "free_energy"]

    def __init__(self, template_atoms, model_path=MODEL, device=DEV):
        super().__init__()
        self._bc = MaceOffBatchCalc(model_path=model_path, device=device, dtype=torch.float64)
        self._bc.prepare([template_atoms.copy()], fixed_nmax=None)
        self.atomic_numbers = self._bc.atomic_numbers
        self.device = self._bc.device

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(self.atoms)
        coord = torch.tensor(self.atoms.get_positions(), dtype=torch.float64, device=self._bc.device)
        self._bc.set_coords_(coord)
        E_Ha, F_Ha = self._bc.get_ef_gpu()
        E = float(E_Ha[0].item())
        F = F_Ha[0, :3 * n].detach().to("cpu").numpy().reshape(n, 3)
        self.results = {"energy": E, "free_energy": E, "forces": F}


def gate_energy_conservation(steps=4000, dt=0.5, seed=1):
    at = ethanol()
    at.calc = _MaceOffSingle(at, device=DEV)
    n = len(at)
    paras = dict(timestep=dt, steps=steps, temperature=300.0,
                 init_velocities=True, random_seed=seed,
                 remove_com=True, remove_angular=True,
                 remove_com_every=0, remove_angular_every=0,   # strict NVE: pure Hamiltonian
                 log_every=1, traj_every=steps, verbose=0)
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        out = f.name
    nve = NVE(out, at, paras=paras)
    nve.run()

    E = np.asarray(nve.logger.energies, dtype=np.float64)        # Ha, per step
    t = np.asarray(nve.logger.times, dtype=np.float64)           # fs
    assert E.size >= steps // 2 and np.all(np.isfinite(E)), "NVE energy series bad"

    burn = int(0.05 * E.size)                                    # drop initial transient
    Eb, tb = E[burn:], t[burn:]
    # secular drift = least-squares slope of H(t)
    A = np.vstack([tb - tb.mean(), np.ones_like(tb)]).T
    slope_ha_per_fs = float(np.linalg.lstsq(A, Eb - Eb.mean(), rcond=None)[0][0])
    drift_mev_atom_ps = slope_ha_per_fs * MEV_PER_HA / n * 1000.0     # meV/atom/ps
    std_mev_atom = float(np.std(Eb)) * MEV_PER_HA / n
    span_ps = (tb[-1] - tb[0]) / 1000.0

    print(f"[NVE-ECONS] steps={steps} dt={dt}fs span={span_ps:.3f}ps n_atoms={n}  "
          f"secular_drift={drift_mev_atom_ps:.3e} meV/atom/ps  "
          f"std(H)={std_mev_atom:.3e} meV/atom  (aspiration drift ~1e-3)")
    ok = abs(drift_mev_atom_ps) < 1e-2 and std_mev_atom < 5.0
    print(f"[NVE-ECONS] {'PASS' if ok else 'FAIL'}")
    return ok, drift_mev_atom_ps, std_mev_atom


def gate_time_reversibility(N=50, dt=0.5, seed=1):
    at = ethanol()
    at.calc = _MaceOffSingle(at, device=DEV)
    n = len(at)
    # physically scaled au velocities (COM + rotation removed) from the real NVE init
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        out = f.name
    nve = NVE(out, at, paras=dict(timestep=dt, steps=1, temperature=300.0,
                                  init_velocities=True, random_seed=seed,
                                  remove_com=True, remove_angular=True, verbose=0))
    v0 = nve._initialize_velocities().copy()

    integ = VelocityVerlet(at, dt)
    pos0 = at.get_positions().copy()
    forces = at.get_forces() * HA_PER_ANG_TO_AU
    v = v0.copy()
    for _ in range(N):
        v, forces = integ.step(v, forces)
    D = float(np.max(np.abs(at.get_positions() - pos0)))         # forward displacement

    v = -v                                                       # reverse time
    forces = at.get_forces() * HA_PER_ANG_TO_AU
    for _ in range(N):
        v, forces = integ.step(v, forces)
    dpos = float(np.max(np.abs(at.get_positions() - pos0)))
    dvel = float(np.max(np.abs((-v) - v0)))

    print(f"[NVE-REVERS] N={N} dt={dt}fs  forward_disp D={D:.3e} A  "
          f"return max|dr|={dpos:.3e} A  max|dv|={dvel:.3e} au  ratio dr/D={dpos/max(D,1e-30):.2e}")
    ok = dpos < 1e-3 and dpos < 1e-2 * max(D, 1e-30) and dvel < 1e-3
    print(f"[NVE-REVERS] {'PASS' if ok else 'FAIL'}")
    return ok, dpos, dvel, D


if __name__ == "__main__":
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    ok1, drift, std = gate_energy_conservation()
    ok2, dpos, dvel, D = gate_time_reversibility()
    allok = ok1 and ok2
    print(f"\n[NVE RESULT] econs_drift={drift:.3e}meV/atom/ps std={std:.3e}meV/atom  "
          f"revers_dr={dpos:.3e}A dv={dvel:.3e}au (D={D:.3e}A)")
    print(f"[NVE RESULT] {'ALL NVE GATES PASS' if allok else 'SOME NVE GATES FAILED'}")
    sys.exit(0 if allok else 1)
