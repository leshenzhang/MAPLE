"""
In-tree correctness gate for the CANONICAL (NVT) kinetic-energy distribution.

Validation axis = ALGORITHM CORRECTNESS of the v-rescale (Bussi 2007) thermostat
in the single-system NVT class (ensemble/nvt.py).  The existing batched-NVT
gates check only PARITY + the *mean* temperature <T>.  The mean is necessary but
NOT sufficient: a wrong thermostat (e.g. Berendsen / deterministic isokinetic)
reproduces <T> yet collapses the KINETIC-ENERGY FLUCTUATION ("flying ice cube").
This gate tests the full instantaneous distribution.

For a canonical ensemble with N_f degrees of freedom the instantaneous kinetic
energy K obeys  2K/(kT) ~ chi^2(N_f), equivalently the Maxwell-Boltzmann KE law
  <K> = (N_f/2) kT      and      Var(K) = (N_f/2) (kT)^2.
N_f must be the number of ACTIVE momentum DOF: remove_com_every=1 projects the
COM every step so the 3 COM modes carry no KE and the thermostat target N_f
(read from thermostat._n_dof = 3N-3 = 24) matches the active subspace exactly --
otherwise (remove_com_every=0 -> N_f=3N=27 while only 24 DOF are live) the
thermostat over-heats the internal modes and the KE marginal is a biased chi^2.
The variance is the discriminating moment.  We assert:
  * <K>/target within +/-3 %,
  * Var(K)/target within +/-20 % (finite-sample tolerance),
  * KS statistic of the decorrelated 2K/kT sample vs chi^2(N_f) small.

Deterministic (fixed seed), small isolated molecule (ethanol, N_f = 3N-3 = 24),
MACE-OFF23 fp64 on a100.  Run as a script (never on import).
"""
import os, sys, tempfile
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes
from scipy import stats as _st

from maple.function.calculator.mace._maceoff_batch_calculator import MaceOffBatchCalc
from maple.function.dispatcher.md.ensemble.nvt import NVT
from maple.function.dispatcher.md.utils import KELVIN_TO_HARTREE

if __name__ != "__main__":
    raise SystemExit("run _test_nvt_canonical.py as a script, not an import")

MODEL = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


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


def gate_canonical_ke(steps=20000, dt=0.5, T=300.0, tau_t=50.0, seed=7):
    at = molecule("CH3CH2OH")
    at.calc = _MaceOffSingle(at, device=DEV)
    paras = dict(timestep=dt, steps=steps, temperature=T, thermostat="v-rescale",
                 tau_t=tau_t, init_velocities=True, random_seed=seed,
                 remove_com=True, remove_com_every=1, remove_angular=False,
                 log_every=1, traj_every=steps, verbose=0)
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        out = f.name
    nvt = NVT(out, at, paras=paras)
    n_dof = int(nvt.thermostat._n_dof)          # DOF the thermostat targets (= 3N-3)
    nvt.run()

    K = np.asarray(nvt.logger.kinetic_energies, dtype=np.float64)   # Ha, per step
    assert K.size >= steps // 2 and np.all(np.isfinite(K)), "KE series bad"
    K = K[int(0.1 * K.size):]                    # drop equilibration

    kT = T * KELVIN_TO_HARTREE
    mean_target = 0.5 * n_dof * kT
    var_target = 0.5 * n_dof * kT * kT
    mean_ratio = float(np.mean(K)) / mean_target
    var_ratio = float(np.var(K)) / var_target
    T_report = 2.0 * float(np.mean(K)) / (n_dof * KELVIN_TO_HARTREE)

    # decorrelate before the KS test (v-rescale KE is time-correlated). Stride by
    # a few tau to approach i.i.d.; keeps ~1000 near-independent samples.
    x = 2.0 * K / kT                              # ~ chi^2(n_dof)
    stride = max(1, int(x.size / 1000))
    xs = x[::stride]
    ks_D, ks_p = _st.kstest(xs, "chi2", args=(n_dof,))

    print(f"[NVT-CANON] steps={steps} T={T}K tau_t={tau_t}fs N_dof={n_dof}  "
          f"<T>report={T_report:.2f}K")
    print(f"[NVT-CANON] <K>/target={mean_ratio:.4f}  Var(K)/target={var_ratio:.4f}  "
          f"(both -> 1 for canonical)")
    print(f"[NVT-CANON] KS(2K/kT vs chi2({n_dof})): D={ks_D:.4f} p={ks_p:.3e} "
          f"(n_eff={xs.size}, stride={stride})")
    ok = (0.97 < mean_ratio < 1.03) and (0.80 < var_ratio < 1.20) and (ks_D < 0.08)
    print(f"[NVT-CANON] {'PASS' if ok else 'FAIL'}")
    return ok, mean_ratio, var_ratio, ks_D


if __name__ == "__main__":
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    ok, mr, vr, D = gate_canonical_ke()
    print(f"\n[NVT-CANON RESULT] mean_ratio={mr:.4f} var_ratio={vr:.4f} KS_D={D:.4f}")
    print(f"[NVT-CANON RESULT] {'CANONICAL KE GATE PASS' if ok else 'CANONICAL KE GATE FAILED'}")
    sys.exit(0 if ok else 1)
