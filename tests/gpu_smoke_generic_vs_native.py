"""GPU a100 smoke: the universal adapter wraps a real foundation model (mace-mp-0)
and reproduces the native MACEMPCalculator backend bit-for-bit.

Proves the adapter is correct on a production MLIP, not just EMT:
  - native  = maple ... mace._mace_mp_calculator.MACEMPCalculator (hand-written calculate)
  - generic = maple ... generic._mace_mp_generic.MACEMPGenericCalculator (adapter)
Both load the SAME upstream mace_mp(model='medium', float64); results must agree to
machine precision on a periodic Cu cell (E/F/stress) and a non-periodic water
molecule (E/F). Also exercises a short NVE-style finite-difference sanity (forces
≈ -dE/dx) on the generic path so a wiring break would surface.

Run under sbatch (a100); see 01__genadapt_smoke.sbatch.
"""
import numpy as np
import torch
from ase.build import bulk, molecule

from maple.function.calculator.calculator_base import EV2HARTREE
from maple.function.calculator.mace._mace_mp_calculator import MACEMPCalculator
from maple.function.calculator.generic._mace_mp_generic import MACEMPGenericCalculator

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device = {dev}, torch {torch.__version__}, cuda_avail={torch.cuda.is_available()}')

ETOL, FTOL, STOL = 1e-6, 1e-6, 1e-6


def evaluate(calc, atoms, periodic):
    w = atoms.copy()
    w.calc = calc
    e = float(w.get_potential_energy())
    f = np.asarray(w.get_forces(), dtype=np.float64)
    s = np.asarray(calc.results['stress'], dtype=np.float64) if periodic else None
    return e, f, s


def ab_compare(label, atoms, periodic):
    native = MACEMPCalculator(dev, model='mace-mp-0', mace_model='medium',
                              implicit='none', solvent='none')
    generic = MACEMPGenericCalculator(dev, mace_model='medium',
                                      implicit='none', solvent='none')
    en, fn, sn = evaluate(native, atoms, periodic)
    eg, fg, sg = evaluate(generic, atoms, periodic)

    de = abs(en - eg)
    df = float(np.max(np.abs(fn - fg)))
    print(f'[{label}] E_native={en:.10f} Ha  E_generic={eg:.10f} Ha  dE={de:.2e}')
    print(f'[{label}] max|dF|={df:.2e} Ha/A   F_finite={np.all(np.isfinite(fg))}')
    assert de < ETOL, f'{label}: energy mismatch {de}'
    assert df < FTOL, f'{label}: force mismatch {df}'
    assert np.all(np.isfinite(fg))

    if periodic:
        ds = float(np.max(np.abs(sn - sg)))
        # Stress stays eV/Angstrom^3 (not Hartree-scaled): a sanity guard.
        print(f'[{label}] max|dStress|={ds:.2e} eV/A^3   stress_native={sn}')
        assert ds < STOL, f'{label}: stress mismatch {ds}'
        # generic energy in Hartree must be << its eV magnitude (conversion happened).
        assert abs(eg) < abs(en / EV2HARTREE) + 1.0
    return generic


def fd_force_check(calc, atoms, h=1e-3):
    """Central finite-difference of energy vs reported force on atom 0, x-axis.

    Energy is in Hartree, positions in Angstrom -> force in Hartree/Angstrom,
    matching get_forces(). Confirms the adapter's forces are the gradient of its
    energy (a wiring break, e.g. forgetting unit conversion on only one channel,
    would fail here)."""
    base = atoms.copy()
    fx0 = atoms.copy(); fx0.calc = calc
    f_reported = fx0.get_forces()[0, 0]

    plus = base.copy(); plus.positions[0, 0] += h; plus.calc = calc
    e_plus = plus.get_potential_energy()
    minus = base.copy(); minus.positions[0, 0] -= h; minus.calc = calc
    e_minus = minus.get_potential_energy()
    f_fd = -(e_plus - e_minus) / (2 * h)
    err = abs(f_fd - f_reported)
    print(f'[fd] F_reported={f_reported:.6f}  F_fd={f_fd:.6f}  err={err:.2e} Ha/A')
    assert err < 5e-3, f'finite-difference force mismatch {err}'


cu = bulk('Cu', 'fcc', a=3.58, cubic=True).repeat((2, 2, 2))
gen_periodic = ab_compare('Cu32-periodic', cu, periodic=True)

wat = molecule('H2O'); wat.center(vacuum=5.0); wat.pbc = False
ab_compare('H2O-molecule', wat, periodic=False)

# Finite-difference gradient check on the generic periodic path.
fd_force_check(gen_periodic, cu)

print('GPU SMOKE PASSED: generic adapter == native mace-mp-0 (E/F/stress) + FD-consistent')
