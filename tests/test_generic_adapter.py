"""CPU unit test for GenericASECalculator using ASE's built-in EMT.

Proves the adapter mechanics with no GPU / network / model download:
  1. energy & forces convert eV -> Hartree exactly via _finalize_results;
  2. stress is left in eV/Angstrom^3 (NOT converted);
  3. stress detection from implemented_properties drives SUPPORTS_PBC inference;
  4. PBC rejection fires for a gas-phase-only wrapper, and a periodic-capable
     wrapper accepts a periodic cell;
  5. the registry wiring for 'mace-mp-0-generic' is present (import-only; no mace).

Run:  PYTHONPATH=<worktree> <plumed-python> test_cpu_emt.py
Exit 0 = pass; raises AssertionError otherwise.
"""
import numpy as np
from ase.build import bulk, molecule
from ase.calculators.emt import EMT

from maple.function.calculator.calculator_base import EV2HARTREE
from maple.function.calculator.generic import GenericASECalculator

TOL = 1e-12


def test_periodic_energy_force_stress_units():
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True).repeat((2, 2, 2))

    # Reference: raw EMT in native eV / eV.Angstrom^-1 / eV.Angstrom^-3.
    ref = atoms.copy()
    ref.calc = EMT()
    e_ev = ref.get_potential_energy()
    f_ev = ref.get_forces()
    s_ev = ref.get_stress(voigt=True)

    calc = GenericASECalculator(EMT())
    # EMT advertises stress -> stress detected -> PBC inferred True.
    assert calc._has_stress is True
    assert calc.SUPPORTS_PBC is True
    assert 'stress' in calc.implemented_properties

    work = atoms.copy()
    work.calc = calc
    e_ha = work.get_potential_energy()
    f_ha = work.get_forces()
    s_out = calc.results['stress']

    # (1) energy & forces converted eV -> Hartree.
    assert abs(e_ha - e_ev * EV2HARTREE) < TOL, (e_ha, e_ev * EV2HARTREE)
    assert np.max(np.abs(f_ha - f_ev * EV2HARTREE)) < TOL
    assert np.all(np.isfinite(f_ha))

    # (2) stress NOT converted: stays eV/Angstrom^3, equals raw EMT stress.
    assert np.max(np.abs(np.asarray(s_out) - s_ev)) < TOL, (s_out, s_ev)
    # And it is genuinely different from a hypothetical converted value.
    assert np.max(np.abs(np.asarray(s_out) - s_ev * EV2HARTREE)) > 1e-9

    # free_energy mirrors energy in Hartree.
    assert abs(calc.results['free_energy'] - e_ev * EV2HARTREE) < TOL
    print('[ok] periodic EMT: E/F -> Hartree, stress kept eV/A^3, finite forces')


def test_pbc_rejection_and_molecule():
    # Force a gas-phase-only wrapper; a periodic cell must be rejected.
    gas = GenericASECalculator(EMT(), supports_pbc=False, name='emt-gasonly')
    assert gas.SUPPORTS_PBC is False
    periodic = bulk('Cu', 'fcc', a=3.6, cubic=True)
    work = periodic.copy()
    work.calc = gas
    raised = False
    try:
        work.get_potential_energy()
    except Exception as exc:  # NotImplementedError-family from reject_periodic_atoms
        raised = True
        print(f'[ok] PBC correctly rejected for gas-only wrapper: {type(exc).__name__}')
    assert raised, 'periodic atoms should be rejected when supports_pbc=False'

    # Non-periodic molecule on the same gas-only wrapper works and converts units.
    mol = molecule('H2O')
    mol.center(vacuum=4.0)
    mol.pbc = False
    ref = mol.copy()
    ref.calc = EMT()
    e_ev = ref.get_potential_energy()
    w2 = mol.copy()
    w2.calc = gas
    e_ha = w2.get_potential_energy()
    assert abs(e_ha - e_ev * EV2HARTREE) < TOL
    # Molecule -> no stress written even though EMT advertises it.
    assert 'stress' not in gas.results
    print('[ok] molecule on gas-only wrapper: E -> Hartree, no stress written')


def test_registry_wiring_present():
    # Importing the builder module registers the name; no mace import needed
    # (mace is lazy-imported inside __init__, which we do NOT call here).
    import maple.function.calculator.generic._mace_mp_generic as gen_mod
    from maple.function.calculator.calculator_base import get_registered_calculator
    from maple.function.calculator.set_calculator import _BUILTIN_NAME_TO_MODULE

    cls = get_registered_calculator('mace-mp-0-generic')
    assert cls is gen_mod.MACEMPGenericCalculator
    assert 'mace-mp-0-generic' in _BUILTIN_NAME_TO_MODULE
    assert _BUILTIN_NAME_TO_MODULE['mace-mp-0-generic'].endswith('generic._mace_mp_generic')
    # Native backend still present and distinct (coexistence, not replacement).
    # The registry is populated lazily on import, so import the native module
    # first (mirrors what _discover_calculator_class does at dispatch time).
    import maple.function.calculator.mace._mace_mp_calculator as native_mod  # noqa: F401
    native = get_registered_calculator('mace-mp-0')
    assert native is not cls
    # Both names resolve to their own distinct modules.
    assert _BUILTIN_NAME_TO_MODULE['mace-mp-0'].endswith('mace._mace_mp_calculator')
    print('[ok] registry: mace-mp-0-generic wired, coexists with native mace-mp-0')


if __name__ == '__main__':
    test_periodic_energy_force_stress_units()
    test_pbc_rejection_and_molecule()
    test_registry_wiring_present()
    print('ALL CPU TESTS PASSED')
