"""Universal ASE-calculator adapter (``GenericASECalculator``).

Wraps an **already-instantiated** ``ase.calculators.calculator.Calculator`` so it
plugs into MAPLE's MD/optimization engine with no bespoke per-potential code. The
engine (integrator / thermostat / barostat / constraints / posres / bias) is
already potential-agnostic: it calls only the ASE protocol and lets
``CalcABC._finalize_results`` perform the single eV->Hartree units conversion.
This adapter is the explicit realization of that fact.

Why per-potential adaptation is mostly redundant (B-36)
-------------------------------------------------------
The only irreducibly per-potential pieces live in the *builder*, not here:

* **(a) construction shim** — how the upstream library loads its model
  (``mace_mp(...)`` vs ``load_predict_unit(...)`` vs ``torch.jit.load(...)``);
  a handful of lines.
* **(b) capability flags** — ``supports_pbc`` / ``supports_charge_mult`` /
  ``energy_unit``: physics of the model, declared once at wrap time.

Everything else — ``calculate()``, unit conversion, results writing, PBC
rejection, the Hessian/HVP scaffolding — is shared and lives in ``CalcABC``.
So a new MLIP = ``GenericASECalculator(build_upstream(), supports_pbc=...)`` +
one ``_BUILTIN_NAME_TO_MODULE`` line. Zero engine edits.

Units
-----
The wrapped calculator reports energy/forces in ``energy_unit`` ('eV' by default,
or 'hartree'). ``_finalize_results`` converts energy/forces to Hartree /
Hartree.Angstrom^-1. **Stress is left in eV/Angstrom^3** (ASE Voigt convention) —
that is exactly what ``dispatcher/md/utils.compute_instantaneous_pressure``
consumes; it is *not* converted to Hartree/Bohr^3.
"""
from __future__ import annotations

import numpy as np
from ase.calculators.calculator import all_changes

from ..calculator_base import CalcABC


def _detect_stress(ase_calc) -> bool:
    """True when the wrapped calculator advertises a configurational stress."""
    props = getattr(ase_calc, 'implemented_properties', None) or ()
    return 'stress' in tuple(props)


class GenericASECalculator(CalcABC):
    """Adapt an arbitrary instantiated ASE Calculator to the ``CalcABC`` contract.

    Parameters
    ----------
    ase_calc : ase.calculators.calculator.Calculator
        An already-constructed upstream calculator (EMT, MACE-MP, fairchem UMA,
        AIMNet2, ...). Its ``implemented_properties`` is introspected to decide
        whether stress is available.
    supports_pbc : bool | None
        Whether the wrapped model produces correct periodic forces/stress.
        ``None`` -> inferred from stress support (a stress-returning calculator
        is treated as periodic-capable). Set explicitly to allow PBC on a
        periodic model that does not export stress, or to reject PBC on a
        gas-phase-only wrapper.
    supports_charge_mult : bool
        Whether the model consumes total charge / spin multiplicity
        (``atoms.info['charge']`` / ``['spin']``).
    energy_unit : {'eV', 'hartree'}
        Unit the wrapped calculator reports energy/forces in.
    name : str | None
        Optional label used in error messages and ``repr``.

    Notes
    -----
    ``MODEL_NAMES`` is intentionally empty: instances are produced by thin
    builder modules that subclass this with a concrete name and a construction
    shim (see ``_mace_mp_generic``). This base class is never registered or
    dispatched directly.
    """

    MODEL_NAMES: tuple = ()
    MODEL_ENERGY_UNIT: str = 'eV'
    SUPPORTED_HESSIAN_MODES: tuple = ('numerical',)
    SUPPORTS_CHARGE_MULT: bool = False
    SUPPORTS_PBC: bool = False

    def __init__(
        self,
        ase_calc,
        *,
        supports_pbc=None,
        supports_charge_mult=False,
        energy_unit='eV',
        name=None,
    ):
        super().__init__()
        if energy_unit not in ('eV', 'hartree'):
            raise ValueError(
                f"energy_unit must be 'eV' or 'hartree', got {energy_unit!r}"
            )
        if ase_calc is None:
            raise ValueError('GenericASECalculator requires an instantiated ASE calculator')

        self._ase_calc = ase_calc
        self._has_stress = _detect_stress(ase_calc)

        # Instance-level capability flags shadow the class defaults, because each
        # wrapped calculator differs. ``_reject_unsupported_pbc`` reads
        # ``self.SUPPORTS_PBC``; the charge path reads ``self.SUPPORTS_CHARGE_MULT``.
        # Default PBC support to "has stress" — a stress-returning model is
        # periodic-capable — unless the caller states otherwise.
        self.SUPPORTS_PBC = bool(self._has_stress) if supports_pbc is None else bool(supports_pbc)
        self.SUPPORTS_CHARGE_MULT = bool(supports_charge_mult)
        self.MODEL_ENERGY_UNIT = energy_unit

        self._name = name or f'GenericASECalculator({type(ase_calc).__name__})'
        # Kept for ctor-parity with CalcABC's implicit-solvent machinery, which
        # this adapter never enables (solvent_correction stays None).
        self.device = getattr(ase_calc, 'device', 'cpu')
        self.solvent_correction = None
        self.hessian = 'numerical'

        # Advertise stress only when the wrapped calc can produce it, so the
        # ensemble capability gate (NPT requires stress) reads the truth.
        props = ['energy', 'forces', 'free_energy']
        if self._has_stress:
            props.append('stress')
        self.implemented_properties = props

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """Delegate to the wrapped ASE calculator, then route through the units seam.

        A copy of the target atoms carries the wrapped calculator so its own
        atoms/cache bookkeeping never aliases the engine's live ``Atoms``. ASE
        caches per-evaluation, so a ``get_forces()`` followed by ``get_stress()``
        on unchanged positions costs a single upstream forward pass.
        """
        properties = self._normalize_properties(properties)
        # Base-class guards: PBC rejection for gas-phase wrappers + implicit-solvent.
        atoms = super().calculate(atoms, properties, system_changes)

        work = atoms.copy()
        work.calc = self._ase_calc
        energy = float(work.get_potential_energy())
        forces = np.asarray(work.get_forces(), dtype=np.float64)

        stress = None
        if self._has_stress and bool(np.any(work.pbc)):
            # eV/Angstrom^3, ASE Voigt-6; left unconverted by _finalize_results.
            stress = np.asarray(work.get_stress(voigt=True), dtype=np.float64)

        self._finalize_results(
            atoms,
            energy=energy,
            forces=forces,
            stress=stress,
            unit=self.MODEL_ENERGY_UNIT,
        )

    def __repr__(self):
        return (
            f'<{self._name} pbc={self.SUPPORTS_PBC} stress={self._has_stress} '
            f'charge_mult={self.SUPPORTS_CHARGE_MULT} unit={self.MODEL_ENERGY_UNIT}>'
        )
