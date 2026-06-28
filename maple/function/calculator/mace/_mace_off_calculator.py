"""Gas-phase MACE-OFF (organic / biomolecular) backend.

Parallel to the periodic MACE-MP-0 backend (``_mace_mp_calculator``): both wrap
the upstream ``mace.calculators`` foundation-model factory and convert the model
output eV / (eV·Å⁻¹) to Hartree / (Hartree·Å⁻¹) through the *same* shared seam,
``CalcABC._finalize_results(..., unit='eV')``. No unit logic is duplicated here.

MACE-OFF23 is trained on organic / biomolecular chemistry (H, C, N, O, F, P, S,
Cl, Br, I) and is the correct foundation model for enzyme / protein MD — unlike
MACE-MP-0, which only covers inorganic crystals and metals and is wrong for
proteins.

This is a *gas-phase* wrapper: ``SUPPORTS_PBC = False`` and no stress is reported
(MACE-OFF23 targets non-periodic molecular systems; the meaningful ensembles are
NVE / NVT). For periodic condensed-phase / NPT runs use the MACE-MP-0 backend.

Units
-----
* energy / forces -> Hartree / (Hartree·Å⁻¹) via ``_finalize_results(unit='eV')``
  (the MAPLE MD integrator then multiplies forces by ``HA_PER_ANG_TO_AU``).
* stress          -> not reported (no PBC; non-periodic organic systems).
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from ase.calculators.calculator import all_changes

from ..calculator_base import CalcABC, register_calculator
from ._mace_mp_calculator import MACEMPCalculator, _device_str


@register_calculator
class MACEOFFCalculator(MACEMPCalculator):
    """ASE calculator wrapping the upstream MACE-OFF23 organic foundation model.

    Inherits the option/registry plumbing of ``MACEMPCalculator``
    (``build_kwargs_from_options``, ``OPTION_KEYS``, ``MODEL_PATH_OPTION``,
    Hartree unit declaration) and overrides only what differs for the gas-phase
    organic model: the foundation-model factory (``mace_off`` instead of
    ``mace_mp``), the registry name, no PBC, and no stress.
    """

    implemented_properties = ['energy', 'forces', 'free_energy']

    MODEL_NAMES = ('mace-off',)
    SUPPORTS_PBC = False

    def __init__(self,
        device,
        model: str = 'mace-off',
        mace_model: str = 'medium',
        model_path: Optional[str] = None,
        overwrite: bool = False,
        implicit: Literal['gbsa', 'none'] = 'none',
        solvent: str = 'none',
        ):
        """
        Args:
            device: compute device (torch.device or 'cuda'/'cpu').
            model: MAPLE registry key; not used for weight loading.
            mace_model: MACE-OFF23 size 'small'|'medium'|'large' (default
                'medium'; ignored when ``model_path`` is given).
            model_path: explicit path to a local MACE model file; overrides the
                foundation-model download/cache lookup.
            overwrite: unused (ctor-signature parity with sibling MACE backends).
            implicit / solvent: implicit-solvent config (energy-only; unused here).
        """
        # Skip MACEMPCalculator.__init__ (it builds the *inorganic* mace_mp
        # model); initialise CalcABC directly and load MACE-OFF instead.
        CalcABC.__init__(self)
        self.device = device
        dev = _device_str(device)

        # Import upstream MACE lazily so MAPLE only needs it when this backend is
        # actually requested.
        from mace.calculators import MACECalculator, mace_off

        if model_path is not None:
            self._mace = MACECalculator(
                model_paths=model_path, device=dev, default_dtype='float64',
            )
        else:
            self._mace = mace_off(
                model=mace_model, device=dev, default_dtype='float64',
            )

        self.overwrite = overwrite
        self.hessian = 'numerical'
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """Evaluate energy + forces in one upstream forward pass.

        No stress is requested: the gas-phase MACE-OFF target is non-periodic, so
        a configurational stress tensor is meaningless (and ``SUPPORTS_PBC`` is
        False, gating out NPT)."""
        properties = self._normalize_properties(properties)
        # Bypass MACEMPCalculator.calculate (it requests stress); go straight to
        # the CalcABC entry point (pbc / implicit-solvent gates + ASE bookkeeping).
        atoms = CalcABC.calculate(self, atoms, properties, system_changes)

        self._mace.calculate(
            atoms,
            properties=['energy', 'forces'],
            system_changes=system_changes,
        )
        e_ev = float(self._mace.results['energy'])
        f_ev = np.asarray(self._mace.results['forces'], dtype=np.float64)

        # eV / (eV·Å⁻¹) -> Hartree / (Hartree·Å⁻¹) via the shared CalcABC seam.
        self._finalize_results(atoms, energy=e_ev, forces=f_ev, unit='eV')
