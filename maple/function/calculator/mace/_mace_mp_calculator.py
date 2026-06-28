"""Periodic MACE-MP-0 backend (PBC + stress) for NPT / condensed-phase MD.

Wraps the upstream ``mace.calculators`` foundation-model calculator (MACE-MP-0,
MPtrj-trained), which natively supports periodic boundary conditions AND returns
a configurational stress tensor.

This is the first MAPLE backend that reports a genuine ``'stress'`` property, so
it is the calculator the NPT barostat needs: the other MACE backends
(``_mace_calculator`` / ``_macepol_calculator``) set ``SUPPORTS_PBC=False`` and
stub stress to zero, and aimnet2 is gas-phase only.

Units
-----
* energy / forces  -> Hartree / (Hartree·Å⁻¹)  via ``_finalize_results``
  (the MAPLE MD integrator multiplies forces by ``HA_PER_ANG_TO_AU``).
* stress           -> left in **eV/Å³** (ASE Voigt convention), because that is
  exactly what ``dispatcher/md/utils.compute_instantaneous_pressure`` consumes
  (its kinetic term is computed in eV).  It is *not* converted to Hartree/Bohr³.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from ..calculator_base import CalcABC, register_calculator


def _device_str(device) -> str:
    """Map a torch.device / str to the 'cuda'|'cpu' string mace_mp expects."""
    if isinstance(device, torch.device):
        return 'cuda' if device.type == 'cuda' else 'cpu'
    return 'cuda' if str(device).startswith('cuda') else 'cpu'


@register_calculator
class MACEMPCalculator(CalcABC):
    """ASE calculator wrapping the upstream periodic MACE-MP-0 foundation model."""

    implemented_properties = ['energy', 'forces', 'free_energy', 'stress']

    MODEL_NAMES = ('mace-mp-0',)
    MODEL_ENERGY_UNIT = 'eV'
    SUPPORTED_HESSIAN_MODES = ('numerical',)
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = True
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = ('mace_model', 'model_path')
    MODEL_PATH_OPTION = 'model_path'

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        kwargs = {}
        size = (options or {}).get('mace_model')
        if size:
            kwargs['mace_model'] = size
        if resolved_model_path is not None:
            kwargs['model_path'] = resolved_model_path
        return kwargs

    def __init__(self,
        device,
        model: str = 'mace-mp-0',
        mace_model: str = 'medium',
        model_path: Optional[str] = None,
        overwrite: bool = False,
        implicit: Literal['gbsa', 'none'] = 'none',
        solvent: str = 'none',
        ):
        """
        Args:
            device: compute device (torch.device or 'cuda'/'cpu').
            model: MAPLE model name (registry key); not used for weight loading.
            mace_model: foundation-model size 'small'|'medium'|'large'
                (ignored when ``model_path`` is given).
            model_path: explicit path to a local MACE model file; overrides the
                foundation-model download/cache lookup.
            overwrite: unused (kept for ctor-signature parity with other backends).
            implicit / solvent: implicit-solvent config (energy-only; unused here).
        """
        super().__init__()
        self.device = device
        dev = _device_str(device)

        # Import upstream MACE lazily so MAPLE only needs it when this backend is
        # actually requested (keeps gas-phase-only environments importable).
        from mace.calculators import MACECalculator, mace_mp

        if model_path is not None:
            self._mace = MACECalculator(
                model_paths=model_path, device=dev, default_dtype='float64',
            )
        else:
            self._mace = mace_mp(
                model=mace_model, device=dev, default_dtype='float64',
            )

        self.overwrite = overwrite
        self.hessian = 'numerical'
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """Main ASE entry point. Always evaluates energy+forces+stress in one
        upstream forward pass; ASE caches the result so a get_forces() followed
        by get_stress() on unchanged positions costs a single evaluation."""
        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)

        # Delegate to the upstream periodic MACE calculator on the SAME atoms
        # (it builds its own minimum-image neighbor list from atoms.cell/pbc).
        self._mace.calculate(
            atoms,
            properties=['energy', 'forces', 'stress'],
            system_changes=system_changes,
        )
        e_ev = float(self._mace.results['energy'])
        f_ev = np.asarray(self._mace.results['forces'], dtype=np.float64)
        s_ev = np.asarray(self._mace.results['stress'], dtype=np.float64)  # eV/Å³, Voigt

        # energy/forces -> Hartree; stress stays eV/Å³ (pressure-consumer convention).
        self._finalize_results(atoms, energy=e_ev, forces=f_ev, unit='eV', stress=s_ev)
