"""MACE-OFF23 (organic / biomolecular) routed through the universal
``GenericASECalculator`` adapter so it gains **PBC + stress** for free.

Registered as ``mace-off-generic`` so it coexists with the native ``mace-off``
backend (``mace._mace_off_calculator.MACEOFFCalculator``). The two differ only in
the periodic capability surface, and that difference is the whole point:

* native ``mace-off``       — SUPPORTS_PBC=False, no stress → NVE / NVT only
  (gas-phase molecular target; the meaningful ensembles are non-periodic).
* ``mace-off-generic`` here — SUPPORTS_PBC=True, stress auto-surfaced by the
  adapter (MACE's upstream ``MACECalculator`` computes a configurational
  virial/stress for periodic cells) → unlocks **NPT** for condensed-phase
  organic / biomolecular systems (the enzyme target: explicit-solvent box,
  pressure coupling).

This is the in-repo promotion of the external ``maceoff_pbc_plugin.py`` used in
the R3 capstone (B-42 … B-45): the capstone proved MACE-OFF runs condensed-phase
PBC pure-ML MD (EM→NVT→NPT) end-to-end, but only via an out-of-tree plugin. This
file makes the same capability a shipped backend — the "把 mace 做成全适配"
completion — following the worked B-36 pattern: a construction shim that loads the
upstream model + capability flags, with **no** bespoke ``calculate()`` or units
code (``GenericASECalculator`` delegates to the upstream ASE calculator and routes
results through the shared ``_finalize_results`` chokepoint).

Why MACE-OFF and not MACE-MP for proteins: MACE-OFF23 is trained on organic /
biomolecular chemistry (H, C, N, O, F, P, S, Cl, Br, I) and is the correct
foundation model for enzyme / protein MD, unlike MACE-MP-0 (inorganic crystals /
metals).

Verified by the capstone: the configurational stress sign + units through this
adapter are correct (B-45 — compressing the box raises pressure toward target;
the eV/Å³ ASE Voigt-6 stress is exactly what MAPLE's pressure routine consumes).
"""
from __future__ import annotations

from typing import Literal, Optional

import torch

from ..calculator_base import register_calculator
from ._generic_ase_calculator import GenericASECalculator


def _device_str(device) -> str:
    """Map a torch.device / str to the 'cuda'|'cpu' string mace_off expects."""
    if isinstance(device, torch.device):
        return 'cuda' if device.type == 'cuda' else 'cpu'
    return 'cuda' if str(device).startswith('cuda') else 'cpu'


@register_calculator
class MACEOFFGenericCalculator(GenericASECalculator):
    """MACE-OFF23 organic foundation model wrapped by the universal adapter."""

    MODEL_NAMES = ('mace-off-generic',)
    MODEL_ENERGY_UNIT = 'eV'
    SUPPORTS_PBC = True
    SUPPORTS_CHARGE_MULT = False
    # MACE-OFF23 loads at hardcoded float64 and its factory takes no `precision`
    # kwarg (unlike mace_mp). Declare no mixed-precision support so SetCalculator
    # does not thread `precision` into the constructor. Matches the native
    # ``mace-off`` backend (SUPPORTS_PRECISION=False).
    SUPPORTS_PRECISION = False
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

    def __init__(
        self,
        device,
        model: str = 'mace-off-generic',
        mace_model: str = 'medium',
        model_path: Optional[str] = None,
        overwrite: bool = False,
        implicit: Literal['gbsa', 'none'] = 'none',
        solvent: str = 'none',
    ):
        # (a) construction shim: load the upstream MACE-OFF organic calculator.
        #     Imported lazily so environments that never request it stay light.
        from mace.calculators import MACECalculator, mace_off

        dev = _device_str(device)
        if model_path is not None:
            upstream = MACECalculator(
                model_paths=model_path, device=dev, default_dtype='float64',
            )
        else:
            upstream = mace_off(model=mace_model, device=dev, default_dtype='float64')

        # (b) capability flags: periodic + stress, eV-native. calculate(), unit
        #     conversion and results writing are all inherited from the adapter,
        #     which introspects the upstream `implemented_properties` to surface
        #     stress and gate NPT on.
        super().__init__(
            upstream,
            supports_pbc=True,
            supports_charge_mult=False,
            energy_unit='eV',
            name='mace-off-generic',
        )
        # Preserve the original torch device object (the adapter only needs a
        # device string for the upstream; keep the real one for parity).
        self.device = device
