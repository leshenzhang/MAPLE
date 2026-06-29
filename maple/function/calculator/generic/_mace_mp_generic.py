"""Proof: mace-mp-0 routed through the universal ``GenericASECalculator`` adapter.

Registered as ``mace-mp-0-generic`` so it coexists with the native ``mace-mp-0``
backend (``mace._mace_mp_calculator.MACEMPCalculator``) for A/B verification.

This is the worked example behind B-36: a periodic foundation model needs only
  (a) a construction shim (load the upstream model — the ``mace_mp(...)`` call), and
  (b) capability flags (``supports_pbc=True``, ``energy_unit='eV'``),
with **no** bespoke ``calculate()`` or units code — ``GenericASECalculator``
delegates to the upstream ASE calculator and routes results through the shared
``_finalize_results`` chokepoint. Compare this file's ~30 substantive lines to the
native backend's hand-written ``calculate()``: that delta is exactly the
per-potential redundancy the adapter removes.

Production routing keeps using the native ``mace-mp-0`` name; this is the proof,
not a replacement.
"""
from __future__ import annotations

from typing import Literal, Optional

import torch

from ..calculator_base import register_calculator
from ._generic_ase_calculator import GenericASECalculator


def _device_str(device) -> str:
    """Map a torch.device / str to the 'cuda'|'cpu' string mace_mp expects."""
    if isinstance(device, torch.device):
        return 'cuda' if device.type == 'cuda' else 'cpu'
    return 'cuda' if str(device).startswith('cuda') else 'cpu'


@register_calculator
class MACEMPGenericCalculator(GenericASECalculator):
    """mace-mp-0 foundation model wrapped by the universal adapter."""

    MODEL_NAMES = ('mace-mp-0-generic',)
    MODEL_ENERGY_UNIT = 'eV'
    SUPPORTS_PBC = True
    SUPPORTS_CHARGE_MULT = False
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
        model: str = 'mace-mp-0-generic',
        mace_model: str = 'medium',
        model_path: Optional[str] = None,
        overwrite: bool = False,
        implicit: Literal['gbsa', 'none'] = 'none',
        solvent: str = 'none',
    ):
        # (a) construction shim: load the upstream periodic MACE-MP calculator.
        #     Imported lazily so gas-phase-only environments stay importable.
        from mace.calculators import MACECalculator, mace_mp

        dev = _device_str(device)
        if model_path is not None:
            upstream = MACECalculator(
                model_paths=model_path, device=dev, default_dtype='float64',
            )
        else:
            upstream = mace_mp(model=mace_model, device=dev, default_dtype='float64')

        # (b) capability flags: periodic + stress, eV-native. calculate(), unit
        #     conversion and results writing are all inherited from the adapter.
        super().__init__(
            upstream,
            supports_pbc=True,
            supports_charge_mult=False,
            energy_unit='eV',
            name='mace-mp-0-generic',
        )
        # Preserve the original torch device object (the adapter only needs a
        # device string for the upstream; keep the real one for parity).
        self.device = device
