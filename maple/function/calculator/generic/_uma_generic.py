"""uma (fairchem UMA) routed through the universal ``GenericASECalculator`` adapter.

Registered as ``uma-generic`` so it coexists with the native ``uma`` backend
(``uma._uma_calculator.UMACalculator``). The native ``UMACalculator`` subclasses
the third-party ``FAIRChemCalculator`` but does **not** implement ``stress`` — so
periodic NPT (the MAPLE ensemble gate requires a PBC + stress calculator) is
blocked on native ``uma`` (this was B-11 堵②). The *upstream*
``FAIRChemCalculator`` itself emits a voigt-6 stress for periodic tasks
(``omat``/``oc20``/...); wrapping that upstream object in
``GenericASECalculator`` surfaces the stress and unblocks NPT — exactly the
per-potential capability gap the universal adapter (B-36) closes with no bespoke
engine code.

Empirically (B-38, Ibex A100 job 47848031): periodic Cu ``omat`` returned a
finite isotropic voigt-6 stress and ``implemented_properties`` containing
``stress``; molecular H2O ``omol`` energy matched. See DECISION_LOG B-38.

Production note: native ``uma`` (omol, charge/spin aware, no stress) remains the
right backend for *molecular* UMA. ``uma-generic`` exists to give *periodic* UMA
the stress needed for NPT. The construction shim is the verified recipe
``load_predict_unit(ckpt, device) -> FAIRChemCalculator(predictor, task_name=task)``.
"""
from __future__ import annotations

import os
from typing import Literal, Optional

import torch

from ..calculator_base import register_calculator
from ._generic_ase_calculator import GenericASECalculator


def _device_str(device) -> str:
    """Map a torch.device / str to the 'cuda'|'cpu' string fairchem expects."""
    if isinstance(device, torch.device):
        return 'cuda' if device.type == 'cuda' else 'cpu'
    return 'cuda' if str(device).startswith('cuda') else 'cpu'


@register_calculator
class UMAGenericCalculator(GenericASECalculator):
    """fairchem UMA wrapped by the universal adapter — exposes stress for NPT."""

    MODEL_NAMES = ('uma-generic',)
    MODEL_ENERGY_UNIT = 'eV'
    SUPPORTS_PBC = True
    SUPPORTS_CHARGE_MULT = True
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = ('task', 'checkpoint_path', 'model_path', 'size')
    MODEL_PATH_OPTION = 'checkpoint_path'

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        opts = options or {}
        kwargs = {}
        if opts.get('task'):
            kwargs['task'] = opts['task']
        ckpt = opts.get('checkpoint_path') or opts.get('model_path') or resolved_model_path
        if ckpt is not None:
            kwargs['checkpoint_path'] = ckpt
        if opts.get('size'):
            kwargs['size'] = opts['size']
        return kwargs

    def __init__(
        self,
        device,
        model: str = 'uma-generic',
        task: str = 'omat',
        checkpoint_path: Optional[str] = None,
        size: Optional[str] = None,
        overwrite: bool = False,
        implicit: Literal['gbsa', 'none'] = 'none',
        solvent: str = 'none',
    ):
        # (a) construction shim — load the upstream fairchem predict-unit + its
        #     native ASE calculator. Lazy import so MD environments without
        #     fairchem (the plumed MD env) stay importable: uma-generic pulls
        #     fairchem in only when actually requested.
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit

        dev = _device_str(device)
        # A self-contained local .pt loads with zero HuggingFace network; a bare
        # HF size name (uma-s-1p1) also works when no local checkpoint is given.
        src = checkpoint_path or size or 'uma-s-1p1'
        predictor = load_predict_unit(src, device=dev)
        upstream = FAIRChemCalculator(predictor, task_name=task)

        # (b) capability flags: a periodic task (omat/...) emits voigt-6 stress,
        #     so SUPPORTS_PBC=True unblocks the NPT gate; UMA is charge/spin
        #     aware (atoms.info['charge']/['spin']). calculate(), unit conversion
        #     and results writing are all inherited from the adapter.
        super().__init__(
            upstream,
            supports_pbc=True,
            supports_charge_mult=True,
            energy_unit='eV',
            name=f'uma-generic[{task}]',
        )
        # Preserve the original torch device object for parity with native uma.
        self.device = device
