from __future__ import annotations

import os
from typing import Literal, Optional

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from ..calculator_base import CalcABC, hessian_via_double_autograd, register_calculator
from ._common import model_float_dtype, one_hot_node_attrs, radius_graph_no_pbc


# ------------------------ Input builder ------------------------

def build_inputs_from_atoms(atoms, model, device='cpu', positions=None, dtype=None):
    """Build model inputs from an ASE Atoms object.

    Returns (positions, node_attrs, edge_index, shifts, batch, ptr).
    The newer MACE wrappers may need total_charge / total_spin, but those are
    handled inside the wrapper.
    """
    device = torch.device(device)
    dtype = dtype or model_float_dtype(model)
    if positions is None:
        pos = torch.tensor(atoms.get_positions(), dtype=dtype, device=device)
    else:
        pos = positions
    Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
    atomic_number_table = [int(z) for z in model.atomic_numbers]

    node_attrs = one_hot_node_attrs(Z, atomic_number_table, dtype=pos.dtype)
    edge_index, shifts = radius_graph_no_pbc(pos, float(model.r_max))

    N = pos.size(0)
    batch = torch.zeros(N, dtype=torch.int64, device=device)
    ptr = torch.tensor([0, N], dtype=torch.int64, device=device)

    return pos, node_attrs, edge_index, shifts, batch, ptr


# ------------------------ Calculator ------------------------

@register_calculator
class MACEModelCalculator(CalcABC):
    """ASE calculator wrapping a traced MACE model."""

    implemented_properties = ['energy', 'forces', 'free_energy', 'hessian']

    MODEL_NAMES = ('maceomol',)
    MODEL_ENERGY_UNIT = 'eV'
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = False
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = True
    OPTION_KEYS = ()
    MODEL_PATH_OPTION = 'model_path'

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        kwargs = {}
        if resolved_model_path is not None:
            kwargs['model_path'] = resolved_model_path
        return kwargs

    def __init__(self,
        device: torch.device,
        model: str = 'maceomol',
        model_path: Optional[str] = None,
        overwrite: bool = False,
        implicit: Literal['gbsa', 'none'] = 'none',
        solvent: str = 'none',
        ):
        """
        Args:
            device (torch.device): compute device
            model (str): model name; expects model/<model>.pt
            model_path (str, optional): explicit path to the scripted model file;
                overrides the model-name lookup when provided.
            overwrite (bool): whether to overwrite existing models
            implicit (str): implicit-solvent model type
            solvent (str): solvent name
        """
        super().__init__()
        if model_path is None:
            model_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.dirname(model_dir)
            model_path = os.path.join(model_dir, 'model', f'{model}.pt')

        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.dtype = model_float_dtype(self.model)
        self.overwrite = overwrite

        self.r_max = float(self.model.r_max)
        self.atomic_numbers = [int(z) for z in self.model.atomic_numbers]
        self.hessian = 'analytic'

        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """Main ASE entry point."""
        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)

        # Single forward; positions carry grad only when forces are requested.
        needs_forces = 'forces' in properties
        positions = torch.tensor(
            atoms.get_positions(), dtype=self.dtype, device=self.device,
            requires_grad=needs_forces,
        )
        inputs = build_inputs_from_atoms(
            atoms, self.model, device=self.device, positions=positions, dtype=self.dtype
        )
        total_energy = self.model(*inputs)
        energy_eV = total_energy.sum()

        forces_np = None
        if needs_forces:
            forces = -torch.autograd.grad(energy_eV, positions)[0]
            forces_np = forces.detach().cpu().numpy()

        hessian = None
        if 'hessian' in properties:
            if self.solvent_correction is not None:
                raise NotImplementedError('Hessian calculation with implicit solvent is not implemented yet.')
            hessian = self.get_hessian(atoms)

        self._finalize_results(atoms, energy=energy_eV.item(), forces=forces_np, hessian=hessian)

    def _analytic_hessian(self, atoms) -> np.ndarray:
        """Analytic Hessian via autograd. Returns (3N, 3N) np.ndarray in Hartree/Å²."""
        from ..calculator_base import EV2HARTREE

        positions = torch.tensor(
            atoms.get_positions(), dtype=self.dtype, device=self.device, requires_grad=True,
        )
        inputs = build_inputs_from_atoms(atoms, self.model, device=self.device, positions=positions)

        def energy_fn():
            return self.model(*inputs).sum() * EV2HARTREE

        return hessian_via_double_autograd(energy_fn, positions)
