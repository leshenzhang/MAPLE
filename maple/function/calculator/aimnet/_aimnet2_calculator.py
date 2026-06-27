from __future__ import annotations

import os
from typing import Dict, Literal

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from ..calculator_base import CalcABC, register_calculator


# --------------------------------------------
# Build dense neighbor list (N+1, M) sentinel padded
# --------------------------------------------
def nblist_dense_padded(coord: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Brute-force dense neighbor list for single molecule.
    Returns: (N+1, M) int32 tensor, sentinel row = N
    """
    device = coord.device
    N = coord.shape[0]
    if N == 0:
        return torch.full((1, 1), 0, dtype=torch.int32, device=device)

    diff = coord[:, None, :] - coord[None, :, :]
    dist2 = torch.sum(diff ** 2, dim=-1)
    dist2[torch.eye(N, dtype=torch.bool, device=device)] = float('inf')
    mask = dist2 <= cutoff ** 2
    M = max(int(mask.sum(dim=1).max().item()), 1)

    nbmat = torch.full((N + 1, M), N, dtype=torch.int32, device=device)
    for i in range(N):
        nb_i = torch.nonzero(mask[i], as_tuple=False).flatten()
        if nb_i.numel() > 0:
            nbmat[i, :min(nb_i.numel(), M)] = nb_i[:min(nb_i.numel(), M)]
    return nbmat

# --------------------------------------------
# Pad helpers
# --------------------------------------------
def pad_dim0(a: torch.Tensor, value=0.0) -> torch.Tensor:
    """
    Pad one row along dim0.
    For (N, C) -> (N+1, C), (N,) -> (N+1,)
    """
    pad_shape = list(a.shape)
    pad_shape[0] = 1
    pad_row = torch.full(pad_shape, value, dtype=a.dtype, device=a.device)
    return torch.cat([a, pad_row], dim=0)

def maybe_pad_dim0(a: torch.Tensor, N: int, value=0.0) -> torch.Tensor:
    """
    If a.shape[0] == N, return as is.
    If a.shape[0] == N-1, pad one row to length N.
    """
    diff = N - a.shape[0]
    assert diff in (0, 1), f"Invalid pad: {a.shape[0]} vs target {N}"
    if diff == 1:
        a = pad_dim0(a, value=value)
    return a

# ==========================================================
# AIMNet2 Calculator (single-molecule minimal version)
# ==========================================================
@register_calculator
class AIMNet2Calculator(CalcABC):
    implemented_properties = ['energy', 'forces', 'free_energy', 'hessian']

    MODEL_NAMES = ('aimnet2', 'aimnet2nse')
    MODEL_ENERGY_UNIT = 'eV'
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = False
    SUPPORTED_COULOMB_METHODS = ('simple', 'dsf')
    CHECKPOINT_FILENAME = {'aimnet2': 'aimnet2.pt', 'aimnet2nse': 'aimnet2nse.pt'}
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = ('coulomb_method',)
    MODEL_PATH_OPTION = 'model_path'

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        kwargs = {}
        coulomb_method = options.get('coulomb_method')
        if coulomb_method is not None:
            kwargs['coulomb_method'] = str(coulomb_method).lower()
        if resolved_model_path is not None:
            kwargs['model_path'] = resolved_model_path
        return kwargs

    def __init__(self, device: torch.device,
                model: str = 'aimnet2',
                model_path: str = None,
                coulomb_method: str = 'simple',
                implicit: Literal['gbsa', 'none'] = 'none',
                solvent: str = 'none',
                ):
        super().__init__()
        self.device = device

        # Load model
        if model_path is None:
            model_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.dirname(model_dir)
            model_path = os.path.join(model_dir, 'model', f'{model}.pt')
        self.model = torch.jit.load(model_path, map_location=device).eval()

        self.cutoff = float(getattr(self.model, 'cutoff'))
        self.cutoff_lr = float(getattr(self.model, 'cutoff_lr', float('inf')))
        # Always provide nbmat_lr to avoid TorchScript KeyError; method routing
        # handles cutoff_lr.
        self.lr = True
        self.hessian: str = 'analytic'

        self._set_lrcoulomb_method(coulomb_method)

        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def _set_lrcoulomb_method(self, method: str, cutoff: float = 15.0, dsf_alpha: float = 0.2):
        """
        Configure the long-range Coulomb interaction method if the model contains a 'lrcoulomb' submodule.
        method: 'simple' or 'dsf'. The historical 'ewald' selector is rejected
        until this wrapper carries validated cell/PBC/MIC inputs.
        cutoff: cutoff distance for long-range interactions
        dsf_alpha: DSF damping parameter (if used)
        """
        method = str(method).lower()
        if method == 'ewald':
            raise NotImplementedError(
                "AIMNet2 coulomb_method='ewald' requires validated PBC/cell/MIC support; "
                "use 'simple' or 'dsf'."
            )
        if method not in self.SUPPORTED_COULOMB_METHODS:
            raise ValueError(
                f"Invalid coulomb_method: {method!r}; expected one of 'simple', 'dsf'."
            )

        def _iter_lrcoulomb_mods(model):
            for name, mod in model.named_modules():
                if name == 'lrcoulomb':
                    yield mod

        for mod in _iter_lrcoulomb_mods(self.model):
            mod.method = method
            if method == 'dsf' and hasattr(mod, 'dsf_alpha'):
                mod.dsf_alpha = dsf_alpha

        self.cutoff_lr = float('inf') if method == 'simple' else float(cutoff)
        self._coulomb_method = method

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)

        needs_grad = ('forces' in properties or 'hessian' in properties)
        coord = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float32,
            device=self.device,
            requires_grad=needs_grad,
        )
        data = self._build_data(coord, atoms)

        # Pure model energy in eV; _finalize_results handles eV→Ha + solvent.
        energy_eV = self._forward_energy(data)

        if 'forces' in properties:
            grad_full = torch.autograd.grad(
                energy_eV, data['coord'], create_graph=('hessian' in properties)
            )[0]
            forces_eV = -grad_full[: coord.shape[0]]
            forces_np = forces_eV.detach().cpu().numpy()
        else:
            forces_np = None

        hessian = None
        if 'hessian' in properties:
            if self.solvent_correction is not None:
                raise NotImplementedError(
                    'Hessian calculation with implicit solvent is not implemented yet.'
                )
            hessian = self.get_hessian(atoms)

        self._finalize_results(atoms, energy=energy_eV.item(), forces=forces_np, hessian=hessian)

    def _build_data(self, coord: torch.Tensor, atoms) -> Dict[str, torch.Tensor]:
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32, device=self.device)
        mol_idx = torch.zeros(coord.shape[0], dtype=torch.int32, device=self.device)
        N = coord.shape[0]
        charge_val = float(atoms.info.get('charge', 0.0))
        mult_val = float(atoms.info.get('mult', 1.0))

        nbmat = nblist_dense_padded(coord, self.cutoff)
        data: Dict[str, torch.Tensor] = {
            'coord': pad_dim0(coord, value=0.0),
            'numbers': pad_dim0(Z, value=0),
            'charge': torch.tensor([charge_val], dtype=torch.float32, device=self.device),
            'mult': torch.tensor([mult_val], dtype=torch.float32, device=self.device),
            'mol_idx': pad_dim0(mol_idx, value=mol_idx[-1].item() if N > 0 else 0),
            'nbmat': nbmat,
        }

        lr_cutoff = self.cutoff_lr if np.isfinite(self.cutoff_lr) else self.cutoff
        data['nbmat_lr'] = nblist_dense_padded(coord, lr_cutoff)
        data['cutoff_lr'] = torch.tensor(lr_cutoff, device=self.device)
        return data

    def _forward_energy(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Pure model forward; returns energy in eV (model's native unit)."""
        with torch.jit.optimized_execution(False):
            out = self.model(data)
        return out['energy'].sum()

    def _analytic_hessian(self, atoms) -> np.ndarray:
        """Analytic Hessian via autograd. Returns (3N, 3N) np.ndarray in Hartree/Å²."""
        from ..calculator_base import EV2HARTREE

        coord = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        N = coord.shape[0]
        data = self._build_data(coord, atoms)

        energy = self._forward_energy(data) * EV2HARTREE

        forces_full = torch.autograd.grad(energy, data['coord'], create_graph=True)[0]
        forces = -forces_full[:N]

        hessian = -torch.stack([
            torch.autograd.grad(f, data['coord'], retain_graph=True)[0]
            for f in forces.flatten().unbind()
        ]).view(-1, 3, N + 1, 3)[:, :, :N, :]

        return hessian.detach().cpu().numpy().reshape(3 * N, 3 * N)
