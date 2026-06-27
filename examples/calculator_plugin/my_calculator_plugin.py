"""Template: a user-supplied MAPLE calculator backend.

Copy this file into your own package (e.g. ``my_lab/maple_plugin.py``), rename
the classes, and fill in the two TODO blocks: how to *load* your model and how
to *run* its forward pass. Everything else is the MAPLE calculator contract and
should not need changing.

This file ships TWO complete backends so you can pick whichever matches your
checkpoint format:

  * ``TemplateJitCalculator``  — loads a TorchScript file (``.pt`` / ``.jpt``)
                                  with ``torch.jit.load``. Use this if your model
                                  was saved with ``torch.jit.script/trace``.
  * ``TemplatePtCalculator``   — loads a plain checkpoint (``.pt`` / ``.model``)
                                  with ``torch.load`` and rebuilds an
                                  ``nn.Module`` from a state_dict. Use this for an
                                  ordinary (non-scripted) PyTorch model.

Both follow the exact contract verified in
``maple/function/calculator/CALCULATOR_REVIEW.md``.

Wire it into MAPLE from an input file, with NO edits to MAPLE itself:

    #model=mytemplate-jit(module=my_calculator_plugin, model_path=/abs/toy_jit.pt)
    #sp
    #device=cpu

or via the environment:

    export MAPLE_CALCULATOR_PLUGINS=my_calculator_plugin

Run ``python my_calculator_plugin.py`` directly to build two toy checkpoints and
drive both calculators end-to-end through MAPLE's factory — no real model needed.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from ase.calculators.calculator import all_changes

# When this file lives inside your own package, import from the installed maple:
from maple.function.calculator.calculator_base import (
    CalcABC,
    EV2HARTREE,
    hessian_via_double_autograd,
    register_calculator,
)


# ===========================================================================
# 0. A toy model standing in for *your* network.
#    Replace this entirely with your own architecture. It only needs to map
#    (atomic_numbers, positions) -> a scalar energy that is differentiable
#    w.r.t. positions. Units here are eV (declared via MODEL_ENERGY_UNIT below).
# ===========================================================================
class ToyHarmonicModel(nn.Module):
    """Energy = sum_{i<j} 0.5 * k * (r_ij - r0)^2 over close pairs (toy only)."""

    def __init__(self, k: float = 1.0, r0: float = 1.2, rcut: float = 3.0):
        super().__init__()
        self.k = float(k)
        self.r0 = float(r0)
        self.rcut = float(rcut)

    def forward(self, numbers: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        rij = positions[:, None, :] - positions[None, :, :]
        d = torch.sqrt((rij * rij).sum(-1) + 1e-12)
        iu = torch.triu_indices(d.shape[0], d.shape[0], offset=1, device=d.device)
        dij = d[iu[0], iu[1]]
        mask = dij < self.rcut
        e = 0.5 * self.k * ((dij[mask] - self.r0) ** 2).sum()
        return e


# ===========================================================================
# 1. TorchScript variant — load a scripted/traced ``.pt`` / ``.jpt``.
# ===========================================================================
@register_calculator
class TemplateJitCalculator(CalcABC):
    # -- Capability declaration (read by SetCalculator before construction) --
    MODEL_NAMES = ('mytemplate-jit',)        # input-header routing keys, lowercase
    MODEL_ENERGY_UNIT = 'eV'                  # 'eV' or 'hartree' — declare honestly
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = False                      # fail-fast on periodic atoms
    CHECKPOINT_FILENAME = None                # no HuggingFace auto-download
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = ()                          # extra model_options keys you accept
    MODEL_PATH_OPTION = 'model_path'          # the kwarg that consumes model_path

    implemented_properties = ['energy', 'forces', 'free_energy', 'hessian']

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        """Translate header options + factory-resolved path into __init__ kwargs."""
        kwargs = {}
        if resolved_model_path is not None:
            kwargs['model_path'] = resolved_model_path
        return kwargs

    def __init__(self, device, model='mytemplate-jit', *, model_path=None,
                 implicit='none', solvent='none', **_ignored):
        super().__init__()
        self.device = device
        self.dtype = torch.float64
        self.hessian = 'analytic'

        # --- TODO: load YOUR scripted model here -------------------------------
        if model_path is None:
            raise ValueError(
                "TemplateJitCalculator needs model_path=/path/to/scripted.pt "
                "(pass it in the #model=...(model_path=...) header)."
            )
        self.model = torch.jit.load(model_path, map_location=device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # ----------------------------------------------------------------------

        # Implicit-solvent hook (no-op unless implicit='gbsa'). Keep this call.
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def _forward(self, atoms, *, requires_grad):
        """Single private forward. Returns (energy_tensor_eV, positions_leaf)."""
        positions = torch.tensor(
            atoms.get_positions(), dtype=self.dtype, device=self.device,
            requires_grad=requires_grad,
        )
        numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        energy_eV = self.model(numbers, positions)          # YOUR forward signature
        return energy_eV, positions

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)  # PBC/solvent guards

        needs_forces = 'forces' in properties
        energy_eV, positions = self._forward(atoms, requires_grad=needs_forces)

        forces_np = None
        if needs_forces:
            forces = -torch.autograd.grad(energy_eV, positions)[0]
            forces_np = forces.detach().cpu().numpy()

        hessian = None
        if 'hessian' in properties:
            hessian = self.get_hessian(atoms)

        # _finalize_results owns eV->Hartree + solvent + writing self.results.
        # Do NOT multiply by EV2HARTREE yourself in this path.
        self._finalize_results(atoms, energy=energy_eV.item(), forces=forces_np, hessian=hessian)

    def _analytic_hessian(self, atoms) -> np.ndarray:
        # The Hessian path converts units itself (it does NOT go through
        # _finalize_results): multiply the energy by EV2HARTREE here.
        energy_eV, positions = self._forward(atoms, requires_grad=True)
        return hessian_via_double_autograd(lambda: energy_eV * EV2HARTREE, positions)


# ===========================================================================
# 2. Plain-checkpoint variant — load a state_dict ``.pt`` / ``.model``.
# ===========================================================================
@register_calculator
class TemplatePtCalculator(CalcABC):
    MODEL_NAMES = ('mytemplate-pt',)
    MODEL_ENERGY_UNIT = 'eV'
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = False
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = ()
    MODEL_PATH_OPTION = 'model_path'

    implemented_properties = ['energy', 'forces', 'free_energy', 'hessian']

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        kwargs = {}
        if resolved_model_path is not None:
            kwargs['model_path'] = resolved_model_path
        return kwargs

    def __init__(self, device, model='mytemplate-pt', *, model_path=None,
                 implicit='none', solvent='none', **_ignored):
        super().__init__()
        self.device = device
        self.dtype = torch.float64
        self.hessian = 'analytic'

        # --- TODO: rebuild YOUR architecture and load weights -----------------
        if model_path is None:
            raise ValueError("TemplatePtCalculator needs model_path=/path/to/model.pt")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        # A real plugin reads hyperparameters from the checkpoint; here the toy
        # model has fixed ones. Construct the module, then load its weights.
        net = ToyHarmonicModel(**ckpt.get('hparams', {}))
        if 'state_dict' in ckpt:
            net.load_state_dict(ckpt['state_dict'])
        self.model = net.to(device).double().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # ----------------------------------------------------------------------

        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def _forward(self, atoms, *, requires_grad):
        positions = torch.tensor(
            atoms.get_positions(), dtype=self.dtype, device=self.device,
            requires_grad=requires_grad,
        )
        numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        energy_eV = self.model(numbers, positions)
        return energy_eV, positions

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)

        needs_forces = 'forces' in properties
        energy_eV, positions = self._forward(atoms, requires_grad=needs_forces)

        forces_np = None
        if needs_forces:
            forces = -torch.autograd.grad(energy_eV, positions)[0]
            forces_np = forces.detach().cpu().numpy()

        hessian = None
        if 'hessian' in properties:
            hessian = self.get_hessian(atoms)

        self._finalize_results(atoms, energy=energy_eV.item(), forces=forces_np, hessian=hessian)

    def _analytic_hessian(self, atoms) -> np.ndarray:
        energy_eV, positions = self._forward(atoms, requires_grad=True)
        return hessian_via_double_autograd(lambda: energy_eV * EV2HARTREE, positions)


# ===========================================================================
# 3. Self-contained demo: build toy checkpoints, then run both backends through
#    MAPLE's real factory (SetCalculator) exactly as an input file would.
# ===========================================================================
def _build_toy_checkpoints(tmpdir):
    import os
    jit_path = os.path.join(tmpdir, 'toy_jit.pt')
    pt_path = os.path.join(tmpdir, 'toy_state.pt')

    model = ToyHarmonicModel().double().eval()
    torch.jit.save(torch.jit.script(model), jit_path)
    torch.save({'hparams': {'k': 1.0, 'r0': 1.2, 'rcut': 3.0},
                'state_dict': model.state_dict()}, pt_path)
    return jit_path, pt_path


def _demo():
    import tempfile
    from ase import Atoms
    from maple.function.calculator.set_calculator import SetCalculator

    atoms = Atoms('OH2', positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])

    with tempfile.TemporaryDirectory() as tmp:
        jit_path, pt_path = _build_toy_checkpoints(tmp)
        cases = [
            ('mytemplate-jit', jit_path),
            ('mytemplate-pt', pt_path),
        ]
        for model_name, path in cases:
            at = atoms.copy()
            sc = SetCalculator(
                'cpu', model_name, '/tmp/template_demo.out', atoms=at,
                model_options={'module': __name__, 'model_path': path},
            )
            at.calc = sc.set_calculator()
            e = at.get_potential_energy()
            f = at.get_forces()
            at.calc.hessian = 'analytic'
            Ha = at.calc.get_hessian(at)
            at.calc.hessian = 'numerical'
            Hn = at.calc.get_hessian(at)
            print(f"{model_name:16s} E={e:.6f} Ha  |F|max={np.abs(f).max():.5f}  "
                  f"H{Ha.shape} sym={np.max(np.abs(Ha - Ha.T)):.1e} "
                  f"|analytic-numerical|={np.max(np.abs(Ha - Hn)):.1e}")
    print("template demo OK")


if __name__ == '__main__':
    _demo()
