from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import ase.calculators.calculator

if TYPE_CHECKING:
    import torch


_REGISTRY: dict[str, type] = {}


def register_calculator(cls):
    """Register a calculator class under each name in cls.MODEL_NAMES.

    Raises ValueError on duplicate registration to surface accidental name
    collisions instead of silently overwriting.
    """
    for raw_name in cls.MODEL_NAMES:
        name = raw_name.lower()
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Calculator name '{name}' is already registered to {existing.__name__}; "
                f"refusing to overwrite with {cls.__name__}."
            )
        _REGISTRY[name] = cls
    return cls


def get_registered_calculator(name: str) -> type:
    return _REGISTRY[name.lower()]


def import_calculator_plugin(module_path: str) -> None:
    import importlib
    importlib.import_module(module_path)


def load_calculator_plugins_from_env() -> None:
    import os
    raw = os.environ.get('MAPLE_CALCULATOR_PLUGINS', '')
    for entry in (s.strip() for s in raw.split(',') if s.strip()):
        import_calculator_plugin(entry)


_NONE_OPTIONS = {'', 'none', 'null', 'false', '0'}


IMPLICIT_SOLVENT_FORCE_ERROR = (
    "Experimental implicit GB-polar solvation is energy-only. Forces, stress, "
    "Hessians, and HVPs are disabled because QEq charges are "
    "geometry-dependent and are not coupled variationally to the solvent "
    "energy."
)
IMPLICIT_SOLVENT_DERIVATIVE_PROPERTIES = {
    "forces",
    "force",
    "stress",
    "stresses",
    "virial",
    "virials",
    "hessian",
}


def reject_implicit_solvent_derivatives(calculator, properties):
    """Fail fast when experimental implicit solvation is asked for derivatives."""
    if not getattr(calculator, "solvent_correction", None):
        return _property_list(properties)
    normalized = _property_list(properties)
    requested = {str(prop).lower() for prop in normalized}
    if requested.intersection(IMPLICIT_SOLVENT_DERIVATIVE_PROPERTIES):
        raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)
    return normalized


def normalize_none_option(value):
    """Normalize user-facing none-like option values to the literal 'none'."""
    if value is None:
        return 'none'
    text = str(value).strip().lower()
    return 'none' if text in _NONE_OPTIONS else text


def validate_implicit_solvent_choice(implicit, solvent):
    """Normalize and validate implicit-solvent selector pair."""
    implicit_norm = normalize_none_option(implicit)
    solvent_norm = normalize_none_option(solvent)
    if implicit_norm == 'gbsa' and solvent_norm == 'none':
        raise ValueError(
            "implicit='gbsa' requires an explicit solvent name such as solvent='water'; "
            "use implicit='none' to disable implicit solvent."
        )
    return implicit_norm, solvent_norm


def parse_bool_option(value, *, name='option'):
    """Parse a model-option flag into a real bool.

    Accepts actual bools and the strings true/false/1/0/yes/no/on/off
    (case-insensitive). Raises ValueError on anything else so a typo like
    d4=flase fails loudly instead of silently enabling the flag — note that
    bool('false') is True, the exact trap this guards against.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in ('true', '1', 'yes', 'on'):
        return True
    if text in ('false', '0', 'no', 'off', ''):
        return False
    raise ValueError(
        f"Cannot parse {name}={value!r} as a boolean; use one of true/false/1/0/yes/no."
    )


EV2HARTREE = 1.0 / 27.211386245988


_PRECISION_ALIASES = {
    'fp64': 'fp64', 'float64': 'fp64', 'double': 'fp64', 'f64': 'fp64', '64': 'fp64',
    'fp32': 'fp32', 'float32': 'fp32', 'single': 'fp32', 'f32': 'fp32', '32': 'fp32',
    'tf32': 'tf32', 'tfloat32': 'tf32', 'tensorfloat32': 'tf32',
}


def normalize_precision(value):
    """Normalize a user precision selector to 'fp64' | 'fp32' | 'tf32'.

    fp64 (default) keeps the historical float64 MACE force path bit-for-bit.
    fp32 runs the model + neighbour-graph in IEEE single precision with TF32
    tensor cores OFF (a true single-precision reference). tf32 uses the same
    float32 storage path but enables A100 TensorFloat-32 matmul (~10-bit
    mantissa) on the big linear layers -- fastest, slightly lossier.
    """
    if value is None:
        return 'fp64'
    text = str(value).strip().lower()
    out = _PRECISION_ALIASES.get(text)
    if out is None:
        raise ValueError(
            f"Unknown precision={value!r}; use one of fp64 / fp32 / tf32."
        )
    return out


def precision_to_torch_dtype(precision):
    """Map a normalized precision string to the torch float dtype of the data path."""
    import torch
    precision = normalize_precision(precision)
    return torch.float64 if precision == 'fp64' else torch.float32


def apply_tf32_backend_flags(precision):
    """Set the global cuBLAS/cuDNN TF32 tensor-core switches for `precision`.

    tf32  -> allow TF32 ON  (A100 tensor cores, ~10-bit-mantissa matmul).
    fp32  -> allow TF32 OFF (true IEEE single, the clean fp32 reference).
    fp64  -> allow TF32 OFF (irrelevant at double precision).

    Writes process-global torch flags, explicitly per precision, so a prior
    tf32 calculator cannot silently bleed reduced-precision matmul into a
    later fp32/fp64 calculator in the same process. No-op on CPU-only builds.
    """
    import torch
    precision = normalize_precision(precision)
    allow = (precision == 'tf32')
    try:
        torch.backends.cuda.matmul.allow_tf32 = allow
        torch.backends.cudnn.allow_tf32 = allow
    except Exception:
        pass
    return precision


def _convert_energy_force_units(energy, forces, *, source_unit):
    """Convert backend (energy, forces) to Hartree and Hartree/Å.

    Backends declare MODEL_ENERGY_UNIT honestly. Hartree is a no-op; eV
    multiplies through by EV2HARTREE.
    """
    if source_unit == 'hartree':
        return energy, forces
    if source_unit == 'eV':
        energy_ha = energy * EV2HARTREE
        forces_ha = forces * EV2HARTREE if forces is not None else None
        return energy_ha, forces_ha
    raise ValueError(
        f"Unknown source_unit: {source_unit!r}; expected 'eV' or 'hartree'."
    )


def init_implicit_solvent(calc, implicit, solvent, device):
    """Shared implicit-solvent initializer.

    Usable by CalcABC subclasses and duck-typed calculators (UMA) so the
    GBSA/QEq construction lives in one place.
    """
    implicit, solvent = validate_implicit_solvent_choice(implicit, solvent)
    if implicit == 'gbsa':
        from .extra_correction import GBSA, QEqTorch

        calc.solvent_correction = GBSA(solvent=solvent, device=device)
        calc.chargecalc = QEqTorch(device=device)
    else:
        calc.solvent_correction = None


def atoms_has_pbc(atoms) -> bool:
    """Return True when an Atoms-like object carries any periodic boundary."""
    return atoms is not None and bool(np.any(getattr(atoms, 'pbc', False)))


def reject_periodic_atoms(atoms, backend_name: str) -> None:
    """Fail loudly for molecular wrappers that do not implement PBC graphs."""
    if atoms_has_pbc(atoms):
        raise NotImplementedError(
            f"{backend_name} is a no-PBC molecular wrapper. "
            "Use UMA or a backend-native PBC calculator for periodic systems."
        )


def numerical_hessian_from_atoms(calc, atoms, delta=0.002):
    """Numerical Hessian via central finite difference on forces.

    Polymorphic: works for any calculator with the ASE protocol
    (calc.calculate(atoms, properties=['forces'], system_changes=...) writes
    calc.results['forces'] as a (N, 3) ndarray). Returns float64 ndarray of
    shape (3N, 3N).
    """
    from ase.constraints import FixAtoms
    from ase.calculators.calculator import all_changes

    old_results = dict(getattr(calc, 'results', {}) or {})
    old_atoms = getattr(calc, 'atoms', None)
    try:
        N = len(atoms)
        pos0 = atoms.get_positions().copy()
        fixed = {
            i
            for c in getattr(atoms, 'constraints', []) or []
            if isinstance(c, FixAtoms)
            for i in c.get_indices()
        }
        movable = [i for i in range(N) if i not in fixed]

        H = np.zeros((3 * N, 3 * N), dtype=np.float64)
        if not movable:
            return H

        def force_at(positions):
            at = atoms.copy()
            at.set_positions(positions)
            if getattr(atoms, 'constraints', None):
                at.set_constraint(atoms.constraints)
            calc.calculate(at, properties=['forces'], system_changes=all_changes)
            return np.asarray(calc.results['forces'], dtype=np.float64)

        for a in movable:
            for k in range(3):
                # Displacing DOF j and measuring all forces yields -dF_i/dx_j = H[i, j],
                # i.e. column j of the Hessian. Fill the column, then symmetrize to
                # absorb finite-difference noise (the exact Hessian is symmetric).
                col = 3 * a + k
                pos_p = pos0.copy(); pos_p[a, k] += delta
                Fp = force_at(pos_p)
                pos_m = pos0.copy(); pos_m[a, k] -= delta
                Fm = force_at(pos_m)
                H[:, col] = (-(Fp - Fm) / (2.0 * delta)).reshape(-1)

        H = 0.5 * (H + H.T)
        if fixed:
            # PHVA embedding: a frozen atom contributes no Hessian row/column.
            # Symmetrization would otherwise smear the movable->fixed force
            # couplings (read from the raw, unconstrained results['forces']) into
            # the fixed DOFs; zero them so fixed atoms decouple cleanly. No-op when
            # there are no constraints, so the common path is unchanged.
            fixed_dofs = [3 * i + k for i in sorted(fixed) for k in range(3)]
            H[fixed_dofs, :] = 0.0
            H[:, fixed_dofs] = 0.0
        return H
    finally:
        calc.results = old_results
        calc.atoms = old_atoms


def hessian_via_double_autograd(energy_fn, leaf):
    """Row-by-row analytic Hessian via double autograd.

    Shared by every CalcABC backend whose ``_analytic_hessian`` builds the
    Hessian one column at a time. ``leaf`` is the position tensor created with
    ``requires_grad=True``; ``energy_fn()`` must return a scalar energy tensor
    that depends on ``leaf`` **already in the target unit** (eV-native backends
    multiply by ``EV2HARTREE`` inside ``energy_fn``; ANI is Hartree-native).

    Returns a ``(3N, 3N)`` float ndarray. Polymorphic over ``leaf`` shape —
    ``(N, 3)`` for MACE-style backends and ``(1, N, 3)`` for ANI both flatten to
    ``3N`` — because it only ever reshapes gradients to 1-D.
    """
    import torch

    energy = energy_fn()
    n = leaf.numel()
    hessian = torch.zeros((n, n), dtype=leaf.dtype, device=leaf.device)
    grad = torch.autograd.grad(energy, leaf, create_graph=True)[0].reshape(-1)
    for i in range(n):
        grad2 = torch.autograd.grad(grad[i], leaf, retain_graph=True)[0].reshape(-1)
        hessian[i, :] = grad2
    return hessian.detach().cpu().numpy()


def _property_list(properties):
    return ['energy'] if properties is None else properties


class CalcABC(ase.calculators.calculator.Calculator):
    # Protocol attributes — each subclass overrides what's relevant.
    MODEL_NAMES: tuple = ()
    MODEL_ENERGY_UNIT: str = 'eV'
    SUPPORTED_HESSIAN_MODES: tuple = ('numerical',)
    SUPPORTS_CHARGE_MULT: bool = False
    # Backends that accept a `precision` ctor kwarg (fp64/fp32/tf32) set this
    # True so SetCalculator threads the MD `precision` mdp key through; all
    # other backends ignore precision and keep their model-native dtype.
    SUPPORTS_PRECISION: bool = False
    SUPPORTS_PBC: bool = False
    CHECKPOINT_FILENAME: dict | None = None
    REQUIRES_LOCAL_MODEL_FILE: bool = False
    # None keeps legacy/plugins permissive. Shipped backends set an explicit
    # tuple so input typos fail before a model is loaded.
    OPTION_KEYS: tuple | None = None
    # Constructor kwarg that accepts an explicit user model_path, if any.
    MODEL_PATH_OPTION: str | None = None

    def __init__(self):
        super().__init__()

    def _reject_unsupported_pbc(self, atoms) -> None:
        if not self.SUPPORTS_PBC:
            reject_periodic_atoms(atoms, type(self).__name__)

    @staticmethod
    def _normalize_properties(properties):
        return _property_list(properties)

    @staticmethod
    def _total_charge_from_atoms(atoms) -> float:
        charge = getattr(atoms, "info", {}).get("charge", None)
        if charge is not None:
            return float(charge)

        if hasattr(atoms, "get_initial_charges"):
            initial_charges = np.asarray(atoms.get_initial_charges(), dtype=float)
            if initial_charges.size and np.all(np.isfinite(initial_charges)):
                return float(initial_charges.sum())

        return 0.0

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=ase.calculators.calculator.all_changes,
    ):
        target_atoms = atoms if atoms is not None else getattr(self, 'atoms', None)
        self._reject_unsupported_pbc(target_atoms)
        properties = reject_implicit_solvent_derivatives(self, properties)
        super().calculate(atoms, properties, system_changes)
        return target_atoms

    @classmethod
    def build_kwargs_from_options(cls, model, model_options, *, resolved_model_path=None):
        """Translate input-header options into ctor kwargs. Backends override."""
        return {}

    def _finalize_results(self, atoms, *, energy, forces=None, hessian=None, stress=None, unit=None):
        """Single entry: unit conversion + implicit-solvent + write self.results.

        Backends pass the pure model outputs (in the unit declared by
        MODEL_ENERGY_UNIT). This method converts to Hartree, then optionally
        adds the implicit-solvent correction, then writes self.results.
        """
        source_unit = unit if unit is not None else self.MODEL_ENERGY_UNIT
        energy_ha, forces_ha = _convert_energy_force_units(
            energy, forces, source_unit=source_unit
        )

        if getattr(self, 'solvent_correction', None) is not None:
            # GB-polar solvation is energy-only. Reaching here with forces under
            # active solvent means the calculate()/get_hessian() guards were
            # bypassed; fail loudly rather than emit a solvent-inconsistent force.
            if forces_ha is not None:
                raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)
            solvent_energy = self.implicit_solv_energy(atoms)
            se = solvent_energy.item() if hasattr(solvent_energy, 'item') else float(solvent_energy)
            energy_ha = energy_ha + se

        # Sole results-writing chokepoint for every CalcABC backend: clear first
        # so an energy-only call cannot inherit stale forces/hessian from a
        # previous forces/hessian call on the same calculator instance.
        self.results = {}
        self.results['energy'] = float(energy_ha)
        self.results['free_energy'] = float(energy_ha)
        if forces_ha is not None:
            self.results['forces'] = forces_ha
        if hessian is not None:
            self.results['hessian'] = hessian
        if stress is not None:
            # Consumed by dispatcher/md/utils.compute_instantaneous_pressure in
            # eV/Å³ (ASE Voigt convention); NOT converted to Hartree/Bohr³.
            self.results['stress'] = np.asarray(stress, dtype=np.float64)

    def get_hessian(self, atoms, delta: float = 0.002):
        """Dispatch on self.hessian. Subclasses may override for backend autograd."""
        self._reject_unsupported_pbc(atoms)
        mode = getattr(self, 'hessian', self.SUPPORTED_HESSIAN_MODES[0])
        if mode == 'analytic':
            if getattr(self, 'solvent_correction', None) is not None:
                raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)
            return np.asarray(self._analytic_hessian(atoms))
        if mode == 'numerical':
            if getattr(self, 'solvent_correction', None) is not None:
                raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)
            return numerical_hessian_from_atoms(self, atoms, delta)
        raise ValueError(f"Unknown hessian mode: {mode!r}")

    def _analytic_hessian(self, atoms):
        """Backend autograd Hessian. Override in subclasses that can autodiff."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement analytic Hessian; "
            "set hessian='numerical' or override _analytic_hessian."
        )


    def log_error(self, error_message: str) -> None:
        """
        Logs error messages to the output file.

        Args:
            error_message: The error message to log.
        """
        with open(self.output, 'a', encoding="utf-8") as file:
            file.write(f"ERROR: {error_message}\n")

    def log_info(self, info_message: list) -> None:
        """
        Logs info messages to the output file.

        Args:
            info_message: The info message to log.
        """
        with open(self.output, 'a', encoding="utf-8") as file:
            for info in info_message:   
                file.write(f"{info}")

    def get_hvp(self, atoms, n: np.ndarray):
        """Hessian-vector product Hn. Override in backends that support Dimer-mode TS.

        There is deliberately no shared default: the autograd forward shape
        differs per backend, so a single implementation would silently misread
        non-matching models. Backends that can autodiff their forward override
        this; everyone else fails loudly here instead of returning garbage.
        """
        if getattr(self, 'solvent_correction', None) is not None:
            raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_hvp; Dimer-mode TS "
            "requires a backend-specific Hessian-vector product."
        )

    def implicit_solv_init(self, implicit: str, solvent: str):
        init_implicit_solvent(self, implicit, solvent, self.device)
    
    def implicit_solv_energy(self, atoms: ase.Atoms) -> torch.Tensor:
        """
        Compute implicit solvent correction energy if applicable.

        Args:
            atoms (ase.Atoms): Atomic structure.

        Returns:
            torch.Tensor: Implicit solvent correction energy in Hartree.
        """
        atoms.atomic_charges = self.chargecalc(
            atoms, total_charge=self._total_charge_from_atoms(atoms)
        )
        solvent_energy,_ = self.solvent_correction.get_energy(atoms)
        return solvent_energy

    def implicit_solv_energy_and_force(self, atoms: ase.Atoms) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute implicit solvent correction energy and forces if applicable.

        Args:
            atoms (ase.Atoms): Atomic structure.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Implicit solvent correction energy in Hartree and forces in Hartree/Å.
        """
        raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)
