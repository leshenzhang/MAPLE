# Authoring MAPLE calculators

MAPLE backends are registered classes that satisfy a narrow protocol. This
document is the contract a new calculator must honor and the migration path
for downstream code that depended on the legacy entry points.

## What MAPLE expects from a calculator

A backend is any class that exposes the public surface below; it does **not**
have to inherit `CalcABC`. UMA, for example, extends third-party
`FAIRChemCalculator` and satisfies the protocol by attribute presence.

**Public methods:**
- `calculate(self, atoms, properties, system_changes)` — ASE entry point.
- `get_hessian(self, atoms, delta=0.002)` — returns a `(3N, 3N) np.ndarray`
  in Hartree / Å². `CalcABC` provides a default that dispatches on
  `self.hessian`.
- `get_hvp(self, atoms, n)` — optional; required only for the HVP-enabled
  Dimer path (`use_hvp=True`). Returns the 3-tuple `(Hn, forces, energy)` as
  torch tensors: `Hn` is the Hessian–vector product `H·n` flattened to `(3N,)`,
  `forces` is the `(3N,)` force vector, and `energy` is a scalar — all in
  Hartree units (energy in Hartree, forces in Hartree/Å, `H` in Hartree/Å²).
  The HVP-enabled Dimer path unpacks all three; the regular Dimer path can use
  finite-difference forces instead. `CalcABC.get_hvp` raises
  `NotImplementedError` by default — there is **no** shared autograd default,
  because the forward shape differs per backend. `ANICalculator` implements it
  for ANI's `self.model(species, coords)` shape; other backends must override
  it before they can use the HVP-enabled Dimer path.

**Class attributes** (read by `SetCalculator` before the class is
instantiated):

| Attribute | Type | Purpose |
|---|---|---|
| `MODEL_NAMES` | `tuple[str, ...]` | Registry routing keys, lowercase. |
| `MODEL_ENERGY_UNIT` | `'eV'` or `'hartree'` | Declares the backend's `calculate()` energy unit. `_finalize_results` converts to Hartree based on this — **do not multiply by `EV2HARTREE` yourself**. |
| `SUPPORTED_HESSIAN_MODES` | `tuple[str, ...]` | Subset of `('analytic', 'numerical')`. |
| `SUPPORTS_CHARGE_MULT` | `bool` | True if the backend honors `atoms.info['charge']` / `atoms.info['mult']`. |
| `SUPPORTS_PBC` | `bool` | True only when the backend constructs a validated periodic graph / neighbor list. `SetCalculator` and `CalcABC.calculate()` reject periodic atoms for false values. |
| `CHECKPOINT_FILENAME` | `dict[str, str] \| None` | Per-name filename for HuggingFace auto-download. `None` if no auto-download. |
| `REQUIRES_LOCAL_MODEL_FILE` | `bool` | Fallback when `CHECKPOINT_FILENAME` does not cover the requested name. |
| `OPTION_KEYS` | `tuple[str, ...] \| None` | Supported backend-specific `model_options`. Shipped backends set this so typos fail loudly; `None` keeps legacy plug-ins permissive. |
| `MODEL_PATH_OPTION` | `str \| None` | Constructor kwarg that consumes an explicit user `model_path` (`'model_path'` or `'checkpoint_path'`). `None` rejects an explicit `model_path` **only when** the constructor also exposes no `model_path`/`checkpoint_path` parameter — `SetCalculator` falls back to inspecting the ctor signature for support detection. The resolved path is still delivered through `build_kwargs_from_options(..., resolved_model_path=...)`, so legacy plug-ins that omit `MODEL_PATH_OPTION` must route that value into their constructor kwargs explicitly. |

`SUPPORTED_COULOMB_METHODS` (`tuple[str, ...]`) is an additional, backend-specific
gate that `SetCalculator` reads before instantiation: AIMNet2 declares it so an
unsupported `coulomb_method` option is rejected before the model is built. It is
not part of the core protocol — `SetCalculator` consults it only for the
`coulomb_method` key — and is noted here so the capability surface is complete.

## Minimum runnable subclass

```python
from maple.function.calculator import CalcABC, register_calculator
import numpy as np


@register_calculator
class FooCalculator(CalcABC):
    MODEL_NAMES = ('foo2x', 'foo1ccx')
    MODEL_ENERGY_UNIT = 'eV'           # or 'hartree' — declare honestly per backend
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = False               # fail fast on periodic atoms unless validated
    CHECKPOINT_FILENAME = {'foo2x': 'foo2x.pt', 'foo1ccx': 'foo1ccx.pt'}
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = ('foo_mode',)
    MODEL_PATH_OPTION = 'model_path'

    implemented_properties = ['energy', 'free_energy', 'forces']

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        # Translate input-header options + factory-resolved path into ctor kwargs.
        return {}

    def __init__(self, device, model, *, implicit='none', solvent='none', **backend_kwargs):
        super().__init__()
        self.device = device
        self.hessian = 'analytic'
        self.implicit_solv_init(implicit=implicit, solvent=solvent)
        # ...load self.model, etc...

    def calculate(self, atoms=None, properties=['energy'], system_changes=None):
        atoms = super().calculate(atoms, properties, system_changes)
        energy, forces = self._forward(atoms)      # private, backend-internal
        self._finalize_results(atoms, energy=energy, forces=forces)

    def _analytic_hessian(self, atoms) -> np.ndarray:
        # backend autograd; return np.ndarray (3N, 3N) in Hartree / Å²
        ...
```

## Unit contract

- Set `MODEL_ENERGY_UNIT` honestly. ANI 's TorchScript model returns Hartree
  natively, so ANI declares `'hartree'` and `_finalize_results` skips the
  conversion. Every other shipped backend declares `'eV'`.
- `_finalize_results` is the **only** path that converts the
  `calculate()` energy / forces flow to Hartree. **Custom calculators must
  not multiply by `EV2HARTREE` themselves** for the `calculate()` path.
- The Hessian path is independent: `_analytic_hessian` must return
  Hartree / Å² directly (each eV-native backend multiplies by `EV2HARTREE`
  inside its analytic method); the numerical path inherits Hartree via
  `numerical_hessian_from_atoms`, which calls back into `calculate()`.

## Implicit solvent

- `_finalize_results` is the only place that adds the GBSA correction.
  **Custom calculators do not call `implicit_solv_energy_and_force()`
  directly** for the `calculate()` flow. If a backend needs special
  handling, override `_finalize_results` rather than duplicating the
  solvent path.
- Solvent setup happens via `init_implicit_solvent(calc, implicit,
  solvent, device)`. `CalcABC.__init__` does not call this for you in the
  current release; subclasses still invoke `self.implicit_solv_init(...)`
  inside their own `__init__`.
- `None`, `none`, `null`, `false`, `0`, and empty strings normalize to
  `none`. `implicit='gbsa'` requires a real solvent name such as
  `solvent='water'`; MAPLE fails early instead of looking for `None.dat`.

## Hessian

- `self.hessian` selects `'analytic'` or `'numerical'`. Numerical falls
  through to the shared `numerical_hessian_from_atoms` helper for free.
- `numerical_hessian_from_atoms` restores the calculator's pre-call
  `results` before returning, so standalone `calc.get_hessian(atoms)` does
  not leave `results` pointing at the final displaced geometry.
- Analytic Hessian with implicit solvent is unsupported and raises
  `NotImplementedError` from `CalcABC.get_hessian`. Document the
  limitation in any backend-specific notes.

## Periodic boundary conditions and stress

- `SUPPORTS_PBC = False` is the default. If any `atoms.pbc` component is
  true, `SetCalculator` rejects the model before construction and
  `CalcABC.calculate()` rejects direct backend use. This prevents molecular
  wrappers from silently treating periodic systems as isolated clusters.
- AIMNet2's hand wrapper is no-PBC. `coulomb_method='ewald'` is disabled
  until cell/PBC/MIC inputs and reference tests exist; use `simple` or `dsf`.
- The hand-wrapped MACE backends are no-PBC and do not provide validated
  stress/virial output. Use UMA or a backend-native periodic calculator for
  periodic stress/NPT workflows.
- UMA delegates PBC graph construction to FAIR-Chem and is the only shipped
  backend with `SUPPORTS_PBC = True`; MAPLE currently rejects UMA
  `stress`/`virial` requests until unit conversion is validated. Periodic UMA
  calculations must set an explicit FAIR-Chem `task=` (`omat`, `oc20`, `oc22`,
  `oc25`, `omc`, or `odac`); MAPLE no longer silently maps every periodic
  system to `omat`, and rejects `task='omol'` with periodic atoms.

## Charge / multiplicity

- `atoms.info['charge']` and `atoms.info['mult']` carry the values. Backends
  that require integer charge/spin inputs must reject non-integer values rather
  than truncating them.
- Set `SUPPORTS_CHARGE_MULT = True` if the backend honors them; otherwise
  `SetCalculator` warns the user that the values will be ignored.
- MACE-POLAR rejects non-integer `mult` before converting multiplicity to the
  model's unpaired-electron `spin = mult - 1` input.
- UMA `omol` charged/open-shell inputs are passed through to FAIR-Chem and
  emit a warning until MAPLE has accepted golden numerical tolerances for
  those states.
- UMA non-`omol` tasks reject non-default `charge`/`mult` because FAIR-Chem's
  current calculator contract only uses charge/spin for the `omol` head.

## Backend-specific kwargs

- Override `build_kwargs_from_options(cls, model, options, *,
  resolved_model_path=None)` to translate input-header `model_options`
  plus the factory-resolved checkpoint path into ctor kwargs. **Do not
  call back into `SetCalculator` from a backend method.** The factory
  passes `resolved_model_path` for backends that need a local file.
- Set `OPTION_KEYS` for every shipped backend-specific option. Unknown keys
  are rejected before model construction so a misspelled production config
  cannot be silently ignored.
- If explicit `model_path` is supported, set `MODEL_PATH_OPTION`. If a user
  passes `model_path` to a backend that does not support it, MAPLE raises a
  configuration error instead of silently falling back to a default checkpoint.

## HVP override

- `CalcABC.get_hvp` raises `NotImplementedError`. Override it only if this
  model will use the HVP-enabled Dimer path (`use_hvp=True`); regular Dimer can
  fall back to finite-difference forces. `ANICalculator` is the only shipped
  backend with an implementation (autograd over the `(species, coords)`
  forward); the rest fail loudly rather than misread a differently-shaped
  forward.
- Unlike the `calculate()` path, `get_hvp` does **not** route through
  `_finalize_results`, so the backend converts units itself: an eV-native
  backend must apply `EV2HARTREE` inside `get_hvp` and return Hartree-unit
  tensors. ANI is Hartree-native, so its implementation needs no conversion.
- ANI HVP rejects implicit-solvent runs because solvent HVP is not implemented;
  returning gas-phase HVP in that mode would be a silent mixed-model result.

## Backend capability matrix

What the shipped backends actually support today. The hand-wrapped MACE
backends build a **no-PBC** radius graph (zero `cell`/`shifts`), so they are
molecule-only regardless of any periodic claim elsewhere. UMA is the only
backend that switches tasks for periodic input.

| Backend (names) | PBC | charge/mult | Hessian | Implicit solvent | D4 | HVP (Dimer) |
|---|---|---|---|---|---|---|
| ANI (`ani2x/1x/1ccx/1xnr`) | no; fail-fast | no | analytic + numerical | yes | yes | yes; no implicit solvent |
| AIMNet2 (`aimnet2`, `aimnet2nse`) | no; fail-fast; no Ewald | yes | analytic + numerical | yes | no | no |
| MACE-OFF (`maceoff23s/m/l`, `egret`) | no; fail-fast | no | analytic + numerical | yes | no | no |
| MACE-omol (`maceomol`) | no; fail-fast | no | analytic + numerical | yes | no | no |
| MACE-POLAR (`macepols/m/l`) | no; fail-fast; no external field | yes (`spin = mult − 1`) | analytic + numerical | yes | no | no |
| UMA (`uma`) | yes; non-PBC auto `omol`; PBC requires explicit non-`omol` task; stress rejected | `omol` only (`spin = mult`); non-`omol` rejects non-default charge/mult | numerical only | yes | no | no |

`spin` semantics differ on purpose: MACE-POLAR's traced interface takes the
number of unpaired electrons (`mult − 1`), UMA's FAIR-Chem path takes the
spin multiplicity (`mult`). Confirm against the specific checkpoint before
trusting open-shell results — neither encoding is verified here.

Observed on a local uma-s-1p1 checkpoint: direct FAIR-Chem and the MAPLE UMA
wrapper agree for H₂O `q=0/+1/-1` and `mult=3`, so charge/spin reaches
FAIR-Chem through MAPLE. The `q=0 → +1` same-geometry energy change is small
(~4.6e-5 Ha) for that checkpoint, so keep a warning and verify charged/open-
shell energetics against FAIR-Chem/reference calculations before relying on
them for production chemistry.

## Plug-in discovery — three layers

| Layer | Mechanism | Status |
|---|---|---|
| 1 | Input header `#model=mymodel(module=my_lab.maple_plugin, model_path=/tmp/m.pt)` | Implemented |
| 2 | Env var `MAPLE_CALCULATOR_PLUGINS=my_lab.maple_plugin,other.plugin` | Implemented (loaded once per process at first `_build_calculator`) |
| 3 | Python entry points: `[project.entry-points."maple.calculators"]` | Preview only; not implemented |

The recommended `pyproject.toml` shape for Layer 3, so packagers can prepare
ahead of time:

```toml
[project.entry-points."maple.calculators"]
my_lab = "my_lab.maple_plugin"
```

## Public vs private API

- **Public**: `calculate`, `get_hessian`, `get_hvp`, the protocol class
  attributes above.
- **Private** (do not depend on from outside the calculator): `_forward_energy`,
  `_build_inputs`, `_analytic_hessian`, `self.model`.
- **Banned**: `calc.get_energy(...)` (removed from every backend), reading
  `self.results` before `calculate()` has been invoked.

## Migration from `calc.get_energy(...)`

Downstream code that previously called `calc.get_energy(atoms)` must
switch to ASE's standard pattern:

```python
atoms.calc = calc
energy = atoms.get_potential_energy()
```

`get_energy` carried three different signatures across the shipped
backends; no single contract was honest. The ASE accessor reads
`calc.results['energy']` after `calculate()` runs, which is now the only
public energy entry point.

## Registration collisions

`@register_calculator` raises `ValueError` on a duplicate name rather than
silently overwriting. If you see this error, the registry already has a
class registered under one of your `MODEL_NAMES` entries — a collision is
a real bug, not a feature.

## UMA is a documented exception

UMA does not inherit `CalcABC` because it already extends
`FAIRChemCalculator`. It satisfies the protocol via attribute presence
and registers via `@register_calculator`. Do not write
`isinstance(calc, CalcABC)` anywhere in the dispatcher — the calculator
surface is duck-typed by design.
