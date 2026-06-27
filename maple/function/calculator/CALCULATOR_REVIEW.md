# MAPLE Calculator Subsystem — Review & Unification Notes

*Branch: `review/calculator-unify` (from `enhance`). Companion to
[`AUTHORING.md`](AUTHORING.md), which is the authoritative backend contract.*

This document records (1) the state of the calculator subsystem as reviewed,
(2) the inconsistencies that were found, (3) what this branch changed to unify
them, and (4) the standard a user-supplied calculator must follow. The runnable
template lives at [`examples/calculator_plugin/`](../../../examples/calculator_plugin/).

---

## 1. What was already unified (before this branch)

The subsystem is **not** un-designed; a prior refactor already established most
of the contract. Worth stating so we don't re-invent it:

- **One protocol base class** — `CalcABC` (`calculator_base.py`) declares
  capability via class attributes (`MODEL_NAMES`, `MODEL_ENERGY_UNIT`,
  `SUPPORTED_HESSIAN_MODES`, `SUPPORTS_CHARGE_MULT`, `SUPPORTS_PBC`,
  `CHECKPOINT_FILENAME`, `REQUIRES_LOCAL_MODEL_FILE`, `OPTION_KEYS`,
  `MODEL_PATH_OPTION`).
- **One registry + 3-layer plug-in discovery** — `register_calculator`,
  input-header `module=`, and the `MAPLE_CALCULATOR_PLUGINS` env var, all funneled
  through the `SetCalculator` factory.
- **`get_energy()` is already gone.** Every job consumes calculators through the
  ASE protocol only: `atoms.get_potential_energy()`, `atoms.get_forces()`,
  `atoms.calc.get_hessian(atoms)`, `atoms.calc.get_hvp(atoms, n)`. There is no
  remaining `calc.get_energy(...)` call anywhere in the dispatcher.
- **One unit/solvent chokepoint** — `_finalize_results()` converts to Hartree
  (per `MODEL_ENERGY_UNIT`) and adds the implicit-solvent term. Backends emit raw
  model outputs only.

So the goal of this branch is **finishing the job**, not designing it from zero.

---

## 2. Inconsistencies found (and their disposition)

| # | Finding | Severity | Status on this branch |
|---|---------|----------|-----------------------|
| 1 | `calculate()` default `properties` varied per backend; **AIMNet2 defaulted to also computing the Hessian** | High (silent waste) | Fixed — all default to `['energy']` |
| 2 | `_radius_graph_no_pbc` / `_one_hot_node_attrs` / `_model_float_dtype` / `_SYMBOL2Z` copy-pasted across the 3 MACE files | High (maintenance) | Fixed — moved to `mace/_common.py` |
| 3 | `_mace_calculator.py` carried **two** builders for the same 20-field dict (`build_data_from_atoms` + `_build_graph_inputs`) | High | Fixed — single `build_data_from_atoms`, used by both `calculate` and `_analytic_hessian` |
| 4 | Identical row-by-row autograd Hessian loop duplicated in ANI/MACE/MACE-omol/MACE-POLAR | Medium | Fixed — shared `hessian_via_double_autograd()` in `calculator_base.py` |
| 5 | MACE family did **two** forwards (energy no-grad, then a second forward for forces) | Medium (perf) | Fixed — single `requires_grad`-gated forward, matching ANI/AIMNet2 |
| 6 | No copy-runnable user template; only an inline snippet in `AUTHORING.md` | Medium (the actual ask) | Fixed — `examples/calculator_plugin/` |
| 7 | Stale `calc.get_energy(...)` examples in `timer.py` docstrings | Low | Fixed — updated to ASE pattern |
| 8 | `implemented_properties` ordering differed in AIMNet2 | Cosmetic | Fixed — uniform order |

**Verification:** ANI-2x, AIMNet2, and MACE (egret) on a water molecule produce
**bit-identical** energy, forces, and Hessian before vs. after the refactor
(checked against an `enhance` worktree). Analytic and numerical Hessians agree to
~1e-4 (ANI ~3e-3, expected for its float32 model).

---

## 3. Still divergent — intentional, documented exceptions

These are **not** bugs; they are flagged so nobody "fixes" them blindly.

- **UMA** does not inherit `CalcABC` (it already extends FAIR-Chem's
  `FAIRChemCalculator`) and **inlines** the eV→Ha + solvent steps that
  `_finalize_results` would otherwise own. Treated as a temporary exception by
  project decision. The cost is a second copy of the unit/solvent logic; if UMA
  is ever brought under the protocol, factor `_finalize_results`' body into a
  free function both can call.
- **`aimnet/_aimnet2_batch_calculator.py`** is deliberately out of the protocol
  (see its module docstring). It is consumed only by `BatchLBFGS`, exposes
  `get_ef_gpu`/`get_efh_gpu` rather than `calculate`, and converts units itself
  (`/EH2EV`). Leave it self-contained until a batch-path retrofit.

---

## 4. The standard a custom calculator must follow

A backend is **any class exposing the surface below** — inheriting `CalcABC` is
recommended but not required (UMA proves duck-typing works).

**Required public surface**
- `calculate(self, atoms=None, properties=['energy'], system_changes=all_changes)`
  — ASE entry point. Must end by calling `self._finalize_results(...)` (if a
  `CalcABC` subclass) or by writing `self.results` with Hartree-unit values.
- The capability class attributes from §1.

**Optional**
- `get_hessian(self, atoms, delta=0.002)` — `CalcABC` provides a default that
  dispatches on `self.hessian` (`'analytic'` → `_analytic_hessian`, `'numerical'`
  → shared finite-difference helper).
- `_analytic_hessian(self, atoms)` — return `(3N, 3N)` ndarray in Hartree/Å². Use
  `hessian_via_double_autograd(energy_fn, leaf)` so the row-by-row loop is not
  re-implemented.
- `get_hvp(self, atoms, n)` — only for the HVP-enabled Dimer path.

**Unit rule (do not violate)**
- Declare `MODEL_ENERGY_UNIT` honestly. `_finalize_results` is the **only** place
  that converts the `calculate()` flow to Hartree. **Never multiply by
  `EV2HARTREE` yourself in `calculate()`.** The Hessian/HVP paths are separate and
  *do* convert inside the backend.

**Loading your own model file (`.pt` / `.jpt` / `.model`)**
- Set `MODEL_PATH_OPTION = 'model_path'` and route it through
  `build_kwargs_from_options(..., resolved_model_path=...)` into your `__init__`.
- Your `__init__` owns the load logic — `torch.jit.load`, `torch.load` +
  reconstruct a custom class, or anything else. MAPLE only hands you a validated,
  existing file path.
- Reach it from input with either:
  - `#model=mymodel(module=my_pkg.my_plugin, model_path=/abs/path/model.jpt)` —
    zero source edits, or
  - drop the file in `maple/function/calculator/model/` for built-in discovery.

See `examples/calculator_plugin/` for two complete, runnable variants (a
TorchScript loader and a `torch.load` + custom-class loader).

---

## 5. Suggested follow-ups (not done here)

- Bring `_forward(atoms, *, requires_grad) -> energy_tensor` in as an explicit,
  documented private contract so `calculate`/`_analytic_hessian` share one forward
  per backend (currently each backend's private forward shape still differs).
- Add a pytest that loads each registered backend on a tiny molecule and asserts
  the ASE protocol + analytic-vs-numerical Hessian tolerance (the manual smoke in
  §2 should become CI).
- Revisit the UMA exception once stress-unit validation lands.
