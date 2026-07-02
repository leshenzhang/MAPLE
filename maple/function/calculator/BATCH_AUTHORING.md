# Authoring MAPLE batch calculators (the `get_ef_gpu` path)

Companion to [`AUTHORING.md`](AUTHORING.md) (single-structure `calculate()` path).
The **batch** path runs B molecules through ONE model forward for GPU throughput.
Base class: [`batch_calculator_base.py`](batch_calculator_base.py) `BatchCalcABC`.

## Contract (what a batch calculator exposes)
One `prepare()` fixes the topology of a batch of B molecules; thereafter the
methods below run batched forwards. Layout: forces `(B, nmax_dof)`, per-structure
Hessian block `(B, nmax_dof, nmax_dof)`, all in **Hartree**.

- `prepare(atoms_list, fixed_nmax=None)` — fix topology + initial coords. `fixed_nmax`
  pins the padded per-structure DOF (multiple of 3, ≥ 3·max_atoms) so a batch
  optimizer shares one layout across iterations.
- `step_cart_(s_cart: (B, nmax_dof))` — in-place displacement (padded per structure).
- `set_coords_(coord: (N,3))` / `backup_coords()` / `restore_coords()`.
- `get_ef_gpu() -> (E (B,), F (B, nmax_dof))` Hartree.
- `get_efh_gpu(movable_masks=None, mode=None) -> (E (B,), F (B, nmax_dof), H (B, nmax_dof, nmax_dof), P (B,))`
  Hartree; `P[i] = Nmax_atoms − n_i` = padding-atom count.
- `hvp(v: (B, nmax_dof)) -> (B, nmax_dof)` Hartree — optional (`HAS_HVP=True`).

Like UMA under `CalcABC`, a backend may satisfy this by attribute presence
without inheriting `BatchCalcABC` — but inheriting is strongly recommended.
Dispatchers must NOT `isinstance`-gate; use `dispatcher/_batch_calc_utils.py`.

## What `BatchCalcABC` gives you (do NOT re-implement)
`prepare` topology (ptr/mol_idx/numbers/`_cols` flat scatter/nmax_dof/`P` + PBC
fail-fast + fixed_nmax validation) · `step_cart_`/`set_coords_`/`backup_coords`/
`restore_coords` · `_pad_forces` · unit conversion (`_to_hartree` per
`MODEL_ENERGY_UNIT`) · `get_ef_gpu` · `get_efh_gpu` mode-dispatch · generic
`_efh_fd` (batched central FD, molecule-parallel) · generic `_efh_analytic`
(seeded double-backward) · `_resolve_movable` · registry.

## What you MUST override (model-specific)
- `_forward(self, coord, need_graph) -> (E (B,), F_all (N,3), leaf|None)` — ONE
  batched forward, in the backend's **native energy unit** (declared by
  `MODEL_ENERGY_UNIT`). `leaf` = the `requires_grad` position tensor when
  `need_graph`, else `None`. **Do NOT convert units here** — the base converts
  once (kills the divide-EH2EV / multiply-EV2HARTREE / none 3-way split).
- `_build_topology(self, atoms_list)` — cache model-specific graph / species /
  AtomicData for the fixed batch (called at end of `prepare`). No-op default if
  the graph is rebuilt inside `_forward`.
- `__init__`: load the model, then `super().__init__(device, dtype)`.

## Capability class attributes (declarative — replace hard-coded class-name sets)
| attr | meaning |
|---|---|
| `MODEL_NAMES` | registry routing keys, lowercase |
| `MODEL_ENERGY_UNIT` | `'eV'` or `'hartree'` — base converts to Hartree once |
| `MODEL_DTYPE` | model compute dtype (forward bridge; master coord stays f64) |
| `SUPPORTS_PBC` | False ⇒ base rejects periodic atoms in `prepare` |
| `SUPPORTED_HESSIAN_MODES` | subset of `('numerical','autograd')` |
| `HAS_HVP` | True ⇒ override `hvp` |
| `SUPPORTS_COUPLING` | True ⇒ perturbing one molecule leaks into others (MACE-POL / charge-eq AIMNet2-native) |
| `BATCHABLE` | False ⇒ potential runs one structure at a time; dispatchers gate B>1 on this |

## Overriding for performance (efficiency is the goal, not just tidy code)
The base `_efh_fd`/`_efh_analytic` are correct + molecule-parallel. A backend that
can pack the ±FD replicas into ONE super-batch (UMA's chunked-plan) or has a
custom cached graph SHOULD **override** the method (not delete it) — keep the
base version as the parity oracle. Never regress throughput: every retrofit
gate = numerical parity (canon-vs-canon) **AND** structures/s + util no-regression.

## Registration + custom potentials
```python
from maple.function.calculator.batch_calculator_base import BatchCalcABC, register_batch_calculator

@register_batch_calculator
class MyBatchCalc(BatchCalcABC):
    MODEL_NAMES = ("mymodel",)
    MODEL_ENERGY_UNIT = "eV"
    def __init__(self, model_path, device="cuda", dtype=None):
        self.model = load(model_path)
        super().__init__(device, dtype or __import__("torch").float64)
    def _forward(self, coord, need_graph):
        ...  # return (E (B,), F_all (N,3), leaf|None)
```
A registered subclass is reachable by name through `make_batch_calc(model, ...)`
and every dispatcher (ts/irc/sp/scan/freq/md) — the batch mirror of the single
`@register_calculator` / `#model=...(module=...)` plug-in path.

## Unit contract (do not violate)
Declare `MODEL_ENERGY_UNIT` honestly. `_forward` returns native units; the base's
`_to_hartree` is the ONLY conversion. Do not multiply by `EV2HARTREE` in `_forward`.

## Parity requirement
Every backend retrofit must reproduce its pre-retrofit `get_ef_gpu`/`get_efh_gpu`
outputs bit-for-bit (canon-vs-canon fp32) on the ts1x test set. The base ships a
`__main__` harmonic self-test proving the plumbing model-independently.
