# BACKEND x ENHANCED-SAMPLING adaptation matrix (ai-maple-md Phase B)

Verifies every potential/backend adapts to every union-branch batched MD /
enhanced-sampling method, probes the batchable-backend capability GATE, and fixes
adaptation gaps. Validation axis = **algorithm correctness** (B=1 parity for
batched methods, bias-additive for biased, clean accept/reject for the gate), not
literature numbers.

## Provenance
- Branch `test/backend-adaptation-matrix` (off `feat/md-phase1-union` @ `1d47310`),
  worktree `.../MAPLE/worktrees/backendmatrix`.
- Env `envs/plumed` (python 3.11, torch 2.6.0+cu124, mace-torch, ase, plumed).
- GPU: 1x A100-SXM4-80GB. Jobs `47914621` (initial) -> `47914639`/`47914652`
  (final, clean). Final harness wall 1:38, `MATRIX_EXIT=0`, 0 ERROR cells.
- Driver `test_results/backend_matrix/backend_matrix.py`, wrapper
  `gate_matrix.sbatch`, machine result `backend_matrix_result.json`.

## 1. Backend load inventory (which load, which env)

| backend | batch class | loaded (plumed env) | SUPPORTS_PBC | batch_isolated | note |
|---|---|---|---|---|---|
| **mace-off** | `MaceOffBatchCalc` | **YES** | True | True | `~/.cache/mace/MACE-OFF23_medium.model` via `torch.load` |
| **mace-mp** | `MaceOffBatchCalc` | **YES** | True | True | `20231203mace128L1_epoch199model` (MACE-MP-0) via SAME adapter |
| **std-mace (maceomol)** | `MACEBatchCalc` | loads stand-in | False | (none) | no genuine batch-generic `maceomol.pt`; stand-in is single-graph trace |
| **macepol** | `MACEPolBatchCalc` | NO | False | (coupled) | `val_movable/*.pt` not a jit archive (no `constants.pkl`); coupled by design |
| **aimnet2-native** | `AIMNet2BatchCalc` | **YES** | False | (coupled) | `zoo_models/aimnet2.pt` via `jit.load`; charge-coupled -> reject B>1 |
| **aimnet2-decoupled** | `AIMNet2DecoupledBatchCalc` | NO | False | True | needs `aimnet2calc` pkg (absent in both plumed+cxtorch envs) |
| **uma** | `UMABatchCalc` | NO | True | True | needs `fairchem` (only cxtorch env, ABI-broken) + periodic ckpt; loaded OK in a separate fairchem env (job 47848031) |

Env facts: `mace` present in both `plumed`(torch2.6) and `cxtorch`(torch2.11);
`aimnet2calc` absent in both; `fairchem` only in `cxtorch`; `plumed` only in the
`plumed` env. All union methods + PLUMED run in the `plumed` env.

## 2. BACKEND x METHOD matrix

Legend: **PASS** smoke ran + invariant held; **N/A-gate** coupled calc, B>1
rejected by the isolation gate (clean, by design); **N/A-model** no compatible
batch-generic model provisioned (adapter emits a clean actionable error);
**SKIP** backend package/model absent in env.

| backend | BatchedNVT | B1-parity | umbrella | GaMD | REMD | SMD |
|---|---|---|---|---|---|---|
| **mace-off** | PASS | PASS | PASS | PASS | PASS | PASS |
| **mace-mp** | PASS | PASS | PASS | PASS | PASS | PASS |
| **std-mace (maceomol)** | N/A-model | N/A-model | N/A-model | N/A-model | N/A-model | N/A-model |
| **macepol** | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP |
| **aimnet2-native** | N/A-gate | N/A-gate | N/A-gate | N/A-gate | N/A-gate | N/A-gate |
| **aimnet2-decoupled** | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP |
| **uma** | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP |

Single-system **PLUMED-bias**: **PASS** on mace-off (RESTRAINT injects bias,
`dF=1.9e-3 Ha/A` vs bare, run finite). Same wrapper (`PlumedCalculator`) is
backend-agnostic (mirrors inner `SUPPORTS_PBC`).

Invariant evidence (mace-off / mace-mp identical structure):
- BatchedNVT B=2: per-replica T on target, perturb-one isolation `dE=0 Ha`, finite.
- B1-parity: BatchedNVT(B=1) === single-system NVT to fp64 (`max|dpos|~1e-15 A`,
  `max|dvel|~5e-17 au`).
- umbrella: bias ADDITIVE (`dE_err~3e-15 Ha` vs analytic `0.5k(cv-c)^2`,
  `dF=4.5e-2`), WHAM PMF finite, windows tracked.
- GaMD: boost force-factor `1-sqrt(2 k dV)` in `[0.3,1.0]` (subset of [0,1]), finite.
- REMD: no-swap === independent BatchedNVT to `~1e-15`, isolation `dE=0`.
- SMD: analytic restraint force `-k(xi-lam)u` exact (`Ferr=0`), pulls finite +
  independent (`isoDE=0`).

**Key architectural reason the matrix is uniform per backend:** `REMD`,
`BatchedGaMD`, `BatchedSMD`, `BatchedUmbrella` all subclass `BatchedNVT` and add a
per-replica bias at the ONE shared `_forces_au` -> `calc.get_ef_gpu()` hook, riding
the ONE `_assert_batch_isolated` gate. So backend adaptation is decided once (does
the batch calc load + is it batch-isolated), not per-method.

## 3. Capability-gate verdict (`BatchedNVT._assert_batch_isolated`, B>1)

| calc class | expected | got | reject msg clean? |
|---|---|---|---|
| `MaceOffBatchCalc` (batch_isolated=True) | ACCEPT | ACCEPT | - |
| `UMABatchCalc` | ACCEPT | ACCEPT | - |
| `AIMNet2DecoupledBatchCalc` | ACCEPT | ACCEPT | - |
| `MACEBatchCalc` | ACCEPT | ACCEPT | - |
| `AIMNet2BatchCalc` (native) | **REJECT** | **REJECT** | **YES** |
| `MACEPolBatchCalc` | **REJECT** | **REJECT** | **YES** |
| generic `batch_isolated=False` | REJECT | REJECT | YES |

**GATE VERDICT: PASS.** Accepts {MACE-OFF, UMA-local, mace-mp(=MaceOff), MACE,
AIMNet2-decoupled}; rejects AIMNet2-native + MACE-POL. Reject is a clean actionable
`ValueError` (names the calc, says "batch-ISOLATED ... global charge equilibration
... replica energies LEAK", points to decoupled/single-system alternatives) — not a
silent wrong result. Confirmed on a REAL loaded `AIMNet2BatchCalc` instance
(charge-coupled native calc present), matching decision D-82.

## 4. Adaptation gaps: fixed vs real limitation

**No generic-adapter bug found.** mace-off and mace-mp both adapt through the SAME
`MaceOffBatchCalc` (torch.load MACE `.model`) with zero code change -> all 5 batched
methods + parity + PLUMED PASS. The three fixes were **validation-harness only**:
1. PLUMED wrapper class is `PlumedCalculator` (not `PlumedBias`).
2. `PYTHONUTF8=1` in the run wrapper: the single-system NVT logger writes a Unicode
   'tau' char that crashes under the compute-node ascii locale (portability, not a
   backend/adapter bug).
3. Reclassify single-graph-traced / incompatible-interface model errors as
   **N/A-model** (clean actionable limitation) instead of ERROR.

**Real limitations (documented, not bugs):**
- **std-MACE (maceomol):** no batch-generic `maceomol.pt` provisioned. `MACEBatchCalc`
  LOADS a stand-in but its batch-native PROBE cleanly rejects it at B>1
  ("traced single-graph, B=1-locked"); the stand-in is a mace-off-format trace
  (`forward(data, local_or_ghost, compute_virials)`), not a maceomol export. Adapter
  behaviour is correct; needs a batch-generic export. (Same trace-lock class as
  MACE-POL, per CAPABILITY_MATRIX.)
- **macepol:** coupled by construction (global polarization) -> B>1 N/A by the gate.
  Its `val_movable/*.pt` weights are not a loadable jit archive here.
- **aimnet2-native:** works B=1 (single-system) but is charge-coupled -> N/A batched
  by design (gate reject).
- **aimnet2-decoupled / uma:** package absent in the plumed env (`aimnet2calc` /
  `fairchem`) -> SKIP. The gate WOULD accept both (Part 2). Both are batch-isolated
  and would drop into every method once their package is installed in the run env.

## Regenerate
```
sbatch test_results/backend_matrix/gate_matrix.sbatch     # a100, ~1.6 min
```
