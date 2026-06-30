# MAPLE MD capability matrix — {NVE, NVT, NPT} x ML potentials

Rows = MD ensemble, columns = ML-potential family. Regenerate with
`tests/smoke_md_capability.py` (see bottom). This file is the committed
snapshot from the run recorded under *Provenance*.

**Legend** — **PASS**: smoke-tested OK on GPU. *supported*: capability gate
allows it but not run in this snapshot. **N/A**: a capability gate refuses the
cell. **PENDING**: backend parked on another branch. **not-installed** /
**no-model**: backend exists in code but is unusable in this env.

**Column -> MAPLE `#model=` selector** (the six task families map onto the six
shipped MACE/AIMNet/UMA backends):
`aimnet2`->`aimnet2`, `macepol`->`macepols` (MACE-POLAR),
`mace`->`maceomol` (generic local-file MACE), `mace-mp-0`->`mace-mp-0`
(periodic MACE-MP-0 foundation), `mace-off`->`maceoff23s` (MACE-OFF23),
`uma`->`uma`.

| ensemble | aimnet2 | macepol | mace | mace-mp-0 | mace-off | uma |
|---|---|---|---|---|---|---|
| **NVE** | not-installed | **PASS** | no-model | **PASS** | PENDING | N/A |
| **NVT** | not-installed | **PASS** | no-model | **PASS** | PENDING | N/A |
| **NPT** | N/A | N/A | N/A | **PASS** | N/A | N/A |

## Why each cell reads as it does

NPT is the discriminating ensemble: `maple/function/dispatcher/md/ensemble/npt.py`
refuses any calculator that is not `SUPPORTS_PBC` **and** does not return a real
`stress` (no kinetic-only/ideal-gas pressure). Only **mace-mp-0**
(`MACEMPCalculator`, `SUPPORTS_PBC=True`, `implemented_properties` includes
`stress`) clears that gate. Every other shipped backend is a no-PBC molecular
wrapper, so its NPT cell is **N/A** by capability — not by failure.

- **NVE x aimnet2** [not-installed]: `aimnet2`/`aimnet2calc` package absent in env (class registers, instantiation impossible).
- **NVT x aimnet2** [not-installed]: same — backend package not installed.
- **NPT x aimnet2** [N/A]: `AIMNet2Calculator` is `SUPPORTS_PBC=False`, no stress (also not installed).
- **NVE x macepol** [PASS]: water, 300 steps, finite, T~147.7 K (300 K init -> equipartition), |dTE/TE|=1.1e-07, forces nonzero.
- **NVT x macepol** [PASS]: water, 2000 steps, v-rescale, T_eq=301.4 +/- 177.9 K (mean on target; large sigma is the 3-atom system, thermostat controls it).
- **NPT x macepol** [N/A]: MACE-POLAR is `SUPPORTS_PBC=False`, no stress -> NPT gate refuses.
- **NVE x mace** [no-model]: `MACEModelCalculator` is `REQUIRES_LOCAL_MODEL_FILE` and no `maceomol.pt` is provisioned in the model dir.
- **NVT x mace** [no-model]: same — no local model file.
- **NPT x mace** [N/A]: maceomol is `SUPPORTS_PBC=False`, no stress.
- **NVE x mace-mp-0** [PASS]: Cu108 fcc, 300 steps, finite, T~152.0 K, |dTE/TE|=9.58e-07, forces nonzero.
- **NVT x mace-mp-0** [PASS]: Cu108, 1000 steps, v-rescale, T_eq=290.0 +/- 38.8 K (controlled to 300 K target).
- **NPT x mace-mp-0** [PASS]: Cu108 started compressed (a=10.6134), 2000 steps v-rescale + c-rescale @1 bar -> box responds: V 1195.58 -> 1243.33 A^3 (slope +25.5 A^3/ps).
- **NVE/NVT x mace-off** [PENDING]: MACE-OFF (`maceoff23*`) MD support is being added in branch `feat/md-mace-off`; not exercised here. (Class is registered and a `MACE-OFF23_medium.model` exists in `~/.cache/mace/`, but it is out of scope for this branch and has no provisioned `maceoff23*.pt` in the model dir.)
- **NPT x mace-off** [N/A]: MACE-OFF is `SUPPORTS_PBC=False`, no stress (and pending).
- **NVE/NVT/NPT x uma** [N/A]: UMA is vetoed (decision **B-11**, double-blocked) and `fairchem` is not installed (the UMA module hard-imports `fairchem.core`, so it does not even register). Note UMA also rejects `stress`, so it could not satisfy the NPT gate regardless of the veto.

## Cells NOT smoke-tested, and why (no silent gaps)

| cell | status | reason it was not run |
|---|---|---|
| aimnet2 x {NVE,NVT} | not-installed | `aimnet`/`aimnet2calc` package not in env; cannot instantiate |
| aimnet2 x NPT | N/A | no PBC+stress backend |
| mace x {NVE,NVT} | no-model | `maceomol` REQUIRES_LOCAL_MODEL_FILE; no `maceomol.pt` provisioned |
| mace x NPT | N/A | no PBC+stress backend |
| macepol x NPT | N/A | MACE-POLAR has no PBC+stress |
| mace-off x {NVE,NVT} | PENDING | MD wiring lives on `feat/md-mace-off` (not merged here) |
| mace-off x NPT | N/A | no PBC+stress backend (and pending) |
| uma x {NVE,NVT,NPT} | N/A | decision B-11 veto + `fairchem` not installed |

The 5 capability-allowed cells (macepol/mace-mp-0 x supported ensembles) were
all smoke-tested and **PASS**.

## Provenance

- Worktree: `feat/md-capability-matrix` (based on `cc9a479`).
- Env: `/ibex/user/xiaox/zls/ai-maple-md/envs/plumed` — python 3.11, torch 2.6.0+cu124, mace-torch 0.3.16, ase 3.29.0.
- Packages probed: mace = present; aimnet/aimnet2calc = absent; fairchem = absent.
- mace-mp-0 weights: upstream `mace_mp(model='medium')` resolved from `~/.cache/mace/` (2023-12-03 MACE-128-L1, epoch 199).
- GPU job: Slurm `47843514` (`01__mmd_capmatrix`), 1x A100-SXM4-80GB on `gpu201-16-r`, harness wall 358 s.
- Run dir (not in git): `/ibex/user/xiaox/zls/ai-maple-md/runs/capmatrix/` (inputs, thermo, `smoke_results.json`).

## Regenerate

Introspect-only (fast, no GPU — re-derives the grid from the live env):

```
PYTHONPATH=<worktree> PYTHONSAFEPATH=1 \
  python tests/smoke_md_capability.py --workdir <run_dir>
```

Full live smoke (GPU; runs MAPLE MD for the runnable cells, then rebuilds):

```
PYTHONPATH=<worktree> PYTHONSAFEPATH=1 \
  python tests/smoke_md_capability.py --run --workdir <run_dir> \
    --json <run_dir>/smoke_results.json --emit-md <run_dir>/CAPABILITY_MATRIX.generated.md
```

Regression gate (CI-friendly, no GPU): `pytest tests/smoke_md_capability.py`
asserts mace-mp-0 is the only PBC+stress backend, the molecular MACE/AIMNet
backends never enable NPT, and uma/mace-off stay N/A/PENDING.

---

# BATCHED MD PBC capability matrix (Phase-1A: NVT-PBC, fixed cell)

This is the **batched** path (`ensemble/nvt_batched.py`, `ensemble/batched.py`,
`B>1` co-batched calculators), distinct from the single-system table above.
Phase-1A lifts the blanket isolated-only reject (`nvt_batched.py:170-172`,
`batched.py:151-157`) and replaces it with a per-replica capability + box-size
gate (`_setup_pbc_gate`): periodic batches require a `SUPPORTS_PBC` backend and
every periodic replica must pass the GROMACS-style box guard (perpendicular width
>= 2*r_max). Cell is calc-internal state set at `prepare()`; the MD loop and
`get_ef_gpu` return contract are **unchanged** (NVT only — NPT-PBC/stress is
Phase 2C, out of scope).

| batched backend | class | SUPPORTS_PBC | NVT-PBC status |
|---|---|---|---|
| **MACE-OFF** | `MaceOffBatchCalc` | **True** | **PASS** — full double-gate (GPU min-image edge builder; A1/A2/A3/A4/A5 + B1/B2). Molecular FM used periodically = extrapolative; validate observables. |
| **UMA-periodic** | `UMABatchCalc` (task=omat/oc20/...) | **True** | **CODE-COMPLETE, runtime DEFERRED** — cell+pbc preserved, periodic-task gate, per-forward AtomicData rebuild (no stale edges). Blocked from runtime validation: cxtorch `fairchem` import broken (pydantic 2.x `IncEx`) + no periodic UMA checkpoint cached. |
| **std-MACE** | `MACEBatchCalc` | False (unchanged) | **DEFERRED (Phase-1A scope cut)** — needs a periodic block-diagonal `radius_graph_pbc` edge builder (1–1.5 wk) + a traced model that accepts nonzero shifts at B>1. |
| **AIMNet2-decoupled** | `AIMNet2DecoupledBatchCalc` | False | **N/A by design** — gas-phase MLIP; periodic batch rejected by the `SUPPORTS_PBC` gate with a clear message. |
| **MACE-POL** | `MacePolBatchCalc` | False | **N/A by design** — non-periodic; rejected by the gate. |

**Batched algorithms all inherit PBC for free** (verified): `REMD`,
`BatchedGaMD`, `BatchedSMD`, `BatchedUmbrella` subclass `BatchedNVT` and call
`super().__init__`, so they ride the ONE lifted gate + the ONE shared
`get_ef_gpu`. No per-method PBC code. GaMD ran a 60-step periodic trajectory with
exactly one forward/step; REMD/SMD/Umbrella admit periodic construction; a
non-`SUPPORTS_PBC` backend is rejected.

Provenance (batched-PBC): worktree `feat/md-pbc` (base `feat/md-r2fixes`
@ `22a1a9e`), env `envs/plumed` (torch 2.6.0+cu124), MACE-OFF23_medium
(r_max=5.0). See `test_results/md-pbc_validation.md` for gate numbers + job IDs.
