# MAPLE GPU-batch Optimizations — branch `wave2-on-f75`

Summary of the GPU batched-compute optimization work on top of `feat/gpu-batch` (base `f75c261d`).
All features are **opt-in, default = byte-identical legacy oracle**; every claim is validated on KAUST Ibex A100 (UMA `uma-s-1p1`, ts1x test set / built large systems). Accuracy floor = UMA fp32 GPU run-to-run nondeterminism ~1e-6 eV.

## Architecture (two layers)
- **Calculator = parallel primitive**: `prepare(atoms_list)` + `get_ef_gpu()` -> (E (B,) Ha, F (B,nmax_dof) Ha/Å) — one MLIP forward over B structures; `get_efh_gpu(movable_masks=None)` batched (partial) Hessian; `set_coords_`/`step_cart_` in-place (re-prepare only on batch-membership change). Block-diagonal graph packing -> cross-molecule isolation exact for LOCAL potentials. Contract: `BATCHED_FORWARD_API_CONTRACT.md`.
- **Dispatcher = batched algorithms** over B independent trajectories, one batched forward per iteration.

## Batchable backends (perturb-one isolation gate)
LOCAL (cross-mol batch ✅): **UMA, MACE, MACE-OFF, Egret, ANI, AIMNet2-decoupled**. COUPLING (B=1 only ❌): **AIMNet2-native / AIMNet2-NSE** (charge equilibration, leak 1.2-2.3e-4 Ha/Å), MACE-POL (polarization). 6/8 of the official MAPLE model zoo are cross-mol batchable.

## Commits (categorized)

### Batched calculators
- `caaca13` autograd double-backward Hessian + HVP (UMA & standard-MACE; UMA eSCN double-backward incomplete -> FD default)
- `eb5a1e5` **AIMNet2DecoupledBatchCalc** — cross-reaction-batchable AIMNet2 (wb97m, non-charge-eq)
- `b2ff1ba` partial (movable-subspace) Hessian for AIMNet2/MACE/MACE-POL (parity 4.8e-7)
- `cb1b177` **ANIBatchCalc** — closes the ANI batch gap (4/8 zoo models; B=8 parity 1.5e-8; ANI autograd Hessian works, FD is the noisy path)
- `8b8a6c8` UMA opt-in fast_inference (ac_off forward, 1.61x, parity 1e-8)

### Batched TS / optimization / IRC
- `f13bd4a` **BPRFO** HVP-iterative eigensolver + Lindh/ts_hessian seeds + NW trust (Sella-class)
- `f6ead0f` dimer autograd-HVP min-mode (opt-in, FD default)
- `e38fb84` multi-band batched **NEB** + DyNEB + variable-spring/CI forces
- `34c2a13` batched optimizers **BatchSD/SDCG/DIIS/RFO** + batched 2D PES scan
- `419e911` **forward elimination** (reuse committed-geometry forwards, BPRFO 132->56 ≈2x, byte-exact)
- `49dde62` **NEB streaming pool** + straggler eviction (1000-reaction campaigns, 1/3 memory)
- `2cd16ef` **NEB on-device** inner loop (set_coords_ kills per-iter prepare; device-sync -82%; util 51.6->54%)
- `0dd78db` **dispatcher-wiring** — list/Molecules job reaches the Batch* classes (opt/prfo/dimer/lqa)
- `78d1510` expose VRAM-adaptive auto_batch through the prfo dispatcher

### Hessian / frequency / partial Hessian (PHVA)
- `7b01c2b` **frequency**: FixAtoms -> partial Hessian -> partial (PHVA) frequencies + fixes the unconditional T/R projection bug for constrained systems
- `54ef6de` **cross-backend movable-autograd** partial Hessian (production-solid): MACE-OFF/ANI native autograd Hessian, 19x fewer forwards; end-to-end enzyme freq through the dispatcher (418-atom 1.6s, RRHO thermochem); enzyme-scale feasibility 70-210x vs full Hessian

### Memory / throughput / multi-GPU
- `5d0acfc` **VRAM-adaptive batch sizing** (auto_chunk Hessian fills VRAM 18.6->44.2%, OOM halve-retry; util-/robustness gain — wall flat since FD-Hessian is compute-bound)
- `3697d12` opt-in exp/Lindh preconditioner (note: build-once does not beat plain L-BFGS on UMA at large N — kept opt-in)

### Engine / MD coordination
- `7a08b96` **input-file UMA fix** (uma/__init__.py env-compat shim: add_safe_globals([slice]) + ray.serve stub -> single+batch .inp jobs run end-to-end)
- `6fb8602`+`f071e85` BatchedMD (replica NVE/NVT) then **deprecated** per the cross-phase parallelism contract (MD physics owned by ai-maple-md; this package owns only the batched-forward primitive)

## Headline validated metrics
- batch-throughput util **37->100%**; multi-GPU **1.73x / 2 GPU** (86.7% eff)
- DyNEB pipeline 1.9x + success 90% (vs paper 86-93%), TS-RMSD 0.067 Å
- partial-Hessian **40-210x** at enzyme scale (450/660-atom; the only feasible route — full Hessian OOM/∝N²)
- batched optimizers 3-12x; forward-elimination ~2x; batched MD 4.18x; ANI full pipeline 3.4x faster than UMA (speed/accuracy tradeoff)

## Honest negatives (tried, do NOT realize at small-molecule scale)
CUDA-graph (eSCN capture infeasible + RNG-poison) · TF32 forward (0.94x, launch-bound) · torch.compile (0.90x) · GSM batching (growing-string control-flow diverges) · multi-fidelity macepol cascade (cheap model gives garbage TS) · exp-precon refresh (does not beat plain L-BFGS) · learned Hessian (out of scope) · cuEquivariance (MACE-only; zoo MACE .pt are single-graph traces anyway).

**Core insight**: at MAPLE's small-molecule TS scale the real wins are **batching / throughput / coverage / partial-Hessian / autodiff-on-MACE-ANI**, not single-forward kernel tricks (the UMA forward is 85% of wall and lives inside fairchem).
