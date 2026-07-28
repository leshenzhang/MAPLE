# A2-phess test harness

| file | what |
|---|---|
| `phess_accuracy.py` | 5 probes vs the EXACT central-FD Hessian: `fd` (forward vs central), `update` (Bofill/PSB/SR1/BFGS fidelity along a real trajectory + secant-residual calibration), `lanczos` (Lanczos vs LOBPCG leftmost eigenpair, cold vs warm), `core` (core-region radius scan), `synth` (solver-only unit check with exact matvecs) |
| `phess_prod.py` | one N=100 end-to-end P-RFO config: wall + gradient-equivalents + exact-Hessian same-saddle metrics |
| `analyze_phess.py` | merges run JSONs into cost / same-saddle tables; REJECTS artifacts that measured nothing |
| `verify_fd_reference.py` | proves the shipped "numerical" Hessian is a real finite difference (hand-rolled FD reference + `_forward` vs `_predict_forces` instrumentation) |
| `acc_full.sbatch` | all 5 accuracy probes (~8.5 min on 1 A100) |
| `synth.sbatch` | solver unit check, CPU-only |
| `regress.sbatch` | default-path no-regression gate |
| `verify_fd.sbatch` | FD-reference verification |
| `run_group.sbatch` + `submit_groups.sh` | N=100 campaign, several configs per GPU allocation |
| `run_cfg.sbatch` | one config per job (superseded by run_group) |

Rules baked in: every step has an artifact gate (a probe that produced 0 rows
exits 2; a run with ge_total==0 / no geometry / 0 saddle cases exits 2), because
`rc == 0` and `State=COMPLETED` are not evidence that anything ran.
