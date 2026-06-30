# MAPLE GPU-batch — Session Progress (2026-06-29)

> Project `ai-maple-gpu` (GPU batch optimization of the ClickFF/MAPLE pure-MLIP TS/OPT/IRC/Hessian library, phase A)
> branch `github.com/leshenzhang/MAPLE:wave2-on-f75` HEAD **d553fde** (12 contributions + validation pushed this session)
> Decision log `DECISION_LOG.json` D-113~D-153 · validation `reports/VALIDATION_MATRIX_2026-06-29.{zh,en}.md` · paper `reports/lit/UniTS_chemrxiv_10001667.pdf`

## TL;DR
1. **12 contributions pushed to wave2-on-f75 this session** (commit chain 334c92d→d553fde), all opt-in default-oracle or byte-identical, validated GREEN, 0 real ISSUEs.
2. **★Profiling measured verdict**: the MLIP forward is **95-99%** of wall; every "non-forward compute optimization" candidate (torch-2.12-eigh / block-LOBPCG / native-scatter / host-sync) is, when measured, a **ghost** (eigh is only 0.31%) → compute-micro-optimization has reached a measured frontier.
3. **★Strategy redirect (data-driven)**: optimizing MAPLE = **reduce the NUMBER of forwards**, not speed up the non-forward 5%. The geodesic front-end already proved it: a better guess saves **65.8% of forwards**.
4. **Two training-free new algorithms landed**: geodesic TS-guess front-end + iso-ARTn single-ended saddle search.
5. **3 SOTA literature surveys** (saddle/path · ML-generative · GPU acceleration) + **Round 1-5 adversarial audits** confirm the code is at the frontier.
6. **In flight**: UniTS-Gen (TS diffusion generative model) integration into MAPLE (env building) + ordered unified validation.

---

## 1. The 12 contributions pushed this session

| # | commit | contribution | nature | key metric | validation |
|---|---|---|---|---|---|
| 1 | 334c92d | 8 gpu-batch opts (BatchDimer shrink / BatchRFO nmax / BPRFO conv-table+H_mw.clone / LQA every-64 / UMA topology device-resident / forward-diff Hessian opt-in / UMACalculator lazy-load / F2 allocator) | opt-in default-oracle / byte-identical | dimer shrink −18.8% atom-fwd · forward-diff Hessian −2× · lazy −15.4s/−1.27GB | large-scale 8-validator GREEN |
| 2 | 5c6422b | validation logs archive (bilingual matrix + 34 .out + decision slice) | docs | — | — |
| 3 | d8687de | **BatchRFO redundant-forward elimination** (port BPRFO FIX#2 + base_ef threading) | default-on + oracle gate | **forward −24.7%** (parity-exact RMSD 1.6e-6) | Round-5 audit airtight |
| 4 | 63b817b | **GSBatch — batched DEFAULT IRC** (Gonzalez-Schlegel; only LQA, a non-default, was batched before) | opt-in, single GS oracle | **9.6×** (TS energy 9.5e-7 noise floor) | parity-faithful, control proves drift=fp32 |
| 5 | a59d559 | **geodesic TS-guess front-end** (geodesic-on-MLIP + FIRE + climb) | opt-in new capability, training-free | **downstream 8/8 valid TS vs crude 5/8 · forward −65.8% · RMSD 0.295<IDPP 0.363** | GPU validation 8 ts1x |
| 6 | d553fde | **iso-ARTn single-ended saddle search** (subclass BatchDimer + activation protocol) | opt-in new capability, training-free | **2/2 valid first-order saddles n_imag=1 · 3.47× batched** (low single-shot yield, needs multi-seed) | GPU validation, honest caveat |

## 2. Validation

### Large-scale parallel validation (D-122~133; 8 validators + cleanup; ~30 optimizations; dual-account)
**All GREEN, 0 real ISSUEs.** Each opt: correctness (batched==single oracle ≤ noise) + positive (benefit≥1) + default-safe.
- **batch-parallelism authoritative**: raw forward `get_ef_gpu` **10.18×@B64**, GPU util 19%→**99%@B128**, parity ≤7e-7 eV.
- V4 NEB soft-HEI divergence / GS path-drift = the **single-vs-batch calculator fp32-Hessian gap, not a batching bug** (control: identical code with NO batching, run twice, diverges the same amount).
- Report `reports/VALIDATION_MATRIX_2026-06-29.{zh,en}.md` + 34 raw .out (GitHub `validation/2026-06-29/`).

### New contributions validated
- **BatchRFO reuse**: reuse vs oracle parity RMSD 1.63e-6 / dE 1.31e-8; forward 77→58 (−24.7%). Round-5 audit confirmed the base_ef invalidation is airtight (no stale-reuse path).
- **GSBatch**: batched vs single GS line-by-line + TS energy 9.5e-7, image counts 7/7 exact on all 6; 9.6×.
- **geodesic**: guess RMSD 0.295 Å (< IDPP 0.363), downstream P-RFO 8/8 valid TS (crude only 5/8, fell into wrong-index stationary points), forward −65.8%.
- **iso-ARTn**: 2/2 converged are valid first-order saddles (n_imag=1), 3.47× batched; honest: 2/8 converged single-shot (single-ended/one-direction property, needs multi-seed restarts).
- **Ordered unified validation** (in flight): full-branch tree (all 12) import smoke PASS (composes without conflict); per-item parity + composition end-to-end on GPU queued.

## 3. ★ Profiling — the session's most valuable finding (job 47885420, torch 2.11 A100, measured)

| component | BPRFO | Frequency | GSBatch IRC |
|---|---|---|---|
| **MLIP forward** | **95.4%** | **98.8%** | **97.3%** |
| μ-bisection (More-Sorensen) | 3.15% | — | — |
| eigh (dense linalg.eigh) | 0.31% | 0.10% | 0.69% |
| FD-Hessian assembly scatter | 0.60% | 2.25% | 0.38% |
| host-sync | 0.20% | ~0 | 1.23% |

**Conclusion**: forward is 95-99% (higher than the 85% headline — the full TS pipeline's FD Hessian is ~6N forwards); everything non-forward sums to <5%.
- **All compute-micro-optimization candidates = ghosts**: torch-2.12-eigh (eigh 0.3%, 100× saves ~0.5%, not worth the torch upgrade + fairchem risk) / block-LOBPCG (n32 cliff is real, 20×, but on a <1% slice) / native-scatter (≤2.25%) / host-sync (<1.3%).
- **The only >eigh non-forward slice** = μ-bisection 3.15%, still 3% + high parity risk.
- → **Don't do this busywork.** The real optimization = **reduce the forward count** (geodesic −65.8% cuts the 95% bottleneck itself).

## 4. Two training-free new algorithms (aligned with the "reduce forward count" strategy)

- **geodesic TS-guess front-end** (`ts/algorithm/geodesic.py`; arXiv 2507.17968 + Zhu/Martinez JCP 2019): geodesic interpolation in a Morse-scaled interatomic-distance metric (pure geometry, no forward) → FIRE relax on the MLIP + climb → TS guess feeding batched P-RFO. A better guess avoids P-RFO falling into wrong-index saddles (8/8 vs 5/8) + saves 65.8% forwards. Complements the UniTS-Gen front-end (physics guess vs ML diffusion guess).
- **iso-ARTn single-ended saddle search** (`ts/algorithm/iso_artn.py`; JCTC 2024): reactant-only (no product / no near-saddle guess) = a single-ended mode MAPLE lacked. Reuses BatchDimer's batched HVP/min-mode. Honest: low single-shot yield, a discovery capability (NOT a forward-reduction optimization — multi-seed would increase forwards, against the strategy).

## 5. SOTA survey roadmap (D-144~147, 3 web-research agents)

- **training-free, directly usable** (done): geodesic front-end ✅ · iso-ARTn ✅.
- **needs training (user excluded)**: LMHE leftmost-eigenvector head / Shoot-from-HIP direct Hessian head (fix UMA broken double-backward) / HORM weights / generative front-ends React-OT/flow-matching / selective BF16.
- **measured ghosts (profiling-rejected)**: torch-2.12-eigh / block-LOBPCG / native-scatter / host-sync.
- **stays rejected**: cuEq-UMA (non-e3nn) / blanket TF32 / torch.compile/CUDA-graph eSCN (launch-bound unchanged).

## 6. In flight
- **UniTS-Gen integration into MAPLE**: env building (torch 2.4/2.8 + PyG + RDKit + OpenBabel + MolOP + QCBot; conflicts with cxtorch → separate env). Integration = a native MAPLE dispatcher orchestrating "UniTS-Gen generates N candidates → batched P-RFO refine → freq validate". Paper Methods captured (D-140).
- **Ordered unified validation**: the composition matrix over all 12 contributions on the full-branch tree (GPU queued).

## 7. Round 1-5 adversarial audit verdict
The code is at the **optimization frontier**: forward-elimination + caching + redundancy all eliminated; the new code (BatchRFO/GSBatch/geodesic/iso-ARTn) is correct + at the frontier. Only marginal micro-syncs remain. Future value is in **new algorithms (reduce forward count) + the UniTS integration**, not in further code-squeezing.

---

*Generated 2026-06-29 | project root `catgo-projects/data/ai-maple-gpu/` | branch wave2-on-f75 @ d553fde | DECISION_LOG D-113~153*
