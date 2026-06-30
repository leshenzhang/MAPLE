# MAPLE GPU-batch — Session 进展存档（2026-06-29）

> 课题 `ai-maple-gpu`（ClickFF/MAPLE 纯 MLIP TS/OPT/IRC/Hessian 库 GPU batch 优化，phase A）
> branch `github.com/leshenzhang/MAPLE:wave2-on-f75` HEAD **d553fde**（本 session push 12 贡献 + 验证）
> 决策日志 `DECISION_LOG.json` D-113~D-153 · 验证报告 `reports/VALIDATION_MATRIX_2026-06-29.{zh,en}.md` · 论文 `reports/lit/UniTS_chemrxiv_10001667.pdf`

## TL;DR
1. **本 session push 12 项贡献到 wave2-on-f75**（commit 链 334c92d→d553fde），全部 opt-in 默认 oracle 或 byte-identical，验证 GREEN，0 真 ISSUE。
2. **★Profiling 实测定论**：MLIP forward 占 wall **95-99%**；所有「非-forward compute 优化」候选（torch-2.12-eigh/block-LOBPCG/native-scatter/host-sync）经测全是**幽灵**（eigh 仅 0.31%）→ compute-micro-optimization 已到 measured frontier。
3. **★战略重定向（数据驱动）**：优化 MAPLE = **减少 forward 数量**，不是加速非-forward 的 5%。geodesic 前端已证：好初猜省 **65.8% forward**。
4. **两个 training-free 新算法落地**：测地线 TS 初猜前端 + iso-ARTn 单端鞍点搜索。
5. **3 轮 SOTA 调研**（saddle/path · ML-generative · GPU 加速）+ **Round 1-5 对抗审计**确认代码前沿。
6. **在途**：UniTS-Gen（TS 扩散生成模型）融入 MAPLE（env 构建）+ 有序统一验证。

---

## 1. 本 session push 的 12 项贡献

| # | commit | 贡献 | 性质 | 关键指标 | 验证 |
|---|---|---|---|---|---|
| 1 | 334c92d | 8 项 gpu-batch 优化(BatchDimer shrink/BatchRFO nmax/BPRFO conv-table+H_mw.clone/LQA every-64/UMA topology device-resident/forward-diff Hessian opt-in/UMACalculator lazy-load/F2 allocator) | opt-in 默认 oracle / byte-identical | dimer shrink −18.8% atom-fwd · forward-diff Hessian −2× · lazy −15.4s/−1.27GB | 大规模平行验证 8 validator GREEN |
| 2 | 5c6422b | 验证日志归档(双语矩阵+34 .out+决策切片) | 文档 | — | — |
| 3 | d8687de | **BatchRFO 减重复计算**(移植 BPRFO FIX#2 + base_ef threading) | 默认 ON+oracle gate | **forward −24.7%**(parity-exact RMSD1.6e-6) | Round-5 审计 airtight |
| 4 | 63b817b | **GSBatch 默认 IRC 批量化**(Gonzalez-Schlegel, 前只批 LQA 非默认) | opt-in 单 GS oracle | **9.6×**(TS 能量 9.5e-7 noise floor) | parity-faithful, control 证 drift=fp32 |
| 5 | a59d559 | **测地线 TS 初猜前端**(MLIP 面 geodesic+FIRE+爬节点) | opt-in 新能力 training-free | **下游 8/8 有效 TS vs crude 5/8 · forward −65.8% · RMSD 0.295<IDPP 0.363** | GPU 验证 8 ts1x |
| 6 | d553fde | **iso-ARTn 单端鞍点搜索**(subclass BatchDimer+activation 协议) | opt-in 新能力 training-free | **2/2 有效一阶鞍点 n_imag=1 · 3.47× batched**(单次低产需 multi-seed) | GPU 验证, 诚实 caveat |

## 2. 验证

### 大规模平行验证（D-122~133，8 validator + cleanup，~30 优化，双账号）
**全 GREEN，0 真 ISSUE**。每 opt 三查：正确性（batched==single oracle ≤噪声底）+ 正向（benefit≥1）+ 默认安全。
- **batch 并行权威**：raw forward `get_ef_gpu` **10.18×@B64**，GPU util 19%→**99%@B128**，parity ≤7e-7 eV。
- V4 NEB 软-HEI 发散 / GS path-drift = **calculator fp32-Hessian gap，非 batching bug**（control 证：同代码无 batching 两跑同等发散）。
- 报告 `reports/VALIDATION_MATRIX_2026-06-29.{zh,en}.md` + 34 原始 .out（GitHub `validation/2026-06-29/`）。

### 新增贡献验证
- **BatchRFO 减重复**：reuse vs oracle parity RMSD 1.63e-6/dE 1.31e-8，forward 77→58（−24.7%）。Round-5 审计确认 base_ef invalidation airtight（无 stale 复用路径）。
- **GSBatch**：batched vs single GS 逐行 + TS 能量 9.5e-7、image 数 7/7 全 6 精确；9.6×。
- **测地线**：guess RMSD 0.295Å（<IDPP 0.363），下游 P-RFO 8/8 有效 TS（crude 仅 5/8 落错指数驻点），forward −65.8%。
- **iso-ARTn**：2/2 收敛为有效一阶鞍点（n_imag=1），3.47× batched；诚实：单次 2/8 收敛（单端单方向固有，需 multi-seed restarts）。
- **有序统一验证**（在途）：full-branch 树（全 12 贡献）import smoke PASS（组合无冲突）；逐项 parity + 组合端到端验证 GPU 排队中。

## 3. ★ Profiling — 本 session 最有价值的发现（job 47885420，torch 2.11 A100 实测）

| 组件 | BPRFO | Frequency | GSBatch IRC |
|---|---|---|---|
| **MLIP forward** | **95.4%** | **98.8%** | **97.3%** |
| μ-bisection(More-Sorensen) | 3.15% | — | — |
| eigh(dense linalg.eigh) | 0.31% | 0.10% | 0.69% |
| FD-Hessian assembly scatter | 0.60% | 2.25% | 0.38% |
| host-sync | 0.20% | ~0 | 1.23% |

**结论**：forward 占 95-99%（比 85% headline 还高——全 TS pipeline 的 FD Hessian 本身 ~6N forwards）；非-forward 全部 <5%。
- **所有 compute-micro-optimization 候选 = 幽灵**：torch-2.12-eigh（eigh 0.3%，100× 只换 ~0.5%，不值 torch 升级破 fairchem 风险）/ block-LOBPCG（n32 cliff 真实 20× 但坐 <1% slice）/ native-scatter（≤2.25%）/ host-sync（<1.3%）。
- **唯一 >eigh 的非-forward** = μ-bisection 3.15%，但仍 3% + 高 parity 风险。
- → **不做这些 busywork**。真优化 = **减少 forward 数**（geodesic −65.8% 砍的是 95% 瓶颈本身）。

## 4. 两个 training-free 新算法（对齐「减 forward 数」战略）

- **测地线 TS 初猜前端**（`ts/algorithm/geodesic.py`，arXiv2507.17968 + Zhu/Martinez JCP2019）：Morse 缩放原子间距度量里测地线插值（纯几何无 forward）→ MLIP 面 FIRE 弛豫 + 爬节点 → TS 猜测喂批量 P-RFO。好初猜避免 P-RFO 落错鞍点（8/8 vs 5/8）+ 省 65.8% forward。与 UniTS-Gen 互补（物理猜 vs ML 扩散猜）。
- **iso-ARTn 单端鞍点**（`ts/algorithm/iso_artn.py`，JCTC2024）：只需反应物（无产物/无近鞍点猜测）= MAPLE 此前没有的单端能力。复用 BatchDimer 的 HVP/min-mode。诚实：单次低产，是 discovery 能力（非 forward-reduction 优化，multi-seed 会增 forward 逆战略）。

## 5. SOTA 调研 roadmap（D-144~147，3 agent web research）

- **training-free 可直接用**（已做）：测地线前端 ✅ · iso-ARTn ✅。
- **需训练（用户排除）**：LMHE 最左本征向量头 / Shoot-from-HIP 直接 Hessian 头（修 UMA 双反向坏）/ HORM 权重 / 生成前端 React-OT/flow-matching / 选择性 BF16。
- **实测幽灵（profiling 否决）**：torch-2.12-eigh / block-LOBPCG / native-scatter / host-sync。
- **仍拒**：cuEq-UMA（非 e3nn）/ blanket TF32 / torch.compile/CUDA-graph eSCN（launch-bound 不变）。

## 6. 在途
- **UniTS-Gen 融入 MAPLE**：env 构建中（torch2.4/2.8+PyG+RDKit+OpenBabel+MolOP+QCBot，与 cxtorch 冲突→独立 env）。集成 = MAPLE 原生 dispatcher 编排「UniTS-Gen 生成 N 候选 → 批量 P-RFO 精修 → freq 验证」。论文 Methods 全在（D-140）。
- **有序统一验证**：full-branch 全 12 贡献的组合矩阵（GPU 排队）。

## 7. Round 1-5 对抗审计结论
代码已到**优化前沿**：forward-elimination + 缓存 + redundancy 全消除；新代码（BatchRFO/GSBatch/geodesic/iso-ARTn）correct + 在 frontier。剩余仅 marginal micro-sync。未来价值在**新算法（减 forward 数）+ UniTS 集成**，非代码再榨。

---

*生成 2026-06-29 | 课题根 `catgo-projects/data/ai-maple-gpu/` | branch wave2-on-f75 @ d553fde | DECISION_LOG D-113~153*
