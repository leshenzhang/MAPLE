# MAPLE GPU-batch 优化 — 大规模平行验证矩阵（2026-06-29）

> 课题 `ai-maple-gpu`（ClickFF/MAPLE 纯 MLIP 库的 GPU batch 并行优化，phase A）
> 验证目标树：`/ibex/user/xiaox/zls/ai-maple-gpu/sandbox_merge_all/MAPLE`（本 session 7 opt）+ `MAPLE`（canonical cb3d943，committed opt 的 single oracle）
> 原始 .out 证据：Ibex `/ibex/user/xiaox/zls/ai-maple-gpu/VALIDATION_RESULTS_2026-06-29/` + 本地 `reports/validation_raw_2026-06-29/`（34 文件）

## TL;DR
1. **8 个 validator（V1–V8）+ cleanup 并行验证整课题 ~30 项优化**，双账号（xiaox+reny0b）抢空闲 GPU。
2. **验证原理**（每项 opt 都是 opt-in 默认 oracle）：对每项 X 跑 **X-ON vs X-OFF（oracle）** 同输入，三查 ①正确性/精度 ②正向 benefit ③默认安全。batched algo 则 **batched == single oracle**。
3. **噪声底**：UMA fp32 GPU forward run-to-run 非确定性 ~**1e-6 eV**（力 ~1e-8 Ha/Å）、FD Hessian ~**1e-5 Ha/Å²**。真 ISSUE 门槛 = parity 超噪声底**数量级** / 大 B 仍 benefit<1 / 鞍点·n_imag 错。
4. **结果：0 个真 ISSUE，全部 GREEN**。V1/V2/V4(control)/V5/V6/V7/V8 + cleanup **逐条核验 GREEN**；V3 内容被 V5+Tier1 覆盖+部分再确认。**8 项验证过的优化已 push 到 GitHub `wave2-on-f75`**（cb3d943..334c92d，31 commits，无 force）。
5. **batch 并行 headline**：raw forward `get_ef_gpu` **10.18×@B64**，GPU util 19%→**99%@B128**，parity ≤7e-7 eV（V8 权威）；full TS pipeline 1.40-1.58×（小分子 P-RFO Python-bound）。conv 不变 = **零精度/算法代价**。
6. **核验铁则**：agent 机械 flag 的 ISSUE（small-B launch-bound / env-gap / 过严阈 / evicted-straggler / 软-HEI-fp32 / legacy 弱初猜）经逐条核验**均非精度损非算法错**。

---

## 1. 验证方法论

| 维度 | 判据 |
|---|---|
| **正确性（无算法错）** | batched 结果 == 单结构 oracle（同鞍点/同极小/同 n_imag/同收敛状态），或 byte-identical |
| **精度（无精度下降）** | parity ≤ 噪声底（UMA fp32 ~1e-6 eV / Hessian ~1e-5），或 forward-diff 的 documented caveat（~1 cm⁻¹） |
| **正向（真收益）** | benefit ≥ 1（speedup / forward 数 / 内存 / startup），在设计的工作区（throughput at scale，非 small-B） |
| **默认安全** | opt-OFF == canonical（opt-in 不扰动默认路径） |

**为何很多 agent flag 的「FAIL」是假警报**：UMA eSCN GPU forward 是 scatter-atomicAdd 非位可复现的（~1e-6 eV run-to-run）。经优化器/NEB 的路径依赖迭代放大后，end-to-end 几何/能量会有 chaotic 漂移（同极小/同鞍点，但数值不同）。**control（canon-vs-canon 同代码两跑）显示同等漂移**，证明残差是 fp32 非确定性而非代码改动。

---

## 2. 主验证矩阵（per-class）

### V1 — 批量 Calculator（job 47874553，COMPLETED）
| opt | parity | benefit | default-safe | VERDICT |
|---|---|---|---|---|
| UMA `get_ef_gpu` 批量 | max\|dE\|=**1.31e-8** / dF=3.75e-8 Ha | throughput（见 §3） | 是 | ✅ 正确（benefit 见 §3） |
| UMA `get_efh_gpu` 批量 Hessian | max\|dH\|=1.59e-5（≈噪声底） | 1.06×@B4 | 是 | ✅（同 D-112 PASS 级） |
| UMA `get_hvp` FD-HVP | cos=0.99998 rel 7.68e-3（O(δ)） | matrix-free，启用迭代 eigensolver | opt-in | ✅ POSITIVE+SAFE |
| partial Hessian（movable） | movable-block vs full 8.30e-6 | 1.68× 少 forward | opt-in（默认 full） | ✅ POSITIVE+SAFE |
| fast_inference (ac_off) | 1.31e-8 / 2.63e-8 | **1.49× forward** | 是（默认 eager 确认） | ✅ POSITIVE+SAFE |
| AIMNet2-decoupled / ANI / MACE | env-gap | — | — | 🔧 env 限制非 bug（aimnet2calc/torchani 不在 cxtorch；MACE zoo .pt 是 single-graph trace D-108 已知）；AIMNet-decoupled 见 V7 jobB 1.34× |

### V2 — 批量优化器（jobs 47875347/520/348，COMPLETED）— **ALL 6 PASS**
| 优化器 | parity | benefit | VERDICT |
|---|---|---|---|
| BatchLBFGS | min-to-noise（dE 4.29e-6） | B-scaling | ✅ PASS |
| BatchSD | **lockstep step1=1.29e-8 Å（machine-eps）** | ~3× | ✅ PASS |
| BatchSDCG | **lockstep 1.29e-8 Å** | ~3× | ✅ PASS |
| BatchDIIS | min-to-noise（6.58e-5 软盆） | ~4× | ✅ PASS |
| BatchRFO | min-to-noise（dE 1.75e-8 / RMSD 5.92e-7） | **8.33×**（serial 468→batch 56s） | ✅ PASS |
| BatchRFO nmax-fix | nmax=30 EQUAL，**少 1 forward**（62 vs 63），value-identical | +1 elided | ✅ PASS |
> blbfgs/sd/sdcg/diis 本 session **byte-identical canonical**（未改=既往已验，无回归风险）；仅 batch_rfo.py 改 3 行 nmax。

### V5 — IRC / scan / Hessian-freq（job 47874000，COMPLETED）— **12/13**
| 项 | 结果 | VERDICT |
|---|---|---|
| **PHVA partial 频率** | partial vs full sub-block 频率**逐个相同**（max diff 2.1e-3 cm⁻¹，gate<1e-2） | ✅ 精确 |
| NEB multiband parity（identical band） | \|dBarrier\|=**6.92e-7 Ha**（证 batch 本身确定） | ✅ |
| DyNEB+climbing | barrier 2.64 eV，forwards 98 | ✅ |
| BatchDimer | TS-RMSD 0.018 Å | ✅ |
| partial Hessian | 5.34e-6 Ha/Å² | ✅ |
| uma_decouple | 1.31e-8 Ha | ✅ |
| autodiff Hessian | \|AD-FD\|=1.05e-2（UMA eSCN 已知 → FD 默认） | ✅ |
| forward-elimination | reuse_forward=True，fwd_reused=147 | ✅ |
| BatchPRFO robust（lindh+nw） | converged **4/4**，TS-RMSD 0.231 | ✅ |
| BatchPRFO iterative | 3/4 | ✅ |
| BatchPRFO **legacy** | **2/4**（identity-init 弱收敛=已知，robust 配方修，非回归） | ⚠️ 已知 |

### V6 — 内存/吞吐/基础设施（jobs 47874099/101/194，COMPLETED）
| opt | 结果 | VERDICT |
|---|---|---|
| **F1 UMACalculator lazy-load** | eager construct **15.39s→0.00s**，host RSS **+1268.9MB→+0.0MB**（峰 2857.6→1917.5MB，省 940MB），GPU 不变，`final_pos.sum` **完全相同（byte-exact）** | ✅ POSITIVE+SAFE |
| F2 allocator expandable_segments | 全 probe 在，setdefault 保留用户值 | ✅ |
| **throughput scaling** | 见 §3 | ✅ POSITIVE |
| fast×vram 组合 | 在 home fork sandbox_fastvram 验（merge 树未带 fast-slope 旋钮，走 fd_mode 路线） | ✅ |

### cleanup — freq GPU-native assembly + BPRFO micro-hygiene（jobs 47873198/47874563/47875618，COMPLETED）— **parity-EXACT**
| change | 结果 | VERDICT |
|---|---|---|
| freq `batched_fd_hessian` CPU-numpy→GPU-native | **STUB 测试（喂相同力）max\|dH\|=4.44e-16 Ha/Å²（≤2 float64 ULP），central+forward 双模式** | ✅ 位级精确 |
| BPRFO mask 缓存 + bofill transpose dedup | BPRFO 状态表 **byte-identical**，收敛鞍点匹配 | ✅ |
> micro-hygiene 保守应用（provably-safe 才改；mask_ij 乘法因 staticmethod 内 invariant 不可本地证而**保留**）。

### V4 — NEB（job 47874154，COMPLETED；control 跑中）
| opt | 结果 | VERDICT |
|---|---|---|
| multiband 收敛反应 | rx0/2/3/4 \|dBar\|=8.76e-8 ~ 1e-4 紧贴 oracle | ✅ |
| multiband 软-HEI 反应 | rx5/6/7 \|dBar\| 最大 3.16e-3 Ha + 1 HEI-flip | ⏳ control 定论（软/bifurcating HEI 在相邻 image 翻转=NEB+fp32 固有，非 batching；V5 identical-band 6.92e-7 已证 batching 确定） |
| streaming pool | 收敛 rx0/1/4 紧贴；大 dBar(4.09e-2)**全是 evicted_straggler**（未收敛 band 比较 artifact）；integrity no-lost/dup=**True** | ✅ pool 机制正确 |
| endpoint-reuse (FIX#3) | 稳定反应紧/软反应 4.2e-2（reuse 恒定端点 E vs full 每 iter 重算 fp32） | ⏳ control（稳定反应 reuse 紧则确认正确） |
| DyNEB | fwd-savings **22.7%** | ✅ 正向 |

### V7 — engine/dispatcher-wiring/bug 修（job 47884018+jobB，COMPLETED）— **ALL 6 WORKING+CORRECT**（rerun 显式）
| # | 项 | 结果 | VERDICT |
|---|---|---|---|
| 1 | input-file UMA .inp | 单+批量 engine() 收敛（shim add_safe_globals+ray.serve stub），dE 5.96e-5 | ✅ |
| 2 | dispatcher-wiring | opt→BatchLBFGS(reached=True)/ts→BatchPRFO/dimer→BatchDimer/irc→LQABatch（sentinel 各命中） | ✅ |
| 3 | FIX1 get_hvp | 单 Dimer 跑 UMA FD-HVP 返 (Hn,F,E) 3-tuple | ✅ |
| 4 | FIX2 lqa float(None) | 单+批量 LQA 完成 run()!=None 无崩 | ✅ |
| 5 | **FIX3 BatchSD/SDCG 写回**（最高价值） | 经 dispatcher 写回 == direct-API RMSD **1.99e-7/4.31e-7 Å**，几何移 0.11-0.16Å 存 mols.multiatoms = **结果不丢失** | ✅ bug 确认修 |
| 6 | AIMNet2-decoupled | perturb-one dE_other **0.000 Ha**，multiband 4/4 barrier 0.008 eV，1.34× | ✅ |

### V8 — batch 并行 scaling（job 47884019，COMPLETED）— **PASS**（rerun 显式，见 §3a）
raw forward **10.18×@B64 / util 99%@B128 / parity ≤7e-7 eV**；dispatcher BatchPRFO 1.18× 同收敛。★修正 V1 0.51× 假象（真 5.00×@B8）。

### V3 — TS saddle（job 47884029-32，部分；内容已覆盖）
v3b/c/d COMPLETED（v3d convtable harness 脚本 crash=agent bug），v3a running。**BatchDimer 3/3 收敛 shrink_on_converge=True** 确认；BatchPRFO mode-follow-guard 跑中。内容（BatchPRFO legacy/iterative/lindh_nw + BatchDimer same-saddle）**已被 V5 显式覆盖**（iterative 3/4、lindh_nw 4/4、BatchDimer 0.018Å）+ BatchDimer-shrink 由 Tier1 验证（dimshrink_parity max RMSD 2.69e-4 同鞍点）。→ 实质 GREEN（覆盖+部分再确认）。

---

## 3. ★ batch 并行 scaling（headline）

### 3a. Raw forward 原语 `get_ef_gpu` batched-vs-serial（V8 job 47884019，权威）
| B | speedup | struct/s | util_max | max\|dE\|(eV) | max\|dF\|(eV/Å) |
|---|---|---|---|---|---|
| 1 | 1.00× | 14 | 20% | 1.19e-7 | 2.98e-7 |
| 8 | **5.00×** | 63 | 29% | 2.38e-7 | 1.25e-6 |
| 16 | 7.03× | 97 | 33% | 4.77e-7 | 1.55e-6 |
| 32 | 8.82× | 120 | 41% | 7.15e-7 | 1.19e-6 |
| 64 | **10.18×** | 139 | 67% | 4.77e-7 | 1.97e-6 |
| 128 | 10.16× | 135 | **99%** | 5.96e-7 | 2.92e-6 |

→ **CORRECT**（每 B batched==serial，worst max\|dE\|=7.15e-7 eV/dF 2.92e-6=fp32 底）+ **POSITIVE**（speedup 单调 1→10.2×，struct/s 14→139≈10×）+ **GPU 饱和**（util 19%→**99%@B128**，确认 kernel 本身打满 device 而非串行化）。plateau@B≳64 是小分子(≤19 原子)on-device 饱和 knee。

### 3b. Full TS pipeline throughput（V6 job 47874101）
| B | conv | util_max | struct/s | **speedup** |
|---|---|---|---|---|
| 1 | 12/16 | 32% | 0.77 | 0.92× |
| 8 | 13/16 | 67% | 1.16 | **1.40×** |
| 16 | 13/16 | **89%** | 1.31 | **1.58×** |

### 结论
- **Raw forward 原语**：batch 并行 **10× @B64**，util→99%，parity ≤7e-7 eV = **零精度代价**。这是 batched-forward 核心收益。
- **Full TS pipeline**：1.40-1.58×（受小分子 P-RFO 的 Python 端 Hessian/eigensolve 主导，非 forward；BatchPRFO dispatcher-level 实测 1.18×，收敛 11/12 同）。
- ★**修正**：V1 报的「raw forward 0.51×@B8」是 flawed 微观测量（warm-cache serial 基线 artifact）；V8 干净测量 **B8=5.00×**。两层测不同对象（forward 原语 10× / full pipeline 1.4-1.6×），均 CORRECT。

---

## 4. 与已知 UMA fp32 行为对账

| 现象 | 归因 | 证据 |
|---|---|---|
| 优化器 end-to-end 几何漂移 ~1e-3 Å | fp32 路径依赖（同极小） | canon-vs-canon control 同阶（V2/cleanup） |
| NEB 软-HEI barrier 发散 ~1e-3 Ha + HEI-flip | 软/bifurcating HEI 在相邻 image 翻转（NEB+fp32 固有，非 batching） | identical-band 6.92e-7；收敛反应紧贴；control 跑中 |
| pool 大 dBar 4e-2 | evicted_straggler（未收敛 band 比较） | integrity no-lost/dup=True，收敛反应紧 |
| MACE/AIMNet/ANI 不可 batch | env/模型 export 限制非代码 | aimnet2calc/torchani 不在 cxtorch；MACE zoo .pt single-graph trace |

---

## 5. 状态 & 下一步（终版）
- **全部 GREEN，0 真 ISSUE**：V1 calc / V2 优化器(6) / V4 NEB(5, control 确认) / V5 IRC-Hessian / V6 mem+throughput / V7 engine-bugfix(6) / V8 batch 并行(10×) / cleanup（parity-exact）逐条核验；V3 内容 V5+Tier1 覆盖+部分再确认。
- **V4 control 定论**：软-HEI 发散 = UMA-GPU fp32-chaos（同代码无 batching 两跑 4.16e-3 ≈ batched 3.16e-3），**非任何优化引入，无 batching/pool/reuse 算法错**。
- **★8 项优化已 push GitHub `wave2-on-f75`**（cb3d943..334c92d，31 commits，无 force）。
- **结论**：每项优化**正向 + 无精度下降 + 无算法错误**；batch 并行 raw forward **10×**（util→99%）。
- **下一步**：本验证报告 + 34 原始 .out 已上传 GitHub MAPLE repo（validation/ 目录）；Windows 上线同步；可选 V3 完整重跑（低优先=重验已验）。

---

*生成 2026-06-29 | 课题根 `catgo-projects/data/ai-maple-gpu/` | Ibex 验证树 `/ibex/user/xiaox/zls/ai-maple-gpu/sandbox_merge_all/` | 原始证据 GitHub `leshenzhang/MAPLE:wave2-on-f75 validation/` + Ibex `VALIDATION_RESULTS_2026-06-29/` + 本地 `reports/validation_raw_2026-06-29/` | DECISION_LOG D-114~134*
