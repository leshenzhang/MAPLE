#!/bin/bash
#SBATCH --job-name=phess_reg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:12:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-gpu/opt2026/phess/logs/reg_%j.out
# DEFAULT-PATH NO-REGRESSION GATE: rerun the exact N=10 config that was measured
# BEFORE the LOBPCG / iter_reorth / quality-recalc commits landed, with all new
# features OFF. Same seed, same start geometries -> the trajectory must reproduce.
set -u
set -o pipefail
ROOT=/ibex/user/xiaox/zls/ai-gpu/opt2026/phess
FORK=$ROOT/MAPLE
export PATH=/home/wangc0i/miniconda3/envs/cxtorch/bin:${PATH}
export PYTHONPATH=$FORK:${PYTHONPATH:-}
python $FORK/tests_campaign/phess/phess_prod.py \
  --fork $FORK --model $FORK/maple/function/calculator/model/uma-s-1p1.pt \
  --pkl /ibex/user/xiaox/zls/ai-gpu/dev/testset/ts1x_seed42_100.pkl \
  --out-dir $ROOT/out --tag regress_base --n 10 --max-iter 40 --recalc 8
rc=$?
echo "RC=$rc"
[ $rc -ne 0 ] && { echo "ARTIFACT_GATE_FAIL regress"; exit 1; }
python - <<'PY'
import json
a=json.load(open('/ibex/user/xiaox/zls/ai-gpu/opt2026/phess/out/smoke_base_r1.json'))
b=json.load(open('/ibex/user/xiaox/zls/ai-gpu/opt2026/phess/out/regress_base_r1.json'))
ka=a['log']; kb=b['log']
print('field            pre-edit      post-edit')
ok=True
for k in ('ge_iter','ge_hess','ge_hvp','ge_total','n_converged','n_evicted','n_maxiter','fwd_calls','fwd_reused'):
    va,vb=ka.get(k),kb.get(k)
    flag='' if va==vb else '  <-- DIFF'
    if va!=vb: ok=False
    print(f'{k:<16} {va!s:>12} {vb!s:>12}{flag}')
print('conv_step equal:', ka.get('conv_step')==kb.get('conv_step'))
if ka.get('conv_step')!=kb.get('conv_step'): ok=False
import math
dmax=0.0
for ga,gb in zip(a['final_geoms'],b['final_geoms']):
    if ga is None or gb is None: continue
    for ra,rb in zip(ga,gb):
        for xa,xb in zip(ra,rb): dmax=max(dmax,abs(xa-xb))
print(f'max |dx| final geometries = {dmax:.3e} A')
print('DEFAULT_PATH_REGRESSION:', 'PASS' if (ok and dmax<1e-8) else 'FAIL')
PY
echo REG_DONE
