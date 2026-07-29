#!/bin/bash
#SBATCH --job-name=ob_pool
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-gpu/opt2026/integrated/out/pool_%j.out
set -o pipefail
ROOT=/ibex/user/xiaox/zls/ai-gpu/opt2026/integrated
INT=$ROOT/MAPLE
PY=/home/wangc0i/miniconda3/envs/cxtorch/bin/python
MODEL=/ibex/user/xiaox/zls/ai-gpu/MAPLE/maple/function/calculator/model/uma-s-1p1.pt
PKL=/ibex/user/xiaox/zls/ai-gpu/dev/testset/ts1x_seed42_100.pkl
OUT=$ROOT/out
PROD=$INT/tests_campaign/phess/phess_prod.py
hostname; FAIL=0

# Pool streams N=100 through an active batch of 16, so the outer-iteration budget
# must cover the QUEUE, not the batch: at max-iter 120 the previous run ended with
# queue_left=18 and only 82/100 structures processed, which made its wall time
# meaningless. Both arms here get the same raised budget so the comparison is matched.
run () { local tag=$1 rep=$2; shift 2
  echo "===== ARM $tag rep$rep ====="
  $PY "$PROD" --fork "$INT" --model "$MODEL" --pkl "$PKL" --out-dir "$OUT" \
      --tag "$tag" --replicate "$rep" --n 100 --seed 20260728 --sigma 0.02 \
      --max-iter 600 --fd-mode forward --recalc 1000000 --fast-inference "$@"
  local rc=$?
  [ $rc -ne 0 ] && { echo "ARM $tag rep$rep FAILED rc=$rc"; FAIL=$((FAIL+1)); }
  [ -s "$OUT/${tag}_r${rep}.json" ] || { echo "ARM $tag rep$rep NO JSON"; FAIL=$((FAIL+1)); }
}
for rep in 1 2; do
  run pool600_nopool "$rep"
  run pool600_pool16 "$rep" --pool 16 --refill-min 4
done
echo "POOL_ARMS_FAIL=$FAIL"; [ $FAIL -eq 0 ] || exit 1; echo POOL_DONE
