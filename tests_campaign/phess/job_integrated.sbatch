#!/bin/bash
#SBATCH --job-name=ob_integ
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-gpu/opt2026/integrated/out/integ_%j.out
set -o pipefail

ROOT=/ibex/user/xiaox/zls/ai-gpu/opt2026/integrated
BASE=/ibex/user/xiaox/zls/ai-gpu/opt2026/hostopt/MAPLE_base
INT=$ROOT/MAPLE
PY=/home/wangc0i/miniconda3/envs/cxtorch/bin/python
MODEL=/ibex/user/xiaox/zls/ai-gpu/MAPLE/maple/function/calculator/model/uma-s-1p1.pt
PKL=/ibex/user/xiaox/zls/ai-gpu/dev/testset/ts1x_seed42_100.pkl
OUT=$ROOT/out
PROD=$INT/tests_campaign/phess/phess_prod.py
mkdir -p "$OUT"
hostname; nvidia-smi -L
FAIL=0

# Every arm runs in THIS job -> same node, same GPU (D-267: within-node band is the
# only band under which a <10% effect is decidable).
run () {  # run <tag> <fork> <rep> <extra args...>
  local tag=$1 fork=$2 rep=$3; shift 3
  echo "===== ARM $tag rep$rep ====="
  $PY "$PROD" --fork "$fork" --model "$MODEL" --pkl "$PKL" --out-dir "$OUT" \
      --tag "$tag" --replicate "$rep" --n 100 --seed 20260728 --sigma 0.02 \
      --max-iter 120 "$@"
  local rc=$?
  [ $rc -ne 0 ] && { echo "ARM $tag rep$rep FAILED rc=$rc"; FAIL=$((FAIL+1)); }
  # product gate: a zero-exit arm that wrote no JSON is still a failure
  if [ ! -s "$OUT/${tag}_r${rep}.json" ]; then
    echo "ARM $tag rep$rep produced NO JSON"; FAIL=$((FAIL+1))
  fi
}

for rep in 1 2; do
  # 1. reference: frozen bc3335e, stock config
  run base        "$BASE" "$rep"
  # 2. no-regression gate: integrated tree, stock config (must match arm 1 in-band)
  run int_default "$INT"  "$rep"
  # 3. full stack: forward-FD + update-only + fast_inference + streaming pool
  run int_stack   "$INT"  "$rep" --fd-mode forward --recalc 1000000 \
                                 --fast-inference --pool 16 --refill-min 4
  # 4. stack minus pool: isolates the pool contribution
  run int_nopool  "$INT"  "$rep" --fd-mode forward --recalc 1000000 --fast-inference
done

echo "INTEGRATED_ARMS_FAIL=$FAIL"
[ $FAIL -eq 0 ] || exit 1
echo INTEGRATED_DONE
