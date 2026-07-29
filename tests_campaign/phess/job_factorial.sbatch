#!/bin/bash
#SBATCH --job-name=ob_fact
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-gpu/opt2026/integrated/out/fact_%j.out
set -o pipefail

# 2^3 factorial over the three optimizations that make up the 5.30x stack:
#   F = forward-FD      (--fd-mode forward   vs central)
#   U = update-only     (--recalc 1000000    vs 8)
#   I = fast_inference  (--fast-inference    vs off)
# All 16 runs live in THIS job, so every comparison is same-node/same-GPU and the
# within-node band (D-267: wall <=1.29%) applies. Geometries are written per arm so
# the same-saddle cost can be computed per structure afterwards against arm 000.
ROOT=/ibex/user/xiaox/zls/ai-gpu/opt2026/integrated
INT=$ROOT/MAPLE
PY=/home/wangc0i/miniconda3/envs/cxtorch/bin/python
MODEL=/ibex/user/xiaox/zls/ai-gpu/MAPLE/maple/function/calculator/model/uma-s-1p1.pt
PKL=/ibex/user/xiaox/zls/ai-gpu/dev/testset/ts1x_seed42_100.pkl
OUT=$ROOT/out_fact
PROD=$INT/tests_campaign/phess/phess_prod.py
mkdir -p "$OUT"; hostname; nvidia-smi -L; FAIL=0

run () {  # run <tag> <rep> <extra...>
  local tag=$1 rep=$2; shift 2
  echo "===== ARM $tag rep$rep ====="
  $PY "$PROD" --fork "$INT" --model "$MODEL" --pkl "$PKL" --out-dir "$OUT" \
      --tag "$tag" --replicate "$rep" --n 100 --seed 20260728 --sigma 0.02 \
      --max-iter 120 "$@"
  local rc=$?
  [ $rc -ne 0 ] && { echo "ARM $tag rep$rep FAILED rc=$rc"; FAIL=$((FAIL+1)); }
  [ -s "$OUT/${tag}_r${rep}.json" ] || { echo "ARM $tag rep$rep NO JSON"; FAIL=$((FAIL+1)); }
}

for rep in 1 2; do
  run f000 "$rep"
  run f100 "$rep" --fd-mode forward
  run f010 "$rep" --recalc 1000000
  run f001 "$rep" --fast-inference
  run f110 "$rep" --fd-mode forward --recalc 1000000
  run f101 "$rep" --fd-mode forward --fast-inference
  run f011 "$rep" --recalc 1000000 --fast-inference
  run f111 "$rep" --fd-mode forward --recalc 1000000 --fast-inference
done
echo "FACTORIAL_ARMS_FAIL=$FAIL"
[ $FAIL -eq 0 ] || exit 1
echo FACTORIAL_DONE
