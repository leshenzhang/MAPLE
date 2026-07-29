#!/bin/bash
#SBATCH --job-name=ob_bsw
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-gpu/opt2026/integrated/out/bsw_%j.out
set -o pipefail
# Batch-axis sweep at FIXED workload (N=100), for the baseline config and for the
# f111 fast stack. Everything in one job -> same node, so D-267's within-node band
# applies. D-274: every earlier phess number was B=100 (one batch); this is the first
# time B is varied here, and D-267 already showed the CHUNKED pipeline regresses at
# B=64, so the stack's optimum B is an open question, not an assumption.
ROOT=/ibex/user/xiaox/zls/ai-gpu/opt2026/integrated
INT=$ROOT/MAPLE
PY=/home/wangc0i/miniconda3/envs/cxtorch/bin/python
MODEL=/ibex/user/xiaox/zls/ai-gpu/MAPLE/maple/function/calculator/model/uma-s-1p1.pt
PKL=/ibex/user/xiaox/zls/ai-gpu/dev/testset/ts1x_seed42_100.pkl
OUT=$ROOT/out_bsw
PROD=$INT/tests_campaign/phess/phess_prod.py
mkdir -p "$OUT"; hostname; nvidia-smi -L; FAIL=0
run () { local tag=$1 rep=$2; shift 2
  echo "===== ARM $tag rep$rep ====="
  $PY "$PROD" --fork "$INT" --model "$MODEL" --pkl "$PKL" --out-dir "$OUT" \
      --tag "$tag" --replicate "$rep" --n 100 --seed 20260728 --sigma 0.02 \
      --max-iter 120 --skip-saddle "$@"
  local rc=$?
  [ $rc -ne 0 ] && { echo "ARM $tag rep$rep FAILED rc=$rc"; FAIL=$((FAIL+1)); }
  [ -s "$OUT/${tag}_r${rep}.json" ] || { echo "ARM $tag rep$rep NO JSON"; FAIL=$((FAIL+1)); }
}
for rep in 1 2; do
  for B in 8 16 32 50 100; do
    run "base_B${B}" "$rep" --chunk "$B"
    run "fast_B${B}" "$rep" --chunk "$B" --fd-mode forward --recalc 1000000 --fast-inference
  done
done
echo "BSWEEP_ARMS_FAIL=$FAIL"; [ $FAIL -eq 0 ] || exit 1; echo BSWEEP_DONE
