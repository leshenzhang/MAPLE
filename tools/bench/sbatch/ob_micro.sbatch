#!/bin/bash
#SBATCH --job-name=ob_micro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-gpu/opt2026/bench/logs/%x_%j.out
set -u
export PATH=/home/wangc0i/miniconda3/envs/cxtorch/bin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=/ibex/user/xiaox/zls/ai-gpu/opt2026/bench/MAPLE
export MAPLE_COMMIT=bc3335e
export MODEL=/ibex/user/xiaox/zls/ai-gpu/MAPLE/maple/function/calculator/model/uma-s-1p1.pt
export PKL=/ibex/user/xiaox/zls/ai-gpu/dev/testset/ts1x_seed42_100.pkl
OUT=/ibex/user/xiaox/zls/ai-gpu/opt2026/bench/runs
mkdir -p $OUT
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
( while true; do echo "[util $(date +%H:%M:%S)] $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; sleep 120; done ) &
TICKER=$!
trap "kill $TICKER 2>/dev/null" EXIT
RM="/home/wangc0i/miniconda3/envs/cxtorch/bin/python /ibex/user/xiaox/zls/ai-gpu/opt2026/bench/MAPLE/tools/bench/run_matrix.py"
$RM --dispatcher forward --backend uma --B 1,8,16,32,64,128 --reps 2 --iters 30 --outdir $OUT --tag base
$RM --dispatcher hessian --backend uma --B 1,16,64 --reps 2 --hess-iters 3 --hess-modes numerical,autograd --outdir $OUT --tag base
$RM --dispatcher parity  --backend uma --reps 1 --outdir $OUT --tag base
echo MICRO_DONE rc=$?
