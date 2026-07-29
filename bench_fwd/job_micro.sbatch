#!/bin/bash
#SBATCH --job-name=fwd_micro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-gpu/opt2026/fwd/out/fwd_micro_%j.out
set -u
set -o pipefail
FAIL=0
export PATH=/home/wangc0i/miniconda3/envs/cxtorch/bin:$PATH
export PYTHONNOUSERSITE=1
ROOT=/ibex/user/xiaox/zls/ai-gpu/opt2026/fwd
PY=/home/wangc0i/miniconda3/envs/cxtorch/bin/python
export PYTHONPATH=$ROOT/MAPLE
export MODEL=/ibex/user/xiaox/zls/ai-gpu/MAPLE/maple/function/calculator/model/uma-s-1p1.pt
export MODEL_DIR=/ibex/user/xiaox/zls/ai-gpu/MAPLE/maple/function/calculator/model
export PKL=/ibex/user/xiaox/zls/ai-gpu/dev/testset/ts1x_seed42_100.pkl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
( while true; do echo "[util $(date +%H:%M:%S)] $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; sleep 120; done ) &
TICKER=$!
trap "kill $TICKER 2>/dev/null" EXIT
echo "===== A3-fwd UMA micro ====="
OUT=$ROOT/out/fwd_micro_uma.json $PY $ROOT/MAPLE/bench_fwd/fwd_micro.py
rc=$?
echo "rc_uma=$rc"
if [ "$rc" -ne 0 ]; then FAIL=$rc; fi
echo "===== A3-fwd MACEPOL micro ====="
OUT=$ROOT/out/macepol_micro.json $PY $ROOT/MAPLE/bench_fwd/macepol_micro.py
rc=$?
echo "rc_macepol=$rc"
if [ "$rc" -ne 0 ]; then FAIL=$rc; fi
echo FWD_MICRO_JOB_DONE
if [ "$FAIL" -ne 0 ]; then echo "JOB_FAILED rc=$FAIL"; fi
exit $FAIL
