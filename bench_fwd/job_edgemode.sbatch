#!/bin/bash
#SBATCH --job-name=fwd_edge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-gpu/opt2026/fwd/out/fwd_edge_%j.out
set -u
set -o pipefail
FAIL=0
export PATH=/home/wangc0i/miniconda3/envs/cxtorch/bin:$PATH
export PYTHONNOUSERSITE=1
ROOT=/ibex/user/xiaox/zls/ai-gpu/opt2026/fwd
PY=/home/wangc0i/miniconda3/envs/cxtorch/bin/python
export PYTHONPATH=$ROOT/MAPLE
export MODEL_DIR=/ibex/user/xiaox/zls/ai-gpu/MAPLE/maple/function/calculator/model
export PKL=/ibex/user/xiaox/zls/ai-gpu/dev/testset/ts1x_seed42_100.pkl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
( while true; do echo "[util $(date +%H:%M:%S)] $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; sleep 60; done ) &
TICKER=$!
trap "kill $TICKER 2>/dev/null" EXIT
echo "===== NON-VACUOUS edge_mode='full' gate (stretched geometries) ====="
OUT=$ROOT/out/gate_edgemode_full.json $PY $ROOT/MAPLE/bench_fwd/gate_edgemode_full.py
rc=$?
echo "rc_edge=$rc"
if [ "$rc" -ne 0 ]; then FAIL=$rc; fi
echo FWD_EDGE_DONE
if [ "$FAIL" -ne 0 ]; then echo "JOB_FAILED rc=$FAIL"; fi
exit $FAIL
