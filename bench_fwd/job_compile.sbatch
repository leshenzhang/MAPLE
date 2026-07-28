#!/bin/bash
#SBATCH --job-name=fwd_comp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-gpu/opt2026/fwd/out/fwd_comp_%j.out
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
export TORCHINDUCTOR_CACHE_DIR=$ROOT/out/inductor_cache
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
( while true; do echo "[util $(date +%H:%M:%S)] $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; sleep 120; done ) &
TICKER=$!
trap "kill $TICKER 2>/dev/null" EXIT
echo "===== A3-fwd MACE-POL torch.compile / cudagraph arms ====="
OUT=$ROOT/out/macepol_compile.json $PY $ROOT/MAPLE/bench_fwd/macepol_compile.py
rc=$?
echo "rc_comp=$rc"
if [ "$rc" -ne 0 ]; then FAIL=$rc; fi
echo FWD_COMPILE_DONE
if [ "$FAIL" -ne 0 ]; then echo "JOB_FAILED rc=$FAIL"; fi
exit $FAIL
