#!/bin/bash
#SBATCH --job-name=fwd_e2e2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-gpu/opt2026/fwd/out/fwd_e2e2_%j.out
set -u
set -o pipefail
FAIL=0
export PATH=/home/wangc0i/miniconda3/envs/cxtorch/bin:$PATH
export PYTHONNOUSERSITE=1
ROOT=/ibex/user/xiaox/zls/ai-gpu/opt2026/fwd
PY=/home/wangc0i/miniconda3/envs/cxtorch/bin/python
export FORK=$ROOT/MAPLE
export PYTHONPATH=$ROOT/MAPLE
export MODEL=/ibex/user/xiaox/zls/ai-gpu/MAPLE/maple/function/calculator/model/uma-s-1p1.pt
export PKL=/ibex/user/xiaox/zls/ai-gpu/dev/testset/ts1x_seed42_100.pkl
export OUTDIR=$ROOT/out
export N=100 B=16 N_IMAGES=10 NEB_MAXITER=150 DYNEB=1 RECALC=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
( while true; do echo "[util $(date +%H:%M:%S)] $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; sleep 120; done ) &
TICKER=$!
trap "kill $TICKER 2>/dev/null" EXIT
echo "===== R3-7 std-MACE byte-equality gate (model-free) ====="
OUT=$ROOT/out/gate_r37.json $PY $ROOT/MAPLE/bench_fwd/gate_r37_stdmace.py
rc=$?
echo "rc_r37=$rc"
if [ "$rc" -ne 0 ]; then FAIL=$rc; fi
echo "===== e2e arm fi (fast_inference=1) replicate 2 ====="
FAST_INFERENCE=1 TAG=e2e_fi_r2 $PY $ROOT/MAPLE/bench_fwd/pipeline_e2e.py
rc=$?
echo "rc_fi2=$rc"
if [ "$rc" -ne 0 ]; then FAIL=$rc; fi
echo "===== e2e arm base (fast_inference=0) replicate 2 ====="
FAST_INFERENCE=0 TAG=e2e_base_r2 $PY $ROOT/MAPLE/bench_fwd/pipeline_e2e.py
rc=$?
echo "rc_base2=$rc"
if [ "$rc" -ne 0 ]; then FAIL=$rc; fi
echo FWD_E2E2_DONE
if [ "$FAIL" -ne 0 ]; then echo "JOB_FAILED rc=$FAIL"; fi
exit $FAIL
