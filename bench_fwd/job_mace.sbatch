#!/bin/bash
#SBATCH --job-name=fwd_mace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-gpu/opt2026/fwd/out/fwd_mace_%j.out
set -u
set -o pipefail
FAIL=0
export PATH=/home/wangc0i/miniconda3/envs/cxtorch/bin:$PATH
export PYTHONNOUSERSITE=1
ROOT=/ibex/user/xiaox/zls/ai-gpu/opt2026/fwd
PY=/home/wangc0i/miniconda3/envs/cxtorch/bin/python
export PYTHONPATH=$ROOT/MAPLE
export PKL=/ibex/user/xiaox/zls/ai-gpu/dev/testset/ts1x_seed42_100.pkl
export MACE_CKPT=/ibex/user/xiaox/zls/ai-chemrobot/models/20231210mace128L0_energy_epoch249model
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
( while true; do echo "[util $(date +%H:%M:%S)] $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; sleep 120; done ) &
TICKER=$!
trap "kill $TICKER 2>/dev/null" EXIT
echo "===== R3-7 std-MACE gate (byte-equality + REAL-forward tier2) ====="
OUT=$ROOT/out/gate_r37.json $PY $ROOT/MAPLE/bench_fwd/gate_r37_stdmace.py
rc=$?
echo "rc_r37=$rc"
if [ "$rc" -ne 0 ]; then FAIL=$rc; fi
echo "===== MACE micro: dtype/TF32 policy (candidate 6) ====="
OUT=$ROOT/out/maceoff_micro.json $PY $ROOT/MAPLE/bench_fwd/maceoff_micro.py
rc=$?
echo "rc_maceoff=$rc"
if [ "$rc" -ne 0 ]; then FAIL=$rc; fi
echo FWD_MACE_DONE
if [ "$FAIL" -ne 0 ]; then echo "JOB_FAILED rc=$FAIL"; fi
exit $FAIL
