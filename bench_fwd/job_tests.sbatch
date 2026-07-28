#!/bin/bash
#SBATCH --job-name=fwd_tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-gpu/opt2026/fwd/out/fwd_tests_%j.out
set -u
export PATH=/home/wangc0i/miniconda3/envs/cxtorch/bin:$PATH
export PYTHONNOUSERSITE=1
ROOT=/ibex/user/xiaox/zls/ai-gpu/opt2026/fwd
PY=/home/wangc0i/miniconda3/envs/cxtorch/bin/python
export PYTHONPATH=$ROOT/MAPLE
hostname
echo "===== in-tree batch-calc contract regression (CPU) ====="
cd $ROOT/MAPLE && $PY -m pytest tests/test_batch_calc_contract.py -q 2>&1 | tail -25
echo "rc_tests=$?"
echo FWD_TESTS_DONE
