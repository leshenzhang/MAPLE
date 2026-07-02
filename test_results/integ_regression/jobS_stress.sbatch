#!/bin/bash
#SBATCH --job-name=integS_stress
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-maple-md/MAPLE/worktrees/integ2/test_results/integ_regression/jobS_%j.out
set -u
WT=/ibex/user/xiaox/zls/ai-maple-md/MAPLE/worktrees/integ2
PY=/ibex/user/xiaox/zls/ai-maple-md/envs/plumed/bin/python
export PYTHONPATH=$WT PYTHONNOUSERSITE=1 PYTHONUTF8=1 LC_ALL=C.UTF-8
cd $WT
echo "=== JOBS node $(hostname) $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
echo; echo '########## BEGIN STRESS ##########'
timeout 900 $PY maple/function/dispatcher/md/ensemble/_test_stress_batched.py
echo "@@ STRESS EXIT=$?"
echo '########## END STRESS ##########'
echo DONE_JOBS $(date)
