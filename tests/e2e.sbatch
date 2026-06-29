#!/bin/bash
#SBATCH --job-name=mmgbsa-e2e
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0:18:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=36G
#SBATCH --output=/ibex/user/xiaox/zls/ai-maple-md/MAPLE/worktrees/mmgbsa-batch/e2e_%j.out
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/ibex/user/xiaox/zls/ai-maple-md/envs/plumed/bin/python
echo "host $(hostname) | started $(date)"
$PY /ibex/user/xiaox/zls/ai-maple-md/MAPLE/worktrees/mmgbsa-batch/tests/run_e2e.py
echo "rc=$? | finished $(date)"
