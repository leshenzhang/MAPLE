#!/bin/bash
#SBATCH --job-name=01__genadapt_smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=genadapt_smoke_%j.out
#SBATCH --error=genadapt_smoke_%j.out
set -euo pipefail
WT=/ibex/user/xiaox/zls/ai-maple-md/worktrees/md-universal-adapter
PY=/ibex/user/xiaox/zls/ai-maple-md/envs/plumed/bin/python
echo "host=$(hostname) date=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$WT"
PYTHONPATH="$WT:${PYTHONPATH:-}" "$PY" tests/gpu_smoke_generic_vs_native.py
echo "SMOKE_EXIT=$?"
