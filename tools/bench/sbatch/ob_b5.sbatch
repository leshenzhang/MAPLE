#!/bin/bash
#SBATCH --job-name=ob_b5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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
export MOLS=/ibex/user/xiaox/zls/ai-gpu/dev/testset/ts1x_seed42_100.pkl
export TOY_MACE=/ibex/user/xiaox/zls/ai-maple-md/MAPLE/worktrees/integ/vfy_b78/out_plumed/toy_maceomol.pt
export ZOO_DIR=/ibex/user/xiaox/zls/ai-maple-gpu/zoo_models
export MACEOFF_RAW=/ibex/user/xiaox/zls/ai-gpu/dev/d2/MACE-OFF23_small.model
for r in 1 2; do
  OUT=$OUT/b5_mace_a100_r${r}.json /home/wangc0i/miniconda3/envs/cxtorch/bin/python /ibex/user/xiaox/zls/ai-maple-gpu/verify_2026-07-14/a3_mace.py
  echo "rc_b5_r${r}=$?"
done
echo B5_DONE
