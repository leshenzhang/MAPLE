#!/bin/bash
#SBATCH --job-name=01__backend_matrix
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-maple-md/MAPLE/worktrees/backendmatrix/test_results/backend_matrix/matrix_%j.out
set -u
WT=/ibex/user/xiaox/zls/ai-maple-md/MAPLE/worktrees/backendmatrix
PY=/ibex/user/xiaox/zls/ai-maple-md/envs/plumed/bin/python
export PYTHONPATH=$WT
export PYTHONNOUSERSITE=1
# Force UTF-8 file I/O: the single-system NVT logger writes a Unicode 'tau' char,
# which crashes under the compute-node ascii locale (harness portability, not a
# backend bug).
export PYTHONUTF8=1
export LC_ALL=C.UTF-8
export MATRIX_JSON=$WT/test_results/backend_matrix/backend_matrix_result.json
cd $WT
echo "=== node $(hostname) $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
echo "################## BACKEND x METHOD ADAPTATION MATRIX ##################"
$PY $WT/test_results/backend_matrix/backend_matrix.py; echo "MATRIX_EXIT=$?"
echo "DONE_MATRIX $(date)"
