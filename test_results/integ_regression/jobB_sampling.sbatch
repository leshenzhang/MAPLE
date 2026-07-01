#!/bin/bash
#SBATCH --job-name=integB_sampling
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-maple-md/MAPLE/worktrees/integ2/test_results/integ_regression/jobB_%j.out
set -u
WT=/ibex/user/xiaox/zls/ai-maple-md/MAPLE/worktrees/integ2
PY=/ibex/user/xiaox/zls/ai-maple-md/envs/plumed/bin/python
export PYTHONPATH=$WT PYTHONNOUSERSITE=1 PYTHONUTF8=1 LC_ALL=C.UTF-8
cd $WT
echo "=== JOBB node $(hostname) $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
run(){ echo; echo "########## BEGIN $1 ##########"; timeout 1200 $PY "$2"; echo "@@ $1 EXIT=$?"; echo "########## END $1 ##########"; }
run REST2   maple/function/dispatcher/md/ensemble/_test_hremd_rest2.py
run METAD   maple/function/dispatcher/md/ensemble/_smoke_metad_batched.py
run PLUMED  maple/function/dispatcher/md/bias/_test_plumed.py
echo "DONE_JOBB $(date)"
