#!/bin/bash
#SBATCH --job-name=phess_vfd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-gpu/opt2026/phess/logs/vfd_%j.out
set -u
set -o pipefail
ROOT=/ibex/user/xiaox/zls/ai-gpu/opt2026/phess
FORK=$ROOT/MAPLE
export PATH=/home/wangc0i/miniconda3/envs/cxtorch/bin:${PATH}
export PYTHONPATH=$FORK:${PYTHONPATH:-}
python $FORK/tests_campaign/phess/verify_fd_reference.py \
  --fork $FORK --model $FORK/maple/function/calculator/model/uma-s-1p1.pt \
  --pkl /ibex/user/xiaox/zls/ai-gpu/dev/testset/ts1x_seed42_100.pkl \
  --out $ROOT/out/verify_fd_reference.json --case 0 --delta 2e-3
rc=$?
echo "RC=$rc"
[ $rc -ne 0 ] && { echo "ARTIFACT_GATE_FAIL verify_fd"; exit $rc; }
echo VFD_DONE
