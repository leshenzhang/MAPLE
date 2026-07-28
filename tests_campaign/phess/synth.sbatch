#!/bin/bash
#SBATCH --job-name=phess_synth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-gpu/opt2026/phess/logs/synth_%j.out
set -u
set -o pipefail
ROOT=/ibex/user/xiaox/zls/ai-gpu/opt2026/phess
FORK=$ROOT/MAPLE
export PATH=/home/wangc0i/miniconda3/envs/cxtorch/bin:${PATH}
export PYTHONPATH=$FORK:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=""
python $FORK/tests_campaign/phess/phess_accuracy.py \
  --fork $FORK --model none --pkl /ibex/user/xiaox/zls/ai-gpu/dev/testset/ts1x_seed42_100.pkl \
  --out-dir $ROOT/out --tag acc_synth --probe synth --n-cases 2 \
  --lanczos-m 4 8 16 30 --lobpcg-iters 2 4 8 --synth-spreads 10 100 1000
rc=$?
echo "RC=$rc"
J=$ROOT/out/acc_synth.json
if [ $rc -ne 0 ] || [ ! -s "$J" ]; then echo "ARTIFACT_GATE_FAIL synth rc=$rc"; echo SYNTH_DONE; exit 1; fi
echo SYNTH_DONE
