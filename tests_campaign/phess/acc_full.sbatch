#!/bin/bash
#SBATCH --job-name=phess_acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-gpu/opt2026/phess/logs/acc_%j.out
set -u
set -o pipefail
FAILED=0
ROOT=/ibex/user/xiaox/zls/ai-gpu/opt2026/phess
FORK=$ROOT/MAPLE
export PATH=/home/wangc0i/miniconda3/envs/cxtorch/bin:${PATH}
export PYTHONPATH=$FORK:${PYTHONPATH:-}
MODEL=$FORK/maple/function/calculator/model/uma-s-1p1.pt
PKL=/ibex/user/xiaox/zls/ai-gpu/dev/testset/ts1x_seed42_100.pkl
mkdir -p $ROOT/out $ROOT/logs
nvidia-smi --query-gpu=name --format=csv,noheader
( while true; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used \
    --format=csv,noheader,nounits >> $ROOT/logs/gpuutil_${SLURM_JOB_ID}.log; sleep 20; done ) &
SAMPLER=$!

echo "########## P1: forward-vs-central FD (30 cases, 3 deltas)"
python $FORK/tests_campaign/phess/phess_accuracy.py \
  --fork $FORK --model $MODEL --pkl $PKL --out-dir $ROOT/out \
  --tag acc_fd --probe fd --n-cases 30 --fd-deltas 1e-3 2e-3 5e-3
rc=$?
echo "RC_fd=$rc"
if [ $rc -ne 0 ] || [ ! -s $ROOT/out/acc_fd.json ]; then echo "ARTIFACT_GATE_FAIL acc_fd rc=$rc"; FAILED=$((FAILED+1)); fi

echo "########## P2: quasi-Newton update fidelity (20 cases, 8 steps)"
python $FORK/tests_campaign/phess/phess_accuracy.py \
  --fork $FORK --model $MODEL --pkl $PKL --out-dir $ROOT/out \
  --tag acc_update --probe update --n-cases 20 --update-steps 8
rc=$?
echo "RC_upd=$rc"
if [ $rc -ne 0 ] || [ ! -s $ROOT/out/acc_update.json ]; then echo "ARTIFACT_GATE_FAIL acc_update rc=$rc"; FAILED=$((FAILED+1)); fi

echo "########## P3: iterative leftmost eigenpair (Lanczos vs LOBPCG, 20 cases)"
python $FORK/tests_campaign/phess/phess_accuracy.py \
  --fork $FORK --model $MODEL --pkl $PKL --out-dir $ROOT/out \
  --tag acc_iter --probe lanczos --n-cases 20 \
  --lanczos-m 4 8 16 32 --lobpcg-iters 2 4 8
rc=$?
echo "RC_iter=$rc"
if [ $rc -ne 0 ] || [ ! -s $ROOT/out/acc_iter.json ]; then echo "ARTIFACT_GATE_FAIL acc_iter rc=$rc"; FAILED=$((FAILED+1)); fi

echo "########## P3b: LOBPCG without preconditioner (ablation)"
python $FORK/tests_campaign/phess/phess_accuracy.py \
  --fork $FORK --model $MODEL --pkl $PKL --out-dir $ROOT/out \
  --tag acc_iter_noprec --probe lanczos --n-cases 20 \
  --lanczos-m 8 --lobpcg-iters 4 8 --precond none
rc=$?
echo "RC_iter_noprec=$rc"
if [ $rc -ne 0 ] || [ ! -s $ROOT/out/acc_iter_noprec.json ]; then echo "ARTIFACT_GATE_FAIL acc_iter_noprec rc=$rc"; FAILED=$((FAILED+1)); fi

echo "########## P4: core-region radius scan (largest structures, >=15 atoms)"
python $FORK/tests_campaign/phess/phess_accuracy.py \
  --fork $FORK --model $MODEL --pkl $PKL --out-dir $ROOT/out \
  --tag acc_core --probe core --n-cases 15 --min-atoms 15 \
  --core-radii 1.5 2.0 2.5 3.0 3.5 4.0 5.0 99.0
rc=$?
echo "RC_core=$rc"
if [ $rc -ne 0 ] || [ ! -s $ROOT/out/acc_core.json ]; then echo "ARTIFACT_GATE_FAIL acc_core rc=$rc"; FAILED=$((FAILED+1)); fi

kill $SAMPLER 2>/dev/null
awk -F, '{g+=$1; m+=$3; n++; if($1>mx)mx=$1} END{if(n)printf "GPUUTIL samples=%d mean_gpu=%.1f%% max_gpu=%.0f%% mean_memMB=%.0f\n",n,g/n,mx,m/n}' \
  $ROOT/logs/gpuutil_${SLURM_JOB_ID}.log
echo "PHESS_ACC_ALL_DONE failed_probes=$FAILED"
exit $FAILED
