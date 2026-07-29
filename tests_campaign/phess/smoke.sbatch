#!/bin/bash
#SBATCH --job-name=phess_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-gpu/opt2026/phess/logs/smoke_%j.out
set -u
ROOT=/ibex/user/xiaox/zls/ai-gpu/opt2026/phess
FORK=$ROOT/MAPLE
export PATH=/home/wangc0i/miniconda3/envs/cxtorch/bin:${PATH}
export PYTHONPATH=$FORK:${PYTHONPATH:-}
MODEL=$FORK/maple/function/calculator/model/uma-s-1p1.pt
PKL=/ibex/user/xiaox/zls/ai-gpu/dev/testset/ts1x_seed42_100.pkl
mkdir -p $ROOT/out $ROOT/logs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# background GPU-util sampler (post-submit utilization check, NRP/Ibex discipline)
( while true; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used \
    --format=csv,noheader,nounits >> $ROOT/logs/gpuutil_${SLURM_JOB_ID}.log; sleep 20; done ) &
SAMPLER=$!

echo "=== accuracy smoke (6 cases, fd+lanczos+core) ==="
python $FORK/tests_campaign/phess/phess_accuracy.py \
  --fork $FORK --model $MODEL --pkl $PKL --out-dir $ROOT/out \
  --tag acc_smoke --probe fd lanczos core --n-cases 6 \
  --lanczos-m 4 8 --core-radii 1.5 2.5 99.0
echo "RC_acc=$?"

echo "=== update-probe smoke (4 cases) ==="
python $FORK/tests_campaign/phess/phess_accuracy.py \
  --fork $FORK --model $MODEL --pkl $PKL --out-dir $ROOT/out \
  --tag acc_upd_smoke --probe update --n-cases 4 --update-steps 5
echo "RC_upd=$?"

echo "=== prod smoke (N=10, recalc=8) ==="
python $FORK/tests_campaign/phess/phess_prod.py \
  --fork $FORK --model $MODEL --pkl $PKL --out-dir $ROOT/out \
  --tag smoke_base --n 10 --max-iter 40 --recalc 8
echo "RC_prod=$?"

kill $SAMPLER 2>/dev/null
echo "=== GPU util summary ==="
awk -F, '{g+=$1; m+=$3; n++; if($1>mx)mx=$1} END{if(n)printf "samples=%d mean_gpu=%.1f%% max_gpu=%.0f%% mean_memMB=%.0f\n",n,g/n,mx,m/n}' \
  $ROOT/logs/gpuutil_${SLURM_JOB_ID}.log
echo SMOKE_ALL_DONE
