#!/bin/bash
#SBATCH --job-name=phess_grp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-gpu/opt2026/phess/logs/grp_%j.out
# Env contract: CONFIGS = "tag|rep|args ; tag|rep|args ; ..."
# One GPU allocation runs several configs back to back -- far kinder to a
# congested queue than one job per config.
set -u
ROOT=/ibex/user/xiaox/zls/ai-gpu/opt2026/phess
FORK=$ROOT/MAPLE
export PATH=/home/wangc0i/miniconda3/envs/cxtorch/bin:${PATH}
export PYTHONPATH=$FORK:${PYTHONPATH:-}
MODEL=$FORK/maple/function/calculator/model/uma-s-1p1.pt
PKL=/ibex/user/xiaox/zls/ai-gpu/dev/testset/ts1x_seed42_100.pkl
NCASE=${NCASE:-100}
MAXIT=${MAXIT:-120}
mkdir -p $ROOT/out $ROOT/logs
nvidia-smi --query-gpu=name --format=csv,noheader
( while true; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used \
    --format=csv,noheader,nounits >> $ROOT/logs/gpuutil_${SLURM_JOB_ID}.log; sleep 20; done ) &
SAMPLER=$!

IFS=';' read -ra ITEMS <<< "$CONFIGS"
for it in "${ITEMS[@]}"; do
  [ -z "${it// }" ] && continue
  TAG=$(echo "$it" | cut -d'|' -f1 | xargs)
  REP=$(echo "$it" | cut -d'|' -f2 | xargs)
  ARGS=$(echo "$it" | cut -d'|' -f3-)
  echo "######## CONFIG tag=$TAG rep=$REP args=$ARGS"
  python $FORK/tests_campaign/phess/phess_prod.py \
    --fork $FORK --model $MODEL --pkl $PKL --out-dir $ROOT/out \
    --tag $TAG --replicate $REP --n $NCASE --max-iter $MAXIT $ARGS
  echo "RC_${TAG}_r${REP}=$?"
done

kill $SAMPLER 2>/dev/null
awk -F, '{g+=$1; m+=$3; n++; if($1>mx)mx=$1} END{if(n)printf "GPUUTIL samples=%d mean_gpu=%.1f%% max_gpu=%.0f%% mean_memMB=%.0f\n",n,g/n,mx,m/n}' \
  $ROOT/logs/gpuutil_${SLURM_JOB_ID}.log
echo PHESS_GROUP_DONE
