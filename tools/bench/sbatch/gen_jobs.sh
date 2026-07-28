#!/bin/bash
# Generate the 2026-07-28 A4-bench sbatch set (run locally; output = this dir).
# All jobs: 1xA100, no --partition (Slurm routes gpu/gpu1/gpu4), walltime <=2h.
cd "$(dirname "$0")"
SB=/ibex/user/xiaox/zls/ai-gpu/opt2026/bench
PY=/home/wangc0i/miniconda3/envs/cxtorch/bin/python
MODEL=/ibex/user/xiaox/zls/ai-gpu/MAPLE/maple/function/calculator/model/uma-s-1p1.pt
PKL=/ibex/user/xiaox/zls/ai-gpu/dev/testset/ts1x_seed42_100.pkl

hdr() { # $1 name $2 time
cat <<EOF
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=$2
#SBATCH --output=$SB/logs/%x_%j.out
set -u
export PATH=/home/wangc0i/miniconda3/envs/cxtorch/bin:\$PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=$SB/MAPLE
export MAPLE_COMMIT=bc3335e
export MODEL=$MODEL
export PKL=$PKL
OUT=$SB/runs
mkdir -p \$OUT
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
( while true; do echo "[util \$(date +%H:%M:%S)] \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; sleep 120; done ) &
TICKER=\$!
trap "kill \$TICKER 2>/dev/null" EXIT
RM="$PY $SB/MAPLE/tools/bench/run_matrix.py"
EOF
}

# --- smoke: every dispatcher tiny (non-production; tagged smoke) ---
{ hdr ob_smoke 00:40:00; cat <<EOF
\$RM --dispatcher pipeline --backend uma --B 2 --N 4 --reps 1 --n-images 5 --neb-maxiter 20 --outdir \$OUT/smoke --tag smoke
\$RM --dispatcher forward  --backend uma --B 1,8 --reps 1 --iters 3 --outdir \$OUT/smoke --tag smoke
\$RM --dispatcher hessian  --backend uma --B 2 --reps 1 --hess-iters 1 --outdir \$OUT/smoke --tag smoke
\$RM --dispatcher parity   --backend uma --reps 1 --outdir \$OUT/smoke --tag smoke
export TOY_MACE=/ibex/user/xiaox/zls/ai-maple-md/MAPLE/worktrees/integ/vfy_b78/out_plumed/toy_maceomol.pt
export MACEOFF_RAW=/ibex/user/xiaox/zls/ai-gpu/dev/d2/MACE-OFF23_small.model
\$RM --dispatcher counter  --backend uma --reps 1 --outdir \$OUT/smoke --tag smoke
\$RM --dispatcher autoneb  --backend uma --arm batched --rxn-count 2 --B-single 2 --aneb-maxiter 10 --reps 1 --outdir \$OUT/smoke --tag smoke
\$RM --dispatcher autoneb  --backend uma --arm serial  --rxn-count 2 --serial-budget-s 240 --aneb-maxiter 10 --reps 1 --outdir \$OUT/smoke --tag smoke
echo SMOKE_DONE rc=\$?
EOF
} > ob_smoke.sbatch

# --- pipeline baseline: one job per (B, rep) ---
for B in 1 16 64; do for r in 1 2; do
  T=02:00:00
  { hdr ob_p${B}r${r} $T; cat <<EOF
\$RM --dispatcher pipeline --backend uma --B $B --N 100 --reps 1 --rep-offset $r --outdir \$OUT --tag base
echo PIPE_DONE rc=\$?
EOF
  } > ob_p${B}_r${r}.sbatch
done; done

# --- micro: raw forward sweep + hessian FD/autograd + parity gate self-test ---
{ hdr ob_micro 02:00:00; cat <<EOF
\$RM --dispatcher forward --backend uma --B 1,8,16,32,64,128 --reps 2 --iters 30 --outdir \$OUT --tag base
\$RM --dispatcher hessian --backend uma --B 1,16,64 --reps 2 --hess-iters 3 --hess-modes numerical,autograd --outdir \$OUT --tag base
\$RM --dispatcher parity  --backend uma --reps 1 --outdir \$OUT --tag base
# FIX-2 emission gate: pin GradCounter against known answers on every backend
export TOY_MACE=/ibex/user/xiaox/zls/ai-maple-md/MAPLE/worktrees/integ/vfy_b78/out_plumed/toy_maceomol.pt
export MACEOFF_RAW=/ibex/user/xiaox/zls/ai-gpu/dev/d2/MACE-OFF23_small.model
\$RM --dispatcher counter --backend uma --counter-backends uma,mace_traced,mace_autograd --reps 1 --outdir \$OUT --tag base
echo MICRO_DONE rc=\$?
EOF
} > ob_micro.sbatch

# --- B3 autoneb: batched arm (2 reps, one job) ---
{ hdr ob_b3bat 02:00:00; cat <<EOF
\$RM --dispatcher autoneb --backend uma --arm batched --rxn-start 0 --rxn-count 16 --B-single 8 --reps 2 --outdir \$OUT --tag b3
echo B3BAT_DONE rc=\$?
EOF
} > ob_b3_bat.sbatch

# --- B3 autoneb: serial arm split 2 halves x 2 reps, per-rxn budget 720 s ---
for h in 1 2; do for r in 1 2; do
  S=$(( (h-1)*8 ))
  { hdr ob_b3s${h}r${r} 02:00:00; cat <<EOF
\$RM --dispatcher autoneb --backend uma --arm serial --rxn-start $S --rxn-count 8 --serial-budget-s 720 --reps 1 --rep-offset $r --outdir \$OUT --tag b3
echo B3SER_DONE rc=\$?
EOF
  } > ob_b3_ser_h${h}_r${r}.sbatch
done; done

# --- B5: MACE traced forward scaling (a3_mace.py verbatim, 2 replicates) ---
{ hdr ob_b5 01:00:00; cat <<EOF
export MOLS=$PKL
export TOY_MACE=/ibex/user/xiaox/zls/ai-maple-md/MAPLE/worktrees/integ/vfy_b78/out_plumed/toy_maceomol.pt
export ZOO_DIR=/ibex/user/xiaox/zls/ai-maple-gpu/zoo_models
export MACEOFF_RAW=/ibex/user/xiaox/zls/ai-gpu/dev/d2/MACE-OFF23_small.model
for r in 1 2; do
  OUT=\$OUT/b5_mace_a100_r\${r}.json $PY /ibex/user/xiaox/zls/ai-maple-gpu/verify_2026-07-14/a3_mace.py
  echo "rc_b5_r\${r}=\$?"
done
echo B5_DONE
EOF
} > ob_b5.sbatch

ls -la *.sbatch
