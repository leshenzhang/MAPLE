#!/bin/bash
#SBATCH --job-name=npt_o1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/ibex/user/xiaox/zls/ai-maple-md/MAPLE/worktrees/npt-opt/o1_validation/npt_o1_%j.log

set -e
ROOT=/ibex/user/xiaox/zls/ai-maple-md
NEW=$ROOT/MAPLE/worktrees/npt-opt               # feat/md-npt-opt (1-forward + stress fix)
OLD=$ROOT/MAPLE/worktrees/batched-core          # feat/md-batched-core (2-forward baseline)
PY=$ROOT/envs/plumed/bin/python
SCRIPT=$NEW/o1_validation/npt_o1_validate.py
ENS=$NEW/maple/function/dispatcher/md/ensemble

# UTF-8 so _log_parameters' tau/gamma symbols write on ascii-locale compute nodes.
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
# Everything runs inside the NEW worktree -> OLD and NEW share the SAME (fixed)
# calculator adapter + identical integrator/barostat/utils; only npt.py differs.
export PYTHONPATH=$NEW
export RUNDIR=$NEW/o1_validation/run_${SLURM_JOB_ID}
mkdir -p $RUNDIR
# 2 fs + H-bond constraints (RATTLE) = 4x ps/step; exercises the harder
# constrained single-forward path. Forward ~215 ms (float64, no cueq) ->
# OLD(2fwd) ~0.4 ns/day, NEW(1fwd) ~0.8 ns/day. tau_p=500 fs decorrelates the
# volume ~4x faster (converged <V> in far fewer steps) AND amplifies any lag
# bias ~4x (conservative test). Sized to ~3 h within the 4 h wall.
export TIMESTEP_FS=2.0
export CONSTRAINTS=h-bonds
export TAU_P=500
export EQUIL_STEPS=7500      # 15 ps equilibration (~30 tau_p -> volume plateaued)
export PROD_STEPS=12500      # 25 ps production each for OLD and NEW
export WARMUP_FRAC=0.3       # discard first ~7.5 ps of production as further equilibration
export SMOKE_STEPS=100
export CALC=${CALC:-generic}     # 'generic' = mace-off-generic (production path); 'selfcontained' = fallback

# Drop the 2-forward baseline npt.py into the package as npt_baseline so the OLD
# control runs in this worktree through the SAME fixed adapter. Gitignored.
cp $OLD/maple/function/dispatcher/md/ensemble/npt.py $ENS/npt_baseline.py

echo "HOST $(hostname)  START $(date)  RUNDIR=$RUNDIR  CALC=$CALC"
nvidia-smi -L || true

# 0) runtime-confirm all 3 thermostat paths are single-forward on the NEW code
echo "=== SMOKE3 (NEW code: v-rescale / langevin / constrained) ==="
$PY $SCRIPT smoke3

# 1) equilibrate with the OLD/baseline code (start from the baseline ensemble)
echo "=== EQUIL (baseline npt_baseline) ==="
$PY $SCRIPT equil npt_baseline

# 2) OLD production (2-forward baseline)
echo "=== PROD old (2-forward, npt_baseline) ==="
$PY $SCRIPT prod old npt_baseline

# 3) NEW production (1-forward)
echo "=== PROD new (1-forward, npt) ==="
$PY $SCRIPT prod new npt

# 4) compare
echo "=== COMPARE ==="
$PY $SCRIPT compare
echo "END $(date)"
