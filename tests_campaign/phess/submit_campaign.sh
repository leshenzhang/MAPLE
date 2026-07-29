#!/bin/bash
# A2-phess N=100 production campaign. One job per (config, replicate).
# Baseline = bc3335e defaults for the Hessian policy: recalc=8, Bofill updates,
# central FD, full-Hessian mode. Every config sees byte-identical start geometries.
set -u
ROOT=/ibex/user/xiaox/zls/ai-gpu/opt2026/phess
SB=$ROOT/MAPLE/tests_campaign/phess/run_cfg.sbatch
NREP=${NREP:-2}

submit () {  # $1=tag  $2=args
  for r in $(seq 1 $NREP); do
    sbatch --export=ALL,TAG=$1,REP=$r,ARGS="$2",NCASE=100,MAXIT=120 \
           --job-name=phess_$1_r$r $SB
  done
}

# --- reference: current production Hessian policy -------------------------
submit base_r8      "--recalc 8"
# --- longer exact-Hessian interval (more Bofill, fewer recalcs) -----------
submit r16          "--recalc 16"
submit r32          "--recalc 32"
# --- update-only: ONE exact Hessian at iter 1, Bofill for the whole run ---
submit updonly      "--recalc 1000000"
# --- update-format ablation at the same cadence ---------------------------
submit updonly_psb  "--recalc 1000000 --hessian-update psb"
submit updonly_sr1  "--recalc 1000000 --hessian-update sr1"
# --- cheaper exact Hessian: forward FD (3m+1 vs 6m forwards) --------------
submit fwd_r8       "--recalc 8 --fd-mode forward"
submit fwd_updonly  "--recalc 1000000 --fd-mode forward"
# --- adaptive recalc (pysisyphus-style ||g|| trigger) ---------------------
submit adapt        "--recalc 1000000 --adapt"
# --- matrix-free leftmost eigenpair (no full Hessian at all) --------------
submit lobpcg       "--hessian-mode iterative --iter-solver lobpcg --lobpcg-max 6 --warm-start --initial-hessian lindh --ts-inject"
submit lanczos      "--hessian-mode iterative --iter-solver lanczos --lanczos-m 8 --warm-start --initial-hessian lindh --ts-inject"
squeue -u xiaox -o '%.10i %.22j %.8T %R' | grep phess
