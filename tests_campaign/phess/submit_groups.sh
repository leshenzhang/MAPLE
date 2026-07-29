#!/bin/bash
# A2-phess N=100 campaign, grouped: 4 configs per GPU allocation.
set -u
ROOT=/ibex/user/xiaox/zls/ai-gpu/opt2026/phess
SB=$ROOT/MAPLE/tests_campaign/phess/run_group.sbatch

sbatch --job-name=phess_gA --export=ALL,NCASE=100,MAXIT=120,CONFIGS="\
base_r8|1|--recalc 8 ; base_r8|2|--recalc 8 ; \
r16|1|--recalc 16 ; r16|2|--recalc 16" $SB

sbatch --job-name=phess_gB --export=ALL,NCASE=100,MAXIT=120,CONFIGS="\
updonly|1|--recalc 1000000 ; updonly|2|--recalc 1000000 ; \
quality|1|--recalc 1000000 --quality --quality-tol 0.5 ; \
quality|2|--recalc 1000000 --quality --quality-tol 0.5" $SB

sbatch --job-name=phess_gC --export=ALL,NCASE=100,MAXIT=120,CONFIGS="\
fwd_r8|1|--recalc 8 --fd-mode forward ; fwd_r8|2|--recalc 8 --fd-mode forward ; \
fwd_updonly|1|--recalc 1000000 --fd-mode forward ; \
fwd_updonly|2|--recalc 1000000 --fd-mode forward" $SB

sbatch --job-name=phess_gD --export=ALL,NCASE=100,MAXIT=120,CONFIGS="\
lobpcg_reorth|1|--hessian-mode iterative --iter-solver lobpcg --lobpcg-max 6 --warm-start --reorth --initial-hessian lindh --ts-inject ; \
lobpcg_reorth|2|--hessian-mode iterative --iter-solver lobpcg --lobpcg-max 6 --warm-start --reorth --initial-hessian lindh --ts-inject ; \
lobpcg|1|--hessian-mode iterative --iter-solver lobpcg --lobpcg-max 6 --warm-start --initial-hessian lindh --ts-inject ; \
lanczos|1|--hessian-mode iterative --iter-solver lanczos --lanczos-m 8 --warm-start --initial-hessian lindh --ts-inject" $SB

sbatch --job-name=phess_gE --export=ALL,NCASE=100,MAXIT=120,CONFIGS="\
adapt|1|--recalc 1000000 --adapt ; adapt|2|--recalc 1000000 --adapt ; \
updonly_sr1|1|--recalc 1000000 --hessian-update sr1 ; \
updonly_psb|1|--recalc 1000000 --hessian-update psb" $SB

squeue -u xiaox -o '%.10i %.14j %.8T %.6l %R' | grep phess
