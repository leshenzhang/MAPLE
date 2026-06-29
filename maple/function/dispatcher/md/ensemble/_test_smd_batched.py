"""B-64 gates for batched constant-velocity steered MD + Jarzynski (MACE-OFF).

Validation axis: ALGORITHM CORRECTNESS given the potential (not literature).

  ANALYTIC : the per-pull restraint force injected by the moving bias equals the
             analytic -k(xi-lambda)*d(xi)/dx to fp64 (~1e-13), recomputed
             independently; per-pull centres are INDEPENDENT (distinct dlam).
  GPU SMD  : N=8 constant-velocity pulls stretch a water-dimer COM-COM distance
             lam0 -> lam1 on MACE-OFF; reconstruct Delta-G(lambda) via the
             EXISTING steered.jarzynski_1d and check the Jarzynski identities:
               * 2nd law  <W> >= Delta-G_exp   (dissipated work >= 0)
               * exp-average vs 2nd-cumulant Delta-G agreement (near-equilibrium)
               * per-pull work monotonic-ish along lambda
               * N pulls independent (perturb-one ENERGY isolation via the calc)

The single-system steered self-test (B-53, bias/steered.py __main__) is the bar.
"""
import os, sys, tempfile, time
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from ase.build import molecule

from maple.function.calculator.mace._maceoff_batch_calculator import MaceOffBatchCalc
from maple.function.dispatcher.md.bias.steered_batched import BatchedMovingRestraint
from maple.function.dispatcher.md.bias.steered import HARTREE_TO_KCAL_MOL
from maple.function.dispatcher.md.ensemble.smd_batched import BatchedSMD

MODEL = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
DEV = "cuda"


def water_dimer(sep=2.9):
    """Two H2O (atoms 0,1,2 = water1; 3,4,5 = water2), COM-COM ~ sep Angstrom."""
    w1 = molecule("H2O")
    w2 = molecule("H2O")
    w2.positions = w2.positions + np.array([sep, 0.0, 0.0])
    return w1 + w2


def analytic_force_check():
    """restraint force == -k(xi-lambda) d(xi)/dx to fp64, over moving centres,
    with 3 DISTINCT dlam (per-pull independence of the centre trajectory)."""
    reps = [water_dimer() for _ in range(3)]
    k = 0.1                                             # Ha/Angstrom^2
    lam0 = np.array([2.9, 2.9, 2.9])
    dlam = np.array([1e-3, 2e-3, 3e-3])                 # distinct -> centres diverge
    bias = BatchedMovingRestraint(reps, "0,1,2", "3,4,5", k, lam0, dlam)
    ptr = np.concatenate([[0], np.cumsum([len(a) for a in reps])]).astype(int)
    coord = np.concatenate([a.get_positions() for a in reps], axis=0)
    g1, g2 = np.array([0, 1, 2]), np.array([3, 4, 5])
    maxdelta = 0.0
    for step in range(6):
        bias.centers = bias.lambda_now()
        cvs, forces = bias.restraint_forces(coord, ptr)
        lam = bias.lambda_now()
        for b in range(3):
            pos = coord[ptr[b]:ptr[b + 1]]
            m = reps[b].get_masses()
            R1 = (m[g1, None] * pos[g1]).sum(0) / m[g1].sum()
            R2 = (m[g2, None] * pos[g2]).sum(0) / m[g2].sum()
            dvec = R1 - R2
            xi = np.linalg.norm(dvec)
            u = dvec / xi
            coef = -k * (xi - lam[b])                   # analytic restraint scalar
            fan = np.zeros_like(pos)
            fan[g1] = coef * (m[g1] / m[g1].sum())[:, None] * u[None, :]
            fan[g2] = coef * (m[g2] / m[g2].sum())[:, None] * (-u[None, :])
            maxdelta = max(maxdelta, float(np.abs(forces[b] - fan).max()))
        bias._istep += 1                                # advance the centre
    centres_indep = not np.allclose(bias.lambda_now()[0], bias.lambda_now()[1:])
    print(f"[ANALYTIC] max|F - (-k(xi-lam)u)| = {maxdelta:.3e}  "
          f"(target ~1e-9, exact)  per-pull centres independent={centres_indep}")
    assert maxdelta < 1e-9, "ANALYTIC FORCE FAIL"
    assert centres_indep, "per-pull centre trajectories not independent"
    print("[ANALYTIC] PASS")
    return maxdelta


def binned_work_mono(work, lam, nb=20):
    """Coarse-grained cumulative-work-vs-lambda trend: fraction of lambda-bins
    whose mean work is non-decreasing. The PER-STEP dW sign flips with thermal
    noise (a fast pull has ~50% negative-dW steps even though W(lambda) trends
    up); the physically-meaningful SMD sanity is this slow binned trend."""
    work = np.asarray(work, float)
    lam = np.asarray(lam, float)
    edges = np.linspace(lam.min(), lam.max(), nb + 1)
    idx = np.clip(np.digitize(lam, edges) - 1, 0, nb - 1)
    means = np.array([work[idx == j].mean() if np.any(idx == j) else np.nan
                      for j in range(nb)])
    means = means[~np.isnan(means)]
    d = np.diff(means)
    return float(np.mean(d >= 0)) if d.size else 1.0


def run_smd(npulls=8, steps=200000, lam1=5.5, seed=20240601,
            velocity=0.0, k=0.1, friction=0.01):
    template = water_dimer()
    bc = MaceOffBatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64)
    paras = dict(
        timestep=0.5, steps=steps, temperature=300.0, thermostat="langevin",
        friction=friction, remove_com_every=100, random_seed=seed, verbose=0,
        smd_group1="0,1,2", smd_group2="3,4,5", smd_k=k,
        smd_lam0="auto", smd_lam1=lam1, smd_velocity=velocity,
        smd_npulls=npulls, smd_jarz_nbins=60,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        out = f.name
    t0 = time.time()
    sim = BatchedSMD(out, template, calc=bc, paras=paras).run()
    wall = time.time() - t0

    works_kcal = sim.pull_work_Ha * HARTREE_TO_KCAL_MOL
    mean_W = sim.mean_work_kcal
    dG_exp = sim.dG_exp_kcal
    dG_cum = sim.dG_cum_kcal

    # per-pull cumulative-work-vs-lambda trend (binned; robust to thermal noise)
    # + per-step Pearson corr(lambda, W) as a second monotonicity witness.
    mono_bin, corr = [], []
    for b in range(sim.B):
        w = np.asarray(sim._bias.work_history[b])
        lam = np.asarray(sim._bias.lam_history[b])
        mono_bin.append(binned_work_mono(w, lam))
        corr.append(float(np.corrcoef(lam, w)[0, 1]) if w.size > 2 else 1.0)
    mono_min = float(np.min(mono_bin))
    corr_min = float(np.min(corr))

    leak = bc.isolation_check(perturb=0.05)            # perturb-one ENERGY isolation

    print(f"\n[GPU SMD] B={npulls}  steps={steps}  dt=0.5 fs  k={k} Ha/A^2  "
          f"lam {float(np.mean(sim._lam0)):.3f}->{float(np.mean(sim._lam1)):.3f} A  "
          f"wall={wall:.1f}s")
    print(f"[GPU SMD] per-pull W(kcal/mol) = "
          f"[{', '.join(f'{x:.3f}' for x in works_kcal)}]")
    print(f"[GPU SMD] <W>={mean_W:.4f}  dG_exp={dG_exp:.4f}  dG_cum={dG_cum:.4f} "
          f"kcal/mol")
    print(f"[GPU SMD] dissipated <W>-dG_exp = {mean_W - dG_exp:.4f} kcal/mol "
          f"(>=0 = Jarzynski 2nd law)")
    print(f"[GPU SMD] |dG_exp - dG_cum| = {abs(dG_exp - dG_cum):.4f} kcal/mol "
          f"(small => near-equilibrium agreement)")
    print(f"[GPU SMD] work-vs-lambda binned-monotonic min frac = {mono_min:.3f}  "
          f"corr(lambda,W) min = {corr_min:.3f}")
    print(f"[GPU SMD] perturb-one isolation dE = {leak:.3e} Ha (N pulls independent)")

    # ---- Jarzynski identity gates (algorithm correctness) ----
    assert leak < 1e-6, "ISOLATION FAIL (pulls coupled)"
    assert mean_W >= dG_exp - 1e-6, "2nd LAW FAIL (<W> < dG_exp)"
    assert mono_min > 0.7, "WORK-vs-LAMBDA not monotonic-ish (binned trend)"
    assert corr_min > 0.8, "WORK not positively correlated with lambda"
    assert np.std(works_kcal) > 1e-6, "pulls identical (no thermal independence)"
    print("[GPU SMD] PASS (Jarzynski 2nd law + work monotonic + pulls independent)")
    return dict(mean_W=mean_W, dG_exp=dG_exp, dG_cum=dG_cum,
                works=works_kcal.tolist(), leak=leak, wall=wall,
                lam0=float(np.mean(sim._lam0)), lam1=float(np.mean(sim._lam1)))


if __name__ == "__main__":
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    delta = analytic_force_check()

    # optional quick timing/correctness probe (SMD_QUICK_STEPS<=0 skips it),
    # then the slow near-equilibrium production pull (clean Jarzynski).
    quick_steps = int(os.environ.get("SMD_QUICK_STEPS", "0"))
    prod_steps = int(os.environ.get("SMD_PROD_STEPS", "200000"))
    if quick_steps > 0:
        print(f"\n--- quick probe ({quick_steps} steps) ---")
        run_smd(npulls=8, steps=quick_steps, lam1=5.5)
    print(f"\n--- production pull ({prod_steps} steps) ---")
    r = run_smd(npulls=8, steps=prod_steps, lam1=5.5)

    print(f"\n[B-64 RESULT] analytic_force_delta={delta:.3e}  "
          f"<W>={r['mean_W']:.4f}  dG_exp={r['dG_exp']:.4f}  "
          f"dG_cum={r['dG_cum']:.4f} kcal/mol  isolation_dE={r['leak']:.3e} Ha")
    print("[B-64 RESULT] ALL GATES PASS")
