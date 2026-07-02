"""
Node-energy REST2 (Hamiltonian replica-exchange) correctness gates (a100, MACE-OFF fp64).

Systematic ALGORITHM-CORRECTNESS validation of the node-energy solute-tempering
scheme (E_m = E_full + (lambda-1)*sum_{solute} node_energy):

  G1  LAMBDA=1 IDENTITY   -- at lambda=1, E_m == E_full and F_m == F_full bit-for-bit
                            (<1e-12), and the all-lambda=1 REST2 trajectory == plain
                            BatchedNVT to <1e-12 (the node-energy machinery reduces to
                            plain NVT).
  G2  FD CONSERVATIVE F   -- at lambda!=1, F_m == -grad E_m by central finite diff of
                            E_m wrt EVERY coordinate (<1e-6). REJECTS the naive
                            "scale the solute forces" scheme (shown to disagree with FD
                            by orders of magnitude, incl. on SOLVENT atoms that feel
                            -grad S through message passing).
  G3  EQUAL-LAMBDA ACCEPT -- two replicas at the SAME lambda: exchange Delta==0 exactly
                            => acceptance probability == 1 (real _attempt_swaps rate=1).
  G4  DETAILED BALANCE    -- (a) toy exact-Gibbs sampler driven by the REAL
                            REST2._attempt_swaps: a tagged walker's lambda-occupancy is
                            UNIFORM over the ladder (flat histogram). (b) short real
                            REST2 MD: lambda set stays a permutation of the ladder every
                            step, all boundaries accept, tagged walker roams the ladder.

Exit 0 iff ALL gates pass.
"""
import os
import sys
import tempfile

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
from ase.build import molecule

from maple.function.calculator.mace._maceoff_batch_calculator import MaceOffBatchCalc
from maple.function.dispatcher.md.ensemble.nvt_batched import BatchedNVT
from maple.function.dispatcher.md.ensemble.hremd_rest2 import REST2, REST2Params

MODEL = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
DEV = "cuda"


def solvated():
    """Ethanol (solute, 9 atoms) + 2 waters (solvent, 6 atoms) within the MACE-OFF
    cutoff so the solute node energies couple to the solvent geometry (=> the tempered
    force has nonzero SOLVENT components -- the crux of G2). Returns (Atoms, solute_idx)."""
    eth = molecule("CH3CH2OH")
    eth.positions -= eth.get_center_of_mass()
    w1 = molecule("H2O"); w1.positions += np.array([3.1, 0.2, 0.1])
    w2 = molecule("H2O"); w2.positions += np.array([-3.0, 0.4, -0.2])
    sysm = eth + w1 + w2                       # 9 + 3 + 3 = 15 atoms
    solute_idx = list(range(len(eth)))         # ethanol = solute
    return sysm, solute_idx


def _tmp():
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        return f.name


# =========================================================== G1: lambda=1 identity
def gate_G1():
    print("\n" + "#" * 72 + "\n[G1] LAMBDA=1 IDENTITY (E_m==E_full, F_m==F_full, traj==BatchedNVT)\n" + "#" * 72)
    sysm, sol = solvated()
    bc = MaceOffBatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64)

    # --- G1a: single-config identity through the FULL node-energy path -------------
    bc.prepare([sysm.copy(), sysm.copy()])
    N = bc.N_atoms
    ptr = np.concatenate(([0], np.cumsum(bc.n_b))).astype(int)
    smask = np.zeros(N, dtype=bool)
    for b in range(bc.B):
        smask[ptr[b] + np.asarray(sol)] = True
    lam1 = np.ones(bc.B)
    Em, Fm, S = bc.get_ef_rest_gpu(lam1, smask)
    Ef, Ff = bc.get_ef_gpu()
    dE = float((Em - Ef).abs().max().item())
    dF = float((Fm - Ff).abs().max().item())
    print(f"[G1a] max|E_m - E_full| = {dE:.3e} Ha   max|F_m - F_full| = {dF:.3e} Ha/A   "
          f"(solute node sum S={S.detach().cpu().numpy()})")
    g1a = (dE < 1e-12) and (dF < 1e-12)

    # --- G1b: all-lambda=1 REST2 trajectory == plain BatchedNVT (bit-exact) --------
    steps, seed, Nr, T0 = 80, 43, 4, 320.0
    paras = dict(timestep=0.5, steps=steps, thermostat="langevin", friction=0.02,
                 temperature=T0, remove_com_every=100, random_seed=seed, verbose=0)
    rp = dict(paras); rp.update(n_replicas=Nr, lambda_ladder=[1.0] * Nr,
                                solute_indices=sol, exchange_every=20)
    sim = REST2(_tmp(), sysm.copy(), calc=bc, paras=rp).run()
    pos_r2 = bc.coord.detach().cpu().numpy().copy()
    v_r2 = sim.v.detach().cpu().numpy().copy()

    reps = [sysm.copy() for _ in range(Nr)]
    bn = BatchedNVT(_tmp(), reps, calc=bc, paras=dict(paras)).run()
    pos_bn = bc.coord.detach().cpu().numpy().copy()
    v_bn = bn.v.detach().cpu().numpy().copy()

    dpos = float(np.max(np.abs(pos_r2 - pos_bn)))
    dvel = float(np.max(np.abs(v_r2 - v_bn)))
    print(f"[G1b] all-lambda=1 REST2 vs BatchedNVT: steps={steps} N={Nr} seed={seed}  "
          f"max|dpos|={dpos:.3e} A  max|dvel|={dvel:.3e} au")
    g1b = (dpos < 1e-12) and (dvel < 1e-12)
    ok = g1a and g1b
    print(f"[G1] {'PASS' if ok else 'FAIL'}  (identity={g1a}, trajectory={g1b})")
    return ok, dict(dE=dE, dF=dF, dpos=dpos, dvel=dvel)


# ================================================= G2: FD conservative force lambda!=1
def gate_G2(h=1e-4):
    print("\n" + "#" * 72 + "\n[G2] FD CONSERVATIVE FORCE at lambda!=1 (rejects force-rescale scheme)\n" + "#" * 72)
    sysm, sol = solvated()
    bc = MaceOffBatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64)
    bc.prepare([sysm.copy(), sysm.copy()])
    N = bc.N_atoms
    ptr = np.concatenate(([0], np.cumsum(bc.n_b))).astype(int)
    sol_arr = np.asarray(sol)
    smask = np.zeros(N, dtype=bool)
    for b in range(bc.B):
        smask[ptr[b] + sol_arr] = True
    lam = np.array([0.5, 0.7])                                   # lambda != 1 for both

    Em0, Fm0, S0 = bc.get_ef_rest_gpu(lam, smask)
    Fm0 = Fm0.detach().cpu().numpy()                            # (B, nmax_dof) analytic tempered
    _, Ff = bc.get_ef_gpu()
    Ff = Ff.detach().cpu().numpy()                             # (B, nmax_dof) untempered full
    coord0 = bc.coord.clone()
    nmax = bc.nmax_dof

    b = 0                                                       # FD replica 0 (lambda=0.5)
    natoms_b = int(bc.n_b[b])
    solute_local = set(sol_arr.tolist())
    err_all, err_solute, err_solvent = 0.0, 0.0, 0.0
    naive_err_all, naive_err_solvent = 0.0, 0.0
    for a in range(natoms_b):
        g = ptr[b] + a
        for c in range(3):
            with torch.no_grad():
                bc.coord[g, c] = coord0[g, c] + h
            Ep = float(bc.get_ef_rest_gpu(lam, smask)[0].detach().cpu().numpy()[b])
            with torch.no_grad():
                bc.coord[g, c] = coord0[g, c] - h
            Emn = float(bc.get_ef_rest_gpu(lam, smask)[0].detach().cpu().numpy()[b])
            with torch.no_grad():
                bc.coord[g, c] = coord0[g, c]
            f_fd = -(Ep - Emn) / (2.0 * h)                     # Ha/A
            f_an = Fm0[b, 3 * a + c]                            # analytic tempered force
            # naive "scale solute forces": F_full with solute cols scaled by lambda_b.
            f_naive = (lam[b] * Ff[b, 3 * a + c]) if a in solute_local else Ff[b, 3 * a + c]
            e = abs(f_an - f_fd)
            en = abs(f_naive - f_fd)
            err_all = max(err_all, e)
            naive_err_all = max(naive_err_all, en)
            if a in solute_local:
                err_solute = max(err_solute, e)
            else:
                err_solvent = max(err_solvent, e)
                naive_err_solvent = max(naive_err_solvent, en)
    print(f"[G2] central-diff h={h} A, replica {b} (lambda={lam[b]}), {natoms_b} atoms")
    print(f"     max|F_m - F_FD|        all={err_all:.3e}  solute={err_solute:.3e}  "
          f"solvent={err_solvent:.3e}  Ha/A")
    print(f"     max|F_naive - F_FD|    all={naive_err_all:.3e}  solvent={naive_err_solvent:.3e}  "
          f"Ha/A  (naive force-rescale)")
    ok_true = err_all < 1e-6
    naive_rejected = naive_err_all > 1e-3                       # naive disagrees w/ FD by >>
    print(f"[G2] {'PASS' if (ok_true and naive_rejected) else 'FAIL'}  "
          f"(true-force FD<1e-6={ok_true}; naive scheme rejected={naive_rejected}, "
          f"ratio={naive_err_all/max(err_all,1e-30):.1e}x worse)")
    return (ok_true and naive_rejected), dict(err_all=err_all, err_solvent=err_solvent,
                                              naive_err_all=naive_err_all)


# ============================================== G3: equal-lambda exchange accept == 1
def gate_G3():
    print("\n" + "#" * 72 + "\n[G3] EQUAL-LAMBDA EXCHANGE ACCEPTANCE == 1 (Delta==0)\n" + "#" * 72)
    beta0 = 1.0 / (3.16681e-6 * 320.0)
    # (i) analytic: Delta == 0 for equal lambda, ANY S_a, S_b.
    deltas = []
    for lam in (1.0, 0.8, 0.5, 0.3):
        for Sa, Sb in [(-12.3, 7.1), (0.0, 0.0), (5.0, -5.0), (100.0, -100.0)]:
            deltas.append(abs(REST2._rest2_delta(lam, lam, Sa, Sb, beta0)))
    dmax = max(deltas)
    print(f"[G3i] max|Delta(lambda_a==lambda_b)| over 16 (lambda,S) cases = {dmax:.3e}")

    # (ii) real _attempt_swaps with a degenerate ladder [0.8,0.8]: every attempt accepts.
    obj = REST2.__new__(REST2)
    obj.B = 2
    obj.beta0 = beta0
    obj.lambdas = np.array([0.8, 0.8])
    obj._swap_rng = np.random.default_rng(7)
    obj._n_attempt = np.zeros(1, dtype=np.int64)
    obj._n_accept = np.zeros(1, dtype=np.int64)
    obj._swap_round = 0
    rng = np.random.default_rng(11)
    for _ in range(5000):
        obj._last_S = rng.normal(0.0, 50.0, size=2)             # arbitrary differing S
        obj._attempt_swaps()
    rate = obj._n_accept[0] / max(1, obj._n_attempt[0])
    print(f"[G3ii] real _attempt_swaps equal-lambda[0.8,0.8]: attempts={int(obj._n_attempt[0])} "
          f"accepts={int(obj._n_accept[0])} rate={rate:.6f}")
    ok = (dmax < 1e-30) and (rate == 1.0)
    print(f"[G3] {'PASS' if ok else 'FAIL'}  (Delta==0 exact={dmax < 1e-30}, accept-rate==1={rate==1.0})")
    return ok, dict(dmax=dmax, rate=rate)


# ================================================ G4a: detailed-balance toy (exact Gibbs)
def gate_G4a(n_sweeps=200000):
    print("\n" + "#" * 72 + "\n[G4a] DETAILED BALANCE -- toy exact-Gibbs + REAL _attempt_swaps (flat histogram)\n" + "#" * 72)
    M = 4
    beta0, k = 1.0, 1.0
    lam_rungs = np.array([1.0, 0.7, 0.5, 0.3])                  # descending ladder
    rungs_sorted = np.sort(lam_rungs)                          # ascending, for occupancy bins
    sigma = (1.0 / (beta0 * k)) ** 0.5

    obj = REST2.__new__(REST2)
    obj.B = M
    obj.beta0 = beta0
    obj.lambdas = lam_rungs.copy()                             # walker b starts at rung b
    obj._swap_rng = np.random.default_rng(2024)
    obj._n_attempt = np.zeros(M - 1, dtype=np.int64)
    obj._n_accept = np.zeros(M - 1, dtype=np.int64)
    obj._swap_round = 0

    rng = np.random.default_rng(99)
    tag = 0
    occ = np.zeros(M)
    for _ in range(n_sweeps):
        # exact within-replica canonical sampling of U_m(x)=0.5 k x^2 + (lambda_m-1) x
        # => x ~ Normal(mean=(1-lambda_m)/k, var=1/(beta0 k)); S(x)=x (solute=whole).
        means = (1.0 - obj.lambdas) / k
        x = rng.normal(means, sigma)
        obj._last_S = x
        obj._attempt_swaps()
        occ[int(np.argmin(np.abs(rungs_sorted - obj.lambdas[tag])))] += 1
    occ /= n_sweeps
    flat = float(np.max(np.abs(occ - 1.0 / M)))
    rates = obj._n_accept / np.maximum(1, obj._n_attempt)
    print(f"[G4a] sweeps={n_sweeps} M={M} ladder={lam_rungs}")
    print(f"      tag-0 lambda occupancy = {np.round(occ, 4)}  (uniform=1/{M}={1.0/M:.4f})")
    print(f"      flatness max|occ-1/M| = {flat:.4f}   per-boundary accept rate = {np.round(rates,3)}")
    ok = (flat < 0.02) and np.all(rates > 0.05)
    print(f"[G4a] {'PASS' if ok else 'FAIL'}  (flat<0.02={flat < 0.02}, all boundaries mixing={np.all(rates>0.05)})")
    return ok, dict(flat=flat, occ=occ.tolist(), rates=rates.tolist())


# ================================================ G4b: detailed-balance real MD sanity
def gate_G4b(steps=4000, exchange_every=20):
    print("\n" + "#" * 72 + "\n[G4b] DETAILED BALANCE -- real REST2 MD (permutation-invariant + roaming)\n" + "#" * 72)
    sysm, sol = solvated()
    bc = MaceOffBatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64)
    Nr, T0 = 4, 320.0
    paras = dict(timestep=0.5, steps=steps, thermostat="langevin", friction=0.02,
                 temperature=T0, remove_com_every=100, random_seed=17, verbose=0,
                 n_replicas=Nr, lambda_min=0.4, lambda_max=1.0, solute_indices=sol,
                 exchange_every=exchange_every)
    sim = REST2(_tmp(), sysm.copy(), calc=bc, paras=paras).run()

    ladder_set = set(np.round(sim.lambda_ladder, 8).tolist())
    lam_hist = np.asarray(sim._hist_lambdas)                    # (nstep, B)
    perm_ok = all(set(np.round(lam_hist[i], 8).tolist()) == ladder_set for i in range(lam_hist.shape[0]))
    rates = [e['rate'] for e in sim.exchange_stats]
    all_accept = all((r is not None and r > 0.0) for r in rates)
    tag0 = sim.results[0]['lambda_residency']
    n_rungs_visited = int(np.sum(np.array(list(tag0.values())) > 0))
    leak = bc.isolation_check(perturb=0.1)
    print(f"[G4b] ladder={np.round(sim.lambda_ladder,4)}  steps={steps}  exchange_every={exchange_every}")
    print(f"      per-boundary accept rate = {[round(r,3) for r in rates]}")
    print(f"      lambda-set == ladder every recorded step: {perm_ok}")
    print(f"      tag-0 residency = { {round(k,3): round(v,3) for k,v in tag0.items()} } "
          f"-> rungs visited = {n_rungs_visited}/{Nr}")
    print(f"      tag-0 flatness dev = {sim.tag0_flatness:.3f}   cross-replica isolation leak = {leak:.2e} Ha")
    ok = perm_ok and all_accept and (n_rungs_visited >= 2) and (leak < 1e-6)
    print(f"[G4b] {'PASS' if ok else 'FAIL'}  (perm={perm_ok}, all-boundaries-accept={all_accept}, "
          f"roams>=2 rungs={n_rungs_visited>=2}, isolated={leak<1e-6})")
    return ok, dict(rates=rates, n_rungs_visited=n_rungs_visited, leak=leak)


if __name__ == "__main__":
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    results = {}
    ok1, r1 = gate_G1(); results["G1"] = ok1
    ok2, r2 = gate_G2(); results["G2"] = ok2
    ok3, r3 = gate_G3(); results["G3"] = ok3
    ok4a, r4a = gate_G4a(); results["G4a"] = ok4a
    ok4b, r4b = gate_G4b(); results["G4b"] = ok4b

    print("\n" + "=" * 72)
    print("REST2 GATE SUMMARY")
    for kk, vv in results.items():
        print(f"  {kk}: {'PASS' if vv else 'FAIL'}")
    all_ok = all(results.values())
    print("=" * 72)
    print(f"[REST2 RESULT] {'ALL GATES PASS' if all_ok else 'SOME GATES FAILED'}")
    sys.exit(0 if all_ok else 1)
