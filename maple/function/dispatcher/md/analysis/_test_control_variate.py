"""
Control-variate observable estimator (FL-1) gates. Gates 1-3 are pure numpy
(no MLIP model / torch needed); gate 4 rides a toy batched run if torch is
present, else AUTO-SKIPS.

  GATE 1  UNBIASEDNESS       : the CV correction  c*(mean(C) - E[C])  is a
          zero-mean term.  (a) ALGEBRAIC: with C_mean set to the sample mean the
          correction is identically zero -> estimate == mean(O) to < 1e-12.
          (b) STATISTICAL: over many independent batches with the TRUE E[C],
          E[O_CV] == E[mean O] (Monte-Carlo mean within a few standard errors).
  GATE 2  VARIANCE REDUCTION : on a synthetic correlated Gaussian batch with a
          KNOWN correlation rho, the returned var_reduction_factor matches the
          analytic  1 - rho^2, and the EMPIRICAL Var(O_CV)/Var(mean O) over many
          batches matches it too; strictly < 1 for rho != 0.
  GATE 3  DEGENERATE (rho->0): an uncorrelated control gives c -> 0, factor -> 1,
          and O_CV == naive mean (even with a known C_mean).
  GATE 4  TRAJECTORY SMOKE   : pull O (a bond-length order parameter) + C (the
          shared potential energy) from a small BatchedNVT run over a toy
          double-well dimer, and confirm a FINITE estimate with
          var_reduction_factor <= 1. AUTO-SKIP if torch/model is unavailable.

The gate-4 fixture ``_DoubleWellDimer`` mirrors the one in ensemble/_test_we.py
(a batch-isolated, MLIP-shaped analytic calc) so it needs only torch (CPU ok),
no MACE model file.
"""
import numpy as np

from maple.function.dispatcher.md.analysis.control_variate import (
    control_variate_estimate,
    batched_O_C,
    batched_control_variate,
)


# ------------------------------------------------------------------ generators
def _correlated_normal(rho, n, mu_O=0.0, mu_C=0.0, sd_O=1.0, sd_C=1.0, rng=None):
    """Draw (O, C) ~ bivariate normal with corr(O, C) = rho (population-exact)."""
    rng = rng or np.random.default_rng(0)
    z1 = rng.standard_normal(n)                       # drives C
    z2 = rng.standard_normal(n)                       # independent part of O
    C = mu_C + sd_C * z1
    O = mu_O + sd_O * (rho * z1 + np.sqrt(max(0.0, 1.0 - rho * rho)) * z2)
    return O, C


# ================================================================= GATE 1
def run_gate1_unbiased(seed=1):
    rng = np.random.default_rng(seed)
    mu_O, mu_C, rho = 3.7, -1.2, 0.8

    # (a) ALGEBRAIC: C_mean = sample mean -> correction identically 0.
    O, C = _correlated_normal(rho, 5000, mu_O, mu_C, 2.0, 0.5, rng)
    est_sm, _se, _f, c = control_variate_estimate(O, C, C_mean=float(C.mean()))
    resid = abs(est_sm - float(O.mean()))
    print(f"[GATE1a algebraic] C_mean=sample-mean -> |estimate - mean(O)| = {resid:.3e}  (c={c:.4f})")
    assert resid < 1e-12, "CV correction not zero when C_mean == sample mean"

    # C_mean=None default must reproduce the same zero-correction estimate.
    est_none, _, _, _ = control_variate_estimate(O, C, C_mean=None)
    assert abs(est_none - float(O.mean())) < 1e-12, "C_mean=None must equal naive mean"

    # (b) STATISTICAL: many independent batches, TRUE E[C] known -> E[O_CV] == mu_O.
    B, trials = 24, 20000
    cv_means, naive_means = np.empty(trials), np.empty(trials)
    for t in range(trials):
        Ot, Ct = _correlated_normal(rho, B, mu_O, mu_C, 2.0, 0.5, rng)
        est, _se, _f, _c = control_variate_estimate(Ot, Ct, C_mean=mu_C)
        cv_means[t] = est
        naive_means[t] = Ot.mean()
    bias_cv = cv_means.mean() - mu_O
    bias_naive = naive_means.mean() - mu_O
    se_cv = cv_means.std(ddof=1) / np.sqrt(trials)
    print(f"[GATE1b statistical] E[O_CV]-mu_O = {bias_cv:+.3e}  (+/-{3*se_cv:.1e}, 3se)   "
          f"E[naive]-mu_O = {bias_naive:+.3e}")
    assert abs(bias_cv) < 4.0 * se_cv, "CV estimator shows a statistically significant bias"
    print("[GATE1] PASS  (correction is zero-mean; estimator unbiased)")
    return resid, bias_cv


# ================================================================= GATE 2
def run_gate2_variance_reduction(seed=2):
    rng = np.random.default_rng(seed)

    for rho in (0.3, 0.6, 0.9):
        analytic = 1.0 - rho * rho

        # returned factor on ONE large sample -> rho_hat -> rho.
        O, C = _correlated_normal(rho, 400000, 1.0, 2.0, 1.3, 0.7, rng)
        _est, stderr, factor, c = control_variate_estimate(O, C, C_mean=2.0)
        err = abs(factor - analytic)
        print(f"[GATE2 rho={rho:.2f}] returned factor={factor:.4f}  analytic(1-rho^2)={analytic:.4f}  "
              f"|d|={err:.3e}  c={c:.4f}")
        assert factor < 1.0, "factor must be < 1 for rho != 0"
        assert err < 5e-3, "returned var_reduction_factor != analytic 1 - rho^2"

        # EMPIRICAL variance ratio over many batches (true c, true C_mean).
        B, trials = 32, 12000
        c_true = rho * (1.3 / 0.7)                    # rho * sd_O / sd_C
        cv, naive = np.empty(trials), np.empty(trials)
        for t in range(trials):
            Ot, Ct = _correlated_normal(rho, B, 1.0, 2.0, 1.3, 0.7, rng)
            cv[t] = Ot.mean() - c_true * (Ct.mean() - 2.0)
            naive[t] = Ot.mean()
        emp = cv.var(ddof=1) / naive.var(ddof=1)
        print(f"[GATE2 rho={rho:.2f}] empirical Var(O_CV)/Var(mean) = {emp:.4f}  (analytic {analytic:.4f})")
        assert emp < 1.0 and abs(emp - analytic) < 0.06, "empirical variance ratio off analytic"

    # stderr must equal naive stderr * sqrt(factor).
    O, C = _correlated_normal(0.7, 5000, 0.0, 0.0, 1.0, 1.0, rng)
    _e, se_cv, fac, _c = control_variate_estimate(O, C, C_mean=0.0)
    se_naive = O.std(ddof=1) / np.sqrt(O.size)
    assert abs(se_cv - se_naive * np.sqrt(fac)) < 1e-10, "stderr != naive_stderr*sqrt(factor)"
    print("[GATE2] PASS  (factor == 1 - rho^2 analytically + empirically; stderr consistent)")
    return True


# ================================================================= GATE 3
def run_gate3_degenerate(seed=3):
    rng = np.random.default_rng(seed)
    # O and C independent, EQUAL scale -> rho ~ 0 and c ~ rho_hat ~ 1/sqrt(N).
    O = 5.0 + 1.0 * rng.standard_normal(200000)
    C = -3.0 + 1.0 * rng.standard_normal(200000)     # uncorrelated with O
    est, _se, factor, c = control_variate_estimate(O, C, C_mean=-3.0)
    dmean = abs(est - float(O.mean()))
    print(f"[GATE3 rho->0] c={c:.3e}  factor={factor:.6f}  |O_CV - mean(O)|={dmean:.3e}")
    assert abs(c) < 5e-3, "c did not collapse toward 0 for an uncorrelated control"
    assert factor > 0.999, "factor did not approach 1 for rho -> 0"
    assert dmean < 5e-3, "O_CV drifted from the naive mean for an uncorrelated control"
    print("[GATE3] PASS  (uncorrelated control -> c->0, factor->1, O_CV == naive mean)")
    return c, factor


# ================================================================= GATE 4
def run_gate4_trajectory_smoke(seed=42):
    try:
        import tempfile
        import torch
        torch.set_default_dtype(torch.float64)
        from ase import Atoms
        from maple.function.calculator.batch_calculator_base import BatchCalcABC
        from maple.function.dispatcher.md.ensemble.nvt_batched import BatchedNVT
        from maple.function.dispatcher.md.utils import KELVIN_TO_HARTREE
    except Exception as e:                            # torch / engine absent
        print(f"[GATE4 trajectory-smoke] SKIP (torch/engine unavailable: {e})")
        return None

    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    class _DoubleWellDimer(BatchCalcABC):
        """B independent 2-atom molecules; bond r sits in a symmetric double well
        E(r) = h*(((r-rmid)/w)^2 - 1)^2 (Ha). Mirrors ensemble/_test_we.py."""
        MODEL_NAMES = ("_cv_doublewell_test",)
        MODEL_ENERGY_UNIT = "hartree"
        MODEL_DTYPE = torch.float64
        SUPPORTED_HESSIAN_MODES = ("numerical",)

        def __init__(self, rmid, w, h, device="cpu", dtype=torch.float64):
            super().__init__(device=device, dtype=dtype)
            self.rmid, self.w, self.h = float(rmid), float(w), float(h)

        def _forward(self, coord, need_graph):
            P0, P1 = coord[0::2], coord[1::2]
            d = P0 - P1
            r = torch.linalg.norm(d, dim=1).clamp_min(1e-8)
            s = ((r - self.rmid) / self.w) ** 2 - 1.0
            E = self.h * s * s
            dEdr = 4.0 * self.h / (self.w ** 2) * s * (r - self.rmid)
            u = d / r[:, None]
            F_all = torch.empty((2 * self._atoms_B, 3), dtype=coord.dtype, device=coord.device)
            F_all[0::2] = -dEdr[:, None] * u
            F_all[1::2] = dEdr[:, None] * u
            return E, F_all, None

    def _dimer(r):
        return Atoms("CC", positions=[[0.0, 0.0, 0.0], [float(r), 0.0, 0.0]])

    def _tmp():
        with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
            return f.name

    T_K, R1, R2 = 300.0, 1.3, 2.1
    rmid, whalf = 0.5 * (R1 + R2), 0.5 * (R2 - R1)
    h = 3.5 * KELVIN_TO_HARTREE * T_K
    calc = _DoubleWellDimer(rmid, whalf, h, device=DEV, dtype=torch.float64)

    B = 8
    starts = [_dimer(R1 + 0.05 * b) for b in range(B)]     # spread across the well
    run = BatchedNVT(_tmp(), [a.copy() for a in starts], calc=calc,
                     paras=dict(timestep=0.5, temperature=T_K, thermostat="langevin",
                                friction=0.02, steps=40, random_seed=seed, verbose=0)).run()

    # O = bond length (order parameter); C = shared potential energy (zero extra fwd).
    bond = lambda p: float(np.linalg.norm(p[0] - p[1]))
    O, C = batched_O_C(run, bond, control="energy")
    est, stderr, factor, c = control_variate_estimate(O, C, C_mean=None)
    est2, stderr2, factor2, c2 = batched_control_variate(run, bond, control="energy")

    print(f"[GATE4 trajectory-smoke] B={B}  O(bond)~[{O.min():.3f},{O.max():.3f}] A  "
          f"C(energy)~[{C.min():.3e},{C.max():.3e}] Ha")
    print(f"[GATE4] estimate={est:.6f} A  stderr={stderr:.3e}  factor={factor:.4f}  c={c:.4e}")
    assert O.shape == (B,) and C.shape == (B,), "batched_O_C shape mismatch"
    assert np.isfinite(est) and np.isfinite(stderr) and np.isfinite(factor), "non-finite estimate"
    assert 0.0 <= factor <= 1.0, "var_reduction_factor out of [0,1]"
    assert abs(est - est2) < 1e-12 and abs(factor - factor2) < 1e-12, "convenience wrapper disagrees"
    print("[GATE4] PASS  (finite CV estimate from a batched run; factor in [0,1])")
    return est, factor


def main():
    print("=" * 72)
    print("CONTROL-VARIATE OBSERVABLE ESTIMATOR (FL-1) GATES")
    print("=" * 72)
    run_gate1_unbiased()
    run_gate2_variance_reduction()
    run_gate3_degenerate()
    run_gate4_trajectory_smoke()
    print("=" * 72)
    print("ALL AVAILABLE CONTROL-VARIATE GATES PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
