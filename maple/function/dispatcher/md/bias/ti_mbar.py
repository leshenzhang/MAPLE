"""Alchemical free-energy estimators for MAPLE MD: Thermodynamic Integration (TI)
+ Multistate Bennett Acceptance Ratio (MBAR), for the linear two-state coupling

    E_lambda(x) = (1 - lambda) E_A(x) + lambda E_B(x)

where ``E_A`` / ``E_B`` are two *end-state potentials* (two MLIP models, or -- for
the toy self-test -- two analytic potentials). This is the MLIP-portable route to
alchemical free energy: unlike a single-model lambda-split (which ``wham2d.py``
correctly flags as non-portable, because one monolithic MLIP forward exposes no
``dU/dlambda``), a *two-end-state mix* makes the coupling derivative EXPLICIT,

    dE_lambda/dlambda = E_B(x) - E_A(x)          (evaluate BOTH end states),

so both TI and MBAR need only the two end-state energies of every collected sample
-- exactly what :class:`bias.lambda_mix.BatchedLambdaMix` records per window.

This module is pure numpy (NO torch), mirroring the estimator layer of
:mod:`bias.umbrella` (``wham_1d`` / ``mbar_1d``) so the FE math is unit-tested
standalone (the ``__main__`` gate) and reused verbatim by
:class:`ensemble.ti_batched.BatchedTI`.

Estimators
----------
* **TI**   : ``dF = integral_0^1 <dU/dlambda>_lambda dlambda`` (trapezoid over the
             lambda grid; ``<dU/dlambda>_lambda = <E_B - E_A>_lambda`` per window).
* **MBAR** : cross-evaluate every window's samples at ALL lambda (the ``u_kn``
             reduced-potential matrix) and solve the self-consistent MBAR equations
             (Shirts & Chodera, J. Chem. Phys. 129, 124105 (2008), eq. 11) by a
             compact fixed-point iteration -> reduced free energies ``f_k``;
             ``dF = kT (f_{K-1} - f_0)``. For K=2 MBAR reduces to BAR (Bennett 1976);
             a standalone ``bar_2state`` is provided as the 2-state fallback.

Units are caller-defined: pass energies and ``kT`` in the SAME unit (the ensemble
passes Hartree + ``kT`` in Hartree; the self-test uses reduced units kT=1). All
returned free energies come back in that unit.
"""

import numpy as np

# numpy>=2 renamed trapz -> trapezoid; keep both working.
_TRAPZ = getattr(np, "trapezoid", getattr(np, "trapz", None))


# --------------------------------------------------------------------------
# 0) numerically stable log-sum-exp (avoid a scipy dependency)
# --------------------------------------------------------------------------
def _logsumexp(a, axis=None):
    """Stable ``log(sum(exp(a)))`` along ``axis`` (handles -inf entries)."""
    a = np.asarray(a, dtype=np.float64)
    if axis is None:
        m = np.max(a)
        m = m if np.isfinite(m) else 0.0                # all -inf -> 0 shift
        return float(m + np.log(np.sum(np.exp(a - m))))
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)                # all -inf slice -> 0 shift
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


# --------------------------------------------------------------------------
# 1) linear two-state coupling helpers (the E_lambda mix + endpoints)
# --------------------------------------------------------------------------
def linear_mix_energy(E_A, E_B, lam):
    """``E_lambda = (1-lambda) E_A + lambda E_B``. lambda=0 -> E_A, lambda=1 -> E_B."""
    lam = np.asarray(lam, dtype=np.float64)
    return (1.0 - lam) * np.asarray(E_A, np.float64) + lam * np.asarray(E_B, np.float64)


def linear_mix_force(F_A, F_B, lam):
    """``F_lambda = (1-lambda) F_A + lambda F_B`` (same linear coupling as the energy)."""
    lam = float(lam)
    return (1.0 - lam) * np.asarray(F_A, np.float64) + lam * np.asarray(F_B, np.float64)


def dudl(E_A, E_B):
    """``dE_lambda/dlambda = E_B - E_A`` (lambda-independent for the linear mix)."""
    return np.asarray(E_B, np.float64) - np.asarray(E_A, np.float64)


# --------------------------------------------------------------------------
# 2) TI: trapezoid of <dU/dlambda> over the lambda grid
# --------------------------------------------------------------------------
def ti_integrate(lambdas, dudl_means):
    """``dF = integral <dU/dlambda> dlambda`` by the trapezoidal rule.

    lambdas     : (K,) the lambda grid (should span the coupling range, e.g. 0..1).
    dudl_means  : (K,) per-window ensemble average ``<E_B - E_A>_lambda``.
    Returns the free-energy difference in the energy unit of ``dudl_means``.
    """
    lambdas = np.asarray(lambdas, np.float64)
    dudl_means = np.asarray(dudl_means, np.float64)
    order = np.argsort(lambdas)                          # trapezoid needs sorted x
    return float(_TRAPZ(dudl_means[order], lambdas[order]))


# --------------------------------------------------------------------------
# 3) MBAR: self-consistent fixed-point solver (Shirts & Chodera 2008)
# --------------------------------------------------------------------------
def mbar_free_energies(u_kn, N_k, *, tol=1e-10, max_iter=100000):
    """Solve the MBAR self-consistent equations for the reduced free energies.

    u_kn : (K, N) reduced potential of sample n evaluated in state k (== E_k(x_n)/kT).
    N_k  : (K,)  number of samples drawn from each state (sum == N).
    Returns f (K,) the reduced (dimensionless) free energies, gauge-fixed f[0]=0.

    Fixed point (eq. 11):  f_k = -ln sum_n exp(-u_kn) / sum_j N_j exp(f_j - u_jn).
    """
    u_kn = np.asarray(u_kn, np.float64)
    K, N = u_kn.shape
    N_k = np.asarray(N_k, np.float64).reshape(-1)
    if N_k.shape[0] != K:
        raise ValueError(f"N_k length {N_k.shape[0]} != K={K}")
    log_N = np.where(N_k > 0, np.log(np.where(N_k > 0, N_k, 1.0)), -np.inf)  # (K,)
    f = np.zeros(K, dtype=np.float64)
    for _ in range(int(max_iter)):
        # log denom_n = logsumexp_k ( log_N_k + f_k - u_kn )
        log_denom = _logsumexp(log_N[:, None] + f[:, None] - u_kn, axis=0)   # (N,)
        f_new = -_logsumexp(-u_kn - log_denom[None, :], axis=1)              # (K,)
        f_new = f_new - f_new[0]                                             # gauge
        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new
    return f


def mbar_delta_f(lambdas, EA_list, EB_list, kT, *, tol=1e-10, max_iter=100000):
    """MBAR free-energy difference for the linear two-state mix.

    lambdas : (K,) the per-window lambda values.
    EA_list / EB_list : lists (one per window) of that window's collected end-state
        energies E_A(x_n) / E_B(x_n) (same energy unit as ``kT``).
    kT      : thermal energy (same unit as the energies).
    Returns ``(dF, f)`` : dF = kT (f[-1] - f[0]) in the energy unit; f the reduced
        free-energy vector.
    """
    lambdas = np.asarray(lambdas, np.float64).reshape(-1)
    K = lambdas.shape[0]
    if len(EA_list) != K or len(EB_list) != K:
        raise ValueError("EA_list/EB_list must have one entry per lambda window.")
    EA = np.concatenate([np.asarray(a, np.float64).reshape(-1) for a in EA_list])
    EB = np.concatenate([np.asarray(b, np.float64).reshape(-1) for b in EB_list])
    N_k = np.array([np.asarray(a).size for a in EA_list], dtype=np.float64)
    # reduced potential of sample n in window k: [(1-lam_k)E_A + lam_k E_B] / kT
    u_kn = ((1.0 - lambdas)[:, None] * EA[None, :]
            + lambdas[:, None] * EB[None, :]) / float(kT)                    # (K, N)
    f = mbar_free_energies(u_kn, N_k, tol=tol, max_iter=max_iter)
    dF = float(kT) * float(f[-1] - f[0])
    return dF, f


# --------------------------------------------------------------------------
# 4) BAR: the 2-state special case (Bennett 1976) -- explicit fallback
# --------------------------------------------------------------------------
def bar_2state(w_F, w_R, kT, *, tol=1e-10, max_iter=100000):
    """Bennett Acceptance Ratio between two states -- the MBAR K=2 reduction.

    w_F : forward work per sample from state 0, ``w_F = [E_1 - E_0](x)`` on state-0 samples.
    w_R : reverse work per sample from state 1, ``w_R = [E_0 - E_1](x)`` on state-1 samples.
    Returns dF = F_1 - F_0 in the energy unit of ``kT``. Implemented by building the
    2-state ``u_kn`` from the works (reference each sample in its own state to 0) and
    calling :func:`mbar_free_energies` -- for K=2 MBAR IS BAR (Bennett 1976), so this
    reuses the tested solver rather than a separate root-find.
    """
    w_F = np.asarray(w_F, np.float64).reshape(-1) / float(kT)   # reduced works
    w_R = np.asarray(w_R, np.float64).reshape(-1) / float(kT)
    nF, nR = w_F.size, w_R.size
    # state-0 samples: [u_0,u_1]=[0, w_F]; state-1 samples: [u_0,u_1]=[w_R, 0].
    u_kn = np.zeros((2, nF + nR), dtype=np.float64)
    u_kn[1, :nF] = w_F                                          # u_1 on state-0 samples
    u_kn[0, nF:] = w_R                                          # u_0 on state-1 samples
    f = mbar_free_energies(u_kn, np.array([nF, nR], float),
                           tol=tol, max_iter=max_iter)
    return float(kT) * float(f[1] - f[0])


# --------------------------------------------------------------------------
# 5) Runnable self-test: two shifted 1-D harmonics with a KNOWN analytic dF.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # E_A(x)=1/2 kA (x-a)^2 ,  E_B(x)=1/2 kB (x-b)^2 .  For the linear mix E_lambda,
    # the exact free energy of a 1-D harmonic is F(lambda) = 1/2 kT ln(K_l/(2 pi kT))
    # + E0(lambda), and E0 vanishes at both endpoints, so the analytic answer is
    #        dF = F(1) - F(0) = 1/2 kT ln(kB / kA).
    rng = np.random.RandomState(0)
    kT = 1.0
    kA, kB = 1.0, 4.0
    a, b = 0.0, 1.0
    lambdas = np.linspace(0.0, 1.0, 11)
    M = 40000

    EA_list, EB_list, dudl_means = [], [], []
    for lam in lambdas:
        K_l = (1.0 - lam) * kA + lam * kB               # mixed spring const
        L = (1.0 - lam) * kA * a + lam * kB * b
        mu, sig = L / K_l, np.sqrt(kT / K_l)            # mixed-Gaussian equilibrium
        x = mu + sig * rng.randn(M)
        EA = 0.5 * kA * (x - a) ** 2
        EB = 0.5 * kB * (x - b) ** 2
        EA_list.append(EA)
        EB_list.append(EB)
        dudl_means.append(float(np.mean(EB - EA)))       # <dU/dlambda>_lambda

    dF_analytic = 0.5 * kT * np.log(kB / kA)
    dF_ti = ti_integrate(lambdas, dudl_means)
    dF_mbar, _f = mbar_delta_f(lambdas, EA_list, EB_list, kT)
    # BAR on the two endpoints (windows 0 and K-1): forward/reverse work.
    wF = EB_list[0] - EA_list[0]                          # [E_B - E_A] on lambda=0 samples
    wR = EA_list[-1] - EB_list[-1]                        # [E_A - E_B] on lambda=1 samples
    dF_bar = bar_2state(wF, wR, kT)

    e_ti = abs(dF_ti - dF_analytic)
    e_mbar = abs(dF_mbar - dF_analytic)
    e_agree = abs(dF_ti - dF_mbar)
    print(f"analytic dF = 1/2 kT ln(kB/kA) = {dF_analytic:.5f}")
    print(f"TI   dF = {dF_ti:.5f}   |TI-analytic|   = {e_ti:.5f}")
    print(f"MBAR dF = {dF_mbar:.5f}   |MBAR-analytic| = {e_mbar:.5f}")
    print(f"BAR  dF = {dF_bar:.5f}   (endpoints only, K=2)")
    print(f"|TI - MBAR| = {e_agree:.5f}")
    ok = (e_ti < 0.03) and (e_mbar < 0.03) and (e_agree < 0.03)
    print("TI/MBAR ESTIMATOR SELF-TEST:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
