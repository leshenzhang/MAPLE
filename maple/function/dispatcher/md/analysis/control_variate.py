"""
Ensemble control-variate observable estimator (variance reduction, FL-1).

MAPLE runs B replicas through ONE batched forward (``BatchedNVT`` /
``WeightedEnsemble`` / ``BatchedNPT``: one ``calc.get_ef_gpu()`` over the padded
(B, nmax_dof) buffer). Those B samples are CORRELATED (shared forward, coupled
initial conditions). The control-variate estimator exploits that correlation to
estimate an observable ``<O>`` at ZERO extra forward:

    given per-sample pairs (O_i, C_i), i = 1..N, with an auxiliary "control"
    quantity C whose mean E[C] is KNOWN (or subtractable),

        O_CV = mean(O) - c * (mean(C) - E[C]),

    which is UNBIASED for E[O] because  E[mean(C) - E[C]] = 0, and whose per-sample
    residual variance  Var(O - cC) = Var(O) - 2c Cov(O,C) + c^2 Var(C)  is minimized
    at the OPTIMAL coefficient

        c* = Cov(O, C) / Var(C),

    giving  Var(O_CV) / Var(mean O) = 1 - rho^2 ,  rho = corr(O, C).

Because C correlates with O across the batch (the natural choice C = the shared
potential energy the forward already produced, or a linear function of the
coordinates with a known mean), the estimator variance drops by the factor
``1 - rho^2 < 1`` for any rho != 0 -- variance reduction with NO additional model
evaluation. This is a STATISTICAL ESTIMATOR layer over what MAPLE already
computes, NOT an MD driver / ensemble.

E[C] KNOWN vs UNKNOWN. The reduction is real ONLY when E[C] is known (or
estimated from data INDEPENDENT of the current batch): pass it as ``C_mean=``.
When ``C_mean is None`` we fall back to the batch sample mean of C, for which the
correction term is identically zero, so ``estimate == mean(O)`` exactly (the point
estimate cannot beat the naive mean without external knowledge of E[C]); the
returned ``var_reduction_factor`` still reports the ACHIEVABLE ``1 - rho_hat^2``
(what you would gain given the true E[C]). Good controls with a known mean:
a coordinate/order-parameter linear in positions whose equilibrium value is known,
a harmonic-restraint CV, or an analytic reference energy.

ponytail: DELIBERATE ceilings -- (1) a SINGLE scalar control variate C (not a
multi-CV regression / vector control with a covariance-matrix solve); (2) one
GLOBAL optimal coefficient c estimated from the same samples (no jackknife /
leave-one-out debiasing of the O(1/N) control-coefficient bias, and no split-sample
c estimation); (3) pure numpy -- no torch at import time (torch is touched only
through the live-run helper, so this module is usable for post-processing without
the engine). All batched-run ceilings are inherited from the run object.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "control_variate_estimate",
    "batched_O_C",
    "batched_control_variate",
]


def control_variate_estimate(O, C, C_mean=None):
    """Control-variate estimate of ``E[O]`` from correlated samples ``(O, C)``.

    Parameters
    ----------
    O : array_like
        Per-sample observable values (per-replica and/or per-frame; flattened).
    C : array_like
        Per-sample control-variate values, SAME length as ``O``. The natural
        choice for a batched MD run is the shared per-replica potential energy
        (see :func:`batched_O_C`), or any quantity correlated with ``O`` whose
        mean is known.
    C_mean : float, optional
        The KNOWN mean ``E[C]`` of the control variate. If ``None`` (default) the
        batch sample mean is used, for which the correction is identically zero
        and ``estimate == mean(O)`` (no reduction possible without external E[C]);
        ``var_reduction_factor`` still reports the achievable ``1 - rho^2``.

    Returns
    -------
    estimate : float
        The control-variate estimate  ``mean(O) - c * (mean(C) - E[C])``.
    stderr : float
        Standard error of ``estimate``: ``sqrt(Var(O - cC) / N)`` (the reduced
        variance; equals the naive standard error times ``sqrt(var_reduction_factor)``).
    var_reduction_factor : float
        ``Var(O_CV) / Var(mean O) = 1 - rho_hat^2`` in [0, 1]; strictly < 1 for
        any nonzero sample correlation, == 1 when C is uncorrelated with O.
    c : float
        The optimal coefficient ``c* = Cov(O, C) / Var(C)`` (0 if ``Var(C) == 0``).
    """
    O = np.asarray(O, dtype=float).ravel()
    C = np.asarray(C, dtype=float).ravel()
    if O.shape != C.shape:
        raise ValueError(
            f"O and C must have the same number of samples, got {O.size} and {C.size}."
        )
    n = O.size
    if n == 0:
        raise ValueError("control_variate_estimate: empty samples.")

    O_bar = float(O.mean())
    C_bar = float(C.mean())
    if C_mean is None:
        C_mean = C_bar                              # zero-correction fallback

    dO = O - O_bar
    dC = C - C_bar
    S_oo = float(dO @ dO)
    S_cc = float(dC @ dC)
    S_oc = float(dO @ dC)

    # optimal coefficient c* = Cov(O,C)/Var(C) (ddof cancels in the ratio).
    c = (S_oc / S_cc) if S_cc > 0.0 else 0.0

    # rho_hat^2 -> variance-reduction factor 1 - rho^2 in [0, 1].
    if S_oo > 0.0 and S_cc > 0.0:
        rho2 = (S_oc * S_oc) / (S_oo * S_cc)
        rho2 = min(1.0, max(0.0, rho2))            # guard fp round-off
    else:
        rho2 = 0.0
    var_reduction_factor = 1.0 - rho2

    # control-variate point estimate (unbiased when C_mean == E[C]).
    estimate = O_bar - c * (C_bar - float(C_mean))

    # standard error of the estimator = sqrt( Var(O - cC) / n ).
    # residual sum-of-squares after the optimal projection = S_oo - S_oc^2/S_cc
    # = S_oo * (1 - rho^2); sample variance uses ddof=1 (n-1).
    if n >= 2:
        resid_ss = S_oo * var_reduction_factor
        var_estimate = resid_ss / (n - 1) / n
        stderr = float(np.sqrt(var_estimate)) if var_estimate > 0.0 else 0.0
    else:
        stderr = float("nan")

    return estimate, stderr, var_reduction_factor, c


# --------------------------------------------------------------------------- #
# Helper: pull per-replica O and C from a LIVE batched run (zero extra forward
# for the energy control -- reuses the forward MAPLE already ran).
# --------------------------------------------------------------------------- #
def _batched_positions(run):
    """Per-replica positions (list of (n_b, 3) Angstrom numpy) from a batched run.

    Prefers the run's own ``_walker_positions`` (WeightedEnsemble); otherwise
    slices the calculator master coords ``calc.coord`` (N_atoms, 3) by the
    per-replica offset ``calc._ptr`` (B+1,) -- the sanctioned batched-calc view.
    """
    fn = getattr(run, "_walker_positions", None)
    if callable(fn):
        return fn()
    calc = run.calc
    coord = calc.coord.detach().to("cpu").numpy()
    ptr = calc._ptr
    if hasattr(ptr, "detach"):                     # CUDA/CPU torch tensor -> numpy
        ptr = ptr.detach().to("cpu").numpy()
    ptr = np.asarray(ptr).tolist()
    B = int(getattr(run, "B", len(ptr) - 1))
    return [coord[ptr[b]:ptr[b + 1]].copy() for b in range(B)]


def _replica_energies(run):
    """Per-replica potential energy (B,) numpy from ONE batched forward.

    Uses the run's ``_forces_au`` (returns E in Hartree, (B,)) so it rides the
    SAME forward machinery as the MD step -- no extra model evaluation beyond the
    single call needed to read the current energies.
    """
    E, _F = run._forces_au()
    if hasattr(E, "detach"):
        E = E.detach().to("cpu").numpy()
    return np.asarray(E, dtype=float).ravel()


def batched_O_C(run, observable, control="energy"):
    """Pull per-replica ``(O, C)`` arrays from a live batched run.

    Parameters
    ----------
    run : BatchedNVT / WeightedEnsemble / BatchedNPT (or any object exposing the
        batched-calc interface ``calc.coord`` + ``calc._ptr`` and ``_forces_au``).
    observable : callable
        ``observable(positions_np (n, 3)) -> float`` computing the per-replica
        order parameter O (e.g. a bond length / distance / any scalar of coords).
    control : {"energy"} or callable, default "energy"
        The control variate C per replica. ``"energy"`` uses the shared potential
        energy (Hartree) from the batched forward (zero extra forward). A callable
        ``control(positions_np) -> float`` uses a coordinate-derived control whose
        mean you can supply to :func:`control_variate_estimate` as ``C_mean``.

    Returns
    -------
    O : (B,) ndarray  -- per-replica observable.
    C : (B,) ndarray  -- per-replica control variate.
    """
    positions = _batched_positions(run)
    O = np.array([float(observable(p)) for p in positions], dtype=float)
    if callable(control):
        C = np.array([float(control(p)) for p in positions], dtype=float)
    elif control == "energy":
        C = _replica_energies(run)
        if C.size != O.size:
            raise ValueError(
                f"energy control has {C.size} replicas but observable produced "
                f"{O.size}; pass a callable control instead."
            )
    else:
        raise ValueError(
            f"Unknown control {control!r}: use 'energy' or a callable(positions)->float."
        )
    return O, C


def batched_control_variate(run, observable, control="energy", C_mean=None):
    """Convenience: pull ``(O, C)`` from ``run`` then control-variate-estimate ``<O>``.

    Equivalent to ``control_variate_estimate(*batched_O_C(run, observable,
    control), C_mean=C_mean)``. Returns ``(estimate, stderr, var_reduction_factor,
    c)`` -- see :func:`control_variate_estimate`.
    """
    O, C = batched_O_C(run, observable, control=control)
    return control_variate_estimate(O, C, C_mean=C_mean)
