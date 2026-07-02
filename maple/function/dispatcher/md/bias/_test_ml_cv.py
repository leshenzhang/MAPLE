"""Correctness gates for the committor-based ML collective variable (CommittorCV).

Validation axis = ALGORITHM CORRECTNESS (autograd vs finite-difference, sigmoid
bound, analytic-committor recovery on a toy double well, CV->metaD interface
parity), NOT a literature FES number. All four gates need ONLY torch (NO MLIP
model, NO GPU): the toy double-well committor (gate 3) is pure torch, and the
metaD parity (gate 4) rides the numpy metaD engine through a tiny fake calc.

  Gate 1  gradient FD : autograd dq/dx vs central finite difference of q(x)
                        on a molecular (pairwise-distance) descriptor  (< 1e-5).
  Gate 2  bounded     : q_theta in [0,1] for arbitrary (incl. extreme) inputs.
  Gate 3  toy committor: on U(x)=H(x^2-1)^2 the variationally-trained q_theta
                        recovers the ANALYTIC 1-D committor (corr + RMSE tol).
  Gate 4  CV<->metaD  : the learned CV plugs into the EXISTING metaD slot (same
                        value/gradient + cvs_and_grads signature); metaD accepts
                        it and the injected bias force is finite and == the exact
                        -dV/dq * dq/dx (points along the learned committor).

Deterministic (fixed seeds). Run as a script:
    python -m maple.function.dispatcher.md.bias._test_ml_cv
"""
import os
import sys

import numpy as np


# --------------------------------------------------------------------------- #
def gate1_gradient_fd():
    print("\n===== GATE 1: autograd dq/dx vs finite difference =====")
    import torch
    torch.set_default_dtype(torch.float64)
    from maple.function.dispatcher.md.bias.ml_cv import (
        CommittorCV, PairwiseDistanceDescriptor)

    # spread-out 5-atom geometry (distances ~1-2 A -> descriptor well-conditioned)
    pos = np.array([[0.0, 0.0, 0.0], [1.10, 0.0, 0.0], [1.9, 0.9, 0.1],
                    [3.0, 0.4, -0.3], [2.4, -1.1, 0.5]], dtype=np.float64)
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 3)]
    cv = CommittorCV(PairwiseDistanceDescriptor(pairs), hidden=(24, 24), seed=3)

    q0, grad = cv.value_and_grad(pos)                     # autograd dq/dx (n,3)
    h = 1e-4
    fd = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for c in range(3):
            pp = pos.copy(); pp[i, c] += h
            pm = pos.copy(); pm[i, c] -= h
            fd[i, c] = (cv.value(pp) - cv.value(pm)) / (2 * h)
    res = float(np.max(np.abs(grad - fd)))
    ok = res < 1e-5 and 0.0 <= q0 <= 1.0
    print(f"  q0={q0:.6f}  max|dq/dx_autograd - dq/dx_FD| = {res:.3e}  (tol 1e-5)")
    print(f"  [GATE 1] {'PASS' if ok else 'FAIL'}")
    return ok, res


# --------------------------------------------------------------------------- #
def gate2_bounded():
    print("\n===== GATE 2: q_theta in [0,1] for arbitrary inputs =====")
    import torch
    torch.set_default_dtype(torch.float64)
    from maple.function.dispatcher.md.bias.ml_cv import (
        CommittorCV, IdentityDescriptor)

    d = 5
    cv = CommittorCV(IdentityDescriptor(d), hidden=(32, 32), seed=7)
    rng = np.random.default_rng(0)
    feats = np.concatenate([
        rng.standard_normal((2000, d)),                    # typical
        1e6 * rng.standard_normal((200, d)),               # extreme +/-
        np.full((10, d), 1e8), np.full((10, d), -1e8),     # saturating
    ], axis=0)
    q = cv.q_of_features(feats)
    qmin, qmax = float(q.min()), float(q.max())
    ok = bool(np.all(np.isfinite(q))) and (qmin >= 0.0) and (qmax <= 1.0)
    print(f"  q range over {feats.shape[0]} inputs = [{qmin:.6e}, {qmax:.6e}] "
          f"(all-finite={bool(np.all(np.isfinite(q)))})")
    print(f"  [GATE 2] {'PASS' if ok else 'FAIL'}  (sigmoid bound)")
    return ok, qmax


# --------------------------------------------------------------------------- #
def _analytic_committor_1d(xs, H, beta):
    """Exact 1-D committor for overdamped dynamics on U(x)=H(x^2-1)^2 with A=[x<=-1],
    B=[x>=1]:  q(x) = INT_{-1}^{x} e^{beta U}/INT_{-1}^{1} e^{beta U}  (clamped)."""
    xf = np.linspace(-1.0, 1.0, 4001)
    g = np.exp(beta * H * (xf ** 2 - 1.0) ** 2)
    dx = xf[1] - xf[0]
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * dx)])
    qf = cdf / cdf[-1]
    return np.interp(np.clip(xs, -1.0, 1.0), xf, qf)


def gate3_toy_committor():
    print("\n===== GATE 3: toy double-well committor recovery (variational) =====")
    import torch
    torch.set_default_dtype(torch.float64)
    from maple.function.dispatcher.md.bias.ml_cv import (
        CommittorCV, IdentityDescriptor)

    H, beta = 2.0, 1.0                                     # barrier ~2 kT (recoverable)
    x = np.linspace(-1.5, 1.5, 501).reshape(-1, 1)         # collocation points
    U = H * (x[:, 0] ** 2 - 1.0) ** 2
    w = np.exp(-beta * U)                                  # Boltzmann/sampled density
    A_mask = x[:, 0] <= -1.0
    B_mask = x[:, 0] >= 1.0

    cv = CommittorCV(IdentityDescriptor(1), hidden=(64, 64), seed=0)
    cv.train(x, A_mask, B_mask, weights=w, epochs=4000, lr=3e-3,
             boundary_weight=50.0, verbose=False)

    q_pred = cv.q_of_features(x)
    q_true = _analytic_committor_1d(x[:, 0], H, beta)
    corr = float(np.corrcoef(q_pred, q_true)[0, 1])
    rmse = float(np.sqrt(np.mean((q_pred - q_true) ** 2)))
    # monotone increasing (committor rises A->B) as a sanity check
    mono = float(np.mean(np.diff(q_pred) >= -1e-3))
    ok = (corr >= 0.98) and (rmse <= 0.06)
    print(f"  H={H} kT barrier;  corr(q_pred,q_true)={corr:.4f}  RMSE={rmse:.4f}  "
          f"(tol corr>=0.98, RMSE<=0.06)")
    print(f"  monotone-increasing fraction={mono:.3f}  "
          f"q_pred range=[{q_pred.min():.3f},{q_pred.max():.3f}]")
    print(f"  [GATE 3] {'PASS' if ok else 'FAIL'}")
    return ok, rmse


# --------------------------------------------------------------------------- #
class _FakeCalc:
    """Minimal batch-calc stand-in for the metaD apply contract: exposes
    ``coord`` (N,3 torch) + ``_ptr`` (B+1 numpy), matching what BatchedNVT gives
    the bias hook (batched_metad.BatchedMetaD.apply)."""

    def __init__(self, coord_np, ptr):
        import torch
        self.coord = torch.as_tensor(np.asarray(coord_np, float), dtype=torch.float64)
        self._ptr = np.asarray(ptr, dtype=int)


def gate4_cv_metad_parity():
    print("\n===== GATE 4: learned CV plugs into the EXISTING metaD slot =====")
    import inspect
    import torch
    torch.set_default_dtype(torch.float64)
    from ase import Atoms
    from maple.function.dispatcher.md.bias.ml_cv import (
        CommittorCV, PairwiseDistanceDescriptor, CommittorMetaD)
    from maple.function.dispatcher.md.bias.batched_metad import (
        BatchedMetaD, WTMetadEngine)
    from maple.function.dispatcher.md.bias.gamd import KB_HA_PER_K, HARTREE_PER_KCAL
    from maple.function.dispatcher.md.utils import HA_PER_ANG_TO_AU

    # --- interface parity: same CV contract the built-in COM CV metaD uses -----
    sig_builtin = list(inspect.signature(BatchedMetaD.cvs_and_grads).parameters)
    sig_learned = list(inspect.signature(CommittorMetaD.cvs_and_grads).parameters)
    parity = (sig_builtin == sig_learned == ["self", "coord_np", "ptr"]) and \
             issubclass(CommittorMetaD, BatchedMetaD)
    print(f"  cvs_and_grads signature builtin={sig_builtin} learned={sig_learned}  "
          f"subclass={issubclass(CommittorMetaD, BatchedMetaD)}")

    # --- build the learned CV + two distinct walkers ---------------------------
    base = np.array([[0.0, 0.0, 0.0], [1.10, 0.0, 0.0],
                     [2.25, 0.35, 0.0], [3.30, 0.0, 0.20]], dtype=np.float64)
    at = Atoms("H4", positions=base)
    pairs = [(0, 1), (1, 2), (2, 3), (0, 3)]
    cv = CommittorCV(PairwiseDistanceDescriptor(pairs), hidden=(24, 24), seed=11)

    coord = np.vstack([base, base + 0.05])                 # walker-1 perturbed
    ptr = [0, 4, 8]
    calc = _FakeCalc(coord, ptr)
    B, n = 2, 4

    kT = KB_HA_PER_K * 300.0
    engine = WTMetadEngine(kT, sigma=0.1, height=0.5 * HARTREE_PER_KCAL,
                           biasfactor=8.0, cv_min=0.0, cv_max=1.0, nbins=200)
    for s0 in (0.2, 0.4, 0.6, 0.8):                        # hills across committor range
        engine.deposit(np.array([s0]))

    bias = CommittorMetaD([at, at], cv, engine, pace=0, deposit=False)

    # --- run the inherited metaD apply hook ------------------------------------
    E = torch.zeros(B, dtype=torch.float64)
    F = torch.zeros(B, 3 * n, dtype=torch.float64)
    E_out, F_out = bias.apply(E.clone(), F.clone(), calc)
    F_np = F_out.detach().cpu().numpy()
    finite = bool(np.all(np.isfinite(F_np))) and bool(np.all(np.isfinite(E_out.detach().cpu().numpy())))

    # --- independent closed form: F_i = -dV/dq * dq/dx_i (a.u.) ----------------
    cvs, grads = bias.cvs_and_grads(coord, ptr)
    _, dVdq = engine.eval(cvs)
    align_ok = True
    max_err = 0.0
    for b in range(B):
        expected = (-(dVdq[b]) * grads[b]).reshape(-1) * HA_PER_ANG_TO_AU
        got = F_np[b, :3 * n]
        max_err = max(max_err, float(np.max(np.abs(got - expected))))
        gflat = grads[b].reshape(-1)
        if np.linalg.norm(gflat) > 1e-8 and abs(dVdq[b]) > 1e-9:
            cos = float(np.dot(got, gflat) / (np.linalg.norm(got) * np.linalg.norm(gflat)))
            align_ok &= abs(abs(cos) - 1.0) < 1e-6         # force points along dq/dx
    print(f"  q per walker = {np.round(cvs, 4).tolist()}  dV/dq = {np.round(dVdq, 5).tolist()}")
    print(f"  bias force finite={finite}  max|F_apply - (-dV/dq*dq/dx)| = {max_err:.3e}  "
          f"(exact)  points-along-dq/dx={align_ok}")
    ok = parity and finite and (max_err < 1e-10) and align_ok
    print(f"  [GATE 4] {'PASS' if ok else 'FAIL'}")
    return ok, max_err


# --------------------------------------------------------------------------- #
def main():
    try:
        import torch  # noqa: F401
    except Exception as e:
        print(f"[ML-CV] torch unavailable ({e}); gates need torch (NO MLIP/GPU). "
              f"SKIPPED locally -- run on the torch/GPU env.")
        return 0
    print(f"torch {__import__('torch').__version__}  "
          f"cuda={__import__('torch').cuda.is_available()}")
    results = {}
    results["gate1_gradient_FD"] = gate1_gradient_fd()
    results["gate2_bounded"] = gate2_bounded()
    results["gate3_toy_committor"] = gate3_toy_committor()
    results["gate4_CV_metaD_parity"] = gate4_cv_metad_parity()
    print("\n" + "=" * 60)
    print(f"{'COMMITTOR ML-CV GATE SUMMARY':^60}")
    print("=" * 60)
    allok = True
    for k, (ok, val) in results.items():
        allok &= ok
        print(f"  {k:<24} {'PASS' if ok else 'FAIL'}   metric={val:.4g}")
    print("=" * 60)
    print("[RESULT] " + ("ALL GATES PASS" if allok else "CHECK ABOVE (some FAIL)"))
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
