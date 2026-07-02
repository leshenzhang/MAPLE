"""Correctness gates for the MLIP-latent-feature collective variable (ES-8).

Extends the committor ML-CV (:mod:`bias.ml_cv`) with a descriptor built from the
MLIP backend's OWN learned per-atom node embeddings (MACE ``out["node_feats"]``,
pooled to a fixed-size vector) -- :class:`bias.ml_cv.MLIPLatentDescriptor`. The CV
/ metaD contract is UNCHANGED; only the descriptor source is the learned latent
space instead of hand-picked internal coordinates.

Validation axis = ALGORITHM CORRECTNESS (not a literature FES number):

  Gate 1  extraction  : MLIPLatentDescriptor returns FINITE, FIXED-size features
                        from a MACE-OFF backend, responsive to geometry.
  Gate 2  CV plumbing : CommittorCV(descriptor=MLIPLatentDescriptor(...)) gives
                        q in [0,1] and a gradient; the descriptor is autograd-
                        DIFFERENTIABLE-THROUGH-POSITIONS, so dq/dx is exact
                        end-to-end -> max|FD - autograd| check (< 1e-4).
  Gate 3  feeds metaD : CommittorMetaD accepts the latent CV (same interface) and
                        injects a FINITE bias force == -dV/dq * dq/dx (exact).
  Gate 4  fallback    : with IdentityDescriptor / PairwiseDistanceDescriptor the
                        extended module reproduces the ORIGINAL committor-CV gates
                        (no regression). Needs ONLY torch (no MLIP/GPU).

Gates 1-3 need torch + mace + e3nn + a loadable MACE-OFF model (auto-SKIP if any
absent; override the model with $MAPLE_MACEOFF_MODEL). Gate 4 needs only torch.
Deterministic (fixed seeds). Run as a script:
    python -m maple.function.dispatcher.md.bias._test_latent_cv
"""
import os

import numpy as np


# --------------------------------------------------------------------------- #
def _make_backend():
    """Return a CPU MaceOffBatchCalc (fresh model load) or None if torch / mace /
    e3nn / a loadable MACE-OFF model is unavailable (-> gates 1-3 auto-SKIP)."""
    try:
        import torch                                    # noqa: F401
        import mace                                     # noqa: F401
        from e3nn import o3                             # noqa: F401
    except Exception:
        return None
    from maple.function.calculator.mace._maceoff_batch_calculator import (
        MaceOffBatchCalc, _DEFAULT_MODEL)
    model_path = os.environ.get("MAPLE_MACEOFF_MODEL", _DEFAULT_MODEL)
    if not os.path.exists(model_path):
        return None
    try:
        import torch
        return MaceOffBatchCalc(model_path=model_path, device="cpu",
                                dtype=torch.float64)
    except Exception as e:                              # pragma: no cover
        print(f"  [latent-CV] backend construct failed: {e}")
        return None


def _water():
    """Compact H2O geometry (all pairwise distances < MACE-OFF r_max -> full graph,
    no cutoff crossing under a 1e-4 FD step -> smooth descriptor)."""
    from ase import Atoms
    pos = np.array([[0.00, 0.00, 0.00],        # O
                    [0.96, 0.00, 0.00],        # H
                    [-0.24, 0.93, 0.00]],      # H
                   dtype=np.float64)
    return Atoms("OH2", positions=pos)


# --------------------------------------------------------------------------- #
def gate1_extraction(backend):
    print("\n===== GATE 1: MLIP-latent feature extraction (MACE node embeddings) =====")
    from maple.function.dispatcher.md.bias.ml_cv import MLIPLatentDescriptor
    at = _water()
    desc = MLIPLatentDescriptor(backend, atoms=at, pool="mean", invariants_only=True)
    F = desc.n_features()
    feat0 = desc(at.get_positions()).detach().cpu().numpy()
    at2 = _water(); p2 = at2.get_positions(); p2[1, 0] += 0.15      # stretch one O-H
    feat1 = desc(p2).detach().cpu().numpy()
    fixed = (feat0.shape == (F,)) and (feat1.shape == (F,)) and (F > 0)
    finite = bool(np.all(np.isfinite(feat0))) and bool(np.all(np.isfinite(feat1)))
    responsive = float(np.max(np.abs(feat0 - feat1))) > 1e-8       # tracks geometry
    ok = fixed and finite and responsive
    print(f"  n_features={F} (fixed)  feat finite={finite}  "
          f"max|feat(geom0)-feat(geom1)|={np.max(np.abs(feat0-feat1)):.3e} (responsive)")
    print(f"  [GATE 1] {'PASS' if ok else 'FAIL'}")
    return ok, float(F)


# --------------------------------------------------------------------------- #
def gate2_cv_plumbing(backend):
    print("\n===== GATE 2: CommittorCV on the latent CV -- dq/dx FD vs autograd =====")
    import torch
    torch.set_default_dtype(torch.float64)
    from maple.function.dispatcher.md.bias.ml_cv import (
        CommittorCV, MLIPLatentDescriptor)
    at = _water()
    desc = MLIPLatentDescriptor(backend, atoms=at, pool="mean",
                                invariants_only=True, differentiable=True)
    cv = CommittorCV(desc, hidden=(16, 16), seed=0)
    pos = at.get_positions()

    q0, grad = cv.value_and_grad(pos)                  # end-to-end autograd dq/dx (n,3)
    h = 1e-4
    fd = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for c in range(3):
            pp = pos.copy(); pp[i, c] += h
            pm = pos.copy(); pm[i, c] -= h
            fd[i, c] = (cv.value(pp) - cv.value(pm)) / (2 * h)
    res = float(np.max(np.abs(grad - fd)))
    bounded = (0.0 <= q0 <= 1.0)
    finite = bool(np.all(np.isfinite(grad)))
    # detached (fixed per-frame descriptor) mode still yields a bounded value.
    desc_fix = MLIPLatentDescriptor(backend, atoms=at, differentiable=False)
    cv_fix = CommittorCV(desc_fix, hidden=(16, 16), seed=0)
    q_fix = cv_fix.value(pos)
    fix_ok = (0.0 <= q_fix <= 1.0)
    ok = bounded and finite and (res < 1e-4) and fix_ok
    print(f"  q0={q0:.6f} in[0,1]={bounded}  grad finite={finite}  "
          f"max|dq/dx_autograd - dq/dx_FD|={res:.3e} (tol 1e-4)")
    print(f"  detached-mode q={q_fix:.6f} in[0,1]={fix_ok} (piecewise-fixed descriptor)")
    print(f"  [GATE 2] {'PASS' if ok else 'FAIL'}")
    return ok, res


# --------------------------------------------------------------------------- #
class _FakeCalc:
    """Batch-calc stand-in for the metaD apply contract (coord + _ptr), matching
    what BatchedNVT hands the bias hook (batched_metad.BatchedMetaD.apply)."""

    def __init__(self, coord_np, ptr):
        import torch
        self.coord = torch.as_tensor(np.asarray(coord_np, float), dtype=torch.float64)
        self._ptr = np.asarray(ptr, dtype=int)


def gate3_feeds_metad(backend):
    print("\n===== GATE 3: latent CV feeds CommittorMetaD (finite bias force) =====")
    import torch
    torch.set_default_dtype(torch.float64)
    from maple.function.dispatcher.md.bias.ml_cv import (
        CommittorCV, MLIPLatentDescriptor, CommittorMetaD)
    from maple.function.dispatcher.md.bias.batched_metad import WTMetadEngine
    from maple.function.dispatcher.md.bias.gamd import KB_HA_PER_K, HARTREE_PER_KCAL
    from maple.function.dispatcher.md.utils import HA_PER_ANG_TO_AU

    at = _water()
    desc = MLIPLatentDescriptor(backend, atoms=at, pool="mean",
                                invariants_only=True, differentiable=True)
    cv = CommittorCV(desc, hidden=(16, 16), seed=5)
    base = at.get_positions()
    coord = np.vstack([base, base + 0.06])             # 2 walkers, same molecule
    ptr = [0, 3, 6]
    calc = _FakeCalc(coord, ptr)
    B, n = 2, 3

    kT = KB_HA_PER_K * 300.0
    engine = WTMetadEngine(kT, sigma=0.1, height=0.5 * HARTREE_PER_KCAL,
                           biasfactor=8.0, cv_min=0.0, cv_max=1.0, nbins=200)
    for s0 in (0.2, 0.4, 0.6, 0.8):                    # hills across the committor range
        engine.deposit(np.array([s0]))

    bias = CommittorMetaD([at, at], cv, engine, pace=0, deposit=False)
    E = torch.zeros(B, dtype=torch.float64)
    F = torch.zeros(B, 3 * n, dtype=torch.float64)
    E_out, F_out = bias.apply(E.clone(), F.clone(), calc)
    F_np = F_out.detach().cpu().numpy()
    finite = (bool(np.all(np.isfinite(F_np)))
              and bool(np.all(np.isfinite(E_out.detach().cpu().numpy()))))

    # independent closed form: F_i = -dV/dq * dq/dx_i (a.u.)
    cvs, grads = bias.cvs_and_grads(coord, ptr)
    _, dVdq = engine.eval(cvs)
    max_err, nonzero = 0.0, False
    for b in range(B):
        expected = (-(dVdq[b]) * grads[b]).reshape(-1) * HA_PER_ANG_TO_AU
        got = F_np[b, :3 * n]
        max_err = max(max_err, float(np.max(np.abs(got - expected))))
        nonzero |= float(np.max(np.abs(got))) > 1e-12
    ok = finite and (max_err < 1e-10) and nonzero and all(0.0 <= q <= 1.0 for q in cvs)
    print(f"  q per walker={np.round(cvs, 4).tolist()}  dV/dq={np.round(dVdq, 5).tolist()}")
    print(f"  bias force finite={finite} nonzero={nonzero}  "
          f"max|F_apply-(-dV/dq*dq/dx)|={max_err:.3e} (exact)")
    print(f"  [GATE 3] {'PASS' if ok else 'FAIL'}")
    return ok, max_err


# --------------------------------------------------------------------------- #
def gate4_fallback_parity():
    print("\n===== GATE 4: IdentityDescriptor fallback reproduces original gates =====")
    from maple.function.dispatcher.md.bias import _test_ml_cv as orig
    subs = {
        "committor_gate1_FD": orig.gate1_gradient_fd(),
        "committor_gate2_bounded": orig.gate2_bounded(),
        "committor_gate3_toy": orig.gate3_toy_committor(),
        "committor_gate4_metaD": orig.gate4_cv_metad_parity(),
    }
    ok = all(v[0] for v in subs.values())
    print(f"  original committor gates: "
          + ", ".join(f"{k}={'PASS' if v[0] else 'FAIL'}" for k, v in subs.items()))
    print(f"  [GATE 4] {'PASS' if ok else 'FAIL'}  (no regression to committor CV)")
    return ok, 0.0


# --------------------------------------------------------------------------- #
def main():
    try:
        import torch  # noqa: F401
    except Exception as e:
        print(f"[latent-CV] torch unavailable ({e}); ALL gates SKIPPED locally -- "
              f"run on the torch/mace/GPU env.")
        return 0
    print(f"torch {__import__('torch').__version__}  "
          f"cuda={__import__('torch').cuda.is_available()}")

    results = {}
    # Gate 4 needs only torch (proves the extended module didn't break the base CV).
    results["gate4_fallback_parity"] = gate4_fallback_parity()

    backend = _make_backend()
    if backend is None:
        print("\n[latent-CV] mace / e3nn / MACE-OFF model unavailable; gates 1-3 "
              "(MLIP extraction, CV plumbing, feeds-metaD) SKIPPED. Set "
              "$MAPLE_MACEOFF_MODEL and run on the torch/mace env.")
    else:
        results["gate1_extraction"] = gate1_extraction(backend)
        results["gate2_cv_plumbing"] = gate2_cv_plumbing(backend)
        results["gate3_feeds_metad"] = gate3_feeds_metad(backend)

    print("\n" + "=" * 62)
    print(f"{'MLIP-LATENT ML-CV (ES-8) GATE SUMMARY':^62}")
    print("=" * 62)
    allok = True
    for k, (ok, val) in results.items():
        allok &= ok
        print(f"  {k:<26} {'PASS' if ok else 'FAIL'}   metric={val:.4g}")
    if backend is None:
        print("  gate1_extraction / gate2_cv_plumbing / gate3_feeds_metad  SKIPPED "
              "(no MACE backend)")
    print("=" * 62)
    print("[RESULT] " + ("ALL RUN GATES PASS" if allok else "CHECK ABOVE (some FAIL)"))
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
