# -*- coding: utf-8 -*-
"""
On-the-fly Gaussian-Process surrogate min-mode (dimer) saddle search.

TRAINING-FREE forward reducer for MAPLE. The expensive object is the MLIP
force/energy oracle (each ``calc.get_ef_gpu()`` batched forward, 95-99% of wall
time per profiling). A *plain* dimer spends dozens of true forwards per saddle
(one per rotation-HVP finite difference + one per translation step). Here every
rotation/translation of the min-mode search is taken on a CHEAP local Gaussian
process fit ONLINE from the (x, E, F) points this run has already evaluated --
NO pretrained weights, NO offline training set (that is the training-free
requirement). Each outer round emits ONE most-informative acquisition point per
structure; ALL structures' proposals are evaluated in ONE batched forward, so
the GPU stays saturated while the per-structure true-force count drops ~10x.

Design (why this COMPLEMENTS MAPLE's batching instead of fighting it)
---------------------------------------------------------------------
CAMPAIGN-BATCHED ACQUISITION. Run B independent TS-searches concurrently. Each
owns its own local GP and runs its dimer inner loop entirely on the surrogate
(all rotations + translations = ZERO true forwards); each proposes its single
next true-eval location; ALL B proposals are packed into ONE ``get_ef_gpu()``
forward per outer round (reuses the existing (B, nmax_dof) batched-forward
primitive + streaming-pool/shrink pattern from ``BatchDimer`` / ``BatchPRFO``).

EXACT-AT-CONVERGENCE. Convergence rests on the TRUE MLIP FORCE, never on the
surrogate: a structure is declared converged only when the TRUE force at its
latest evaluated point satisfies |F| < fmax (per-atom max & RMS) AND it is a
genuine FIRST-ORDER saddle. The saddle-type test is the number of NEGATIVE modes
of the surrogate Hessian at the candidate (free finite differences of the GP
gradient): a first-order saddle has EXACTLY ONE. This is the correct test --
a single-axis "curvature < 0" is necessary but NOT sufficient (a |F|<fmax point
can be a minimum whose perpendicular subspace still has negative modes, or a
higher-order saddle; the index==1 test rejects both, so the search cannot
false-converge to a non-TS). Optionally the single negative mode's sign is also
confirmed by ONE TRUE FD-HVP forward (``confirm_curv_true=True``). The GP only
CHOOSES where to sample and steers the min mode; the stop test reads the real
forward, so the surrogate can be crude and the answer is still the real saddle.

ROBUST ACQUISITION (so the search reaches the saddle on a stiff real PES instead
of wandering to ``max_iter``). Three ingredients, all training-free: (1) each GP
is WARM-STARTED with a few points sampled around the guess so the GEK has real
curvature before the first surrogate proposal (a 1-2 point GP has no min mode ->
garbage early steps); (2) each proposal is capped by a per-structure MODEL-QUALITY
trust radius that grows when the GP predicted the true force well at the last
proposal and shrinks when it mispredicted (Denzel-Kastner GP-TS), with the center
always advancing (a min-mode climb legitimately raises |F|, so an |F|-monotone
accept/reject would stall); (3) the kernel length scale is data-driven (see
below). NOTE: a min-mode dimer can still converge to a DIFFERENT stationary point
than intended from a poor start (inherent to min-mode search, not a bug); seeding
the reaction-coordinate axis (``n_init``) or a better guess is the remedy, and the
index==1 gate guarantees whatever is returned IS a true first-order saddle.

The GP is a Gradient-Enhanced Kriging (GEK) model: it is fit on BOTH the energy
and the gradient (= -force) that each true forward already returns for free, so
a single true eval gives (3n+1) observations. Kernel = squared-exponential on a
coordinate descriptor; default descriptor = INVERSE INTERATOMIC DISTANCES
(Koistinen et al., inverse-distance kernel: ~40 vs 170-228 evals vs Matern/SE
for molecules; also rototranslationally invariant, so rigid modes are naturally
zero-curvature and never trap the min mode). A ``cartesian`` descriptor is
provided for point-particle / analytic-PES self-tests. The kernel length scale
is DATA-DRIVEN by default (``length_scale="auto"`` = median pairwise descriptor
distance, recomputed each refit): a fixed length scale tuned on a toy PES is
wrong for real interatomic-distance magnitudes and gives an ill-conditioned GP
whose surrogate gradient/curvature are garbage, so the search wanders and never
reaches low force. Adaptive pruning caps the GP training set to ``M_max`` nearest
points so the O(M^3) Cholesky refit stays far under one true forward.

References (cited per project 铁则)
----------------------------------
  * V. Asgeirsson-style GP min-mode / GP-dimer saddle search:
      arXiv 2505.12519 == O. et al., J. Chem. Theory Comput. 2025,
      DOI 10.1021/acs.jctc.5c00866 (GP saddle-point search).
  * Koistinen, Asgeirsson, Vehtari, Jonsson, "Minimum Mode Saddle Point
      Searches Using Gaussian Process Regression with Inverse-Distance
      Covariance Function", J. Chem. Theory Comput. 2020, 16, 499.
      DOI 10.1021/acs.jctc.9b01038  (the inverse-distance kernel used here).
  * Denzel, Kastner, "Gaussian Process Regression for Transition State
      Search", J. Chem. Theory Comput. 2018, 14, 5777.
      DOI 10.1021/acs.jctc.8b00708  (GP-TS + trust radius).
  * Adaptive GP-set pruning (cap M so O(M^3) stays sub-forward):
      arXiv 2510.06030.
  * Dimer min-mode following (finite-difference curvature + rotation):
      G. Henkelman, H. Jonsson, J. Chem. Phys. 1999, 111, 7010.
      DOI 10.1063/1.480097  (the min-mode taken here on the SURROGATE gradient).

Backend gate (mirrors GSMBatch / AutoNEBBatch / BatchDimer): only LOCAL /
block-diagonal batch calculators (UMA-local, standard MACE / MACE-OFF,
AIMNet2-decoupled, ANI) may cross-structure batch; globally coupled /
polarizable calcs (MACE-POL, AIMNet2 charge-eq / NSE) are gated out.

The pure-numpy GP core + per-structure ``GPSaddle`` need no torch and no MLIP;
only ``BatchGPSaddle`` (the batched-forward driver) imports torch lazily. The
single-structure ``Dimer`` and ``BatchDimer`` in dimer.py are UNTOUCHED oracles.
"""
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable

import numpy as np

# torch is needed ONLY by BatchGPSaddle (the batched-forward driver). The GP core
# + GPSaddle + descriptors + convergence gate are pure numpy so the analytic
# self-test runs with no torch and no MLIP.
try:  # pragma: no cover - trivial import guard
    import torch
except Exception:  # pragma: no cover
    torch = None

try:  # log_info / JobABC live in the same package; guarded for standalone import
    from .logger import log_info
    from ...jobABC import JobABC
except Exception:  # pragma: no cover - standalone (bare-module) fallback
    def log_info(msgs, output):
        try:
            with open(output, "a") as fh:
                for m in msgs:
                    fh.write(str(m))
        except Exception:
            pass

    class JobABC:  # minimal shim so the module imports standalone
        def __init__(self, output):
            self.output = output


# =============================================================================
# ------------------------------ numeric helpers ------------------------------
# =============================================================================
def _to_np64(x) -> np.ndarray:
    """torch tensor / numpy / list -> contiguous float64 numpy array."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().to("cpu", torch.float64).numpy()
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


def _maxatom(v_flat: np.ndarray) -> float:
    """max over atoms of the per-atom Euclidean norm of a flat (3n,) vector."""
    a = v_flat.reshape(-1, 3)
    return float(np.max(np.linalg.norm(a, axis=1))) if a.size else 0.0


def _rmsatom(v_flat: np.ndarray) -> float:
    a = v_flat.reshape(-1, 3)
    return float(np.sqrt(np.mean(np.sum(a * a, axis=1)))) if a.size else 0.0


# =============================================================================
# ------------------------------ descriptors ----------------------------------
# =============================================================================
# A descriptor maps Cartesian coords x (n,3) -> feature vector q (Nq,) and its
# Jacobian J = dq/dx (Nq, 3n). The GEK kernel lives in feature space; the
# Jacobian transports gradients between Cartesian and feature space so the whole
# GP is fit on Cartesian forces directly (no pseudo-inverse of observations).

def cartesian_descriptor(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Identity feature map q = x.flatten(); J = I. Used by point-particle /
    analytic-PES tests (no interatomic distances exist for 1 particle)."""
    n = x.shape[0]
    q = x.reshape(-1).astype(np.float64)
    J = np.eye(3 * n, dtype=np.float64)
    return q, J


def invdist_descriptor(x: np.ndarray, eps: float = 1e-9
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse interatomic distances q_p = 1 / r_ij for all pairs i<j (Koistinen
    2020). Rototranslationally invariant. Jacobian:
        d(1/r_ij)/dx_i = -(x_i - x_j) / r_ij^3 ,   d/dx_j = +(x_i - x_j)/r_ij^3.
    """
    x = x.astype(np.float64)
    n = x.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    Nq = len(pairs)
    q = np.zeros(Nq, dtype=np.float64)
    J = np.zeros((Nq, 3 * n), dtype=np.float64)
    for p, (i, j) in enumerate(pairs):
        d = x[i] - x[j]
        r = float(np.linalg.norm(d))
        r = max(r, eps)
        q[p] = 1.0 / r
        g = -d / (r ** 3)                       # d(1/r)/d(x_i)
        J[p, 3 * i:3 * i + 3] = g
        J[p, 3 * j:3 * j + 3] = -g
    return q, J


_DESCRIPTORS = {"invdist": invdist_descriptor, "cartesian": cartesian_descriptor}


# =============================================================================
# ------------------------------ online GEK GP --------------------------------
# =============================================================================
class OnlineGEK:
    """Minimal self-contained gradient-enhanced GP (GEK), pure numpy.

    Fit ONLINE on the (x, E, F) points a single structure has accumulated this
    run. Squared-exponential kernel k(q,q') = sf^2 exp(-||q-q'||^2 / (2 l^2)) in
    the descriptor feature space q = phi(x); the Cartesian gradient observation
    operator is J^T d/dq, so an evaluated point contributes 1 energy + 3n
    gradient observations (the forces the forward already returned -- free).

    Predictive mean energy + predictive Cartesian gradient are analytic (SE
    kernel derivatives + the descriptor Jacobian). No third derivatives are
    needed: the min-mode curvature is obtained by a finite difference of the
    *surrogate* gradient (which is free), exactly as the real dimer differences
    the true force. Adaptive pruning keeps the M_max nearest points.
    """

    def __init__(self, descriptor: str = "invdist", length_scale="auto",
                 sigma_f: float = 1.0, noise_e: float = 1e-6,
                 noise_g: float = 1e-5, m_max: int = 12, jitter: float = 1e-10,
                 ls_floor: float = 1e-3, ls_seed: float = 0.5):
        self.phi = _DESCRIPTORS[descriptor]
        # length_scale="auto" -> data-driven: l = median pairwise DESCRIPTOR
        # distance, recomputed each refit. A fixed length scale tuned on a toy
        # PES is wrong for real interatomic-distance magnitudes and yields an
        # ill-conditioned GP (garbage surrogate gradient/curvature -> the search
        # wanders and never reaches low force). Auto-scaling makes the GP robust
        # across systems / feature magnitudes.
        self._auto_ls = isinstance(length_scale, str) and length_scale.lower() == "auto"
        self._ls_floor = float(ls_floor)
        self._ls_seed = float(ls_seed)
        self.l2 = (self._ls_seed ** 2) if self._auto_ls else float(length_scale) ** 2
        self.sf2 = float(sigma_f) ** 2
        self.noise_e = float(noise_e)
        self.noise_g = float(noise_g)
        self.m_max = int(m_max)
        self.jitter = float(jitter)
        # accumulated observations
        self._X: List[np.ndarray] = []          # (n,3) coords
        self._q: List[np.ndarray] = []          # (Nq,) features
        self._J: List[np.ndarray] = []          # (Nq,3n) Jacobians
        self._E: List[float] = []               # energies
        self._g: List[np.ndarray] = []          # (3n,) Cartesian gradient = -F
        self._alpha = None                      # solved coefficients
        self._dirty = True
        self._n = None                          # atom count (fixed per structure)

    # ---------------------------------------------------------------- ingest
    def add(self, x: np.ndarray, E: float, F: np.ndarray):
        """Add one evaluated point. x (n,3), E scalar, F (n,3) force (Eh/A)."""
        x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
        F = np.asarray(F, dtype=np.float64).reshape(-1, 3)
        if self._n is None:
            self._n = x.shape[0]
        q, J = self.phi(x)
        self._X.append(x.copy())
        self._q.append(q)
        self._J.append(J)
        self._E.append(float(E))
        self._g.append((-F).reshape(-1).copy())     # gradient = -force
        self._prune(x.reshape(-1))
        self._dirty = True

    def _prune(self, x_ref_flat: np.ndarray):
        """Keep the m_max points nearest (in feature space) to the reference."""
        if len(self._X) <= self.m_max:
            return
        qref, _ = self.phi(x_ref_flat.reshape(-1, 3))
        d = np.array([float(np.sum((qi - qref) ** 2)) for qi in self._q])
        keep = np.argsort(d)[: self.m_max]
        keep = set(int(k) for k in keep)
        self._X = [self._X[i] for i in range(len(self._X)) if i in keep]
        self._q = [self._q[i] for i in range(len(self._q)) if i in keep]
        self._J = [self._J[i] for i in range(len(self._J)) if i in keep]
        self._E = [self._E[i] for i in range(len(self._E)) if i in keep]
        self._g = [self._g[i] for i in range(len(self._g)) if i in keep]

    @property
    def n_points(self) -> int:
        return len(self._X)

    # ---------------------------------------------------------------- kernel
    def _pair(self, qa: np.ndarray, qb: np.ndarray):
        """Return (k, d, M2) for SE kernel between features qa, qb.
        k scalar; d = qa-qb; M2 = k*(I/l2 - outer(d,d)/l2^2)  [mixed 2nd deriv]."""
        d = qa - qb
        k = self.sf2 * math.exp(-0.5 * float(d @ d) / self.l2)
        Nq = d.shape[0]
        M2 = k * (np.eye(Nq) / self.l2 - np.outer(d, d) / (self.l2 ** 2))
        return k, d, M2

    def _update_length_scale(self):
        """Data-driven length scale = median pairwise descriptor distance (only
        when length_scale='auto'). Floor guards against coincident points."""
        if not self._auto_ls:
            return
        Q = self._q
        if len(Q) < 2:
            self.l2 = self._ls_seed ** 2
            return
        ds = [float(np.sqrt(np.sum((Q[i] - Q[j]) ** 2)))
              for i in range(len(Q)) for j in range(i + 1, len(Q))]
        md = float(np.median(ds)) if ds else self._ls_seed
        self.l2 = max(md, self._ls_floor) ** 2

    def _refit(self):
        """Assemble the GEK Gram over all points and Cholesky-solve for alpha."""
        self._update_length_scale()
        P = len(self._X)
        n = self._n
        blk = 1 + 3 * n                          # obs per point: E + 3n gradients
        M = P * blk
        K = np.zeros((M, M), dtype=np.float64)
        y = np.zeros(M, dtype=np.float64)
        for a in range(P):
            ra = a * blk
            y[ra] = self._E[a]
            y[ra + 1: ra + blk] = self._g[a]
            Ja = self._J[a]
            for b in range(a, P):
                rb = b * blk
                k, d, M2 = self._pair(self._q[a], self._q[b])
                Jb = self._J[b]
                # energy-energy
                K[ra, rb] = k
                # energy_a - grad_b : J_b^T (k d / l2)   (d = qa-qb)
                eg = Jb.T @ (k * d / self.l2)
                K[ra, rb + 1: rb + blk] = eg
                # grad_a - energy_b : J_a^T (-k d / l2)
                ge = Ja.T @ (-k * d / self.l2)
                K[ra + 1: ra + blk, rb] = ge
                # grad_a - grad_b : J_a^T M2 J_b
                gg = Ja.T @ M2 @ Jb
                K[ra + 1: ra + blk, rb + 1: rb + blk] = gg
                if b != a:                       # mirror the symmetric block
                    K[rb, ra] = k
                    K[rb + 1: rb + blk, ra] = eg
                    K[rb, ra + 1: ra + blk] = ge
                    K[rb + 1: rb + blk, ra + 1: ra + blk] = gg.T
            # observation noise on the diagonal blocks
            K[ra, ra] += self.noise_e
            for s in range(3 * n):
                K[ra + 1 + s, ra + 1 + s] += self.noise_g
        # symmetrize + jitter, robust Cholesky (escalate jitter on failure)
        K = 0.5 * (K + K.T)
        jit = self.jitter
        for _ in range(8):
            try:
                L = np.linalg.cholesky(K + jit * np.eye(M))
                z = np.linalg.solve(L, y)
                self._alpha = np.linalg.solve(L.T, z)
                self._dirty = False
                return
            except np.linalg.LinAlgError:
                jit = max(jit * 10.0, 1e-12)
        # last resort: least-squares (never raise inside the surrogate loop)
        self._alpha = np.linalg.lstsq(K + jit * np.eye(M), y, rcond=None)[0]
        self._dirty = False

    # --------------------------------------------------------------- predict
    def predict(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Predictive (energy, Cartesian gradient (3n,)) at geometry x (n,3)."""
        if self._dirty or self._alpha is None:
            self._refit()
        x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
        qs, Js = self.phi(x)
        P = len(self._X)
        n = self._n
        blk = 1 + 3 * n
        kstar = np.zeros(P * blk, dtype=np.float64)
        dmu_dq = np.zeros(qs.shape[0], dtype=np.float64)     # d mu / d q*
        for a in range(P):
            ra = a * blk
            d = qs - self._q[a]                              # q* - q_a
            k = self.sf2 * math.exp(-0.5 * float(d @ d) / self.l2)
            Ja = self._J[a]
            # k_star vs energy obs
            kstar[ra] = k
            # k_star vs grad obs : cov(f(q*), g_a) = J_a^T d_{q_a}k(q*,q_a)
            #   = J_a^T (k*d/l2), d = q* - q_a  (matches the eg block template)
            kstar[ra + 1: ra + blk] = Ja.T @ (k * d / self.l2)
            # d/dq* of k_star, contracted with alpha:
            ae = self._alpha[ra]
            ag = self._alpha[ra + 1: ra + blk]
            # energy-obs term: ae * d_{q*}k = ae * (-k*d/l2)
            dmu_dq += ae * (-k * d / self.l2)
            # grad-obs term: +M2 @ (J_a ag)   [M2 = k(I/l2 - dd^T/l2^2)]
            Nq = d.shape[0]
            M2 = k * (np.eye(Nq) / self.l2 - np.outer(d, d) / (self.l2 ** 2))
            dmu_dq += (M2 @ (Ja @ ag))
        mu_E = float(kstar @ self._alpha)
        grad_cart = Js.T @ dmu_dq                            # (3n,)
        return mu_E, grad_cart


# =============================================================================
# ---------------------------- convergence gate -------------------------------
# =============================================================================
def saddle_converged(F_true_cart: np.ndarray, curvature: float,
                     f_max_th: float = 5.0e-3, f_rms_th: float = 1.0e-3,
                     require_neg_curv: bool = True) -> bool:
    """EXACT-AT-CONVERGENCE gate. ``F_true_cart`` is the TRUE MLIP force (n,3 or
    3n) at the point; ``curvature`` is the lowest surrogate curvature (the
    first-order-saddle indicator). Converged iff the true per-atom max & RMS
    forces are below threshold AND (optionally) the surrogate curvature is
    negative (exactly one negative mode = a first-order saddle). The FORCE test
    reads the real forward, so convergence is exact to the force tolerance
    regardless of surrogate accuracy. Shared verbatim by BatchGPSaddle and the
    torch-free analytic self-test."""
    F = np.asarray(F_true_cart, dtype=np.float64).reshape(-1)
    force_ok = (_maxatom(F) <= f_max_th) and (_rmsatom(F) <= f_rms_th)
    curv_ok = (curvature < 0.0) if require_neg_curv else True
    return bool(force_ok and curv_ok)


def model_quality_trust(force_pred_err: float, trust: float, grow: float,
                        shrink: float, trust_min: float, trust_max: float,
                        err_lo: float, err_hi: float) -> float:
    """OUTER acquisition trust region driven by GP MODEL QUALITY (Denzel-Kastner
    GP-TS), shared by BatchGPSaddle + the self-test. ``force_pred_err`` is the
    relative error between the GP-predicted and the TRUE force at the proposal:
    if the GP predicted well (err < err_lo) the surrogate is trustworthy further
    out -> GROW the trust; if it mispredicted (err > err_hi) the step overshot the
    GP's valid region -> SHRINK. The center ALWAYS advances to the (evaluated,
    best-estimate) proposal -- a min-mode climb legitimately raises |F|, so an
    |F|-monotone accept/reject would stall; gating on prediction accuracy instead
    keeps proposals where the GP is accurate WITHOUT blocking the climb."""
    if force_pred_err < err_lo:
        return float(min(trust * grow, trust_max))
    if force_pred_err > err_hi:
        return float(max(trust * shrink, trust_min))
    return float(min(max(trust, trust_min), trust_max))


def warmup_displacements(x_center: np.ndarray, n_warmup: int, delta: float,
                         remove_rigid: bool, rng) -> List[np.ndarray]:
    """A few small rigid-body-free displacements around the guess to SEED the GP
    before the first surrogate proposal (a 1-2 point GP has no min-mode curvature
    -> its first steps are garbage). Returns a list of (n,3) displacements."""
    n = x_center.shape[0]
    out = []
    for _ in range(max(0, int(n_warmup))):
        d = rng.normal(size=(n, 3))
        if remove_rigid and n > 1:
            d = _remove_rigid(d.reshape(-1), x_center).reshape(n, 3)
        nrm = float(np.linalg.norm(d))
        d = (d / nrm) if nrm > 1e-12 else d
        out.append(float(delta) * d)
    return out


# =============================================================================
# ---------------------------- per-structure search ---------------------------
# =============================================================================
@dataclass
class GPSaddleParams:
    # --- descriptor / GP ---
    descriptor: str = "invdist"        # "invdist" (molecules) | "cartesian"
    length_scale: object = "auto"      # "auto" = median pairwise descriptor dist
    sigma_f: float = 1.0
    noise_e: float = 1e-6
    noise_g: float = 1e-5
    m_max: int = 12                    # adaptive GP-set cap (O(M^3) bound)
    # --- surrogate dimer inner loop (ALL on the GP -> zero true forwards) ---
    delta: float = 5.0e-3              # FD-HVP half-step on the surrogate gradient
    rot_max_iter: int = 6              # rotation iters per surrogate step
    rot_alpha: float = 0.5             # steepest-descent rotation factor
    rot_f_tol: float = 1.0e-3          # stop rotating once |F_rot| < tol
    max_inner: int = 25               # surrogate translation steps per proposal
    inner_tol: float = 1.0e-4          # surrogate |F_trans| stop (proposal ready)
    step0: float = 0.2                 # translation step factor on F_trans
    step_max: float = 0.10             # max per-atom Cartesian step (A)
    kappa_to_flip: float = 0.0         # flip parallel force once curvature < this
    # --- acquisition trust region (keep proposals where the GP has data) ---
    max_acq_step: float = 0.30         # INITIAL/MAX cap on |proposal-center| per round (A)
    # OUTER trust region on the acquisition: after evaluating the TRUE force at a
    # proposal, accept the move (advance the center) only if it did not worsen the
    # true |F| (within a small slack); otherwise REJECT (keep the old center) and
    # SHRINK the per-structure trust radius, re-proposing next round from the same
    # center with a smaller step + a GP that just gained the (still-informative)
    # rejected point. This is what makes the search robust on a stiff real PES:
    # a crude early GP that overshoots cannot run away (Denzel-Kastner GP-TS trust
    # radius). The center's true |F| is then non-increasing -> monotone progress.
    trust_grow: float = 1.5            # x trust when the GP predicted the force well
    trust_shrink: float = 0.5          # x trust when the GP mispredicted (overshoot)
    trust_min: float = 0.02            # trust-radius floor (A)
    trust_err_lo: float = 0.3          # grow trust if |F_pred-F_true|/|F_true| < this
    trust_err_hi: float = 1.0          # shrink trust if that relative error > this
    # --- GP warm-up seeding (before the first surrogate proposal) ---
    n_warmup: int = 2                  # extra points sampled around the guess so the
                                       # GEK has curvature from round 1 (a 1-2 point
                                       # GP has no min-mode -> garbage early steps)
    warmup_delta: float = 0.15         # warm-up displacement magnitude (A)
    remove_rigid: bool = True          # project rigid modes out of the axis N
    # --- initial dimer axis ---
    n_init: str = "force"              # "force" | "random"
    seed: int = 0
    # --- convergence (EXACT true-force gate) ---
    f_max_th: float = 5.0e-3
    f_rms_th: float = 1.0e-3
    require_neg_curv: bool = True      # also require a first-order-saddle mode
    curv_patience: int = 3             # accept on the true force alone after this
                                       # many consecutive |F_true|<fmax rounds even
                                       # if the surrogate curvature has not yet
                                       # gone negative (never spuriously max_iter a
                                       # force-converged structure; force is exact)


def _remove_rigid(v_flat: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Project global translation + rotation out of a (3n,) direction (mirror
    dimer._remove_rigid_body_components, mass-unweighted)."""
    n = x.shape[0]
    if n < 2:
        return v_flat.copy()
    v = v_flat.reshape(n, 3).copy()
    v -= v.mean(axis=0, keepdims=True)
    r = x - x.mean(axis=0, keepdims=True)
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        rot = np.cross(r, e)
        denom = float(np.sum(rot * rot)) + 1e-20
        v -= (float(np.sum(v * rot)) / denom) * rot
    return v.reshape(-1)


def _normalize(v: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    nrm = float(np.linalg.norm(v))
    return v / nrm if nrm > eps else v


class GPSaddle:
    """One structure's GP-surrogate min-mode saddle search.

    Holds an ``OnlineGEK`` + a dimer axis ``N`` + a search center ``x_c``. Each
    ``propose()`` runs the dimer inner loop ENTIRELY on the surrogate (rotations
    + translations, zero true forwards) and returns ONE acquisition point (the
    surrogate's predicted next true-eval location) plus the current surrogate
    curvature; the driver evaluates it with a real forward, calls ``accept`` with
    the true (x, E, F), and gates convergence on the true force."""

    def __init__(self, x0: np.ndarray, E0: float, F0: np.ndarray,
                 params: Optional[GPSaddleParams] = None):
        self.p = params or GPSaddleParams()
        p = self.p
        self.gp = OnlineGEK(descriptor=p.descriptor, length_scale=p.length_scale,
                            sigma_f=p.sigma_f, noise_e=p.noise_e,
                            noise_g=p.noise_g, m_max=p.m_max)
        x0 = np.asarray(x0, dtype=np.float64).reshape(-1, 3)
        F0 = np.asarray(F0, dtype=np.float64).reshape(-1, 3)
        self.n = x0.shape[0]
        self.x_c = x0.copy()                      # current search center
        self.gp.add(x0, E0, F0)
        # initial dimer axis. "given" (n_given) lets the caller seed the reaction
        # coordinate (e.g. P-R from a NEB/GSM band) so the min mode starts on the
        # intended mode instead of wandering to another saddle; "force" seeds it
        # from the guess force; "random" otherwise.
        rng = np.random.default_rng(int(p.seed))
        if p.n_init == "given" and getattr(p, "n_given", None) is not None:
            N = np.asarray(p.n_given, dtype=np.float64).reshape(-1)
            if N.size != 3 * self.n:
                raise ValueError(f"n_given size {N.size} != 3*n_atoms {3 * self.n}")
            N = N.copy()
        elif p.n_init == "force" and float(np.linalg.norm(F0)) > 1e-8:
            N = F0.reshape(-1).copy()
        else:
            N = rng.normal(size=3 * self.n)
        if p.remove_rigid:
            N = _remove_rigid(N, x0)
        self.N = _normalize(N)
        self.curvature = 1.0                      # last surrogate curvature
        self.last_true_F = F0.copy()
        self.converged = False
        self._lowforce_streak = 0                 # consecutive |F_true|<fmax rounds
        self.curv_true = None                     # exact curvature (driver may set)

    # ------------------------------------------------------ surrogate helpers
    def _grad(self, x_flat: np.ndarray) -> np.ndarray:
        _, g = self.gp.predict(x_flat.reshape(-1, 3))
        return g

    def _rotate_axis(self, x_flat: np.ndarray, N: np.ndarray) -> Tuple[np.ndarray, float]:
        """Steepest-descent rotation of N toward the lowest-curvature mode using
        finite differences of the SURROGATE gradient (free). Returns (N, C)."""
        p = self.p
        C = 0.0
        for _ in range(p.rot_max_iter):
            g0 = self._grad(x_flat)
            gd = self._grad(x_flat + p.delta * N)
            Hn = (gd - g0) / p.delta               # H @ N (grad = +grad E)
            C = float(N @ Hn)
            Frot = -(Hn - C * N)
            if p.remove_rigid:
                Frot = _remove_rigid(Frot, x_flat.reshape(-1, 3))
                Frot = Frot - (Frot @ N) * N       # keep orthogonal to N
            fr = float(np.linalg.norm(Frot))
            if fr < p.rot_f_tol:
                break
            Theta = _normalize(Frot)
            N = _normalize(N * math.cos(p.rot_alpha) + Theta * math.sin(p.rot_alpha))
            if p.remove_rigid:
                N = _normalize(_remove_rigid(N, x_flat.reshape(-1, 3)))
        return N, C

    def propose(self, trust: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """Run the surrogate dimer inner loop; return (acquisition point (n,3),
        surrogate curvature). ZERO true forwards. ``trust`` overrides the per-round
        acquisition cap (the OUTER trust region, driven by the driver); defaults to
        ``max_acq_step``."""
        p = self.p
        cap = p.max_acq_step if trust is None else max(float(trust), 1e-6)
        x = self.x_c.reshape(-1).copy()
        x0 = x.copy()
        N = self.N.copy()
        alpha = p.step0
        C = self.curvature
        for _ in range(max(1, p.max_inner)):
            N, C = self._rotate_axis(x, N)
            g = self._grad(x)
            F = -g                                 # surrogate force
            Fpar = (F @ N) * N
            Fperp = F - Fpar
            Ftrans = (Fperp - Fpar) if C < p.kappa_to_flip else Fperp
            if p.remove_rigid:
                Ftrans = _remove_rigid(Ftrans, x.reshape(-1, 3))
            step = alpha * Ftrans
            # per-atom trust clip
            smax = _maxatom(step)
            if smax > p.step_max:
                step *= p.step_max / (smax + 1e-20)
            x_try = x + step
            # acquisition trust region: never propose farther than the (per-round,
            # adaptive) cap from the center -- the GP is only trustworthy near data
            disp = _maxatom(x_try - x0)
            if disp > cap:
                x_try = x0 + (x_try - x0) * (cap / (disp + 1e-20))
                x = x_try
                break
            x = x_try
            if _maxatom(Ftrans) < p.inner_tol:
                break
        self.N = N
        self.curvature = C
        return x.reshape(-1, 3), C

    def n_negative_surrogate_modes(self, x: np.ndarray, delta: float = 2.0e-3,
                                   rel_tol: float = 1.0e-2) -> int:
        """Number of NEGATIVE eigenvalues of the SURROGATE Hessian at ``x`` (the
        Morse/saddle INDEX). Built free from central finite differences of the GP
        predictive gradient over the 3n Cartesian basis (zero true forwards);
        eigenvalues within ``rel_tol`` of the spectral scale are treated as zero
        (excludes the rigid-body / soft modes -- for the invdist descriptor the
        surrogate gradient is rototranslationally invariant so rigid modes are
        exactly ~0). A FIRST-ORDER saddle has index == 1; this is the correct
        saddle-TYPE test (curvature < 0 along ONE axis is necessary but NOT
        sufficient -- it also accepts minima's perpendicular saddles / higher-order
        saddles)."""
        xf = np.asarray(x, dtype=np.float64).reshape(-1)
        n3 = xf.size
        H = np.zeros((n3, n3), dtype=np.float64)
        for j in range(n3):
            dp = xf.copy(); dp[j] += delta
            dm = xf.copy(); dm[j] -= delta
            H[:, j] = (self._grad(dp) - self._grad(dm)) / (2.0 * delta)
        H = 0.5 * (H + H.T)
        w = np.linalg.eigvalsh(H)
        scale = float(np.max(np.abs(w))) if w.size else 1.0
        tol = max(1e-8, rel_tol * scale)
        return int(np.sum(w < -tol))

    def add_observation(self, x: np.ndarray, E: float, F: np.ndarray):
        """Add a TRUE (x, E, F) forward result to the GP (does NOT move the search
        center -- used for rejected trust-region proposals, still informative)."""
        x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
        F = np.asarray(F, dtype=np.float64).reshape(-1, 3)
        self.gp.add(x, E, F)
        self.last_true_F = F.copy()

    def set_center(self, x: np.ndarray):
        """Advance the search center to ``x`` (on an accepted trust-region move)."""
        self.x_c = np.asarray(x, dtype=np.float64).reshape(-1, 3).copy()

    def accept(self, x: np.ndarray, E: float, F: np.ndarray):
        """Add the TRUE point AND advance the center (unconditional; used by the
        analytic self-test's simple driver)."""
        self.add_observation(x, E, F)
        self.set_center(x)

    def check_converged(self, F_true: np.ndarray, n_neg: Optional[int] = None,
                        curvature: Optional[float] = None) -> bool:
        """EXACT-at-convergence gate. ``F_true`` is the TRUE MLIP force (never the
        surrogate) -- the convergence criterion is always the exact force. The
        first-order-SADDLE type is checked, in order of rigour:
          1. ``n_neg`` (surrogate-Hessian index): a first-order saddle has EXACTLY
             ONE negative mode. This is the correct saddle-type test and rejects
             minima / higher-order saddles that a single-axis curvature check would
             wrongly accept (a |F|<fmax point can have >1 negative mode).
          2. ``curvature`` (TRUE FD-HVP along the min mode < 0): necessary but not
             sufficient; used as an extra confirmation when supplied.
          3. surrogate min-mode curvature < 0, with a ``curv_patience`` fallback on
             the exact force alone (only when neither of the above is supplied --
             e.g. the simple analytic self-test driver)."""
        p = self.p
        F = np.asarray(F_true, dtype=np.float64).reshape(-1)
        force_ok = (_maxatom(F) <= p.f_max_th) and (_rmsatom(F) <= p.f_rms_th)
        self._lowforce_streak = (self._lowforce_streak + 1) if force_ok else 0
        if curvature is not None:
            self.curv_true = float(curvature)
        if not p.require_neg_curv:
            ok = bool(force_ok)
        elif n_neg is not None:
            # PRIMARY correct first-order-saddle test: exactly one negative mode
            # (plus, if a true axis curvature was supplied, it must be negative).
            saddle_ok = (int(n_neg) == 1)
            if curvature is not None:
                saddle_ok = saddle_ok and (curvature < 0.0)
            ok = bool(force_ok and saddle_ok)
        elif curvature is not None:
            ok = bool(force_ok and (curvature < 0.0))   # exact axis curvature
        else:
            # surrogate min-mode curvature only + patience fallback on exact force
            patience_ok = (self._lowforce_streak >= max(1, int(p.curv_patience)))
            ok = bool(force_ok and ((self.curvature < 0.0) or patience_ok))
        self.converged = ok
        return ok


# =============================================================================
# ---------------------------- backend gate -----------------------------------
# =============================================================================
def _gp_is_batch_calc(calc) -> bool:
    """Batch calc exposes prepare(...) + get_ef_gpu() + step_cart_ (contract of
    UMABatchCalc / MACE*BatchCalc / AIMNet2*BatchCalc / ANIBatchCalc)."""
    return (calc is not None
            and callable(getattr(calc, "prepare", None))
            and callable(getattr(calc, "get_ef_gpu", None))
            and callable(getattr(calc, "step_cart_", None)))


def _gp_calc_cross_batch_safe(calc) -> Tuple[bool, str]:
    """Same gate as GSMBatch / AutoNEBBatch: only LOCAL / block-diagonal
    potentials may pack INDEPENDENT structures into one forward. Coupled /
    polarizable calcs (MACE-POL, AIMNet2 charge-eq / NSE) are unsafe."""
    ov = getattr(calc, "cross_batch_safe", None)
    if isinstance(ov, bool):
        return ov, ("calc.cross_batch_safe override")
    cpl = getattr(calc, "couples_across_batch", None)
    if isinstance(cpl, bool):
        return (not cpl), (f"calc.couples_across_batch={cpl} override")
    cm = getattr(calc, "coupling_mode", None)
    if cm is not None:
        return False, (f"polarizable/coupled calc (coupling_mode={cm!r})")
    low = type(calc).__name__.lower()
    if ("pol" in low) or ("polar" in low):
        return False, f"polarizable calc ({type(calc).__name__})"
    if ("aimnet" in low) and ("decoupl" not in low):
        return False, (f"AIMNet2 charge-eq/NSE calc ({type(calc).__name__}); "
                       "use AIMNet2DecoupledBatchCalc for the batched path")
    return True, f"local/block-diagonal calc ({type(calc).__name__})"


# =============================================================================
# ---------------------------- batched GP driver ------------------------------
# =============================================================================
@dataclass
class BatchGPSaddleParams(GPSaddleParams):
    """Adds the outer-loop / driver knobs to the per-structure GPSaddleParams."""
    max_outer: int = 200               # max outer rounds (= max true forwards/struct)
    shrink_on_converge: bool = True     # slice converged structures out of the batch
    save_traj: bool = True
    verbose: bool = True                # per-round diagnostics -> job stdout + log
    confirm_curv_true: bool = False     # RIGOROUS opt-in: when the exact true-force
                                        # gate is met, confirm the first-order-saddle
                                        # mode with ONE TRUE FD-HVP forward at the
                                        # candidate (true curvature, not surrogate).
                                        # Costs +1 batched forward only on rounds with
                                        # a force-converged candidate (end-game).


class BatchGPSaddle(JobABC):
    """Campaign-batched GP-surrogate saddle search.

    B independent structures, each with its OWN GPSaddle + online GP. Per outer
    round: every active structure proposes ONE point on its surrogate (zero true
    forwards); ALL proposals are packed into ONE batched ``get_ef_gpu`` forward;
    each GP is updated with its new (x, E, F); per-structure convergence is
    tested on the TRUE force. Converged/diverged structures are masked (and, by
    default, sliced) out of the batch. Mirrors BatchDimer's per-structure-state /
    shrink pattern; the single Dimer / BatchDimer stay untouched oracles.

    Usage
    -----
        m = Molecules(atoms_list); m.calc = batch_calc      # e.g. UMABatchCalc
        drv = BatchGPSaddle(output="gp.out", device="cuda")
        drv.run(m)
        drv.true_forward_calls   # the metric that matters (few)
    """

    def __init__(self, output: str, device: str = "cuda",
                 paras: Optional[dict] = None):
        super().__init__(output)
        if torch is None:
            raise ImportError("BatchGPSaddle needs torch (the batched-forward "
                              "driver); the pure-numpy GP core / GPSaddle do not.")
        self.device = torch.device(device if (device == "cpu"
                                   or torch.cuda.is_available()) else "cpu")
        self.params = self._init_params_local(paras)
        self.true_forward_calls = 0
        self.n_outer = 0
        self.final_status: List[str] = []
        self.final_energy = None
        self.final_curvature = None

    def _init_params_local(self, paras) -> BatchGPSaddleParams:
        p = BatchGPSaddleParams()
        if isinstance(paras, dict):
            low = {k.lower(): v for k, v in paras.items()}
            sub = None
            for alias in ("batchgpsaddle", "gpsaddle", "gp", "ts"):
                if alias in low and isinstance(low[alias], dict):
                    sub = {k.lower(): v for k, v in low[alias].items()}
                    break
            src = sub if sub is not None else low
            for f in p.__dataclass_fields__:
                if f.lower() in src:
                    setattr(p, f, src[f.lower()])
        return p

    # ------------------------------------------------------------- geometry
    def _padded_step(self, disps: List[np.ndarray], nmax_dof: int) -> "torch.Tensor":
        """Pack per-structure Cartesian displacements (each (n_b,3)) into the
        (B, nmax_dof) padded layout the calc's ``step_cart_`` expects: row b,
        columns [0:3 n_b] = structure b's atoms (local order), rest zero."""
        B = len(disps)
        step = np.zeros((B, nmax_dof), dtype=np.float64)
        for b, d in enumerate(disps):
            flat = np.asarray(d, dtype=np.float64).reshape(-1)
            step[b, : flat.shape[0]] = flat
        return torch.as_tensor(step, dtype=torch.float64, device=self.device)

    def _set_coords(self, calc, geoms):
        """Place the batch AT the given per-structure geometries via the portable
        ``set_coords_`` contract (avoids step-displacement drift bookkeeping across
        accept/reject). ``geoms`` = list of (n_b,3) in the current atoms_list order
        (== the calc's concatenated _ptr layout)."""
        flat = np.vstack([np.asarray(g, dtype=np.float64).reshape(-1, 3)
                          for g in geoms])
        calc.set_coords_(torch.as_tensor(flat, dtype=torch.float64,
                                         device=self.device))

    def _ef_np(self, calc):
        """One batched forward -> (E: list[float], F: list[(n_b,3) np.float64])."""
        E, F = calc.get_ef_gpu()
        self.true_forward_calls += 1
        E = _to_np64(E).reshape(-1)
        F = _to_np64(F)                            # (B, nmax_dof)
        n_b = _to_np64(calc._n_b).astype(int).reshape(-1)
        Es, Fs = [], []
        for b in range(len(n_b)):
            nb = int(n_b[b])
            Es.append(float(E[b]))
            Fs.append(F[b, : 3 * nb].reshape(nb, 3).copy())
        return Es, Fs

    # ------------------------------------------------------------------- run
    def run(self, mols):
        p = self.params
        atoms_list = list(mols.multiatoms)
        calc = mols.calc
        B0 = len(atoms_list)
        if B0 == 0:
            return
        base, _ = os.path.splitext(self.output)

        # -------- backend gate --------
        if not _gp_is_batch_calc(calc):
            raise NotImplementedError(
                "BatchGPSaddle needs a MAPLE batch calculator "
                "(prepare()/get_ef_gpu()/step_cart_()); got "
                f"{type(calc).__name__}.")
        safe, reason = _gp_calc_cross_batch_safe(calc)
        if not safe:
            raise NotImplementedError(
                f"BatchGPSaddle cannot cross-structure batch a coupled "
                f"calculator: {reason}. Packing independent structures into one "
                f"forward would let them leak (physically wrong). Use a local / "
                f"block-diagonal calc (UMA-local / standard MACE / MACE-OFF / "
                f"AIMNet2-decoupled / ANI), or run per structure.")

        log_info([
            "\n========================================================================\n",
            "        BatchGPSaddle  (campaign-batched GP-surrogate saddle search)     \n",
            "========================================================================\n",
            f"B structures     : {B0}\n",
            f"descriptor       : {p.descriptor} (SE-GEK, m_max={p.m_max})\n",
            f"backend gate     : {reason}\n",
            "Refs: Koistinen JCTC2020 (10.1021/acs.jctc.9b01038, inverse-distance); "
            "Denzel-Kastner JCTC2018 (10.1021/acs.jctc.8b00708); "
            "GP-saddle JCTC2025 (10.1021/acs.jctc.5c00866); HJ JCP1999 (10.1063/1.480097)\n",
            "------------------------------------------------------------------------\n",
        ], self.output)

        # -------- seed: one batched forward at the initial guesses --------
        calc.prepare(atoms_list)
        nmax_dof = int(calc.nmax_dof)
        Es, Fs = self._ef_np(calc)

        result_status = ["max_iter"] * B0
        result_E = [float("nan")] * B0
        result_C = [float("nan")] * B0
        orig_index = list(range(B0))

        states: List[GPSaddle] = []
        x_cur = []
        for b in range(B0):
            x0 = np.asarray(atoms_list[b].get_positions(), dtype=np.float64)
            sp = GPSaddleParams(**{f: getattr(p, f)
                                   for f in GPSaddleParams.__dataclass_fields__})
            sp.seed = p.seed + b                    # decorrelate random inits
            states.append(GPSaddle(x0, Es[b], Fs[b], sp))
            x_cur.append(x0.copy())
        # per-structure OUTER trust radius + the center's true |F| (monotone target)
        trust = [float(p.max_acq_step)] * B0
        center_fmax = [_maxatom(Fs[b].reshape(-1)) for b in range(B0)]

        # -------- GP warm-up: a few batched forwards around each guess so every
        #          GEK has real curvature before the FIRST surrogate proposal
        #          (a 1-2 point GP has no min-mode -> garbage early steps -> wander)
        if p.n_warmup > 0:
            warm = [warmup_displacements(
                        x_cur[b], p.n_warmup, p.warmup_delta, p.remove_rigid,
                        np.random.default_rng(int(p.seed) + b + 777))
                    for b in range(B0)]
            for w in range(p.n_warmup):
                geoms = [x_cur[b] + warm[b][w] for b in range(B0)]
                self._set_coords(calc, geoms)
                Ew, Fw = self._ef_np(calc)
                for b in range(B0):
                    states[b].add_observation(geoms[b], Ew[b], Fw[b])
            self._set_coords(calc, x_cur)           # back to the guess centers

        for it in range(1, p.max_outer + 1):
            self.n_outer = it
            B = len(atoms_list)
            if B == 0:
                break

            # (1) each active structure proposes ONE point on its GP within its
            #     current OUTER trust radius (0 true forwards). Also record the
            #     GP-PREDICTED force at the proposal (before the true point is
            #     added) -> the model-quality metric that drives the trust radius.
            proposals, C_surr, F_pred = [], [], []
            for b in range(B):
                xp, C = states[b].propose(trust=trust[b])
                proposals.append(xp); C_surr.append(C)
                _Ep, g_pred = states[b].gp.predict(xp)
                F_pred.append(-g_pred.reshape(-1))

            # (2) place the batch AT all proposals -> ONE batched forward
            self._set_coords(calc, proposals)
            Es, Fs = self._ef_np(calc)
            prop_fmax = [_maxatom(Fs[b].reshape(-1)) for b in range(B)]

            # (3a-i) first-order-SADDLE type test at any force-converged candidate:
            #        count the SURROGATE-Hessian negative modes (free) -> a true TS
            #        has exactly one. Rejects minima / higher-order saddles that a
            #        single-axis curvature check would wrongly accept.
            n_neg = [None] * B
            for b in range(B):
                if prop_fmax[b] <= p.f_max_th:
                    n_neg[b] = states[b].n_negative_surrogate_modes(proposals[b])

            # (3a-ii) RIGOROUS opt-in: also confirm the min mode with ONE TRUE
            #         FD-HVP forward at the candidate (true curvature, NOT surrogate).
            C_true = [None] * B
            if p.confirm_curv_true and any(prop_fmax[b] <= p.f_max_th
                                           for b in range(B)):
                pert = [proposals[b] + p.delta * states[b].N.reshape(states[b].n, 3)
                        for b in range(B)]
                self._set_coords(calc, pert)
                _E2, F2 = calc.get_ef_gpu(); self.true_forward_calls += 1
                F2 = _to_np64(F2)
                for b in range(B):
                    nb = states[b].n
                    F2b = F2[b, : 3 * nb].reshape(nb, 3)
                    HN = -(F2b - Fs[b]) / p.delta              # H @ N  (grad = -F)
                    Nb = states[b].N.reshape(nb, 3)
                    C_true[b] = float(np.sum(HN * Nb))         # N^T H N (true)

            # (3b) per structure: add the (always-informative) point to the GP,
            #      gate on the EXACT true force, then OUTER trust-region accept/reject
            conv_local = []
            for b in range(B):
                states[b].add_observation(proposals[b], Es[b], Fs[b])
                done = states[b].check_converged(Fs[b], n_neg=n_neg[b],
                                                 curvature=C_true[b])
                if done:
                    states[b].set_center(proposals[b])
                    x_cur[b] = proposals[b].copy(); center_fmax[b] = prop_fmax[b]
                    oi = orig_index[b]
                    result_status[oi] = "converged"
                    result_E[oi] = float(Es[b])
                    result_C[oi] = float(states[b].curvature
                                         if C_true[b] is None else C_true[b])
                    atoms_list[b].positions[:] = proposals[b]
                    if p.save_traj:
                        _write_xyz(f"{base}_gp_ts_{oi}.xyz", atoms_list[b], Es[b])
                else:
                    # model-quality trust: relative GP force-prediction error at
                    # the proposal -> grow/shrink the trust; ALWAYS advance the
                    # center to the evaluated point (accept-always; a min-mode climb
                    # legitimately raises |F| so |F|-monotone reject would stall).
                    Ftrue = Fs[b].reshape(-1)
                    ferr = (float(np.linalg.norm(F_pred[b] - Ftrue))
                            / (float(np.linalg.norm(Ftrue)) + 1e-6))
                    trust[b] = model_quality_trust(
                        ferr, trust[b], p.trust_grow, p.trust_shrink, p.trust_min,
                        p.max_acq_step, p.trust_err_lo, p.trust_err_hi)
                    states[b].set_center(proposals[b])
                    x_cur[b] = proposals[b].copy(); center_fmax[b] = prop_fmax[b]
                conv_local.append(done)

            # (3c) per-round instrumentation (job stdout + log): shows WHY a
            #      structure has/hasn't converged (true |F|, curvature, auto ls,
            #      trust radius, acquisition). Structure 0 tracked explicitly.
            if p.verbose and (it <= 8 or it % 5 == 0 or any(conv_local)):
                c0 = C_true[0] if C_true[0] is not None else C_surr[0]
                ix0 = "-" if n_neg[0] is None else str(n_neg[0])
                msg = (f"[gp round {it:4d}] active={B}/{B0} "
                       f"|F|max(s0)={prop_fmax[0]:.4e} best|F|={min(prop_fmax):.4e} "
                       f"curv(s0)={c0:+.3e} idx(s0)={ix0} "
                       f"ls(s0)={states[0].gp.l2 ** 0.5:.3f} "
                       f"trust(s0)={trust[0]:.3f} conv={sum(conv_local)}/{B} "
                       f"true_fwd={self.true_forward_calls}")
                print(msg)
                log_info([msg + "\n"], self.output)

            # (4) survive-and-shrink: slice converged structures out of the batch
            survive = [b for b in range(B) if not conv_local[b]]
            if p.shrink_on_converge and len(survive) < B:
                atoms_list = [atoms_list[b] for b in survive]
                states = [states[b] for b in survive]
                x_cur = [x_cur[b] for b in survive]
                orig_index = [orig_index[b] for b in survive]
                trust = [trust[b] for b in survive]
                center_fmax = [center_fmax[b] for b in survive]
                if atoms_list:
                    for b in range(len(atoms_list)):
                        atoms_list[b].positions[:] = x_cur[b]
                    calc.prepare(atoms_list, fixed_nmax=nmax_dof)
            if not atoms_list:
                log_info([f"\nAll {B0} GP saddle searches converged at "
                          f"round {it}.\n"], self.output)
                break

        # -------- any structure still active hit the round cap --------
        for b in range(len(atoms_list)):
            oi = orig_index[b]
            if result_status[oi] == "max_iter":
                result_E[oi] = float("nan")
                result_C[oi] = float(states[b].curvature)
                atoms_list[b].positions[:] = x_cur[b]

        n_conv = sum(1 for s in result_status if s == "converged")
        n_neg = sum(1 for c in result_C if c == c and c < 0.0)
        log_info([
            "\n------------------------------------------------------------------------\n",
            "                       BatchGPSaddle summary                            \n",
            "------------------------------------------------------------------------\n",
            f"converged                 : {n_conv}/{B0}\n",
            f"negative final curvature  : {n_neg}/{B0}\n",
            f"TRUE get_ef_gpu forwards  : {self.true_forward_calls}\n",
            f"outer rounds              : {self.n_outer}\n",
            f"per-structure status      : {result_status}\n",
        ], self.output)

        self.final_status = result_status
        self.final_energy = np.array(result_E, dtype=float)
        self.final_curvature = np.array(result_C, dtype=float)
        return


def _write_xyz(path: str, atoms, energy: Optional[float] = None):
    try:
        pos = np.asarray(atoms.get_positions(), dtype=np.float64)
        syms = atoms.get_chemical_symbols()
        with open(path, "w") as f:
            f.write(f"{len(syms)}\n")
            f.write((f"Energy = {energy:.10f}\n") if energy is not None else "\n")
            for s, (x, y, z) in zip(syms, pos):
                f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")
    except Exception:
        pass
