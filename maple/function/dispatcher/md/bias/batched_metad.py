"""Batch-aware metadynamics + OPES for MAPLE's batched MD core (rides
``ensemble/nvt_batched.py`` via the SAME additive bias hook GaMD / umbrella use:
``BatchedNVT._forces_au`` calls ``job._bias.apply(E, F, calc)`` once per forward;
``nvt_batched.py`` is NOT modified).

This is the GPU-batched enhanced-sampling engine (Paper-2 headline): B walkers =
the batch dimension of ONE ``calc.get_ef_gpu()`` forward, ALL sharing ONE
in-memory bias over the collective variable. Multiple-walker metadynamics
(Raiteri, Laio, Gervasio, Micheletti, Parrinello, JPCB 2006,
10.1021/jp054359r) normally pays a disk/MPI sync so W separately-running walkers
can read+write one shared hill file; here the W walkers co-reside in ONE batched
forward and ONE address space, so the coupling (each walker deposits into + feels
the same bias) is FREE -- no file sync, no MPI.

Methods implemented (all CV-based; a CV-free method is GaMD, already in
``bias.batched``):

WELL-TEMPERED METADYNAMICS  (:class:`WTMetadEngine`)
    History-dependent Gaussian hills deposited along the CV (Laio & Parrinello,
    PNAS 2002, 10.1073/pnas.202427399). Well-tempered variant (Barducci, Bussi &
    Parrinello, PRL 2008, 10.1103/PhysRevLett.100.020603): hill height decays as
    ``w = w0 * exp(-V_bias(s)/(k*dT))`` with ``k*dT = (gamma-1)*kT`` and bias
    factor ``gamma = (T+dT)/T``. At convergence the deposited bias satisfies the
    WELL-TEMPERED ASYMPTOTE ``F(xi) = -(gamma/(gamma-1)) * V_bias(xi) + C`` (gamma
    -> inf recovers standard, non-tempered metaD, constant height). A c(t)
    time-independent reweighting (Tiwary & Parrinello, JPCB 2015,
    10.1021/jp504920s) is provided for reweighting biased frames.

OPES (metad-like)  (:class:`OPESMetadEngine`)
    On-the-fly Probability Enhanced Sampling (Invernizzi & Parrinello, JPCL 2020,
    10.1021/acs.jpclett.0c00497). A weighted KDE estimate ``P(xi)`` of the
    (reweighted) unbiased CV density drives the bias
    ``V(xi) = (1-1/gamma)*kT*log(P(xi)/Z + epsilon)`` with a running normalization
    ``Z`` (kept bounded) and a barrier cap ``epsilon = exp(-beta*dE/(1-1/gamma))``
    so ``V in [-dE, V_max]`` stays bounded. Free energy ``F(xi) = -kT*log P(xi)``.

Both engines are 1-D and CV-agnostic (they act on scalar CV values + the CV's
spatial gradient), so the SAME engine that rides the MLIP forward here also runs
against a toy analytic potential in the correctness gates (validation axis =
ALGORITHM CORRECTNESS, not a literature FES number).

Energy / force representation
-----------------------------
Energy and force are evaluated ANALYTICALLY from the shared kernel list (a sum of
Gaussians for metaD, a weighted KDE for OPES), so the bias force
``F_i = -dV/dxi * dxi/dx_i`` is the EXACT gradient of the bias energy ``V(xi(x))``
to machine precision (gate 1: FD residual < 1e-10). A rendered GRID (``grid_V`` /
``grid_dV`` for metaD; ``grid_P`` / ``grid_dP`` for OPES) is accumulated in lock
step for O(1) evaluation and FES output; the toy-potential gates use the grid path
(cheap over 1e5+ CV steps) and a grid-vs-analytic agreement check ties the two.
On the MLIP path the analytic sum is negligible next to the MACE forward, so
``apply`` uses the EXACT analytic path.

Collective variable -- distance between the centres of mass of two atom groups
(reuses the group selection + isolated no-min-image convention of the rest of the
batch bias layer, :mod:`bias.batched`)::

    R1 = sum_{i in g1} m_i x_i / M1 ,  R2 likewise            [Angstrom]
    xi = |R1 - R2|                                            [Angstrom]
    dxi/dx_i =  (m_i/M1) u   (i in g1),  -(m_j/M2) u  (j in g2),  u = (R1-R2)/xi

Isolated (non-periodic) replicas only (inherited from ``BatchedNVT``).
"""

import numpy as np

from .posres_calc import _build_selection     # reuse group selection (all/heavy/idx)
from .batched import _ptr_to_np               # reuse CUDA-aware _ptr coercion (C2)
from ..utils import HA_PER_ANG_TO_AU          # Ha/Angstrom -> Ha/Bohr (a.u. force)
from .gamd import KB_HA_PER_K, HARTREE_PER_KCAL  # reuse Boltzmann + kcal->Ha


# --------------------------------------------------------------------------- #
#  Collective variable: COM-COM distance + analytic spatial gradient dxi/dx
# --------------------------------------------------------------------------- #
def _com_distance_cv_grad(pos, g1, g2, m):
    """CV = |COM(g1) - COM(g2)| (Angstrom) and its per-atom gradient dxi/dx (n,3,
    dimensionless). Isolated (no minimum image). ``pos`` (n,3) Angstrom; ``g1``/
    ``g2`` int index arrays; ``m`` (n,) masses (amu; only ratios enter the COM)."""
    M1, M2 = m[g1].sum(), m[g2].sum()
    R1 = (m[g1, None] * pos[g1]).sum(0) / M1
    R2 = (m[g2, None] * pos[g2]).sum(0) / M2
    dvec = R1 - R2
    xi = float(np.linalg.norm(dvec))
    u = dvec / xi if xi > 1e-9 else np.zeros(3)
    grad = np.zeros_like(pos)
    if xi > 1e-9:
        grad[g1] += (m[g1] / M1)[:, None] * u[None, :]
        grad[g2] += -(m[g2] / M2)[:, None] * u[None, :]
    return xi, grad


# --------------------------------------------------------------------------- #
#  Well-tempered metadynamics engine (1-D, CV-agnostic, shared across walkers)
# --------------------------------------------------------------------------- #
class WTMetadEngine:
    """Shared well-tempered metadynamics bias over one CV.

    Parameters
    ----------
    kT : float
        Thermal energy [Hartree] (``KB_HA_PER_K * T``).
    sigma : float
        Gaussian hill width [CV units, Angstrom for a distance CV].
    height : float
        Base hill height ``w0`` [Hartree].
    biasfactor : float | None
        Well-tempered bias factor ``gamma``. ``None`` / ``inf`` / ``<=1`` selects
        STANDARD (non-tempered) metadynamics (constant height ``w0``).
    cv_min, cv_max, nbins : float, float, int
        Grid range + resolution (for the O(1) grid path + FES output).
    """

    def __init__(self, kT, sigma, height, biasfactor=None,
                 cv_min=0.0, cv_max=1.0, nbins=300):
        self.kT = float(kT)
        self.sigma = float(sigma)
        self.w0 = float(height)
        g = biasfactor
        self.gamma = (None if g is None or g == 0 or not np.isfinite(g) or g <= 1.0
                      else float(g))
        self.cv_min, self.cv_max, self.nbins = float(cv_min), float(cv_max), int(nbins)
        # shared kernel list (analytic source of truth) -- all walkers append here.
        self.s = np.zeros(0, dtype=np.float64)      # hill centres
        self.h = np.zeros(0, dtype=np.float64)      # hill heights [Ha]
        self.t_dep = np.zeros(0, dtype=np.float64)  # deposit height BEFORE WT scale
        # rendered grid (accumulated in lock step for the fast path / FES).
        self.grid_x = np.linspace(self.cv_min, self.cv_max, self.nbins)
        self.grid_V = np.zeros(self.nbins, dtype=np.float64)
        self.grid_dV = np.zeros(self.nbins, dtype=np.float64)

    # -- well-tempered height scaling: k*dT = (gamma-1)*kT -------------------- #
    def _wt_scale(self, V_at_center):
        if self.gamma is None:
            return np.ones_like(np.atleast_1d(V_at_center))
        return np.exp(-np.atleast_1d(V_at_center) / ((self.gamma - 1.0) * self.kT))

    # -- EXACT analytic evaluation (sum of Gaussians) ------------------------ #
    def eval(self, xi):
        """Analytic (exact) bias. ``xi`` (M,) -> (V (M,) Ha, dV/dxi (M,) Ha/A)."""
        xi = np.atleast_1d(np.asarray(xi, dtype=np.float64))
        if self.s.size == 0:
            z = np.zeros_like(xi)
            return z, z.copy()
        d = xi[:, None] - self.s[None, :]                        # (M, K)
        g = self.h[None, :] * np.exp(-0.5 * (d / self.sigma) ** 2)
        V = g.sum(1)
        dV = (g * (-d / self.sigma ** 2)).sum(1)
        return V, dV

    # -- fast grid evaluation (linear interp of the accumulated grid) -------- #
    def eval_grid(self, xi):
        xi = np.atleast_1d(np.asarray(xi, dtype=np.float64))
        V = np.interp(xi, self.grid_x, self.grid_V)
        dV = np.interp(xi, self.grid_x, self.grid_dV)
        return V, dV

    # -- deposit one hill per walker (simultaneous, shared) ------------------ #
    def deposit(self, xi_dep, use_grid=False):
        """Deposit one WT-scaled Gaussian per walker at ``xi_dep`` (B,). The height
        uses the CURRENT shared bias at each centre (multi-walker: heights are
        computed from the pre-deposit bias, then all B hills are appended)."""
        xi_dep = np.atleast_1d(np.asarray(xi_dep, dtype=np.float64))
        V_at, _ = (self.eval_grid(xi_dep) if use_grid else self.eval(xi_dep))
        heights = self.w0 * self._wt_scale(V_at)
        # append to the analytic kernel list (source of truth).
        self.s = np.concatenate([self.s, xi_dep])
        self.h = np.concatenate([self.h, heights])
        self.t_dep = np.concatenate([self.t_dep, self.w0 * np.ones_like(heights)])
        # accumulate onto the grid in lock step (value + analytic derivative).
        gx = self.grid_x[:, None]                                # (nbins, B)
        d = gx - xi_dep[None, :]
        gg = heights[None, :] * np.exp(-0.5 * (d / self.sigma) ** 2)
        self.grid_V += gg.sum(1)
        self.grid_dV += (gg * (-d / self.sigma ** 2)).sum(1)

    # -- free energy surface from the deposited bias ------------------------- #
    def fes(self, grid=None, use_grid=False):
        """Reconstructed FES [Ha] (zeroed at its minimum). Well-tempered:
        ``F = -(gamma/(gamma-1)) V``; standard metaD: ``F = -V`` (asymptotic)."""
        if grid is None:
            grid = self.grid_x
        V, _ = (self.eval_grid(grid) if use_grid else self.eval(grid))
        F = (-V if self.gamma is None else -(self.gamma / (self.gamma - 1.0)) * V)
        F = F - np.nanmin(F)
        return grid, F

    # -- c(t) reweighting factor (Tiwary-Parrinello 2015) -------------------- #
    def ct(self):
        """Time-independent free-energy shift c(t) [Ha] for the CURRENT bias
        (Tiwary & Parrinello 2015 eq 8): c = kT log( <exp(gamma/(gamma-1) beta V)>
        / <exp(1/(gamma-1) beta V)> ), averages over the CV grid. Used to reweight
        biased frames back to the unbiased ensemble; not needed for the WT
        asymptote gate (that uses F = -(gamma/(gamma-1)) V directly)."""
        if self.gamma is None or self.s.size == 0:
            return 0.0
        V, _ = self.eval(self.grid_x)
        beta = 1.0 / self.kT
        g = self.gamma
        num = np.mean(np.exp((g / (g - 1.0)) * beta * V))
        den = np.mean(np.exp((1.0 / (g - 1.0)) * beta * V))
        return float(self.kT * np.log(num / den))


# --------------------------------------------------------------------------- #
#  OPES (metad-like) engine (1-D, CV-agnostic, shared across walkers)
# --------------------------------------------------------------------------- #
class OPESMetadEngine:
    """Shared OPES (metad-like) bias over one CV (Invernizzi & Parrinello 2020).

    Weighted KDE of the reweighted CV density ``P(xi)`` (kernels at sampled points,
    weight ``w_k = exp(beta V_k)`` to undo the bias). Bias
    ``V(xi) = (1-1/gamma) kT log(P/Z + epsilon)`` with barrier cap
    ``epsilon = exp(-beta*dE/(1-1/gamma))`` (so ``V in [-dE, V_max]`` bounded) and a
    running normalization ``Z`` (mean of ``P`` over the kernels; bounded ->
    ``P/Z ~ O(1)``). Free energy ``F(xi) = -kT log P(xi)``.
    """

    def __init__(self, kT, sigma, biasfactor, barrier,
                 cv_min=0.0, cv_max=1.0, nbins=300):
        self.kT = float(kT)
        self.beta = 1.0 / self.kT
        self.sigma = float(sigma)
        self.gamma = float(biasfactor)
        if self.gamma <= 1.0:
            raise ValueError("OPES needs biasfactor gamma > 1.")
        self.barrier = float(barrier)                       # dE [Ha]
        self.prefac = (1.0 - 1.0 / self.gamma) * self.kT    # (1-1/gamma) kT
        self.epsilon = float(np.exp(-self.beta * self.barrier / (1.0 - 1.0 / self.gamma)))
        self.cv_min, self.cv_max, self.nbins = float(cv_min), float(cv_max), int(nbins)
        self._c = 1.0 / (self.sigma * np.sqrt(2.0 * np.pi))  # normalized-Gaussian const
        # shared kernel list (analytic source of truth).
        self.centers = np.zeros(0, dtype=np.float64)
        self.logw = np.zeros(0, dtype=np.float64)           # log kernel weights
        self.Z = 1.0
        self.Z_history = []
        # rendered UNNORMALIZED grids: sum_k w_k G(x,x_k) and its derivative.
        self.grid_x = np.linspace(self.cv_min, self.cv_max, self.nbins)
        self.grid_Praw = np.zeros(self.nbins, dtype=np.float64)   # sum w_k G
        self.grid_dPraw = np.zeros(self.nbins, dtype=np.float64)  # sum w_k dG
        self.sumw = 0.0

    # -- weighted normalized KDE (analytic exact) ---------------------------- #
    def _kde(self, xi):
        xi = np.atleast_1d(np.asarray(xi, dtype=np.float64))
        if self.centers.size == 0:
            z = np.zeros_like(xi)
            return z, z.copy()
        w = np.exp(self.logw - self.logw.max())
        norm = w.sum()
        d = xi[:, None] - self.centers[None, :]
        G = self._c * np.exp(-0.5 * (d / self.sigma) ** 2)
        P = (w[None, :] * G).sum(1) / norm
        dP = (w[None, :] * G * (-d / self.sigma ** 2)).sum(1) / norm
        return P, dP

    def _kde_grid(self, xi):
        xi = np.atleast_1d(np.asarray(xi, dtype=np.float64))
        if self.sumw <= 0.0:
            z = np.zeros_like(xi)
            return z, z.copy()
        P = np.interp(xi, self.grid_x, self.grid_Praw) / self.sumw
        dP = np.interp(xi, self.grid_x, self.grid_dPraw) / self.sumw
        return P, dP

    # -- bias (analytic exact) ----------------------------------------------- #
    def eval(self, xi):
        P, dP = self._kde(xi)
        arg = P / self.Z + self.epsilon
        V = self.prefac * np.log(arg)
        dV = self.prefac * (dP / self.Z) / arg
        return V, dV

    def eval_grid(self, xi):
        P, dP = self._kde_grid(xi)
        arg = P / self.Z + self.epsilon
        V = self.prefac * np.log(arg)
        dV = self.prefac * (dP / self.Z) / arg
        return V, dV

    # -- deposit one kernel per walker --------------------------------------- #
    def deposit(self, xi_dep, use_grid=False):
        xi_dep = np.atleast_1d(np.asarray(xi_dep, dtype=np.float64))
        V_at, _ = (self.eval_grid(xi_dep) if use_grid else self.eval(xi_dep))
        logw_new = self.beta * V_at                     # w_k = exp(beta V_k)
        self.centers = np.concatenate([self.centers, xi_dep])
        self.logw = np.concatenate([self.logw, logw_new])
        # accumulate the UNNORMALIZED weighted-kernel grids.
        w_new = np.exp(logw_new)
        gx = self.grid_x[:, None]
        d = gx - xi_dep[None, :]
        G = self._c * np.exp(-0.5 * (d / self.sigma) ** 2)
        self.grid_Praw += (w_new[None, :] * G).sum(1)
        self.grid_dPraw += (w_new[None, :] * G * (-d / self.sigma ** 2)).sum(1)
        self.sumw += float(w_new.sum())
        self._update_Z(use_grid=use_grid)

    def _update_Z(self, use_grid=False):
        """Z = mean of P over the kernel centres (bounded typical density). Keeps
        P/Z ~ O(1) so the log stays bounded; floored at epsilon."""
        P, _ = (self._kde_grid(self.centers) if use_grid else self._kde(self.centers))
        self.Z = max(float(P.mean()), self.epsilon)
        self.Z_history.append(self.Z)

    # -- free energy --------------------------------------------------------- #
    def fes(self, grid=None, use_grid=False):
        if grid is None:
            grid = self.grid_x
        P, _ = (self._kde_grid(grid) if use_grid else self._kde(grid))
        F = -self.kT * np.log(np.maximum(P, 1e-300))
        F = F - np.nanmin(F)
        return grid, F


# --------------------------------------------------------------------------- #
#  Batch-aware bias: rides BatchedNVT._forces_au (apply(E,F,calc) -> (E,F))
# --------------------------------------------------------------------------- #
class BatchedMetaD:
    """Per-walker metadynamics / OPES bias on the BatchedNVT force hook.

    All B walkers SHARE ``engine`` (the shared in-memory bias): each walker feels
    the same bias and deposits into it (multiple-walker metaD, Raiteri 2006 --
    free here since the walkers co-reside in one batched forward + one address
    space). ``deposit=False`` makes it a passive CV logger (unbiased reference).

    Bias contract (see :mod:`bias.batched`): ``apply(E, F, calc) -> (E, F)``;
    ``E`` (B,) Ha, ``F`` (B, nmax_dof) a.u. modified IN PLACE, ``calc.coord``
    (N,3 Angstrom) + ``calc._ptr`` (B+1). Force added is
    ``F_i = -dV/dxi * dxi/dx_i`` (a.u.); energy added is ``V(xi_b)``.
    """

    KIND = "metad"

    def __init__(self, atoms_list, group1, group2, engine, pace, deposit=True):
        self.B = len(atoms_list)
        if self.B == 0:
            raise ValueError("BatchedMetaD needs >= 1 walker.")
        self._g1 = [np.asarray(_build_selection(group1, at), dtype=int)
                    for at in atoms_list]
        self._g2 = [np.asarray(_build_selection(group2, at), dtype=int)
                    for at in atoms_list]
        self._m = [np.asarray(at.get_masses(), dtype=np.float64) for at in atoms_list]
        self._n = [len(at) for at in atoms_list]
        for b in range(self.B):
            if len(self._g1[b]) == 0 or len(self._g2[b]) == 0:
                raise ValueError("metaD CV groups must be non-empty.")
            if set(self._g1[b].tolist()) & set(self._g2[b].tolist()):
                raise ValueError("metaD CV groups must not overlap.")
        self.engine = engine
        self.pace = int(pace)
        self.deposit = bool(deposit)
        self._napply = 0
        # per-walker logs (once per force evaluation).
        self.cv_history = [[] for _ in range(self.B)]
        self.bias_history = [[] for _ in range(self.B)]
        self.deposit_steps = []

    # ---------------------------------------------------------------- CV + grad
    def cvs_and_grads(self, coord_np, ptr):
        """Per-walker (CV value, CV spatial gradient (n,3)) from master coords."""
        cvs = np.empty(self.B, dtype=np.float64)
        grads = []
        for b in range(self.B):
            pos = np.asarray(coord_np[ptr[b]:ptr[b + 1]], dtype=np.float64)
            xi, gr = _com_distance_cv_grad(pos, self._g1[b], self._g2[b], self._m[b])
            cvs[b] = xi
            grads.append(gr)
        return cvs, grads

    # ------------------------------------------------------------- bias contract
    def apply(self, E, F, calc):
        import torch
        coord_np = calc.coord.detach().to("cpu").numpy()
        ptr = _ptr_to_np(calc._ptr)
        cvs, grads = self.cvs_and_grads(coord_np, ptr)
        V, dVdxi = self.engine.eval(cvs)                 # EXACT analytic (B,), (B,)
        dE = np.zeros(self.B, dtype=np.float64)
        for b in range(self.B):
            n = self._n[b]
            f_ha = -(dVdxi[b]) * grads[b]                # -dV/dxi * dxi/dx  [Ha/A]
            f_au = f_ha.reshape(-1) * HA_PER_ANG_TO_AU   # -> a.u. (Ha/Bohr)
            F[b, :3 * n] += torch.as_tensor(f_au, dtype=F.dtype, device=F.device)
            dE[b] = V[b]
            self.cv_history[b].append(float(cvs[b]))
            self.bias_history[b].append(float(V[b]))
        E = E + torch.as_tensor(dE, dtype=E.dtype, device=E.device)
        # deposit a hill per walker every ``pace`` force evaluations.
        if self.deposit and self.pace > 0 and (self._napply % self.pace == 0):
            self.engine.deposit(cvs)
            self.deposit_steps.append(self._napply)
        self._napply += 1
        return E, F


# --------------------------------------------------------------------------- #
#  Standalone numpy self-check (no torch / no GPU; no MLIP). Run:
#     PYTHONSAFEPATH=1 python -m maple.function.dispatcher.md.bias.batched_metad
#  Quick FD (gate-1 kernel) + WT-asymptote sanity + OPES bound sanity on a toy
#  1-D CV. The full a100 gates live in ensemble/_smoke_metad_batched.py.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from ase import Atoms

    ok = True
    kT = KB_HA_PER_K * 300.0

    # (1) FD: bias force -dV/dxi*dxi/dx == -d/dx of V(xi(x)) to ~1e-10 (analytic).
    a = Atoms("H2", positions=[[0.0, 0, 0], [2.5, 0, 0]])
    m = a.get_masses()
    g1, g2 = np.array([0]), np.array([1])
    eng = WTMetadEngine(kT, sigma=0.2, height=0.5 * HARTREE_PER_KCAL,
                        biasfactor=10.0, cv_min=1.0, cv_max=4.0, nbins=400)
    for s0 in (2.1, 2.6, 3.0):                    # deposit a few hills
        eng.deposit(np.array([s0]))
    pos = a.get_positions()
    xi0, grad = _com_distance_cv_grad(pos, g1, g2, m)
    _, dV = eng.eval(np.array([xi0]))
    f_analytic = -(dV[0]) * grad                  # (n,3) Ha/A  (= -dVbias/dx)
    h = 1e-6
    f_fd = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for c in range(3):
            pp = pos.copy(); pp[i, c] += h
            xp, _ = _com_distance_cv_grad(pp, g1, g2, m)
            Vp, _ = eng.eval(np.array([xp]))
            pm = pos.copy(); pm[i, c] -= h
            xm, _ = _com_distance_cv_grad(pm, g1, g2, m)
            Vm, _ = eng.eval(np.array([xm]))
            f_fd[i, c] = -(Vp[0] - Vm[0]) / (2 * h)   # -dV/dx
    fd_res = float(np.max(np.abs(f_analytic - f_fd)))
    ok &= fd_res < 1e-9
    print(f"(1) FD residual |F_analytic - F_fd| = {fd_res:.2e} (exp < 1e-9)")

    # (2) grid path agrees with analytic path.
    xg = np.linspace(1.2, 3.8, 50)
    Va, _ = eng.eval(xg)
    Vg, _ = eng.eval_grid(xg)
    ga = float(np.max(np.abs(Va - Vg)))
    ok &= ga < 1e-2
    print(f"(2) max|V_analytic - V_grid| = {ga:.2e} (exp small)")

    # (3) OPES: epsilon bounds |V| by dE, Z stays finite/positive over deposits.
    op = OPESMetadEngine(kT, sigma=0.2, biasfactor=10.0,
                         barrier=5.0 * HARTREE_PER_KCAL,
                         cv_min=1.0, cv_max=4.0, nbins=400)
    rng = np.random.default_rng(0)
    for _ in range(200):
        op.deposit(1.0 + 3.0 * rng.random(4))
    Vop, _ = op.eval(np.linspace(1.2, 3.8, 50))
    zmin, zmax = min(op.Z_history), max(op.Z_history)
    ok &= (zmin > 0) and np.isfinite(zmax) and (Vop.min() >= -5.0 * HARTREE_PER_KCAL - 1e-9)
    print(f"(3) OPES Z in [{zmin:.3e},{zmax:.3e}] (bounded, >0); "
          f"V_min={Vop.min()/HARTREE_PER_KCAL:.3f} >= -dE={-5.0:.1f} kcal/mol")

    print("BATCHED-METAD/OPES SELF-CHECK:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
