"""Extended-system Adaptive Biasing Force (eABF) + CZAR estimator for MAPLE's
batched MD core (rides ``ensemble/nvt_batched.py`` via the SAME additive bias hook
umbrella / metaD / GaMD use: ``BatchedNVT._forces_au`` -> ``job._bias.apply(E, F,
calc)`` once per forward; ``nvt_batched.py`` is NOT modified).

Method
------
Extended-system ABF (Lelièvre, Rousset & Stoltz 2007/2010; Comer, Gumbart, Hénin,
Lelièvre, Pohorille & Chipot, JPCB 2015, 10.1021/jp506633n) with the CZAR
estimator (Lesage, Lelièvre, Stoltz & Chipot, JPCB 2017, 10.1021/acs.jpcb.6b10055).

EXTENDED LAGRANGIAN.  Each replica carries a fictitious variable ``lambda`` coupled
to the physical collective variable ``xi(x)`` by a stiff harmonic spring::

    U_ext = 1/2 k (xi(x) - lambda)^2

The spring force acts on BOTH degrees of freedom:
  * physical coords:  F_i   = -k (xi - lambda) dxi/dx_i   (added to the force buffer)
  * extended var    :  F_lam = +k (xi - lambda)           (drives lambda's dynamics)
``lambda`` has its own mass ``m_lam`` and a Langevin thermostat at the system
temperature (BAOAB, advanced once per force evaluation with ``xi`` held fixed --
a first-order Trotter split of the physical/extended subsystems), so it explores
its coordinate as an extra, light, fast degree of freedom.

ABF ON lambda.  The running average of the force along ``lambda`` is accumulated
in bins; the NEGATIVE ramped running mean is applied back to ``lambda``, cancelling
the systematic force so ``lambda`` diffuses freely -> uniform sampling along
``lambda`` (hence along ``xi``, via the tight spring). ``ramp = min(1, n_j/
full_samples)`` gates the applied force until a bin is populated (the standard ABF
``fullSamples`` ramp, Darve 2008; Comer 2015).

CZAR ESTIMATOR.  The true PMF of the PHYSICAL CV ``xi`` is recovered from the joint
(xi, lambda) time series -- NOT from the ABF accumulator, which estimates A(lambda)
of the *smeared* extended variable::

    A'(xi) = -(1/beta) d ln rho~(xi)/dxi + k (<lambda>_xi - xi)

with ``rho~(xi)`` the biased histogram of ``xi`` and ``<lambda>_xi`` the mean of
``lambda`` conditioned on the ``xi`` bin. Integrating A'(xi) (cumulative trapezoid)
gives the PMF, zeroed at its minimum. CZAR is exact in the stiff-spring / well
sampled limit and robust to the bias applied to ``lambda``.

Batch axis = replicas
---------------------
Each replica shares the ONE batched ``calc.get_ef_gpu()`` forward and carries its
OWN ``lambda`` + ABF accumulator (per-replica, the default). ``shared_grid=True``
pools all replicas into ONE ABF grid = MULTI-WALKER eABF (free here: the walkers
co-reside in one batched forward + one address space, so the shared-bias coupling
needs no file/MPI sync -- cf. multiple-walker metaD in ``bias.batched_metad``). The
final CZAR PMF pools the (xi, lambda) samples across all replicas.

Units (real MLIP path): ``xi``, ``lambda`` [Angstrom]; ``k`` [Ha/Angstrom^2];
``kT`` [Ha]; ``dt`` [fs]; ``m_lam`` [Ha*fs^2/Angstrom^2]; ``v_lam`` [Angstrom/fs];
``lam_friction`` [1/fs]. The spring force on the physical atoms is converted
Ha/Angstrom -> a.u. (Ha/Bohr) via ``HA_PER_ANG_TO_AU`` (as every batch bias); the
lambda dynamics is self-contained and never touches the MD buffer. The toy 1-D
driver (:func:`eabf_toy_1d`) runs the IDENTICAL BAOAB/ABF/CZAR math in reduced
units (calc-free) -- the correctness anchor, mirroring ``bias/umbrella.py``.

Collective variable -- COM-COM distance between two atom groups (isolated, no
minimum image), reusing :func:`bias.batched_metad._com_distance_cv_grad` (value +
analytic spatial gradient dxi/dx), identical to the rest of the batch bias layer.
"""

import numpy as np

from .posres_calc import _build_selection      # reuse group selection (all/heavy/idx)
from .batched import _ptr_to_np                # reuse CUDA-aware _ptr coercion (C2)
from .batched_metad import _com_distance_cv_grad   # CV value + analytic dxi/dx
from ..utils import HA_PER_ANG_TO_AU           # Ha/Angstrom -> Ha/Bohr (a.u. force)
from .gamd import KB_HA_PER_K, HARTREE_PER_KCAL    # Boltzmann (Ha/K) + kcal->Ha

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
#  Running mean-force accumulator (the ABF bias along the extended variable)
# --------------------------------------------------------------------------- #
class ABFGrid:
    """1-D running mean-force accumulator = the ABF bias along ``lambda``.

    Per bin ``j`` keeps ``Sum F_lam`` (the instantaneous spring force on
    ``lambda``, ``k(xi-lambda)``) and the count ``n_j``. The mean force estimate
    is ``m_j = Sum F_j / n_j``; the applied biasing force is ``-m_j * ramp`` with
    ``ramp = min(1, n_j/full_samples)``. One grid per replica (default) or one
    shared across replicas (multi-walker eABF)."""

    def __init__(self, cv_min, cv_max, nbins, full_samples=200):
        self.cv_min = float(cv_min)
        self.cv_max = float(cv_max)
        self.nbins = int(nbins)
        if not (self.cv_max > self.cv_min) or self.nbins < 1:
            raise ValueError("ABFGrid needs cv_max > cv_min and nbins >= 1.")
        self.dx = (self.cv_max - self.cv_min) / self.nbins
        self.full = float(full_samples)
        self.force_sum = np.zeros(self.nbins, dtype=np.float64)
        self.count = np.zeros(self.nbins, dtype=np.float64)

    def _bin(self, x):
        j = int(np.floor((float(x) - self.cv_min) / self.dx))
        return min(max(j, 0), self.nbins - 1)

    def sample(self, lam, f_lam):
        """Accumulate the instantaneous force ``f_lam`` at extended-var ``lam``."""
        j = self._bin(lam)
        self.force_sum[j] += float(f_lam)
        self.count[j] += 1.0

    def bias_force(self, lam):
        """ABF biasing force at ``lam`` = -(ramped running mean force in its bin)."""
        j = self._bin(lam)
        n = self.count[j]
        if n <= 0.0:
            return 0.0
        mean_f = self.force_sum[j] / n
        ramp = min(1.0, n / self.full) if self.full > 0.0 else 1.0
        return -mean_f * ramp

    def mean_force(self):
        """Per-bin running mean force estimate (NaN where unvisited)."""
        m = np.full(self.nbins, np.nan)
        nz = self.count > 0.0
        m[nz] = self.force_sum[nz] / self.count[nz]
        return m

    def bin_centers(self):
        return self.cv_min + (np.arange(self.nbins) + 0.5) * self.dx


# --------------------------------------------------------------------------- #
#  CZAR estimator: true PMF of the physical CV xi from the joint (xi, lambda)
# --------------------------------------------------------------------------- #
def czar_pmf(xi_samples, lam_samples, k, beta, cv_min, cv_max, nbins):
    """CZAR PMF of the physical CV ``xi`` from joint (xi, lambda) samples.

    ``A'(xi) = -(1/beta) d ln rho~/dxi + k (<lambda>_xi - xi)`` (Lesage et al.
    2017); integrated (cumulative trapezoid over populated bins) -> A(xi), min 0.
    Returns ``(bin_centres, pmf)`` in the ENERGY UNIT implied by ``k`` and
    ``beta`` (Ha if ``k`` in Ha/Angstrom^2 and ``beta`` in 1/Ha; reduced units
    otherwise). Bins with no ``xi`` sample -> NaN in the PMF."""
    xi = np.asarray(xi_samples, dtype=np.float64).ravel()
    lam = np.asarray(lam_samples, dtype=np.float64).ravel()
    edges = np.linspace(float(cv_min), float(cv_max), int(nbins) + 1)
    xc = 0.5 * (edges[:-1] + edges[1:])
    dx = xc[1] - xc[0]
    idx = np.clip(np.digitize(xi, edges) - 1, 0, int(nbins) - 1)
    count = np.zeros(int(nbins), dtype=np.float64)
    lam_sum = np.zeros(int(nbins), dtype=np.float64)
    np.add.at(count, idx, 1.0)
    np.add.at(lam_sum, idx, lam)

    ntot = max(count.sum(), 1.0)
    rho = count / ntot / dx                                  # normalized biased density
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_mean = np.where(count > 0.0, lam_sum / np.maximum(count, 1.0), np.nan)
        ln_rho = np.where(rho > 0.0, np.log(rho), np.nan)
    # d ln rho / dxi via central finite difference on the interior bins.
    dln = np.full(int(nbins), np.nan)
    dln[1:-1] = (ln_rho[2:] - ln_rho[:-2]) / (2.0 * dx)
    # CZAR mean force A'(xi) = -(1/beta) d ln rho/dxi + k(<lambda>_xi - xi).
    grad_A = -(1.0 / float(beta)) * dln + float(k) * (lam_mean - xc)

    pmf = np.full(int(nbins), np.nan)
    finite = np.isfinite(grad_A)
    if int(finite.sum()) >= 2:
        xf = xc[finite]
        gf = grad_A[finite]
        Af = np.concatenate([[0.0],
                             np.cumsum(0.5 * (gf[1:] + gf[:-1]) * np.diff(xf))])
        Af = Af - np.nanmin(Af)
        pmf[finite] = Af
    return xc, pmf


# --------------------------------------------------------------------------- #
#  Batch-aware bias: rides BatchedNVT._forces_au (apply(E, F, calc) -> (E, F))
# --------------------------------------------------------------------------- #
class ExtendedABF:
    """Extended-system ABF bias + CZAR estimator on the BatchedNVT force hook.

    One extended variable ``lambda_b`` per replica on a COM-COM distance CV
    (isolated; ``dxi/dx`` via :func:`_com_distance_cv_grad`). The spring couples
    ``xi(x) <-> lambda``; ABF flattens ``A(lambda)``; CZAR recovers ``A(xi)``. The
    ABF grid is per-replica by default (``shared_grid=True`` -> one grid across all
    replicas = multi-walker eABF). ``apply_abf=False`` makes it a passive
    spring+logger (extended dynamics with NO flattening -- the reference for the
    flattening gate).

    Bias contract (see :mod:`bias.batched`): ``apply(E, F, calc) -> (E, F)``;
    ``E`` (B,) Ha, ``F`` (B, nmax_dof) a.u. modified IN PLACE, ``calc.coord``
    (N,3 Angstrom) + ``calc._ptr`` (B+1). The spring force added to the physical
    buffer is ``F_i = -k(xi-lambda) dxi/dx_i`` (a.u.); the energy added is
    ``1/2 k (xi-lambda)^2``. The extended variable is advanced (BAOAB) once per
    call; (xi, lambda) are logged for CZAR."""

    KIND = "eabf"

    def __init__(self, atoms_list, group1, group2, *, k, kT, dt_fs,
                 cv_min, cv_max, nbins=100, full_samples=200,
                 lam_tau_fs=100.0, lam_mass=None, lam_friction=0.01,
                 shared_grid=False, apply_abf=True, lam0=None, seed=None):
        self.B = len(atoms_list)
        if self.B == 0:
            raise ValueError("ExtendedABF needs >= 1 replica.")
        self._g1 = [np.asarray(_build_selection(group1, at), dtype=int)
                    for at in atoms_list]
        self._g2 = [np.asarray(_build_selection(group2, at), dtype=int)
                    for at in atoms_list]
        self._m = [np.asarray(at.get_masses(), dtype=np.float64) for at in atoms_list]
        self._n = [len(at) for at in atoms_list]
        for b in range(self.B):
            if len(self._g1[b]) == 0 or len(self._g2[b]) == 0:
                raise ValueError("eABF CV groups must be non-empty.")
            if set(self._g1[b].tolist()) & set(self._g2[b].tolist()):
                raise ValueError("eABF CV groups must not overlap.")
        self.k = float(k)                                    # Ha/Angstrom^2
        self.kT = float(kT)                                  # Ha
        self.beta = 1.0 / self.kT
        self.dt_fs = float(dt_fs)                            # fs
        self.cv_min, self.cv_max = float(cv_min), float(cv_max)
        self.nbins = int(nbins)
        self.full_samples = float(full_samples)
        self.lam_friction = float(lam_friction)              # 1/fs
        self.apply_abf = bool(apply_abf)
        self.shared_grid = bool(shared_grid)
        self._lam0 = None if lam0 is None else float(lam0)
        # m_lam: explicit, else from the desired lambda oscillation period tau
        # (harmonic-oscillator omega=sqrt(k/m_lam)): m_lam = k (tau/2pi)^2, so a
        # stiffer spring gets a heavier extended mass (fixed period). Units
        # [Ha/Angstrom^2]*[fs^2] = Ha*fs^2/Angstrom^2 -> a=F/m_lam in Angstrom/fs^2.
        if lam_mass is not None:
            self.m_lam = float(lam_mass)
        else:
            self.m_lam = self.k * (float(lam_tau_fs) / TWO_PI) ** 2
        if self.m_lam <= 0.0:
            raise ValueError("ExtendedABF: m_lam must be > 0 (check k / lam_tau_fs).")
        # OU coefficients for the lambda Langevin (BAOAB middle step).
        self._c1 = float(np.exp(-self.lam_friction * self.dt_fs))
        self._c2 = float(np.sqrt(max(1.0 - self._c1 * self._c1, 0.0)
                                 * self.kT / self.m_lam))
        self._rng = np.random.default_rng(seed)
        # ABF grid(s): shared (multi-walker) or per-replica (default).
        if self.shared_grid:
            self._grid = ABFGrid(cv_min, cv_max, nbins, full_samples)
            self._grids = [self._grid] * self.B
        else:
            self._grids = [ABFGrid(cv_min, cv_max, nbins, full_samples)
                           for _ in range(self.B)]
        # extended-variable state (lazily initialized to xi(x0) on the first call).
        self._lam = [None] * self.B
        self._vlam = [0.0] * self.B
        # per-replica logs (appended once per force evaluation).
        self.xi_history = [[] for _ in range(self.B)]
        self.lam_history = [[] for _ in range(self.B)]
        self._napply = 0

    # --------------------------------------------------- extended-var integrator
    def _lambda_force(self, b, xi, lam):
        """Total force on ``lambda_b`` [Ha/Angstrom]: spring k(xi-lambda) plus the
        ABF biasing force (negative ramped running mean, if ``apply_abf``)."""
        f = self.k * (xi - lam)
        if self.apply_abf:
            f += self._grids[b].bias_force(lam)
        return f

    def _advance_lambda(self, b, xi, lam, v):
        """One BAOAB Langevin step for ``lambda_b`` (``xi`` held fixed within the
        call; a first-order Trotter split of the physical/extended subsystems)."""
        dt, m = self.dt_fs, self.m_lam
        v = v + 0.5 * dt * self._lambda_force(b, xi, lam) / m       # B
        lam = lam + 0.5 * dt * v                                     # A
        v = self._c1 * v + self._c2 * self._rng.standard_normal()    # O
        lam = lam + 0.5 * dt * v                                     # A
        v = v + 0.5 * dt * self._lambda_force(b, xi, lam) / m        # B
        return lam, v

    # ---------------------------------------------------------------- CV + grad
    def cvs_and_grads(self, coord_np, ptr):
        """Per-replica (CV value, CV spatial gradient (n,3)) from master coords."""
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
        """Add the spring force to the padded buffer + advance every ``lambda``.

        E : (B,) Ha ; F : (B, nmax_dof) a.u. (Ha/Bohr) modified IN PLACE ;
        ``calc.coord`` (N,3 Angstrom) + ``calc._ptr`` (B+1)."""
        import torch
        coord_np = calc.coord.detach().to("cpu").numpy()
        ptr = _ptr_to_np(calc._ptr)
        cvs, grads = self.cvs_and_grads(coord_np, ptr)
        dE = np.zeros(self.B, dtype=np.float64)
        for b in range(self.B):
            xi = float(cvs[b])
            if self._lam[b] is None:                      # lazy init lambda = xi(x0)
                self._lam[b] = xi if self._lam0 is None else self._lam0
                self._vlam[b] = float(self._rng.standard_normal()
                                      * np.sqrt(self.kT / self.m_lam))
            lam = self._lam[b]
            diff = xi - lam                               # xi - lambda
            # (1) spring force on the PHYSICAL coords at the CURRENT lambda.
            n = self._n[b]
            f_ha = -(self.k * diff) * grads[b]            # (n,3) Ha/Angstrom
            f_au = f_ha.reshape(-1) * HA_PER_ANG_TO_AU    # -> a.u. (Ha/Bohr)
            F[b, :3 * n] += torch.as_tensor(f_au, dtype=F.dtype, device=F.device)
            dE[b] = 0.5 * self.k * diff * diff            # 1/2 k (xi-lambda)^2  [Ha]
            # (2) accumulate ABF + CZAR at the ENTRY state (xi(t), lambda(t)).
            self._grids[b].sample(lam, self.k * diff)     # instantaneous force on lambda
            self.xi_history[b].append(xi)
            self.lam_history[b].append(lam)
            # (3) advance lambda one BAOAB step (xi held fixed at xi(t)).
            self._lam[b], self._vlam[b] = self._advance_lambda(b, xi, lam,
                                                               self._vlam[b])
        E = E + torch.as_tensor(dE, dtype=E.dtype, device=E.device)
        self._napply += 1
        return E, F

    # ----------------------------------------------------------------- CZAR PMF
    def czar(self, eq_frac=0.1, per_replica=False):
        """CZAR PMF from the logged (xi, lambda) samples.

        Default pools ALL replicas (multi-walker CZAR). ``per_replica=True``
        returns a list of ``(xc, pmf)`` (one per replica). ``eq_frac`` drops the
        leading fraction of each replica's series as equilibration."""
        def _series(b):
            xi = np.asarray(self.xi_history[b], dtype=np.float64)
            lam = np.asarray(self.lam_history[b], dtype=np.float64)
            if eq_frac > 0.0 and xi.size:
                s = int(eq_frac * xi.size)
                xi, lam = xi[s:], lam[s:]
            return xi, lam

        if per_replica:
            return [czar_pmf(*_series(b), self.k, self.beta,
                             self.cv_min, self.cv_max, self.nbins)
                    for b in range(self.B)]
        xis, lams = [], []
        for b in range(self.B):
            xi, lam = _series(b)
            xis.append(xi)
            lams.append(lam)
        return czar_pmf(np.concatenate(xis) if xis else np.zeros(0),
                        np.concatenate(lams) if lams else np.zeros(0),
                        self.k, self.beta, self.cv_min, self.cv_max, self.nbins)

    def lambda_histogram(self, eq_frac=0.1):
        """Pooled histogram (counts) of ``lambda`` over [cv_min, cv_max] -- the
        sampling-flatness diagnostic (flatter with ABF on)."""
        edges = np.linspace(self.cv_min, self.cv_max, self.nbins + 1)
        lams = []
        for b in range(self.B):
            lam = np.asarray(self.lam_history[b], dtype=np.float64)
            if eq_frac > 0.0 and lam.size:
                lam = lam[int(eq_frac * lam.size):]
            lams.append(lam)
        allx = np.concatenate(lams) if lams else np.zeros(0)
        hist, _ = np.histogram(allx, bins=edges)
        return 0.5 * (edges[:-1] + edges[1:]), hist


def attach_eabf(job, group1, group2, *, k, cv_min, cv_max, nbins=100,
                full_samples=200, lam_tau_fs=100.0, lam_mass=None,
                lam_friction=0.01, shared_grid=False, apply_abf=True,
                lam0=None, seed=None):
    """Build an :class:`ExtendedABF` from a batched-MD ``job`` and attach it on the
    batch bias hook (``job._bias``), reading ``kT``/``dt`` from ``job.params``.

    Returns the bias. ``k`` in Ha/Angstrom^2; ``cv_min``/``cv_max`` in Angstrom.
    Mirrors the sanctioned pattern (cf. ``BatchedUmbrella`` setting ``self._bias``);
    no MD-loop change -- ``BatchedNVT._forces_au`` picks the bias up automatically."""
    kT = KB_HA_PER_K * float(job.params.temperature)         # Ha
    bias = ExtendedABF(job.atoms_list, group1, group2, k=k, kT=kT,
                       dt_fs=float(job.params.timestep),
                       cv_min=cv_min, cv_max=cv_max, nbins=nbins,
                       full_samples=full_samples, lam_tau_fs=lam_tau_fs,
                       lam_mass=lam_mass, lam_friction=lam_friction,
                       shared_grid=shared_grid, apply_abf=apply_abf,
                       lam0=lam0, seed=seed)
    job._bias = bias
    return bias


# --------------------------------------------------------------------------- #
#  Self-contained numpy eABF driver on a toy 1-D potential (CALC-FREE anchor)
# --------------------------------------------------------------------------- #
def eabf_toy_1d(U, dU, *, x0, m_x, m_lam, k, kT, dt, gamma_x, gamma_lam,
                cv_min, cv_max, nbins, nsteps, full_samples=200,
                apply_abf=True, seed=0, equil_frac=0.1):
    """eABF on a 1-D toy potential ``U(x)`` with the IDENTITY CV ``xi = x``
    (``dxi/dx = 1``). Physical ``x``: BAOAB Langevin (mass ``m_x``, friction
    ``gamma_x``) under ``U(x)`` + spring ``-k(x-lambda)``. Extended ``lambda``:
    BAOAB Langevin (mass ``m_lam``, friction ``gamma_lam``) under the spring + ABF.

    Runs the IDENTICAL ABF / CZAR / BAOAB math as :class:`ExtendedABF` (reduced
    units), so the correctness gates exercise the production code path without a
    calculator/GPU (mirrors ``bias/umbrella.py``'s analytic WHAM self-test).
    Deterministic (seeded). Returns a dict with the post-equilibration ``xi`` /
    ``lambda`` series, the :class:`ABFGrid`, and the CZAR PMF ``(czar_x, czar_pmf)``."""
    rng = np.random.default_rng(seed)
    grid = ABFGrid(cv_min, cv_max, nbins, full_samples)
    x = float(x0)
    vx = float(rng.standard_normal() * np.sqrt(kT / m_x))
    lam = float(x0)
    vlam = float(rng.standard_normal() * np.sqrt(kT / m_lam))
    cx1 = np.exp(-gamma_x * dt)
    cx2 = np.sqrt(max(1.0 - cx1 * cx1, 0.0) * kT / m_x)
    cl1 = np.exp(-gamma_lam * dt)
    cl2 = np.sqrt(max(1.0 - cl1 * cl1, 0.0) * kT / m_lam)
    xi_hist = np.empty(nsteps, dtype=np.float64)
    lam_hist = np.empty(nsteps, dtype=np.float64)

    def f_lambda(ll, xi):
        f = k * (xi - ll)
        if apply_abf:
            f += grid.bias_force(ll)
        return f

    for t in range(nsteps):
        xi = x                                       # identity CV: xi = x, dxi/dx = 1
        diff = xi - lam
        # accumulate ABF + CZAR at the entry state (xi(t), lambda(t)).
        grid.sample(lam, k * diff)
        xi_hist[t] = xi
        lam_hist[t] = lam
        # --- physical x: BAOAB under U(x) + spring -k(x - lambda) ---
        fx0 = -dU(x) - k * (x - lam)
        vx = vx + 0.5 * dt * fx0 / m_x
        x = x + 0.5 * dt * vx
        vx = cx1 * vx + cx2 * rng.standard_normal()
        x = x + 0.5 * dt * vx
        fx1 = -dU(x) - k * (x - lam)
        vx = vx + 0.5 * dt * fx1 / m_x
        # --- extended lambda: BAOAB under spring + ABF (xi held at xi(t)) ---
        vlam = vlam + 0.5 * dt * f_lambda(lam, xi) / m_lam
        lam = lam + 0.5 * dt * vlam
        vlam = cl1 * vlam + cl2 * rng.standard_normal()
        lam = lam + 0.5 * dt * vlam
        vlam = vlam + 0.5 * dt * f_lambda(lam, xi) / m_lam

    n0 = int(equil_frac * nsteps)
    xi_p, lam_p = xi_hist[n0:], lam_hist[n0:]
    xc, pmf = czar_pmf(xi_p, lam_p, k, 1.0 / kT, cv_min, cv_max, nbins)
    return dict(xi=xi_p, lam=lam_p, grid=grid, czar_x=xc, czar_pmf=pmf)


def _langevin_window(U, dU, center, kappa, *, x0, m_x, kT, dt, gamma, nsteps,
                     seed, equil_frac=0.2):
    """Short BAOAB Langevin of physical ``x`` under ``U(x)`` + harmonic restraint
    ``1/2 kappa (x - center)^2`` -> one umbrella window's ``x`` samples (post-equil).
    Used only to build the umbrella+WHAM REFERENCE PMF for the CZAR-parity gate."""
    rng = np.random.default_rng(seed)
    x = float(x0)
    v = float(rng.standard_normal() * np.sqrt(kT / m_x))
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt(max(1.0 - c1 * c1, 0.0) * kT / m_x)
    xs = np.empty(nsteps, dtype=np.float64)
    for t in range(nsteps):
        xs[t] = x
        f0 = -dU(x) - kappa * (x - center)
        v = v + 0.5 * dt * f0 / m_x
        x = x + 0.5 * dt * v
        v = c1 * v + c2 * rng.standard_normal()
        x = x + 0.5 * dt * v
        f1 = -dU(x) - kappa * (x - center)
        v = v + 0.5 * dt * f1 / m_x
    return xs[int(equil_frac * nsteps):]


def umbrella_wham_reference(U, dU, *, cv_min, cv_max, nwin, kappa, m_x, kT, dt,
                            gamma, nsteps, nbins, seed=0):
    """Umbrella-sampling + WHAM PMF of ``U(x)`` (identity CV) -- the INDEPENDENT
    reference for the CZAR-parity gate. Reuses the EXISTING WHAM
    (:func:`bias.umbrella.wham_1d`, NOT reimplemented). Reduced units: the
    ``temperature`` passed to ``wham_1d`` is chosen so its internal
    ``kbt = KCAL_PER_MOL_K * T`` equals the reduced ``kT`` here. Returns
    ``(bin_centres, pmf)`` in reduced energy units (min shifted to 0)."""
    from .umbrella import make_windows, wham_1d, KCAL_PER_MOL_K
    centers = make_windows(cv_min, cv_max, nwin)
    samples = [_langevin_window(U, dU, c, kappa, x0=c, m_x=m_x, kT=kT, dt=dt,
                                gamma=gamma, nsteps=nsteps, seed=seed + i)
               for i, c in enumerate(centers)]
    T_reduced = kT / KCAL_PER_MOL_K                  # => kbt = KCAL_PER_MOL_K*T = kT
    xb, pmf = wham_1d(centers, kappa, samples, T_reduced, nbins=nbins,
                      cv_range=(cv_min, cv_max))
    return xb, pmf


# --------------------------------------------------------------------------- #
#  Standalone numpy self-check (no torch / no GPU / no MLIP). Run:
#     PYTHONSAFEPATH=1 python -m maple.function.dispatcher.md.bias.eabf
#  Quick spring-force FD (gate-1 kernel) + a short toy CZAR sanity. The full 4
#  gates + the a100 pipeline smoke live in ensemble/../bias/_test_eabf.py.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from ase import Atoms

    ok = True
    kT = KB_HA_PER_K * 300.0

    # (1) FD: spring force -k(xi-lambda) dxi/dx == -d/dx of U_ext = 1/2 k (xi-lam)^2
    #     to < 1e-6 (lambda held fixed). COM-COM distance CV on an H2 replica.
    a = Atoms("H2", positions=[[0.0, 0, 0], [2.5, 0, 0]])
    m = a.get_masses()
    g1, g2 = np.array([0]), np.array([1])
    k = 1.5                                          # Ha/Angstrom^2
    lam = 2.0                                        # fixed extended-var value
    pos = a.get_positions()
    xi0, grad = _com_distance_cv_grad(pos, g1, g2, m)
    f_analytic = -(k * (xi0 - lam)) * grad           # (n,3) Ha/Angstrom
    h = 1e-6
    f_fd = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for c in range(3):
            pp = pos.copy(); pp[i, c] += h
            xp, _ = _com_distance_cv_grad(pp, g1, g2, m)
            Up = 0.5 * k * (xp - lam) ** 2
            pm = pos.copy(); pm[i, c] -= h
            xm, _ = _com_distance_cv_grad(pm, g1, g2, m)
            Um = 0.5 * k * (xm - lam) ** 2
            f_fd[i, c] = -(Up - Um) / (2 * h)        # -dU_ext/dx
    fd_res = float(np.max(np.abs(f_analytic - f_fd)))
    ok &= fd_res < 1e-6
    print(f"(1) spring-force FD residual = {fd_res:.2e} (exp < 1e-6)")

    # (2) toy double-well CZAR sanity: PMF finite + recovers a barrier > 0.
    h_b, a_w = 4.0 * kT, 1.0
    def U(x):  return h_b * ((x / a_w) ** 2 - 1.0) ** 2
    def dU(x): return h_b * 2.0 * ((x / a_w) ** 2 - 1.0) * (2.0 * x / a_w ** 2)
    res = eabf_toy_1d(U, dU, x0=-a_w, m_x=1.0, m_lam=1.0, k=200.0 * kT / a_w ** 2,
                      kT=kT, dt=0.002, gamma_x=5.0, gamma_lam=5.0,
                      cv_min=-1.6 * a_w, cv_max=1.6 * a_w, nbins=48,
                      nsteps=60000, full_samples=100, seed=0)
    pmf = res["czar_pmf"]
    finite = np.isfinite(pmf)
    ok &= bool(finite.any()) and (np.nanmax(pmf) - np.nanmin(pmf) > 0.5 * kT)
    print(f"(2) toy CZAR PMF: finite bins={int(finite.sum())}/{pmf.size}  "
          f"range={np.nanmax(pmf) - np.nanmin(pmf):.4f} (kT={kT:.3e})")

    print("eABF/CZAR SELF-CHECK:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
