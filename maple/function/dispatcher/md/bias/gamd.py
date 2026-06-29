"""Gaussian Accelerated MD (GaMD) boost for MAPLE MD — pure-MLIP form.

Same single-injection-point architecture as the rest of the bias layer
(:mod:`plumed_calc` / :mod:`colvars_calc` / :mod:`posres_calc`): a calculator
wrapper recomputes the boost once per MD step at ``atoms.get_forces`` and folds
it into the MLIP force, so NVE/NVT/NPT all gain GaMD with **no integrator
change** and it composes with any active PLUMED/Colvars/posres wrapper.

Why GaMD here
-------------
GaMD (Miao, Feher, McCammon, JCTC 2015, 10.1021/acs.jctc.5b00436) is a
**collective-variable-free** enhanced-sampling method: it adds a harmonic boost
to the potential wherever the energy is below a threshold, flattening barriers
without any pre-chosen reaction coordinate. That complements the PLUMED backend
(metadynamics / OPES need a CV) for systems whose slow CV is unknown a priori —
e.g. enzyme conformational change, the ultimate MAPLE MACE-OFF target.

Pure-MLIP reduction (the key simplification)
--------------------------------------------
GaMD boost potential, applied to the *total* potential energy V(r) when V<E::

    ΔV(r) = 1/2 · k · (E - V(r))²            (0 otherwise)

On a pure-MLIP engine the boost force is just the physical force scaled by a
scalar — no force-field energy decomposition needed::

    F_boost = -dΔV/dr = -k(E-V)·(dV/dr) = +k(E-V)·F_phys      (since F_phys=-dV/dr)
    F_total = F_phys + F_boost = F_phys · (1 - k(E - V))

The constant ``k`` is bounded so the factor (1 - k(E-V)) stays in [0, 1] (the
boost never flips a force, keeps the distribution near-Gaussian, and stays
analytically reweightable). So GaMD in a "atoms.get_forces() every step" engine
is literally: evaluate V, and if V<E multiply the force by (1 - k(E-V)).

Parameters from short-MD statistics (Miao 2015, eqs 7-11)
--------------------------------------------------------
Collect Vmin, Vmax, Vavg, σV over a conventional-MD prep window, then::

    k0  ≤ 1 (effective force-constant ratio); k = k0 / (Vmax - Vmin)
    lower-bound (E = Vmax):  k0 = min(1, (σ0/σV)·(Vmax-Vmin)/(Vmax-Vavg))
    upper-bound (E = Vmin + (Vmax-Vmin)/k0):
                             k0 = (1 - σ0/σV)·(Vmax-Vmin)/(Vavg-Vmin)
                             valid only if 0 < k0 ≤ 1, else fall back to lower.

``σ0`` caps the standard deviation of the boost (anharmonicity control); a
typical value is 6 kcal/mol. Here it is carried in **Hartree** internally
(``GamdCalculator`` accepts kcal/mol and converts), matching MAPLE's Ha units.

Reweighting (Miao, Sinko, Walker, JCTC 2014, 10.1021/ct500090q)
---------------------------------------------------------------
Each frame's ΔV is logged; the unbiased PMF along any reaction coordinate is
recovered by cumulant expansion to 2nd order (CE2)::

    F(A_j) = -kT·ln p*(A_j) - ⟨ΔV⟩_j - (β/2)·σ²_{ΔV,j} + C

(``p*`` = biased histogram; the two correction terms are C1=⟨ΔV⟩, C2=(β/2)σ²,
the 1st/2nd cumulants). :func:`gamd_reweight_1d` does this; ``maclaurin`` mode
gives the alternative exp-average expansion.

Not portable: LiGaMD's *selective* boost (only the ligand non-bonded term) needs
a force-field energy decomposition the MLIP does not expose — plain GaMD (boost
the whole MLIP potential) is the portable variant and is what this implements.
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

HARTREE_PER_KCAL = 1.0 / 627.5094740631      # kcal/mol → Ha
KB_HA_PER_K = 8.617333262e-5 / 27.211386245988  # Boltzmann const, Ha/K


def gamd_enabled(gamd):
    """True if a gamd setting requests boosting (falsey/'off' ⇒ disabled)."""
    if gamd is None:
        return False
    if isinstance(gamd, bool):
        return gamd
    if isinstance(gamd, str):
        return gamd.strip().lower() not in ("", "off", "no", "false", "none", "0")
    return bool(gamd)


def gamd_params(Vmin, Vmax, Vavg, sigmaV, sigma0, mode="lower"):
    """GaMD boost params from energy statistics (all energies same unit, e.g. Ha).

    Returns dict {mode, k0, k, E, Vmin, Vmax}. ``mode`` is the *effective* mode
    after the upper→lower fallback. ``k`` has unit 1/energy; ``E`` is the
    threshold in the energy unit of the inputs. See Miao 2015 eqs 7-11.
    """
    Vmin, Vmax, Vavg, sigmaV = map(float, (Vmin, Vmax, Vavg, sigmaV))
    sigma0 = float(sigma0)
    span = Vmax - Vmin
    if span <= 0.0:
        # degenerate (flat/insufficient sampling): no boost
        return {"mode": "off", "k0": 0.0, "k": 0.0, "E": Vmax,
                "Vmin": Vmin, "Vmax": Vmax}

    eff = mode.strip().lower() if isinstance(mode, str) else "lower"
    if eff == "upper":
        denom = Vavg - Vmin
        if denom > 0.0:
            k0 = (1.0 - sigma0 / sigmaV) * span / denom if sigmaV > 0 else 1.0
            if 0.0 < k0 <= 1.0:
                E = Vmin + span / k0
                return {"mode": "upper", "k0": k0, "k": k0 / span, "E": E,
                        "Vmin": Vmin, "Vmax": Vmax}
        eff = "lower"  # fall back

    # lower bound: E = Vmax
    denom = Vmax - Vavg
    if sigmaV > 0.0 and denom > 0.0:
        k0 = min(1.0, (sigma0 / sigmaV) * span / denom)
    else:
        k0 = 1.0
    k0 = max(0.0, min(1.0, k0))
    return {"mode": "lower", "k0": k0, "k": k0 / span, "E": Vmax,
            "Vmin": Vmin, "Vmax": Vmax}


class _Welford:
    """Running min/max/mean/variance (Welford) over scalar energies."""

    __slots__ = ("n", "mean", "M2", "vmin", "vmax")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.vmin = np.inf
        self.vmax = -np.inf

    def push(self, x):
        x = float(x)
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)
        if x < self.vmin:
            self.vmin = x
        if x > self.vmax:
            self.vmax = x

    @property
    def sigma(self):
        return (self.M2 / self.n) ** 0.5 if self.n > 1 else 0.0


class GamdCalculator(Calculator):
    """Wrap a MAPLE MLIP (or bias) calculator with a GaMD total-potential boost.

    Protocol (Miao 2015): the first ``prep_steps`` force evaluations run
    *conventional* MD (no boost) while collecting Vmin/Vmax/Vavg/σV; the boost
    params (k, E) are then frozen and the boost is applied for the rest of the
    run. Supply ``params=`` to skip collection and use fixed params from a prior
    prep stage. Each step logs (step, V, ΔV, k, E, phase) to ``{output}_gamd.dat``
    and accumulates ΔV in ``self.dv_history`` (Ha) for reweighting.
    """

    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

    def __init__(self, inner, *, mode="lower", sigma0_kcal=6.0, prep_steps=2000,
                 temperature=300.0, params=None, atoms=None, output='run',
                 restart_step=0, log=True):
        Calculator.__init__(self)
        self.inner = inner
        self.SUPPORTS_PBC = getattr(inner, 'SUPPORTS_PBC', False)
        self._mode = mode
        self._sigma0 = float(sigma0_kcal) * HARTREE_PER_KCAL   # Ha
        self._prep = max(int(prep_steps), 0)
        self._kT = KB_HA_PER_K * float(temperature)            # Ha
        self._istep = int(restart_step)
        self._output = output
        self._log = bool(log)
        self._stats = _Welford()
        self._params = dict(params) if params else None        # frozen if given
        self.dv_history = []                                   # Ha per step
        if atoms is not None:
            self.atoms = atoms.copy()
        self._logfile = None

    # -- boost params: collect during prep, freeze once -------------------- #
    def _ensure_params(self, V):
        if self._params is not None:
            return self._params
        self._stats.push(V)
        if self._istep + 1 >= self._prep and self._stats.n > 1:
            self._params = gamd_params(self._stats.vmin, self._stats.vmax,
                                       self._stats.mean, self._stats.sigma,
                                       self._sigma0, self._mode)
        return self._params  # None ⇒ still in prep (no boost yet)

    def _write_log(self, V, dV, k, E, phase):
        if not self._log:
            return
        if self._logfile is None:
            self._logfile = open(f"{self._output}_gamd.dat", "a")
            self._logfile.write("# step  V[Ha]  dV[Ha]  k[1/Ha]  E[Ha]  phase\n")
        self._logfile.write(f"{self._istep} {V:.10g} {dV:.10g} {k:.6g} "
                            f"{E:.10g} {phase}\n")
        self._logfile.flush()

    def calculate(self, atoms=None, properties=('energy', 'forces'),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        atoms = self.atoms

        self.inner.calculate(atoms, list(properties), system_changes)
        V = float(self.inner.results['energy'])                       # Ha
        forces = np.array(self.inner.results['forces'], np.float64)   # Ha/Å (copy)

        p = self._ensure_params(V)
        dV = 0.0
        k = 0.0
        E = V
        phase = "prep"
        if p is not None and p["k"] > 0.0:
            k, E = p["k"], p["E"]
            phase = "prod"
            if V < E:
                factor = 1.0 - k * (E - V)                # ∈ [0, 1] by construction
                forces *= factor
                dV = 0.5 * k * (E - V) ** 2               # Ha
        self._write_log(V, dV, k, E, phase)
        self.dv_history.append(dV)
        self._istep += 1

        self.results = {
            'energy': V + dV,
            'free_energy': V + dV,
            'forces': forces,
            'gamd_boost': dV,
            'gamd_phase': phase,
        }
        if 'stress' in self.inner.results:
            # boost scales the configurational force; the virial scales the same
            # way in production. Prep ⇒ untouched.
            s = np.asarray(self.inner.results['stress'], np.float64)
            self.results['stress'] = s * (1.0 - k * (E - V)) if (phase == "prod"
                                                                 and V < E) else s

    def __del__(self):
        try:
            if self._logfile is not None:
                self._logfile.close()
        except Exception:
            pass


def gamd_reweight_1d(cv, dV_ha, temperature=300.0, bins=50, mode="ce2"):
    """Recover the unbiased 1-D PMF from a GaMD run (kcal/mol).

    ``cv``    : (M,) reaction-coordinate value per frame (production frames only).
    ``dV_ha`` : (M,) boost ΔV per frame in Hartree (``GamdCalculator.dv_history``).
    Returns (centers, pmf_kcal). ``ce2`` = cumulant expansion to 2nd order
    (default, robust near-Gaussian ΔV); ``maclaurin`` = exp-average reweight.
    """
    cv = np.asarray(cv, np.float64)
    dV = np.asarray(dV_ha, np.float64)
    kT = KB_HA_PER_K * float(temperature)          # Ha
    beta = 1.0 / kT
    edges = np.histogram_bin_edges(cv, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    which = np.clip(np.digitize(cv, edges) - 1, 0, len(centers) - 1)
    pmf = np.full(len(centers), np.nan)
    for j in range(len(centers)):
        m = which == j
        nj = int(m.sum())
        if nj == 0:
            continue
        pstar = nj / len(cv)
        if mode == "maclaurin":
            # ⟨exp(βΔV)⟩_j via Maclaurin series to 2nd order
            x = beta * dV[m]
            avg = np.mean(1.0 + x + 0.5 * x * x)
            corr_ha = kT * np.log(max(avg, 1e-300))
        else:  # ce2
            dvm = dV[m].mean()
            dvv = dV[m].var()
            corr_ha = dvm + 0.5 * beta * dvv         # C1 + C2, in Ha
        pmf[j] = (-kT * np.log(pstar) - corr_ha)     # Ha
    pmf -= np.nanmin(pmf)
    return centers, pmf / HARTREE_PER_KCAL           # → kcal/mol


# --------------------------------------------------------------------------- #
# Standalone numpy/ASE self-check (no maple import, no GPU). Run:
#   PYTHONSAFEPATH=1 python gamd.py
# (1) param eqs + force-factor bound; (2) ΔV & force-scaling identity on a toy
# inner calc; (3) CE2 reweighting analytically recovers a harmonic PMF from
# boosted samples.
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    from ase import Atoms

    ok = True

    # (1) param equations + factor ∈ [0,1] over the sampled energy range.
    Vmin, Vmax, Vavg, sV = -100.0, -90.0, -95.0, 2.0
    pl = gamd_params(Vmin, Vmax, Vavg, sV, sigma0=1.0, mode="lower")
    assert pl["mode"] == "lower" and pl["E"] == Vmax
    assert 0.0 < pl["k0"] <= 1.0
    fac = [1.0 - pl["k"] * (pl["E"] - V) for V in (Vmin, Vavg, Vmax)]
    ok &= all(-1e-12 <= f <= 1.0 + 1e-12 for f in fac)
    pu = gamd_params(Vmin, Vmax, Vavg, sV, sigma0=1.0, mode="upper")
    ok &= (0.0 < pu["k0"] <= 1.0) and pu["E"] >= Vmax - 1e-9
    print(f"(1) lower k0={pl['k0']:.4f} E={pl['E']:.1f} factors={['%.3f'%f for f in fac]} "
          f"| upper k0={pu['k0']:.4f} E={pu['E']:.2f}")

    # (2) ΔV = 1/2 k (E-V)² and F_tot = F_phys·(1 - k(E-V)) on a toy inner calc.
    class _LinCalc(Calculator):
        implemented_properties = ['energy', 'forces']

        def __init__(self, V0, F0):
            Calculator.__init__(self)
            self._V0, self._F0 = V0, F0

        def calculate(self, atoms=None, properties=('energy', 'forces'),
                      system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            n = len(self.atoms)
            self.results = {'energy': self._V0,
                            'forces': np.tile(self._F0, (n, 1))}

    a = Atoms('H', positions=[[0., 0, 0]])
    Vtest = -95.0
    a.calc = _LinCalc(Vtest, np.array([1.0, 0.0, 0.0]))
    g = GamdCalculator(a.calc, params=pl, atoms=a, output='/tmp/_gamd_st',
                       log=False)
    a.calc = g
    f = a.get_forces()
    e = a.get_potential_energy()
    factor_exp = 1.0 - pl["k"] * (pl["E"] - Vtest)
    dV_exp = 0.5 * pl["k"] * (pl["E"] - Vtest) ** 2
    ok &= abs(f[0, 0] - factor_exp) < 1e-12
    ok &= abs(e - (Vtest + dV_exp)) < 1e-10
    ok &= abs(g.dv_history[-1] - dV_exp) < 1e-12
    print(f"(2) factor={f[0,0]:.6f} (exp {factor_exp:.6f})  "
          f"ΔV={g.dv_history[-1]:.6e} (exp {dV_exp:.6e})")

    # (3) CE2 reweight recovers a harmonic PMF V(x)=½κx² from boosted samples.
    #     Draw x ∝ exp(-β(V+ΔV)) on a grid, assign βΔV, reweight, compare.
    rng = np.random.default_rng(0)
    T = 300.0
    kT = KB_HA_PER_K * T
    kappa = 8.0 * kT                       # Ha/Å² ⇒ unbiased std = sqrt(kT/κ)
    xs = np.linspace(-4.0, 4.0, 4001)      # Å grid
    Vx = 0.5 * kappa * xs ** 2             # Ha
    # GaMD params from the *unbiased* energy stats on this grid (Boltzmann-weighted)
    wb = np.exp(-(Vx - Vx.min()) / kT)
    wb /= wb.sum()
    Vavg_g = float((Vx * wb).sum())
    sV_g = float(np.sqrt(((Vx - Vavg_g) ** 2 * wb).sum()))
    pg = gamd_params(Vx.min(), Vx.max(), Vavg_g, sV_g,
                     sigma0=3.0 * kT, mode="lower")
    k, E = pg["k"], pg["E"]
    dVx = np.where(Vx < E, 0.5 * k * (E - Vx) ** 2, 0.0)        # Ha
    pstar = np.exp(-(Vx + dVx) / kT)
    pstar /= pstar.sum()
    idx = rng.choice(len(xs), size=400000, p=pstar)            # boosted samples
    cv = xs[idx]
    dV_samp = dVx[idx]
    centers, pmf = gamd_reweight_1d(cv, dV_samp, temperature=T, bins=60,
                                    mode="ce2")
    true_pmf = (0.5 * kappa * centers ** 2) / HARTREE_PER_KCAL  # kcal/mol
    true_pmf -= np.nanmin(true_pmf)
    good = np.isfinite(pmf) & (np.abs(centers) < 2.0)          # central, well-sampled
    err = np.nanmax(np.abs(pmf[good] - true_pmf[good]))
    # boosted dist barely populates the tails; CE2 should recover the well to <0.3 kcal
    ok &= err < 0.3
    print(f"(3) reweight max|PMF-true|={err:.3f} kcal/mol over |x|<2Å "
          f"(k0={pg['k0']:.3f}); curvature recovered")

    print("SELF-CHECK:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
