"""Steered MD (SMD) + Jarzynski free energy for MAPLE MD.

Constant-velocity steered MD: a harmonic restraint on a *distance* collective
variable -- the distance between the centres of mass of two atom groups --
whose centre ``lambda(t)`` moves linearly from ``lam0`` to ``lam1`` across the
run. The external pulling work is accumulated every step, and Jarzynski's
equality reconstructs the PMF / Delta-G along the pull from an ensemble of
independent pulls.

Same single-injection-point architecture as :mod:`posres_calc` / the bias
layer: a calculator wrapper recomputes the steering force once per MD step at
``atoms.get_forces`` and folds it into the MLIP force, so NVE/NVT/NPT all gain
SMD with no integrator change, and it composes with any active PLUMED/Colvars
bias or position restraint.

CV -- distance between the COMs of group1 and group2::

    R1 = sum_{i in g1} m_i x_i / M1 ,  R2 likewise        (Angstrom)
    d  = | minimum_image(R1 - R2) |                        (Angstrom)

Moving harmonic restraint (lambda moves lam0 -> lam1 over ``steps``)::

    lam(n) = lam0 + (lam1 - lam0) * n / steps              (Angstrom)
    V      = 1/2 k (d - lam)^2                             (Hartree)
    F_i    = -k (d - lam) d(d)/d(x_i)                      (Ha/Angstrom)
    d(d)/d(x_i) =  (m_i/M1) u   (i in g1),  -(m_j/M2) u  (j in g2),  u=(R1-R2)/d

External (Jarzynski) work, accumulated per step (Park & Schulten 2004)::

    dW = (dV/dlam) dlam = -k (d - lam) dlam = k (lam - d) dlam   (Hartree)

The force constant ``smd_k`` is in **Ha/Angstrom^2** (MAPLE units, identical
convention to ``posres_fc``); convert a GROMACS k via the factor documented in
:mod:`posres_calc`. ``lam0=None`` starts the pull at the current CV distance.

Refs: Park & Schulten, J. Chem. Phys. 120, 5946 (2004) (SMD + Jarzynski,
doi:10.1063/1.1651473); Jarzynski, Phys. Rev. Lett. 78, 2690 (1997)
(doi:10.1103/PhysRevLett.78.2690); Hummer & Szabo, PNAS 98, 3658 (2001)
(doi:10.1073/pnas.071034098).
"""

import os
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .posres_calc import _build_selection   # reuse group selection (all/heavy/idx)

HARTREE_TO_KCAL_MOL = 627.5094740631          # 1 Ha -> kcal/mol
KELVIN_TO_HARTREE = 3.1668114e-6              # k_B in Hartree/K (matches md/utils.py)


def smd_enabled(spec):
    """True if an ``smd`` setting requests steered MD (falsey/'off' => off)."""
    if spec is None:
        return False
    if isinstance(spec, bool):
        return spec
    if isinstance(spec, str):
        return spec.strip().lower() not in ("", "off", "no", "false", "none", "0")
    return bool(spec)


def _mic_vec(vec, cell, pbc):
    """Minimum-image of a single cartesian displacement (3,) for periodic dirs."""
    pbc = np.asarray(pbc, bool)
    if not np.any(pbc):
        return vec
    cell = np.asarray(cell, np.float64)
    if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-12:
        return vec
    frac = vec @ np.linalg.inv(cell)
    shift = np.round(frac)
    shift[~pbc] = 0.0
    return vec - shift @ cell


class SteeredMDCalculator(Calculator):
    """Wrap a MAPLE MLIP (or bias) calculator with constant-velocity SMD."""

    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

    def __init__(self, inner, group1, group2, k, lam0, lam1, total_steps,
                 *, atoms=None, output='run', restart_step=0, log_every=10):
        Calculator.__init__(self)
        if atoms is None:
            raise ValueError("SteeredMDCalculator needs the initial atoms to "
                             "resolve the pull groups and the start CV.")
        self.inner = inner
        self._istep = int(restart_step)
        self._total_steps = max(int(total_steps or 0), 1)
        self._output = output
        self._log_every = max(int(log_every or 1), 1)
        self.SUPPORTS_PBC = getattr(inner, 'SUPPORTS_PBC', False)

        self.atoms = atoms.copy()
        self._g1 = np.asarray(_build_selection(group1, atoms), dtype=int)
        self._g2 = np.asarray(_build_selection(group2, atoms), dtype=int)
        if len(self._g1) == 0 or len(self._g2) == 0:
            raise ValueError("SMD pull groups must be non-empty.")
        if set(self._g1.tolist()) & set(self._g2.tolist()):
            raise ValueError("SMD pull groups must not overlap.")
        self._m = np.asarray(atoms.get_masses(), np.float64)
        self._k = float(k)
        self._lam1 = float(lam1)
        # lam0=None => start at the current CV distance (pull from where we are)
        self._lam0 = float(lam0) if lam0 is not None else self._cv(
            np.asarray(atoms.get_positions(), np.float64),
            np.asarray(atoms.get_cell()[:], np.float64), atoms.pbc)[0]

        self._work = 0.0          # accumulated external work (Ha)
        self._lam_prev = self._lam0
        self._logpath = f"{output}_smd.dat"
        self._log_init = False

    def _com(self, pos, sel):
        m = self._m[sel]
        return (m[:, None] * pos[sel]).sum(0) / m.sum()

    def _cv(self, pos, cell, pbc):
        """Return (d, unit_vector u=(R1-R2)/d, R1, R2)."""
        R1 = self._com(pos, self._g1)
        R2 = self._com(pos, self._g2)
        dvec = _mic_vec(R1 - R2, cell, pbc)
        d = float(np.linalg.norm(dvec))
        u = dvec / d if d > 1e-9 else np.zeros(3)
        return d, u, R1, R2

    def _lambda_now(self):
        f = self._istep / self._total_steps
        f = min(1.0, max(0.0, f))
        return self._lam0 + (self._lam1 - self._lam0) * f

    def calculate(self, atoms=None, properties=('energy', 'forces'),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        atoms = self.atoms

        self.inner.calculate(atoms, list(properties), system_changes)
        energy = float(self.inner.results['energy'])                 # Ha
        forces = np.array(self.inner.results['forces'], np.float64)  # Ha/A (copy)

        pos = np.asarray(atoms.get_positions(), np.float64)
        cell = np.asarray(atoms.get_cell()[:], np.float64)
        d, u, _, _ = self._cv(pos, cell, atoms.pbc)
        lam = self._lambda_now()

        # restraint force: F_i = -k (d - lam) d(d)/d(x_i)
        diff = d - lam
        e_res = 0.5 * self._k * diff * diff
        if self._k != 0.0 and d > 1e-9:
            g1, g2 = self._g1, self._g2
            w1 = (self._m[g1] / self._m[g1].sum())[:, None]   # (n1,1)
            w2 = (self._m[g2] / self._m[g2].sum())[:, None]
            coef = -self._k * diff                            # scalar (Ha/A)
            forces[g1] += coef * w1 * u[None, :]              # +coef*(m/M)u
            forces[g2] += coef * w2 * (-u[None, :])

        # external work increment for moving the centre this step (Park 2004):
        #   dW = k (lam - d) dlam ,  dlam = lam - lam_prev
        self._work += self._k * (lam - d) * (lam - self._lam_prev)
        self._lam_prev = lam

        if (self._istep % self._log_every) == 0:
            self._write_log(self._istep, lam, d,
                            (-self._k * diff), self._work)
        self._istep += 1

        self.results = {
            'energy': energy + e_res,
            'free_energy': energy + e_res,
            'forces': forces,
            'smd_cv': d,
            'smd_lambda': lam,
            'smd_work': self._work,
            'smd_restraint_force': -self._k * diff,
        }
        if 'stress' in self.inner.results:
            self.results['stress'] = np.asarray(self.inner.results['stress'])

    def _write_log(self, step, lam, cv, force, work):
        if not self._log_init:
            mode = 'a' if (step > 0 and os.path.exists(self._logpath)) else 'w'
            self._fh = open(self._logpath, mode)
            if mode == 'w':
                self._fh.write("# Step Lambda(A) CV(A) RestraintForce(Ha/A) "
                               "Work(Ha) Work(kcal/mol)\n")
            self._log_init = True
        self._fh.write(f"{step:8d} {lam:14.6f} {cv:14.6f} {force:16.8e} "
                       f"{work:16.8e} {work*HARTREE_TO_KCAL_MOL:16.8e}\n")
        self._fh.flush()


# --------------------------------------------------------------------------- #
# Post-processing: Jarzynski PMF / Delta-G from an ensemble of pulls.
# --------------------------------------------------------------------------- #
def read_smd_work(path):
    """Read a ``*_smd.dat`` file -> (lambda[A], work[Ha]) arrays."""
    lam, w = [], []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("#") or not ln.strip():
                continue
            p = ln.split()
            lam.append(float(p[1]))
            w.append(float(p[4]))
    return np.asarray(lam, float), np.asarray(w, float)


def jarzynski_1d(lams_list, works_list, temperature, *, nbins=50):
    """Jarzynski Delta-G(lambda) from N constant-velocity pulls.

    Each pull contributes a work-vs-lambda series (same pull protocol, different
    random seed / start). Returns ``(lambda_grid, dG_exp, dG_cumulant)`` with
    free energies in **kcal/mol**, the start of the pull shifted to 0.

    * ``dG_exp``      : exponential average  -kT ln < exp(-W/kT) >  (Jarzynski).
    * ``dG_cumulant`` : 2nd-order cumulant   <W> - (1/2kT) Var(W)   (exact for
      Gaussian work, lower-variance but biased if work is non-Gaussian).

    Both estimators are interpolated onto a common lambda grid so pulls with
    slightly different sampled lambdas combine. By the 2nd law
    ``dG_exp <= <W>`` always (the inequality is the standard SMD sanity check).
    """
    kbt = float(temperature) * KELVIN_TO_HARTREE          # Ha
    lams_list = [np.asarray(l, float) for l in lams_list]
    works_list = [np.asarray(w, float) for w in works_list]
    lo = max(l.min() for l in lams_list)
    hi = min(l.max() for l in lams_list)
    grid = np.linspace(lo, hi, nbins)
    # interpolate each pull's work onto the common grid
    W = np.array([np.interp(grid, l, w) for l, w in zip(lams_list, works_list)])
    # exponential average, done in a shift-stabilised way
    shift = W.min(axis=0)
    expavg = np.log(np.mean(np.exp(-(W - shift) / kbt), axis=0)) - shift / kbt
    dG_exp = -kbt * expavg                                # Ha
    dG_cum = W.mean(axis=0) - 0.5 * W.var(axis=0) / kbt   # Ha
    dG_exp = (dG_exp - dG_exp[0]) * HARTREE_TO_KCAL_MOL
    dG_cum = (dG_cum - dG_cum[0]) * HARTREE_TO_KCAL_MOL
    return grid, dG_exp, dG_cum


# --------------------------------------------------------------------------- #
# Standalone numpy/ASE self-check (no maple import; no GPU). Run:
#   PYTHONSAFEPATH=1 python -m maple.function.dispatcher.md.bias.steered
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    from ase import Atoms

    class _ZeroCalc(Calculator):
        implemented_properties = ['energy', 'forces']

        def calculate(self, atoms=None, properties=('energy', 'forces'),
                      system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            n = len(self.atoms)
            self.results = {'energy': 0.0, 'forces': np.zeros((n, 3))}

    ok = True

    # (1) force exactness: two equal-mass atoms on x at d=2.0, lam0=2.0 (no force
    #     at start), single-atom groups. Move so d=2.5 with lam still ~2.0 =>
    #     F = -k(d-lam) u on atom0, opposite on atom1.
    a = Atoms('H2', positions=[[0., 0, 0], [2.0, 0, 0]])
    a.calc = _ZeroCalc()
    w = SteeredMDCalculator(a.calc, [0], [1], k=1.0, lam0=2.0, lam1=2.0,
                            total_steps=100, atoms=a, output='/tmp/_smd_t1')
    a.calc = w
    a.positions[1] = [2.5, 0.0, 0.0]            # d = 2.5, lam = 2.0 (istep 0)
    f = a.get_forces()
    # u points from R1(atom0@0) to R2(atom1@2.5) => R1-R2 = -x => u=-x.
    # diff=d-lam=0.5; coef=-k*diff=-0.5; F0 = coef*(m/M)*u = -0.5*1*(-x)=+0.5 x
    ok &= abs(f[0, 0] - 0.5) < 1e-9 and abs(f[1, 0] + 0.5) < 1e-9
    e = a.get_potential_energy()
    ok &= abs(e - 0.125) < 1e-9                 # 0.5*1*0.5^2
    print(f"(1) F0x={f[0,0]:+.6f}(exp +0.5) F1x={f[1,0]:+.6f}(exp -0.5) "
          f"E={e:.6f}(exp 0.125)")

    # (2) COM weighting: group1={0,1} (a methylene-like pair), group2={2}.
    a2 = Atoms('H3', positions=[[0., 0, 0], [0, 0, 0], [3.0, 0, 0]])
    a2.calc = _ZeroCalc()
    w2 = SteeredMDCalculator(a2.calc, [0, 1], [2], k=2.0, lam0=None, lam1=5.0,
                             total_steps=100, atoms=a2, output='/tmp/_smd_t2')
    # lam0 auto = current CV = COM(g1)=0, COM(g2)=3 => d=3.0
    ok &= abs(w2._lam0 - 3.0) < 1e-9
    a2.calc = w2
    f2 = a2.get_forces()                        # istep0: lam=3.0=d => zero force
    ok &= np.allclose(f2, 0.0, atol=1e-9)
    print(f"(2) lam0(auto)={w2._lam0:.4f}(exp 3.0) max|F|@lam==d={np.abs(f2).max():.1e}")

    # (3) work sign + Jarzynski 2nd-law bound on synthetic Gaussian work.
    #     For W ~ N(mu, s^2): dG_cumulant = mu - s^2/(2kT) exactly; dG_exp <= mu.
    T = 300.0
    kbt = T * KELVIN_TO_HARTREE
    rng = np.random.RandomState(0)
    grid = np.linspace(0.0, 1.0, 20)
    mu = 0.02 * grid                            # Ha, linear "true work" ramp
    s = 0.5 * kbt                               # work spread ~ 0.5 kT
    lams_list, works_list = [], []
    for _ in range(400):
        noise = s * rng.randn(grid.size)
        lams_list.append(grid.copy())
        works_list.append(mu + noise)
    gr, dG_exp, dG_cum = jarzynski_1d(lams_list, works_list, T, nbins=20)
    # the per-bin work variance is constant here, so the -Var/2kT term cancels in
    # the start-shift => the TRUE Jarzynski/cumulant Delta-G(end) both equal the
    # work-ramp (mu_end - mu_0). The exp-average estimator is biased slightly HIGH
    # at finite N (ln-concavity), so check closeness to the analytic target, not
    # the (true-value-only) 2nd-law inequality dG_exp <= <W>.
    true_dG_end = (mu[-1] - mu[0]) * HARTREE_TO_KCAL_MOL          # 12.55 kcal/mol
    ok &= abs(dG_exp[-1] - true_dG_end) < 0.3                     # Jarzynski exp-avg
    ok &= abs(dG_cum[-1] - true_dG_end) < 0.1                     # cumulant (tighter)
    print(f"(3) dG_exp_end={dG_exp[-1]:.4f} dG_cum_end={dG_cum[-1]:.4f} "
          f"(analytic true {true_dG_end:.4f})")

    # (4) disabled / overlap / empty-group guards.
    ok &= (smd_enabled('') is False and smd_enabled('off') is False
           and smd_enabled('distance') is True and smd_enabled(True) is True)
    try:
        SteeredMDCalculator(_ZeroCalc(), [0], [0], 1.0, 0.0, 1.0, 10,
                            atoms=Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]]))
        ok = False  # should have raised on overlap
    except ValueError:
        pass
    print(f"(4) enable/guard checks ok={ok}")

    print("STEERED SELF-CHECK:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
