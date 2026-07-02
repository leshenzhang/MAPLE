"""Batched extended-system ABF (eABF) for MAPLE -- N eABF walkers ride
``BatchedNVT`` as the batch dimension, each carrying its OWN extended variable
``lambda`` + ABF accumulator, then the EXISTING CZAR estimator
(``bias/eabf.py``) gives the PMF of the physical CV ``xi``.

This is the FIRST-CLASS ensemble entry for eABF -- the SAME sanctioned pattern as
``BatchedUmbrella`` / ``BatchedGaMD`` / ``BatchedSMD`` / ``BatchedMetaD``: a
``BatchedNVT`` subclass with a ``*Params(BatchedNVTParams)`` dataclass, replica
replication, ``self._bias`` set to the validated bias object, and ``run()`` =
``super().run()`` + the estimator post-processing (here CZAR). Before this wrapper
existed eABF had to be hand-wired (``BatchedNVT`` + ``bias.eabf.attach_eabf``);
now ``paras`` routes it exactly like every other enhanced-sampling ensemble.

Extended-system ABF (Lelievre/Rousset/Stoltz; Comer et al. JPCB 2015; CZAR: Lesage
et al. JPCB 2017) couples a fictitious variable ``lambda`` to the physical CV
``xi(x)`` by a stiff spring ``1/2 k (xi-lambda)^2``; ABF flattens ``A(lambda)`` so
``lambda`` (hence ``xi``) diffuses freely, and CZAR recovers the TRUE PMF of ``xi``
from the joint ``(xi, lambda)`` log. N such walkers are mutually independent --
exactly the batch dimension ``BatchedNVT`` advances with ONE ``calc.get_ef_gpu()``
per step. ``shared_grid=True`` pools all walkers into ONE ABF grid = multi-walker
eABF (free here: the walkers co-reside in one batched forward + address space).

Pipeline:
  1. replicate the template system to N walkers (B = N), or use a pre-built list;
  2. attach the per-replica :class:`bias.eabf.ExtendedABF` (spring + extended-var
     Langevin + ABF grid) on the batch bias hook (``BatchedNVT._forces_au``) via
     the EXISTING ``bias.eabf.attach_eabf`` (NOT reimplemented);
  3. run the batched NVT -- the spring force is folded into the padded
     ``(B, nmax_dof)`` buffer each step and each walker's ``(xi, lambda)`` logged;
  4. feed the pooled ``(xi, lambda)`` log to the EXISTING ``bias.eabf.czar`` /
     ``czar_pmf`` (NOT reimplemented) -> PMF of the physical CV ``xi``.

Units (real MLIP path): ``k`` [Ha/Angstrom^2]; ``xi``/``lambda``/``cv_min``/
``cv_max`` [Angstrom]; the CZAR PMF is in Ha (``k`` in Ha/Angstrom^2, ``beta`` in
1/Ha). The CV is a COM-COM distance between two atom groups (``cv_group1`` /
``cv_group2``), the only CV type the validated ``ExtendedABF`` implements.

PBC (Phase-1A): periodic walkers ARE supported, inherited from ``BatchedNVT`` (the
eABF spring rides the same shared get_ef_gpu that carries the per-replica cell; no
per-method PBC code) on a SUPPORTS_PBC batch backend.

ponytail: the ABF/CZAR/spring math is REUSED verbatim (``ExtendedABF`` + ``czar_pmf``
+ ``attach_eabf``); this wrapper is only the ensemble entry + param plumbing + the
CZAR call in ``run()``. The CV atom spec (``cv_group1`` / ``cv_group2``) mirrors the
GaMD sibling and is the honest realization of a distance CV for the two-group
COM-COM CV ``ExtendedABF`` supports (``cv_type`` reserved for future CV kinds).
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from ase import Atoms

from maple.function.utility import Molecules
from .nvt_batched import BatchedNVT, BatchedNVTParams
from ..bias.eabf import ExtendedABF, attach_eabf   # ExtendedABF + CZAR reused verbatim


@dataclass
class ExtendedABFParams(BatchedNVTParams):
    """Batched-eABF parameters = the batched-NVT params PLUS the eABF block. All
    fields are DECLARED (B-51: ``_init_params`` keeps the keys) and the bare names
    map 1:1 onto :class:`bias.eabf.ExtendedABF`'s constructor kwargs. Defaults track
    ``ExtendedABF``'s own defaults."""
    # --- collective variable (COM-COM distance between two atom groups) ---
    cv_type:      str = "distance"    # only 'distance' (COM-COM) is implemented
    cv_group1:    str = ""            # CV group 1 (all|heavy|"0,1,2"); required
    cv_group2:    str = ""            # CV group 2; required, non-overlapping
    n_walkers:    int = 4             # number of eABF walkers = batch B (if replicating)
    # --- spring (extended Lagrangian coupling) ---
    k:            float = 2.0         # spring force const [Ha/Angstrom^2]
    # --- extended variable lambda (fictitious dynamics) ---
    lam_mass:     Optional[float] = None   # explicit m_lam [Ha*fs^2/Angstrom^2]; else from tau
    lam_tau_fs:   float = 100.0       # lambda oscillation period -> m_lam = k (tau/2pi)^2
    lam_friction: float = 0.01        # lambda Langevin friction [1/fs]
    # --- ABF grid along lambda ---
    cv_min:       float = 0.0         # grid min [Angstrom]; required (cv_max > cv_min)
    cv_max:       float = 0.0         # grid max [Angstrom]; required
    n_bins:       int = 100           # ABF / CZAR histogram bins
    full_samples: int = 200           # ABF ramp fullSamples (Darve 2008; Comer 2015)
    shared_grid:  bool = False        # True => one ABF grid across walkers (multi-walker)
    apply_abf:    bool = True         # False => passive spring+logger (no flattening ref)
    lam0:         Optional[float] = None   # initial lambda; None => xi(x0) per walker
    eq_frac:      float = 0.1         # leading fraction of each (xi,lambda) log dropped for CZAR


class BatchedEABF(BatchedNVT):
    """N eABF walkers ridden on BatchedNVT (one forward/step) -> CZAR PMF."""

    _ALIASES = ("md", "MD", "nvt", "NVT", "batched", "batchnvt",
                "eabf", "eABF", "extended_abf", "eabf_batched")

    def __init__(self, output: str,
                 systems: Union[Molecules, List[Atoms], Atoms],
                 calc=None,
                 paras: Optional[dict] = None):
        # 1) parse eABF config first (need N before replicating the template).
        ecfg = self._init_params(ExtendedABFParams, paras, self._ALIASES)
        if str(ecfg.cv_type).strip().lower() != "distance":
            raise NotImplementedError(
                f"BatchedEABF cv_type='{ecfg.cv_type}' not implemented; the validated "
                "ExtendedABF CV is a COM-COM distance between two atom groups "
                "(cv_type='distance').")
        if not str(ecfg.cv_group1).strip() or not str(ecfg.cv_group2).strip():
            raise ValueError("BatchedEABF requires cv_group1 and cv_group2 (the two "
                             "CV groups, e.g. '0,1,2' / 'heavy').")
        if not (ecfg.cv_max > ecfg.cv_min):
            raise ValueError("BatchedEABF requires cv_max > cv_min "
                             f"(got min={ecfg.cv_min}, max={ecfg.cv_max}).")
        n = int(ecfg.n_walkers)

        # 2) resolve the template + replicate to N walkers (B = N). A pre-built list
        #    of >=2 systems is used as-is (B = len; e.g. distinct start conformers).
        template, calc = self._resolve_template(systems, calc)
        if isinstance(systems, (list, tuple)) and len(systems) >= 2:
            replicas = [a.copy() for a in systems]
        else:
            if n < 1:
                raise ValueError("BatchedEABF needs n_walkers >= 1.")
            replicas = [template.copy() for _ in range(n)]

        # 3) build the batched NVT over the N walkers (calc prepared on B = N).
        super().__init__(output, replicas, calc=calc, paras=paras)
        # re-attach the eABF fields (super() parsed only BatchedNVTParams).
        self.params = self._init_params(ExtendedABFParams, paras, self._ALIASES)
        p = self.params

        # 4) attach the per-walker ExtendedABF bias on the batch hook. attach_eabf
        #    reads kT (from p.temperature) + dt (from p.timestep), builds the bias,
        #    and sets self._bias -- REUSED verbatim (ponytail; no ABF/CZAR rewrite).
        self._bias = attach_eabf(
            self, p.cv_group1, p.cv_group2, k=p.k, cv_min=p.cv_min, cv_max=p.cv_max,
            nbins=int(p.n_bins), full_samples=int(p.full_samples),
            lam_tau_fs=p.lam_tau_fs, lam_mass=p.lam_mass, lam_friction=p.lam_friction,
            shared_grid=bool(p.shared_grid), apply_abf=bool(p.apply_abf),
            lam0=p.lam0, seed=p.random_seed)

        # CZAR PMF results (filled by run()).
        self.pmf_x = None
        self.pmf = None
        self.walker_mean_xi = None
        self.walker_mean_lambda = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _resolve_template(systems, calc):
        if isinstance(systems, Molecules):
            template = systems.multiatoms[0]
            calc = calc if calc is not None else systems.calc
        elif isinstance(systems, (list, tuple)):
            if not systems:
                raise ValueError("BatchedEABF: empty systems list.")
            template = systems[0]
        else:
            template = systems              # single ase.Atoms template
        return template, calc

    # ====================================================================== run
    def run(self):
        super().run()                        # BatchedNVT advances all N walkers
        self._compute_czar_pmf()
        return self

    def _compute_czar_pmf(self):
        p = self.params
        # pooled multi-walker CZAR PMF of the physical CV xi (EXISTING estimator).
        self.pmf_x, self.pmf = self._bias.czar(eq_frac=float(p.eq_frac))
        # per-walker <xi> / <lambda> tracking (post-equilibration).
        eq = float(p.eq_frac)
        mean_xi, mean_lam = [], []
        for b in range(self.B):
            xi = np.asarray(self._bias.xi_history[b], dtype=np.float64)
            lam = np.asarray(self._bias.lam_history[b], dtype=np.float64)
            if eq > 0.0 and xi.size:
                s = int(eq * xi.size)
                xi, lam = xi[s:], lam[s:]
            mean_xi.append(float(xi.mean()) if xi.size else np.nan)
            mean_lam.append(float(lam.mean()) if lam.size else np.nan)
        self.walker_mean_xi = np.array(mean_xi)
        self.walker_mean_lambda = np.array(mean_lam)
        self._log_eabf_summary()

    def _log_eabf_summary(self):
        pmf = self.pmf[np.isfinite(self.pmf)] if self.pmf is not None else np.array([])
        pmf_range = float(pmf.max() - pmf.min()) if pmf.size else float("nan")
        p = self.params
        lines = ["\n" + "=" * 72 + "\n",
                 f"{'BATCHED eABF -> CZAR PMF':^72}\n", "=" * 72 + "\n",
                 f"  walkers (B):   {self.B}\n",
                 f"  CV:            COM-COM distance  g1='{p.cv_group1}'  g2='{p.cv_group2}'\n",
                 f"  k:             {self._bias.k:.6f} Ha/Angstrom^2\n",
                 f"  m_lam:         {self._bias.m_lam:.4g} Ha*fs^2/Angstrom^2 "
                 f"(tau={p.lam_tau_fs:.1f} fs)\n",
                 f"  ABF grid:      [{self._bias.cv_min:.3f},{self._bias.cv_max:.3f}] "
                 f"Angstrom  {self._bias.nbins} bins  fullSamples={int(p.full_samples)}\n",
                 f"  shared_grid:   {bool(p.shared_grid)}  apply_abf={bool(p.apply_abf)}\n",
                 f"  T:             {p.temperature:.1f} K\n",
                 f"  PMF range:     {pmf_range:.6f} Ha\n",
                 "\n  wlk    <xi>(A)   <lambda>(A)   |<xi>-<lambda>|(A)\n"]
        for b in range(self.B):
            dx = abs(self.walker_mean_xi[b] - self.walker_mean_lambda[b])
            lines.append(f"  {b:>3}   {self.walker_mean_xi[b]:>8.3f}   "
                         f"{self.walker_mean_lambda[b]:>9.3f}   {dx:>12.4f}\n")
        lines.append("=" * 72 + "\n")
        self.log_info(lines)
