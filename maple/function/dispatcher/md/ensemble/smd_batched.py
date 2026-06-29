"""Batched constant-velocity steered MD (SMD) for MAPLE -- N independent pulls
ride ``BatchedNVT`` as the batch dimension, then Jarzynski's equality
(``bias/steered.py``) reconstructs Delta-G(lambda) from the N accumulated works.

Constant-velocity SMD is a MOVING harmonic restraint on a collective variable:
the centre ``lambda_b(t) = lam0_b + v_b t`` drags replica ``b``'s COM-COM distance
from ``lam0_b`` toward ``lam1_b`` while the external pulling work ``W_b`` is
accumulated each step (Park & Schulten, J. Chem. Phys. 120, 5946 (2004)). N such
pulls are mutually independent -- exactly the batch dimension ``BatchedNVT``
advances with ONE ``calc.get_ef_gpu()`` per step. So a batched SMD run = a single
``BatchedNVT`` over B = N replicas (same system, each its own pull + per-replica
thermostat RNG stream), via the batch-aware bias
:class:`bias.steered_batched.BatchedMovingRestraint` attached at the
force-assembly hook (``BatchedNVT._forces_au``).

Jarzynski (Phys. Rev. Lett. 78, 2690 (1997)):
``Delta-G(lambda) = -kT ln <exp(-W/kT)>`` over the N pull works (exp-average),
plus the 2nd-cumulant estimator ``<W> - Var(W)/2kT`` (exact for Gaussian work).
Both come from the EXISTING :func:`bias.steered.jarzynski_1d` (NOT reimplemented).
2nd law: ``<W> >= Delta-G`` (dissipated work >= 0).

Pipeline:
  1. replicate the template system to N pulls (B = N), or use a pre-built list;
  2. resolve per-replica ``lam0`` (``smd_lam0`` or ``"auto"`` = current CV) and
     the centre speed (``smd_velocity`` [Angstrom/fs], else derived from the
     endpoints ``lam0 -> smd_lam1`` over ``steps``);
  3. attach a per-replica moving restraint on a COM-COM distance CV;
  4. run the batched NVT -- the restraint force is folded into the padded
     ``(B, nmax_dof)`` buffer each step and each pull's (CV, lambda, work) logged;
  5. feed the N (lambda, work) series to ``steered.jarzynski_1d`` -> Delta-G(lambda).

Units: ``smd_k`` is in **Ha/Angstrom^2** (the single-system ``bias.steered`` /
``posres_fc`` convention); CV / lambda in Angstrom; work / Delta-G reported in
kcal/mol. Isolated (non-periodic) replicas only (inherited from BatchedNVT).
ponytail: ``jarzynski_1d`` + the CV gradient are REUSED, not rewritten; the only
new physics is the moving centre + per-pull work accumulator in the bias.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from ase import Atoms

from maple.function.utility import Molecules
from .nvt_batched import BatchedNVT, BatchedNVTParams
from ..bias.batched import BatchedHarmonicRestraint
from ..bias.steered_batched import BatchedMovingRestraint
from ..bias.steered import jarzynski_1d, HARTREE_TO_KCAL_MOL


@dataclass
class BatchedSMDParams(BatchedNVTParams):
    """Batched-SMD params = the batched-NVT params PLUS the SMD pull block. All
    fields are DECLARED (B-51: _init_params keeps the keys). ``smd_lam0`` /
    ``smd_lam1`` / ``smd_velocity`` accept a scalar (broadcast to all N pulls)
    OR a length-N sequence (per-pull lam0/v); ``smd_lam0`` also accepts
    ``""`` / ``"auto"`` => start each pull at that replica's current CV."""
    smd_group1:     str   = ""        # CV group 1 (all|heavy|"0,1,2"); required
    smd_group2:     str   = ""        # CV group 2; required, non-overlapping
    smd_k:          float = 0.0       # restraint force const [Ha/Angstrom^2]; required
    smd_lam0:       str   = "auto"    # start centre(s) [Angstrom]; "auto" => current CV
    smd_lam1:       float = 0.0       # end centre(s) [Angstrom] (endpoint mode)
    smd_velocity:   float = 0.0       # pull speed(s) [Angstrom/fs]; 0 => derive from endpoints
    smd_npulls:     int   = 8         # number of independent pulls = batch B
    smd_jarz_nbins: int   = 50        # Jarzynski Delta-G(lambda) grid bins


class BatchedSMD(BatchedNVT):
    """N constant-velocity SMD pulls ridden on BatchedNVT -> Jarzynski Delta-G."""

    _ALIASES = ("md", "MD", "nvt", "NVT", "batched", "batchnvt",
                "smd", "steered", "smd_batched")

    def __init__(self, output: str,
                 systems: Union[Molecules, List[Atoms], Atoms],
                 calc=None,
                 paras: Optional[dict] = None):
        # 1) parse SMD config first (need N before replicating the template).
        scfg = self._init_params(BatchedSMDParams, paras, self._ALIASES)
        n = int(scfg.smd_npulls)
        if n < 1:
            raise ValueError("BatchedSMD needs smd_npulls >= 1.")
        if not str(scfg.smd_group1).strip() or not str(scfg.smd_group2).strip():
            raise ValueError("BatchedSMD requires smd_group1 and smd_group2 "
                             "(the two CV groups, e.g. '0,1,2' / 'heavy').")
        if float(scfg.smd_k) <= 0.0:
            raise ValueError("BatchedSMD requires smd_k > 0 [Ha/Angstrom^2].")

        # 2) resolve template + replicate to N pulls (B = N). A pre-built list of
        #    exactly N systems is used as-is (e.g. distinct start conformers).
        template, calc = self._resolve_template(systems, calc)
        if isinstance(systems, (list, tuple)) and len(systems) == n:
            replicas = [a.copy() for a in systems]
        else:
            replicas = [template.copy() for _ in range(n)]

        # 3) build the batched NVT over the N replicas (calc prepared on B = N).
        super().__init__(output, replicas, calc=calc, paras=paras)
        # re-attach the SMD fields (super() parsed only BatchedNVTParams).
        self.params = self._init_params(BatchedSMDParams, paras, self._ALIASES)

        # 4) resolve per-replica lam0 + centre speed, then attach the moving bias.
        self._k_ha = float(scfg.smd_k)
        lam0 = self._resolve_lam0(scfg.smd_lam0, scfg.smd_group1, scfg.smd_group2, n)
        steps = max(int(self.params.steps), 1)
        dt_fs = float(self.params.timestep)
        vel = self._broadcast(scfg.smd_velocity, n)
        if np.any(vel != 0.0):
            dlam = vel * dt_fs                              # Angstrom per force-eval
            self._lam1 = lam0 + dlam * steps
            self._mode = "velocity"
        else:
            lam1 = self._broadcast(scfg.smd_lam1, n)
            dlam = (lam1 - lam0) / steps                    # endpoint interpolation
            self._lam1 = lam1
            self._mode = "endpoint"
        self._lam0 = lam0
        self._dlam = dlam
        self._bias = BatchedMovingRestraint(
            self.atoms_list, scfg.smd_group1, scfg.smd_group2,
            self._k_ha, lam0, dlam)
        # Jarzynski results (filled by run()).
        self.jarz_lambda = None
        self.jarz_dG_exp = None
        self.jarz_dG_cum = None
        self.pull_work_Ha = None
        self.mean_work_kcal = None
        self.dG_exp_kcal = None
        self.dG_cum_kcal = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _resolve_template(systems, calc):
        if isinstance(systems, Molecules):
            template = systems.multiatoms[0]
            calc = calc if calc is not None else systems.calc
        elif isinstance(systems, (list, tuple)):
            if not systems:
                raise ValueError("BatchedSMD: empty systems list.")
            template = systems[0]
        else:
            template = systems              # single ase.Atoms template
        return template, calc

    @staticmethod
    def _broadcast(val, B):
        """scalar / comma-string / sequence -> (B,) float array."""
        if isinstance(val, str):
            parts = [p for p in val.replace(",", " ").split() if p]
            arr = np.array([float(p) for p in parts], np.float64)
        elif np.isscalar(val):
            arr = np.array([float(val)], np.float64)
        else:
            arr = np.asarray(val, np.float64).reshape(-1)
        if arr.size == 1:
            arr = np.full(B, float(arr[0]), np.float64)
        if arr.size != B:
            raise ValueError(f"per-pull value length {arr.size} != B={B}")
        return arr

    def _resolve_lam0(self, raw, g1, g2, B):
        """lam0 = explicit value(s), or 'auto'/'' => each replica's current CV."""
        if (raw is None or (isinstance(raw, str)
                            and str(raw).strip().lower() in ("", "auto", "none"))):
            # current CV per replica from the initial geometries (k=0 probe reuses
            # the analytic COM-COM-distance machinery of the static restraint).
            probe = BatchedHarmonicRestraint(self.atoms_list, g1, g2, 0.0,
                                             np.zeros(B))
            ptr = np.concatenate([[0], np.cumsum(
                [len(at) for at in self.atoms_list])]).astype(int)
            coord = np.concatenate([at.get_positions()
                                    for at in self.atoms_list], axis=0)
            cvs, _ = probe.restraint_forces(coord, ptr)
            return np.asarray(cvs, np.float64)
        return self._broadcast(raw, B)

    # ====================================================================== run
    def run(self):
        super().run()                        # BatchedNVT advances all N pulls
        self._reconstruct_jarzynski()
        return self

    def _reconstruct_jarzynski(self):
        p = self.params
        lams_list = [np.asarray(self._bias.lam_history[b], np.float64)
                     for b in range(self.B)]
        works_list = [np.asarray(self._bias.work_history[b], np.float64)
                      for b in range(self.B)]                    # Hartree
        self.pull_work_Ha = np.array([w[-1] if w.size else np.nan
                                      for w in works_list])
        self.mean_work_kcal = float(np.nanmean(self.pull_work_Ha)
                                    * HARTREE_TO_KCAL_MOL)
        # EXISTING Jarzynski reconstruction (Ha works -> kcal/mol Delta-G).
        self.jarz_lambda, self.jarz_dG_exp, self.jarz_dG_cum = jarzynski_1d(
            lams_list, works_list, p.temperature, nbins=int(p.smd_jarz_nbins))
        self.dG_exp_kcal = float(self.jarz_dG_exp[-1])
        self.dG_cum_kcal = float(self.jarz_dG_cum[-1])
        self._log_smd_summary()

    def _log_smd_summary(self):
        lam0_mean = float(np.mean(self._lam0))
        lam1_mean = float(np.mean(self._lam1))
        second_law = (self.mean_work_kcal >= self.dG_exp_kcal - 1e-9)
        lines = ["\n" + "=" * 72 + "\n",
                 f"{'BATCHED STEERED MD -> JARZYNSKI Delta-G':^72}\n", "=" * 72 + "\n",
                 f"  pulls (B):     {self.B}\n",
                 f"  mode:          {self._mode}\n",
                 f"  k:             {self._k_ha:.6f} Ha/Angstrom^2\n",
                 f"  lambda:        {lam0_mean:.3f} -> {lam1_mean:.3f} Angstrom\n",
                 f"  T:             {self.params.temperature:.1f} K\n",
                 f"  <W>:           {self.mean_work_kcal:.4f} kcal/mol\n",
                 f"  dG_exp(end):   {self.dG_exp_kcal:.4f} kcal/mol (Jarzynski exp-avg)\n",
                 f"  dG_cum(end):   {self.dG_cum_kcal:.4f} kcal/mol (2nd cumulant)\n",
                 f"  2nd law <W> >= dG_exp: {'OK' if second_law else 'VIOLATED'}\n",
                 "\n  pull   W(kcal/mol)\n"]
        for b in range(self.B):
            lines.append(f"  {b:>4}   "
                         f"{self.pull_work_Ha[b] * HARTREE_TO_KCAL_MOL:>10.4f}\n")
        lines.append("=" * 72 + "\n")
        self.log_info(lines)
