"""Batched umbrella sampling for MAPLE -- N windows ride ``BatchedNVT`` as the
batch dimension, then the EXISTING WHAM (``bias/umbrella.py``) gives the PMF.

Classical umbrella sampling runs N independent windows, each a harmonic restraint
``1/2 k (xi - xi0_i)^2`` on a collective variable ``xi`` at a different centre
``xi0_i`` spanning a range. Those N windows are mutually independent -- exactly
the batch dimension ``BatchedNVT`` advances with ONE ``calc.get_ef_gpu()`` per
step. So a batched umbrella run = a single ``BatchedNVT`` over B = N replicas,
all the same system, each carrying a per-replica restraint centre, via the
batch-aware bias :class:`bias.batched.BatchedHarmonicRestraint` attached at the
force-assembly point (``BatchedNVT._forces_au``).

Pipeline:
  1. replicate the template system to N windows (B = N);
  2. build centres ``make_windows(cv_min, cv_max, N)`` (reuses ``bias.umbrella``);
  3. attach a per-replica harmonic restraint (centre i for replica i) on a
     COM-COM distance CV;
  4. run the batched NVT -- the restraint force is folded into the padded
     ``(B, nmax_dof)`` buffer each step and each window's CV is logged;
  5. feed the per-window CV time series to the EXISTING ``bias.umbrella.wham_1d``
     (NOT reimplemented here) -> PMF in kcal/mol.

Units: restraint force constant ``us_kappa`` is in **kcal/mol/Angstrom^2** (the
chemist convention used by ``bias.umbrella`` / its PLUMED window writer); it is
converted to Ha/Angstrom^2 for the in-buffer restraint. The CV and PMF are in
Angstrom / kcal/mol.

PBC (Phase-1A): periodic windows ARE supported, inherited from ``BatchedNVT``
(the umbrella bias rides the same shared get_ef_gpu that carries the cell);
PLUMED/Colvars-driven windows stay on the single-system path (this is the native
batched umbrella, not a PLUMED batcher). WHAM is reused, not rewritten.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from ase import Atoms

from maple.function.utility import Molecules
from .nvt_batched import BatchedNVT, BatchedNVTParams
from ..bias.batched import BatchedHarmonicRestraint
from ..bias.umbrella import make_windows, wham_1d, histogram_overlap

HARTREE_TO_KCAL_MOL = 627.5094740631     # 1 Ha -> kcal/mol (matches bias/steered.py)


@dataclass
class BatchedUmbrellaParams(BatchedNVTParams):
    """Batched-umbrella parameters = the batched-NVT params PLUS the umbrella
    block. All fields are DECLARED (B-51: _init_params keeps the keys)."""
    us_group1:       str   = ""        # CV group 1 (all|heavy|"0,1,2"); required
    us_group2:       str   = ""        # CV group 2; required, non-overlapping
    us_kappa:        float = 150.0     # restraint force const [kcal/mol/Angstrom^2]
    us_cv_min:       float = 0.0       # first window centre [Angstrom]
    us_cv_max:       float = 0.0       # last window centre  [Angstrom]
    us_nwindows:     int   = 8         # number of windows = batch B
    us_discard_frac: float = 0.2       # leading fraction of each CV series dropped
    us_nbins:        int   = 80        # WHAM histogram bins


class BatchedUmbrella(BatchedNVT):
    """N umbrella windows ridden on BatchedNVT (one forward/step) -> WHAM PMF."""

    _ALIASES = ("md", "MD", "nvt", "NVT", "batched", "batchnvt",
                "umbrella", "us", "umbrella_batched")

    def __init__(self, output: str,
                 systems: Union[Molecules, List[Atoms], Atoms],
                 calc=None,
                 paras: Optional[dict] = None):
        # 1) parse umbrella config first (need N before replicating the template).
        ucfg = self._init_params(BatchedUmbrellaParams, paras, self._ALIASES)
        n = int(ucfg.us_nwindows)
        if n < 2:
            raise ValueError("BatchedUmbrella needs us_nwindows >= 2.")
        if not str(ucfg.us_group1).strip() or not str(ucfg.us_group2).strip():
            raise ValueError("BatchedUmbrella requires us_group1 and us_group2 "
                             "(the two CV groups, e.g. '0,1,2' / 'heavy').")
        if not (ucfg.us_cv_max > ucfg.us_cv_min):
            raise ValueError("BatchedUmbrella requires us_cv_max > us_cv_min "
                             f"(got min={ucfg.us_cv_min}, max={ucfg.us_cv_max}).")

        # 2) resolve the template + replicate to N windows (B = N). A pre-built
        #    list of exactly N systems is used as-is (e.g. seeded from an SMD pull).
        template, calc = self._resolve_template(systems, calc)
        if isinstance(systems, (list, tuple)) and len(systems) == n:
            replicas = [a.copy() for a in systems]
        else:
            replicas = [template.copy() for _ in range(n)]

        # 3) build the batched NVT over the N replicas (calc prepared on B = N).
        super().__init__(output, replicas, calc=calc, paras=paras)
        # re-attach the umbrella fields (super() parsed only BatchedNVTParams).
        self.params = self._init_params(BatchedUmbrellaParams, paras, self._ALIASES)

        # 4) per-window centres + the per-replica harmonic restraint bias.
        self._centers = make_windows(ucfg.us_cv_min, ucfg.us_cv_max, n)   # (N,) Angstrom
        self._kappa_kcal = float(ucfg.us_kappa)
        k_ha = self._kappa_kcal / HARTREE_TO_KCAL_MOL                      # Ha/Angstrom^2
        self._bias = BatchedHarmonicRestraint(
            self.atoms_list, ucfg.us_group1, ucfg.us_group2, k_ha, self._centers)
        # PMF results (filled by run()).
        self.pmf_x = None
        self.pmf = None
        self.window_mean_cv = None
        self.window_std_cv = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _resolve_template(systems, calc):
        if isinstance(systems, Molecules):
            template = systems.multiatoms[0]
            calc = calc if calc is not None else systems.calc
        elif isinstance(systems, (list, tuple)):
            if not systems:
                raise ValueError("BatchedUmbrella: empty systems list.")
            template = systems[0]
        else:
            template = systems              # single ase.Atoms template
        return template, calc

    # ====================================================================== run
    def run(self):
        super().run()                        # BatchedNVT advances all N windows
        self._compute_pmf()
        return self

    def _compute_pmf(self):
        p = self.params
        disc = float(p.us_discard_frac)
        samples = []
        for b in range(self.B):
            s = np.asarray(self._bias.cv_history[b], dtype=np.float64)
            if disc > 0.0 and s.size:
                s = s[int(disc * s.size):]
            samples.append(s)
        self.window_mean_cv = np.array([float(s.mean()) if s.size else np.nan
                                        for s in samples])
        self.window_std_cv = np.array([float(s.std()) if s.size else np.nan
                                       for s in samples])
        # reuse the EXISTING WHAM (kcal/mol/Angstrom^2 kappa, Angstrom CV).
        self.pmf_x, self.pmf = wham_1d(
            self._centers, self._kappa_kcal, samples, p.temperature,
            nbins=int(p.us_nbins))
        try:
            overlaps, _ = histogram_overlap(samples, nbins=int(p.us_nbins))
        except Exception:
            overlaps = np.array([])
        self._log_umbrella_summary(overlaps)

    def _log_umbrella_summary(self, overlaps):
        pmf = self.pmf[np.isfinite(self.pmf)]
        pmf_range = (float(pmf.max() - pmf.min()) if pmf.size else float("nan"))
        lines = ["\n" + "=" * 72 + "\n",
                 f"{'BATCHED UMBRELLA -> WHAM PMF':^72}\n", "=" * 72 + "\n",
                 f"  windows (B):   {self.B}\n",
                 f"  kappa:         {self._kappa_kcal:.3f} kcal/mol/Angstrom^2\n",
                 f"  centres:       {self._centers[0]:.3f} .. {self._centers[-1]:.3f} "
                 f"Angstrom\n",
                 f"  T:             {self.params.temperature:.1f} K\n",
                 f"  PMF range:     {pmf_range:.4f} kcal/mol\n",
                 (f"  mean overlap:  {float(np.mean(overlaps)):.3f}\n"
                  if overlaps.size else ""),
                 "\n  win   centre(A)   <CV>(A)    sig(CV)(A)\n"]
        for b in range(self.B):
            lines.append(f"  {b:>3}   {self._centers[b]:>8.3f}   "
                         f"{self.window_mean_cv[b]:>8.3f}   "
                         f"{self.window_std_cv[b]:>9.4f}\n")
        lines.append("=" * 72 + "\n")
        self.log_info(lines)
