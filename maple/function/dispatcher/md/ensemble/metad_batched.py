"""Batched metadynamics + OPES for MAPLE -- N walkers ride ``BatchedNVT`` as the
batch dimension, ALL sharing ONE in-memory bias over the collective variable.

Paper-2 headline: GPU-batched enhanced sampling on an MLIP. The N walkers are the
batch B of ONE ``calc.get_ef_gpu()`` forward; a single shared bias engine
(:class:`bias.batched_metad.WTMetadEngine` or :class:`~bias.batched_metad.
OPESMetadEngine`) is attached at the EXISTING batch bias hook (``BatchedNVT.
_forces_au`` -> ``job._bias.apply(E, F, calc)``; ``nvt_batched.py`` is NOT
modified). Because the walkers co-reside in one batched forward + one address
space, multiple-walker metadynamics (Raiteri, Laio, Gervasio, Micheletti,
Parrinello, JPCB 2006) needs NO hill-file / MPI sync -- every walker deposits into
and feels the same bias for free.

Methods
-------
* ``method="wt"``  -- well-tempered metadynamics (Laio & Parrinello, PNAS 2002;
  Barducci, Bussi & Parrinello, PRL 2008). FES = ``-(gamma/(gamma-1)) V_bias``;
  c(t) reweighting available (Tiwary & Parrinello, JPCB 2015).
* ``method="opes"`` -- OPES metad-like (Invernizzi & Parrinello, JPCL 2020);
  weighted-KDE ``P(xi)`` bias with bounded running ``Z``; FES = ``-kT log P``.

``metad="off"`` (or ``mtd_deposit=false``) makes the bias a passive CV logger ->
the unbiased reference run. Isolated (non-periodic) replicas only (inherited from
``BatchedNVT``). CV = COM-COM distance between two atom groups (see
``bias.batched_metad``).
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from ase import Atoms

from maple.function.utility import Molecules
from .nvt_batched import BatchedNVT, BatchedNVTParams
from ..bias.batched_metad import (
    BatchedMetaD as BatchedMetaDBias, WTMetadEngine, OPESMetadEngine)
from ..bias.gamd import gamd_enabled, KB_HA_PER_K, HARTREE_PER_KCAL


@dataclass
class BatchedMetaDParams(BatchedNVTParams):
    """Batched-metaD parameters = batched-NVT params PLUS the metaD/OPES block.
    All fields are DECLARED (B-51: ``_init_params`` keeps the keys)."""
    metad:          str   = "on"       # on/off (off => plain MD + CV logger)
    method:         str   = "wt"       # 'wt' (well-tempered) | 'opes'
    nwalkers:       int   = 8          # number of walkers = batch B (if replicating)
    cv_group1:      str   = ""         # CV group 1 (all|heavy|"0,1,2"); required
    cv_group2:      str   = ""         # CV group 2; required, non-overlapping
    mtd_sigma:      float = 0.2        # Gaussian hill width [Angstrom]
    mtd_height:     float = 0.5        # base hill height w0 [kcal/mol]
    mtd_pace:       int   = 200        # force evaluations between hill deposits
    mtd_biasfactor: float = 10.0       # well-tempered gamma (WT + OPES); <=1 => std metaD
    mtd_barrier:    float = 5.0        # OPES barrier dE [kcal/mol]
    mtd_cv_min:     float = 0.0        # FES grid range min [Angstrom]
    mtd_cv_max:     float = 0.0        # FES grid range max [Angstrom]
    mtd_nbins:      int   = 300        # FES grid bins
    mtd_deposit:    bool  = True       # False => passive CV logger (reference)


class BatchedMetaD(BatchedNVT):
    """N metadynamics/OPES walkers ridden on BatchedNVT (one forward/step),
    sharing one bias -> FES along the CV."""

    _ALIASES = ("md", "MD", "nvt", "NVT", "batched", "batchnvt",
                "metad", "MetaD", "metadynamics", "opes", "OPES", "metad_batched")

    def __init__(self, output: str,
                 systems: Union[Molecules, List[Atoms], Atoms],
                 calc=None,
                 paras: Optional[dict] = None):
        # 1) parse metaD config first (need nwalkers before replicating template).
        mcfg = self._init_params(BatchedMetaDParams, paras, self._ALIASES)
        if not str(mcfg.cv_group1).strip() or not str(mcfg.cv_group2).strip():
            raise ValueError("BatchedMetaD requires cv_group1 and cv_group2 (the two "
                             "CV groups, e.g. '0,1,2' / 'heavy').")
        if str(mcfg.method).strip().lower() not in ("wt", "opes"):
            raise ValueError("BatchedMetaD method must be 'wt' or 'opes'.")
        n = int(mcfg.nwalkers)

        # 2) resolve the template + replicate to N walkers (B = N). A pre-built
        #    list of >= 2 systems is used as-is (B = len).
        template, calc = self._resolve_template(systems, calc)
        if isinstance(systems, (list, tuple)) and len(systems) >= 2:
            replicas = [a.copy() for a in systems]
        else:
            if n < 1:
                raise ValueError("BatchedMetaD needs nwalkers >= 1.")
            replicas = [template.copy() for _ in range(n)]

        # 3) build the batched NVT over the N walkers (calc prepared on B = N).
        super().__init__(output, replicas, calc=calc, paras=paras)
        self.params = self._init_params(BatchedMetaDParams, paras, self._ALIASES)
        p = self.params

        # 4) build the shared bias engine + attach on the batch hook.
        self._deposit_on = gamd_enabled(p.metad) and bool(p.mtd_deposit)
        self._method = str(p.method).strip().lower()
        kT = KB_HA_PER_K * float(p.temperature)
        cv_min, cv_max = float(p.mtd_cv_min), float(p.mtd_cv_max)
        if not (cv_max > cv_min):
            raise ValueError("BatchedMetaD requires mtd_cv_max > mtd_cv_min "
                             f"(FES/grid range; got min={cv_min}, max={cv_max}).")
        sigma = float(p.mtd_sigma)
        nbins = int(p.mtd_nbins)
        if self._method == "opes":
            self.engine = OPESMetadEngine(
                kT, sigma=sigma, biasfactor=float(p.mtd_biasfactor),
                barrier=float(p.mtd_barrier) * HARTREE_PER_KCAL,
                cv_min=cv_min, cv_max=cv_max, nbins=nbins)
        else:
            bf = float(p.mtd_biasfactor)
            self.engine = WTMetadEngine(
                kT, sigma=sigma, height=float(p.mtd_height) * HARTREE_PER_KCAL,
                biasfactor=(None if bf <= 1.0 else bf),
                cv_min=cv_min, cv_max=cv_max, nbins=nbins)
        self._bias = BatchedMetaDBias(
            self.atoms_list, p.cv_group1, p.cv_group2, self.engine,
            pace=int(p.mtd_pace), deposit=self._deposit_on)

        # FES results (filled by run()).
        self.fes_x = None
        self.fes = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _resolve_template(systems, calc):
        if isinstance(systems, Molecules):
            template = systems.multiatoms[0]
            calc = calc if calc is not None else systems.calc
        elif isinstance(systems, (list, tuple)):
            if not systems:
                raise ValueError("BatchedMetaD: empty systems list.")
            template = systems[0]
        else:
            template = systems              # single ase.Atoms template
        return template, calc

    # --- subclass hook: metaD is implemented here, so DON'T reject it ---------
    def _reject_ceiling_features(self):
        """metaD/OPES is implemented in this subclass; keep rejecting the OTHER
        batched ceilings (constraints/gamd/smd/plumed/colvars/posres)."""
        p = self.params
        bad = []
        if str(p.constraints or "none").strip().lower() not in ("", "none"):
            bad.append(f"constraints={p.constraints}")
        for nm in ("gamd", "smd", "plumed", "colvars", "posres"):
            val = getattr(p, nm, "")
            if str(val or "").strip().lower() not in ("", "off", "none", "no", "false", "0"):
                bad.append(f"{nm}={val}")
        if bad:
            raise NotImplementedError(
                "ponytail: the batched kernel does not implement "
                f"{', '.join(bad)} (deliberate C3 ceiling). Use the single-system NVT.")

    # ====================================================================== run
    def run(self):
        super().run()                       # BatchedNVT advances all walkers
        self._compute_fes()
        self._log_metad_summary()
        return self

    def production_cv(self, eq_frac=0.2):
        """Pooled CV frames across all walkers (drop leading ``eq_frac``)."""
        cv = []
        for b in range(self.B):
            s = np.asarray(self._bias.cv_history[b], dtype=np.float64)
            if s.size:
                s = s[int(eq_frac * s.size):]
            cv.append(s)
        return np.concatenate(cv) if cv else np.zeros(0)

    def _compute_fes(self):
        grid = np.linspace(self.params.mtd_cv_min, self.params.mtd_cv_max,
                           int(self.params.mtd_nbins))
        self.fes_x, self.fes = self.engine.fes(grid)

    # ----------------------------------------------------------------- logging
    def _log_metad_summary(self):
        p = self.params
        cv = self.production_cv()
        n_hills = int(self.engine.s.size) if self._method == "wt" \
            else int(self.engine.centers.size)
        lines = ["\n" + "=" * 72 + "\n",
                 f"{'BATCHED METADYNAMICS / OPES SUMMARY':^72}\n", "=" * 72 + "\n",
                 f"  walkers (B):   {self.B}\n",
                 f"  method:        {self._method}"
                 f"{'  (deposit off: CV logger)' if not self._deposit_on else ''}\n",
                 f"  CV groups:     g1='{p.cv_group1}'  g2='{p.cv_group2}'\n",
                 f"  hill sigma:    {p.mtd_sigma:.4f} A   pace {p.mtd_pace} evals\n"]
        if self._method == "opes":
            lines.append(f"  OPES:          gamma={p.mtd_biasfactor:.1f}  "
                         f"dE={p.mtd_barrier:.2f} kcal/mol  eps={self.engine.epsilon:.3e}\n")
            if self.engine.Z_history:
                zh = np.asarray(self.engine.Z_history)
                lines.append(f"  Z:             [{zh.min():.4e}, {zh.max():.4e}] "
                             f"(bounded, {zh.size} updates)\n")
        else:
            gtxt = ("std metaD" if self.engine.gamma is None
                    else f"gamma={self.engine.gamma:.1f}")
            lines.append(f"  well-tempered: {gtxt}  w0={p.mtd_height:.3f} kcal/mol\n")
        lines.append(f"  hills:         {n_hills}\n")
        if cv.size:
            lines.append(f"  CV sampled:    [{cv.min():.3f}, {cv.max():.3f}] A  "
                         f"mean={cv.mean():.3f}\n")
        if self.fes is not None:
            fin = self.fes[np.isfinite(self.fes)]
            rng = float(fin.max() - fin.min()) if fin.size else float("nan")
            lines.append(f"  FES range:     {rng / HARTREE_PER_KCAL:.4f} kcal/mol "
                         f"over {int(p.mtd_nbins)} bins\n")
        lines.append("=" * 72 + "\n")
        self.log_info(lines)
