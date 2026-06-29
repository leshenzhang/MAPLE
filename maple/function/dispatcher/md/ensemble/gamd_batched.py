"""Batched GaMD + ParGaMD Weighted-Ensemble for MAPLE -- N walkers ride
``BatchedNVT`` as the batch dimension, each carrying a pure-MLIP GaMD boost, with
optional periodic Weighted-Ensemble (WE) resampling along a 1-D progress
coordinate (ParGaMD, Sonti..Ahn, JCTC 2026, 10.1021/acs.jctc.6c00557).

Two layers, both riding the EXISTING batch-aware bias hook (``BatchedNVT.
_forces_au`` calls ``job._bias.apply(E, F, calc)`` once per forward; ``nvt_batched.
py`` is NOT modified):

GaMD layer
----------
The N walkers = the batch B. A single :class:`bias.batched.BatchedGaMD` boosts
every walker: on a pure MLIP the GaMD boost ``DeltaV = 1/2 k (E - V)^2`` (V<E)
reduces to scaling the physical force by ``(1 - k(E-V)) in [0,1]`` (see
:mod:`bias.gamd`). The boost params ``(k, E)`` are estimated ONCE from a shared
prep window -- energies POOLED across all B walkers -- then frozen for every
walker (paper Sec 2.4 step 1: a single finalized ``(E, Vmax, Vmin, k)`` is held
fixed for all walkers). The unbiased PMF along the CV is recovered with the
EXISTING :func:`bias.gamd.gamd_reweight_1d` (CE2 / Maclaurin) -- NOT reimplemented.
``gamd="off"`` makes the bias a passive CV logger (no boost) -> the unbiased
reference run.

ParGaMD WE layer
----------------
Every ``we_resample_every`` steps (production only -- after the GaMD prep) the B
walkers are binned along the COM-COM distance CV and RESAMPLED, conserving total
statistical weight: high-weight walkers in under-populated bins are SPLIT (state
copied into new slots, weight halved) and low-weight walkers in over-populated
bins are MERGED (survivor drawn by weight, weights summed), exactly per the paper
(Sec 2.3; each walker ends with weight ``P_i/n_w`` or ``2 P_i/n_w``). Walker state
(coords + velocities) is copied GPU-side.

ponytail (deliberate, fixed-B WE): the batched engine carries a FIXED population B
(fixed padded buffers / thermostat list / prepared calc), so rather than the
variable-population WESTPA target ``n_w`` per bin, the B walkers are flattened
ACROSS the occupied bins (each occupied bin -> ``floor(B/n_occ)`` or
``+1`` walkers, sum = B). The split/merge mechanics and weight bookkeeping are the
paper's; only the per-bin target is set to keep B fixed. This is the honest
adaptation to a fixed-batch MD core and is exactly what the WE gate checks
(total weight conserved + occupancy flattened). Isolated (non-periodic) replicas
only (inherited from ``BatchedNVT``).
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from ase import Atoms

from maple.function.utility import Molecules
from .nvt_batched import BatchedNVT, BatchedNVTParams
from ..bias.batched import BatchedGaMD as BatchedGaMDBias
from ..bias.gamd import gamd_enabled, gamd_reweight_1d


@dataclass
class BatchedGaMDParams(BatchedNVTParams):
    """Batched-GaMD parameters = the batched-NVT params PLUS the GaMD + WE block.
    All fields are DECLARED (B-51: ``_init_params`` keeps the keys)."""
    # --- GaMD (the boost is implemented here; overrides the BatchedNVT ceiling) ---
    gamd:            str   = "on"      # on/off (off => plain MD + CV logger, no boost)
    nwalkers:        int   = 8         # number of walkers = batch B (if replicating)
    cv_group1:       str   = ""        # CV group 1 (all|heavy|"0,1,2"); required
    cv_group2:       str   = ""        # CV group 2; required, non-overlapping
    gamd_mode:       str   = "lower"   # 'lower' | 'upper' (Miao 2015 eqs 7-11)
    gamd_sigma0:     float = 6.0       # sigma0 anharmonicity cap [kcal/mol]
    gamd_prep_steps: int   = 2000      # conventional-MD prep window (shared, pooled)
    gamd_nbins:      int   = 50        # reweight histogram bins
    gamd_reweight:   str   = "ce2"     # 'ce2' | 'maclaurin'
    # --- ParGaMD Weighted Ensemble (WE) ---
    we:                str = "off"     # on/off
    we_resample_every: int = 0         # steps between WE resamples (0 => off)
    we_cv_min:         float = 0.0     # WE bin range min [Angstrom]
    we_cv_max:         float = 0.0     # WE bin range max [Angstrom]
    we_nbins:          int = 5         # WE bins along the CV


class BatchedGaMD(BatchedNVT):
    """N GaMD walkers ridden on BatchedNVT (one forward/step) + optional WE."""

    _ALIASES = ("md", "MD", "nvt", "NVT", "batched", "batchnvt",
                "gamd", "GaMD", "pargamd", "ParGaMD", "gamd_batched")

    def __init__(self, output: str,
                 systems: Union[Molecules, List[Atoms], Atoms],
                 calc=None,
                 paras: Optional[dict] = None):
        # 1) parse GaMD config first (need nwalkers before replicating template).
        gcfg = self._init_params(BatchedGaMDParams, paras, self._ALIASES)
        if not str(gcfg.cv_group1).strip() or not str(gcfg.cv_group2).strip():
            raise ValueError("BatchedGaMD requires cv_group1 and cv_group2 (the two "
                             "CV groups, e.g. '0,1,2' / 'heavy' -- the GaMD reweight "
                             "/ WE progress coordinate).")
        n = int(gcfg.nwalkers)

        # 2) resolve the template + replicate to N walkers (B = N). A pre-built list
        #    of >=2 systems is used as-is (B = len).
        template, calc = self._resolve_template(systems, calc)
        if isinstance(systems, (list, tuple)) and len(systems) >= 2:
            replicas = [a.copy() for a in systems]
        else:
            if n < 1:
                raise ValueError("BatchedGaMD needs nwalkers >= 1.")
            replicas = [template.copy() for _ in range(n)]

        # 3) build the batched NVT over the N walkers (calc prepared on B = N).
        super().__init__(output, replicas, calc=calc, paras=paras)
        # re-attach the GaMD fields (super() parsed only BatchedNVTParams).
        self.params = self._init_params(BatchedGaMDParams, paras, self._ALIASES)
        p = self.params

        # 4) attach the per-walker GaMD boost bias on the batch hook. boost=False
        #    (gamd off) => passive CV logger -> the unbiased reference run.
        self._boost_on = gamd_enabled(p.gamd)
        self._bias = BatchedGaMDBias(
            self.atoms_list, p.cv_group1, p.cv_group2,
            boost=self._boost_on, mode=p.gamd_mode, sigma0_kcal=p.gamd_sigma0,
            prep_steps=int(p.gamd_prep_steps), temperature=p.temperature)

        # 5) WE (ParGaMD) setup: fixed-population, equal weights to start.
        self._we_on = (gamd_enabled(p.we) and int(p.we_resample_every) > 0)
        self._we_weights = np.full(self.B, 1.0 / self.B, dtype=np.float64)
        self._we_rng = (np.random.default_rng(p.random_seed) if p.random_seed
                        is not None else np.random.default_rng())
        self._we_log = []
        if self._we_on and not (p.we_cv_max > p.we_cv_min):
            raise ValueError("BatchedGaMD WE requires we_cv_max > we_cv_min "
                             f"(got min={p.we_cv_min}, max={p.we_cv_max}).")
        # PMF results (filled by run() when boosting).
        self.pmf_x = None
        self.pmf = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _resolve_template(systems, calc):
        if isinstance(systems, Molecules):
            template = systems.multiatoms[0]
            calc = calc if calc is not None else systems.calc
        elif isinstance(systems, (list, tuple)):
            if not systems:
                raise ValueError("BatchedGaMD: empty systems list.")
            template = systems[0]
        else:
            template = systems              # single ase.Atoms template
        return template, calc

    # --- subclass hook: GaMD is implemented here, so DON'T reject it -----------
    def _reject_ceiling_features(self):
        """GaMD is implemented in this subclass; keep rejecting the OTHER batched
        ceilings (constraints/smd/plumed/colvars/posres) exactly as the base."""
        p = self.params
        bad = []
        if str(p.constraints or "none").strip().lower() not in ("", "none"):
            bad.append(f"constraints={p.constraints}")
        for nm in ("smd", "plumed", "colvars", "posres"):
            val = getattr(p, nm, "")
            if str(val or "").strip().lower() not in ("", "off", "none", "no", "false", "0"):
                bad.append(f"{nm}={val}")
        if bad:
            raise NotImplementedError(
                "ponytail: the batched kernel does not implement "
                f"{', '.join(bad)} (deliberate C3 ceiling). Use the single-system NVT.")

    # ====================================================================== run
    def run(self):
        super().run()                       # BatchedNVT advances all walkers (+WE)
        if self._boost_on:
            self._compute_pmf()
        self._log_gamd_summary()
        return self

    # --- WE rides the run loop: override only to inject periodic resampling ----
    def _run_langevin(self):
        if not self._we_on:
            return super()._run_langevin()
        return self._run_we(vrescale=False)

    def _run_vrescale(self):
        if not self._we_on:
            return super()._run_vrescale()
        return self._run_we(vrescale=True)

    def _run_we(self, vrescale):
        """Base BatchedNVT VV loop (faithfully mirrored: ``nvt_batched._run_vrescale``
        / ``_run_langevin``) with a periodic WE resample appended to the step body
        (production only). The GaMD boost itself rides ``_forces_au`` via the bias,
        so no force-loop change is needed for it -- only WE needs loop access."""
        torch = self._torch
        p = self.params
        re = int(p.we_resample_every)
        prep = int(p.gamd_prep_steps) if self._boost_on else 0
        v = self.v
        E, F = self._forces_au()
        if not vrescale:
            v = v - 0.5 * F / self.mass * self.dt_au   # standard -> LF-Middle carried
        for step in range(1, p.steps + 1):
            self._set_anneal_T(step)
            if vrescale:
                v = v + 0.5 * F / self.mass * self.dt_au
                self._displace(v, 1.0)
                E, F = self._forces_au()
                v = v + 0.5 * F / self.mass * self.dt_au
                v = self._apply_thermostat(v, vrescale=True)
                v = self._apply_projection(v, step)
                ke = 0.5 * (self.mass * v * v).sum(dim=1)
            else:
                v = v + F / self.mass * self.dt_au
                self._displace(v, 0.5)
                v = self._apply_thermostat(v, vrescale=False)
                self._displace(v, 0.5)
                E, F = self._forces_au()
                v = self._apply_projection(v, step)
                v_sync = v + 0.5 * F / self.mass * self.dt_au
                ke = 0.5 * (self.mass * v_sync * v_sync).sum(dim=1)
            self._record(ke, E)
            self._steps_done = step
            if re and step % re == 0 and step > prep:
                v = self._we_resample(v, step)
                E, F = self._forces_au()                 # refresh F for moved walkers
                # the refresh double-logs this step's frame -> drop the duplicate.
                for b in range(self.B):
                    self._bias.cv_history[b].pop()
                    self._bias.dv_history[b].pop()
                self._bias.phase_history.pop()
                self._bias._istep -= 1
        self.v = v

    # --------------------------------------------------------- WE resampling
    def _resample_bin(self, items, t):
        """Resample one bin's walkers to EXACTLY ``t`` walkers, conserving the
        bin's total weight (paper Sec 2.3 split/merge). ``items`` = list of
        ``(source_slot, weight)``. Returns the new ``items`` (len == t)."""
        items = list(items)
        if t <= 0 or not items:
            return []
        if len(items) < t:                                   # SPLIT (replicate)
            while len(items) < t:
                i = max(range(len(items)), key=lambda k: items[k][1])
                s, ws = items[i]
                items[i] = (s, ws * 0.5)
                items.append((s, ws * 0.5))
        elif len(items) > t:                                 # MERGE (combine)
            while len(items) > t:
                items.sort(key=lambda it: it[1])
                (sa, wa), (sb, wb) = items[0], items[1]
                wsum = wa + wb
                survivor = sa if (wsum <= 0.0 or self._we_rng.random() < wa / wsum) else sb
                items = items[2:] + [(survivor, wsum)]
        return items

    def _we_resample(self, v, step):
        """Bin the B walkers along the CV, split/merge to flatten occupancy across
        occupied bins (sum == B), copy walker state (coords + velocities) GPU-side,
        carry per-walker weights. Logs the resample event for the WE gate."""
        torch = self._torch
        p = self.params
        B = self.B
        cvs = np.array([self._bias.cv_history[b][-1] for b in range(B)], dtype=np.float64)
        nb = int(p.we_nbins)
        edges = np.linspace(p.we_cv_min, p.we_cv_max, nb + 1)
        bin_id = np.clip(np.digitize(cvs, edges) - 1, 0, nb - 1)
        occ = sorted(set(int(x) for x in bin_id.tolist()))
        n_occ = len(occ)
        members = {ob: [int(i) for i in np.where(bin_id == ob)[0]] for ob in occ}
        occ_before = np.array([len(members[ob]) for ob in occ], dtype=int)
        # flatten B walkers across occupied bins: base or base+1 each (sum = B);
        # the +1 goes to the currently most-populated bins.
        base = B // n_occ
        rem = B - base * n_occ
        order = sorted(occ, key=lambda ob: -len(members[ob]))
        target = {ob: base + (1 if rank < rem else 0) for rank, ob in enumerate(order)}
        occ_after = np.array([target[ob] for ob in occ], dtype=int)

        w = self._we_weights
        wsum_before = float(w.sum())
        plan = []                                            # (source_slot, weight) * B
        for ob in occ:
            items = [(s, float(w[s])) for s in members[ob]]
            plan.extend(self._resample_bin(items, target[ob]))
        assert len(plan) == B, f"WE plan length {len(plan)} != B={B}"

        # snapshot state, then realize the plan into the B physical slots.
        coord_snap = self.calc.coord.detach().clone()
        v_snap = v.detach().clone()
        cvh = [list(self._bias.cv_history[b]) for b in range(B)]
        dvh = [list(self._bias.dv_history[b]) for b in range(B)]
        ptr = self._bias._ptr_np(self.calc)
        new_w = np.empty(B, dtype=np.float64)
        with torch.no_grad():
            for j, (src, wt) in enumerate(plan):
                new_w[j] = wt
                v[j] = v_snap[src]
                self.calc.coord[ptr[j]:ptr[j + 1]] = coord_snap[ptr[src]:ptr[src + 1]]
                self._bias.cv_history[j] = list(cvh[src])
                self._bias.dv_history[j] = list(dvh[src])
        self._we_weights = new_w
        self._we_log.append(dict(
            step=int(step), n_occ=int(n_occ),
            weight_before=wsum_before, weight_after=float(new_w.sum()),
            occ_before=occ_before, occ_after=occ_after,
            occ_std_before=float(occ_before.std()), occ_std_after=float(occ_after.std())))
        return v

    # --------------------------------------------------------- reweight / PMF
    def production_samples(self):
        """Pooled production (boost-phase) frames across all walkers:
        ``(cv (M,), dV (M,) Ha)``. Prep / log-only frames are excluded."""
        ph = np.asarray(self._bias.phase_history)
        prod = (ph == "prod")
        cv_all, dv_all = [], []
        for b in range(self.B):
            cvb = np.asarray(self._bias.cv_history[b], dtype=np.float64)
            dvb = np.asarray(self._bias.dv_history[b], dtype=np.float64)
            m = prod[:len(cvb)]
            cv_all.append(cvb[m])
            dv_all.append(dvb[m])
        return np.concatenate(cv_all), np.concatenate(dv_all)

    def _compute_pmf(self):
        p = self.params
        cv, dV = self.production_samples()
        if cv.size < 2:
            return
        self.pmf_x, self.pmf = gamd_reweight_1d(
            cv, dV, temperature=p.temperature, bins=int(p.gamd_nbins),
            mode=p.gamd_reweight)

    # ----------------------------------------------------------------- logging
    def _log_gamd_summary(self):
        p = self.params
        lines = ["\n" + "=" * 72 + "\n",
                 f"{'BATCHED GaMD / ParGaMD SUMMARY':^72}\n", "=" * 72 + "\n",
                 f"  walkers (B):   {self.B}\n",
                 f"  boost:         {'on' if self._boost_on else 'off (CV logger)'}\n",
                 f"  CV groups:     g1='{p.cv_group1}'  g2='{p.cv_group2}'\n"]
        if self._boost_on and self._bias.params is not None:
            bp = self._bias.params
            lines.append(f"  GaMD params:   mode={bp['mode']} k0={bp['k0']:.4f} "
                         f"k={bp['k']:.6g}/Ha E={bp['E']:.6f} Ha\n")
            lines.append(f"  prep steps:    {p.gamd_prep_steps} (shared/pooled)\n")
        if self.pmf is not None:
            fin = self.pmf[np.isfinite(self.pmf)]
            rng = float(fin.max() - fin.min()) if fin.size else float("nan")
            lines.append(f"  reweight:      {p.gamd_reweight}  PMF range "
                         f"{rng:.4f} kcal/mol over {int(p.gamd_nbins)} bins\n")
        if self._we_on:
            lines.append(f"  WE:            on  every {p.we_resample_every} steps  "
                         f"{int(p.we_nbins)} bins [{p.we_cv_min:.3f},{p.we_cv_max:.3f}] A\n")
            lines.append(f"  WE resamples:  {len(self._we_log)}\n")
            for ev in self._we_log:
                lines.append(f"    step {ev['step']:>6}: n_occ={ev['n_occ']} "
                             f"Wtot {ev['weight_before']:.10f}->{ev['weight_after']:.10f} "
                             f"occ_std {ev['occ_std_before']:.3f}->{ev['occ_std_after']:.3f}\n")
        lines.append("=" * 72 + "\n")
        self.log_info(lines)
