"""
Population Annealing MD (PA-MD) -- rides C3 (BatchedNVT).

Population Annealing (Hukushima & Iba, AIP Conf. Proc. 690, 200 (2003);
Machta, Phys. Rev. E 82, 026704 (2010)) evolves a POPULATION of M replicas
(== the batch axis) while ANNEALING the inverse temperature along a ladder
beta_0 -> beta_target. It is an unbiased, sequential-Monte-Carlo free-energy
method: NO collective variable, NO binning. The population itself carries the
statistics (the maple advantage: the population IS the batch, so one MLIP
forward advances all M replicas per step).

At annealing NODE k (temperature ladder T_0..T_K, beta linear from beta_0 to
beta_target, K = ``n_anneal_steps`` intervals):
  1. PROPAGATE all M replicas ``n_sweep`` UNBIASED MD steps at T_k -- the
     inherited ``BatchedNVT`` kernel (ONE ``calc.get_ef_gpu()``/step over the
     padded ``(B=M, nmax_dof)`` buffer). The Velocity-Verlet / thermostat
     physics is NOT reimplemented: PA calls the factored ``_step_langevin`` /
     ``_step_vrescale`` / ``_step_nhc`` so the resampling is interleaved
     between byte-identical NVT segments (exactly how ``remd.py`` interleaves
     swaps and ``weighted_ensemble.py`` interleaves split/merge).
  2. REWEIGHT toward the next rung: w_i proportional to
     exp(-(beta_{k+1}-beta_k) * E_i), with E_i the per-replica POTENTIAL energy
     already returned by that node's last forward (no extra force eval). Weights
     are formed in log-space (log-sum-exp) so they are stable and sum to 1.
  3. RESAMPLE M replicas proportional to w_i (systematic or residual; population
     size stays M), reset weights uniform. Systematic resampling with uniform
     weights is EXACTLY the identity map -> the resample is a strict NO-OP
     (v/F/coords/RNG/thermostats untouched), so a single-temperature ladder
     (beta_0 == beta_target) is BIT-IDENTICAL to a plain ``BatchedNVT`` run of
     the same total steps + seed (the degenerate-reduction gate).
The last node (k == K) is a final equilibration sweep at T_target (no reweight),
so the surviving population is a Boltzmann sample at beta_target.

FREE ENERGY (the ladder result): each interval estimates the partition-function
ratio  Z(beta_{k+1})/Z(beta_k) = <exp(-Delta_beta E)>_{beta_k}, so
  ``lnZ_ratio`` = sum_k ln <exp(-Delta_beta_k E)> = ln[ Z(beta_target)/Z(beta_0) ]
is accumulated UNBIASED along the ladder, and the dimensionless reduced
free-energy difference is
  ``delta_betaF`` = beta_target*F(beta_target) - beta_0*F(beta_0) = -lnZ_ratio
(= integral of U dbeta, the Gibbs-Helmholtz temperature TI that a temperature
ladder measures -- an absolute F at a single T needs a separate reference and is
out of scope). ANNEALED IMPORTANCE WEIGHTS per node are exposed
(``node_energies`` / ``node_importance_weights``) for reweighted observables /
density of states, and ``observable_mean`` averages any coordinate function over
the final beta_target population.

ponytail: DELIBERATE ceilings -- (1) LINEAR-in-beta ladder only (no adaptive /
constant-ESS schedule); (2) SYSTEMATIC resampling is the default (residual is
offered; multinomial / stratified-optimal are not); (3) on a resample that
actually changes the population the per-replica NHC / v-rescale thermostat
INTERNAL state is reset for clones (Langevin's OU is stateless, so it is
unaffected; and the degenerate NO-OP path never rebuilds so bit-identity holds);
(4) the free energy is the temperature-ladder log-Z ratio (reduced free energy),
not an absolute F. All ``BatchedNVT`` ceilings (RATTLE / GaMD / SMD / PLUMED /
Colvars / posres, charge-coupled batch calcs) are inherited unchanged.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from ase import Atoms

from maple.function.timer import timer
from maple.function.utility import Molecules

from ..utils import KELVIN_TO_HARTREE
from .nvt_batched import BatchedNVT, BatchedNVTParams


HARTREE_TO_KCAL_MOL = 627.5094740631  # Ha -> kcal/mol


# ===================================================== resampling (pure numpy)
def systematic_resample(weights, rng) -> np.ndarray:
    """Systematic (a.k.a. universal / low-variance) resampling of M weighted
    replicas -> M parent indices. ONE uniform draw u ~ U[0,1) places M evenly
    spaced pointers (u+j)/M, j=0..M-1, along the cumulative weight; pointer j
    selects the replica whose cumulative interval it lands in. Expected count of
    replica i is exactly M*w_i (unbiased), with the minimum resampling variance
    of the standard schemes.

    Pure numpy (no torch) so it is unit-testable in isolation. With UNIFORM
    weights (w_i == 1/M) the pointers land one-per-interval and the returned
    indices are EXACTLY ``arange(M)`` (the identity), which the driver uses as a
    bit-identity NO-OP fast path."""
    w = np.asarray(weights, dtype=float)
    M = w.size
    if M == 0:
        return np.empty(0, dtype=int)
    tot = w.sum()
    if not (tot > 0):
        raise ValueError(f"systematic_resample: non-positive weight sum {tot!r}.")
    w = w / tot
    positions = (rng.random() + np.arange(M)) / M
    cumsum = np.cumsum(w)
    cumsum[-1] = 1.0                                  # guard fp round-off at the top
    # side='right' -> smallest i with cumsum[i] > position; gives identity for
    # uniform weights for ANY u in [0,1) (side='left' can miss the edge at u==0).
    return np.searchsorted(cumsum, positions, side="right").astype(int)


def residual_resample(weights, rng) -> np.ndarray:
    """Residual resampling -> M parent indices. Deterministic part: n_i =
    floor(M*w_i) guaranteed copies of replica i. Residual part: the remaining
    R = M - sum n_i slots are drawn from the normalized residual weights
    (M*w_i - n_i). Lower variance than plain multinomial. Pure numpy."""
    w = np.asarray(weights, dtype=float)
    M = w.size
    if M == 0:
        return np.empty(0, dtype=int)
    tot = w.sum()
    if not (tot > 0):
        raise ValueError(f"residual_resample: non-positive weight sum {tot!r}.")
    w = w / tot
    scaled = M * w
    counts = np.floor(scaled).astype(int)
    idx = np.repeat(np.arange(M), counts)
    R = int(M - counts.sum())
    if R > 0:
        resid = scaled - counts
        s = resid.sum()
        resid = (resid / s) if s > 0 else np.full(M, 1.0 / M)
        extra = rng.choice(M, size=R, replace=True, p=resid)
        idx = np.concatenate([idx, extra])
    return idx.astype(int)


def resample_indices(weights, rng, method="systematic") -> np.ndarray:
    m = str(method or "systematic").strip().lower()
    if m in ("systematic", "sys", "low-variance", "universal"):
        return systematic_resample(weights, rng)
    if m in ("residual", "res"):
        return residual_resample(weights, rng)
    raise ValueError(f"Unknown resample method {method!r} (use 'systematic'|'residual').")


# ============================================= free-energy ladder increment (numpy)
def reduced_free_energy_increment(energies, dbeta):
    """One annealing interval's log partition-function ratio + normalized weights.

    ``delta_lnZ`` = ln[ (1/M) sum_i exp(-dbeta * E_i) ] estimates
    ln[ Z(beta+dbeta)/Z(beta) ] from a population equilibrated at ``beta``
    (unbiased Monte-Carlo estimator of <exp(-dbeta E)>_beta). Formed with the
    log-sum-exp trick so it is stable for large |dbeta*E| and either sign of
    ``dbeta`` (cooling dbeta>0 or heating dbeta<0). Returns
    ``(delta_lnZ, weights)`` with weights = softmax(-dbeta E) (sum == 1)."""
    e = np.asarray(energies, dtype=float).reshape(-1)
    M = e.size
    if M == 0:
        raise ValueError("reduced_free_energy_increment: empty energy array.")
    logw = -float(dbeta) * e
    c = float(logw.max())
    shifted = np.exp(logw - c)
    denom = float(shifted.sum())
    lse = c + np.log(denom)                           # ln sum_i exp(logw_i)
    delta_lnZ = lse - np.log(M)                       # ln mean_i exp(logw_i)
    weights = shifted / denom                         # softmax, sum == 1
    return float(delta_lnZ), weights


@dataclass
class PopulationAnnealingParams(BatchedNVTParams):
    """PA-MD params = the batched-NVT params PLUS the annealing controls.

    Inherits every ``BatchedNVTParams`` field (timestep / thermostat / friction /
    tau_t / hmr / remove_com_every ... honored by the inherited batched-NVT
    kernel). ``temperature`` is pinned to ``temp_start`` at construction (the
    velocity-init + node-0 temperature; per-node targets come from the ladder).
    ``steps`` is overwritten with ``n_sweep * (n_anneal_steps + 1)`` (PA drives
    the segments itself)."""
    n_replicas:      int   = 64             # M = population = initial batch B
    temp_start:      float = 600.0          # K at beta_0 (ladder start)
    temp_target:     float = 300.0          # K at beta_target (ladder end)
    n_anneal_steps:  int   = 20             # K annealing INTERVALS (ladder has K+1 nodes)
    n_sweep:         int   = 50             # unbiased MD steps per node
    schedule:        str   = "linear-beta"  # inverse-temperature ladder (linear in beta only)
    resample_method: str   = "systematic"   # "systematic" | "residual"
    resample_seed:   Optional[int] = None   # RNG for resampling + clone streams


class PopulationAnnealing(BatchedNVT):
    """Population Annealing MD built ON TOP of ``BatchedNVT`` (reuses its VV /
    thermostat / projection kernel; adds only the beta ladder + reweight/resample
    + the log-Z free-energy accumulator)."""

    _ALIASES = ("pa", "PA", "popann", "population_annealing", "populationannealing",
                "md", "MD", "nvt", "NVT", "batched")
    _SCHEDULES = {"linear-beta", "linear_beta", "beta", "linear", ""}

    def __init__(self, output: str,
                 systems: Union[Molecules, List[Atoms], Atoms],
                 calc=None,
                 paras: Optional[dict] = None):
        # ---- parse PA config first (need n_replicas BEFORE building the batch) --
        pp = self._init_params(PopulationAnnealingParams, paras, self._ALIASES)
        M = int(pp.n_replicas)
        if M < 1:
            raise ValueError(f"PA needs n_replicas >= 1 (got {M}).")
        if int(pp.n_anneal_steps) < 1:
            raise ValueError("PA needs n_anneal_steps >= 1.")
        if int(pp.n_sweep) < 1:
            raise ValueError("PA needs n_sweep >= 1.")
        if not (pp.temp_start > 0.0 and pp.temp_target > 0.0):
            raise ValueError(f"PA needs temp_start ({pp.temp_start}) and temp_target "
                             f"({pp.temp_target}) > 0 K.")
        if str(pp.schedule or "").strip().lower() not in self._SCHEDULES:
            raise ValueError(f"PA schedule '{pp.schedule}' not supported "
                             f"(ponytail ceiling: linear-in-beta ladder only).")
        # velocity-init + node-0 temperature = the ladder start; anneal disabled so
        # the PA-controlled per-node thermostat temps are not clobbered by BatchedNVT.
        pp.temperature = float(pp.temp_start)
        pp.anneal = ""

        # ---- resolve template + initial population (B = M) ---------------------
        template, calc = self._resolve_template(systems, calc)
        if isinstance(systems, (list, tuple)) and len(systems) == M:
            replicas = [a.copy() for a in systems]          # pre-built distinct starts
        else:
            replicas = [template.copy() for _ in range(M)]
        n_per = len(template)
        if any(len(r) != n_per for r in replicas):
            raise ValueError("PA requires a HOMOGENEOUS population topology (same atom "
                             "count/species): the population is resampled on the batch "
                             "axis over interchangeable replicas.")

        # ---- build the M-replica batch and run BatchedNVT's setup/gates --------
        super().__init__(output, replicas, calc=calc, paras=paras)
        # BatchedNVT parsed a BatchedNVTParams; swap in the richer PA params (superset,
        # so the inherited _prepare_buffers / _step_* read it unchanged).
        self.params = pp
        if self.params.thermostat not in self._THERMOSTAT_CHOICES:
            raise ValueError(f"Unknown thermostat '{self.params.thermostat}'. "
                             f"Choose from: {self._THERMOSTAT_CHOICES}")
        self._reject_ceiling_features()

        self._M = M
        self._K = int(pp.n_anneal_steps)
        self._n_sweep = int(pp.n_sweep)
        # PA drives its own segment loop; steps = full trajectory length so the
        # inherited _record's "always record the final step" cadence is correct.
        self.params.steps = self._n_sweep * (self._K + 1)
        self._template = template.copy()
        self._n_per = n_per

        # ---- inverse-temperature ladder (linear in beta) -----------------------
        kB = KELVIN_TO_HARTREE                           # Hartree / K
        beta_0 = 1.0 / (kB * float(pp.temp_start))
        beta_K = 1.0 / (kB * float(pp.temp_target))
        self._betas = np.linspace(beta_0, beta_K, self._K + 1)     # (K+1,) 1/Ha
        self._temps = 1.0 / (kB * self._betas)                     # (K+1,) K

        # ---- resampling controls -----------------------------------------------
        self._resample_method = str(pp.resample_method or "systematic").strip().lower()
        if self._resample_method not in ("systematic", "residual"):
            raise ValueError(f"PA resample_method '{pp.resample_method}' unknown "
                             "(use 'systematic' or 'residual').")
        self._resample_seed = pp.resample_seed
        self._cur_T = float(pp.temp_start)

        # results (filled by run()/_finalize_pa)
        self.weights = None
        self.lnZ_ratio = 0.0
        self.delta_betaF = None
        self.delta_lnZ_per_step = None
        self.ess = None
        self.betas = None
        self.temps_ladder = None
        self.node_energies = None
        self.node_importance_weights = None
        self.n_replicas_hist = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _resolve_template(systems, calc):
        if isinstance(systems, Molecules):
            template = systems.multiatoms[0]
            calc = calc if calc is not None else systems.calc
        elif isinstance(systems, (list, tuple)):
            if not systems:
                raise ValueError("PA: empty systems list.")
            template = systems[0]
        else:
            template = systems              # single ase.Atoms template
        return template, calc

    def _replica_positions(self):
        """Per-replica positions (list of (n,3) Angstrom numpy) from the calc master."""
        from ..bias.batched import _ptr_to_np  # sanctioned torch|numpy _ptr coercion
        coord = self.calc.coord.detach().to("cpu").numpy()
        ptr = _ptr_to_np(self.calc._ptr).tolist()
        return [coord[ptr[b]:ptr[b + 1]].copy() for b in range(self.B)]

    def _rebuild_thermostats(self):
        """Rebuild the per-replica AUTHORITATIVE thermostat objects keyed to the
        current RNG streams (used after a resample that reorders replicas). Built at
        the current node temperature ``self._cur_T``; the loop re-sets the target for
        the next node. ponytail: NHC / v-rescale INTERNAL state is reset here (clones
        start a fresh chain); Langevin's OU is stateless so it is unaffected, and the
        degenerate NO-OP path never calls this so bit-identity is preserved."""
        from ..thermostat.langevin import LangevinThermostat
        from ..thermostat.vrescale import VRescaleThermostat
        from ..thermostat.nose_hoover import NoseHooverChain
        self._thermostats = []
        for b, at in enumerate(self.atoms_list):
            if self.params.thermostat == "v-rescale":
                th = VRescaleThermostat(at, temperature=self._cur_T,
                                        tau_t=self.params.tau_t,
                                        timestep=self.params.timestep,
                                        rng=self._rngs[b], n_dof=int(self.n_dof[b]))
            elif self.params.thermostat in ("nose-hoover", "nhc"):
                th = NoseHooverChain(at, temperature=self._cur_T,
                                     tau_t=self.params.tau_t,
                                     timestep=self.params.timestep,
                                     n_dof=int(self.n_dof[b]),
                                     chain_length=self.params.chain_length,
                                     n_respa=self.params.nhc_n_respa,
                                     n_yoshida=self.params.nhc_n_yoshida)
            else:
                th = LangevinThermostat(at, temperature=self._cur_T,
                                        friction=self.params.friction,
                                        timestep=self.params.timestep,
                                        rng=self._rngs[b])
            self._thermostats.append(th)

    def _set_node_temperature(self, T):
        """Set every replica's thermostat target to the current ladder rung T (K).
        set_temperature only recomputes the noise scale (no RNG draw), so calling it
        with an unchanged T is a no-op -> the degenerate single-T ladder is unperturbed."""
        self._cur_T = float(T)
        for th in self._thermostats:
            th.set_temperature(self._cur_T)
        # fused-loop path: set_temperature retuned each thermostat's ._c2 (noise scale
        # ~ sqrt(kT)); the on-device _c2_dev built at _prepare_buffers() is now stale
        # for the new rung T and would SILENTLY inject the wrong noise amplitude (no
        # crash -- B is constant -- but a biased free-energy ladder). Re-pull c2 into
        # _c2_dev from the just-retuned thermostats (also picks up any resample reorder
        # of _thermostats). c1 is T-independent so it never needs a refresh. Guard
        # mirrors _prepare_buffers; no-op unless fused Langevin is active.
        if getattr(self, "_fused", False) and self.params.thermostat == "langevin":
            self._refresh_c2_dev()

    # ============================================================= resample step
    def _reindex(self, idx, v_std):
        """Reorder the population by parent-index array ``idx`` (length M): gather
        positions (write back to the calc master coords), gather standard velocities
        (row gather of the padded buffer -- homogeneous topology, B constant), gather
        RNG streams (a duplicated parent -> a fresh stream for each clone so clones
        decorrelate), and rebuild the per-replica thermostats. mass/mask buffers are
        unchanged (constant B, identical topology). Returns the reordered v_std."""
        torch = self._torch
        idx = np.asarray(idx, dtype=int)
        # positions: gather per-replica, reorder, overwrite the master coord buffer.
        pos = self._replica_positions()
        new_flat = np.concatenate([pos[i] for i in idx], axis=0)
        self.calc.set_coords_(torch.tensor(new_flat, dtype=self.dtype, device=self.device))
        # standard velocities: clean row gather (all replicas share nmax_dof).
        gather = torch.as_tensor(idx, dtype=torch.long, device=v_std.device)
        v_new = v_std[gather].clone()
        # RNG streams: first occurrence keeps the parent stream; each further copy of
        # the same parent (a split clone) gets a fresh independent stream.
        old_rng = list(self._rngs)
        new_rng = []
        used = {}
        for i in idx:
            i = int(i)
            c = used.get(i, 0)
            used[i] = c + 1
            if c == 0:
                new_rng.append(old_rng[i])
            else:
                new_rng.append(np.random.default_rng(int(self._resample_rng.integers(1 << 62))))
        self._rngs = new_rng
        self._rebuild_thermostats()
        return v_new

    def _resample(self, w, v, F, use_carried):
        """Resample the population proportional to ``w`` toward the next rung. On the
        identity plan (systematic resampling of uniform weights == arange(M)) it is a
        strict NO-OP: v/F/coords/RNG/thermostats untouched (bit-identity for the
        degenerate single-T ladder). Otherwise the population is reordered and (v, F)
        recomputed for it. Returns (v, F)."""
        # convert carried -> standard velocity for a clean reorder (mirror WE).
        v_std = (v + 0.5 * F / self.mass * self.dt_au) if use_carried else v
        idx = resample_indices(w, self._resample_rng, self._resample_method)
        if idx.size != self.B:
            raise AssertionError(f"PA resample changed population size: {idx.size} != {self.B}")

        # identity NO-OP fast path (preserves degenerate == BatchedNVT bit-identity).
        if np.array_equal(idx, np.arange(self.B)):
            self._weights = np.full(self.B, 1.0 / self.B, dtype=float)
            return v, F

        v_std = self._reindex(idx, v_std)
        self.v = v_std
        self._weights = np.full(self.B, 1.0 / self.B, dtype=float)   # reset uniform
        # recompute the force for the reordered configuration; re-derive carried v.
        _E, F = self._forces_au()
        v = (v_std - 0.5 * F / self.mass * self.dt_au) if use_carried else v_std
        return v, F

    # ====================================================================== run
    def run(self):
        with timer(f"Population Annealing (PA-MD, M={self.B})"):
            self._log_parameters_pa()
            self._prepare_buffers()                      # init v (std), rngs seed+b, thermostats
            self._init_pa_state()
            self._run_pa()
            self._finalize_pa()
        return self

    def _init_pa_state(self):
        self._weights = np.full(self.B, 1.0 / self.B, dtype=float)
        self._resample_rng = np.random.default_rng(self._resample_seed)
        self._delta_lnZ = []
        self._ess = []
        self._node_E = []
        self._node_w = []
        self.n_replicas_hist = []
        self.lnZ_ratio = 0.0

    def _run_pa(self):
        """PA loop: for each ladder node, equilibrate the population ``n_sweep`` steps
        via the inherited batched kernel, then (except at the last node) reweight to
        the next rung and resample. Mirrors ``remd._run_remd`` / ``we._run_we`` --
        segments of the factored NVT step with the population bookkeeping interleaved."""
        th = self.params.thermostat
        if th == "v-rescale":
            step_fn, use_carried = self._step_vrescale, False
        elif th in ("nose-hoover", "nhc"):
            step_fn, use_carried = self._step_nhc, False
        else:
            step_fn, use_carried = self._step_langevin, True

        v = self.v
        E, F = self._forces_au()
        if use_carried:                                  # std -> LF-Middle carried, ONCE
            v = v - 0.5 * F / self.mass * self.dt_au
        step = 0
        for k in range(self._K + 1):
            self._set_node_temperature(float(self._temps[k]))
            for _ in range(self._n_sweep):
                step += 1
                v, E, F = step_fn(v, F, step)
            if k < self._K:
                dbeta = float(self._betas[k + 1] - self._betas[k])
                E_np = E.detach().to("cpu").numpy().reshape(-1)      # (M,) PE [Ha]
                delta_lnZ, w = reduced_free_energy_increment(E_np, dbeta)
                # GATE 3 (runtime): weight normalization to 1e-12 after every reweight.
                assert abs(float(w.sum()) - 1.0) < 1e-12, \
                    f"PA weights not normalized: sum w = {w.sum()!r} (node {k})"
                self.lnZ_ratio += delta_lnZ
                self._delta_lnZ.append(delta_lnZ)
                self._ess.append(1.0 / float(np.sum(w * w)))         # effective sample size
                self._node_E.append(E_np.copy())
                self._node_w.append(w.copy())
                v, F = self._resample(w, v, F, use_carried)
                self.n_replicas_hist.append(self.B)
        self.v = (v + 0.5 * F / self.mass * self.dt_au) if use_carried else v

    # --------------------------------------------------------------- finalize
    def _finalize_pa(self):
        self.weights = self._weights.copy()
        self.lnZ_ratio = float(self.lnZ_ratio)                 # ln[ Z(beta_target)/Z(beta_0) ]
        self.delta_betaF = -self.lnZ_ratio                     # beta_t F_t - beta_0 F_0 (reduced)
        self.delta_lnZ_per_step = np.asarray(self._delta_lnZ, dtype=float)
        self.ess = np.asarray(self._ess, dtype=float)
        self.betas = self._betas.copy()
        self.temps_ladder = self._temps.copy()
        self.node_energies = list(self._node_E)                # per-interval PE [Ha]
        self.node_importance_weights = list(self._node_w)      # per-interval annealed weights
        self.n_replicas_hist = np.asarray(self.n_replicas_hist, dtype=int)
        self.beta_start = float(self._betas[0])
        self.beta_target = float(self._betas[-1])
        self.temp_start_K = float(self._temps[0])
        self.temp_target_K = float(self._temps[-1])
        self._log_summary_pa()

    # -------------------------------------------------------------- observables
    def observable_mean(self, obs_fn):
        """Population average <O>_{beta_target} of a coordinate function over the FINAL
        (uniform-weight, Boltzmann-at-beta_target) population. ``obs_fn`` maps a
        replica's positions (n,3) Angstrom to a scalar."""
        pos = self._replica_positions()
        O = np.array([float(obs_fn(pos[b])) for b in range(self.B)], dtype=float)
        return float(np.dot(self._weights, O))

    def reweighted_observable(self, obs_fn, node=-1):
        """Annealed-importance-weighted <O>_{beta_{node+1}} from node ``node``'s
        pre-resample population (weights = ``node_importance_weights[node]`` applied to
        the energies/positions at that node). Provided for reweighted observables / DOS
        on the fly; ``observable_mean`` (final population) is the simplest estimator."""
        w = self._node_w[node]
        pos = self._replica_positions()   # NOTE: positions are the CURRENT population
        O = np.array([float(obs_fn(pos[b])) for b in range(len(pos))], dtype=float)
        return float(np.dot(w, O) / w.sum())

    # ----------------------------------------------------------------- logging
    def _log_parameters_pa(self):
        p = self.params
        L = " / ".join(f"{t:.1f}" for t in self._temps)
        lines = ["\n" + "=" * 72 + "\n", f"{'POPULATION ANNEALING (PA-MD) PARAMETERS':^72}\n",
                 "=" * 72 + "\n",
                 f"Population (M):  {self.B}\n",
                 f"Calculator:      {type(self.calc).__name__} (one get_ef_gpu/step)\n",
                 f"Thermostat:      {p.thermostat}\n",
                 f"Timestep:        {p.timestep:.3f} fs\n",
                 f"n_sweep:         {self._n_sweep}   anneal intervals (K): {self._K}\n",
                 f"Total MD steps:  {p.steps}  = n_sweep x (K+1)\n",
                 f"Schedule:        {p.schedule} (linear in beta)\n",
                 f"T ladder (K):    {L}\n",
                 f"Resampling:      {self._resample_method}\n"]
        if p.resample_seed is not None:
            lines.append(f"Resample seed:   {p.resample_seed}\n")
        if p.random_seed is not None:
            lines.append(f"MD seed:         {p.random_seed}\n")
        lines.append("=" * 72 + "\n")
        self.log_info(lines)

    def _log_summary_pa(self):
        kT_t = self.temp_target_K * KELVIN_TO_HARTREE
        lines = ["\n" + "=" * 72 + "\n", f"{'PA-MD SUMMARY':^72}\n", "=" * 72 + "\n",
                 f"  nodes={self._K + 1}  M={self.B}  sum w={float(self._weights.sum()):.12f}\n",
                 f"  T_0={self.temp_start_K:.2f} K  ->  T_target={self.temp_target_K:.2f} K\n",
                 "\n  -- free-energy ladder (partition-function ratio) --\n",
                 f"  ln[ Z(T_target)/Z(T_0) ]           = {self.lnZ_ratio:+.6f}\n",
                 f"  reduced dF  (beta_t F_t - beta_0 F_0) = {self.delta_betaF:+.6f}  (dimensionless)\n"]
        if self.ess.size:
            lines.append(f"  ESS per interval: min={self.ess.min():.1f}  "
                         f"mean={self.ess.mean():.1f}  max={self.ess.max():.1f}  (of M={self.B})\n")
        lines.append("=" * 72 + "\n")
        self.log_info(lines)
