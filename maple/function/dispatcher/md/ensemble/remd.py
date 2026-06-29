"""
Temperature replica-exchange MD (T-REMD / parallel tempering) -- rides C3.

REMD runs N copies of the SAME system on a geometric temperature ladder and
periodically attempts Metropolis swaps between ladder-adjacent replicas, so a
barrier crossing sampled at high T propagates down to the 300 K replica. Between
swaps this is EXACTLY N independent NVT trajectories at different T -- i.e. the N
replicas ARE the batch dimension of ``BatchedNVT``. There are NO bias forces
(unlike umbrella/GaMD/metaD), so REMD rides ``BatchedNVT`` DIRECTLY:

  * Per-step propagation of all N replicas is the inherited ``BatchedNVT`` kernel
    -- ONE ``calc.get_ef_gpu()`` per step over the padded ``(B=N, nmax_dof)``
    buffer. The Velocity-Verlet / thermostat physics is NOT reimplemented here;
    REMD calls the factored ``_step_vrescale`` / ``_step_langevin`` so the swap
    bookkeeping is interleaved between byte-identical NVT steps.
  * Each replica's thermostat target = its ladder temperature T_i (geometric
    ladder T_i = T_min * (T_max/T_min)^(i/(N-1))). Velocities are initialized at
    T_min then rescaled per replica to T_i (Maxwell-Boltzmann is linear in
    sqrt(T), so this == drawing at T_i with the same RNG draws).
  * Every ``exchange_every`` steps: attempt swaps on ALTERNATING even/odd
    ladder-adjacent neighbor pairs. Acceptance on the POTENTIAL energy already
    returned by that step's ``get_ef_gpu`` (NO extra force eval):
        p = min(1, exp((beta_i - beta_j)(E_i - E_j))),  beta = 1/(k_B T).
    On accept (RELABEL convention, task spec): swap the two replicas' TARGET
    temperatures and rescale each velocity set by sqrt(T_new/T_old) so the KE
    matches the new target. The swap proposal is symmetric (Metropolis), so
    detailed balance holds.

ponytail: TEMPERATURE-REMD ONLY. REST2 / Hamiltonian-REMD / alchemical lambda
ladders need a force-field ENERGY DECOMPOSITION (scaled solute-solute /
solute-solvent terms) that a pure black-box MLIP does not expose -- they are
REJECTED here with a clear error (declared ``mode`` field so jobABC keeps the
key -- B-51 -- but only temperature mode is implemented). All ``BatchedNVT``
ceilings (RATTLE / GaMD / SMD / PLUMED / Colvars / posres, periodic cells,
charge-coupled batch calcs) are inherited unchanged.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from ase import Atoms

from ...jobABC import JobABC
from maple.function.timer import timer
from maple.function.utility import Molecules

from ..utils import KELVIN_TO_HARTREE
from .nvt_batched import BatchedNVT, BatchedNVTParams


@dataclass
class REMDParams(BatchedNVTParams):
    """Parameters for temperature-REMD. Inherits every ``BatchedNVTParams`` field
    (timestep/thermostat/friction/tau_t/hmr/... are honored by the inherited
    batched NVT kernel) and adds the ladder + exchange controls.

    ``temperature`` is reused as the velocity-initialization temperature and is
    pinned to ``temp_min`` at construction (per-replica targets come from the
    ladder, not this scalar). ``mode`` is DECLARED so jobABC does not strip it
    (B-51) but only ``temperature`` is implemented."""
    n_replicas:    int   = 4                # N replicas = batch B (>= 2)
    temp_min:      float = 300.0            # K (target / lowest rung)
    temp_max:      float = 500.0            # K (highest rung)
    exchange_every: int  = 100              # steps between swap sweeps (<=0 => never)
    mode:          str   = "temperature"    # temperature | temp | t-remd  (only mode)
    swap_seed:     Optional[int] = None     # RNG for swap accept draws (default: derived)


class REMD(BatchedNVT):
    """Temperature replica-exchange MD built ON TOP of ``BatchedNVT`` (reuses its
    VV/thermostat/projection kernel; adds only the ladder + Metropolis swaps)."""

    _TEMP_MODES = {"", "temperature", "temp", "t", "t-remd", "tremd", "temp-remd",
                   "parallel-tempering", "pt"}
    _REJECT_MODE_HINTS = ("rest2", "rest", "hamiltonian", "h-remd", "hremd",
                          "alchemical", "lambda", "fep")

    def __init__(self, output: str,
                 system: Union[Molecules, List[Atoms], Atoms],
                 calc=None,
                 paras: Optional[dict] = None):
        # ---- resolve the single template system (REMD replicates ONE system) ---
        if isinstance(system, Molecules):
            templates = list(system.multiatoms)
            calc = calc if calc is not None else system.calc
        elif isinstance(system, Atoms):
            templates = [system]
        else:
            templates = list(system)
        if not templates:
            raise ValueError("REMD requires one template system to replicate.")
        if len(templates) > 1:
            raise ValueError(
                "REMD replicates a SINGLE system across the temperature ladder; got "
                f"{len(templates)} distinct systems. Pass one Atoms (or a Molecules / "
                "list holding exactly one structure).")
        template = templates[0]

        # ---- parse REMD params (need n_replicas BEFORE building the batch) -----
        rp = REMDParams()
        rp = self._update_dataclass_from_dict(
            rp, self._select_subdict(paras, ("remd", "REMD", "tremd", "replica", "md", "MD")))
        self._validate_mode(rp.mode)
        n = int(rp.n_replicas)
        if n < 2:
            raise ValueError(f"REMD needs n_replicas >= 2 (got {n}); for one replica "
                             "use the single-system NVT or BatchedNVT.")
        if not (rp.temp_max > rp.temp_min > 0.0):
            raise ValueError(f"REMD ladder needs temp_max ({rp.temp_max}) > temp_min "
                             f"({rp.temp_min}) > 0.")

        # geometric ladder (ascending); replica b is initially labeled ladder[b].
        ladder = rp.temp_min * (rp.temp_max / rp.temp_min) ** (
            np.arange(n) / (n - 1))
        self.ladder = ladder.astype(float)

        # velocities are drawn at T_min then rescaled to the ladder (-> draw at T_i).
        rp.temperature = float(rp.temp_min)
        rp.anneal = ""                       # ladder targets are fixed, not annealed

        # ---- build the N-replica batch and run BatchedNVT's setup/gates --------
        replicas = [template.copy() for _ in range(n)]
        super().__init__(output, replicas, calc=calc, paras=paras)
        # BatchedNVT parsed a BatchedNVTParams; swap in the richer REMDParams (it is
        # a superset, so the inherited _prepare_buffers / _step_* read it unchanged).
        self.params = rp
        # re-validate against the REMD params (thermostat choice + inherited ceilings).
        if self.params.thermostat not in self._THERMOSTAT_CHOICES:
            raise ValueError(f"Unknown thermostat '{self.params.thermostat}'. "
                             f"Choose from: {self._THERMOSTAT_CHOICES}")
        self._reject_ceiling_features()

        self.exchange_every = int(rp.exchange_every)
        self.temps = self.ladder.copy()      # current per-replica TARGET temps (K)
        # dedicated swap-accept RNG (separate from per-replica thermostat streams so
        # swaps never perturb the NVT noise -> no-swap REMD == plain BatchedNVT).
        ss = rp.swap_seed
        if ss is None and rp.random_seed is not None:
            ss = int(rp.random_seed) + 1_000_003
        self._swap_rng = np.random.default_rng(ss)
        # per ladder-rank-boundary swap counters (rank r <-> rank r+1), r=0..N-2.
        self._n_attempt = np.zeros(self.B - 1, dtype=np.int64)
        self._n_accept = np.zeros(self.B - 1, dtype=np.int64)
        self._swap_round = 0
        self._hist_temps: List[np.ndarray] = []

    # --------------------------------------------------------------- mode gate
    @classmethod
    def _validate_mode(cls, mode):
        m = str(mode or "").strip().lower()
        if any(h in m for h in cls._REJECT_MODE_HINTS):
            raise NotImplementedError(
                f"REMD mode '{mode}' (Hamiltonian/REST2/alchemical replica exchange) "
                "needs a force-field ENERGY DECOMPOSITION (scaled solute-solute / "
                "solute-solvent terms) that a pure black-box MLIP does not expose. "
                "Only TEMPERATURE-REMD is implemented (mode='temperature').")
        if m not in cls._TEMP_MODES:
            raise ValueError(f"Unknown REMD mode '{mode}'. Only temperature-REMD is "
                             f"supported (mode in {sorted(cls._TEMP_MODES)}).")

    # ----------------------------------------------- ladder -> buffers (post-prepare)
    def _apply_ladder(self):
        """Set each replica's thermostat target to its ladder temperature and
        rescale its (standard, freshly-initialized) velocities from T_min to T_i.
        Called AFTER ``_prepare_buffers`` (which seeded v at T_min)."""
        for b in range(self.B):
            T_i = float(self.ladder[b])
            scale = (T_i / float(self.params.temp_min)) ** 0.5
            if scale != 1.0:
                self.v[b] = self.v[b] * scale
            self._thermostats[b].set_temperature(T_i)
        self.temps = self.ladder.copy()

    # ============================================================= swap attempt
    def _attempt_swaps(self, v, F, E, langevin: bool):
        """Attempt Metropolis swaps on alternating ladder-adjacent pairs using the
        PE ``E`` already returned this step (NO extra force eval). RELABEL
        convention: on accept, swap the two replicas' TARGET temperatures and
        rescale their velocities by sqrt(T_new/T_old)."""
        kB = KELVIN_TO_HARTREE                                   # Hartree / K
        E_np = E.detach().to("cpu").numpy().reshape(-1)          # (B,) PE [Ha]
        temps = self.temps
        order = np.argsort(temps, kind="stable")                 # replicas by asc T
        parity = self._swap_round % 2                            # 0 even, 1 odd sweep
        self._swap_round += 1

        # physical (standard) velocities for the KE-matching rescale; langevin
        # carries v_carried = v_std - 0.5*(F/m)*dt -> convert with the cached F.
        half = 0.5 * F / self.mass * self.dt_au
        v_std = (v + half) if langevin else v
        v_std = v_std.clone()

        for r in range(parity, self.B - 1, 2):
            a = int(order[r]); b = int(order[r + 1])             # ladder neighbors
            Ta = float(temps[a]); Tb = float(temps[b])           # Ta <= Tb
            if Tb <= Ta:
                continue
            beta_a = 1.0 / (kB * Ta)
            beta_b = 1.0 / (kB * Tb)
            delta = (beta_a - beta_b) * (E_np[a] - E_np[b])
            self._n_attempt[r] += 1
            p = 1.0 if delta >= 0.0 else float(np.exp(delta))
            if self._swap_rng.random() < p:
                self._n_accept[r] += 1
                temps[a], temps[b] = Tb, Ta                      # relabel targets
                v_std[a] = v_std[a] * (Tb / Ta) ** 0.5           # a now targets Tb
                v_std[b] = v_std[b] * (Ta / Tb) ** 0.5           # b now targets Ta
                self._thermostats[a].set_temperature(temps[a])
                self._thermostats[b].set_temperature(temps[b])

        return (v_std - half) if langevin else v_std

    # ================================================================== run loop
    def _run_remd(self):
        """N-replica NVT via the inherited batched kernel, with swap sweeps every
        ``exchange_every`` steps. exchange_every<=0 => never swap => the trajectory
        is EXACTLY the independent N-replica BatchedNVT run (parity gate)."""
        langevin = (self.params.thermostat != "v-rescale")
        v = self.v
        E, F = self._forces_au()
        if langevin:                                             # std -> carried at t=0
            v = v - 0.5 * F / self.mass * self.dt_au
        ex = self.exchange_every
        self._hist_temps = []
        for step in range(1, self.params.steps + 1):
            if langevin:
                v, E, F = self._step_langevin(v, F, step)
            else:
                v, E, F = self._step_vrescale(v, F, step)
            self._hist_temps.append(self.temps.copy())           # label DURING this step
            if ex and ex > 0 and step % ex == 0 and step < self.params.steps:
                v = self._attempt_swaps(v, F, E, langevin)
        self.v = v

    # ====================================================================== run
    def run(self):
        with timer(f"REMD (T-ladder, N={self.B})"):
            self._log_parameters_remd()
            self._prepare_buffers()
            self._apply_ladder()
            self._run_remd()
            self._finalize_remd()
        return self

    # --------------------------------------------------------------- finalize
    def _finalize_remd(self):
        np_ = np
        torch = self._torch
        if not self._hist_T:
            self.results, self.exchange_stats, self.slot_results = [], [], []
            return
        T = torch.stack(self._hist_T, 0).detach().to("cpu").numpy()      # (nstep, B)
        KE = torch.stack(self._hist_KE, 0).detach().to("cpu").numpy()
        PE = torch.stack(self._hist_PE, 0).detach().to("cpu").numpy()
        temps_hist = np_.asarray(self._hist_temps)                       # (nstep, B)
        nstep = T.shape[0]
        burn = nstep // 2                                                 # last 50%

        # ---- per ladder-temperature slot: measured kinetic T (relabel-aware) ----
        # group instantaneous T by the TARGET label each walker held that step.
        self.slot_results = []
        for tk in self.ladder:
            mask = np_.isclose(temps_hist, tk)
            occ = int(mask.any(axis=1).sum())                            # steps occupied
            tail = mask.copy()
            tail[:burn] = False
            vals = T[tail]
            self.slot_results.append(dict(
                T_target=float(tk),
                T_measured=float(np_.mean(vals)) if vals.size else float("nan"),
                T_std=float(np_.std(vals)) if vals.size else float("nan"),
                steps_occupied=occ))

        # ---- per ladder-adjacent pair: swap acceptance --------------------------
        self.exchange_stats = []
        for r in range(self.B - 1):
            att = int(self._n_attempt[r]); acc = int(self._n_accept[r])
            self.exchange_stats.append(dict(
                pair_rank=(r, r + 1),
                T_lo=float(self.ladder[r]), T_hi=float(self.ladder[r + 1]),
                attempts=att, accepts=acc,
                rate=(acc / att) if att else float("nan")))

        # ---- per walker (fixed-identity replica): tail mean T + label residency --
        self.results = []
        for b in range(self.B):
            t_b = T[:, b]
            resid = {float(tk): float(np_.mean(np_.isclose(temps_hist[:, b], tk)))
                     for tk in self.ladder}
            self.results.append(dict(
                replica=b, natoms=int(self.n_b[b]), n_dof=int(self.n_dof[b]),
                T_K=t_b.copy(), KE_Ha=KE[:, b].copy(), PE_Ha=PE[:, b].copy(),
                T_tail_mean=float(np_.mean(t_b[burn:])),
                T_mean=float(np_.mean(t_b)), T_std=float(np_.std(t_b)),
                label_residency=resid))

        # the lowest rung (target) is occupied by exactly one walker every step.
        self.target_slot_steps = int(np_.isclose(temps_hist, self.ladder[0]).any(axis=1).sum())
        self.nstep = nstep
        self._log_summary()

    # ----------------------------------------------------------------- logging
    def _log_parameters_remd(self):
        p = self.params
        L = " / ".join(f"{t:.1f}" for t in self.ladder)
        lines = ["\n" + "=" * 72 + "\n", f"{'TEMPERATURE-REMD PARAMETERS':^72}\n",
                 "=" * 72 + "\n",
                 f"Replicas (B):    {self.B}\n",
                 f"Ladder (K):      {L}\n",
                 f"Calculator:      {type(self.calc).__name__} (one get_ef_gpu/step)\n",
                 f"Thermostat:      {p.thermostat}\n",
                 f"Timestep:        {p.timestep:.3f} fs\n",
                 f"Total steps:     {p.steps}\n",
                 f"Exchange every:  {self.exchange_every} steps"
                 f"{'  (NO swaps)' if self.exchange_every <= 0 else ''}\n"]
        if p.random_seed is not None:
            lines.append(f"Random seed:     {p.random_seed}\n")
        lines.append("=" * 72 + "\n")
        self.log_info(lines)

    def _log_summary(self):
        lines = ["\n" + "=" * 72 + "\n", f"{'REMD SUMMARY':^72}\n", "=" * 72 + "\n",
                 f"  steps={self.nstep}  exchange_every={self.exchange_every}\n",
                 "\n  -- per ladder slot (measured kinetic T, tail) --\n",
                 "  T_target(K)   <T>meas(K)   sig(K)   steps_occ\n"]
        for s in self.slot_results:
            lines.append(f"  {s['T_target']:>10.1f}   {s['T_measured']:>9.2f}   "
                         f"{s['T_std']:>6.2f}   {s['steps_occupied']:>7d}\n")
        lines.append("\n  -- per ladder-adjacent pair (swap acceptance) --\n")
        lines.append("  T_lo->T_hi (K)        attempts  accepts   rate\n")
        for e in self.exchange_stats:
            lines.append(f"  {e['T_lo']:>6.1f} -> {e['T_hi']:>6.1f}        "
                         f"{e['attempts']:>6d}   {e['accepts']:>6d}   {e['rate']:>6.3f}\n")
        lines.append("=" * 72 + "\n")
        self.log_info(lines)
