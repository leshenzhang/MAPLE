"""
Hamiltonian replica-exchange via NODE-ENERGY solute tempering (REST2-style) -- rides C3.

REST2 (Replica-Exchange with Solute Tempering, 2) runs N copies of ONE system at
the SAME reference temperature T_0 but on a lambda-ladder that scales the SOLUTE's
contribution to the potential, so the solute crosses barriers at an effective high
temperature while the solvent stays cold. Classic FF-REST2 scales the per-term
force-field solute-solute / solute-solvent energies -- IMPOSSIBLE for a black-box
MLIP, which exposes no FF energy decomposition (this is exactly what the
temperature-REMD wrapper (``remd.py``) REJECTS with mode='rest2').

NODE-ENERGY PARTITION -- the only defensible scheme for a black-box MLIP.
MACE returns per-atom NODE energies ``E_i`` (``out["node_energy"]``; they sum per
graph to the total energy, so they are a real conservative partition of E_full).
Define the tempered energy of replica m::

    S       = sum_{i in solute} E_i                          (solute node-energy sum)
    E_m     = E_full + (lambda_m - 1) * S                    (tempered SCALAR energy)

``E_m`` is a TRUE conservative scalar of ALL positions, so ``F_m = -grad E_m`` is a
real force (built by ``MaceOffBatchCalc.get_ef_rest_gpu`` via autograd of the
weighted node-energy sum). Because the solute node energies depend on the WHOLE
geometry through message passing, ``F_m`` has nonzero components on SOLVENT atoms
too -- it is NOT the naive "scale the solute forces" scheme (which the FD gate G2
rejects). Marketed as node-energy solute tempering, NOT FF-REST2.

Reuse of the batched NVT kernel (~80% of ``remd.py``): the N replicas ARE the batch
dimension of ``BatchedNVT``. Per-step propagation is the inherited VV/thermostat
kernel; ONLY the force eval is swapped (``_forces_au`` -> ``get_ef_rest_gpu`` with
the CURRENT per-replica lambda) and the swap bookkeeping is interleaved:

  * All replicas share T_0 (single temperature) -- velocities are seeded at T_0 and
    NEVER rescaled on a swap (unlike T-REMD). Replica b's lambda is ladder[b].
  * Every ``exchange_every`` steps, attempt Metropolis swaps on ALTERNATING
    ladder-adjacent (in lambda) pairs. The REST2 exchange criterion (relabel-lambda
    convention; E_full and the constant node_e0 offset CANCEL exactly) is::
        Delta = beta_0 (lambda_i - lambda_j) (S_j - S_i),   beta_0 = 1/(k_B T_0)
        p = min(1, exp(-Delta))
    using the solute node sums ``S`` already returned by that step's forward (NO
    extra force eval). On accept, swap the two replicas' lambda LABELS (same T_0 =>
    no velocity rescale). Symmetric proposal => detailed balance holds (gate G4).

Ceilings inherited from ``BatchedNVT`` unchanged (RATTLE / GaMD / SMD / PLUMED /
Colvars / posres rejected; isolated OR homogeneous-periodic replicas; local /
decoupled batch calc). The backend MUST expose ``get_ef_rest_gpu`` (MACE-OFF does).
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from ase import Atoms

from maple.function.timer import timer
from maple.function.utility import Molecules

from ..utils import KELVIN_TO_HARTREE, HA_PER_ANG_TO_AU
from .nvt_batched import BatchedNVT, BatchedNVTParams


@dataclass
class REST2Params(BatchedNVTParams):
    """Parameters for node-energy REST2 (Hamiltonian replica exchange). Inherits every
    ``BatchedNVTParams`` field (timestep/thermostat/friction/tau_t/hmr/... honored by
    the inherited batched kernel) and adds the lambda-ladder + exchange controls.

    ``temperature`` is the SHARED reference T_0 for ALL replicas (REST2 tempers via
    lambda, not T). ``mode`` is declared so jobABC does not strip it (B-51)."""
    n_replicas:     int   = 4                # N replicas = batch B (>= 2)
    lambda_max:     float = 1.0              # physical rung (production target; lambda=1)
    lambda_min:     float = 0.5              # most-tempered rung (solute least coupled)
    lambda_ladder:  Optional[List[float]] = None   # explicit ladder (overrides min/max/N)
    solute_indices: Optional[List[int]] = None     # solute atom indices in the TEMPLATE
    solute:         Optional[str] = None           # "a:b" range string (alt to indices)
    exchange_every: int   = 100              # steps between swap sweeps (<=0 => never)
    mode:           str   = "rest2"          # rest2 | node-energy | solute-temper
    swap_seed:      Optional[int] = None     # RNG for swap accept draws (default: derived)


class REST2(BatchedNVT):
    """Node-energy solute-tempering Hamiltonian replica-exchange built ON TOP of
    ``BatchedNVT`` (reuses its VV/thermostat/projection kernel; swaps the force eval
    to the tempered node-energy force and adds lambda-ladder Metropolis swaps)."""

    def __init__(self, output: str,
                 system: Union[Molecules, List[Atoms], Atoms],
                 calc=None,
                 paras: Optional[dict] = None):
        # ---- resolve the single template system (REST2 replicates ONE system) ----
        if isinstance(system, Molecules):
            templates = list(system.multiatoms)
            calc = calc if calc is not None else system.calc
        elif isinstance(system, Atoms):
            templates = [system]
        else:
            templates = list(system)
        if not templates:
            raise ValueError("REST2 requires one template system to replicate.")
        if len(templates) > 1:
            raise ValueError(
                "REST2 replicates a SINGLE system across the lambda-ladder; got "
                f"{len(templates)} distinct systems. Pass one Atoms (or a Molecules / "
                "list holding exactly one structure).")
        template = templates[0]
        n_template = len(template)

        # ---- parse REST2 params (need n_replicas BEFORE building the batch) -------
        rp = REST2Params()
        rp = self._update_dataclass_from_dict(
            rp, self._select_subdict(paras, ("rest2", "REST2", "hremd", "hrex",
                                             "replica", "md", "MD")))
        n = int(rp.n_replicas)
        if n < 2:
            raise ValueError(f"REST2 needs n_replicas >= 2 (got {n}); for one replica "
                             "use the single-system NVT or BatchedNVT.")

        # ---- lambda ladder (descending from lambda_max=physical to lambda_min) ----
        if rp.lambda_ladder is not None:
            ladder = np.asarray(rp.lambda_ladder, dtype=float).reshape(-1)
            if ladder.size != n:
                raise ValueError(f"lambda_ladder has {ladder.size} entries but "
                                 f"n_replicas={n}.")
        else:
            if not (rp.lambda_max >= rp.lambda_min > 0.0):
                raise ValueError(f"REST2 ladder needs lambda_max ({rp.lambda_max}) >= "
                                 f"lambda_min ({rp.lambda_min}) > 0.")
            # geometric ladder, DESCENDING (rung 0 = lambda_max = physical/production).
            ladder = rp.lambda_max * (rp.lambda_min / rp.lambda_max) ** (
                np.arange(n) / (n - 1))
        self.lambda_ladder = ladder.astype(float)               # per-rung lambda
        self.lambda_init = ladder.astype(float).copy()          # replica b starts at ladder[b]

        # ---- solute atom indices in the TEMPLATE (same for every replica) ---------
        self._solute_template = self._resolve_solute(rp, n_template)

        # velocities seeded at the shared reference T_0 = temperature (NO ladder rescale).
        rp.anneal = ""

        # ---- build the N-replica batch and run BatchedNVT's setup/gates -----------
        replicas = [template.copy() for _ in range(n)]
        super().__init__(output, replicas, calc=calc, paras=paras)
        # BatchedNVT parsed a BatchedNVTParams; swap in the richer REST2Params.
        self.params = rp
        if self.params.thermostat not in self._THERMOSTAT_CHOICES:
            raise ValueError(f"Unknown thermostat '{self.params.thermostat}'. "
                             f"Choose from: {self._THERMOSTAT_CHOICES}")
        self._reject_ceiling_features()

        # backend MUST expose the node-energy tempered force primitive.
        if not callable(getattr(calc, "get_ef_rest_gpu", None)):
            raise TypeError(
                f"REST2 needs a calculator exposing get_ef_rest_gpu(lambdas, "
                f"solute_mask) (node-energy tempered E_m + force). '{type(calc).__name__}' "
                f"does not. Use MaceOffBatchCalc (MACE-OFF exposes per-atom node energies).")

        self.exchange_every = int(rp.exchange_every)
        self.lambdas = self.lambda_init.copy()   # current per-replica lambda (mutated on swap)
        # dedicated swap-accept RNG (separate from per-replica thermostat streams so
        # swaps never perturb the NVT noise -> no-swap / all-lambda=1 == BatchedNVT).
        ss = rp.swap_seed
        if ss is None and rp.random_seed is not None:
            ss = int(rp.random_seed) + 2_000_003
        self._swap_rng = np.random.default_rng(ss)
        self._n_attempt = np.zeros(self.B - 1, dtype=np.int64)   # per ladder boundary r<->r+1
        self._n_accept = np.zeros(self.B - 1, dtype=np.int64)
        self._swap_round = 0
        self._last_S = np.zeros(self.B, dtype=float)             # solute node sums (Ha), cur config
        self._hist_lambdas: List[np.ndarray] = []

    # ----------------------------------------------------------- solute resolution
    @staticmethod
    def _resolve_solute(rp, n_template):
        """Resolve the solute atom index set (in the template) from params. Priority:
        explicit ``solute_indices`` -> ``solute`` "a:b" range string -> ALL atoms
        (whole-system tempering; documented degenerate case)."""
        if rp.solute_indices is not None:
            idx = np.asarray(rp.solute_indices, dtype=int).reshape(-1)
        elif rp.solute is not None and str(rp.solute).strip():
            s = str(rp.solute).strip()
            if ":" in s:
                a, b = s.split(":")
                idx = np.arange(int(a or 0), int(b if b else n_template), dtype=int)
            else:
                idx = np.asarray([int(t) for t in s.replace(",", " ").split()], dtype=int)
        else:
            idx = np.arange(n_template, dtype=int)               # default: whole system
        if idx.size == 0:
            raise ValueError("REST2 solute selection is empty.")
        if idx.min() < 0 or idx.max() >= n_template:
            raise ValueError(f"REST2 solute indices out of range [0,{n_template}).")
        return np.unique(idx)

    # ----------------------------------------------- ladder -> buffers (post-prepare)
    def _apply_lambda_ladder(self):
        """Pin ALL thermostats to the shared reference T_0, cache beta_0, and build the
        GLOBAL solute mask over the concatenated (replica-major) atom ordering. Called
        AFTER ``_prepare_buffers`` (which seeded v at T_0 for every replica)."""
        T0 = float(self.params.temperature)
        if T0 <= 0.0:
            raise ValueError(f"REST2 reference temperature must be > 0 (got {T0}).")
        self.beta0 = 1.0 / (KELVIN_TO_HARTREE * T0)              # 1/Hartree
        for b in range(self.B):
            self._thermostats[b].set_temperature(T0)            # all replicas share T_0
        self.lambdas = self.lambda_init.copy()
        # global solute mask: replica b occupies concatenated atoms [ptr[b], ptr[b+1]).
        ptr = np.concatenate(([0], np.cumsum(self.n_b))).astype(int)
        N_atoms = int(ptr[-1])
        mask = np.zeros(N_atoms, dtype=bool)
        for b in range(self.B):
            mask[ptr[b] + self._solute_template] = True
        self._solute_mask_global = mask

    # ------------------------------------------------- tempered force (override seam)
    def _forces_au(self):
        """REST2 force eval: ONE batched forward -> (E_m (B,) Ha, F_m (B,nmax_dof) a.u.)
        via the node-energy tempered scalar with the CURRENT per-replica lambda. Caches
        the per-replica solute node-energy sums ``S`` (Ha) for the swap criterion. This
        REPLACES BatchedNVT._forces_au (get_ef_gpu); no bias hook (REST2 rejects the
        C3 ceiling features)."""
        E_Ha, F_Ha, S_Ha = self.calc.get_ef_rest_gpu(self.lambdas, self._solute_mask_global)
        F = F_Ha.to(self.device, self.dtype) * HA_PER_ANG_TO_AU
        E = E_Ha.to(self.device, self.dtype).reshape(-1)
        self._last_S = S_Ha.detach().to("cpu").numpy().reshape(-1).astype(float)   # (B,) Ha
        return E, F

    # ============================================================= exchange criterion
    @staticmethod
    def _rest2_delta(lam_a, lam_b, S_a, S_b, beta0):
        """REST2 relabel-lambda exchange exponent for neighbors (i=a, j=b):
            Delta = beta_0 (lambda_a - lambda_b) (S_b - S_a).
        E_full and the constant node_e0 offset cancel EXACTLY; only the config-dependent
        solute node-energy sums remain. lambda_a == lambda_b => Delta == 0 (gate G3)."""
        return beta0 * (lam_a - lam_b) * (S_b - S_a)

    def _attempt_swaps(self):
        """Attempt Metropolis swaps on alternating lambda-adjacent pairs using the solute
        node sums ``self._last_S`` from this step's forward (NO extra force eval).
        RELABEL convention: on accept swap the two replicas' lambda LABELS. All replicas
        share T_0 => velocities are UNCHANGED on a swap (no sqrt(T'/T) rescale)."""
        S = self._last_S
        lam = self.lambdas
        order = np.argsort(lam, kind="stable")                  # replicas by ascending lambda
        parity = self._swap_round % 2                           # 0 even, 1 odd sweep
        self._swap_round += 1
        for r in range(parity, self.B - 1, 2):
            a = int(order[r]); b = int(order[r + 1])            # lambda neighbors (lam_a <= lam_b)
            lam_a = float(lam[a]); lam_b = float(lam[b])
            delta = self._rest2_delta(lam_a, lam_b, float(S[a]), float(S[b]), self.beta0)
            self._n_attempt[r] += 1
            p = 1.0 if delta <= 0.0 else float(np.exp(-delta))
            if self._swap_rng.random() < p:
                self._n_accept[r] += 1
                lam[a], lam[b] = lam_b, lam_a                   # relabel lambdas (same T_0 => no v rescale)

    # ================================================================== run loop
    def _run_rest2(self):
        """N-replica NVT at shared T_0 via the inherited batched kernel (tempered force),
        with swap sweeps every ``exchange_every`` steps. exchange_every<=0 => never swap
        => B independent tempered-NVT replicas (each frozen at its ladder lambda)."""
        langevin = (self.params.thermostat != "v-rescale")
        v = self.v
        E, F = self._forces_au()                                # also caches self._last_S
        if langevin:                                            # std -> carried at t=0
            v = v - 0.5 * F / self.mass * self.dt_au
        ex = self.exchange_every
        le = max(1, int(self.params.log_every or 1))
        self._hist_lambdas = []
        for step in range(1, self.params.steps + 1):
            if langevin:
                v, E, F = self._step_langevin(v, F, step)
            else:
                v, E, F = self._step_vrescale(v, F, step)
            if (step % le == 0) or (step == self.params.steps):
                self._hist_lambdas.append(self.lambdas.copy())  # label DURING this recorded step
            if ex and ex > 0 and step % ex == 0 and step < self.params.steps:
                self._attempt_swaps()
        self.v = v

    # ====================================================================== run
    def run(self):
        with timer(f"REST2 (lambda-ladder, N={self.B})"):
            self._log_parameters_rest2()
            self._prepare_buffers()
            self._apply_lambda_ladder()
            self._run_rest2()
            self._finalize_rest2()
        return self

    # --------------------------------------------------------------- finalize
    def _finalize_rest2(self):
        np_ = np
        torch = self._torch
        if not self._hist_T:
            self.results, self.exchange_stats, self.slot_results = [], [], []
            return
        T = torch.stack(self._hist_T, 0).detach().to("cpu").numpy()      # (nstep, B)
        KE = torch.stack(self._hist_KE, 0).detach().to("cpu").numpy()
        PE = torch.stack(self._hist_PE, 0).detach().to("cpu").numpy()
        lam_hist = np_.asarray(self._hist_lambdas)                       # (nstep, B)
        nstep = T.shape[0]
        burn = nstep // 2

        # ---- per lambda-slot: measured kinetic T (all share T_0) + occupancy --------
        self.slot_results = []
        for lk in self.lambda_ladder:
            mask = np_.isclose(lam_hist, lk)
            occ = int(mask.any(axis=1).sum())
            tail = mask.copy(); tail[:burn] = False
            vals = T[tail]
            self.slot_results.append(dict(
                lam_target=float(lk),
                T_measured=float(np_.mean(vals)) if vals.size else float("nan"),
                T_std=float(np_.std(vals)) if vals.size else float("nan"),
                steps_occupied=occ))

        # ---- per lambda-adjacent pair: swap acceptance ------------------------------
        lam_sorted = np_.sort(self.lambda_ladder)
        self.exchange_stats = []
        for r in range(self.B - 1):
            att = int(self._n_attempt[r]); acc = int(self._n_accept[r])
            self.exchange_stats.append(dict(
                pair_rank=(r, r + 1),
                lam_lo=float(lam_sorted[r]), lam_hi=float(lam_sorted[r + 1]),
                attempts=att, accepts=acc,
                rate=(acc / att) if att else float("nan")))

        # ---- per walker (fixed identity): lambda residency (flat histogram / G4) -----
        self.results = []
        for b in range(self.B):
            resid = {float(lk): float(np_.mean(np_.isclose(lam_hist[:, b], lk)))
                     for lk in self.lambda_ladder}
            self.results.append(dict(
                replica=b, natoms=int(self.n_b[b]), n_dof=int(self.n_dof[b]),
                T_K=T[:, b].copy(), KE_Ha=KE[:, b].copy(), PE_Ha=PE[:, b].copy(),
                lambda_residency=resid))

        # occupancy flatness of a TAGGED walker across the ladder (uniform => G4 pass).
        M = self.B
        res0 = np_.array([self.results[0]["lambda_residency"][float(lk)]
                          for lk in self.lambda_ladder])
        self.tag0_residency = res0
        self.tag0_flatness = float(np_.max(np_.abs(res0 - 1.0 / M)))
        # physical rung (lambda_max) occupied by exactly one walker every step.
        self.target_slot_steps = int(np_.isclose(lam_hist, self.lambda_ladder.max()).any(axis=1).sum())
        self.nstep = nstep
        self._log_summary()

    # ----------------------------------------------------------------- logging
    def _log_parameters_rest2(self):
        p = self.params
        L = " / ".join(f"{x:.4f}" for x in self.lambda_ladder)
        sol = self._solute_template
        lines = ["\n" + "=" * 72 + "\n",
                 f"{'NODE-ENERGY REST2 (Hamiltonian REX) PARAMETERS':^72}\n",
                 "=" * 72 + "\n",
                 f"Replicas (B):    {self.B}\n",
                 f"Lambda ladder:   {L}\n",
                 f"Reference T_0:   {p.temperature:.1f} K (shared by ALL replicas)\n",
                 f"Solute atoms:    {sol.size} of {self.n_b[0] if hasattr(self,'n_b') else '?'} "
                 f"(indices {sol[:8]}{'...' if sol.size > 8 else ''})\n",
                 f"Calculator:      {type(self.calc).__name__} (get_ef_rest_gpu/step)\n",
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
        lines = ["\n" + "=" * 72 + "\n", f"{'REST2 SUMMARY':^72}\n", "=" * 72 + "\n",
                 f"  steps={self.nstep}  exchange_every={self.exchange_every}  "
                 f"T_0={self.params.temperature:.1f} K\n",
                 "\n  -- per lambda slot (measured kinetic T, tail) --\n",
                 "  lambda      <T>meas(K)   sig(K)   steps_occ\n"]
        for s in self.slot_results:
            lines.append(f"  {s['lam_target']:>8.4f}   {s['T_measured']:>9.2f}   "
                         f"{s['T_std']:>6.2f}   {s['steps_occupied']:>7d}\n")
        lines.append("\n  -- per lambda-adjacent pair (swap acceptance) --\n")
        lines.append("  lam_lo->lam_hi        attempts  accepts   rate\n")
        for e in self.exchange_stats:
            lines.append(f"  {e['lam_lo']:>6.4f} -> {e['lam_hi']:>6.4f}      "
                         f"{e['attempts']:>6d}   {e['accepts']:>6d}   {e['rate']:>6.3f}\n")
        lines.append(f"\n  tag-0 lambda occupancy = {np.round(self.tag0_residency, 3)} "
                     f"(uniform=1/{self.B}={1.0/self.B:.3f}); flatness dev={self.tag0_flatness:.3f}\n")
        lines.append("=" * 72 + "\n")
        self.log_info(lines)
