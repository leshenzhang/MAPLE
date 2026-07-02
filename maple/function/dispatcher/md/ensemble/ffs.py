"""
Forward Flux Sampling (FFS, direct) -- interface-based rates, rides C3.

FFS (Allen, Warren & ten Wolde, Phys. Rev. Lett. 94, 018104 (2005); Allen,
Frenkel & ten Wolde, J. Chem. Phys. 124, 194111 (2006)) computes the rate
constant k_AB of an IRREVERSIBLE / HIGH-BARRIER rare event A -> B by ratcheting
the system across a sequence of non-intersecting interfaces of an order parameter
lambda(x):

    lambda_A = lambda_0 < lambda_1 < ... < lambda_n = lambda_B

  1. FLUX Phi_A : run UNBIASED MD in basin A and count the effective POSITIVE
     crossings of the first interface lambda_0 per unit time (a crossing counts
     only after the trajectory has returned to A -- the effective-positive-flux
     rule of van Erp). Each crossing configuration (+ its velocity) is stored: it
     seeds the first interface. Phi_A = N_0 / T.
  2. INTERFACE PROBABILITIES : at interface i, fire N TRIAL SHOTS (the BATCH axis)
     from the stored lambda_i configs. Each shot runs the natural dynamics until it
     either reaches lambda_{i+1} (SUCCESS -> its config is stored for interface i+1)
     or falls back to basin A, lambda <= lambda_A (FAIL). The conditional crossing
     probability is P(lambda_{i+1}|lambda_i) = successes / N.
  3. RATE : k_AB = Phi_A * prod_i P(lambda_{i+1}|lambda_i). Because the flux is
     high (lambda_0 sits just outside A) and the product is tiny, FFS reaches rates
     far below brute-force MD reach. The interface placement cancels in the product
     (the FFS invariance), so k_AB is independent of where the interfaces sit.

BATCH = the N shots at ONE interface: ``BatchedNVT`` advances all N candidate
trajectories with ONE ``calc.get_ef_gpu()`` per step (shots == the batch axis B).
An entire interface's transition probability is estimated in a single forward
series -- the MAPLE advantage. PROPAGATION reuses ``BatchedNVT`` verbatim (its
LF-Middle Langevin VV kernel + per-replica force buffer); the integrator is NOT
reimplemented. lambda(x) + basin detection reuse ``tps.OrderParameter`` unchanged.

Dynamics: FFS is defined for STOCHASTIC dynamics; the shots are Langevin NVT by
default (``friction`` > 0). Setting ``friction=0`` gives NVE shots whose only
randomness is the fresh Maxwell-Boltzmann velocity draw (``redraw_velocities``,
the TPS aimless-shooting flavour). By default a shot CONTINUES the dynamics from
the stored phase point (positions + velocities), which is the rigorous direct-FFS
estimator.

Deliverables (filled by ``run()``): ``self.flux`` (Phi_A, 1/fs), ``self.P_interfaces``
(the per-interface probabilities), ``self.k_AB`` (1/fs), ``self.mfpt_fs`` = 1/k_AB
(directly comparable to ``WeightedEnsemble.mfpt_fs`` and a brute-force first-passage
time on the SAME toy), and the per-interface config pools.

Self-contained: interfaces / basins are defined ONLY by this module's interface
list + ``OrderParameter`` thresholds; no dependence on any saddle/TS module.

ponytail: this is DIRECT FFS (fire shots straight from the previous interface).
It is NOT branched-growth FFS (BG-FFS) and NOT the Rosenbluth variant (no per-shot
branching weights), and NOT TIS/RETIS (no full path ensembles / crossing
histograms). Interfaces are FIXED (user list or a linspace) -- no adaptive interface
placement. Crossings are detected on the ``op_every`` grid (fine grid -> exact).
All ``BatchedNVT`` ceilings (RATTLE / GaMD / SMD / PLUMED / Colvars / posres;
charge-coupled batch calcs; isolated non-periodic replicas) are inherited unchanged;
FFS adds no bias.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
from ase import Atoms

from maple.function.timer import timer
from maple.function.utility import Molecules

from ..utils import initialize_velocities
from .nvt_batched import BatchedNVT, BatchedNVTParams
from .tps import OrderParameter


# =============================================================================
# Pure-numpy effective-positive-flux crossing counter (no torch/GPU).
# =============================================================================
def count_interface_crossings(lam: Sequence[float],
                              lambda0: float,
                              lambda_A: float):
    """Effective POSITIVE crossings of interface ``lambda0`` along a lambda time
    series, gated by a return to basin A (``lambda <= lambda_A``).

    A crossing is counted the first time ``lambda`` rises through ``lambda0``
    (``prev < lambda0 <= cur``) while ARMED; counting then disarms until the
    trajectory returns to A (``cur <= lambda_A``), which re-arms it. This is the
    effective-positive-flux rule (Allen-Frenkel-ten Wolde / van Erp): recrossings
    that never fall back to A are counted once. For ``lambda_A == lambda0`` (the
    default, first interface == basin-A boundary) re-arming is a dip back to the
    boundary; for ``lambda_A < lambda0`` the gap is a hysteresis band.

    Returns ``(n_crossings, crossing_indices)`` where each index is the series
    position at which a crossing was counted. Pure numpy -- shared as the canonical
    definition by ``ForwardFluxSampling._run_flux`` (mirrored online there) and the
    numpy flux gate."""
    lam = np.asarray(lam, dtype=float).reshape(-1)
    if lam.size < 2:
        return 0, []
    n = 0
    idxs: List[int] = []
    armed = bool(lam[0] <= lambda_A)          # armed only if starting inside A
    prev = float(lam[0])
    for i in range(1, lam.size):
        cur = float(lam[i])
        if armed and (prev < lambda0 <= cur):
            n += 1
            idxs.append(i)
            armed = False
        if cur <= lambda_A:
            armed = True
        prev = cur
    return n, idxs


def ffs_rate(flux: float, probabilities: Sequence[float]) -> float:
    """k_AB = Phi_A * prod_i P(lambda_{i+1}|lambda_i). Pure numpy (shared by the
    driver + the numpy rate gate). Any probability == 0 => k_AB == 0."""
    k = float(flux)
    for p in probabilities:
        k *= float(p)
    return k


# =============================================================================
# FFS parameters
# =============================================================================
@dataclass
class FFSParams(BatchedNVTParams):
    """Forward-Flux-Sampling params = batched-NVT params PLUS the FFS controls.

    Inherits every ``BatchedNVTParams`` field. ``thermostat`` is forced to
    ``'langevin'`` in ``ForwardFluxSampling.__init__`` (the OU path; ``friction``
    is free -- ``friction>0`` = NVT shots, ``friction=0`` = NVE shots)."""
    n_shots:        int   = 16       # parallel trial shots per interface = batch B
    # --- interfaces lambda_0 < ... < lambda_n (lambda_0 = basin-A boundary) ------
    interfaces:     Optional[list] = None   # explicit [lam_0, ..., lam_n]; overrides auto
    n_interfaces:   int   = 7        # #interfaces when auto (linspace a_max..b_min)
    # --- flux (basin-A exploration) ---------------------------------------------
    flux_steps:     int   = 3000     # MD steps of basin-A run (per flux walker)
    # --- interface shots --------------------------------------------------------
    shot_max_steps: int   = 3000     # cap per shot (undetermined at cap => FAIL)
    op_every:       int   = 5        # steps between lambda / crossing checks
    redraw_velocities: bool = False  # False: continue stored v (rigorous direct FFS)
                                     # True : fresh Maxwell-Boltzmann per shot (TPS flavour)
    ffs_seed:       Optional[int] = None    # RNG for velocity draws + config-pool sampling
    # --- order parameter (used only when no OrderParameter object is passed) -----
    op_kind:        str   = "distance"      # distance | dihedral | custom(needs obj)
    op_indices:     str   = ""              # "i,j" (distance) / "i,j,k,l" (dihedral)
    op_group1:      str   = ""              # distance COM group 1 (overrides op_indices)
    op_group2:      str   = ""              # distance COM group 2
    state_a_max:    float = 0.0             # lambda_A = lambda_0 (basin A boundary)
    state_b_min:    float = 0.0             # lambda_B = lambda_n (basin B boundary)
    op_degrees:     bool  = True            # dihedral in degrees


class ForwardFluxSampling(BatchedNVT):
    """Direct Forward Flux Sampling built ON TOP of ``BatchedNVT``.

    Reuses the batched VV/force kernel for shot propagation (batch axis = N trial
    shots per interface) and ``tps.OrderParameter`` for lambda(x) + basin
    detection; adds only the flux measurement, the interface ratchet, and the rate
    product k_AB = Phi_A * prod P(lambda_{i+1}|lambda_i)."""

    _ALIASES = ("ffs", "FFS", "forward_flux", "forwardflux", "md", "MD",
                "nvt", "NVT", "batched", "batchnvt")

    def __init__(self, output: str,
                 system: Union[Molecules, List[Atoms], Atoms],
                 calc=None,
                 paras: Optional[dict] = None,
                 order_parameter: Optional[OrderParameter] = None):
        # 1) parse FFS config first (need n_shots before replicating the template).
        cfg = self._init_params(FFSParams, paras, self._ALIASES)
        n = int(cfg.n_shots)
        if n < 1:
            raise ValueError("FFS needs n_shots >= 1 (batch = parallel trial shots).")
        if int(cfg.flux_steps) < 1 or int(cfg.shot_max_steps) < 1:
            raise ValueError("FFS needs flux_steps >= 1 and shot_max_steps >= 1.")

        # 2) resolve template + basin-A start config (the flux run seeds).
        template, calc, basinA = self._resolve_system(system, calc)
        self._basinA_config = basinA
        self._n_atoms_tmpl = len(template)

        # 3) build the N-shot batch (calc prepared on B = n_shots copies of template).
        replicas = [template.copy() for _ in range(n)]
        super().__init__(output, replicas, calc=calc, paras=paras)
        self.params = cfg                              # richer FFSParams (superset)

        # 4) FFS shots use the OU (Langevin) path; friction is free (NVT or NVE).
        self.params.thermostat = "langevin"
        self.params.anneal = ""
        self._reject_ceiling_features()                # inherited ceilings

        # 5) order parameter (user object wins; else build from params).
        self.op = order_parameter if order_parameter is not None \
            else self._build_op(self.params)
        if self.op.a_max is None or self.op.b_min is None:
            raise ValueError("FFS OrderParameter needs a_max (=lambda_0) and "
                             "b_min (=lambda_n) thresholds set.")

        # 6) interfaces lambda_0 < ... < lambda_n (lambda_0 = a_max, lambda_n = b_min).
        self._interfaces = self._build_interfaces(self.params, self.op)
        self._n_transitions = len(self._interfaces) - 1

        # 7) RNGs: per-shot velocity streams (independent => batch-isolated draws)
        #    + a dedicated pool-sampling RNG (never perturbs the shooting streams).
        seed = self.params.ffs_seed
        if seed is None:
            seed = self.params.random_seed
        self._shoot_rngs = [np.random.default_rng((seed + b) if seed is not None else None)
                            for b in range(n)]
        self._pool_rng = np.random.default_rng(
            (seed + 777_733) if seed is not None else None)

        self._prepared_ffs = False

        # deliverables (filled by run()).
        self.flux = float("nan")               # Phi_A (1/fs)
        self.n_crossings = 0
        self.flux_time_fs = 0.0
        self.P_interfaces: List[float] = []
        self.n_success: List[int] = []
        self.pools: List[list] = []            # per-interface config pools [(pos, v)]
        self.k_AB = float("nan")               # 1/fs
        self.rate_per_fs = float("nan")
        self.mfpt_fs = float("inf")

    # ------------------------------------------------------------- construction
    @staticmethod
    def _resolve_system(system, calc):
        """Return (template Atoms, calc, basin-A positions (n,3)). The basin-A
        config is the first frame (single Atoms / first of a list / first
        multiatom); the flux run replicates it across the batch."""
        if isinstance(system, Molecules):
            mols = list(system.multiatoms)
            calc = calc if calc is not None else system.calc
            template = mols[0]
        elif isinstance(system, Atoms):
            template = system
        else:
            frames = list(system)
            if not frames:
                raise ValueError("FFS: empty systems list.")
            template = frames[0]
        return template, calc, np.asarray(template.get_positions(), dtype=float)

    @staticmethod
    def _build_op(p: FFSParams) -> OrderParameter:
        if p.op_kind == "custom":
            raise ValueError("op_kind='custom' requires passing order_parameter=... "
                             "(a callable cannot come from the params dict).")
        if str(p.op_group1).strip() and str(p.op_group2).strip():
            return OrderParameter(kind="distance", groups=[p.op_group1, p.op_group2],
                                  a_max=p.state_a_max, b_min=p.state_b_min,
                                  degrees=p.op_degrees)
        return OrderParameter(kind=p.op_kind, indices=(p.op_indices or None),
                              a_max=p.state_a_max, b_min=p.state_b_min,
                              degrees=p.op_degrees)

    @staticmethod
    def _build_interfaces(p: FFSParams, op: OrderParameter) -> np.ndarray:
        """lambda_0 < ... < lambda_n. Explicit ``interfaces`` list wins; else a
        linspace of ``n_interfaces`` points from a_max (lambda_0) to b_min
        (lambda_n). Endpoints are pinned to the basin boundaries so lambda_0 is the
        basin-A boundary (the effective-positive-flux reference)."""
        if p.interfaces is not None:
            lam = np.asarray(p.interfaces, dtype=float).reshape(-1)
            if lam.size < 2 or np.any(np.diff(lam) <= 0):
                raise ValueError("FFS interfaces must be strictly increasing, "
                                 "length >= 2 (lambda_0 < ... < lambda_n).")
            return lam
        ni = int(p.n_interfaces)
        if ni < 2:
            raise ValueError("FFS needs n_interfaces >= 2 (or an explicit list).")
        if not (op.a_max < op.b_min):
            raise ValueError("FFS auto-interfaces need a_max < b_min.")
        return np.linspace(float(op.a_max), float(op.b_min), ni)

    # ------------------------------------------------------------------- setup
    def _ensure_prepared(self):
        if not self._prepared_ffs:
            self._prepare_buffers()          # prepares calc(B), thermostats, rngs, v
            self._prepared_ffs = True

    # -------------------------------------------------- per-replica buffer <-> np
    def _ptr(self) -> np.ndarray:
        return np.concatenate(([0], np.cumsum(self.n_b))).astype(int)

    def _positions_per_replica(self) -> List[np.ndarray]:
        cc = self.calc.coord.detach().to("cpu").numpy()
        ptr = self._ptr()
        return [cc[ptr[b]:ptr[b + 1]].copy() for b in range(self.B)]

    def _set_coords(self, pos_list: Sequence[np.ndarray]):
        arr = np.concatenate([np.asarray(p, dtype=float) for p in pos_list], axis=0)
        self.calc.set_coords_(self._torch.tensor(arr, dtype=self.dtype,
                                                 device=self.device))

    def _load_v_std(self, v_list: Sequence[np.ndarray]):
        v = self._torch.zeros((self.B, self.nmax_dof), dtype=self.dtype,
                              device=self.device)
        for b in range(self.B):
            self._set_v_real(v, b, np.asarray(v_list[b], dtype=float))
        self.v = v

    def _lambdas_now(self) -> np.ndarray:
        return np.array([self.op.value(p) for p in self._positions_per_replica()])

    def _draw_shoot_vel(self, b: int) -> np.ndarray:
        """Fresh Maxwell-Boltzmann velocities (COM removed) for shot b, from shot
        b's INDEPENDENT stream (=> perturbing one shot's draw leaves the others
        untouched -- batch isolation)."""
        at, T, ndof = self.atoms_list[b], self.params.temperature, int(self.n_dof[b])
        return initialize_velocities(atoms=at, temperature=T, remove_com=True,
                                     remove_angular=False, target_n_dof=ndof,
                                     rng=self._shoot_rngs[b])

    # ---------------------------------------------------------------- propagation
    def _carry_from_std(self, v_std, F):
        return v_std - 0.5 * F / self.mass * self.dt_au     # std -> LF-Middle carried

    def _std_from_carry(self, v_carried, F):
        return v_carried + 0.5 * F / self.mass * self.dt_au  # LF carried -> std

    # ----------------------------------------------------------------- FLUX Phi_A
    def _run_flux(self):
        """Basin-A exploration: B walkers, one per batch slot, all started in A with
        fresh MB velocities; propagate ``flux_steps`` Langevin steps. Count the
        effective positive crossings of lambda_0 (mirrors ``count_interface_crossings``
        online) and capture the phase point (positions + std velocity) at each
        crossing. Returns (Phi_A [1/fs], pool [(pos, v)], n_cross, total_time_fs)."""
        lam0 = float(self._interfaces[0])
        lamA = float(self.op.a_max)
        oe = max(1, int(self.params.op_every))
        n_steps = int(self.params.flux_steps)

        self._set_coords([self._basinA_config] * self.B)
        self._load_v_std([self._draw_shoot_vel(b) for b in range(self.B)])
        self._hist_T, self._hist_KE, self._hist_PE = [], [], []
        v = self.v.clone()
        E, F = self._forces_au()
        v = self._carry_from_std(v, F)

        prev = self._lambdas_now()
        armed = (prev <= lamA)                             # (B,) bool
        n_cross = 0
        pool: List[tuple] = []
        for step in range(1, n_steps + 1):
            v, E, F = self._step_langevin(v, F, step)
            if step % oe == 0 or step == n_steps:
                lams = self._lambdas_now()
                v_std = self._std_from_carry(v, F)
                pos_now = None
                for b in range(self.B):
                    if armed[b] and (prev[b] < lam0 <= lams[b]):
                        n_cross += 1
                        if pos_now is None:
                            pos_now = self._positions_per_replica()
                        pool.append((pos_now[b].copy(), self._v_real(v_std, b).copy()))
                        armed[b] = False
                    if lams[b] <= lamA:
                        armed[b] = True
                prev = lams
        total_time_fs = float(self.B * n_steps * self.params.timestep)
        phi_A = (n_cross / total_time_fs) if total_time_fs > 0 else 0.0
        return phi_A, pool, n_cross, total_time_fs

    # -------------------------------------------------- INTERFACE transition P(i)
    def _propagate_shots(self, starts, vels, lam_next, max_steps):
        """Fire B shots from ``starts`` (list of B (n,3)) toward ``lam_next``. Each
        shot runs until lambda >= lam_next (SUCCESS -> capture phase point) or
        lambda <= lambda_A (FAIL); undetermined at ``max_steps`` counts as FAIL.
        ``vels`` = per-shot stored std velocities (used when redraw_velocities is
        False); else fresh MB is drawn. Returns (outcome ['S'|'F'], captured
        [(pos, v) | None])."""
        lamA = float(self.op.a_max)
        oe = max(1, int(self.params.op_every))
        self._set_coords(starts)
        if self.params.redraw_velocities:
            vlist = [self._draw_shoot_vel(b) for b in range(self.B)]
        else:
            vlist = [np.asarray(vels[b], dtype=float) for b in range(self.B)]
        self._load_v_std(vlist)
        self._hist_T, self._hist_KE, self._hist_PE = [], [], []
        v = self.v.clone()
        E, F = self._forces_au()
        v = self._carry_from_std(v, F)

        outcome: List[Optional[str]] = [None] * self.B
        captured: List[Optional[tuple]] = [None] * self.B
        for step in range(1, max_steps + 1):
            v, E, F = self._step_langevin(v, F, step)
            if step % oe == 0 or step == max_steps:
                lams = self._lambdas_now()
                v_std = None
                pos_now = None
                for b in range(self.B):
                    if outcome[b] is not None:
                        continue
                    if lams[b] >= lam_next:
                        outcome[b] = "S"
                        if pos_now is None:
                            pos_now = self._positions_per_replica()
                            v_std = self._std_from_carry(v, F)
                        captured[b] = (pos_now[b].copy(), self._v_real(v_std, b).copy())
                    elif lams[b] <= lamA:
                        outcome[b] = "F"
                if all(o is not None for o in outcome):
                    break
        outcome = [o if o is not None else "F" for o in outcome]   # undetermined -> FAIL
        return outcome, captured

    def _run_interface(self, i: int, pool_in):
        """One interface transition i -> i+1. Sample B start configs from ``pool_in``
        (with replacement), fire the shots, return (P, pool_out, n_success)."""
        if not pool_in:
            return 0.0, [], 0
        idx = self._pool_rng.integers(len(pool_in), size=self.B)
        starts = [pool_in[k][0] for k in idx]
        vels = [pool_in[k][1] for k in idx]
        lam_next = float(self._interfaces[i + 1])
        outcome, captured = self._propagate_shots(starts, vels, lam_next,
                                                  int(self.params.shot_max_steps))
        ns = sum(1 for o in outcome if o == "S")
        P = ns / self.B
        pool_out = [captured[b] for b in range(self.B) if outcome[b] == "S"]
        return P, pool_out, ns

    # ====================================================================== run
    def run(self):
        with timer(f"Forward Flux Sampling (shots={self.params.n_shots}, "
                   f"interfaces={len(self._interfaces)})"):
            self._log_parameters_ffs()
            self._ensure_prepared()

            # Phase 1 -- flux Phi_A + seed pool at lambda_0.
            phi_A, pool0, n_cross, T = self._run_flux()
            self.flux = phi_A
            self.n_crossings = int(n_cross)
            self.flux_time_fs = float(T)
            self.pools = [pool0]

            # Phase 2 -- interface transition probabilities (batched shots).
            self.P_interfaces, self.n_success = [], []
            for i in range(self._n_transitions):
                P, pool_next, ns = self._run_interface(i, self.pools[-1])
                self.P_interfaces.append(P)
                self.n_success.append(int(ns))
                self.pools.append(pool_next)
                if not pool_next:            # dead end -> all downstream P = 0
                    for _ in range(i + 1, self._n_transitions):
                        self.P_interfaces.append(0.0)
                        self.n_success.append(0)
                        self.pools.append([])
                    break

            # Phase 3 -- rate product k_AB = Phi_A * prod P.
            self.k_AB = ffs_rate(self.flux, self.P_interfaces)
            self.rate_per_fs = self.k_AB
            self.mfpt_fs = (1.0 / self.k_AB) if self.k_AB > 0 else float("inf")
            self._finalize_ffs()
        return self

    # ----------------------------------------------------------------- logging
    def _log_parameters_ffs(self):
        p, op = self.params, self.op
        lam = " ".join(f"{x:.4f}" for x in self._interfaces)
        dyn = "NVE (friction=0)" if float(p.friction) == 0.0 else \
            f"NVT (Langevin, friction={p.friction} 1/fs)"
        lines = ["\n" + "=" * 72 + "\n",
                 f"{'FORWARD FLUX SAMPLING (direct FFS) PARAMETERS':^72}\n",
                 "=" * 72 + "\n",
                 f"Shots (batch B): {self.B}\n",
                 f"Calculator:      {type(self.calc).__name__} (one get_ef_gpu/step)\n",
                 f"Order param:     kind={op.kind}   lambda_A={op.a_max}  lambda_B={op.b_min}\n",
                 f"Interfaces (n+1={len(self._interfaces)}): {lam}\n",
                 f"Dynamics:        {dyn} @ dt={p.timestep} fs\n",
                 f"Flux steps:      {p.flux_steps} (basin-A run, per walker)\n",
                 f"Shot max steps:  {p.shot_max_steps}   op_every: {p.op_every}\n",
                 f"Velocities:      {'fresh MB per shot' if p.redraw_velocities else 'continue stored phase point'}\n",
                 f"Temperature:     {p.temperature:.2f} K\n"]
        if p.ffs_seed is not None or p.random_seed is not None:
            lines.append(f"Seed:            "
                         f"{p.ffs_seed if p.ffs_seed is not None else p.random_seed}\n")
        lines.append("=" * 72 + "\n")
        self.log_info(lines)

    def _finalize_ffs(self):
        lines = ["\n" + "=" * 72 + "\n", f"{'FORWARD FLUX SAMPLING SUMMARY':^72}\n",
                 "=" * 72 + "\n",
                 f"  flux Phi_A = {self.flux:.6e} /fs  "
                 f"({self.n_crossings} crossings of lambda_0={self._interfaces[0]:.4f} "
                 f"over {self.flux_time_fs:.1f} fs)\n",
                 "\n  i   lambda_i -> lambda_i+1     P(i+1|i)   successes/B\n"]
        for i, P in enumerate(self.P_interfaces):
            lines.append(f"  {i:>2}   {self._interfaces[i]:>8.4f} -> "
                         f"{self._interfaces[i + 1]:>8.4f}     {P:>7.4f}   "
                         f"{self.n_success[i]:>3}/{self.B}\n")
        prod = 1.0
        for P in self.P_interfaces:
            prod *= P
        lines.append(f"\n  prod P = {prod:.6e}\n")
        lines.append(f"  k_AB = Phi_A * prod P = {self.k_AB:.6e} /fs\n")
        mfpt = self.mfpt_fs
        lines.append(f"  MFPT = 1/k_AB = {mfpt:.6e} fs"
                     f"{'' if not np.isfinite(mfpt) else f' ({mfpt / 1000.0:.4f} ps)'}\n")
        lines.append("=" * 72 + "\n")
        self.log_info(lines)
