"""
Transition Path Sampling (TPS) via aimless shooting -- rides C3.

TPS samples the ensemble of *reactive* trajectories connecting two stable states
A and B (defined by an order parameter lambda(x)) WITHOUT any bias potential, so
it never distorts the dynamics or the mechanism. This module implements the
Bolhuis-Dellago-Chandler two-way shooting move in the Peters-Trout AIMLESS
SHOOTING flavour (Peters & Trout, J. Chem. Phys. 125, 054108 (2006)):

  * SHOOTING MOVE: pick a shooting point x0 from the current transition path,
    draw FRESH Maxwell-Boltzmann velocities v (optionally a small MB-preserving
    perturbation), and integrate a segment FORWARD (x0, +v) and BACKWARD
    (x0, -v) in time. Backward-in-time == forward integration of the SAME
    time-reversible Velocity-Verlet map with NEGATED velocities. The reversed
    backward half spliced to the forward half is the new candidate path.
  * ACCEPT/REJECT: the move is accepted iff the new path is REACTIVE (one end
    commits to A, the other to B). For aimless shooting with fresh MB velocities
    the shooting-point selection weight and the velocity proposal are symmetric,
    so the Metropolis-Hastings acceptance on the path ensemble reduces to
    ``accept iff reactive`` (Peters & Trout Sec. II).
  * BATCH = N PARALLEL SHOOTING TRIALS: the N shots fired from a shooting point
    (each its own MB velocity draw) ARE the batch dimension B of ``BatchedNVT``.
    One ``calc.get_ef_gpu()`` per step advances all N candidate half-paths, so a
    TPS iteration costs the SAME number of forwards as a single NVT segment. The
    forward halves of those N shots simultaneously give the COMMITTOR estimate at
    the shooting point ``pB(x0) = #(shots reaching B) / #(shots that committed)``.

PROPAGATION reuses ``BatchedNVT`` verbatim (its VV kernel + per-replica force
buffer) -- the integrator is NOT reimplemented. TPS segments are MICROCANONICAL
(NVE): the inherited LF-Middle Langevin path is driven with ``friction=0`` (the
OU update ``v' = c1 v + c2 xi`` collapses to the identity: ``c1=exp(0)=1``,
``c2=sqrt((1-1)kT/m)=0``) and runtime COM/angular projection OFF, so a segment is
pure time-reversible Velocity-Verlet -- exactly what the two-way shooting +
committor need. State detection uses FIRST-HITTING of the A/B thresholds along
the segment (well-defined under NVE without dissipation).

Deliverables (filled by ``run()``): the ensemble of accepted reactive paths
(``self.paths``), the committor at every shooting point (``self.committor_shooting``),
and the transition-state ensemble (``self.ts_ensemble`` = shooting configs with
committor ~ 0.5). ``self.acceptance_rate`` is the reactive fraction.

Self-contained: A/B are defined ONLY by this module's ``OrderParameter`` (a
distance / dihedral / user callable + thresholds); it does NOT depend on any
saddle/TS/Hessian module elsewhere in the tree.

ponytail: this is TPS AIMLESS SHOOTING, not full TIS/RETIS (no interface
ensembles / crossing histograms) and not the exact effective-positive-flux RATE
(Bolhuis-Dellago-Chandler-Geissler reactive flux) -- ``self.acceptance_rate``
reports only the reactive fraction. Upgrade path: add interfaces + WHAM-style
flux factor for k_AB, or one-way shooting for stiff systems. All ``BatchedNVT``
ceilings (RATTLE / GaMD / SMD / PLUMED / Colvars / posres; charge-coupled batch
calcs) are inherited unchanged; TPS itself adds no bias.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from ase import Atoms

from maple.function.timer import timer
from maple.function.utility import Molecules

from ..utils import initialize_velocities
from .nvt_batched import BatchedNVT, BatchedNVTParams


# =============================================================================
# Order parameter lambda(x) + A/B basin definition (pure numpy; no torch/GPU)
# =============================================================================
class OrderParameter:
    """User-supplied order parameter lambda(x) and the two stable-state basins.

    ``kind``:
      * ``"distance"`` -- lambda = | COM(group1) - COM(group2) | [Angstrom].
        ``groups`` = ``[g1, g2]`` (each a list of 0-based atom indices; a
        single-atom group => an interatomic distance). Or pass ``indices="i,j"``
        for the two single-atom shorthand.
      * ``"dihedral"`` -- lambda = dihedral(i, j, k, l) [degrees in (-180, 180]
        by default, or radians if ``degrees=False``]. ``indices="i,j,k,l"``.
      * ``"custom"``   -- lambda = ``func(positions_ang)`` for a user callable
        (positions: ``(n_atoms, 3)`` numpy in Angstrom) -> float. This is the
        general "user-supplied lambda(x)" entry point.

    Basin convention (Peters-Trout): A is the LOW-lambda well (``lambda <= a_max``),
    B is the HIGH-lambda well (``lambda >= b_min``), with ``a_max < b_min`` so
    ``(a_max, b_min)`` is the transition region. ``basin(lam)`` -> ``"A"`` / ``"B"``
    / ``None``.
    """

    def __init__(self,
                 kind: str = "distance",
                 indices: Optional[Union[str, Sequence[int]]] = None,
                 groups: Optional[Sequence[Sequence[int]]] = None,
                 a_max: Optional[float] = None,
                 b_min: Optional[float] = None,
                 func: Optional[Callable[[np.ndarray], float]] = None,
                 masses: Optional[Sequence[float]] = None,
                 degrees: bool = True):
        self.kind = str(kind).strip().lower()
        self.a_max = None if a_max is None else float(a_max)
        self.b_min = None if b_min is None else float(b_min)
        self.func = func
        self.degrees = bool(degrees)
        self.masses = None if masses is None else np.asarray(masses, dtype=float)

        if self.kind == "custom":
            if not callable(func):
                raise ValueError("OrderParameter(kind='custom') needs func=callable.")
        elif self.kind == "distance":
            self.groups = self._resolve_groups(indices, groups)
        elif self.kind == "dihedral":
            idx = self._resolve_indices(indices, groups, need=4)
            self.idx = idx
        else:
            raise ValueError(f"Unknown OrderParameter kind '{kind}' "
                             "(use 'distance', 'dihedral', or 'custom').")

        if self.a_max is not None and self.b_min is not None \
                and not (self.a_max < self.b_min):
            raise ValueError(
                f"OrderParameter needs a_max ({self.a_max}) < b_min ({self.b_min}) "
                "(A is the low-lambda basin, B the high-lambda basin).")

    # ------------------------------------------------------------ index parsing
    @staticmethod
    def _parse_int_list(spec) -> List[int]:
        if spec is None:
            return []
        if isinstance(spec, str):
            return [int(t) for t in spec.replace(",", " ").split() if t.strip()]
        return [int(t) for t in spec]

    def _resolve_groups(self, indices, groups) -> List[np.ndarray]:
        if groups is not None:
            g = [np.asarray(self._parse_int_list(x), dtype=int) for x in groups]
        else:
            idx = self._parse_int_list(indices)
            if len(idx) != 2:
                raise ValueError("distance OrderParameter needs two atom indices "
                                 "(indices='i,j') or groups=[g1, g2].")
            g = [np.array([idx[0]], dtype=int), np.array([idx[1]], dtype=int)]
        if len(g) != 2 or any(gi.size == 0 for gi in g):
            raise ValueError("distance OrderParameter needs two non-empty groups.")
        if set(g[0].tolist()) & set(g[1].tolist()):
            raise ValueError("distance OrderParameter groups must not overlap.")
        return g

    def _resolve_indices(self, indices, groups, need) -> np.ndarray:
        idx = self._parse_int_list(indices)
        if not idx and groups is not None:
            idx = [int(np.asarray(g).reshape(-1)[0]) for g in groups]
        if len(idx) != need:
            raise ValueError(f"{self.kind} OrderParameter needs {need} atom indices "
                             f"(indices='{','.join(str(i) for i in range(need))}').")
        return np.asarray(idx, dtype=int)

    # --------------------------------------------------------------- evaluation
    def _com(self, positions: np.ndarray, g: np.ndarray) -> np.ndarray:
        pos = positions[g]
        if self.masses is not None:
            w = self.masses[g]
            return (w[:, None] * pos).sum(0) / w.sum()
        return pos.mean(0)

    def value(self, positions: np.ndarray) -> float:
        """lambda(x) for one configuration ``positions`` (n_atoms, 3) [Angstrom]."""
        p = np.asarray(positions, dtype=float)
        if self.kind == "custom":
            return float(self.func(p))
        if self.kind == "distance":
            return float(np.linalg.norm(self._com(p, self.groups[0])
                                        - self._com(p, self.groups[1])))
        # dihedral
        i, j, k, l = self.idx
        b0 = p[i] - p[j]
        b1 = p[k] - p[j]
        b2 = p[l] - p[k]
        n1 = np.linalg.norm(b1)
        if n1 < 1e-12:
            return 0.0
        b1n = b1 / n1
        v = b0 - np.dot(b0, b1n) * b1n
        w = b2 - np.dot(b2, b1n) * b1n
        x = np.dot(v, w)
        y = np.dot(np.cross(b1n, v), w)
        ang = np.arctan2(y, x)
        return float(np.degrees(ang)) if self.degrees else float(ang)

    def basin(self, lam: float) -> Optional[str]:
        """Classify a lambda value into basin ``"A"`` / ``"B"`` / ``None``."""
        if self.a_max is None or self.b_min is None:
            raise ValueError("OrderParameter has no a_max/b_min thresholds set.")
        if lam <= self.a_max:
            return "A"
        if lam >= self.b_min:
            return "B"
        return None

    def basin_of(self, positions: np.ndarray) -> Optional[str]:
        return self.basin(self.value(positions))


def committor_fraction(first_basins: Sequence[Optional[str]]) -> float:
    """Committor estimate pB = #(committed to B) / #(committed to A or B).

    ``first_basins`` is the per-shot first-hit basin ("A"/"B"/None). Uncommitted
    shots (None) are excluded from the denominator. Returns NaN if none committed.
    Pure numpy -- shared by ``TPS.committor`` and the numpy toy gate."""
    nB = sum(1 for s in first_basins if s == "B")
    nA = sum(1 for s in first_basins if s == "A")
    n = nA + nB
    return (nB / n) if n else float("nan")


# =============================================================================
# TPS parameters
# =============================================================================
@dataclass
class TPSParams(BatchedNVTParams):
    """Aimless-shooting TPS params = batched-NVT params PLUS the shooting block.

    NVE is enforced in ``TPS.__init__`` (``friction=0``, ``remove_com_every=0``,
    ``remove_angular_every=0``, ``thermostat='langevin'``, ``anneal=''``) so
    segments are time-reversible Velocity-Verlet; the inherited thermostat/anneal
    fields are therefore effectively fixed."""
    n_shots:        int   = 8        # parallel shooting trials per iteration = batch B
    segment_steps:  int   = 200      # MD steps per half (forward AND backward)
    n_iterations:   int   = 20       # aimless-shooting MC iterations
    op_every:       int   = 10       # steps between order-parameter / basin checks
    shoot_dt:       int   = 5        # +/- steps for the aimless x0 +/- dt neighbours
    vel_perturb:    float = 0.0      # 0 => fresh MB; (0,1) => MB-preserving blend
    shoot_seed:     Optional[int] = None   # RNG for velocity draws + chain moves
    ts_committor_tol: float = 0.1    # |pB - 0.5| <= tol => TS-ensemble member
    # --- order parameter (used only when no OrderParameter object is passed) ---
    op_kind:        str   = "distance"     # distance | dihedral | custom(needs obj)
    op_indices:     str   = ""             # "i,j" (distance) / "i,j,k,l" (dihedral)
    op_group1:      str   = ""             # distance COM group 1 (overrides op_indices)
    op_group2:      str   = ""             # distance COM group 2
    state_a_max:    float = 0.0            # lambda <= a_max => basin A
    state_b_min:    float = 0.0            # lambda >= b_min => basin B
    op_degrees:     bool  = True           # dihedral in degrees


class TPS(BatchedNVT):
    """Transition Path Sampling (aimless shooting) built ON TOP of ``BatchedNVT``.

    Reuses the batched VV/force kernel for propagation (batch axis = N shooting
    trials); adds only the shooting move + A/B state detection + accept/reject +
    committor. Segments are NVE (friction=0)."""

    _ALIASES = ("tps", "TPS", "shooting", "aimless", "md", "MD", "nvt", "NVT",
                "batched", "batchnvt")

    def __init__(self, output: str,
                 system: Union[Molecules, List[Atoms], Atoms],
                 calc=None,
                 paras: Optional[dict] = None,
                 order_parameter: Optional[OrderParameter] = None,
                 initial_path: Optional[List[Atoms]] = None):
        # 1) parse TPS config first (need n_shots before replicating the template).
        cfg = self._init_params(TPSParams, paras, self._ALIASES)
        n = int(cfg.n_shots)
        if n < 1:
            raise ValueError("TPS needs n_shots >= 1 (batch = parallel shooting trials).")
        if int(cfg.segment_steps) < 1:
            raise ValueError("TPS needs segment_steps >= 1.")

        # 2) resolve the template topology + the initial path frames (shooting-point
        #    source). Accept: initial_path=[frames]; OR system=list (len 2 = A,B
        #    endpoints -> linear-interpolated frames; len>2 = path frames); OR a
        #    single Atoms (single-frame path, assumed near the barrier).
        template, calc, frames = self._resolve_frames(system, calc, initial_path)
        self._path_frames = frames                          # list of (n,3) arrays
        self._n_atoms_tmpl = len(template)

        # 3) build the N-shot batch (calc prepared on B = n_shots copies of template).
        replicas = [template.copy() for _ in range(n)]
        super().__init__(output, replicas, calc=calc, paras=paras)
        self.params = cfg                                   # richer TPSParams (superset)

        # 4) ENFORCE NVE segments (time-reversible VV; committor first-hitting well-
        #    defined). friction=0 => OU thermostat is the identity; projections OFF.
        self.params.thermostat = "langevin"
        self.params.friction = 0.0
        self.params.remove_com_every = 0
        self.params.remove_angular_every = 0
        self.params.anneal = ""
        self._reject_ceiling_features()                     # inherited ceilings

        # 5) order parameter (user object wins; else build from params).
        self.op = order_parameter if order_parameter is not None \
            else self._build_op(self.params)

        # 6) RNGs: per-shot velocity streams (independent => batch-isolated draws)
        #    + a dedicated chain-move RNG (never perturbs the shooting streams).
        seed = self.params.shoot_seed
        if seed is None:
            seed = self.params.random_seed
        self._shoot_rngs = [np.random.default_rng((seed + b) if seed is not None else None)
                            for b in range(n)]
        self._chain_rng = np.random.default_rng(
            (seed + 999_983) if seed is not None else None)

        self.n_iterations = int(self.params.n_iterations)
        self._dt_frames = max(1, int(self.params.shoot_dt))
        self._prepared_tps = False
        # initial shooting point = path frame nearest the A/B separatrix.
        self._cur_shoot = self._initial_shoot_point()

        # deliverables (filled by run()).
        self.paths: List[dict] = []
        self.all_shots: List[dict] = []
        self.committor_shooting: List[dict] = []
        self.ts_ensemble: List[dict] = []
        self.acceptance_rate = float("nan")
        self.rate_estimate = None      # ponytail: reactive-flux rate not implemented

    # ------------------------------------------------------------- construction
    @staticmethod
    def _resolve_frames(system, calc, initial_path):
        """Return (template Atoms, calc, list-of-(n,3)-frame-arrays)."""
        if isinstance(system, Molecules):
            mols = list(system.multiatoms)
            calc = calc if calc is not None else system.calc
            template = mols[0]
            sys_frames = mols
        elif isinstance(system, Atoms):
            template = system
            sys_frames = [system]
        else:
            sys_frames = list(system)
            if not sys_frames:
                raise ValueError("TPS: empty systems list.")
            template = sys_frames[0]

        if initial_path is not None:
            src = list(initial_path)
            frames = [np.asarray(a.get_positions(), dtype=float) for a in src]
        elif len(sys_frames) == 2:
            # two endpoints A, B -> linear-interpolated initial guess path.
            pA = np.asarray(sys_frames[0].get_positions(), dtype=float)
            pB = np.asarray(sys_frames[1].get_positions(), dtype=float)
            frames = [pA + (pB - pA) * t for t in np.linspace(0.0, 1.0, 9)]
        else:
            frames = [np.asarray(a.get_positions(), dtype=float) for a in sys_frames]
        return template, calc, frames

    @staticmethod
    def _build_op(p: TPSParams) -> OrderParameter:
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

    def _initial_shoot_point(self) -> np.ndarray:
        """Pick the path frame whose lambda is nearest the A/B separatrix midpoint
        (best transition-region guess); fall back to the middle frame."""
        if (self.op.a_max is not None) and (self.op.b_min is not None):
            mid = 0.5 * (self.op.a_max + self.op.b_min)
            lams = np.array([self.op.value(fr) for fr in self._path_frames])
            return self._path_frames[int(np.argmin(np.abs(lams - mid)))].copy()
        return self._path_frames[len(self._path_frames) // 2].copy()

    # ------------------------------------------------------------------- setup
    def _ensure_prepared(self):
        if not self._prepared_tps:
            self._prepare_buffers()          # prepares calc(B) + friction=0 thermostats
            self._assert_nve()
            self._prepared_tps = True

    def _assert_nve(self):
        p = self.params
        if float(p.friction) != 0.0 or int(p.remove_com_every) != 0 \
                or int(p.remove_angular_every) != 0 or p.thermostat != "langevin":
            raise RuntimeError(
                "TPS segments must be NVE (friction=0, remove_com_every=0, "
                "remove_angular_every=0, thermostat='langevin') for time-reversible "
                "two-way shooting; params were mutated away from that.")

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
        """Fresh Maxwell-Boltzmann velocities (COM removed) for shot b. With
        ``vel_perturb`` in (0,1) blend two independent MB draws -- the blend
        ``sqrt(1-p^2) v1 + p v2`` is still MB at the target T (Gaussian sum), so
        the proposal stays symmetric (accept-iff-reactive holds)."""
        at, T, ndof = self.atoms_list[b], self.params.temperature, int(self.n_dof[b])
        v1 = initialize_velocities(atoms=at, temperature=T, remove_com=True,
                                   remove_angular=False, target_n_dof=ndof,
                                   rng=self._shoot_rngs[b])
        p = float(self.params.vel_perturb)
        if p > 0.0:
            v2 = initialize_velocities(atoms=at, temperature=T, remove_com=True,
                                       remove_angular=False, target_n_dof=ndof,
                                       rng=self._shoot_rngs[b])
            v1 = np.sqrt(max(0.0, 1.0 - p * p)) * v1 + p * v2
        return v1

    # ---------------------------------------------------------- NVE propagation
    def _propagate(self, n_steps: int, detect: bool = True, capture_at: int = 0):
        """Advance the current batch NVE for ``n_steps`` (reuses BatchedNVT's VV
        step). ``self.v`` in, STANDARD velocities; calc.coord = start config.

        Returns ``(v_std_final, first_basin_list, captured)``:
          * ``first_basin_list`` -- per-shot FIRST-HIT basin (only if ``detect``).
          * ``captured`` -- per-shot positions snapshot at step ``capture_at`` (the
            aimless +/- dt neighbour), or None.
        Segment history is reset each call so buffers never grow across shots."""
        torch = self._torch
        self._hist_T, self._hist_KE, self._hist_PE = [], [], []
        v = self.v.clone()
        E, F = self._forces_au()
        v = v - 0.5 * F / self.mass * self.dt_au             # standard -> LF carried
        first: List[Optional[str]] = [None] * self.B
        captured = None
        oe = max(1, int(self.params.op_every))
        for step in range(1, n_steps + 1):
            v, E, F = self._step_langevin(v, F, step)        # inherited VV (gamma=0)
            if capture_at and step == capture_at:
                captured = self._positions_per_replica()
            if detect and (step % oe == 0 or step == n_steps):
                self._commit_check(first)
                if all(b is not None for b in first):
                    break
        v_std = v + 0.5 * F / self.mass * self.dt_au         # LF carried -> standard
        self.v = v_std
        return v_std, first, captured

    def _commit_check(self, first: List[Optional[str]]):
        lams = self._lambdas_now()
        for b in range(self.B):
            if first[b] is None:
                s = self.op.basin(lams[b])
                if s is not None:
                    first[b] = s

    # --------------------------------------------------------------- committor
    def committor(self, config, v_init: Optional[Sequence[np.ndarray]] = None):
        """Committor pB at a single configuration via ``self.B`` FORWARD shots.

        Fires B trajectories from ``config`` with fresh MB velocities (batch axis =
        shots), each integrated ``segment_steps`` under NVE, and returns
        ``(pB, first_basins)`` where pB = fraction of committed shots reaching B
        (first-hitting). ``config``: ase.Atoms or (n_atoms,3) array. ``v_init``:
        optional per-shot velocities (else drawn). pB ~ 0.5 => transition state."""
        self._ensure_prepared()
        pos = config.get_positions() if hasattr(config, "get_positions") \
            else np.asarray(config, dtype=float)
        self._set_coords([pos] * self.B)
        vlist = v_init if v_init is not None \
            else [self._draw_shoot_vel(b) for b in range(self.B)]
        self._load_v_std(vlist)
        _, first, _ = self._propagate(self.params.segment_steps, detect=True)
        return committor_fraction(first), first

    # ------------------------------------------------------- two-way shooting
    def _shoot_iteration(self, x0: np.ndarray):
        """One aimless-shooting iteration: B shots from shooting point ``x0``, each
        two-way (forward +v / backward -v). Returns (records, pB(x0), n_committed).

        pB(x0) is the committor from the FORWARD halves (fresh MB draws). A shot is
        REACTIVE iff its forward and backward halves commit to the two DIFFERENT
        basins (path connects A<->B)."""
        B = self.B
        vlist = [self._draw_shoot_vel(b) for b in range(B)]

        # forward half: (x0, +v)
        self._set_coords([x0] * B)
        self._load_v_std(vlist)
        _, fwd_first, x_plus = self._propagate(self.params.segment_steps,
                                               detect=True, capture_at=self._dt_frames)
        # backward half: (x0, -v)  (backward-in-time == negated velocities)
        self._set_coords([x0] * B)
        self._load_v_std([-v for v in vlist])
        _, bwd_first, x_minus = self._propagate(self.params.segment_steps,
                                                detect=True, capture_at=self._dt_frames)

        lam0 = float(self.op.value(x0))
        recs = []
        nA = nB = 0
        for b in range(B):
            fb, bb = fwd_first[b], bwd_first[b]
            reactive = (fb is not None and bb is not None and {fb, bb} == {"A", "B"})
            if fb == "B":
                nB += 1
            elif fb == "A":
                nA += 1
            recs.append(dict(
                shot=b, fwd_basin=fb, bwd_basin=bb, reactive=bool(reactive),
                lam0=lam0,
                x0=np.asarray(x0, dtype=float).copy(),
                x_plus=(x_plus[b] if x_plus is not None else None),
                x_minus=(x_minus[b] if x_minus is not None else None)))
        ncommit = nA + nB
        pB = (nB / ncommit) if ncommit else float("nan")
        return recs, pB, ncommit

    # ====================================================================== run
    def run(self):
        with timer(f"TPS aimless shooting (shots={self.params.n_shots}, "
                   f"iters={self.n_iterations})"):
            self._log_parameters_tps()
            self._ensure_prepared()
            x0 = self._cur_shoot
            n_accept = n_trials = 0
            for it in range(self.n_iterations):
                recs, pB, ncommit = self._shoot_iteration(x0)
                self.all_shots.extend(recs)
                n_trials += self.B
                reactive = [r for r in recs if r["reactive"]]
                n_accept += len(reactive)
                self.paths.extend(reactive)
                self.committor_shooting.append(dict(
                    iteration=it, lam=float(self.op.value(x0)), pB=pB,
                    n_committed=int(ncommit), n_reactive=len(reactive),
                    x0=np.asarray(x0, dtype=float).copy()))
                # advance the MC chain (aimless x0 +/- dt from an accepted path);
                # reject => keep x0 (standard Metropolis stay).
                if reactive:
                    pick = reactive[int(self._chain_rng.integers(len(reactive)))]
                    cands = [x0] + [c for c in (pick["x_plus"], pick["x_minus"])
                                    if c is not None]
                    x0 = np.asarray(cands[int(self._chain_rng.integers(len(cands)))],
                                    dtype=float).copy()
            self._cur_shoot = x0
            self.acceptance_rate = (n_accept / n_trials) if n_trials else float("nan")
            self.n_accept, self.n_trials = n_accept, n_trials
            self._harvest_ts_ensemble()
            self._finalize_tps()
        return self

    def _harvest_ts_ensemble(self):
        """TS ensemble = shooting configs with committor ~ 0.5 (within
        ``ts_committor_tol``) and a majority of shots committed."""
        tol = float(self.params.ts_committor_tol)
        need = max(1, self.B // 2)
        self.ts_ensemble = [
            dict(iteration=r["iteration"], lam=r["lam"], pB=r["pB"],
                 n_committed=r["n_committed"], x0=r["x0"])
            for r in self.committor_shooting
            if (r["n_committed"] >= need and not np.isnan(r["pB"])
                and abs(r["pB"] - 0.5) <= tol)]

    # --------------------------------------------------------- time-reversal API
    def time_reversal_residual(self, n_steps: int,
                               v0_list: Optional[Sequence[np.ndarray]] = None,
                               x0_pos_list: Optional[Sequence[np.ndarray]] = None):
        """Integrate NVE forward ``n_steps`` from (x0, v0), negate velocities, and
        integrate ``n_steps`` more; return ``(max|dpos|, max|dvel|)`` vs the start.
        For time-reversible Velocity-Verlet both are ~ fp64 rounding (validates the
        backward = negated-velocity leg of two-way shooting). Gate #1."""
        self._ensure_prepared()
        if x0_pos_list is not None:
            self._set_coords(x0_pos_list)
        x0 = self.calc.coord.detach().to("cpu").numpy().copy()      # (N_atoms,3)
        if v0_list is None:
            v0_list = [self._draw_shoot_vel(b) for b in range(self.B)]
        v0_list = [np.asarray(v, dtype=float).copy() for v in v0_list]

        self._load_v_std(v0_list)
        v_std_n, _, _ = self._propagate(n_steps, detect=False)
        vn = [self._v_real(v_std_n, b) for b in range(self.B)]

        self._load_v_std([-x for x in vn])                          # backward
        v_back, _, _ = self._propagate(n_steps, detect=False)
        xback = self.calc.coord.detach().to("cpu").numpy()
        vb = [self._v_real(v_back, b) for b in range(self.B)]

        dpos = float(np.max(np.abs(xback - x0)))
        dvel = float(max(np.max(np.abs(vb[b] - (-v0_list[b]))) for b in range(self.B)))
        return dpos, dvel

    # ----------------------------------------------------------------- logging
    def _log_parameters_tps(self):
        p = self.params
        op = self.op
        thr = (f"A: lambda<={op.a_max}   B: lambda>={op.b_min}"
               if (op.a_max is not None and op.b_min is not None) else "(unset)")
        lines = ["\n" + "=" * 72 + "\n",
                 f"{'TRANSITION PATH SAMPLING (aimless shooting) PARAMETERS':^72}\n",
                 "=" * 72 + "\n",
                 f"Shots (batch B): {self.B}\n",
                 f"Calculator:      {type(self.calc).__name__} (one get_ef_gpu/step)\n",
                 f"Order param:     kind={op.kind}   {thr}\n",
                 f"Segment steps:   {p.segment_steps} (forward AND backward) @ "
                 f"dt={p.timestep} fs\n",
                 f"Iterations:      {p.n_iterations}\n",
                 f"Ensemble:        NVE (friction=0, projections off; VV time-reversible)\n",
                 f"Temperature:     {p.temperature:.2f} K (Maxwell-Boltzmann draws)\n",
                 f"op_every:        {p.op_every} steps   shoot_dt: {p.shoot_dt}\n"]
        if p.shoot_seed is not None or p.random_seed is not None:
            lines.append(f"Seed:            "
                         f"{p.shoot_seed if p.shoot_seed is not None else p.random_seed}\n")
        lines.append("=" * 72 + "\n")
        self.log_info(lines)

    def _finalize_tps(self):
        lines = ["\n" + "=" * 72 + "\n", f"{'TPS AIMLESS SHOOTING SUMMARY':^72}\n",
                 "=" * 72 + "\n",
                 f"  iterations={self.n_iterations}  shots/iter={self.B}  "
                 f"trials={self.n_trials}\n",
                 f"  accepted (reactive) paths: {self.n_accept}  "
                 f"acceptance/reactive fraction: {self.acceptance_rate:.3f}\n",
                 f"  TS-ensemble members (|pB-0.5|<={self.params.ts_committor_tol}): "
                 f"{len(self.ts_ensemble)}\n",
                 "\n  iter   lambda(x0)   pB(x0)   committed   reactive\n"]
        for r in self.committor_shooting:
            pB = r["pB"]
            lines.append(f"  {r['iteration']:>4}   {r['lam']:>9.4f}   "
                         f"{(f'{pB:.3f}' if not np.isnan(pB) else '  nan '):>6}   "
                         f"{r['n_committed']:>9}   {r['n_reactive']:>8}\n")
        lines.append("=" * 72 + "\n")
        self.log_info(lines)
