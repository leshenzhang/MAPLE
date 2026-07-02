"""
Weighted Ensemble (WE) rare-event sampling -- rides C3 (BatchedNVT).

WE (Huber & Kim, Biophys. J. 70, 97 (1996); Zuckerman & Chong, Annu. Rev.
Biophys. 46, 43 (2017)) runs N *weighted* walkers (Sum w = 1) of UNBIASED MD and
periodically resamples them by a progress coordinate xi(x): occupied bins are
split (clone a walker, halve its weight) / merged (combine low-weight walkers,
survivor picked proportional to weight, weights summed) toward a target count
``m`` per bin. Dynamics are NEVER biased -- only walker bookkeeping changes, and
weight is conserved EXACTLY -- so the weighted ensemble average of any observable
is unbiased, while walkers are steered by population control into the rare region.

  * Propagation of ALL N walkers for ``tau_steps`` is the inherited ``BatchedNVT``
    kernel -- ONE ``calc.get_ef_gpu()`` per step over the padded ``(B=N, nmax_dof)``
    buffer (walkers == the batch axis). The Velocity-Verlet / thermostat physics is
    NOT reimplemented: WE calls the factored ``_step_langevin`` / ``_step_vrescale``
    / ``_step_nhc`` so the resampling is interleaved between byte-identical NVT
    segments (exactly how ``remd.py`` interleaves swaps).
  * Between segments the walkers are binned on xi(x) and resampled. When the
    resample plan is the identity (e.g. 1 bin with ``walkers_per_bin == N``) it is a
    strict NO-OP: v/F/positions/RNG/calc are untouched, so the trajectory is
    BIT-IDENTICAL to a plain ``BatchedNVT`` run of the same total steps + seed (the
    degenerate-reduction gate). Only when split/merge/recycle actually fire is the
    (variable) walker set rebuilt.
  * Rates: with a ``target_bin`` + steady-state recycling, the weight that crosses
    into the target each segment is the flux; MFPT = 1/flux at steady state. Free
    energy comes from the weighted histogram of xi, F(xi) = -kT ln P(xi).

xi interface (constructor ``xi=``): a callable ``xi(positions_np (n,3)) -> float``,
or a dict spec ``{"type": "distance", "atoms": [i, j]}`` /
``{"type": "dihedral", "atoms": [i, j, k, l]}`` (also settable via
``paras['we']['cv_type']`` + ``['cv_atoms']``). B is small so xi is looped on host.

ponytail: DELIBERATE ceilings -- (1) FIXED, static bins only (no adaptive / MAB /
WESTPA Voronoi binning); (2) a SINGLE scalar progress coordinate; (3) split/merge
is WESTPA's split-heaviest / merge-lightest pairwise scheme (not the optimal
allocation); (4) on a resample that actually changes the walker set the calc
topology + buffers are rebuilt (calc.prepare) and, for NHC / v-rescale, the
per-walker thermostat *internal* state is reset (Langevin's OU is stateless, so it
is unaffected; and the degenerate NO-OP path never rebuilds, so bit-identity holds).
All ``BatchedNVT`` ceilings (RATTLE / GaMD / SMD / PLUMED / Colvars / posres,
charge-coupled batch calcs) are inherited unchanged.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from ase import Atoms

from maple.function.timer import timer
from maple.function.utility import Molecules

from ..utils import (
    AMU_TO_AU,
    KELVIN_TO_HARTREE,
    initialize_velocities,
)
from .nvt_batched import BatchedNVT, BatchedNVTParams


HARTREE_TO_KCAL_MOL = 627.5094740631  # Ha -> kcal/mol


# ============================================================ xi (progress CV)
def dihedral_deg(p0, p1, p2, p3) -> float:
    """Signed dihedral angle (degrees) of the four points (Praxeolitic formula)."""
    b0 = np.asarray(p0, float) - np.asarray(p1, float)
    b1 = np.asarray(p2, float) - np.asarray(p1, float)
    b2 = np.asarray(p3, float) - np.asarray(p2, float)
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def make_xi_fn(spec):
    """Resolve a xi definition into a callable ``xi(positions_np (n,3)) -> float``.

    ``spec`` is a callable (returned as-is), or a dict:
      {"type": "distance"|"bond", "atoms": [i, j]}  -> |r_i - r_j| (Angstrom)
      {"type": "dihedral"|"torsion", "atoms": [i, j, k, l]} -> angle (degrees)
      {"type": "coordinate", "atom": i, "axis": 0|1|2} -> that Cartesian component
    """
    if callable(spec):
        return spec
    if isinstance(spec, dict):
        t = str(spec.get("type", "")).strip().lower()
        atoms = spec.get("atoms")
        if t in ("distance", "dist", "bond"):
            i, j = int(atoms[0]), int(atoms[1])
            return lambda pos: float(np.linalg.norm(pos[i] - pos[j]))
        if t in ("dihedral", "torsion", "dih"):
            i, j, k, l = (int(a) for a in atoms[:4])
            return lambda pos: dihedral_deg(pos[i], pos[j], pos[k], pos[l])
        if t in ("coordinate", "coord", "cartesian"):
            i = int(spec.get("atom", atoms[0] if atoms else 0))
            ax = int(spec.get("axis", 0))
            return lambda pos: float(pos[i, ax])
    raise ValueError(
        f"Unrecognized xi spec {spec!r}. Pass a callable(positions)->float, or a "
        "dict {'type':'distance','atoms':[i,j]} / {'type':'dihedral','atoms':[i,j,k,l]}.")


# ============================================================ split / merge core
def we_split_merge(ids, weights, target_m, rng):
    """WESTPA-style split(heaviest) / merge(lightest) toward ``target_m`` walkers.

    Pure-numpy walker bookkeeping (no torch) so it is unit-testable in isolation.
    ``ids`` / ``weights`` are the walkers in ONE bin. Returns a list of
    ``[source_id, weight]``: a repeated ``source_id`` is a SPLIT clone (each half
    the parent's weight); MERGE combines the two lightest walkers into one whose
    ``source_id`` is the survivor (picked with prob proportional to weight) and
    whose weight is the SUM. Weight is conserved exactly:
      * split: w -> w/2 + w/2   (exact in binary floating point);
      * merge: {w1, w2} -> w1 + w2.
    """
    items = [[int(i), float(w)] for i, w in zip(ids, weights)]
    if not items:
        return []
    m = int(target_m)
    if m < 1:
        raise ValueError(f"target_m must be >= 1 (got {m}).")
    # MERGE the two lightest until the bin holds exactly m walkers.
    while len(items) > m:
        items.sort(key=lambda x: x[1])                 # ascending weight
        (i1, w1), (i2, w2) = items[0], items[1]
        W = w1 + w2
        # survivor proportional to weight (Zuckerman & Chong 2017, sec. 3.2)
        surv = i1 if (W <= 0.0 or rng.random() * W < w1) else i2
        items = items[2:] + [[surv, W]]
    # SPLIT the heaviest until the bin holds exactly m walkers.
    while len(items) < m:
        items.sort(key=lambda x: -x[1])                # descending weight
        i0, w0 = items[0]
        items[0] = [i0, 0.5 * w0]
        items.append([i0, 0.5 * w0])
    return items


@dataclass
class WEParams(BatchedNVTParams):
    """Weighted-Ensemble params = the batched-NVT params PLUS the WE controls.

    Inherits every ``BatchedNVTParams`` field (timestep / thermostat / friction /
    tau_t / hmr / remove_com_every ... honored by the inherited batched-NVT
    kernel). ``steps`` is overwritten with ``tau_steps * n_iterations`` at
    construction (WE drives the segments itself)."""
    n_walkers:       int   = 16             # initial walker count = initial batch B
    tau_steps:       int   = 50             # unbiased MD steps per WE iteration
    n_iterations:    int   = 20             # number of WE iterations
    walkers_per_bin: int   = 4              # target walkers m per OCCUPIED bin
    # --- binning (fixed / static) ---
    n_bins:          int   = 1              # used with xi_min/xi_max if no bin_edges
    xi_min:          float = 0.0
    xi_max:          float = 1.0
    bin_edges:       Optional[list] = None  # explicit edges (len n_bins+1); overrides above
    # --- steady-state flux / MFPT ---
    target_bin:      Optional[int] = None   # target-state bin; None => no recycling/flux
    source_bin:      int   = 0              # recycle destination (source basin)
    recycle:         bool  = True           # steady-state recycling (only if target_bin set)
    resample_seed:   Optional[int] = None   # RNG for merge-survivor + clone streams
    # --- xi spec via paras (JSON-friendly alternative to the xi= ctor arg) ---
    cv_type:         str   = ""             # "distance" | "dihedral" | "coordinate"
    cv_atoms:        str   = ""             # e.g. "0,1" or "0,1,2,3"


class WeightedEnsemble(BatchedNVT):
    """Weighted Ensemble rare-event sampling built ON TOP of ``BatchedNVT`` (reuses
    its VV / thermostat / projection kernel; adds only binning + split/merge +
    steady-state recycling)."""

    _ALIASES = ("we", "WE", "weighted", "weighted_ensemble", "md", "MD",
                "nvt", "NVT", "batched")

    def __init__(self, output: str,
                 systems: Union[Molecules, List[Atoms], Atoms],
                 calc=None,
                 xi=None,
                 paras: Optional[dict] = None):
        # ---- parse WE config first (need n_walkers BEFORE building the batch) ---
        wp = self._init_params(WEParams, paras, self._ALIASES)
        n = int(wp.n_walkers)
        if n < 1:
            raise ValueError(f"WE needs n_walkers >= 1 (got {n}).")
        if int(wp.walkers_per_bin) < 1:
            raise ValueError("WE needs walkers_per_bin >= 1.")
        if int(wp.tau_steps) < 1 or int(wp.n_iterations) < 1:
            raise ValueError("WE needs tau_steps >= 1 and n_iterations >= 1.")

        # ---- resolve template + initial walker set (B = n_walkers) -------------
        template, calc = self._resolve_template(systems, calc)
        if isinstance(systems, (list, tuple)) and len(systems) == n:
            walkers = [a.copy() for a in systems]          # pre-built distinct starts
        else:
            walkers = [template.copy() for _ in range(n)]
        n_per = len(template)
        if any(len(w) != n_per for w in walkers):
            raise ValueError("WE requires a HOMOGENEOUS walker topology (same atom "
                             "count/species): binning uses one scalar progress "
                             "coordinate over interchangeable walkers.")

        # ---- build the N-walker batch and run BatchedNVT's setup/gates ---------
        super().__init__(output, walkers, calc=calc, paras=paras)
        # BatchedNVT parsed a BatchedNVTParams; swap in the richer WEParams (superset,
        # so the inherited _prepare_buffers / _step_* read it unchanged).
        self.params = wp
        if self.params.thermostat not in self._THERMOSTAT_CHOICES:
            raise ValueError(f"Unknown thermostat '{self.params.thermostat}'. "
                             f"Choose from: {self._THERMOSTAT_CHOICES}")
        self._reject_ceiling_features()
        # WE drives its own segment loop; steps = the full trajectory length so the
        # inherited _record's "always record the final step" cadence is correct.
        self._tau_steps = int(wp.tau_steps)
        self._n_iterations = int(wp.n_iterations)
        self._m = int(wp.walkers_per_bin)
        self.params.steps = self._tau_steps * self._n_iterations
        self._template = template.copy()
        self._n_per = n_per

        # ---- xi (progress coordinate) ------------------------------------------
        if xi is None:
            ct = str(wp.cv_type or "").strip()
            if not ct:
                raise ValueError(
                    "WE requires a progress coordinate: pass xi=callable / dict, or "
                    "set paras['we']['cv_type'] (+ 'cv_atoms').")
            atoms_idx = [int(a) for a in str(wp.cv_atoms).replace(",", " ").split()]
            xi = {"type": ct, "atoms": atoms_idx}
        self._xi_fn = make_xi_fn(xi)

        # ---- binning (fixed / static) ------------------------------------------
        if wp.bin_edges is not None:
            edges = np.asarray(wp.bin_edges, dtype=float).reshape(-1)
            if edges.size < 2 or np.any(np.diff(edges) <= 0):
                raise ValueError("bin_edges must be strictly increasing, length >= 2.")
        else:
            if int(wp.n_bins) < 1:
                raise ValueError("n_bins must be >= 1.")
            edges = np.linspace(float(wp.xi_min), float(wp.xi_max), int(wp.n_bins) + 1)
        self._bin_edges = edges
        self.n_bins = int(edges.size - 1)

        # ---- flux / recycling controls -----------------------------------------
        self._target_bin = None if wp.target_bin is None else int(wp.target_bin)
        self._source_bin = int(wp.source_bin)
        self._recycle = bool(wp.recycle) and (self._target_bin is not None)
        for nm, val in (("target_bin", self._target_bin), ("source_bin", self._source_bin)):
            if val is not None and not (0 <= val < self.n_bins):
                raise ValueError(f"{nm}={val} out of range [0, {self.n_bins}).")
        self._resample_seed = wp.resample_seed

        # results (filled by run()/_finalize_we)
        self.weights = None
        self.free_energy_kT = None
        self.free_energy_kcal = None
        self.bin_centers = None
        self.flux_total = 0.0
        self.flux_per_iter = None
        self.mfpt_fs = float("inf")
        self.rate_per_fs = 0.0
        self.n_walkers_hist = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _resolve_template(systems, calc):
        if isinstance(systems, Molecules):
            template = systems.multiatoms[0]
            calc = calc if calc is not None else systems.calc
        elif isinstance(systems, (list, tuple)):
            if not systems:
                raise ValueError("WE: empty systems list.")
            template = systems[0]
        else:
            template = systems              # single ase.Atoms template
        return template, calc

    def _walker_positions(self):
        """Per-walker positions (list of (n,3) Angstrom numpy) from the calc master."""
        from ..bias.batched import _ptr_to_np  # sanctioned torch|numpy _ptr coercion
        coord = self.calc.coord.detach().to("cpu").numpy()
        ptr = _ptr_to_np(self.calc._ptr).tolist()
        return [coord[ptr[b]:ptr[b + 1]].copy() for b in range(self.B)]

    def _compute_xi(self):
        """Progress coordinate xi(x) per walker (looped on host; B is small)."""
        pos = self._walker_positions()
        return np.array([self._xi_fn(pos[b]) for b in range(self.B)], dtype=float)

    def _bin_index(self, xi):
        """Static-bin assignment of xi -> [0, n_bins-1] (values outside are clamped)."""
        interior = self._bin_edges[1:-1]
        idx = np.digitize(xi, interior)                 # 0..n_bins-1 (empty edges => 0)
        return np.clip(idx, 0, self.n_bins - 1).astype(int)

    def _draw_mb_velocities(self, positions, rng):
        """Fresh Maxwell-Boltzmann std velocities (n,3) for a recycled walker,
        via the SAME ``initialize_velocities`` the inherited _prepare_buffers uses."""
        at = self._template.copy()
        at.set_positions(positions)
        n = len(at)
        n_dof = max(3 * n - 3, 1) if self.params.remove_com else max(3 * n, 1)
        return initialize_velocities(
            atoms=at, temperature=self.params.temperature,
            remove_com=self.params.remove_com, remove_rotation=False,
            remove_angular=False, target_n_dof=int(n_dof), rng=rng)

    # ============================================================ buffer rebuild
    def _rebuild(self, positions_list, vstd_list, rngs):
        """Rebuild the (variable-B) batched buffers for a NEW walker set, CARRYING
        each walker's velocity + RNG stream (unlike _prepare_buffers, which re-inits
        them). Preserves the accumulated T/KE/PE history (does NOT reset it).

        positions_list : list of (n,3) Angstrom for the new walkers
        vstd_list      : list of (n,3) STANDARD velocities (a.u.) for the new walkers
        rngs           : list of np.random.Generator (parent stream / fresh clone)
        """
        torch = self._torch
        B = len(positions_list)
        # new topology holders (homogeneous -> template clones with the new coords)
        atoms_list = []
        for p in positions_list:
            at = self._template.copy()
            at.set_positions(p)
            atoms_list.append(at)
        self.atoms_list = atoms_list
        self.B = B

        # re-prepare the calc for the new B (topology + master coords) --------
        self.calc.prepare(self.atoms_list, fixed_nmax=None)
        self.nmax_dof = int(self.calc.nmax_dof)
        # explicit master-coord set (prepare already read them; belt-and-suspenders).
        if self.calc.N_atoms > 0:
            flat = np.concatenate(positions_list, axis=0)
            self.calc.set_coords_(torch.tensor(flat, dtype=self.dtype, device=self.device))

        self.n_b = np.array([len(at) for at in self.atoms_list], dtype=int)
        self.n_dof = np.maximum(3 * self.n_b - 3, 1)

        dev, dt = self.device, self.dtype
        mass = torch.ones((B, self.nmax_dof), dtype=dt, device=dev)
        mask = torch.zeros((B, self.nmax_dof), dtype=dt, device=dev)
        for b, at in enumerate(self.atoms_list):
            m_au = torch.tensor(at.get_masses(), dtype=dt, device=dev) * AMU_TO_AU
            n = m_au.shape[0]
            mass[b, :3 * n] = m_au.repeat_interleave(3)
            mask[b, :3 * n] = 1.0
        self.mass = mass
        self.mask = mask

        # carried RNG streams (parent keeps continuity; clones get a fresh stream).
        self._rngs = list(rngs)

        # per-walker AUTHORITATIVE thermostat objects (B's classes; new obj per
        # walker keyed to its carried RNG). ponytail ceiling: NHC / v-rescale
        # internal chain state is reset here -- Langevin's OU is stateless so it is
        # unaffected; the degenerate NO-OP path never calls _rebuild so bit-identity
        # is preserved regardless.
        from ..thermostat.langevin import LangevinThermostat
        from ..thermostat.vrescale import VRescaleThermostat
        from ..thermostat.nose_hoover import NoseHooverChain
        self._thermostats = []
        for b, at in enumerate(self.atoms_list):
            if self.params.thermostat == "v-rescale":
                th = VRescaleThermostat(at, temperature=self.params.temperature,
                                        tau_t=self.params.tau_t,
                                        timestep=self.params.timestep,
                                        rng=self._rngs[b], n_dof=int(self.n_dof[b]))
            elif self.params.thermostat in ("nose-hoover", "nhc"):
                th = NoseHooverChain(at, temperature=self.params.temperature,
                                     tau_t=self.params.tau_t,
                                     timestep=self.params.timestep,
                                     n_dof=int(self.n_dof[b]),
                                     chain_length=self.params.chain_length,
                                     n_respa=self.params.nhc_n_respa,
                                     n_yoshida=self.params.nhc_n_yoshida)
            else:
                th = LangevinThermostat(at, temperature=self.params.temperature,
                                        friction=self.params.friction,
                                        timestep=self.params.timestep,
                                        rng=self._rngs[b])
            self._thermostats.append(th)

        # new STANDARD velocity buffer (B, nmax_dof).
        v = torch.zeros((B, self.nmax_dof), dtype=dt, device=dev)
        for b in range(B):
            self._set_v_real(v, b, np.asarray(vstd_list[b]))
        self.v = v

    # ============================================================= resample step
    def _resample(self, it, step, v, F, use_carried):
        """Bin the walkers on xi(x), (optionally) recycle target-state crossings for
        the steady-state flux, and split/merge each occupied bin toward ``m``
        walkers. Returns ``(changed, v, F)``: on a pure identity plan (NO-OP) v/F
        are returned UNCHANGED and the calc/buffers/RNG are untouched (bit-identity);
        otherwise the walker set is rebuilt and (v, F) recomputed for it."""
        torch = self._torch
        xi = self._compute_xi()                          # (B,)  positions only
        w = self._weights.copy()                         # (B,)
        bins = self._bin_index(xi)                       # (B,)

        # free-energy histogram (weighted, pre-resample ensemble).
        for b in range(self.B):
            self._fe_hist[bins[b]] += w[b]
        self._fe_count += 1

        # --- steady-state recycling: target crossings contribute to the flux and
        #     are relocated back to the source basin (weight preserved). ---------
        recycled = np.zeros(self.B, dtype=bool)
        if self._recycle:
            for b in range(self.B):
                if bins[b] == self._target_bin:
                    self._flux += float(w[b])
                    recycled[b] = True
                    bins[b] = self._source_bin           # now in the source basin

        # --- split/merge plan per occupied bin ----------------------------------
        plan = []                                        # list of [source_id, weight]
        for bidx in range(self.n_bins):
            members = np.nonzero(bins == bidx)[0]
            if members.size == 0:
                continue
            plan.extend(we_split_merge(members.tolist(), w[members].tolist(),
                                       self._m, self._resample_rng))
        src_ids = [p[0] for p in plan]
        new_w = np.array([p[1] for p in plan], dtype=float)

        # GATE 1 -- weight conservation after every resampling step.
        assert abs(float(new_w.sum()) - 1.0) < 1e-12, \
            f"WE weight not conserved: Sum w = {new_w.sum()!r} (iter {it})"

        # identity NO-OP fast path (preserves the degenerate == BatchedNVT bit-identity)
        is_identity = ((not recycled.any())
                       and len(plan) == self.B
                       and src_ids == list(range(self.B))
                       and np.array_equal(new_w, w))
        if is_identity:
            self._weights = new_w
            self.n_walkers_hist.append(self.B)
            return False, v, F

        # --- apply the plan: gather new positions / velocities / RNG streams ----
        v_std = (v + 0.5 * F / self.mass * self.dt_au) if use_carried else v
        old_pos = self._walker_positions()
        old_v = [self._v_real(v_std, b).copy() for b in range(self.B)]
        old_rng = list(self._rngs)
        # relocate recycled walkers to a random source-basin config + fresh MB draw.
        for b in np.nonzero(recycled)[0]:
            cfg = self._source_configs[self._resample_rng.integers(len(self._source_configs))]
            fresh = np.random.default_rng(int(self._resample_rng.integers(1 << 62)))
            old_pos[b] = cfg.copy()
            old_v[b] = self._draw_mb_velocities(cfg, fresh)
            old_rng[b] = fresh

        new_pos, new_v, new_rngs = [], [], []
        used = {}
        for sid in src_ids:
            c = used.get(sid, 0)
            used[sid] = c + 1
            new_pos.append(old_pos[sid].copy())
            new_v.append(old_v[sid].copy())
            if c == 0:
                new_rngs.append(old_rng[sid])            # first copy keeps the stream
            else:                                        # split clone -> fresh stream
                new_rngs.append(np.random.default_rng(int(self._resample_rng.integers(1 << 62))))

        self._rebuild(new_pos, new_v, new_rngs)
        self._weights = new_w
        self.n_walkers_hist.append(self.B)

        # recompute the force for the rebuilt set; re-derive the carried velocity.
        E, F = self._forces_au()
        v = (self.v - 0.5 * F / self.mass * self.dt_au) if use_carried else self.v
        return True, v, F

    # ====================================================================== run
    def run(self):
        with timer(f"Weighted Ensemble (WE, N0={self.B})"):
            self._log_parameters_we()
            self._prepare_buffers()                      # init v (std), rngs seed+b, thermostats
            self._init_we_state()
            self._run_we()
            self._finalize_we()
        return self

    def _init_we_state(self):
        """WE bookkeeping, initialized AFTER _prepare_buffers (calc prepared -> xi
        computable). Equal initial weights; capture the source-basin configs used to
        re-inject recycled walkers."""
        self._weights = np.full(self.B, 1.0 / self.B, dtype=float)
        self._resample_rng = np.random.default_rng(self._resample_seed)
        self._fe_hist = np.zeros(self.n_bins, dtype=float)
        self._fe_count = 0
        self._flux = 0.0
        self._flux_per_iter = []
        self.n_walkers_hist = []
        xi0 = self._compute_xi()
        bins0 = self._bin_index(xi0)
        pos0 = self._walker_positions()
        src = [pos0[b] for b in range(self.B) if bins0[b] == self._source_bin]
        self._source_configs = src if src else [p for p in pos0]

    def _run_we(self):
        """WE loop: propagate ALL walkers ``tau_steps`` via the inherited batched
        kernel, then resample. Mirrors ``remd._run_remd`` (segments of the factored
        NVT step, bookkeeping interleaved) -- but the segment count is fixed
        (``tau_steps``) and the interleaved op is split/merge instead of swaps."""
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
        for it in range(self._n_iterations):
            flux0 = self._flux
            for _ in range(self._tau_steps):
                step += 1
                v, E, F = step_fn(v, F, step)
            _changed, v, F = self._resample(it, step, v, F, use_carried)
            self._flux_per_iter.append(self._flux - flux0)
        self.v = (v + 0.5 * F / self.mass * self.dt_au) if use_carried else v

    # --------------------------------------------------------------- finalize
    def _finalize_we(self):
        self.weights = self._weights.copy()
        self.flux_total = float(self._flux)
        self.flux_per_iter = np.asarray(self._flux_per_iter, dtype=float)
        self.n_walkers_hist = np.asarray(self.n_walkers_hist, dtype=int)

        # free energy F(xi) = -kT ln P(xi) (min shifted to 0).
        self.bin_centers = 0.5 * (self._bin_edges[:-1] + self._bin_edges[1:])
        tot = float(self._fe_hist.sum())
        if tot > 0:
            P = self._fe_hist / tot
            with np.errstate(divide="ignore"):
                fe_kT = -np.log(np.where(P > 0, P, np.nan))
            fe_kT = fe_kT - np.nanmin(fe_kT)
            self.free_energy_kT = fe_kT
            kT_ha = self.params.temperature * KELVIN_TO_HARTREE
            self.free_energy_kcal = fe_kT * kT_ha * HARTREE_TO_KCAL_MOL
        else:
            self.free_energy_kT = np.full(self.n_bins, np.nan)
            self.free_energy_kcal = np.full(self.n_bins, np.nan)

        # steady-state flux -> MFPT (tail-averaged to skip the WE transient).
        tau_time_fs = self._tau_steps * float(self.params.timestep)   # fs / iteration
        niter = len(self._flux_per_iter)
        if self._recycle and niter > 0:
            tail = max(1, niter // 2)
            flux_tail = float(np.mean(self._flux_per_iter[-tail:]))   # weight / iteration
            self.rate_per_fs = flux_tail / tau_time_fs
            self.mfpt_fs = (1.0 / self.rate_per_fs) if self.rate_per_fs > 0 else float("inf")
        self._log_summary()

    # ----------------------------------------------------------------- logging
    def _log_parameters_we(self):
        p = self.params
        E = " / ".join(f"{e:.3f}" for e in self._bin_edges)
        lines = ["\n" + "=" * 72 + "\n", f"{'WEIGHTED ENSEMBLE (WE) PARAMETERS':^72}\n",
                 "=" * 72 + "\n",
                 f"Initial walkers: {self.B}\n",
                 f"Calculator:      {type(self.calc).__name__} (one get_ef_gpu/step)\n",
                 f"Thermostat:      {p.thermostat}\n",
                 f"Timestep:        {p.timestep:.3f} fs\n",
                 f"tau_steps:       {self._tau_steps}   iterations: {self._n_iterations}\n",
                 f"walkers/bin (m): {self._m}\n",
                 f"n_bins:          {self.n_bins}   edges: {E}\n"]
        if self._target_bin is not None:
            lines.append(f"target/source:   bin {self._target_bin} / {self._source_bin}"
                         f"   recycle={self._recycle}\n")
        if p.resample_seed is not None:
            lines.append(f"resample seed:   {p.resample_seed}\n")
        if p.random_seed is not None:
            lines.append(f"MD seed:         {p.random_seed}\n")
        lines.append("=" * 72 + "\n")
        self.log_info(lines)

    def _log_summary(self):
        lines = ["\n" + "=" * 72 + "\n", f"{'WE SUMMARY':^72}\n", "=" * 72 + "\n",
                 f"  iterations={self._n_iterations}  final walkers={self.B}  "
                 f"Sum w={float(self._weights.sum()):.12f}\n",
                 "\n  -- free energy F(xi) = -kT ln P (weighted histogram) --\n",
                 "  bin   xi_center    P(xi)        F(kT)     F(kcal/mol)\n"]
        tot = float(self._fe_hist.sum()) or 1.0
        for i in range(self.n_bins):
            P = self._fe_hist[i] / tot
            fkT = self.free_energy_kT[i]
            fkc = self.free_energy_kcal[i]
            lines.append(f"  {i:>3}   {self.bin_centers[i]:>9.4f}   {P:>9.4e}   "
                         f"{fkT:>8.3f}   {fkc:>10.4f}\n")
        if self._target_bin is not None:
            lines.append("\n  -- steady-state flux / MFPT --\n")
            lines.append(f"  total flux weight into target bin {self._target_bin}: "
                         f"{self.flux_total:.6e}\n")
            lines.append(f"  rate = {self.rate_per_fs:.6e} /fs   "
                         f"MFPT = {self.mfpt_fs:.6e} fs "
                         f"({self.mfpt_fs / 1000.0:.4f} ps)\n")
        lines.append("=" * 72 + "\n")
        self.log_info(lines)
