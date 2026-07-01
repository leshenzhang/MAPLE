"""
[TASK#9] Bond constraints for the MAPLE velocity-Verlet integrator.

Adds GROMACS-style holonomic distance constraints so the MD timestep can be
raised from ~0.5 fs to 2 fs by freezing the fast X-H stretch (and, for water,
the full rigid 3-site geometry).

GROMACS ``constraints`` mdp key (identical meaning):
    none      no constraints (default; backward compatible — the unconstrained
              integrator path is untouched).
    h-bonds   constrain every covalent bond involving a hydrogen.
    all-bonds constrain every covalent bond.
    h-angles  all-bonds PLUS convert angles that involve a hydrogen into 1-3
              distance constraints (so e.g. the H-X-H opening is also frozen).

Rigid water (this matters most — water dominates enzyme systems):
    Any oxygen covalently bonded to EXACTLY two hydrogens (and nothing else) is
    detected purely by topology and made fully rigid with three distance
    constraints (O-H1, O-H2, H1-H2), removing 3 DOF per molecule.  This
    reproduces the rigid geometry that GROMACS enforces with SETTLE
    (Miyamoto & Kollman, J. Comput. Chem. 13, 952, 1992).  Water is rigidified
    whenever ``constraints != none`` (matching the common enzyme setup where the
    water model ships a ``[settles]`` block).  The reduction of rigid water to
    three distance constraints is solved by the same iterative RATTLE projection
    used for the other constraints; per the task this is explicitly equivalent
    to the analytic SETTLE result (bond invariance < ``tol`` = 1e-8 A).

Algorithm — RATTLE (velocity-Verlet correct; Andersen, J. Comput. Phys. 52, 24,
1983):
    * project_positions: after the unconstrained drift, restore every bond to
      its frozen length d0 with a SHAKE sweep that uses the reference geometry
      r(t) for the constraint gradients, and apply the matching correction to
      the half-step velocities (this is the RATTLE position stage).
    * project_velocities: after each velocity half-kick, remove the relative
      velocity component along every bond ((v_i - v_j).r_ij = 0).  This is the
      RATTLE velocity stage.
    Plain position-only SHAKE is NOT velocity-Verlet correct; RATTLE constrains
    BOTH positions and velocities, which is required here.

``constraint_algorithm`` mdp key (GROMACS values accepted):
    lincs (default, = GROMACS default) | shake
    Both are solved by the iterative RATTLE projection here (they converge to
    the same constrained manifold to ``tol``); a dedicated LINCS solver
    (Hess et al., J. Comput. Chem. 18, 1463, 1997) is a documented future option.

Bond topology:
    MAPLE carries no bonded topology, so bonds are inferred ONCE at t=0 from
    covalent radii (ase.neighborlist.natural_cutoffs scaled by ``bond_mult``)
    and the bond list is then FROZEN for the whole trajectory.

Units:
    positions A, masses amu (only mass ratios enter), velocities atomic units
    (Bohr / a.u.-time).  The integrator advances r[A] += v[au]*dt[au]*BOHR2ANG,
    so a position correction dr[A] maps to a velocity correction
    dv[au] = dr / (dt_au * BOHR2ANG).
"""

import numpy as np
from ase import Atoms
from ase.neighborlist import natural_cutoffs, neighbor_list

from .utils import BOHR_TO_ANGSTROM

_DEFAULT_BOND_MULT = 1.2      # scale on summed covalent radii for bond inference
_DEFAULT_TOL = 1e-8           # A; convergence on |bond - d0|
_DEFAULT_VTOL = 1e-10         # convergence on |(v_i-v_j).r_ij| (au*A)
_DEFAULT_MAX_ITER = 500

_VALID_MODES = {"none", "h-bonds", "all-bonds", "h-angles"}
_VALID_ALGOS = {"lincs", "shake"}


# ----------------------------------------------------------------------------
# Topology helpers (run ONCE at t=0)
# ----------------------------------------------------------------------------
def _infer_bonds(atoms: Atoms, mult: float):
    """Infer the frozen bond list from covalent radii at t=0 (unique i<j)."""
    cutoffs = natural_cutoffs(atoms, mult=mult)
    i_idx, j_idx = neighbor_list("ij", atoms, cutoffs)
    pairs = set()
    for a, b in zip(i_idx.tolist(), j_idx.tolist()):
        pairs.add((a, b) if a < b else (b, a))
    return sorted(pairs)


def _neighbor_map(n_atoms: int, bonds):
    neigh = {k: [] for k in range(n_atoms)}
    for a, b in bonds:
        neigh[a].append(b)
        neigh[b].append(a)
    return neigh


def _detect_water(atoms: Atoms, neigh):
    """Water = O bonded to exactly two H and nothing else. Returns (groups, set)."""
    symbols = atoms.get_chemical_symbols()
    groups = []
    water_atoms = set()
    for o in range(len(atoms)):
        if symbols[o] != "O":
            continue
        nb = neigh[o]
        h = [x for x in nb if symbols[x] == "H"]
        if len(nb) == 2 and len(h) == 2:
            groups.append((o, h[0], h[1]))
            water_atoms.update((o, h[0], h[1]))
    return groups, water_atoms


class ConstraintManager:
    """Holds the frozen constraint set and applies RATTLE position/velocity
    projections during the velocity-Verlet step.

    Constraints are stored as flat arrays (ai, aj, d0) over distance pairs.
    Rigid water contributes its three pairs like any other distance constraint;
    each distance pair removes exactly one DOF (rigid water => 3 DOF).
    """

    def __init__(self, atoms: Atoms, mode: str,
                 algorithm: str = "lincs",
                 bond_mult: float = _DEFAULT_BOND_MULT,
                 tol: float = _DEFAULT_TOL,
                 vtol: float = _DEFAULT_VTOL,
                 max_iter: int = _DEFAULT_MAX_ITER):
        self.mode = mode
        self.algorithm = algorithm
        self.tol = tol
        self.vtol = vtol
        self.max_iter = max_iter

        # PBC / minimum-image setup
        self.pbc = bool(np.any(atoms.pbc))
        if self.pbc:
            self._cell = np.asarray(atoms.cell.array, dtype=float)
            self._cell_inv = np.linalg.inv(self._cell)
        else:
            self._cell = None
            self._cell_inv = None

        bonds = _infer_bonds(atoms, bond_mult)
        neigh = _neighbor_map(len(atoms), bonds)
        self.water_groups, water_atoms = _detect_water(atoms, neigh)
        self.n_water = len(self.water_groups)

        symbols = atoms.get_chemical_symbols()
        pairs = []          # (i, j) constraint pairs
        seen = set()

        def _add(i, j):
            key = (i, j) if i < j else (j, i)
            if key not in seen:
                seen.add(key)
                pairs.append(key)

        # 1) rigid water -> 3 distance constraints each (= SETTLE geometry)
        for (o, h1, h2) in self.water_groups:
            _add(o, h1)
            _add(o, h2)
            _add(h1, h2)

        # 2) non-water bonds per GROMACS mode
        for (a, b) in bonds:
            if a in water_atoms and b in water_atoms:
                continue                       # already rigid via water
            is_h = symbols[a] == "H" or symbols[b] == "H"
            if mode == "h-bonds" and not is_h:
                continue
            # all-bonds and h-angles keep every (non-water) bond
            _add(a, b)

        # 3) h-angles: add 1-3 distance constraints for angles involving H
        if mode == "h-angles":
            for j in range(len(atoms)):
                nb = neigh[j]
                for x in range(len(nb)):
                    for y in range(x + 1, len(nb)):
                        i_, k_ = nb[x], nb[y]
                        if i_ in water_atoms and k_ in water_atoms:
                            continue
                        if symbols[i_] == "H" or symbols[k_] == "H":
                            _add(i_, k_)

        self.ai = np.array([p[0] for p in pairs], dtype=int)
        self.aj = np.array([p[1] for p in pairs], dtype=int)
        self.n_constraints = len(pairs)

        # frozen equilibrium lengths from the t=0 geometry (minimum image)
        d0vec = self._mic(atoms.get_positions()[self.ai]
                          - atoms.get_positions()[self.aj])
        self.d0 = np.linalg.norm(d0vec, axis=1)
        self.d0sq = self.d0 ** 2

        masses = atoms.get_masses()            # amu (ratios only)
        self.inv_mi = 1.0 / masses[self.ai]
        self.inv_mj = 1.0 / masses[self.aj]
        self.inv_mu = self.inv_mi + self.inv_mj

        # Jacobi convergence-iteration counters (last sweep); 0 before first call.
        self.last_pos_iters = 0
        self.last_vel_iters = 0

    # -- minimum-image displacement ------------------------------------------
    def _mic(self, d):
        """Minimum-image displacement for an (M,3) array (no-op when not PBC)."""
        if not self.pbc:
            return d
        f = d @ self._cell_inv
        f -= np.round(f)
        return f @ self._cell

    # -- cell refresh (NPT only) ---------------------------------------------
    def sync_cell(self, atoms: Atoms) -> None:
        """Refresh the cached cell used for minimum-image constraint geometry.

        NVE / NVT keep a fixed cell, so the t=0 cache is exact and this is never
        needed. NPT changes the cell every step (the barostat rescales it), so a
        frozen t=0 cell would mis-image any constrained bond whose two atoms
        straddle a periodic face. Call this once per NPT step (after the barostat
        rescale) so the RATTLE position/velocity projections use the current
        cell. No-op for non-periodic systems."""
        if not self.pbc:
            return
        self._cell = np.asarray(atoms.cell.array, dtype=float)
        self._cell_inv = np.linalg.inv(self._cell)

    # -- DOF bookkeeping -----------------------------------------------------
    @property
    def n_dof_removed(self) -> int:
        """Each distance constraint removes one DOF (rigid water = 3)."""
        return int(self.n_constraints)

    def summary(self) -> str:
        return (f"constraints={self.mode} algorithm={self.algorithm} "
                f"(solver=RATTLE) n_constraints={self.n_constraints} "
                f"(rigid_water={self.n_water}, removes {self.n_dof_removed} DOF)")

    # -- RATTLE position stage (SHAKE + half-step velocity correction) -------
    def project_positions(self, atoms: Atoms, ref_positions: np.ndarray,
                          velocities: np.ndarray, dt_au: float) -> np.ndarray:
        """Restore all bond lengths (positions modified in place on ``atoms``)
        and apply the matching correction to ``velocities``. Returns the
        corrected velocities.

        Vectorized batched-JACOBI RATTLE sweep. Every constraint correction in a
        sweep is computed from the SAME pre-sweep positions in a single ``(M,3)``
        minimum-image op, and the M corrections are applied SIMULTANEOUSLY via
        ``np.add.at`` scatter (Jacobi update), instead of sequentially reusing
        partially-updated positions (Gauss-Seidel). Jacobi is order-free and fully
        vectorized; it converges to the identical constrained manifold as the
        serial SHAKE sweep, only needing more outer iterations. The outer
        convergence-iteration loop is retained, and a post-loop residual check
        (below) guards the new path against returning an unconverged geometry."""
        if self.n_constraints == 0:
            return velocities

        ai, aj = self.ai, self.aj
        inv_mi, inv_mj, inv_mu = self.inv_mi, self.inv_mj, self.inv_mu
        d0, d0sq = self.d0, self.d0sq
        tol = self.tol
        # reference bond directions r_i(t) - r_j(t) (classic SHAKE gradient); fixed
        ref = self._mic(ref_positions[ai] - ref_positions[aj])
        # position correction dr[A] -> velocity correction dr/(dt_au*BOHR2ANG)
        vfac = 1.0 / (dt_au * BOHR_TO_ANGSTROM)

        pos = atoms.get_positions()
        v = velocities.copy()

        n_iter = 0
        for _ in range(self.max_iter):
            n_iter += 1
            # all M current bond vectors from the SAME pre-sweep positions (Jacobi)
            rij = self._mic(pos[ai] - pos[aj])              # (M,3)
            d2 = np.einsum("ij,ij->i", rij, rij)            # (M,)
            err = np.abs(np.sqrt(d2) - d0)                  # (M,)
            max_err = float(err.max())
            if max_err <= tol:
                break
            active = err > tol
            diff = d2 - d0sq                                # (M,)
            rij_dot_s = np.einsum("ij,ij->i", rij, ref)     # (M,)
            with np.errstate(divide="ignore", invalid="ignore"):
                g = np.where(active, diff / (2.0 * inv_mu * rij_dot_s), 0.0)
            di = (g * inv_mi)[:, None] * ref                # (M,3)
            dj = (g * inv_mj)[:, None] * ref                # (M,3)
            # Jacobi scatter: all M corrections applied simultaneously (an atom
            # shared by several constraints accumulates via np.add.at, unbuffered)
            np.add.at(pos, ai, -di)
            np.add.at(pos, aj,  dj)
            np.add.at(v,   ai, -di * vfac)
            np.add.at(v,   aj,  dj * vfac)

        # post-loop convergence guard (Jacobi needs more sweeps than Gauss-Seidel)
        rij = self._mic(pos[ai] - pos[aj])
        max_res = float(np.abs(np.sqrt(np.einsum("ij,ij->i", rij, rij)) - d0).max())
        if max_res > tol:
            raise RuntimeError(
                f"RATTLE position projection (Jacobi) failed to converge: max bond "
                f"residual {max_res:.3e} A > tol {tol:.3e} A after {self.max_iter} "
                f"iterations. Raise ConstraintManager max_iter.")
        self.last_pos_iters = n_iter

        atoms.set_positions(pos)
        return v

    # -- RATTLE velocity stage ----------------------------------------------
    def project_velocities(self, atoms: Atoms, velocities: np.ndarray) -> np.ndarray:
        """Project velocities so the relative velocity along every bond is zero
        (RATTLE velocity stage). Returns the corrected velocities.

        Vectorized batched-JACOBI sweep (same scheme as ``project_positions``):
        all M relative-velocity corrections are computed from the SAME pre-sweep
        velocities and scattered simultaneously via ``np.add.at`` scatter, instead
        of the serial Gauss-Seidel per-constraint update. A post-loop residual
        check guards against returning unconverged velocities."""
        if self.n_constraints == 0:
            return velocities

        ai, aj = self.ai, self.aj
        inv_mi, inv_mj, inv_mu = self.inv_mi, self.inv_mj, self.inv_mu
        vtol = self.vtol
        pos = atoms.get_positions()
        v = velocities.copy()

        # current bond vectors r(t+dt) (fixed during the velocity sweep)
        bond = self._mic(pos[ai] - pos[aj])
        bond2 = np.einsum("ij,ij->i", bond, bond)

        n_iter = 0
        for _ in range(self.max_iter):
            n_iter += 1
            # relative velocity along each bond from the SAME pre-sweep v (Jacobi)
            rv = np.einsum("ij,ij->i", v[ai] - v[aj], bond)   # (M,)
            max_err = float(np.abs(rv).max())
            if max_err <= vtol:
                break
            active = np.abs(rv) > vtol
            with np.errstate(divide="ignore", invalid="ignore"):
                kk = np.where(active, -rv / (inv_mu * bond2), 0.0)   # (M,)
            dvi = (kk * inv_mi)[:, None] * bond               # (M,3)
            dvj = (kk * inv_mj)[:, None] * bond               # (M,3)
            # Jacobi scatter: all M velocity corrections applied simultaneously
            np.add.at(v, ai,  dvi)
            np.add.at(v, aj, -dvj)

        # post-loop convergence guard
        rv = np.einsum("ij,ij->i", v[ai] - v[aj], bond)
        max_res = float(np.abs(rv).max())
        if max_res > vtol:
            raise RuntimeError(
                f"RATTLE velocity projection (Jacobi) failed to converge: max "
                f"|(v_i-v_j).r_ij| {max_res:.3e} > vtol {vtol:.3e} after "
                f"{self.max_iter} iterations. Raise ConstraintManager max_iter.")
        self.last_vel_iters = n_iter
        return v


def build_constraint_manager(atoms: Atoms, params):
    """Construct a ConstraintManager from an MD params dataclass.

    Returns None when ``constraints == none`` (unconstrained path unchanged).
    ``params`` must expose ``constraints`` and ``constraint_algorithm``.
    """
    mode = str(getattr(params, "constraints", "none") or "none").strip().lower()
    if mode in ("", "none"):
        return None
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown constraints='{mode}'. Choose from: "
            f"{sorted(_VALID_MODES)} (GROMACS-identical meaning)."
        )
    algo = str(getattr(params, "constraint_algorithm", "lincs") or "lincs").strip().lower()
    if algo not in _VALID_ALGOS:
        raise ValueError(
            f"Unknown constraint_algorithm='{algo}'. Choose from: {sorted(_VALID_ALGOS)}."
        )
    return ConstraintManager(atoms, mode=mode, algorithm=algo)


# ----------------------------------------------------------------------------
# Hydrogen Mass Repartitioning (HMR) — enables a 4 fs time step together with
# H-bond constraints (Hopkins, Le Grand, Walker & Roitberg, JCTC 2015,
# 10.1021/ct5010406). Mass-only edit, so it composes with any MLIP force engine
# and with the RATTLE constraint manager above (TASK#9).
# ----------------------------------------------------------------------------
_HMR_FLOOR = 1.0     # amu; a heavy donor may not drop to/below this


def _hmr_factor(raw, explicit):
    """Resolve the HMR scale factor from params.hmr (+ optional params.hmr_factor).

    Falsey / 'off' / 'no' / 'false' / 'none' / '0' => 0.0 (disabled).
    True / 'on' / 'yes' / 'true' => 3.0 (standard ~3.024 amu hydrogen).
    A number (or numeric string) => that factor.
    """
    if explicit is not None:
        try:
            return max(0.0, float(explicit))
        except (TypeError, ValueError):
            pass
    if isinstance(raw, bool):
        return 3.0 if raw else 0.0
    if isinstance(raw, (int, float)):
        return max(0.0, float(raw))
    s = str(raw).strip().lower()
    if s in ("", "off", "no", "false", "none", "0"):
        return 0.0
    if s in ("on", "yes", "true"):
        return 3.0
    try:
        return max(0.0, float(s))
    except ValueError:
        return 3.0     # 'hmr' truthy but unparseable -> standard factor


def maybe_repartition_masses(atoms: Atoms, params, *, log: bool = True) -> float:
    """Repartition hydrogen masses in place (AMBER-style HMR); return the factor.

    Each hydrogen mass is scaled to ``factor * m_H`` and the added mass
    ``m_H*(factor-1)`` is subtracted from the single heavy atom it is covalently
    bonded to. Bonds are inferred from covalent radii via the same topology
    helper the constraint manager uses, so HMR and H-bond constraints see an
    identical bond set. Total mass is conserved exactly; forces/energies are
    untouched (pure MLIP-safe). Slowing the fastest X-H motions is what permits a
    constrained integrator to use up to a 4 fs step.

    Controlled by ``params.hmr`` (falsey => no-op) and optional
    ``params.hmr_factor`` / ``params.hmr_bond_mult``. Idempotent (guarded by
    ``atoms.info['hmr_applied']``). Returns 0.0 when disabled.

    Reads params via getattr so it composes with any params object, but
    hmr / hmr_factor / hmr_bond_mult are also declared as real NVE/NVT/NPTParams
    dataclass fields -- otherwise _init_params (jobABC) would strip the unknown
    keys when building the dataclass and HMR would silently no-op on the
    production ensemble path (the bug the gamd/* fields had until B-51).
    """
    factor = _hmr_factor(getattr(params, "hmr", ""), getattr(params, "hmr_factor", None))
    if factor <= 0.0:
        return 0.0
    if atoms.info.get("hmr_applied"):
        return float(atoms.info["hmr_applied"])        # already applied; no double-count

    masses = atoms.get_masses().astype(float)
    numbers = atoms.get_atomic_numbers()
    mult = float(getattr(params, "hmr_bond_mult", _DEFAULT_BOND_MULT))
    neigh = _neighbor_map(len(atoms), _infer_bonds(atoms, mult))
    symbols = atoms.get_chemical_symbols()

    h_idx = [i for i in range(len(atoms)) if numbers[i] == 1]
    moved = 0
    for h in h_idx:
        heavy = [x for x in neigh[h] if numbers[x] > 1]
        if len(heavy) != 1:
            # bridging H / isolated H / H2: no unique heavy donor -> leave it (AMBER skips)
            if log:
                print(f"[HMR] H atom {h} has {len(heavy)} heavy neighbour(s); skipped.")
            continue
        donor = heavy[0]
        d = masses[h] * (factor - 1.0)
        if masses[donor] - d <= _HMR_FLOOR:
            raise ValueError(
                f"[HMR] heavy atom {donor} ({symbols[donor]}, {masses[donor]:.3f} amu) "
                f"cannot donate {d:.3f} amu at factor {factor:g}; lower hmr_factor.")
        masses[h] = masses[h] * factor
        masses[donor] -= d
        moved += 1

    atoms.set_masses(masses)
    atoms.info["hmr_applied"] = float(factor)
    if log:
        print(f"[HMR] factor={factor:g}: repartitioned {moved}/{len(h_idx)} H; "
              f"total mass {masses.sum():.4f} amu (conserved).")
    return float(factor)


if __name__ == "__main__":
    # Self-test: HMR conserves mass, scales H, draws from the bonded heavy atom,
    # is idempotent, no-ops when disabled, and guards an impossible factor.
    from types import SimpleNamespace
    from ase.build import molecule

    def _masses_of(name, **pkw):
        a = molecule(name)
        maybe_repartition_masses(a, SimpleNamespace(**pkw), log=False)
        return a, a.get_masses().copy()

    base = molecule("CH3CH2OH")
    m0 = base.get_masses().copy()
    nums = base.get_atomic_numbers()

    # (1) factor 3: total mass conserved, every H == 3*m_H
    a, m = _masses_of("CH3CH2OH", hmr=True)
    assert abs(m.sum() - m0.sum()) < 1e-9, (m.sum(), m0.sum())
    for i in range(len(a)):
        if nums[i] == 1:
            assert abs(m[i] - 3.0 * m0[i]) < 1e-9, (i, m[i], m0[i])
    # heavy atoms only ever lose mass
    for i in range(len(a)):
        if nums[i] > 1:
            assert m[i] <= m0[i] + 1e-12, (i, m[i], m0[i])
    print(f"(1) factor3 ethanol OK: sum {m.sum():.4f}==={m0.sum():.4f}")

    # (2) explicit numeric factor via hmr_factor overrides hmr
    a2, m2 = _masses_of("H2O", hmr=True, hmr_factor=2.0)
    for i in range(len(a2)):
        if a2.get_atomic_numbers()[i] == 1:
            assert abs(m2[i] - 2.0 * 1.008) < 1e-2, (i, m2[i])
    assert abs(m2.sum() - molecule("H2O").get_masses().sum()) < 1e-9
    print(f"(2) factor2 water OK: O={m2[0]:.4f} H={m2[1]:.4f}")

    # (3) idempotent: a second call does not double-apply
    maybe_repartition_masses(a, SimpleNamespace(hmr=True), log=False)
    assert np.allclose(a.get_masses(), m), "HMR not idempotent"
    print("(3) idempotent OK")

    # (4) disabled => factor 0, masses untouched
    a4 = molecule("CH3CH2OH")
    f4 = maybe_repartition_masses(a4, SimpleNamespace(hmr="off"), log=False)
    assert f4 == 0.0 and np.allclose(a4.get_masses(), m0)
    print("(4) disabled no-op OK")

    # (5) floor guard: methane carbon (12 amu) cannot feed 4 H at factor 5
    raised = False
    try:
        maybe_repartition_masses(molecule("CH4"), SimpleNamespace(hmr=5.0), log=False)
    except ValueError:
        raised = True
    assert raised, "floor guard did not fire"
    print("(5) floor guard OK")

    print("HMR SELF-CHECK PASS")
