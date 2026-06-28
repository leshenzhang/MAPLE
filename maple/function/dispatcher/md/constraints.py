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

    # -- minimum-image displacement ------------------------------------------
    def _mic(self, d):
        """Minimum-image displacement for an (M,3) array (no-op when not PBC)."""
        if not self.pbc:
            return d
        f = d @ self._cell_inv
        f -= np.round(f)
        return f @ self._cell

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
        corrected velocities."""
        if self.n_constraints == 0:
            return velocities

        ai, aj = self.ai, self.aj
        inv_mi, inv_mj = self.inv_mi, self.inv_mj
        d0sq = self.d0sq
        tol = self.tol
        # reference bond directions r_i(t) - r_j(t) (classic SHAKE gradient)
        ref = self._mic(ref_positions[ai] - ref_positions[aj])
        # position correction dr[A] -> velocity correction dr/(dt_au*BOHR2ANG)
        vfac = 1.0 / (dt_au * BOHR_TO_ANGSTROM)

        pos = atoms.get_positions()
        v = velocities.copy()
        cell, cell_inv, pbc = self._cell, self._cell_inv, self.pbc

        for _ in range(self.max_iter):
            max_err = 0.0
            for k in range(self.n_constraints):
                i, j = ai[k], aj[k]
                rij = pos[i] - pos[j]
                if pbc:
                    fr = rij @ cell_inv
                    fr -= np.round(fr)
                    rij = fr @ cell
                d2 = rij @ rij
                diff = d2 - d0sq[k]
                err = abs(np.sqrt(d2) - self.d0[k])
                if err > max_err:
                    max_err = err
                if err <= tol:
                    continue
                s = ref[k]
                g = diff / (2.0 * self.inv_mu[k] * (rij @ s))
                di = (g * inv_mi[k]) * s
                dj = (g * inv_mj[k]) * s
                pos[i] -= di
                pos[j] += dj
                v[i] -= di * vfac
                v[j] += dj * vfac
            if max_err <= tol:
                break

        atoms.set_positions(pos)
        return v

    # -- RATTLE velocity stage ----------------------------------------------
    def project_velocities(self, atoms: Atoms, velocities: np.ndarray) -> np.ndarray:
        """Project velocities so the relative velocity along every bond is zero
        (RATTLE velocity stage). Returns the corrected velocities."""
        if self.n_constraints == 0:
            return velocities

        ai, aj = self.ai, self.aj
        inv_mi, inv_mj = self.inv_mi, self.inv_mj
        vtol = self.vtol
        pos = atoms.get_positions()
        v = velocities.copy()
        cell, cell_inv, pbc = self._cell, self._cell_inv, self.pbc

        # current bond vectors r(t+dt) (fixed during the velocity sweep)
        bond = self._mic(pos[ai] - pos[aj])
        bond2 = np.einsum("ij,ij->i", bond, bond)

        for _ in range(self.max_iter):
            max_err = 0.0
            for k in range(self.n_constraints):
                i, j = ai[k], aj[k]
                rij = bond[k]
                rv = (v[i] - v[j]) @ rij
                if abs(rv) > max_err:
                    max_err = abs(rv)
                if abs(rv) <= vtol:
                    continue
                kk = -rv / (self.inv_mu[k] * bond2[k])
                v[i] += (kk * inv_mi[k]) * rij
                v[j] -= (kk * inv_mj[k]) * rij
            if max_err <= vtol:
                break
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
