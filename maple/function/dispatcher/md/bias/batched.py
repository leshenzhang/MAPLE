"""Batch-aware bias for MAPLE's batched MD core (rides ``ensemble/nvt_batched.py``).

The single-system bias layer (:mod:`posres_calc` / :mod:`steered` / :mod:`gamd` /
PLUMED / Colvars) wraps ``atoms.calc`` and recomputes one extra force per step at
``atoms.get_forces`` -- that architecture is SINGLE-system. ``BatchedNVT`` instead
advances B replicas with ONE ``calc.get_ef_gpu()`` per step over a padded
``(B, nmax_dof)`` force buffer, so a batch-aware bias must add a PER-REPLICA force
into that same buffer at the force-assembly point (``BatchedNVT._forces_au``).

ponytail / C3 ceiling: PLUMED and Colvars are NOT batched (their CV/bias engines
are single-walker, per-``atoms`` objects) -- they stay on the single-system path
and are rejected for B>1 by the batched kernel's ``_reject_ceiling_features``.
What IS implemented natively here is the one bias every batched method needs as
its batch dimension: a HARMONIC RESTRAINT on a collective variable, with a
per-replica centre. N umbrella windows = B replicas, each restrained to a
different centre along the same CV.

Batch-aware-bias contract (what the batched kernel calls; the SAME hook the
orchestrator can target to batch GaMD / steered-MD next)::

    bias.apply(E, F, calc) -> (E, F)

    E    : (B,)            replica potential energies   [Hartree]
    F    : (B, nmax_dof)   padded replica force buffer   [a.u. = Ha/Bohr]
    calc : the batch calculator; ``calc.coord`` (N_atoms,3, Angstrom) holds the
           live master coordinates and ``calc._ptr`` (B+1,) the per-replica atom
           offsets. The bias reads positions from it, computes its per-replica
           force, and ADDS it (in a.u.) into ``F`` in place; it returns the
           bias-augmented (E, F). A missing/None bias is an exact no-op
           (unbiased / single-system parity preserved).

To attach: set ``job._bias = BatchedHarmonicRestraint(...)`` on a ``BatchedNVT``
instance (``BatchedUmbrella`` does exactly this). Any object exposing the
``apply(E, F, calc)`` signature above is a valid batch-aware bias.

Collective variable -- distance between the centres of mass of two atom groups
(same indices in every replica; a single-atom group reduces it to an interatomic
distance), identical to the steered-MD CV (:mod:`steered`)::

    R1 = sum_{i in g1} m_i x_i / M1 ,  R2 likewise          [Angstrom]
    xi = | R1 - R2 |                                         [Angstrom]
    d(xi)/d(x_i) =  (m_i/M1) u   (i in g1),  -(m_j/M2) u  (j in g2),  u=(R1-R2)/xi

Per-replica harmonic restraint at centre ``xi0_b`` (force constant ``k`` in
**Ha/Angstrom^2**, the MAPLE force convention shared with ``posres_fc`` /
``smd_k``)::

    V_b   = 1/2 k_b (xi_b - xi0_b)^2                         [Hartree]
    F_i,b = -k_b (xi_b - xi0_b) d(xi)/d(x_i)                 [Ha/Angstrom]

Isolated (non-periodic) replicas only -- the batch backends carry no cell, so no
minimum-image is applied (matching ``BatchedNVT``'s isolated-only contract).
"""

import numpy as np

from .posres_calc import _build_selection     # reuse group selection (all/heavy/idx)
from ..utils import HA_PER_ANG_TO_AU          # Ha/Angstrom -> Ha/Bohr (a.u. force)
from .gamd import gamd_params, _Welford, HARTREE_PER_KCAL, KB_HA_PER_K  # reuse GaMD math


class BatchedHarmonicRestraint:
    """Per-replica harmonic restraint on a COM-COM distance CV for BatchedNVT.

    Parameters
    ----------
    atoms_list : list[ase.Atoms]
        The B replicas (used for masses + the group selection + atom counts).
        All replicas are assumed to share the same topology (the umbrella case:
        B copies of one system), so the CV group indices are resolved per replica
        but normally coincide.
    group1, group2 : str | sequence[int]
        The two CV groups (``"all"`` / ``"heavy"`` / a 0-based index list like
        ``"0,1,2"`` -- see :func:`posres_calc._build_selection`). Must be
        non-empty and non-overlapping.
    k : float | sequence[float]
        Force constant in **Ha/Angstrom^2** (scalar broadcast to all replicas, or
        one value per replica).
    centers : sequence[float]
        Per-replica restraint centre ``xi0_b`` (length B) in **Angstrom**.
    """

    KIND = "distance"   # COM-COM distance collective variable

    def __init__(self, atoms_list, group1, group2, k, centers):
        self.B = len(atoms_list)
        if self.B == 0:
            raise ValueError("BatchedHarmonicRestraint needs >= 1 replica.")
        self._g1 = [np.asarray(_build_selection(group1, at), dtype=int)
                    for at in atoms_list]
        self._g2 = [np.asarray(_build_selection(group2, at), dtype=int)
                    for at in atoms_list]
        self._m = [np.asarray(at.get_masses(), dtype=np.float64) for at in atoms_list]
        self._n = [len(at) for at in atoms_list]
        for b in range(self.B):
            if len(self._g1[b]) == 0 or len(self._g2[b]) == 0:
                raise ValueError("restraint CV groups must be non-empty.")
            if set(self._g1[b].tolist()) & set(self._g2[b].tolist()):
                raise ValueError("restraint CV groups must not overlap.")
        self.k = np.broadcast_to(np.asarray(k, np.float64), (self.B,)).astype(np.float64)
        self.centers = np.asarray(centers, np.float64).reshape(-1)
        if self.centers.shape[0] != self.B:
            raise ValueError(f"centers length {self.centers.shape[0]} != B={self.B}")
        # per-replica CV time series, appended once per force evaluation.
        self.cv_history = [[] for _ in range(self.B)]

    # ---------------------------------------------------------------- CV + grad
    def _cv_grad(self, pos, b):
        """Return (xi, restraint_force (n_b,3) Ha/Angstrom) for replica b.

        pos : (n_b, 3) Angstrom. Force = -k_b (xi - xi0_b) d(xi)/d(x_i)."""
        g1, g2, m = self._g1[b], self._g2[b], self._m[b]
        M1, M2 = m[g1].sum(), m[g2].sum()
        R1 = (m[g1, None] * pos[g1]).sum(0) / M1
        R2 = (m[g2, None] * pos[g2]).sum(0) / M2
        dvec = R1 - R2                                   # isolated -> no min-image
        xi = float(np.linalg.norm(dvec))
        u = dvec / xi if xi > 1e-9 else np.zeros(3)
        f = np.zeros((self._n[b], 3), dtype=np.float64)
        diff = xi - self.centers[b]
        coef = -self.k[b] * diff                         # scalar (Ha/Angstrom)
        if self.k[b] != 0.0 and xi > 1e-9:
            f[g1] += coef * (m[g1] / M1)[:, None] * u[None, :]
            f[g2] += coef * (m[g2] / M2)[:, None] * (-u[None, :])
        return xi, f

    def restraint_forces(self, coord_np, ptr):
        """Per-replica (CV, force) from master coords. Pure numpy (no torch/GPU).

        coord_np : (N_atoms, 3) Angstrom (replicas concatenated).
        ptr      : (B+1,) per-replica atom offsets (``calc._ptr``).
        Returns ``(cvs (B,), forces list[(n_b,3) Ha/Angstrom])``."""
        cvs = np.empty(self.B, dtype=np.float64)
        forces = []
        for b in range(self.B):
            pos = np.asarray(coord_np[ptr[b]:ptr[b + 1]], dtype=np.float64)
            xi, f = self._cv_grad(pos, b)
            cvs[b] = xi
            forces.append(f)
        return cvs, forces

    # ------------------------------------------------------------- bias contract
    def apply(self, E, F, calc):
        """Add the per-replica restraint to the padded force buffer + energy.

        E : (B,) Ha ; F : (B, nmax_dof) a.u. (Ha/Bohr) ; calc carries ``coord``
        (N_atoms,3 Angstrom) and ``_ptr`` (B+1). F is modified IN PLACE; the
        CV of each replica is recorded into ``self.cv_history``."""
        import torch
        coord_np = calc.coord.detach().to("cpu").numpy()
        cvs, forces = self.restraint_forces(coord_np, np.asarray(calc._ptr))
        dE = np.zeros(self.B, dtype=np.float64)
        for b in range(self.B):
            n = self._n[b]
            f_au = forces[b].reshape(-1) * HA_PER_ANG_TO_AU         # Ha/Bohr
            F[b, :3 * n] += torch.as_tensor(f_au, dtype=F.dtype, device=F.device)
            diff = cvs[b] - self.centers[b]
            dE[b] = 0.5 * self.k[b] * diff * diff
            self.cv_history[b].append(float(cvs[b]))
        E = E + torch.as_tensor(dE, dtype=E.dtype, device=E.device)
        return E, F


def _com_distance_cv(coord_np, ptr, g1_list, g2_list, m_list):
    """Per-replica COM-COM distance CV (Angstrom) from concatenated master coords.

    The SAME collective variable as :class:`BatchedHarmonicRestraint` (isolated;
    no minimum image), but value-only (no gradient): GaMD does NOT restrain along
    the CV -- it boosts the whole potential -- so only the CV *value* is needed,
    for reweighting and for the WE progress coordinate.

    coord_np : (N_atoms,3) Angstrom (replicas concatenated).
    ptr      : (B+1,) per-replica atom offsets (``calc._ptr``).
    g1_list/g2_list : per-replica int index arrays (the two CV groups).
    m_list   : per-replica mass arrays (amu; only ratios matter for the COM).
    Returns ``cvs`` (B,) in Angstrom."""
    B = len(g1_list)
    cvs = np.empty(B, dtype=np.float64)
    for b in range(B):
        pos = np.asarray(coord_np[ptr[b]:ptr[b + 1]], dtype=np.float64)
        g1, g2, m = g1_list[b], g2_list[b], m_list[b]
        R1 = (m[g1, None] * pos[g1]).sum(0) / m[g1].sum()
        R2 = (m[g2, None] * pos[g2]).sum(0) / m[g2].sum()
        cvs[b] = float(np.linalg.norm(R1 - R2))
    return cvs


class BatchedGaMD:
    """Per-replica pure-MLIP GaMD boost on the BatchedNVT bias hook.

    This is the batch-aware sibling of the single-system :class:`bias.gamd.
    GamdCalculator`: it reuses that module's parameter estimator
    (:func:`bias.gamd.gamd_params`, Miao 2015 eqs 7-11), its Welford accumulator
    and its reweighting math (:func:`bias.gamd.gamd_reweight_1d`) -- only the
    *injection point* differs (per-replica force buffer vs ASE ``get_forces``).

    Pure-MLIP reduction (see :mod:`bias.gamd`): the boost ``DeltaV=1/2 k (E-V)^2``
    (when ``V<E``) reduces, on a pure MLIP, to scaling the physical force by the
    scalar ``(1 - k(E-V)) in [0,1]``. So per step, per replica ``b``::

        if V_b < E:  F_b *= (1 - k*(E - V_b));  DeltaV_b = 1/2 k (E - V_b)^2

    The boost params ``(k, E)`` are estimated ONCE from a shared prep window
    (energies POOLED across all B replicas -- ParGaMD holds one finalized
    ``(E, Vmax, Vmin, k)`` fixed for every walker, paper Sec 2.4 step 1) and then
    frozen. ``boost=False`` makes this a passive CV logger (no force change,
    ``DeltaV=0``) -- used for the unbiased reference run.

    Bias contract: ``apply(E, F, calc) -> (E, F)`` (see module docstring). The
    per-replica CV is logged to ``cv_history`` and the boost to ``dv_history``
    (Ha) every force evaluation -- feed (concatenated production frames of)
    ``cv_history``/``dv_history`` to :func:`bias.gamd.gamd_reweight_1d` for the
    unbiased PMF.
    """

    KIND = "gamd"

    def __init__(self, atoms_list, group1, group2, *, boost=True, mode="lower",
                 sigma0_kcal=6.0, prep_steps=2000, temperature=300.0, params=None):
        self.B = len(atoms_list)
        if self.B == 0:
            raise ValueError("BatchedGaMD needs >= 1 replica.")
        self._g1 = [np.asarray(_build_selection(group1, at), dtype=int)
                    for at in atoms_list]
        self._g2 = [np.asarray(_build_selection(group2, at), dtype=int)
                    for at in atoms_list]
        self._m = [np.asarray(at.get_masses(), dtype=np.float64) for at in atoms_list]
        self._n = [len(at) for at in atoms_list]
        for b in range(self.B):
            if len(self._g1[b]) == 0 or len(self._g2[b]) == 0:
                raise ValueError("GaMD CV groups must be non-empty.")
            if set(self._g1[b].tolist()) & set(self._g2[b].tolist()):
                raise ValueError("GaMD CV groups must not overlap.")
        self.boost = bool(boost)
        self._mode = mode
        self._sigma0 = float(sigma0_kcal) * HARTREE_PER_KCAL   # Ha
        self._prep = max(int(prep_steps), 0)
        self._kT = KB_HA_PER_K * float(temperature)            # Ha
        self._stats = _Welford()                               # shared (pooled) stats
        self._params = dict(params) if params else None        # frozen if supplied
        self._istep = 0
        self.k = 0.0
        self.E_thr = 0.0
        # per-replica per-step logs (Ha for dV, Angstrom for CV); phase per step.
        self.dv_history = [[] for _ in range(self.B)]
        self.cv_history = [[] for _ in range(self.B)]
        self.phase_history = []

    @property
    def params(self):
        return self._params

    # -- shared boost params: pool all B replicas during prep, freeze once ----- #
    def _ensure_params(self, V_all):
        if self._params is not None:
            return self._params
        for V in V_all:
            self._stats.push(float(V))
        if self._istep + 1 >= self._prep and self._stats.n > 1:
            self._params = gamd_params(self._stats.vmin, self._stats.vmax,
                                       self._stats.mean, self._stats.sigma,
                                       self._sigma0, self._mode)
        return self._params  # None => still in prep (no boost yet)

    @staticmethod
    def _ptr_np(calc):
        ptr = calc._ptr
        return (ptr.detach().to("cpu").numpy() if hasattr(ptr, "detach")
                else np.asarray(ptr))

    # ------------------------------------------------------------- bias contract
    def apply(self, E, F, calc):
        """Scale each replica's force by the GaMD factor + log CV/boost.

        E : (B,) Ha (physical V) ; F : (B, nmax_dof) a.u. (modified IN PLACE) ;
        ``calc.coord`` (N_atoms,3 Angstrom) + ``calc._ptr`` (B+1). Returns the
        boosted (E+DeltaV, F)."""
        import torch
        coord_np = calc.coord.detach().to("cpu").numpy()
        ptr = self._ptr_np(calc)
        cvs = _com_distance_cv(coord_np, ptr, self._g1, self._g2, self._m)
        V_all = np.asarray(E.detach().to("cpu").numpy(), dtype=np.float64).reshape(-1)
        dV = np.zeros(self.B, dtype=np.float64)
        phase = "logonly"
        if self.boost:
            p = self._ensure_params(V_all)
            if p is not None and p["k"] > 0.0:
                k, Ethr = p["k"], p["E"]
                self.k, self.E_thr = k, Ethr
                phase = "prod"
                for b in range(self.B):
                    Vb = V_all[b]
                    if Vb < Ethr:
                        factor = 1.0 - k * (Ethr - Vb)        # in [0,1] by construction
                        n = self._n[b]
                        F[b, :3 * n].mul_(float(factor))
                        dV[b] = 0.5 * k * (Ethr - Vb) ** 2     # Ha
            else:
                phase = "prep"
        for b in range(self.B):
            self.cv_history[b].append(float(cvs[b]))
            self.dv_history[b].append(float(dV[b]))
        self.phase_history.append(phase)
        self._istep += 1
        if dV.any():
            E = E + torch.as_tensor(dV, dtype=E.dtype, device=E.device)
        return E, F


# --------------------------------------------------------------------------- #
# Standalone numpy self-check (no torch / no GPU; mirrors posres_calc/steered):
#   PYTHONSAFEPATH=1 python -m maple.function.dispatcher.md.bias.batched
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from ase import Atoms

    ok = True

    # (1) single-window (B=1), two single-atom groups on x at d=2.5, centre 2.0,
    #     k=1.5 Ha/A^2. restraint force MUST equal analytic -k(xi-xi0) u.
    a = Atoms("H2", positions=[[0.0, 0, 0], [2.5, 0, 0]])
    r = BatchedHarmonicRestraint([a], [0], [1], 1.5, [2.0])
    ptr = np.array([0, 2])
    coord = a.get_positions()
    cvs, forces = r.restraint_forces(coord, ptr)
    xi = cvs[0]
    u = (coord[0] - coord[1]) / np.linalg.norm(coord[0] - coord[1])   # (R1-R2)/xi
    analytic0 = -1.5 * (xi - 2.0) * u                                 # on group1 atom
    d0 = np.abs(forces[0][0] - analytic0).max()
    d1 = np.abs(forces[0][1] + analytic0).max()                      # equal/opposite
    ok &= (abs(xi - 2.5) < 1e-12) and (d0 < 1e-13) and (d1 < 1e-13)
    print(f"(1) xi={xi:.6f}(exp 2.5)  |F0-analytic|={d0:.2e}  "
          f"|F1+analytic|={d1:.2e}  (exp ~0)")

    # (2) COM weighting: g1={0,1} (two H), g2={2}; CV = |COM(g1)-COM(g2)|.
    a2 = Atoms("H3", positions=[[0.0, 0, 0], [0, 0, 0], [3.0, 0, 0]])
    r2 = BatchedHarmonicRestraint([a2], [0, 1], [2], 2.0, [3.0])     # centre = current CV
    cvs2, f2 = r2.restraint_forces(a2.get_positions(), np.array([0, 3]))
    ok &= abs(cvs2[0] - 3.0) < 1e-12 and np.abs(f2[0]).max() < 1e-12  # xi==centre -> 0 force
    print(f"(2) COM CV={cvs2[0]:.6f}(exp 3.0)  max|F|@xi==centre={np.abs(f2[0]).max():.1e}")

    # (3) per-replica centres: B=3 same geometry, 3 centres -> 3 different forces.
    reps = [Atoms("H2", positions=[[0.0, 0, 0], [2.5, 0, 0]]) for _ in range(3)]
    r3 = BatchedHarmonicRestraint(reps, [0], [1], 1.0, [2.0, 2.5, 3.0])
    ptr3 = np.array([0, 2, 4, 6])
    coord3 = np.concatenate([rr.get_positions() for rr in reps], axis=0)
    cvs3, f3 = r3.restraint_forces(coord3, ptr3)
    # diff = 2.5-centre = [0.5, 0.0, -0.5]; F0x = -k*diff*u_x, u_x=-1 -> [0.5,0,-0.5]
    fx = np.array([f3[b][0, 0] for b in range(3)])
    ok &= np.allclose(fx, [0.5, 0.0, -0.5], atol=1e-12)
    print(f"(3) per-window F0x={fx} (exp [0.5, 0.0, -0.5])")

    # (4) non-overlap / non-empty guards.
    try:
        BatchedHarmonicRestraint([a], [0], [0], 1.0, [2.0]); ok = False
    except ValueError:
        pass
    print(f"(4) overlap guard ok={ok}")

    # (5) GaMD CV helper (numpy) + boost-factor identity. B=2 H2 replicas at
    #     d=2.0 and 3.0; g1={0} g2={1} -> CV = bond length.
    repsg = [Atoms("H2", positions=[[0., 0, 0], [2.0, 0, 0]]),
             Atoms("H2", positions=[[0., 0, 0], [3.0, 0, 0]])]
    g = BatchedGaMD(repsg, [0], [1], boost=True, prep_steps=1,
                    sigma0_kcal=6.0, temperature=300.0)
    coordg = np.concatenate([r.get_positions() for r in repsg], axis=0)
    cvg = _com_distance_cv(coordg, np.array([0, 2, 4]), g._g1, g._g2, g._m)
    ok &= np.allclose(cvg, [2.0, 3.0], atol=1e-12)
    print(f"(5) GaMD CV={cvg} (exp [2.0, 3.0])")

    # (6) boost factor in [0,1] and DeltaV>=0 from frozen params over a V sweep.
    pg = gamd_params(-100.0, -90.0, -95.0, 2.0, sigma0=1.0 * HARTREE_PER_KCAL,
                     mode="lower")
    kk, EE = pg["k"], pg["E"]
    facs = [1.0 - kk * (EE - V) for V in (-100.0, -95.0, -90.0)]
    dvs = [0.5 * kk * (EE - V) ** 2 for V in (-100.0, -95.0, -90.0)]
    ok &= all(-1e-12 <= f <= 1.0 + 1e-12 for f in facs) and all(d >= 0 for d in dvs)
    print(f"(6) boost factors={['%.3f' % f for f in facs]} in [0,1]; "
          f"DeltaV>=0 ok")

    print("BATCHED-RESTRAINT/GaMD SELF-CHECK:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
