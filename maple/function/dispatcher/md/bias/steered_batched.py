"""Batch-aware CONSTANT-VELOCITY steered-MD bias for MAPLE's batched MD core.

Rides the SAME batch-aware-bias hook as
:class:`bias.batched.BatchedHarmonicRestraint` (``bias.apply(E, F, calc)`` at
``BatchedNVT._forces_au``), but the per-replica restraint CENTRE MOVES with the
MD step -- this is steered MD (SMD): N independent constant-velocity pulls run
as the batch dimension B, and Jarzynski's equality
(:func:`bias.steered.jarzynski_1d`) reconstructs Delta-G(lambda) from the N
accumulated works.

Physics (Park & Schulten, J. Chem. Phys. 120, 5946 (2004), doi:10.1063/1.1651473;
Jarzynski, Phys. Rev. Lett. 78, 2690 (1997)). Per replica ``b``, on the COM-COM
distance CV ``xi_b`` (the SAME CV as the static restraint / single-system
:mod:`bias.steered`)::

    lambda_b(n) = lam0_b + dlam_b * n                      [Angstrom]
    V_b         = 1/2 k_b (xi_b - lambda_b)^2              [Hartree]
    F_i,b       = -k_b (xi_b - lambda_b) d(xi)/d(x_i)      [Ha/Angstrom]
    dW_b        = k_b (lambda_b - xi_b) dlambda_b          [Hartree]  (accumulated)

``dlam_b`` = the per-force-evaluation centre increment = ``v_b * dt`` (Angstrom).
``n`` counts force evaluations: ``n = 0`` at the first ``apply`` (lambda_b =
lam0_b, dW_b = 0), reaching ``lam0_b + dlam_b * steps`` at the final force eval.
This mirrors the single-system :class:`bias.steered.SteeredMDCalculator` math
bit-for-bit (same CV gradient ``_cv_grad``, same Park-2004 work increment), so
the single-system steered self-test (B-53) is the reference bar. The restraint
force itself REUSES the base class's analytic ``_cv_grad`` / ``restraint_forces``
(only the centre is swapped per step).

Isolated (non-periodic) replicas only (inherited contract). ``k`` in
**Ha/Angstrom^2** (the ``bias.steered`` / ``posres_fc`` convention).
"""

import numpy as np

from .batched import BatchedHarmonicRestraint, _ptr_to_np
from ..utils import HA_PER_ANG_TO_AU          # Ha/Angstrom -> Ha/Bohr (a.u. force)


class BatchedMovingRestraint(BatchedHarmonicRestraint):
    """Per-replica constant-velocity moving harmonic restraint (steered MD).

    Parameters
    ----------
    atoms_list : list[ase.Atoms]
        The B replicas (masses + group selection + atom counts), one pull each.
    group1, group2 : str | sequence[int]
        The two CV groups (``"all"`` / ``"heavy"`` / ``"0,1,2"`` --
        see :func:`posres_calc._build_selection`); non-empty, non-overlapping.
    k : float | sequence[float]
        Force constant in **Ha/Angstrom^2** (scalar broadcast, or one per pull).
    lam0 : sequence[float]
        Per-replica start centre ``lambda0_b`` (length B) in **Angstrom**.
    dlam_per_step : float | sequence[float]
        Per-force-evaluation centre increment ``dlam_b = v_b * dt`` (Angstrom);
        scalar broadcast to all pulls, or one value per pull.
    """

    KIND = "distance"   # COM-COM distance collective variable (moving centre)

    def __init__(self, atoms_list, group1, group2, k, lam0, dlam_per_step):
        lam0 = np.asarray(lam0, np.float64).reshape(-1)
        # initialise the static base with centres = lam0 (reuses _cv_grad infra).
        super().__init__(atoms_list, group1, group2, k, lam0)
        if lam0.shape[0] != self.B:
            raise ValueError(f"lam0 length {lam0.shape[0]} != B={self.B}")
        self.lam0 = lam0.copy()
        self.dlam = np.broadcast_to(np.asarray(dlam_per_step, np.float64),
                                    (self.B,)).astype(np.float64).copy()
        self._istep = 0
        self.work = np.zeros(self.B, np.float64)        # accumulated work [Ha]
        self._lam_prev = self.lam0.copy()
        # per-replica time series (one append per force evaluation).
        self.lam_history = [[] for _ in range(self.B)]
        self.work_history = [[] for _ in range(self.B)]
        # cv_history is created by the base class.

    def lambda_now(self):
        """Per-replica restraint centre at the current force-eval index (B,)."""
        return self.lam0 + self.dlam * self._istep

    # ------------------------------------------------------------- bias contract
    def apply(self, E, F, calc):
        """Add the per-replica MOVING restraint to (E, F) and accumulate work.

        Mirrors :meth:`BatchedHarmonicRestraint.apply` but (i) advances the
        centre to ``lambda_now()`` before computing the analytic force and
        (ii) accumulates the Park-2004 external work per replica. ``F`` is
        modified IN PLACE; returns the bias-augmented ``(E, F)``."""
        import torch
        lam = self.lambda_now()
        self.centers = lam                              # move the restraint centre
        coord_np = calc.coord.detach().to("cpu").numpy()
        cvs, forces = self.restraint_forces(coord_np, _ptr_to_np(calc._ptr))
        dE = np.zeros(self.B, np.float64)
        for b in range(self.B):
            n = self._n[b]
            f_au = forces[b].reshape(-1) * HA_PER_ANG_TO_AU      # Ha/Bohr
            F[b, :3 * n] += torch.as_tensor(f_au, dtype=F.dtype, device=F.device)
            diff = cvs[b] - lam[b]
            dE[b] = 0.5 * self.k[b] * diff * diff
            # external work increment (Park & Schulten 2004): dW = k (lam - xi) dlam
            dlam = lam[b] - self._lam_prev[b]
            self.work[b] += self.k[b] * (lam[b] - cvs[b]) * dlam
            self.cv_history[b].append(float(cvs[b]))
            self.lam_history[b].append(float(lam[b]))
            self.work_history[b].append(float(self.work[b]))
        self._lam_prev = lam.copy()
        self._istep += 1
        E = E + torch.as_tensor(dE, dtype=E.dtype, device=E.device)
        return E, F


# --------------------------------------------------------------------------- #
# Standalone numpy self-check (no torch / no GPU; mirrors bias/batched.py):
#   PYTHONSAFEPATH=1 python -m maple.function.dispatcher.md.bias.steered_batched
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from ase import Atoms

    ok = True

    # (1) moving centre: B=1 two single-atom groups on x at d=2.5, lam0=2.0,
    #     dlam=0.1, k=1.5. At n=0 centre=2.0 -> force = -k(2.5-2.0)u; at n=2
    #     centre=2.2 -> force = -k(2.5-2.2)u. restraint_forces must match.
    a = Atoms("H2", positions=[[0.0, 0, 0], [2.5, 0, 0]])
    r = BatchedMovingRestraint([a], [0], [1], 1.5, [2.0], 0.1)
    ptr = np.array([0, 2])
    coord = a.get_positions()
    u = (coord[0] - coord[1]) / np.linalg.norm(coord[0] - coord[1])   # (R1-R2)/xi
    for n in range(3):
        r.centers = r.lambda_now()
        cvs, forces = r.restraint_forces(coord, ptr)
        lam = 2.0 + 0.1 * n
        analytic0 = -1.5 * (cvs[0] - lam) * u
        ok &= (np.abs(forces[0][0] - analytic0).max() < 1e-12
               and np.abs(forces[0][1] + analytic0).max() < 1e-12)
        r._istep += 1
    print(f"(1) moving-centre force matches -k(xi-lam)u over 3 steps  ok={ok}")

    # (2) work accumulation sign + dlam=0 first step. Two atoms, attractive pull:
    #     synthetic ZeroCalc-like: emulate apply() bookkeeping by hand on a frozen
    #     CV xi=2.5, lam0=2.0, dlam=0.1, k=1.0. dW = k(lam-xi)dlam.
    r2 = BatchedMovingRestraint([a], [0], [1], 1.0, [2.0], 0.1)
    works = []
    lam_prev = 2.0
    for n in range(4):
        lam = 2.0 + 0.1 * n
        dW = 1.0 * (lam - 2.5) * (lam - lam_prev)
        r2.work[0] += dW
        works.append(r2.work[0])
        lam_prev = lam
    # n=0: dW=0; pulling lam toward xi=2.5 from below -> (lam-xi)<0, dlam>0 -> W<0
    ok &= abs(works[0]) < 1e-15 and works[-1] < 0.0
    print(f"(2) work[0]={works[0]:.2e}(exp 0)  work_end={works[-1]:.4f} (exp <0)")

    # (3) per-pull INDEPENDENCE: B=3 same geom, 3 distinct dlam -> 3 distinct
    #     centre trajectories -> 3 distinct forces.
    reps = [Atoms("H2", positions=[[0.0, 0, 0], [2.5, 0, 0]]) for _ in range(3)]
    r3 = BatchedMovingRestraint(reps, [0], [1], 1.0, [2.0, 2.0, 2.0],
                                [0.0, 0.1, 0.2])
    ptr3 = np.array([0, 2, 4, 6])
    coord3 = np.concatenate([rr.get_positions() for rr in reps], axis=0)
    r3._istep = 2                                  # advance the centre
    r3.centers = r3.lambda_now()                   # [2.0, 2.2, 2.4]
    cvs3, f3 = r3.restraint_forces(coord3, ptr3)
    # diff = 2.5 - centre = [0.5, 0.3, 0.1]; F0x = -k*diff*u_x, u_x=-1 -> [0.5,0.3,0.1]
    fx = np.array([f3[b][0, 0] for b in range(3)])
    ok &= np.allclose(fx, [0.5, 0.3, 0.1], atol=1e-12)
    ok &= np.allclose(r3.lambda_now(), [2.0, 2.2, 2.4], atol=1e-12)
    print(f"(3) per-pull F0x={fx} (exp [0.5, 0.3, 0.1])  centres independent")

    print("BATCHED-MOVING-RESTRAINT SELF-CHECK:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
