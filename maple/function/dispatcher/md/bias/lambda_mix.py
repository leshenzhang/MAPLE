"""Batch-aware two-state lambda-mixing bias for MAPLE's batched MD core
(rides ``ensemble/nvt_batched.py`` exactly like ``bias.batched.BatchedHarmonicRestraint``).

Thermodynamic Integration / MBAR need the LINEAR two-state coupling

    E_lambda(x) = (1 - lambda) E_A(x) + lambda E_B(x)
    F_lambda(x) = (1 - lambda) F_A(x) + lambda F_B(x)

with N_lambda windows placed on the BATCH axis: replica ``b`` carries its own
``lambda_b`` and each ``BatchedNVT`` step advances all windows with ONE
``calc.get_ef_gpu()`` -- the batched forward IS end-state A (the base MLIP model,
so E_A/F_A arrive already in the padded ``(B, nmax_dof)`` buffer), and this bias
adds the lambda-mixing correction toward end-state B at the force-assembly hook
(``BatchedNVT._forces_au`` -> ``bias.apply(E, F, calc)``):

    E   += lambda_b (E_B - E_A)                         => (1-lambda)E_A + lambda E_B
    F_b += lambda_b (F_B - F_A)   [a.u.]                => (1-lambda)F_A + lambda F_B

Because the buffer already holds (E_A, F_A), the mix is expressed as the *additive
delta* ``lambda (B - A)`` -- the SAME "add my per-replica force into the buffer"
contract every batch-aware bias obeys (missing bias -> exact no-op). Per window it
records ``E_A``, ``E_B`` and ``dU/dlambda = E_B - E_A`` (the only quantities TI +
MBAR consume; see :mod:`bias.ti_mbar`).

End-state B is any object exposing ``energy_forces(coord_np, ptr) -> (E_B (B,) Ha,
forces_B list[(n_b,3) Ha/Angstrom])``. The default toy is :class:`HarmonicEndState`
-- a per-atom harmonic tether to each replica's reference geometry (the alchemical
"Einstein-crystal" end state; analytic, gradient-exact, cheap, needs no second
forward). A second MLIP model can be dropped in the same slot (that variant costs
one extra forward per step -- still one batched forward over ALL lambda windows).

ponytail: ONLY the linear two-state mix (no soft-core / no alchemical topology
transformation); end-state B is an analytic tether (no second MLIP forward in the
core path). Isolated (non-periodic) replicas -- the batch backends carry no cell.
"""

import numpy as np

from ..utils import HA_PER_ANG_TO_AU          # Ha/Angstrom -> Ha/Bohr (a.u. force)
from .ti_mbar import linear_mix_energy, linear_mix_force   # endpoint helpers (reused)


def _ptr_to_np(ptr):
    """Coerce a batch calc's per-replica atom-offset ``_ptr`` (B+1,) to numpy.

    UMA's ``_ptr`` is a CUDA torch tensor (``np.asarray`` on it raises), MACE-OFF's
    is already numpy -- mirror ``bias.batched._ptr_to_np`` (C2 CUDA-safe coercion)."""
    if hasattr(ptr, "detach"):                 # torch tensor (CPU or CUDA)
        return ptr.detach().to("cpu").numpy()
    return np.asarray(ptr)


class HarmonicEndState:
    """Toy analytic end-state B: a per-atom harmonic tether to a reference geometry.

        E_B(x) = sum_i 1/2 kappa |x_i - x0_i|^2         [Hartree]
        F_B,i  = -kappa (x_i - x0_i)                     [Ha/Angstrom]

    This is the standard alchemical "Einstein crystal" reference: analytic,
    gradient-exact, and (unlike a second MLIP) needs no extra forward. ``ref`` is
    the per-replica reference positions (list of (n_b,3) Angstrom), ``kappa`` in
    Ha/Angstrom^2 (scalar broadcast to all replicas, or one per replica)."""

    KIND = "harmonic_einstein"

    def __init__(self, ref_positions, kappa, ptr=None):
        self._ref = [np.asarray(r, np.float64).reshape(-1, 3) for r in ref_positions]
        self.B = len(self._ref)
        self.kappa = np.broadcast_to(np.asarray(kappa, np.float64),
                                     (self.B,)).astype(np.float64)
        self._n = [r.shape[0] for r in self._ref]

    def energy_forces(self, coord_np, ptr):
        """Return ``(E_B (B,) Ha, forces_B list[(n_b,3) Ha/Angstrom])``.

        coord_np : (N_atoms,3) Angstrom (replicas concatenated); ptr : (B+1,)."""
        E_B = np.empty(self.B, dtype=np.float64)
        forces = []
        for b in range(self.B):
            pos = np.asarray(coord_np[ptr[b]:ptr[b + 1]], np.float64).reshape(-1, 3)
            d = pos - self._ref[b]                        # (n_b,3)
            E_B[b] = 0.5 * self.kappa[b] * float(np.sum(d * d))
            forces.append(-self.kappa[b] * d)             # Ha/Angstrom
        return E_B, forces


class BatchedLambdaMix:
    """Per-replica linear two-state lambda-mixing on the BatchedNVT bias hook.

    Parameters
    ----------
    atoms_list : list[ase.Atoms]
        The B replicas (used for per-replica atom counts).
    lambdas : sequence[float]
        Per-replica coupling ``lambda_b`` in [0,1] (length B). lambda=0 -> pure
        end-state A, lambda=1 -> pure end-state B.
    end_state_B : object
        Exposes ``energy_forces(coord_np, ptr) -> (E_B (B,) Ha, forces list)``
        (e.g. :class:`HarmonicEndState`, or a second MLIP wrapper).
    """

    KIND = "lambda_mix"

    def __init__(self, atoms_list, lambdas, end_state_B):
        self.B = len(atoms_list)
        if self.B == 0:
            raise ValueError("BatchedLambdaMix needs >= 1 replica.")
        self.lambdas = np.asarray(lambdas, np.float64).reshape(-1)
        if self.lambdas.shape[0] != self.B:
            raise ValueError(f"lambdas length {self.lambdas.shape[0]} != B={self.B}")
        self._n = [len(at) for at in atoms_list]
        self.end_state_B = end_state_B
        # per-window histories (one entry per force evaluation).
        self.eA_history = [[] for _ in range(self.B)]     # E_A(x)          [Ha]
        self.eB_history = [[] for _ in range(self.B)]     # E_B(x)          [Ha]
        self.dudl_history = [[] for _ in range(self.B)]   # dU/dlambda=E_B-E_A [Ha]

    # ------------------------------------------------------------- bias contract
    def apply(self, E, F, calc):
        """Mix toward end-state B and log (E_A, E_B, dU/dlambda) per window.

        E : (B,) Ha == E_A (base calc energy) ; F : (B, nmax_dof) a.u. == F_A
        (modified IN PLACE to F_lambda) ; ``calc.coord`` (N_atoms,3 Angstrom) +
        ``calc._ptr`` (B+1). Returns ``(E_lambda, F_lambda)``."""
        import torch
        coord_np = calc.coord.detach().to("cpu").numpy()
        ptr = _ptr_to_np(calc._ptr)
        E_A = np.asarray(E.detach().to("cpu").numpy(), np.float64).reshape(-1)   # (B,)
        E_B, forces_B = self.end_state_B.energy_forces(coord_np, ptr)
        lam = self.lambdas
        dE = np.zeros(self.B, dtype=np.float64)
        for b in range(self.B):
            n = self._n[b]
            fA_au = F[b, :3 * n]                                    # a.u. (Ha/Bohr)
            fB_au = torch.as_tensor(forces_B[b].reshape(-1) * HA_PER_ANG_TO_AU,
                                    dtype=F.dtype, device=F.device)
            # F_lambda = F_A + lambda (F_B - F_A) = (1-lambda)F_A + lambda F_B
            F[b, :3 * n] = fA_au + float(lam[b]) * (fB_au - fA_au)
            dudl_b = float(E_B[b] - E_A[b])
            dE[b] = float(lam[b]) * dudl_b                          # E_lambda - E_A
            self.eA_history[b].append(float(E_A[b]))
            self.eB_history[b].append(float(E_B[b]))
            self.dudl_history[b].append(dudl_b)
        E = E + torch.as_tensor(dE, dtype=E.dtype, device=E.device)
        return E, F


# --------------------------------------------------------------------------- #
# Standalone numpy self-check (no torch / no GPU; mirrors bias.batched):
#   PYTHONSAFEPATH=1 python -m maple.function.dispatcher.md.bias.lambda_mix
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    _mixE, _mixF = linear_mix_energy, linear_mix_force     # run via `python -m ...`

    ok = True

    # (1) HarmonicEndState energy/force vs analytic on B=2 replicas of H2.
    ref = [np.array([[0.0, 0, 0], [1.0, 0, 0]]),
           np.array([[0.0, 0, 0], [1.0, 0, 0]])]
    es = HarmonicEndState(ref, kappa=2.0)
    coord = np.array([[0.0, 0, 0], [1.3, 0, 0],       # replica 0: atom1 displaced +0.3x
                      [0.0, 0, 0], [1.0, 0, 0]])       # replica 1: at reference
    ptr = np.array([0, 2, 4])
    E_B, fB = es.energy_forces(coord, ptr)
    ok &= abs(E_B[0] - 0.5 * 2.0 * 0.3 ** 2) < 1e-12      # 1/2 k d^2
    ok &= abs(E_B[1]) < 1e-12
    ok &= abs(fB[0][1, 0] - (-2.0 * 0.3)) < 1e-12         # -k*d on atom1 x
    print(f"(1) E_B={E_B} (exp [{0.5*2*0.09:.3f}, 0.0])  F_B[0][1,0]={fB[0][1,0]:.3f}"
          f" (exp {-0.6:.3f})")

    # (2) endpoint coupling: lambda=0 -> E_A/F_A ; lambda=1 -> E_B/F_B.
    E_A, E_Bv = 3.0, 5.0
    F_A = np.array([1.0, -2.0, 0.5])
    F_Bv = np.array([-1.0, 0.0, 2.0])
    e0, e1 = _mixE(E_A, E_Bv, 0.0), _mixE(E_A, E_Bv, 1.0)
    f0, f1 = _mixF(F_A, F_Bv, 0.0), _mixF(F_A, F_Bv, 1.0)
    half = _mixE(E_A, E_Bv, 0.5)
    ok &= abs(e0 - E_A) < 1e-12 and abs(e1 - E_Bv) < 1e-12
    ok &= np.allclose(f0, F_A) and np.allclose(f1, F_Bv)
    ok &= abs(half - 4.0) < 1e-12                          # midpoint average
    print(f"(2) lambda=0 E={e0}(exp {E_A})  lambda=1 E={e1}(exp {E_Bv})  "
          f"lambda=0.5 E={half}(exp 4.0)  force endpoints ok")

    # (3) lambdas-length guard.
    try:
        BatchedLambdaMix([1, 2, 3], [0.0, 1.0], es); ok = False   # 3 replicas, 2 lambdas
    except ValueError:
        pass
    print(f"(3) lambdas-length guard ok={ok}")

    print("LAMBDA-MIX SELF-CHECK:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
