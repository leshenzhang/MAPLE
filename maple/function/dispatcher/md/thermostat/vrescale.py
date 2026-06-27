"""
V-rescale (stochastic velocity rescaling) thermostat for NVT molecular dynamics.

V-rescale corrects the Berendsen thermostat by adding a stochastic term to the
kinetic energy update, ensuring the canonical (NVT) distribution is sampled
exactly. The kinetic energy after rescaling follows a chi-squared distribution
with N_f degrees of freedom.

Algorithm (Bussi et al., 2007, Eq. A7):
    At each step, velocities are uniformly scaled by α where α² is drawn from:

        α² = e^{-Δt/τ}
           + (K̄/(N_f·K))·(1 - e^{-Δt/τ})·(R₁² + Σᵢ₌₂^{N_f} Rᵢ²)
           + 2·e^{-Δt/(2τ)}·sqrt(K̄/(N_f·K)·(1 - e^{-Δt/τ}))·R₁

    where K̄ = (N_f/2)·kT is the target kinetic energy, K is the current
    kinetic energy, N_f is the number of degrees of freedom, and
    R₁, R₂, ..., R_{N_f} are independent standard normal variates.

Notes:
    - Produces the correct canonical ensemble unlike plain Berendsen rescaling.
    - No per-atom friction; global kinetic energy is rescaled uniformly.
    - τ → 0 approaches very strong stochastic coupling; this is not the same as
      a deterministic isokinetic constraint and should not be described as one.
    - τ → ∞ reduces to NVE (no coupling).

Reference:
    Bussi, Donadio & Parrinello, J. Chem. Phys. 126, 014101 (2007).
"""

import numpy as np
from ase import Atoms
from typing import Optional

from ..utils import AMU_TO_AU, FS_TO_AU, KELVIN_TO_HARTREE


class VRescaleThermostat:
    """
    Stochastic velocity rescaling thermostat (V-rescale).

    Rescales all velocities by a global factor α each step, where the new
    kinetic energy is drawn from the correct canonical distribution.
    """

    def __init__(
        self,
        atoms: Atoms,
        temperature: float,
        tau_t: float,
        timestep: float,
        rng: Optional[np.random.Generator] = None,
        n_dof: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        atoms : ase.Atoms
            Molecular system
        temperature : float
            Target temperature in Kelvin
        tau_t : float
            Temperature coupling time constant in fs.
            Larger τ_T = weaker coupling.  Recommended: 100–500 fs for MLP.
        timestep : float
            MD timestep in fs
        rng : np.random.Generator, optional
            Random number generator for reproducibility
        """
        self.atoms = atoms
        self.temperature = temperature
        self.tau_t = tau_t * FS_TO_AU       # fs → a.u.
        self.timestep = timestep * FS_TO_AU # fs → a.u.
        self.masses = atoms.get_masses() * AMU_TO_AU
        self.rng = rng if rng is not None else np.random.default_rng()

        n_atoms = len(atoms)
        # Runtime N_dof should be provided by the central DOF policy; fall back to
        # the legacy rule only for not-yet-migrated callers.
        self._n_dof = n_dof if n_dof is not None else (3 * n_atoms if any(atoms.pbc) else 3 * n_atoms - 3)
        self._kT_target = temperature * KELVIN_TO_HARTREE
        self._ke_target = 0.5 * self._n_dof * self._kT_target

        # exp(-dt/τ_T) precomputed
        self._decay = np.exp(-self.timestep / self.tau_t)

        # Whether the system is periodic.  For non-periodic systems the caller
        # should project out the COM velocity before applying the thermostat so
        # that the input kinetic energy lives in the intended 3N-3 subspace.
        self._is_periodic = any(atoms.pbc)

    def _sample_chi2(self, n: int) -> float:
        """
        Sample from χ²(n) distribution as the exact sum of n squared normals.

        This implementation always uses the direct sum-of-squares form rather
        than a large-n normal approximation.
        """
        if n <= 0:
            return 0.0
        if n == 1:
            w = self.rng.standard_normal()
            return w * w
        # Use sum of squares directly (exact, vectorised)
        w = self.rng.standard_normal(n)
        return float(np.dot(w, w))

    def apply(self, velocities: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Apply one V-rescale step: globally rescale velocities.

        Implements Eq. A7 of Bussi, Donadio & Parrinello, J. Chem. Phys.
        126, 014101 (2007):

            α² = e^{-Δt/τ} + (K̄/(N_f·K))·(1 - e^{-Δt/τ})·(R₁² + Σᵢ₌₂^{N_f} Rᵢ²)
               + 2·e^{-Δt/(2τ)}·sqrt(K̄/(N_f·K)·(1 - e^{-Δt/τ}))·R₁

        where K̄ = (N_f/2)·kT is the target kinetic energy, K is the current
        kinetic energy, and R₁...R_{N_f} are independent standard normals.

        Also returns the thermostat work ΔW = (α² − 1)·K for computing the
        conserved energy H̃ (Bussi 2007, Eq. 15).

        Parameters
        ----------
        velocities : np.ndarray
            Current velocities in atomic units, shape (N_atoms, 3)

        Returns
        -------
        tuple[np.ndarray, float]
            (rescaled_velocities, delta_w) where delta_w = (α² − 1)·K
            is the energy injected by the thermostat this step (Hartree).
        """
        ke = 0.5 * np.sum(self.masses[:, np.newaxis] * velocities ** 2)

        if ke < 1e-30:
            return velocities, 0.0

        f = self._decay                     # e^{-Δt/τ}
        ke_ref = self._ke_target            # K̄ = N_f/2 · kT
        n_dof = self._n_dof                 # N_f

        one_minus_f = 1.0 - f               # 1 - e^{-Δt/τ}
        c = ke_ref / (n_dof * ke)           # K̄ / (N_f · K)

        # R₁ (shared between cross and quadratic terms)
        r1 = self.rng.standard_normal()

        # Σᵢ₌₂^{N_f} Rᵢ²  ~  χ²(N_f - 1)
        sum_r2 = self._sample_chi2(n_dof - 1)

        # Eq. A7: α² = f + c·(1-f)·(R₁² + Σ Rᵢ²) + 2·sqrt(f)·sqrt(c·(1-f))·R₁
        alpha_sq = (f
                    + c * one_minus_f * (r1 * r1 + sum_r2)
                    + 2.0 * np.sqrt(f) * np.sqrt(c * one_minus_f) * r1)
        alpha_sq = max(alpha_sq, 0.0)

        alpha = np.sqrt(alpha_sq)
        v_new = alpha * velocities

        # Thermostat work per step: ΔW_k = (α²_k − 1)·K_k
        #
        # This is the standard Bussi 2007 definition.  The caller is responsible
        # for projecting out COM motion (for isolated systems) before invoking
        # apply(), so that K is already the kinetic energy in the intended 3N-3
        # subspace.
        delta_w = (alpha_sq - 1.0) * ke

        return v_new, delta_w
