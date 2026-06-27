"""
LFMiddle Langevin thermostat for NVT molecular dynamics.

This module implements the thermostat-only Ornstein-Uhlenbeck (OU) velocity
update used in the LFMiddle formulation discussed by Leimkuhler & Matthews,
J. Chem. Phys. 138, 174102 (2013), and by Zhang et al., J. Phys. Chem. A 123,
6056-6079 (2019):

    v' = c1 * v + c2 * xi
    c1 = exp(-gamma * dt)
    c2 = sqrt((1 - c1**2) * k_B * T / m)
    xi ~ N(0,1)

The surrounding LFMiddle splitting is handled outside this thermostat. In the
current codebase, the full sequence is assembled in the ensemble layer using
LF-middle carried velocities rather than exposing the textbook BAOAB half-step
notation directly in this module. This module does not apply COM or angular-
motion projection. If runtime motion projection is desired, it must be handled
by the caller or by an ensemble-level policy outside this thermostat.

References
----------
Leimkuhler & Matthews, J. Chem. Phys. 138, 174102 (2013).
Zhang et al., J. Phys. Chem. A 123, 6056-6079 (2019).
"""

import numpy as np
from ase import Atoms
from typing import Optional

from ..utils import AMU_TO_AU, FS_TO_AU, KELVIN_TO_HARTREE


class LangevinThermostat:
    """
    LFMiddle Langevin thermostat.

    Applies the thermostat-only Ornstein-Uhlenbeck velocity update used by the
    LFMiddle formulation. The complete splitting is assembled in the ensemble
    layer; this class only implements the OU thermostat substep for the
    LF-middle carried-velocity representation. This class does not apply COM or
    angular-motion projection; if such runtime projection is desired, it
    remains a caller- or ensemble-level policy outside this thermostat.
    """

    def __init__(
        self,
        atoms: Atoms,
        temperature: float,
        friction: float,
        timestep: float,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Parameters
        ----------
        atoms : ase.Atoms
            Molecular system.
        temperature : float
            Target temperature in Kelvin.
        friction : float
            Friction coefficient γ in 1/fs.
        timestep : float
            MD timestep in fs.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.
        """
        self.atoms       = atoms
        self.temperature = temperature
        # NOTE: `timestep` is converted to atomic units below. The matching unit
        # treatment for `friction` must be kept consistent so that γ·dt remains
        # dimensionless in the OU factor exp(-γ·dt); verify carefully if this
        # line is ever changed.
        self.friction    = friction / FS_TO_AU           # 1/fs → 1/a.u.
        self.timestep    = timestep * FS_TO_AU           # fs   → a.u.
        self.masses      = atoms.get_masses() * AMU_TO_AU  # amu → a.u.
        self.rng         = rng if rng is not None else np.random.default_rng()

        # Motion projection is handled by the ensemble-level central policy.
        # Precompute OU coefficients for the LFMiddle thermostat step:
        #   c1 = exp(-γ dt)
        #   c2 = sqrt((1 - c1²) k_B T / m)
        kT        = self.temperature * KELVIN_TO_HARTREE
        self._c1  = np.exp(-self.friction * self.timestep)
        self._c2  = np.sqrt((1.0 - self._c1**2) * kT / self.masses)

    def apply(self, velocities: np.ndarray) -> np.ndarray:
        """
        Apply the LFMiddle Ornstein-Uhlenbeck thermostat step to velocities.

        This method does not apply COM or angular-motion projection; any such
        runtime constraint is the caller's responsibility.

            v' = c1 * v + c2 * xi,   xi ~ N(0, 1)

        Any runtime COM or angular projection is applied by the ensemble-level
        central motion policy after the thermostat step.

        Parameters
        ----------
        velocities : np.ndarray
            Current velocities in atomic units, shape (N_atoms, 3).

        Returns
        -------
        np.ndarray
            Thermostatted velocities in atomic units, shape (N_atoms, 3).
        """
        noise = self.rng.standard_normal(velocities.shape)
        v_new = self._c1 * velocities + self._c2[:, np.newaxis] * noise
        return v_new
