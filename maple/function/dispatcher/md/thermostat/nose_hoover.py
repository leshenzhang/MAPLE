"""
Nosé-Hoover chains (NHC) thermostat for NVT molecular dynamics.

This is a DETERMINISTIC, time-reversible canonical velocity-space map: unlike the
stochastic langevin / v-rescale thermostats it injects no random noise, so it
preserves the dynamical correlations (diffusion, VACF, vibrational spectra) that
stochastic thermostats distort, while still sampling the canonical (NVT)
distribution.  A *chain* of M coupled thermostats (Martyna, Klein & Tuckerman
1992) is used instead of a single Nosé-Hoover variable: a single NH thermostat is
famously non-ergodic for stiff / few-mode systems (e.g. the harmonic oscillator),
whereas a chain restores ergodicity.

Algorithm
---------
The thermostat acts purely on the velocities (no forces, no positions, no cell).
Each call to :meth:`apply` advances the extended NHC sub-system by *half* the MD
timestep using the symmetric, time-reversible Trotter factorisation of Martyna,
Tuckerman, Tobias & Klein, Mol. Phys. 87, 1117 (1996), with a Suzuki-Yoshida
decomposition (``n_yoshida`` weights) and ``n_respa`` multiple-timestep
sub-steps for the (stiff) chain integration.  The ensemble layer wraps a full
velocity-Verlet step between two half :meth:`apply` calls, giving the standard
symmetric NHC-VV splitting

    exp(iL dt) = exp(iL_NHC dt/2) exp(iL_VV dt) exp(iL_NHC dt/2).

Chain masses follow Martyna 1992 in the GROMACS ``tau_t`` (coupling-period)
convention:

    Q_0 = N_f k_B T tau^2 ,   Q_i = k_B T tau^2   (i = 1 .. M-1).

Conserved quantity
-------------------
The dynamics conserve the extended ("bath") Hamiltonian

    H~ = H_phys + Σ_i ½ Q_i v_{η_i}²  +  N_f k_B T η_0  +  k_B T Σ_{i≥1} η_i ,

where η_i, v_{η_i} are the chain positions/velocities.  :meth:`apply` returns the
current bath energy (the chain part of H~) so the ensemble loop can log H~ and
verify its drift (the NHC analogue of v-rescale's work ledger).  H~ is computed
from the chain STATE (not an accumulated work) precisely so that its small
discretisation drift remains observable.

References
----------
Martyna, Klein & Tuckerman, J. Chem. Phys. 97, 2635 (1992).
Martyna, Tuckerman, Tobias & Klein, Mol. Phys. 87, 1117 (1996).
"""

import numpy as np
from ase import Atoms
from typing import Optional

from ..utils import AMU_TO_AU, FS_TO_AU, KELVIN_TO_HARTREE


def _suzuki_yoshida_weights(n: int) -> np.ndarray:
    """Symmetric Suzuki-Yoshida weights w_j (Σ w_j = 1) for orders 1, 3, 5, 7."""
    if n == 1:
        return np.array([1.0])
    if n == 3:
        w1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
        return np.array([w1, 1.0 - 2.0 * w1, w1])
    if n == 5:
        w1 = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        return np.array([w1, w1, 1.0 - 4.0 * w1, w1, w1])
    if n == 7:
        # Suzuki 1990 / Yoshida 1990 7-term coefficients (symmetric).
        w1 = 0.784513610477560
        w2 = 0.235573213359357
        w3 = -1.17767998417887
        w0 = 1.0 - 2.0 * (w1 + w2 + w3)
        return np.array([w3, w2, w1, w0, w1, w2, w3])
    raise ValueError(f"n_yoshida must be one of {{1,3,5,7}}, got {n}")


class NoseHooverChain:
    """Deterministic Nosé-Hoover chains thermostat (canonical velocity-space map).

    Mirrors the LangevinThermostat / VRescaleThermostat interface:
    ``__init__(atoms, temperature, tau_t, timestep, n_dof, ...)``,
    ``set_temperature(T)`` annealing hook, and ``apply(velocities)`` returning
    ``(v_new, bath_energy)``.  ``apply`` advances the chain by *half* a timestep,
    so it is called twice per MD step (symmetric NHC-VV split).
    """

    def __init__(
        self,
        atoms: Atoms,
        temperature: float,
        tau_t: float,
        timestep: float,
        n_dof: int,
        chain_length: int = 3,
        n_respa: int = 1,
        n_yoshida: int = 3,
        rng: Optional[np.random.Generator] = None,  # unused; NHC is deterministic
    ):
        """
        Parameters
        ----------
        atoms : ase.Atoms
            Molecular system.
        temperature : float
            Target temperature in Kelvin.
        tau_t : float
            Thermostat coupling time τ in fs (GROMACS convention; sets the chain
            masses Q = N_f k_B T τ²).  Recommended 50-500 fs for MLP MD.
        timestep : float
            MD timestep in fs (the FULL step; apply() uses dt/2 internally).
        n_dof : int
            Number of physical degrees of freedom N_f (from the central DOF
            policy).  Must be supplied so the canonical target is correct.
        chain_length : int
            Number of thermostats M in the chain (>=1; default 3, Martyna 1996).
            M=1 reduces to a single Nosé-Hoover (non-ergodic for stiff systems).
        n_respa : int
            Number of RESPA sub-steps for the chain integration (default 1).
        n_yoshida : int
            Suzuki-Yoshida order for the chain integration (1, 3, 5 or 7;
            default 3).
        rng : ignored
            Accepted for interface symmetry with the stochastic thermostats; the
            NHC map is fully deterministic.
        """
        self.atoms = atoms
        self.temperature = float(temperature)
        self.tau_t = tau_t * FS_TO_AU         # fs -> a.u.
        self.timestep = timestep * FS_TO_AU   # fs -> a.u. (full step)
        self.masses = atoms.get_masses() * AMU_TO_AU
        self.n_dof = int(n_dof)

        self.M = int(chain_length)
        if self.M < 1:
            raise ValueError(f"chain_length must be >= 1, got {self.M}")
        self.n_respa = max(1, int(n_respa))
        self._ys_w = _suzuki_yoshida_weights(int(n_yoshida))
        self.n_yoshida = len(self._ys_w)

        self._kT = self.temperature * KELVIN_TO_HARTREE
        self._init_chain_masses()

        # Extended-system state: chain positions η_i and velocities v_{η_i}.
        self.eta = np.zeros(self.M, dtype=np.float64)
        self.v_eta = np.zeros(self.M, dtype=np.float64)

    # ------------------------------------------------------------------ masses
    def _init_chain_masses(self) -> None:
        """Q_0 = N_f k_B T τ² ; Q_i = k_B T τ²  (Martyna 1992)."""
        tau2 = self.tau_t ** 2
        Q = np.full(self.M, self._kT * tau2, dtype=np.float64)
        Q[0] = self.n_dof * self._kT * tau2
        self.Q = Q

    def set_temperature(self, temperature: float) -> None:
        """Update the target temperature (K) and rescale kT and the chain masses
        Q ∝ kT for a simulated-annealing schedule (mirrors langevin / v-rescale).
        Chain positions/velocities are preserved; strict H~ conservation does not
        apply across an anneal step (as for any time-dependent thermostat)."""
        self.temperature = float(temperature)
        self._kT = self.temperature * KELVIN_TO_HARTREE
        self._init_chain_masses()

    # ------------------------------------------------------------------ energy
    def bath_energy(self) -> float:
        """Current chain contribution to the conserved quantity H~ (Hartree):
        Σ ½ Q_i v_{η_i}² + N_f k_B T η_0 + k_B T Σ_{i≥1} η_i."""
        ke_chain = 0.5 * float(np.dot(self.Q, self.v_eta ** 2))
        pe_chain = self.n_dof * self._kT * self.eta[0] + self._kT * float(np.sum(self.eta[1:]))
        return ke_chain + pe_chain

    # ------------------------------------------------------------------ propagate
    def _integrate_chain(self, akin: float, dt: float) -> float:
        """Symmetric NHC propagator over time ``dt`` (Martyna 1996 NHCINT).

        ``akin`` is twice the particle kinetic energy (Σ m v² = 2·KE).  Returns the
        scalar factor by which the particle velocities must be multiplied; the
        chain state (eta, v_eta) is advanced in place.
        """
        M, Q, kT, Nf = self.M, self.Q, self._kT, self.n_dof
        v_eta, eta = self.v_eta, self.eta

        G = np.empty(M, dtype=np.float64)
        G[0] = (akin - Nf * kT) / Q[0]
        for i in range(1, M):
            G[i] = (Q[i - 1] * v_eta[i - 1] ** 2 - kT) / Q[i]

        scale = 1.0
        for _ in range(self.n_respa):
            for w in self._ys_w:
                wdt = w * dt / self.n_respa
                wdt2, wdt4, wdt8 = 0.5 * wdt, 0.25 * wdt, 0.125 * wdt

                # --- backward sweep: last chain velocity, then M-2 .. 0 ---
                v_eta[M - 1] += G[M - 1] * wdt4
                for k in range(M - 2, -1, -1):
                    aa = np.exp(-wdt8 * v_eta[k + 1])
                    v_eta[k] = v_eta[k] * aa * aa + wdt4 * G[k] * aa

                # --- scale particle velocities; track akin = 2·KE ---
                aa = np.exp(-wdt2 * v_eta[0])
                scale *= aa
                akin *= aa * aa

                # --- chain positions ---
                eta += v_eta * wdt2

                # --- forward sweep: G[0] recompute, then 0 .. M-2 ---
                G[0] = (akin - Nf * kT) / Q[0]
                for k in range(M - 1):
                    aa = np.exp(-wdt8 * v_eta[k + 1])
                    v_eta[k] = v_eta[k] * aa * aa + wdt4 * G[k] * aa
                    G[k + 1] = (Q[k] * v_eta[k] ** 2 - kT) / Q[k + 1]
                v_eta[M - 1] += G[M - 1] * wdt4

        return scale

    def apply(self, velocities: np.ndarray) -> tuple[np.ndarray, float]:
        """Advance the chain by half an MD step and rescale the velocities.

        Called twice per MD step (around the velocity-Verlet step) to realise the
        symmetric NHC-VV Trotter splitting.

        Parameters
        ----------
        velocities : np.ndarray
            Current velocities in atomic units, shape (N_atoms, 3).

        Returns
        -------
        tuple[np.ndarray, float]
            ``(v_new, bath_energy)`` where ``bath_energy`` is the current chain
            contribution to the conserved quantity H~ (Hartree).
        """
        akin = float(np.sum(self.masses[:, np.newaxis] * velocities ** 2))  # = 2·KE
        scale = self._integrate_chain(akin, 0.5 * self.timestep)
        return velocities * scale, self.bath_energy()
