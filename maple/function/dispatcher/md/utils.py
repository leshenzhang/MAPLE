"""
Utility functions for molecular dynamics simulations.

This module provides essential calculations for MD:
- Temperature from velocities
- Kinetic energy calculations
- Velocity initialization from Maxwell-Boltzmann distribution
- Instantaneous pressure calculation
- XYZ trajectory writing utilities
"""

import warnings

import numpy as np
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from typing import Dict, Optional, Tuple


# ========== Physical Constants and Unit Conversions ==========

# Temperature conversions
KELVIN_TO_HARTREE = 3.1668114e-6  # k_B in Hartree/K
HARTREE_TO_KELVIN = 1.0 / KELVIN_TO_HARTREE

# Mass conversions
AMU_TO_AU = 1822.888486209  # atomic mass unit to atomic units

# Time conversions
FS_TO_AU = 41.341374575751  # femtoseconds to atomic units
AU_TO_FS = 1.0 / FS_TO_AU

# Energy conversions
HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

# Length conversions
# NIST CODATA 2018: 1 Bohr = 0.529177210903 Å (exact to 12 sig. fig.)
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# Force conversions
# MAPLE calculators (AIMNet2, MACE, UMA) return forces in Ha/Å.
# The MD integrator needs Ha/Bohr (atomic units).
# Ha/Å → Ha/Bohr:  F[Ha/Bohr] = F[Ha/Å] × (dr_Å / dr_Bohr) = F[Ha/Å] × BOHR_TO_ANGSTROM
# (1 Bohr = 0.5292 Å, so force per Bohr is smaller than force per Å)
HA_PER_ANG_TO_AU = BOHR_TO_ANGSTROM  # Ha/Å → Ha/Bohr ≈ 0.5292

# Legacy alias kept for backward compatibility (was used when forces were assumed eV/Å)
EV_PER_ANG_TO_AU = 1.0 / (27.211386245988 * BOHR_TO_ANGSTROM)  # ≈ 0.019447

# Pressure unit conversions
# Derivation: 1 eV = 1.6021766208e-19 J, 1 Å³ = 1e-30 m³ → 1 eV/Å³ = 1.6021766208e11 Pa = 1.6021766208e6 bar
EV_PER_ANG3_TO_BAR = 1.6021766208e-19 / 1e-30 * 1e-5   # eV/Å³ → bar
BAR_TO_EV_PER_ANG3 = 1.0 / EV_PER_ANG3_TO_BAR          # bar → eV/Å³

# Kinetic energy conversion for pressure calculation
# 1 amu·Å²/fs² = 1.66054e-27 kg × 1e-20 m² / 1e-30 s² = 1.66054e-17 J
# → 1.66054e-17 / 1.60218e-19 eV ≈ 103.6427 eV
AMU_ANG2_PER_FS2_TO_EV = 1.03642695e+2

# Default isothermal compressibility (liquid water at 300 K, 1 bar)
# Reference: CRC Handbook of Chemistry and Physics
DEFAULT_COMPRESSIBILITY = 4.5e-5   # 1/bar

# Velocity representation metadata
VELOCITY_REPR_STANDARD = "standard"
VELOCITY_REPR_LFMIDDLE_CARRIED = "lfmiddle_carried"
_VALID_VELOCITY_REPRESENTATIONS = {
    VELOCITY_REPR_STANDARD,
    VELOCITY_REPR_LFMIDDLE_CARRIED,
}


# ========== Core MD Calculations ==========

def is_linear_molecule(atoms: Atoms, tol: float = 1e-8) -> bool:
    """Return True if a non-periodic system is effectively linear."""
    if any(atoms.pbc):
        return False

    n_atoms = len(atoms)
    if n_atoms <= 1:
        return False
    if n_atoms == 2:
        return True

    positions = atoms.get_positions()
    masses = atoms.get_masses()
    total_mass = np.sum(masses)
    if total_mass <= 0:
        return False

    com = np.sum(masses[:, np.newaxis] * positions, axis=0) / total_mass
    centered = positions - com
    if np.linalg.matrix_rank(centered, tol=tol) <= 1:
        return True

    centered_bohr = centered * ANGSTROM_TO_BOHR
    masses_au = masses * AMU_TO_AU
    inertia = np.zeros((3, 3))
    for mi, ri in zip(masses_au, centered_bohr):
        inertia += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
    eigvals = np.sort(np.linalg.eigvalsh(inertia))
    return bool(eigvals[0] < tol * max(eigvals[-1], 1.0))


def get_initialization_dof_policy(
    atoms: Atoms,
    remove_com: bool = True,
    remove_angular: bool = False,
) -> Dict[str, object]:
    """Return initialization DOF policy for velocity generation.

    `remove_com` and `remove_angular` are parallel initialization settings.
    `remove_com` controls initialization-only COM removal. `remove_angular`
    controls initialization-only angular projection and always includes COM
    removal first.
    """
    warnings_list = []
    is_pbc = any(atoms.pbc)
    angular_active = bool(remove_angular and not is_pbc)
    if remove_angular and is_pbc:
        warnings_list.append(
            "remove_angular is ignored for periodic systems because global rigid-body rotation is not well-defined under PBC."
        )

    linear_active = bool(remove_com or angular_active)
    rotational_removed = 0
    if angular_active:
        rotational_removed = 2 if is_linear_molecule(atoms) else 3

    return {
        "atoms": atoms,
        "is_pbc": is_pbc,
        "angular_active": angular_active,
        "linear_active": linear_active,
        "rotational_dof_removed": rotational_removed,
        "warnings": warnings_list,
    }


def get_runtime_dof_policy(
    atoms: Atoms,
    remove_com_every: int = 0,
    remove_angular_every: int = 0,
) -> Dict[str, object]:
    """Return runtime DOF policy for temperature control and logging."""
    warnings_list = []
    is_pbc = any(atoms.pbc)
    angular_requested = remove_angular_every > 0
    linear_requested = remove_com_every > 0

    if is_pbc and angular_requested:
        warnings_list.append(
            "remove_angular_every is ignored for periodic systems because global rigid-body rotation is not well-defined under PBC. remove_com_every remains an independent optional runtime COM-drift removal under PBC."
        )
        angular_requested = False

    angular_active = bool(angular_requested)
    linear_active = bool(angular_active or linear_requested)
    rotational_removed = 0
    if angular_active:
        rotational_removed = 2 if is_linear_molecule(atoms) else 3

    return {
        "atoms": atoms,
        "is_pbc": is_pbc,
        "angular_active": angular_active,
        "linear_active": linear_active,
        "rotational_dof_removed": rotational_removed,
        "warnings": warnings_list,
    }


def get_n_dof_from_policy(policy: Dict[str, object], n_atoms: Optional[int] = None) -> int:
    """Convert a DOF policy dictionary into an active N_dof count."""
    if n_atoms is None:
        atoms = policy.get("atoms")
        if atoms is None:
            raise ValueError("n_atoms is required when policy does not include atoms")
        n_atoms = len(atoms)

    n_dof = 3 * n_atoms
    if policy.get("linear_active", False):
        n_dof -= 3
    n_dof -= int(policy.get("rotational_dof_removed", 0))
    return max(n_dof, 1)


def describe_dof_policy(policy: Dict[str, object]) -> str:
    """Return a short human-readable DOF description for logs/summaries."""
    if policy.get("is_pbc", False):
        if policy.get("linear_active", False):
            return "PBC: 3N - 3 (runtime COM removal)"
        return "PBC: 3N"

    rotational = int(policy.get("rotational_dof_removed", 0))
    if policy.get("angular_active", False):
        return f"isolated: 3N - 3 - {rotational} (runtime angular removal)"
    if policy.get("linear_active", False):
        return "isolated: 3N - 3 (runtime COM removal)"
    return "isolated: 3N"


def calculate_temperature(atoms: Atoms, velocities: np.ndarray, n_dof: Optional[int] = None) -> float:
    """
    Calculate instantaneous temperature from velocities.

    Uses the equipartition theorem:
        T = 2 * KE / (N_dof * k_B)

    In the migrated MD code paths, `n_dof` is supplied explicitly from the
    runtime motion/DOF policy so that temperature reporting, thermostat target
    kinetic energy, and logger summaries all use the same active subspace.
    The built-in fallback (`3N` for PBC, `3N-3` for non-PBC) is retained only
    for legacy callers that have not yet been migrated to the central policy.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system
    velocities : np.ndarray
        Atomic velocities in atomic units (Bohr/a.u. time)
        Shape: (N_atoms, 3)
    n_dof : int, optional
        Active number of degrees of freedom. When omitted, a legacy fallback is
        used (`3N` for PBC, `3N-3` for non-PBC).

    Returns
    -------
    float
        Temperature in Kelvin
    """
    masses = atoms.get_masses() * AMU_TO_AU  # Convert to atomic units
    kinetic = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)

    if n_dof is None:
        n_atoms = len(atoms)
        # Backward-compatible default until all callers migrate to explicit policy.
        if any(atoms.pbc):
            n_dof = 3 * n_atoms
        else:
            n_dof = 3 * n_atoms - 3

    if n_dof <= 0:
        return 0.0

    temperature = 2.0 * kinetic / (n_dof * KELVIN_TO_HARTREE)
    return temperature


def calculate_kinetic_energy(atoms: Atoms, velocities: np.ndarray) -> float:
    """
    Calculate total kinetic energy.

    KE = 0.5 * sum(m_i * v_i^2)

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system
    velocities : np.ndarray
        Atomic velocities in atomic units
        Shape: (N_atoms, 3)

    Returns
    -------
    float
        Kinetic energy in Hartree
    """
    masses = atoms.get_masses() * AMU_TO_AU
    kinetic = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
    return kinetic


def compute_instantaneous_pressure(
    atoms: Atoms,
    velocities: np.ndarray,
    stress_warned: bool = False,
    class_name: str = "Barostat",
) -> Tuple[float, bool]:
    """
    Compute instantaneous pressure from the virial theorem.

    P = (2*KE + W) / (3*V)

    where W = -V * (σ_xx + σ_yy + σ_zz) is the virial from the stress tensor.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system with attached calculator.
    velocities : np.ndarray
        Current velocities in atomic units (Bohr/a.u. time), shape (N_atoms, 3).
    stress_warned : bool, default=False
        Flag indicating whether stress-unavailable warning has already been issued.
        If False and stress is unavailable, a warning is emitted and the flag is
        set to True in the return value.
    class_name : str, default="Barostat"
        Class name to include in warning message.

    Returns
    -------
    tuple of (float, bool)
        (pressure_in_bar, new_stress_warned_flag)
        pressure_in_bar: Instantaneous pressure in bar.
        new_stress_warned_flag: Updated warning flag (True if warning was issued).

    Notes
    -----
    If the calculator does not support stress tensor, the ideal-gas approximation
    (W=0) is used, which underestimates pressure for dense systems.

    References
    ----------
    Allen & Tildesley, Computer Simulation of Liquids, 2nd ed. (2017), §3.3.
    """
    volume = atoms.get_volume()   # Å³

    # Kinetic contribution (in eV)
    masses_amu = atoms.get_masses()
    # v in a.u. (Bohr/a.u.time) → convert to Å/fs
    v_ang_per_fs = velocities * BOHR_TO_ANGSTROM / AU_TO_FS
    # KE in eV: 0.5 * m[amu] * v²[Å²/fs²] * (amu·Å²/fs² → eV)
    ke_ev = 0.5 * np.sum(masses_amu[:, np.newaxis] * v_ang_per_fs**2) * AMU_ANG2_PER_FS2_TO_EV

    # Virial contribution from stress tensor (eV)
    virial_ev = 0.0
    new_stress_warned = stress_warned
    try:
        stress = atoms.get_stress(voigt=True)   # eV/Å³, Voigt: xx,yy,zz,yz,xz,xy
        # Hydrostatic virial: W = -V * (σ_xx + σ_yy + σ_zz)
        virial_ev = -volume * (stress[0] + stress[1] + stress[2])
    except (PropertyNotImplementedError, RuntimeError):
        # Calculator does not support stress; fall back to ideal-gas pressure (virial = 0).
        # Warn once per barostat instance so the user is aware.
        if not stress_warned:
            warnings.warn(
                f"{class_name}: calculator does not provide a stress tensor; "
                "pressure estimated from kinetic term only (ideal-gas approximation). "
                "For accurate NPT simulations, use a calculator that supports stress.",
                UserWarning, stacklevel=2
            )
            new_stress_warned = True

    # P = (2*KE + W) / (3*V)  in eV/Å³, then convert to bar
    pressure_ev_ang3 = (2.0 * ke_ev + virial_ev) / (3.0 * volume)
    return pressure_ev_ang3 * EV_PER_ANG3_TO_BAR, new_stress_warned


def initialize_velocities(
    atoms: Atoms,
    temperature: float,
    remove_com: bool = True,
    remove_rotation: bool = False,
    remove_angular: Optional[bool] = None,
    target_n_dof: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Initialize velocities from a Maxwell-Boltzmann distribution.

    For each atom i with mass m_i at temperature T:
        v_i ~ N(0, sqrt(k_B * T / m_i))

    Initialization projection and runtime projection are intentionally distinct:
    `remove_com` / `remove_angular` act only on the initial velocity draw,
    whereas `remove_com_every` / `remove_angular_every` act during dynamics in
    the ensemble loops. To keep initialization and runtime thermodynamic targets
    consistent, callers should pass `target_n_dof` from the runtime DOF policy.
    If `target_n_dof` is omitted, a legacy fallback based on the initialization
    projection is used.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system.
    temperature : float
        Target temperature in Kelvin.
    remove_com : bool, default=True
        Initialization-only COM removal.
    remove_rotation : bool, default=False
        Legacy alias path used to support older call sites. Prefer
        `remove_angular`, which represents initialization-only COM + rigid-body
        rotation projection.
    remove_angular : bool, optional
        Initialization-only angular projection. When true, COM removal is always
        applied first and rigid-body rotation is projected out for non-periodic
        systems.
    target_n_dof : int, optional
        DOF used for the final temperature rescaling. In migrated MD paths this
        should come from the runtime policy so initialization and thermostat
        targets remain consistent.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Velocities in atomic units with shape ``(N_atoms, 3)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    if remove_angular is None:
        remove_angular = bool(remove_rotation)
    if remove_angular:
        remove_com = True
        remove_rotation = True

    kT = temperature * KELVIN_TO_HARTREE
    masses = atoms.get_masses() * AMU_TO_AU
    n_atoms = len(atoms)

    # Generate random velocities from standard normal distribution
    velocities = rng.standard_normal(size=(n_atoms, 3))

    # Scale each atom's velocity by sqrt(kT/m)
    for i, mass in enumerate(masses):
        sigma = np.sqrt(kT / mass)
        velocities[i] *= sigma

    # Remove center of mass motion, then rescale to restore target temperature.
    # COM removal reduces the number of active DOF by 3, which lowers the
    # instantaneous kinetic energy below the target; rescaling corrects this.
    # Ref: Allen & Tildesley, Computer Simulation of Liquids, 2nd ed. (2017), §3.2
    if remove_com:
        total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
        total_mass = np.sum(masses)
        velocities -= total_momentum / total_mass

    # Remove overall rigid-body rotation (non-PBC only).
    #
    # Method (Shirts 2013, §2):
    #   1. Compute angular momentum L = Σ r_i × (m_i v_i)  in the COM frame.
    #   2. Compute the inertia tensor I = Σ m_i (|r_i|² E − r_i ⊗ r_i).
    #   3. Solve ω = I⁻¹ L  for the rigid-body angular velocity.
    #   4. Subtract the rigid-rotation contribution: v_i -= ω × r_i.
    #
    # This is a linear projection onto the subspace orthogonal to the three
    # infinitesimal rotation generators; internal DOF are exactly preserved.
    if remove_rotation and not any(atoms.pbc):
        positions_au = atoms.get_positions() * ANGSTROM_TO_BOHR  # Å → Bohr

        # Step 1 — COM frame positions
        total_mass = np.sum(masses)
        com = np.sum(masses[:, np.newaxis] * positions_au, axis=0) / total_mass
        r = positions_au - com  # (N, 3)

        # Step 2 — Angular momentum
        L = np.sum(
            masses[:, np.newaxis] * np.cross(r, velocities),
            axis=0
        )  # (3,)

        # Step 3 — Inertia tensor
        I = np.zeros((3, 3))
        for mi, ri in zip(masses, r):
            I += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))

        # Step 4 — Solve for ω; use pseudoinverse to handle near-singular I
        # (e.g. linear molecules where one principal moment is ~0)
        try:
            omega = np.linalg.solve(I, L)
        except np.linalg.LinAlgError:
            omega = np.linalg.lstsq(I, L, rcond=None)[0]

        # Step 5 — Subtract rigid rotation from each atom
        velocities -= np.cross(omega, r)  # v_i -= ω × r_i

    # Rescale to exact target temperature using the runtime DOF policy.
    # Initialization projection (`remove_com` / `remove_angular`) and runtime
    # projection (`remove_com_every` / `remove_angular_every`) are parallel
    # concepts. To keep initialization and thermostat targets consistent, the
    # target DOF for velocity scaling is provided explicitly by the caller and
    # should match the runtime policy. Fall back to the initialization policy only
    # for legacy callers that do not pass target_n_dof.
    if target_n_dof is None:
        init_policy = get_initialization_dof_policy(
            atoms,
            remove_com=bool(remove_com),
            remove_angular=bool(remove_angular),
        )
        n_dof = get_n_dof_from_policy(init_policy, n_atoms=n_atoms)
    else:
        n_dof = target_n_dof
    current_ke2 = np.sum(masses[:, np.newaxis] * velocities**2)  # 2*KE
    if n_dof > 0 and current_ke2 > 0:
        actual_temp = current_ke2 / (n_dof * KELVIN_TO_HARTREE)
        velocities *= np.sqrt(temperature / actual_temp)

    return velocities


# ========== Trajectory I/O ==========

def write_xyz_frame(
    file_handle,
    atoms: Atoms,
    energy: float,
    frame_number: int,
    velocity: Optional[np.ndarray] = None,
    include_velocities: bool = False,
    rng_state: Optional[str] = None,
    velocity_representation: Optional[str] = None,
):
    """
    Write a single frame to XYZ file.

    Format:
        N_atoms
        Frame <number>  Energy = <energy> Hartree[  Cell = ...]
        Symbol  x  y  z  [vx  vy  vz]

    Parameters
    ----------
    file_handle : file object
        Opened file handle
    atoms : ase.Atoms
        Atomic system
    energy : float
        Total energy in Hartree
    frame_number : int
        Frame index
    velocity : np.ndarray, optional
        Velocities available for optional debug output (in atomic units)
        Shape: (N_atoms, 3)
    include_velocities : bool, default=False
        Whether to include velocity columns in the XYZ atom lines.
    rng_state : str, optional
        Reserved for API compatibility. Strict restart state is written only to
        RST checkpoints, never to XYZ comment lines.
    velocity_representation : str, optional
        Reserved for API compatibility. Velocity representation metadata is
        stored only in RST checkpoints, never in XYZ comment lines.
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # Header lines
    file_handle.write(f"{len(symbols)}\n")
    cell_str = ""
    if any(atoms.pbc):
        cp = atoms.cell.cellpar()  # [a, b, c, alpha, beta, gamma]
        pbc_tokens = ["T" if periodic else "F" for periodic in atoms.pbc]
        cell_str = (f"  Cell = {cp[0]:.6f} {cp[1]:.6f} {cp[2]:.6f}"
                    f" {cp[3]:.6f} {cp[4]:.6f} {cp[5]:.6f}"
                    f"  PBC = {' '.join(pbc_tokens)}")
    # frame_number stores the MD step number (not sequential frame index) so that
    # resume_simulation() can recover the exact step offset without knowing traj_every.
    file_handle.write(
        f"Frame {frame_number}  Energy = {energy:.10f} Hartree"
        f"{cell_str}\n"
    )
    # NOTE: frame_number is the MD *step* number (passed as `step` from the ensemble loop).
    # The regex _TRAJ_COMMENT_RE parses this as frame_num; resume_simulation uses it
    # directly as step_offset (no multiplication by traj_every needed).

    # Atomic coordinates (and optionally velocities)
    if include_velocities and velocity is not None:
        for symbol, (x, y, z), (vx, vy, vz) in zip(symbols, positions, velocity):
            file_handle.write(
                f"{symbol:2s} {x:15.8f} {y:15.8f} {z:15.8f}  "
                f"{vx:12.6f} {vy:12.6f} {vz:12.6f}\n"
            )
    else:
        for symbol, (x, y, z) in zip(symbols, positions):
            file_handle.write(f"{symbol:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")


def write_xyz_trajectory(
    filename: str,
    atoms_list: list,
    energies: list,
    append: bool = False
):
    """
    Write multiple frames to XYZ file.

    Parameters
    ----------
    filename : str
        Output file path
    atoms_list : list of ase.Atoms
        List of atomic configurations
    energies : list of float
        Corresponding energies in Hartree
    append : bool, default=False
        Append to existing file or overwrite
    """
    mode = 'a' if append else 'w'

    with open(filename, mode) as f:
        for i, (atoms, energy) in enumerate(zip(atoms_list, energies)):
            write_xyz_frame(f, atoms, energy, frame_number=i)


# ========== Velocity Utilities ==========

def remove_rigid_body_rotation(atoms: Atoms, velocities: np.ndarray) -> np.ndarray:
    """Project out rigid-body rotation in the center-of-mass frame."""
    if any(atoms.pbc) or len(atoms) <= 1:
        return velocities.copy()

    masses = atoms.get_masses() * AMU_TO_AU
    positions_au = atoms.get_positions() * ANGSTROM_TO_BOHR
    total_mass = np.sum(masses)
    com = np.sum(masses[:, np.newaxis] * positions_au, axis=0) / total_mass
    r = positions_au - com

    L = np.sum(masses[:, np.newaxis] * np.cross(r, velocities), axis=0)
    I = np.zeros((3, 3))
    for mi, ri in zip(masses, r):
        I += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))

    try:
        omega = np.linalg.solve(I, L)
    except np.linalg.LinAlgError:
        omega = np.linalg.lstsq(I, L, rcond=None)[0]

    return velocities - np.cross(omega, r)


def apply_runtime_motion_projection(
    atoms: Atoms,
    velocities: np.ndarray,
    step: int,
    remove_com_every: int = 0,
    remove_angular_every: int = 0,
) -> tuple[np.ndarray, str]:
    """Apply runtime COM/angular projection according to the configured cadence.

    `remove_com_every` and `remove_angular_every` are parallel settings, not
    enable/disable toggles. The former controls runtime COM removal only; the
    latter controls runtime angular projection. Under PBC, runtime COM removal
    may still be applied as an optional numerical COM-drift control, whereas
    runtime angular projection is ignored because global rigid-body rotation is
    not well-defined. If an angular projection fires, it always includes COM
    removal first and therefore supersedes COM-only removal for that step.
    """
    out = velocities.copy()
    if any(atoms.pbc):
        if remove_com_every and step % remove_com_every == 0:
            return remove_center_of_mass_motion(atoms, out), "com"
        return out, "none"

    if remove_angular_every and step % remove_angular_every == 0:
        out = remove_center_of_mass_motion(atoms, out)
        out = remove_rigid_body_rotation(atoms, out)
        return out, "angular"

    if remove_com_every and step % remove_com_every == 0:
        return remove_center_of_mass_motion(atoms, out), "com"

    return out, "none"


def remove_center_of_mass_motion(atoms: Atoms, velocities: np.ndarray) -> np.ndarray:
    """
    Remove center of mass translational motion.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system
    velocities : np.ndarray
        Atomic velocities
        Shape: (N_atoms, 3)

    Returns
    -------
    np.ndarray
        Velocities with COM motion removed
    """
    masses = atoms.get_masses() * AMU_TO_AU
    total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    total_mass = np.sum(masses)

    velocities_corrected = velocities - total_momentum / total_mass
    return velocities_corrected


def normalize_velocity_representation(representation: Optional[str]) -> str:
    """Return a supported velocity representation label."""
    if representation in _VALID_VELOCITY_REPRESENTATIONS:
        return representation
    return VELOCITY_REPR_STANDARD


def get_atoms_velocity_representation(atoms: Atoms) -> str:
    """Read velocity representation metadata from ``atoms.info``."""
    return normalize_velocity_representation(atoms.info.get("velocity_representation"))


def set_atoms_velocity_representation(atoms: Atoms, representation: str) -> str:
    """Store normalized velocity representation metadata on ``atoms.info``."""
    normalized = normalize_velocity_representation(representation)
    atoms.info["velocity_representation"] = normalized
    return normalized


def standard_to_lfmiddle_carried(
    atoms: Atoms,
    velocities: np.ndarray,
    forces: np.ndarray,
    timestep_au: float,
) -> np.ndarray:
    """
    Convert standard velocities into the carried LF-Middle velocities.

    Zhang et al. (JPCA 2019, Eq. 16/19) formulate LF-Middle with leapfrog
    carried momentum/velocity as the internal state. For a fresh start from a
    standard velocity defined at the same coordinates, the corresponding
    carried velocity is obtained by a backward half-kick:

        v_carried = v_standard - 0.5 * (F / m) * dt

    Parameters
    ----------
    atoms : Atoms
        Atomic system.
    velocities : np.ndarray
        Standard velocities in atomic units.
    forces : np.ndarray
        Forces in atomic units (Ha/Bohr).
    timestep_au : float
        Timestep in atomic units.
    """
    masses = atoms.get_masses() * AMU_TO_AU
    return velocities - 0.5 * timestep_au * forces / masses[:, np.newaxis]


def lfmiddle_carried_to_standard(
    atoms: Atoms,
    velocities: np.ndarray,
    forces: np.ndarray,
    timestep_au: float,
) -> np.ndarray:
    """
    Convert carried LF-Middle velocities back into standard velocities.

    This is the inverse of ``standard_to_lfmiddle_carried()`` at the same
    coordinates/forces:

        v_standard = v_carried + 0.5 * (F / m) * dt
    """
    masses = atoms.get_masses() * AMU_TO_AU
    return velocities + 0.5 * timestep_au * forces / masses[:, np.newaxis]


def scale_velocities_to_temperature(
    atoms: Atoms,
    velocities: np.ndarray,
    target_temperature: float
) -> np.ndarray:
    """
    Scale velocities to match target temperature.

    Useful for initialization or re-thermalization.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system
    velocities : np.ndarray
        Current velocities
    target_temperature : float
        Target temperature in Kelvin

    Returns
    -------
    np.ndarray
        Scaled velocities
    """
    current_temp = calculate_temperature(atoms, velocities)

    if current_temp < 1e-10:  # Avoid division by zero
        return velocities

    scale_factor = np.sqrt(target_temperature / current_temp)
    return velocities * scale_factor


# ========== Statistics ==========

def calculate_momentum(atoms: Atoms, velocities: np.ndarray) -> np.ndarray:
    """
    Calculate total momentum.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system
    velocities : np.ndarray
        Atomic velocities

    Returns
    -------
    np.ndarray
        Total momentum vector (3,)
    """
    masses = atoms.get_masses() * AMU_TO_AU
    total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    return total_momentum


def calculate_angular_momentum(
    atoms: Atoms,
    velocities: np.ndarray,
    origin: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate total angular momentum in atomic units.

    L = sum_i (r_i - origin) × (m_i * v_i)

    All quantities are converted to atomic units before computation:
    positions Å → Bohr, masses amu → a.u., velocities already in a.u.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system
    velocities : np.ndarray
        Atomic velocities in atomic units (Bohr/a.u. time)
    origin : np.ndarray, optional
        Reference point in Å (default: center of mass).
        Converted to Bohr internally.

    Returns
    -------
    np.ndarray
        Angular momentum vector in atomic units (3,)
    """
    masses = atoms.get_masses() * AMU_TO_AU
    positions = atoms.get_positions() * ANGSTROM_TO_BOHR  # Å → Bohr

    if origin is None:
        # Center of mass in Bohr
        origin = np.sum(masses[:, np.newaxis] * positions, axis=0) / np.sum(masses)
    else:
        origin = np.asarray(origin) * ANGSTROM_TO_BOHR

    angular_momentum = np.zeros(3)
    for mass, pos, vel in zip(masses, positions, velocities):
        r = pos - origin
        p = mass * vel
        angular_momentum += np.cross(r, p)

    return angular_momentum
