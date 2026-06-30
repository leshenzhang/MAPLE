# -*- coding: utf-8 -*-
"""Native, differentiable OBC-II (Onufriev-Bashford-Case GB-II) implicit solvent.

This module is the *physics core* of MAPLE's ML-GBSA implicit-solvent stack:
real OBC-II / HCT Born radii (the proper pairwise descreening integral, NOT the
legacy heuristic Gaussian-volume Psi), the generalized-Born (Still) polar energy,
and a differentiable ACE surface-area nonpolar term.

It is intentionally *unit-locked to OpenMM's GBSAOBCForce* so it can be
reference-matched bit-for-bit-ish:
  * lengths internal = nanometres (caller passes coords / radii in Angstrom),
  * energy           = kJ/mol via ONE_4PI_EPS0 = 138.935456 kJ*nm/mol/e^2,
  * dielectric offset = 0.009 nm, probe radius = 0.14 nm,
  * OBC-II constants alpha=1.0, beta=0.8, gamma=4.85 (igb=5 / OBC2),
  * ACE nonpolar prefactor = 4*pi*surfaceAreaEnergy (default 28.3919551).

Everything is plain torch and fully autograd-differentiable w.r.t. coordinates,
so the GB force is just ``-autograd.grad(E, coords)`` at FIXED point charges
(classical ML-GBSA assumption: charges parameterise the GB-polar Coulomb term;
the gas-phase energy comes from the MLIP, not from any MM force field).

All functions accept an optional leading batch dimension, so a whole trajectory
of frames vectorises to ``[B, N, 3]`` -> ``[B]`` with a single ``[B, N, N]``
pairwise tensor (this is the per-frame batched-rescoring axis; the GB term is a
cheap O(N^2) torch add, never a second MLIP forward).
"""
from __future__ import annotations

import torch

# --- OpenMM-locked constants (nm / kJ-mol / e) ---
ONE_4PI_EPS0 = 138.935456          # kJ * nm / mol / e^2   (OpenMM)
KJ_PER_MOL_TO_HARTREE = 1.0 / 2625.4996394798254
ANG_TO_NM = 0.1
DIELECTRIC_OFFSET_NM = 0.009       # OBC/Amber dielectric offset
PROBE_RADIUS_NM = 0.14             # ACE probe radius
ACE_SURFACE_AREA_ENERGY = 2.25936  # kJ/mol/nm^2 (OpenMM default surfaceAreaEnergy)

# OBC-II (igb=5 / OBC2) screening polynomial constants
OBC2_ALPHA = 1.0
OBC2_BETA = 0.8
OBC2_GAMMA = 4.85
# OBC-I (igb=2) for completeness
OBC1_ALPHA = 0.8
OBC1_BETA = 0.0
OBC1_GAMMA = 2.909


def obc_born_radii(
    coords_nm: torch.Tensor,
    radii_nm: torch.Tensor,
    scale: torch.Tensor,
    *,
    offset: float = DIELECTRIC_OFFSET_NM,
    alpha: float = OBC2_ALPHA,
    beta: float = OBC2_BETA,
    gamma: float = OBC2_GAMMA,
) -> torch.Tensor:
    """Effective OBC Born radii (nm), differentiable w.r.t. ``coords_nm``.

    Replicates OpenMM ``ReferenceObc::computeBornRadii`` exactly: the
    Hawkins-Cramer-Truhlar pairwise descreening integral followed by the OBC
    tanh rescaling.

    Args:
        coords_nm: ``[..., N, 3]`` coordinates in nm.
        radii_nm:  ``[N]`` or ``[..., N]`` intrinsic (mbondi) radii in nm.
        scale:     ``[N]`` or ``[..., N]`` HCT screening factors (prmtop ``screen``).

    Returns:
        ``[..., N]`` effective Born radii in nm.
    """
    N = coords_nm.shape[-2]
    dtype = coords_nm.dtype
    dev = coords_nm.device

    rho = radii_nm - offset                       # offset radius rho_i  [...,N]
    sr = rho * scale                              # scaled offset radius  [...,N]

    # pairwise distances; protect the diagonal so 1/r and log are finite, then mask.
    diff = coords_nm.unsqueeze(-2) - coords_nm.unsqueeze(-3)   # [...,N,N,3]
    r = torch.linalg.norm(diff, dim=-1)                       # [...,N,N]
    eye = torch.eye(N, dtype=torch.bool, device=dev)
    r_safe = r + eye.to(dtype)                                # diagonal -> 1 nm (dummy)

    rho_i = rho.unsqueeze(-1)            # [...,N,1]
    sr_j = sr.unsqueeze(-2)             # [...,1,N]
    rsr = r_safe + sr_j                 # r + scaledRadiusJ

    # L_ij = max(rho_i, |r - sr_j|),  U_ij = r + sr_j
    L = torch.maximum(rho_i, torch.abs(r_safe - sr_j))
    U = rsr
    linv = 1.0 / L
    uinv = 1.0 / U
    linv2 = linv * linv
    uinv2 = uinv * uinv

    term = (
        linv - uinv
        + 0.25 * r_safe * (uinv2 - linv2)
        + (0.5 / r_safe) * torch.log(uinv / linv)
        + (0.25 * sr_j * sr_j / r_safe) * (linv2 - uinv2)
    )

    # fully-engulfed correction: rho_i < sr_j - r
    engulf = (rho_i < (sr_j - r_safe))
    term = term + torch.where(engulf, 2.0 * (1.0 / rho_i - linv),
                              torch.zeros((), dtype=dtype, device=dev))

    # contribute only when rho_i < r + sr_j and i != j
    contrib = (rho_i < rsr) & (~eye)
    term = term * contrib.to(dtype)

    I = 0.5 * term.sum(dim=-1)                    # descreening integral  [...,N]
    psi = rho * I                                 # OBC psi = offsetRadius * I
    poly = alpha * psi - beta * psi * psi + gamma * psi * psi * psi
    inv_born = (1.0 / rho) - torch.tanh(poly) / radii_nm
    born = 1.0 / inv_born
    return born


def gb_polar_energy(
    coords_nm: torch.Tensor,
    charges: torch.Tensor,
    born_nm: torch.Tensor,
    *,
    solute_dielectric: float = 1.0,
    solvent_dielectric: float = 78.5,
) -> torch.Tensor:
    """Generalized-Born (Still) polar solvation energy in kJ/mol.

    ``E = -0.5 * keps * ONE_4PI_EPS0 * sum_{i,j} q_i q_j / f_GB`` over the FULL
    double sum (the i==j term is the Born self energy q_i^2/R_i). This matches
    OpenMM ``GBSAOBCForce`` (preFactor = -ONE_4PI_EPS0*keps, loop i<=j with the
    diagonal halved, which equals the -0.5 full-sum form here).
    """
    keps = (1.0 / solute_dielectric) - (1.0 / solvent_dielectric)

    diff = coords_nm.unsqueeze(-2) - coords_nm.unsqueeze(-3)
    r2 = (diff * diff).sum(dim=-1)                # [...,N,N]; diagonal = 0
    bi = born_nm.unsqueeze(-1)
    bj = born_nm.unsqueeze(-2)
    alpha2 = bi * bj
    fgb = torch.sqrt(r2 + alpha2 * torch.exp(-r2 / (4.0 * alpha2)))
    qi = charges.unsqueeze(-1)
    qj = charges.unsqueeze(-2)
    e = (qi * qj) / fgb                            # [...,N,N]
    return -0.5 * keps * ONE_4PI_EPS0 * e.sum(dim=(-1, -2))


def ace_sa_energy(
    radii_nm: torch.Tensor,
    born_nm: torch.Tensor,
    *,
    surface_area_energy: float = ACE_SURFACE_AREA_ENERGY,
    probe: float = PROBE_RADIUS_NM,
) -> torch.Tensor:
    """ACE nonpolar surface-area energy in kJ/mol (OpenMM GBSAOBCForce term).

    ``E_SA = 4*pi*gamma * sum_i (r_i + probe)^2 (r_i / R_i)^6``.
    """
    factor = 4.0 * torch.pi * surface_area_energy
    r = radii_nm + probe
    ratio6 = (radii_nm / born_nm) ** 6
    return factor * (r * r * ratio6).sum(dim=-1)


def gbsa_energy_hartree(
    coords_ang: torch.Tensor,
    charges: torch.Tensor,
    radii_nm: torch.Tensor,
    scale: torch.Tensor,
    *,
    solute_dielectric: float = 1.0,
    solvent_dielectric: float = 78.5,
    surface_area_energy: float = ACE_SURFACE_AREA_ENERGY,
    include_sa: bool = True,
    alpha: float = OBC2_ALPHA,
    beta: float = OBC2_BETA,
    gamma: float = OBC2_GAMMA,
    return_components: bool = False,
):
    """Total OBC-II GBSA energy in **Hartree** from coordinates in **Angstrom**.

    Differentiable w.r.t. ``coords_ang`` (keep it a leaf with requires_grad to get
    the GB force as ``-autograd.grad``). Supports an optional leading batch axis
    on ``coords_ang`` (``[B, N, 3]`` -> ``[B]``).
    """
    coords_nm = coords_ang * ANG_TO_NM
    born = obc_born_radii(coords_nm, radii_nm, scale, alpha=alpha, beta=beta, gamma=gamma)
    e_pol = gb_polar_energy(coords_nm, charges, born,
                            solute_dielectric=solute_dielectric,
                            solvent_dielectric=solvent_dielectric)
    e_sa = (ace_sa_energy(radii_nm, born, surface_area_energy=surface_area_energy)
            if include_sa else torch.zeros_like(e_pol))
    e_kj = e_pol + e_sa
    e_ha = e_kj * KJ_PER_MOL_TO_HARTREE
    if return_components:
        return e_ha, {
            "born_nm": born,
            "e_polar_kjmol": e_pol,
            "e_sa_kjmol": e_sa,
            "e_total_kjmol": e_kj,
        }
    return e_ha
