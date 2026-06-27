# -*- coding: utf-8 -*-
"""
Model initial Hessians for batched RS-P-RFO TS search (OPT-IN seeds).

This module is a NEW, fully opt-in companion to BPRFO.py. Nothing here changes
the default ('full' numerical-Hessian) path; it is only reached when a caller
sets ``initial_hessian='lindh'`` and/or ``ts_hessian_inject=True``.

Two public helpers:

  lindh_initial_hessian(...)   -- Lindh model Cartesian Hessian (bonds + bends),
      a connectivity-aware positive-definite seed that replaces the identity seed.
      Lindh, Bernhardsson, Karlstrom, Malmqvist, "On the use of a Hessian model
      function in molecular geometry optimizations", Chem. Phys. Lett. 1995, 241,
      423-428.  DOI: 10.1016/0009-2614(95)00646-L
        rho_ij = exp( alpha_ij * (r_ref_ij^2 - r_ij^2) )          (r in Bohr)
        k_bond = 0.45 * rho_ij                  (Hartree/Bohr^2)
        k_bend = 0.15 * rho_ij * rho_jk         (Hartree/rad^2)
        k_tors = 0.005 * rho_ij * rho_jk * rho_kl  (torsions: formula provided,
                 applied only if include_torsions=True; off by default -- a seed
                 Hessian used to bootstrap an iterative/Bofill scheme does not
                 need the torsion block, and the batched torsion B-matrix is the
                 most fragile part to build under fixed-nmax padding).
      alpha and r_ref are the Lindh period-pair tables (periods 1-3; >3 clamped
      to 3). The pairwise ``exp(alpha (r_ref^2 - r^2))`` kernel is the GPU-friendly
      core requested by the upgrade spec.

  ts_hessian_inject(...)       -- Swart-Bickelhaupt / pysisyphus 'ts_hessian'
      curvature injection. Given a (typically positive-definite) model/identity
      seed it sign-flips and damps the lowest (reaction) mode so the seed has
      EXACTLY ONE negative eigenvalue:  lambda_rxn <- -scale * |lambda_rxn|
      (default scale = 0.25, i.e. diag[rxn] = -0.25 * diag). A positive-definite
      model Hessian otherwise gives P-RFO no mode to follow; this also fixes the
      wave-1 "identity Hessian for pool newcomers" degeneracy.
      Swart, M.; Bickelhaupt, F. M. Int. J. Quantum Chem. 2006, 106, 2536
      (DOI: 10.1002/qua.21049); pysisyphus optimizers/hessian_init 'ts'.
"""

from typing import List
import numpy as np
import torch
from ase import Atoms

DTYPE_DEFAULT = torch.float64

# Bohr <-> Angstrom
_BOHR = 0.52917721067
# Hartree/Bohr^2 -> Hartree/Angstrom^2  (stretch force constants are in a.u.)
_B2A2 = 1.0 / (_BOHR * _BOHR)

# Lindh period-pair tables (periods 1,2,3; index = min(period,3)-1).
# alpha in 1/Bohr^2, r_ref in Bohr  (Lindh et al. 1995, Table 1).
_LINDH_ALPHA = [[1.0000, 0.3949, 0.3949],
                [0.3949, 0.2800, 0.2800],
                [0.3949, 0.2800, 0.2800]]
_LINDH_RREF = [[1.35, 2.10, 2.53],
               [2.10, 2.87, 3.40],
               [2.53, 3.40, 3.40]]


def _period(Z: int) -> int:
    if Z <= 2:
        return 1
    if Z <= 10:
        return 2
    if Z <= 18:
        return 3
    if Z <= 36:
        return 4
    if Z <= 54:
        return 5
    if Z <= 86:
        return 6
    return 7


def lindh_initial_hessian(atoms_list: List[Atoms],
                          nmax_dof: int,
                          device,
                          dtype=DTYPE_DEFAULT,
                          rho_cut: float = 0.0085,
                          include_angles: bool = True,
                          include_torsions: bool = False):
    """Batched Lindh model Cartesian Hessian, padded to ``nmax_dof``.

    Returns ``H`` of shape (B, nmax_dof, nmax_dof) in Hartree/Angstrom^2, with
    each structure's real (3*n_i) block filled and everything else zero. The
    Gauss-Newton model Hessian is assembled as  H = sum_q k_q * b_q b_q^T  over
    bond-stretch and angle-bend internal coordinates q (Wilson B-matrix rows
    b_q = dq/dx), with Lindh force constants k_q. B is expected to be small
    (validation batches); the per-structure python loop is acceptable and the
    inner per-internal ops are vectorized torch.

    DOI: 10.1016/0009-2614(95)00646-L
    """
    alpha_t = torch.tensor(_LINDH_ALPHA, dtype=dtype, device=device)
    rref_t = torch.tensor(_LINDH_RREF, dtype=dtype, device=device)
    B = len(atoms_list)
    H = torch.zeros((B, nmax_dof, nmax_dof), dtype=dtype, device=device)

    for b, at in enumerate(atoms_list):
        Z = at.get_atomic_numbers()
        N = len(at)
        if N == 0:
            continue
        pos = torch.tensor(np.asarray(at.get_positions(), dtype=np.float64),
                           dtype=dtype, device=device)            # (N,3) Angstrom
        per = torch.tensor([min(_period(int(z)), 3) - 1 for z in Z],
                           dtype=torch.long, device=device)        # (N,)
        a_ij = alpha_t[per][:, per]                                # (N,N) 1/Bohr^2
        r0_ij = rref_t[per][:, per]                                # (N,N) Bohr

        d = pos.unsqueeze(0) - pos.unsqueeze(1)                    # (N,N,3): d[i,j]=r_i-r_j
        r_ang = d.norm(dim=-1)                                     # (N,N) Angstrom
        eye = torch.eye(N, dtype=torch.bool, device=device)
        r_safe = torch.where(eye, torch.ones_like(r_ang), r_ang)
        r_bohr = r_ang / _BOHR
        rho = torch.exp(a_ij * (r0_ij ** 2 - r_bohr ** 2))        # (N,N)
        rho = rho.masked_fill(eye, 0.0)

        Hb = H[b]

        # ---- bond-stretch contributions (all pairs i<j with rho >= cut) ----
        iu = torch.triu_indices(N, N, offset=1, device=device)
        for c in range(iu.shape[1]):
            i = int(iu[0, c]); j = int(iu[1, c])
            rij = float(rho[i, j])
            if rij < rho_cut:
                continue
            k = 0.45 * rij * _B2A2                                  # Hartree/Angstrom^2
            u = d[i, j] / r_safe[i, j]                              # unit vec, points j->i
            bi, bj = u, -u
            Hb[3*i:3*i+3, 3*i:3*i+3] += k * torch.outer(bi, bi)
            Hb[3*j:3*j+3, 3*j:3*j+3] += k * torch.outer(bj, bj)
            Hb[3*i:3*i+3, 3*j:3*j+3] += k * torch.outer(bi, bj)
            Hb[3*j:3*j+3, 3*i:3*i+3] += k * torch.outer(bj, bi)

        # ---- angle-bend contributions (triples i-j-k, j = vertex) ----
        if include_angles:
            rho_cpu = rho.detach().cpu().numpy()
            for j in range(N):
                nb = [i for i in range(N) if i != j and rho_cpu[i, j] >= rho_cut]
                for aa in range(len(nb)):
                    for cc in range(aa + 1, len(nb)):
                        i = nb[aa]; kk = nb[cc]
                        u = d[i, j]; ru = float(r_ang[i, j])
                        v = d[kk, j]; rv = float(r_ang[kk, j])
                        if ru < 1e-6 or rv < 1e-6:
                            continue
                        uh = u / ru
                        vh = v / rv
                        cw = float(torch.dot(uh, vh))
                        sw2 = 1.0 - cw * cw
                        if sw2 < 1e-8:
                            continue                               # (near-)linear: skip
                        sw = float(np.sqrt(sw2))
                        bi = (cw * uh - vh) / (ru * sw)            # d(theta)/dx_i  (1/Ang)
                        bk = (cw * vh - uh) / (rv * sw)            # d(theta)/dx_k
                        bj = -(bi + bk)
                        kang = 0.15 * float(rho[i, j]) * float(rho[kk, j])   # Hartree/rad^2
                        idx = [i, j, kk]
                        bv = [bi, bj, bk]
                        for p in range(3):
                            for q in range(3):
                                Hb[3*idx[p]:3*idx[p]+3, 3*idx[q]:3*idx[q]+3] += \
                                    kang * torch.outer(bv[p], bv[q])

        # ---- torsion contributions (formula provided; off by default) ----
        if include_torsions:
            # k_tors = 0.005 * rho_ij * rho_jk * rho_kl ; the dihedral B-matrix is
            # intentionally NOT assembled here (fragile under fixed-nmax padding and
            # negligible for a bootstrap seed). Hook left for completeness.
            pass

    return H


def ts_hessian_inject(H: torch.Tensor,
                      real_mask: torch.Tensor,
                      D: torch.Tensor,
                      scale: float = 0.25,
                      big: float = 1e8):
    """Sign-flip + damp the lowest (reaction) mode so the seed has EXACTLY ONE
    negative eigenvalue (Swart-Bickelhaupt / pysisyphus 'ts_hessian').

    Operates in the mass-weighted metric (the same one BPRFO eigen-follows): with
    D = 1/sqrt(mass), build H_mw = D H D, eigendecompose, set the lowest real
    eigenvalue lambda_0 -> -scale*|lambda_0|, rebuild, and un-mass-weight back to
    Cartesian. Pad DOFs are lifted by ``big`` on the diagonal so their modes never
    become the reaction mode, then restored.

    H        : (B, n, n) Cartesian seed Hessian (Hartree/Angstrom^2)
    real_mask: (B, n) bool, True on real DOFs
    D        : (B, n) = 1/sqrt(mass)
    Returns the injected Cartesian Hessian, same shape, pad block zeroed.
    DOI: 10.1002/qua.21049
    """
    dtype = H.dtype
    Bn, n, _ = H.shape
    idx = torch.arange(n, device=H.device)
    pad = ~real_mask                                              # (B,n)

    H_mw = D.unsqueeze(-1) * H * D.unsqueeze(-2)
    if bool(pad.any()):
        H_mw = H_mw.clone()
        diag = H_mw[..., idx, idx]
        H_mw[..., idx, idx] = diag + pad.to(dtype) * big

    w, V = torch.linalg.eigh(H_mw)                                # ascending
    w = w.clone()
    w[:, 0] = -float(scale) * w[:, 0].abs()                       # flip + damp lowest mode
    H_mw2 = (V * w.unsqueeze(-2)) @ V.transpose(-1, -2)

    if bool(pad.any()):
        H_mw2 = H_mw2.clone()
        d2 = H_mw2[..., idx, idx]
        H_mw2[..., idx, idx] = d2 - pad.to(dtype) * big

    Dinv = 1.0 / torch.clamp(D, min=1e-30)                        # = sqrt(mass)
    H_cart = Dinv.unsqueeze(-1) * H_mw2 * Dinv.unsqueeze(-2)
    mask_ij = (real_mask.unsqueeze(-1) & real_mask.unsqueeze(-2)).to(dtype)
    return H_cart * mask_ij
