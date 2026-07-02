import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from ase import Atoms

from . import obc as _obc

# --- physical constants ---
ANG2BOHR = 1.8897259886  # 1 Å = 1.8897 bohr
OBC_RADIUS_OFFSET_BOHR = 0.09 * ANG2BOHR  # OpenMM/Amber OBC offset: 0.009 nm

# --- default OBC-II per-element parameters (Angstrom radii + HCT screen) ---
# mbondi-style intrinsic radii (Å) and standard OBC/HCT screening factors.
# Used only by the model='obc2' path when explicit per-atom radii/screen are not
# supplied; binding-FE rescoring should pass prmtop-sourced radii/screen instead.
_MBONDI_RADIUS_ANG = {
    1: 1.20, 6: 1.70, 7: 1.55, 8: 1.50, 9: 1.50, 15: 1.85, 16: 1.80, 17: 1.70,
    35: 1.85, 53: 1.98,
}
_OBC_SCREEN = {
    1: 0.85, 6: 0.72, 7: 0.79, 8: 0.85, 9: 0.88, 15: 0.86, 16: 0.96, 17: 0.80,
    35: 0.80, 53: 0.80,
}
_DEFAULT_RADIUS_ANG = 1.50
_DEFAULT_SCREEN = 0.80


def default_obc_radii_screen(atomic_numbers):
    """Return (radii_nm, screen) numpy arrays for the given atomic numbers."""
    Z = np.asarray(atomic_numbers, dtype=int)
    radii_ang = np.array([_MBONDI_RADIUS_ANG.get(int(z), _DEFAULT_RADIUS_ANG) for z in Z])
    screen = np.array([_OBC_SCREEN.get(int(z), _DEFAULT_SCREEN) for z in Z])
    return radii_ang * 0.1, screen  # radii in nm


def load_gbsa_params(solvent="water"):
    """
    Read solvent parameters from ./data/{solvent}.dat.
    Format:
      Line 1: 8 constants (eps, molarMass, refracIndex, gamma, beta, born_scale, born_offset, reserved)
      then '# array1' and '# array2' sections with comma-separated floats.
      We only use: eps, born_scale, born_offset, array1 (shift), array2 (vdw_ref).
    """
    path = Path(__file__).parent / "data" / f"{solvent}.dat"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    header = [float(x) for x in lines[0].split(",")]
    eps, _, _, gamma, beta, born_scale, born_offset, _ = header

    allnums = []
    for l in lines[1:]:
        allnums.extend([float(x) for x in l.split(",") if x])

    half = len(allnums) // 2
    array1 = allnums[:half]   # element-wise shift (Bohr-like, from your file)
    array2 = allnums[half:]   # element-wise vdw_ref (Bohr)

    return dict(
        eps=eps,
        gamma=gamma,
        beta=beta,
        born_scale=born_scale,
        born_offset=born_offset,
        array1=array1,
        array2=array2,
    )

class GBSA(nn.Module):
    """
    GB implicit-solvent correction for MAPLE.

    Two models are available, selected with ``model=``:

    * ``model='legacy'`` (DEFAULT, unchanged): the original experimental
      GB-polar correction with a MAPLE-heuristic Gaussian-volume descreening and
      no surface-area term. Energy-only, public forces disabled
      (``SUPPORTS_FORCES = False``) -- bit-for-bit identical to the historical
      behaviour so existing energy-only paths are untouched.

    * ``model='obc2'`` (NEW): production OBC-II / HCT Born radii (proper pairwise
      descreening integral, OpenMM-locked), GB-polar (Still) energy, and a
      differentiable ACE surface-area nonpolar term. This path advertises
      ``SUPPORTS_FORCES = True``: at FIXED point charges the GB+SA force is the
      exact analytic gradient ``-autograd.grad(E, coords)`` (the ML-GBSA
      assumption -- charges parameterise the GB Coulomb term, the gas-phase
      energy comes from the MLIP, so there is no dQ/dR consistency problem).

    Input coords in Å, output energy in Hartree.
    """

    #: capability flag read by calculator_base to (un)gate solvent derivatives.
    SUPPORTS_FORCES = False

    def __init__(self, solvent="water", device="cpu", model="legacy",
                 radii_nm=None, screen=None, solvent_dielectric=None,
                 include_sa=True):
        super().__init__()
        params = load_gbsa_params(solvent)
        self.device = torch.device(device)
        self.model = str(model).lower()
        self.include_sa = bool(include_sa)

        # dielectric
        self.eps = float(params["eps"])
        # Still/GB dielectric prefactor. xTB uses keps = (1/eps - 1)/(1+alpbet).
        # 无 ALPB 时用最常见的 (1 - 1/eps)，与你之前保持一致。
        self.keps = float(1.0 - 1.0 / self.eps)

        # element parameters (in Bohr domain, as in your file)
        self.born_scale  = float(params["born_scale"])
        self.born_offset = float(params["born_offset"])
        self.register_buffer("born_shift",
                             torch.tensor(params["array1"], dtype=torch.float32, device=self.device))
        self.register_buffer("vdw_ref",
                             torch.tensor(params["array2"], dtype=torch.float32, device=self.device))

        # OBC-II constants (standard, unitless)
        # These control how burial deepness increases effective Born radii.
        self.alpha = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        self.beta  = torch.tensor(0.8, dtype=torch.float32, device=self.device)
        self.gamma = torch.tensor(4.85, dtype=torch.float32, device=self.device)

        # small eps to keep numerics stable
        self.eps_dist = torch.tensor(1e-8, dtype=torch.float32, device=self.device)

        # ---- OBC-II (production) configuration ----
        if self.model == "obc2":
            self.SUPPORTS_FORCES = True
            # OBC-II / ai-pbsa-cal recipe lock: extdiel=78.5, intdiel=1.0.
            self.solvent_dielectric = float(
                solvent_dielectric if solvent_dielectric is not None else 78.5)
            self.solute_dielectric = 1.0
            # optional fixed per-atom radii/screen (e.g. from a prmtop)
            self._radii_nm = None if radii_nm is None else np.asarray(radii_nm, dtype=float)
            self._screen = None if screen is None else np.asarray(screen, dtype=float)
        elif self.model != "legacy":
            raise ValueError(f"Unknown GBSA model {self.model!r}; use 'legacy' or 'obc2'.")

    # ---- intrinsic radii from element table (Bohr) ----
    def intrinsic_radius(self, atom_index: torch.Tensor) -> torch.Tensor:
        """
        ri0: intrinsic Born-like radius from your element tables (Bohr).
        ri0 = born_scale * (vdw_ref[Z] + born_shift[Z]) + born_offset
        """
        vdwr  = self.vdw_ref[atom_index]      # Bohr
        shift = self.born_shift[atom_index]   # Bohr
        ri0 = self.born_scale * (vdwr + shift) + self.born_offset
        # physical lower bound
        ri0 = torch.clamp(ri0, min=0.5)
        return ri0

    # ---- geometry-dependent Born radii via an OBC-like transform ----
    def compute_born_radius(self, coords_B: torch.Tensor, atom_index: torch.Tensor) -> torch.Tensor:
        """
        Effective radii (Bohr), differentiable w.r.t. coords_B.

        The tanh transform follows the OBC-II direction: increasing burial
        increases the effective Born radius.  The Psi term above remains a
        Gaussian-volume approximation, not the standard OBC descreening
        integral; production use remains disabled by default.
        """
        ri0 = self.intrinsic_radius(atom_index)          # (N,)

        # pairwise distances in Bohr
        Rij = torch.cdist(coords_B, coords_B, p=2)       # (N,N)
        Rij = torch.clamp(Rij, min=self.eps_dist)        # avoid 0

        # volumes ~ r^3
        Vi = (ri0 ** 3)                                  # (N,)
        Vj = Vi.unsqueeze(0)                             # (1,N)

        # form Psi_i = sum_{j!=i}  V_j * exp(-Rij^2 / (4 ri0_i ri0_j)) / Rij^2
        ri0i = ri0.view(-1, 1)                           # (N,1)
        ri0j = ri0.view(1, -1)                           # (1,N)
        expo = torch.exp(- (Rij**2) / (4.0 * ri0i * ri0j + self.eps_dist))
        term = (Vj * expo) / (Rij**2)
        # zero self terms
        term = term - torch.diag_embed(torch.diag(term))
        Psi = torch.sum(term, dim=1)                     # (N,)

        # OBC-like transform:
        #   R_i = 1 / (rho_i^-1 - r_i^-1 tanh(alpha*Psi-beta*Psi^2+gamma*Psi^3))
        # This preserves the physically expected radius increase with burial.
        rho = torch.clamp(ri0 - OBC_RADIUS_OFFSET_BOHR, min=0.5)
        poly = self.alpha * Psi - self.beta * (Psi**2) + self.gamma * (Psi**3)
        inv_R = (1.0 / rho) - (torch.tanh(poly) / ri0)
        Ri = 1.0 / torch.clamp(inv_R, min=1e-6)

        # lower bound to avoid degenerate f_ij
        Ri = torch.clamp(Ri, min=0.5, max=100.0)
        return Ri

    # ---- OBC-II helpers (production path) ----
    def _resolve_radii_screen(self, atoms: Atoms):
        """Per-atom (radii_nm, screen) torch tensors for the OBC-II path."""
        if self._radii_nm is not None and self._screen is not None:
            radii_nm, screen = self._radii_nm, self._screen
        else:
            radii_nm, screen = default_obc_radii_screen(atoms.get_atomic_numbers())
        rn = torch.as_tensor(radii_nm, dtype=torch.float64, device=self.device)
        sc = torch.as_tensor(screen, dtype=torch.float64, device=self.device)
        return rn, sc

    def _obc_energy(self, atoms: Atoms, return_components=False):
        q_np = atoms.atomic_charges
        if q_np is None or np.allclose(q_np, 0):
            raise ValueError("Atoms object must have nonzero partial charges.")
        q = torch.as_tensor(q_np, dtype=torch.float64, device=self.device)
        coords_A = torch.as_tensor(atoms.get_positions(), dtype=torch.float64,
                                   device=self.device)
        coords_A.requires_grad_(True)
        radii_nm, screen = self._resolve_radii_screen(atoms)
        out = _obc.gbsa_energy_hartree(
            coords_A, q, radii_nm, screen,
            solute_dielectric=self.solute_dielectric,
            solvent_dielectric=self.solvent_dielectric,
            include_sa=self.include_sa,
            return_components=return_components,
        )
        if return_components:
            e_ha, comp = out
            return e_ha, coords_A, comp
        return out, coords_A

    # ---- API ----
    def get_energy(self, atoms: Atoms):
        """
        Returns: energy (Eh), coords_A (Å, requires_grad=True)
        legacy: GB polar term only.  obc2: GB polar + ACE surface-area.
        """
        if self.model == "obc2":
            return self._obc_energy(atoms)

        # charges
        q_np = atoms.atomic_charges
        if q_np is None or np.allclose(q_np, 0):
            raise ValueError("Atoms object must have nonzero partial charges.")
        q = torch.as_tensor(q_np, dtype=torch.float32, device=self.device)  # e

        # coordinates Å as leaf
        coords_A = torch.as_tensor(atoms.get_positions(),
                                   dtype=torch.float32,
                                   device=self.device)  # (N,3) Å
        coords_A.requires_grad_(True)
        coords_B = coords_A * ANG2BOHR                      # Bohr

        # atom indices
        Z = torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        atom_index = Z - 1

        # geometry-dependent Born radii (Bohr)
        Ri = self.compute_born_radius(coords_B, atom_index)          # (N,)

        # Still f_ij
        Rij = torch.cdist(coords_B, coords_B, p=2) + self.eps_dist   # Bohr
        RiRj = Ri.view(-1, 1) * Ri.view(1, -1)                       # Bohr^2
        f_ij = torch.sqrt(Rij**2 + RiRj * torch.exp(-Rij**2 / (4.0 * RiRj + self.eps_dist)))  # Bohr

        # GB electrostatic energy
        q_i = q.view(-1, 1)
        q_j = q.view(1, -1)
        E_polar = -0.5 * self.keps * torch.sum((q_i * q_j) / f_ij)   # Eh

        return E_polar, coords_A

    def get_energy_and_force(self, atoms: Atoms):
        """Energy (Eh) and force (Eh/Å) at FIXED point charges.

        Only available on the production ``model='obc2'`` path, where the GB+SA
        energy is a proper OBC-II expression and the fixed-charge gradient is the
        physically correct ML-GBSA force.
        """
        if self.model != "obc2":
            raise NotImplementedError(
                "Experimental legacy GB-polar/QEq solvation is energy-only. Forces "
                "are disabled because MAPLE obtains those charges from "
                "geometry-dependent QEq and the heuristic GB-polar correction does "
                "not include the variational charge response dQ/dR. Use "
                "model='obc2' (production OBC-II, fixed point charges) for "
                "dynamics-capable implicit-solvent forces."
            )
        energy, coords_A = self.get_energy(atoms)
        force = -torch.autograd.grad(energy, coords_A, create_graph=False)[0]
        return energy.detach(), force.detach()

    def _debug_energy_gradient_fixed_charges(self, atoms: Atoms):
        """
        Debug-only fixed-charge gradient of the (legacy) GB-polar expression.

        This is not a MAPLE implicit-solvent force on the legacy path: it treats
        ``atoms.atomic_charges`` as externally fixed constants and omits the
        QEq charge response dQ/dR. Do not use it for optimization, MD, TS, or
        frequency workflows. Use ``model='obc2'`` + ``get_energy_and_force`` for
        a production force.
        """
        energy, coords_A = self.get_energy(atoms)
        force = -torch.autograd.grad(energy, coords_A, create_graph=False)[0]  # Eh/Å
        return energy, force
