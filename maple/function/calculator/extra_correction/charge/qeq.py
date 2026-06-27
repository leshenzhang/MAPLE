import os

import torch
import numpy as np

COULOMB_EV_ANGSTROM = 14.3996454784255
QEQ_CHARGE_TOL = 1.0e-8


class QEqTorch:
    """
    GPU-compatible Charge Equilibration (QEq) solver implemented with PyTorch.
    Reads per-element parameters (electronegativity, hardness, Gaussian radius)
    from ./data/qeq.dat and computes atomic charges for an ASE Atoms object.

    Reference:
      A.K. Rappé and W.A. Goddard III, J. Phys. Chem. 95 (1991): 3358–3363.
    """

    def __init__(self, data_file="./data/qeq.dat", device="cpu", eps0=1.0):
        """
        Args:
            data_file (str): Path to QEq parameter file.
            device (str): 'cpu' or 'cuda' for GPU acceleration.
            eps0 (float): Relative dielectric scaling for the Coulomb term.
        """
        self.device = torch.device(device)
        self.eps0 = eps0
        data_path = os.path.join(os.path.dirname(__file__), data_file)
        self.params = self._load_params(data_path)

    def _load_params(self, path):
        """Read QEq params: Element, electronegativity(eV/e), hardness(eV/e²), radius(Å)."""
        table = {}
        with open(path, "r") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                elt, chi, J, radius = line.split()[:4]
                table[elt] = {"chi": float(chi), "J": float(J), "sigma": float(radius)}
        return table

    def _get_param_tensor(self, symbols, dtype=torch.float64):
        """Convert element parameters into PyTorch tensors."""
        chi, J, sigma = [], [], []
        for s in symbols:
            if s not in self.params:
                raise ValueError(f"Element {s} not found in qeq.dat")
            p = self.params[s]
            chi.append(p["chi"])
            J.append(p["J"])
            sigma.append(p["sigma"])
        return (
            torch.tensor(chi, dtype=dtype, device=self.device),
            torch.tensor(J, dtype=dtype, device=self.device),
            torch.tensor(sigma, dtype=dtype, device=self.device),
        )

    @staticmethod
    def _infer_total_charge(atoms) -> float:
        """Infer the QEq charge constraint from ASE metadata, defaulting to neutral."""
        charge = getattr(atoms, "info", {}).get("charge", None)
        if charge is not None:
            return float(charge)

        if hasattr(atoms, "get_initial_charges"):
            initial_charges = np.asarray(atoms.get_initial_charges(), dtype=float)
            if initial_charges.size and np.all(np.isfinite(initial_charges)):
                return float(initial_charges.sum())

        if hasattr(atoms, "get_initial_charge"):
            try:
                return float(atoms.get_initial_charge())
            except Exception:
                pass

        return 0.0

    def forward(self, atoms, total_charge=None):
        """
        Compute QEq charges for an ASE Atoms object.

        Args:
            atoms (ase.Atoms): ASE Atoms object.
            total_charge (float, optional): Total charge constraint.
                If None, uses atoms.info["charge"], then initial charges,
                then defaults to 0.0.

        Returns:
            torch.Tensor: Atomic charges (N,)
        """
        coords = torch.tensor(atoms.get_positions(), dtype=torch.float64, device=self.device)
        symbols = atoms.get_chemical_symbols()
        total_charge = self._infer_total_charge(atoms) if total_charge is None else float(total_charge)

        N = len(symbols)
        chi, J, sigma = self._get_param_tensor(symbols, dtype=coords.dtype)

        # Build hardness matrix and RHS vector
        H = torch.zeros((N + 1, N + 1), dtype=coords.dtype, device=self.device)
        V = torch.zeros(N + 1, dtype=coords.dtype, device=self.device)

        # Diagonal: atomic hardness
        H[:N, :N] = torch.diag(J)

        # Off-diagonal: Gaussian-screened Coulomb interaction
        rij = torch.cdist(coords, coords, p=2) + 1e-6
        a = sigma.view(-1, 1)
        b = sigma.view(1, -1)
        p = torch.sqrt(a * b / (a**2 + b**2))
        coulomb = torch.erf(p * rij) / rij

        H[:N, :N] += (COULOMB_EV_ANGSTROM / self.eps0) * (
            coulomb - torch.diag(torch.diag(coulomb))
        )

        # Charge conservation constraint
        H[N, :N] = 1.0
        H[:N, N] = 1.0
        V[:N] = chi
        # The system is solved as Hx = -V, so use -Q to enforce sum(q)=Q.
        V[N] = -total_charge

        try:
            q = torch.linalg.solve(H, -V)
        except RuntimeError as exc:
            raise ValueError(
                "QEq solve failed; check geometry, total charge, and element parameters."
            ) from exc

        if not torch.isfinite(q).all():
            raise ValueError("QEq produced non-finite charges.")

        charges = q[:-1]
        charge_error = abs(float(charges.sum().detach().cpu()) - total_charge)
        if charge_error > QEQ_CHARGE_TOL:
            raise ValueError(
                "QEq charge conservation failed: "
                f"sum={float(charges.sum().detach().cpu()):.12g}, "
                f"target={total_charge:.12g}."
            )

        return charges.detach().cpu()

    __call__ = forward
