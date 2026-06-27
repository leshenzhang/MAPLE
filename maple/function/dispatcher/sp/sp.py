from typing import Union, List, Optional
from dataclasses import dataclass

from ase import Atoms

from ..jobABC import JobABC
from maple.function.timer import timer

@dataclass
class SPParams:
    """Parameters for Single Point calculation."""
    verbose: int = 0  # 0=coordinates+energy+charge/mult, 1=+gradients

class SinglePoint(JobABC):

    def __init__(self, output: str, atoms: Union[Atoms, List[Atoms]],
                 paras: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        self.is_trajectory = isinstance(atoms, list)

        # Initialize params
        self.params = self._init_params(SPParams, paras, ("sp", "SP"))
        self.verbose = self.params.verbose

    def run(self):
        if self.is_trajectory:
            self._run_trajectory()
        else:
            self._run_single()

    def _run_single(self):
        """Original single-point calculation logic."""
        with timer("Single Point Energy Calculation"):
            energy = self.atoms.get_potential_energy()
            self.log_info(self._single_energy_lines(energy))

    def _single_energy_lines(self, energy: float) -> list:
        """Return single-structure SP result lines for the selected verbosity."""
        lines = ["\n"]
        lines.extend(self._charge_mult_lines(self.atoms))
        lines.append(f"Energy: {energy:.10f} Hartree\n")
        if self.verbose >= 1:
            lines.extend(self._gradient_lines(self.atoms))
        return lines

    def _charge_mult_lines(self, atoms: Atoms) -> list:
        """Return charge and multiplicity metadata lines for SP output."""
        charge = atoms.info.get('charge', 0)
        mult = atoms.info.get('mult', 1)
        return [f"Charge: {charge}, Multiplicity: {mult}\n"]

    def _gradient_lines(self, atoms: Atoms) -> list:
        """Return per-atom energy gradients for detailed SP output."""
        forces = atoms.get_forces()
        gradients = -forces
        symbols = atoms.get_chemical_symbols()
        lines = [
            "\nGradients (Hartree/Angstrom):\n",
            "  Gradient = -Force\n",
            "  Atom  El"
            "        Gx              Gy              Gz\n",
        ]
        for i, (sym, gradient) in enumerate(zip(symbols, gradients), start=1):
            lines.append(
                f"  {i:<4} {sym:<2}"
                f" {gradient[0]:>15.8f} {gradient[1]:>15.8f} {gradient[2]:>15.8f}\n"
            )
        return lines

    def _trajectory_frame_lines(self, idx: int, atoms_frame: Atoms, energy_hartree: float) -> list:
        """Return trajectory-frame SP result lines for the selected verbosity."""
        lines = [
            f"\n{('Frame ' + str(idx)):=^80}\n",
        ]
        lines.extend(self._charge_mult_lines(atoms_frame))
        lines.extend([
            f"Energy: {energy_hartree:.10f} Hartree\n\n",
            "Coordinates (Angstrom):\n",
        ])
        symbols = atoms_frame.get_chemical_symbols()
        positions = atoms_frame.get_positions()
        for i, (sym, pos) in enumerate(zip(symbols, positions), start=1):
            lines.append(f"  {i:<4} {sym:<2} {pos[0]:>15.8f} {pos[1]:>15.8f} {pos[2]:>15.8f}\n")
        if self.verbose >= 1:
            lines.extend(self._gradient_lines(atoms_frame))
        lines.append("=" * 80 + "\n")
        return lines

    def _run_trajectory(self):
        """Process multiple structures sequentially."""
        with timer("Single Point Energy Calculation (Trajectory)"):
            n_frames = len(self.atoms)
            self.log_info([f"\nProcessing {n_frames} structures from trajectory...\n"])
            self.log_info(["=" * 80 + "\n"])

            energies_hartree = []

            for idx, atoms_frame in enumerate(self.atoms, start=1):
                # Calculate energy
                energy_hartree = atoms_frame.get_potential_energy()
                energies_hartree.append(energy_hartree)

                self.log_info(self._trajectory_frame_lines(idx, atoms_frame, energy_hartree))

            # Summary (always shown)
            self.log_info([f"\n{' SUMMARY ':=^80}\n"])
            self.log_info([f"Total frames processed: {n_frames}\n"])
            self.log_info([f"Energy range: {min(energies_hartree):.10f} to {max(energies_hartree):.10f} Hartree\n"])
            energy_span = max(energies_hartree) - min(energies_hartree)
            self.log_info([f"Energy span: {energy_span:.10f} Hartree ({energy_span * 627.509:.4f} kcal/mol)\n"])
            self.log_info(["=" * 80 + "\n"])
