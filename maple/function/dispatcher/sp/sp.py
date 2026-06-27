from typing import Union, List, Optional
from dataclasses import dataclass

import numpy as np
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

        # Per-frame results (Hartree / Hartree.A^-1), populated by _run_trajectory.
        self.energies_hartree: List[float] = []
        self.forces_hartree: List[Optional[np.ndarray]] = []

    def run(self):
        if self.is_trajectory:
            self._run_trajectory()
        else:
            self._run_single()

    def _run_single(self):
        """Original single-point calculation logic (Hartree-native)."""
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

    def _gradient_lines(self, atoms: Atoms,
                        forces: Optional[np.ndarray] = None) -> list:
        """Return per-atom energy gradients for detailed SP output.

        ``forces`` (Hartree/Angstrom), when provided, is used directly -- e.g. the
        per-frame forces unpacked from ONE batched forward -- so the calculator is
        not re-invoked. When None, falls back to ``atoms.get_forces()`` (Hartree-
        native: f75 calculators convert backend forces to Hartree/A inside
        ``calculate()``, so no eV->Hartree rescale is applied here).
        """
        if forces is None:
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

    def _trajectory_frame_lines(self, idx: int, atoms_frame: Atoms,
                                energy_hartree: float,
                                forces: Optional[np.ndarray] = None) -> list:
        """Return trajectory-frame SP result lines for the selected verbosity.

        ``forces`` (Hartree/Angstrom) is the precomputed per-frame force from the
        batched/serial compute pass; it is threaded into ``_gradient_lines`` for
        verbose>=1 output so the calculator is never re-called per frame.
        """
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
            lines.extend(self._gradient_lines(atoms_frame, forces=forces))
        lines.append("=" * 80 + "\n")
        return lines

    # ------------------------------------------------------------------ batch
    def _get_batch_calc(self):
        """Return a batch-capable calculator if the trajectory carries one.

        Duck-typed: a batch calculator exposes ``prepare(atoms_list, fixed_nmax)``
        and ``get_ef_gpu() -> (E (B,), F (B, nmax_dof))`` in Hartree (mirrors
        UMABatchCalc / AIMNet2BatchCalc / MACE*BatchCalc). All frames share one
        calculator in the engine flow, so atoms[0] is representative. Returns
        ``None`` for plain ASE calculators -> serial fallback.
        """
        if not self.atoms:
            return None
        calc = getattr(self.atoms[0], "calc", None)
        if (calc is not None
                and callable(getattr(calc, "prepare", None))
                and callable(getattr(calc, "get_ef_gpu", None))):
            return calc
        return None

    def _compute_batched(self, calc) -> List[float]:
        """ONE prepare + ONE forward for the whole trajectory; unpack per frame.

        Eliminates the redundant per-frame graph/nblist rebuild of the serial
        path: ``prepare`` fixes the (block-diagonal) batched topology once and a
        single ``get_ef_gpu`` evaluates all B frames together (the D1.2 win).
        Forces come back padded as (B, nmax_dof) with per-molecule layout (atom a
        -> cols 3a..3a+2); we unpad each frame's leading 3*n_i columns into
        (n_i, 3).

        UMA/AIMNet/MACE batch calcs return Hartree directly (their get_ef_gpu
        applies EV2HARTREE internally), so -- like the serial path -- NO eV->
        Hartree rescale is applied here.
        """
        # One topology fix + one batched forward (reuses the nblist for all frames).
        calc.prepare(self.atoms, fixed_nmax=None)
        E_Ha, F_Ha = calc.get_ef_gpu()  # E (B,) Ha ; F (B, nmax_dof) Ha.A^-1

        E_cpu = E_Ha.detach().to("cpu")
        F_cpu = F_Ha.detach().to("cpu")

        energies_hartree: List[float] = []
        forces_hartree: List[Optional[np.ndarray]] = []
        for i, atoms_frame in enumerate(self.atoms):
            n_i = len(atoms_frame)
            energies_hartree.append(float(E_cpu[i].item()))
            # Unpad: atom a of frame i lives in row i, columns 3a..3a+2.
            f_i = F_cpu[i, : 3 * n_i].reshape(n_i, 3).numpy()
            forces_hartree.append(f_i)

        self.energies_hartree = energies_hartree
        self.forces_hartree = forces_hartree
        return energies_hartree

    def _compute_serial(self) -> List[float]:
        """Serial fallback for plain (non-batch) ASE calculators.

        Hartree-native: f75 calculators convert backend energy/force to Hartree
        inside ``calculate()``, so ``get_potential_energy()`` already returns
        Hartree and ``get_forces()`` Hartree/Angstrom -- NO eV->Hartree rescale.
        Forces are only fetched when verbose>=1 needs them (saves a force eval
        on energy-only runs).
        """
        energies_hartree: List[float] = []
        forces_hartree: List[Optional[np.ndarray]] = []
        for atoms_frame in self.atoms:
            energies_hartree.append(atoms_frame.get_potential_energy())
            if self.verbose >= 1:
                try:
                    forces_hartree.append(atoms_frame.get_forces())
                except Exception:
                    forces_hartree.append(None)
            else:
                forces_hartree.append(None)
        self.energies_hartree = energies_hartree
        self.forces_hartree = forces_hartree
        return energies_hartree

    def _run_trajectory(self):
        """Process multiple structures: ONE batched forward when the calculator
        supports it, else a serial fallback. The compute path is chosen ONCE,
        energies (and verbose>=1 forces) are precomputed, then a single output
        loop reuses f75's frame formatter -- the calculator is never re-called
        inside the loop.
        """
        with timer("Single Point Energy Calculation (Trajectory)"):
            n_frames = len(self.atoms)
            self.log_info([f"\nProcessing {n_frames} structures from trajectory...\n"])

            # --- pick compute path ONCE; precompute energies (+forces if verbose>=1) ---
            batch_calc = self._get_batch_calc()
            if batch_calc is not None:
                self.log_info([
                    f"Batched single-point: 1 prepare + 1 forward over "
                    f"{n_frames} frames ({type(batch_calc).__name__}).\n"
                ])
                energies_hartree = self._compute_batched(batch_calc)
            else:
                self.log_info(["Serial single-point (non-batch calculator).\n"])
                energies_hartree = self._compute_serial()

            self.log_info(["=" * 80 + "\n"])

            # --- single output loop (f75 formatter; precomputed E/F, no re-call) ---
            for idx, (atoms_frame, energy_hartree) in enumerate(
                zip(self.atoms, energies_hartree), start=1
            ):
                forces = self.forces_hartree[idx - 1] if self.verbose >= 1 else None
                self.log_info(self._trajectory_frame_lines(
                    idx, atoms_frame, energy_hartree, forces=forces))

            # Summary (always shown)
            self.log_info([f"\n{' SUMMARY ':=^80}\n"])
            self.log_info([f"Total frames processed: {n_frames}\n"])
            self.log_info([f"Energy range: {min(energies_hartree):.10f} to {max(energies_hartree):.10f} Hartree\n"])
            energy_span = max(energies_hartree) - min(energies_hartree)
            self.log_info([f"Energy span: {energy_span:.10f} Hartree ({energy_span * 627.509:.4f} kcal/mol)\n"])
            self.log_info(["=" * 80 + "\n"])
