"""
MD simulation logging and output management.

Handles:
    - Thermodynamic data output (.dat file)
    - XYZ/DCD trajectory output
    - Progress logging to main output
    - Final summary statistics
"""

import time as _time
import numpy as np
from pathlib import Path
from typing import Optional, TextIO, Any
from ase import Atoms

from .utils import (
    VELOCITY_REPR_STANDARD,
    normalize_velocity_representation,
    set_atoms_velocity_representation,
    write_xyz_frame,
)
from .rst_io import read_rst, rotate_rst_checkpoint
from .dcd_writer import DCDWriter


def _backup_file(path: Path) -> Optional[Path]:
    """
    GROMACS-style file backup: if *path* exists, rename it to
    #<name>.<ext>.1# (incrementing until a free slot is found).

    Returns the backup path if a backup was made, else None.
    """
    if not path.exists():
        return None
    n = 1
    while True:
        backup = path.parent / f"#{path.name}.{n}#"
        if not backup.exists():
            path.rename(backup)
            return backup
        n += 1


# ========== Unit Conversion Constants ==========

# 1 Hartree = 2625.4996 kJ/mol  (NIST CODATA 2018)
HARTREE_TO_KJ_PER_MOL = 2625.4996
# 1 Hartree = 627.5095 kcal/mol (1 kJ = 0.239006 kcal)
HARTREE_TO_KCAL_PER_MOL = 627.5095
# 1 fs = 1e-6 ns
FS_TO_NS = 1e-6


# ========== Progress Formatting Helpers ==========

def _fmt_duration(seconds: float) -> str:
    """Format wall-clock duration into human-readable string, e.g. '1h 23m 45s'."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _fmt_eta(seconds: float) -> str:
    """Format ETA; show '∞' when remaining time is unknown or very large."""
    if seconds <= 0 or seconds > 1e7:   # > ~4 months
        return "∞"
    return _fmt_duration(seconds)


class MDLogger:
    """
    Manages MD simulation output and logging.

    Attributes:
        output_base: Base path for output files (without extension)
        thermo_file: Thermodynamics data file handle
        traj_file: XYZ trajectory file handle
        main_output: Main output file path
        log_every: Log frequency (steps)
        traj_every: Trajectory write frequency (steps)
    """

    def __init__(
        self,
        output_path: str,
        log_every: int = 100,
        traj_every: int = 10,
        verbose: int = 1,
        traj_format: str = "xyz",
        debug: bool = False,
    ):
        """
        Initialize MD logger.

        Args:
            output_path: Main output file path (e.g., "task.out")
            log_every:   Frequency to log thermodynamic data (steps)
            traj_every:  Frequency to write trajectory frames (steps)
            verbose:     Progress verbosity level
                           0 = no progress lines (thermo file still written)
                           1 = GROMACS-style progress line every log_every steps
                           2 = verbose (step-level timing detail)
            traj_format: Trajectory output format: "xyz" (text) or "dcd" (binary)
            debug:       Enable one-shot restart/load_state diagnostics
        """
        self.main_output = output_path
        self.log_every   = log_every
        self.traj_every  = traj_every
        self.verbose      = verbose
        self.traj_format  = traj_format.lower()
        self.debug        = debug

        # Generate file paths
        base   = Path(output_path).stem
        parent = Path(output_path).parent

        self.thermo_path   = parent / f"{base}_md_thermo.dat"
        self.traj_path     = parent / f"{base}_md_traj.{self.traj_format}"
        self.summary_path  = parent / f"{base}_md_summary.txt"
        self.final_path    = parent / f"{base}_final.xyz"   # GROMACS confout.gro equivalent
        self.rst_path      = parent / f"{base}_md.rst"
        self.rst_prev_path = parent / f"{base}_md_prev.rst"

        # File handles (opened in start_simulation)
        self.thermo_file: Optional[TextIO] = None
        self.traj_file:   Any = None  # TextIO for XYZ, DCDWriter for DCD

        # Statistics tracking
        self.energies            = []
        self.temperatures        = []
        self.times               = []
        self.pressures           = []   # NPT only
        self.kinetic_energies    = []   # raw KE history
        self.potential_energies  = []   # raw PE history
        self.analysis_energies   = []   # summary/progress energy history (sync when available)
        self.analysis_temperatures = [] # summary/progress temperature history (sync when available)
        self.analysis_kinetic_energies = []  # summary/progress KE history (sync when available)
        self.analysis_label      = "raw"
        self.conserved_energies  = []   # V-rescale conserved-energy history: H̃ = H − ΣΔW_external
        self._n_atoms            = 0    # set in start_simulation
        self._is_pbc             = False  # set in start_simulation; affects fallback N_dof
        self._n_dof_override: Optional[int] = None
        self._dof_description: Optional[str] = None

        # Performance / progress tracking (set in start_simulation)
        self._n_steps:    int   = 0
        self._timestep:   float = 0.0  # fs
        self._wall_start: float = 0.0  # time.perf_counter() at simulation start
        # Wall time of the last log_step call (for rolling speed estimate)
        self._last_wall:  float = 0.0
        self._last_step:  int   = 0
        self._time_col_w: int   = 14   # width of Time(ps) column header
        self._total_ps:   float = 0.0  # total simulation time in ps

        # RNG state restored from the last trajectory frame on resume (None if not present)
        self.resumed_rng_state: Optional[str] = None
        self.resumed_velocity_representation: str = VELOCITY_REPR_STANDARD
        self.resumed_timestep: Optional[float] = None
        self.velocity_representation: str = VELOCITY_REPR_STANDARD
        # Fresh-start backup should not archive checkpoint files that are being
        # used as explicit rst_file inputs for the current run.
        self._protected_restart_inputs: set[Path] = set()

    def log_debug_initial_state(self, atoms: Atoms, velocities: np.ndarray,
                                mode: str, effective_step: int,
                                source: Optional[str] = None,
                                rst_step: Optional[int] = None,
                                velocity_representation: Optional[str] = None):
        if not self.debug:
            return

        from .utils import calculate_kinetic_energy, calculate_temperature

        forces = atoms.get_forces()
        kinetic_energy = calculate_kinetic_energy(atoms, velocities)
        potential_energy = atoms.get_potential_energy()
        temperature_inst = calculate_temperature(atoms, velocities)
        atom0_pos = atoms.get_positions()[0]
        atom0_vel = velocities[0]
        atom0_force = forces[0]

        lines = [
            "\n[MD_DEBUG_INIT]\n",
            f"  mode:             {mode}\n",
            f"  effective_step:   {effective_step}\n",
            f"  atom0_pos(A):     [{atom0_pos[0]: .10f}, {atom0_pos[1]: .10f}, {atom0_pos[2]: .10f}]\n",
            f"  atom0_vel(au):    [{atom0_vel[0]: .10f}, {atom0_vel[1]: .10f}, {atom0_vel[2]: .10f}]\n",
            f"  atom0_force(Ha/A):[{atom0_force[0]: .10f}, {atom0_force[1]: .10f}, {atom0_force[2]: .10f}]\n",
            f"  PE(Ha):           {potential_energy: .12f}\n",
            f"  KE(Ha):           {kinetic_energy: .12f}\n",
            f"  T(K):             {temperature_inst: .6f}\n",
        ]
        if source is not None:
            lines.insert(2, f"  source:           {source}\n")
        if rst_step is not None:
            insert_at = 3 if source is not None else 2
            lines.insert(insert_at, f"  rst_step:         {rst_step}\n")
        if velocity_representation is not None:
            normalized = normalize_velocity_representation(velocity_representation)
            insert_at = 4 if (source is not None and rst_step is not None) else (3 if (source is not None or rst_step is not None) else 2)
            lines.insert(insert_at, f"  velocity_repr:    {normalized}\n")

        self.log_main(lines)

    def start_simulation(self, ensemble: str, timestep: float, n_steps: int,
                        temperature: float, atoms: Atoms,
                        pressure: float = None, step_offset: int = 0,
                        velocity_representation: str = VELOCITY_REPR_STANDARD,
                        n_dof: Optional[int] = None,
                        dof_description: Optional[str] = None,
                        write_sync_thermo: bool = False,
                        write_conserved_energy: bool = False):
        """
        Initialize output files and write headers.

        Args:
            ensemble: Ensemble type (nve, nvt, npt)
            timestep: Timestep in fs
            n_steps: Number of steps to run in this segment
            temperature: Target temperature in K
            atoms: ASE Atoms object
            pressure: Target pressure in bar (NPT only)
            step_offset: Step number already completed (for resume)
            velocity_representation: Velocity semantics used for stored velocities.
            write_sync_thermo: Append sync-corrected thermo columns for
                Langevin LF-Middle carried-velocity runs.
            write_conserved_energy: Append conserved-energy columns for
                ensembles/thermostats that define one explicitly.
        """
        self._ensemble    = ensemble.lower()
        self.velocity_representation = normalize_velocity_representation(velocity_representation)
        set_atoms_velocity_representation(atoms, self.velocity_representation)
        self._n_atoms     = len(atoms)
        self._is_pbc      = bool(any(atoms.pbc))
        self._timestep    = timestep
        self._step_offset = step_offset
        self._n_dof_override = n_dof
        self._dof_description = dof_description
        self._write_sync_thermo = bool(write_sync_thermo)
        self._write_conserved_energy = bool(write_conserved_energy)
        # Total steps across the full run (for progress %)
        self._n_steps     = n_steps + step_offset

        # Record wall-clock start; also seed the rolling-speed anchor
        self._wall_start = _time.perf_counter()
        self._last_wall  = self._wall_start
        self._last_step  = step_offset

        if step_offset == 0:
            # Open files (back up any pre-existing files first, GROMACS-style)
            backup_msgs = []
            for p in (
                self.thermo_path,
                self.traj_path,
                self.summary_path,
                self.final_path,
                self.rst_path,
                self.rst_prev_path,
            ):
                if p in self._protected_restart_inputs:
                    continue
                backup = _backup_file(p)
                if backup is not None:
                    backup_msgs.append(f"  Backed up existing file: {p.name} -> {backup.name}\n")

            self.thermo_file = open(self.thermo_path, 'w')
            # Open trajectory file based on format
            if self.traj_format == 'dcd':
                self.traj_file = DCDWriter(
                    path=self.traj_path,
                    natoms=len(atoms),
                    timestep=timestep,
                    is_periodic=any(atoms.pbc),
                    first_step=step_offset,
                )
            else:  # xyz
                self.traj_file = open(self.traj_path, 'w')
        # else: files already opened in append mode by restart_simulation()

        # Write main output header
        total_steps_display = n_steps + step_offset
        self.log_main([
            "\n" + "="*80 + "\n",
            f"{'MD SIMULATION':^80}\n",
            "="*80 + "\n",
            f"Ensemble:        {ensemble.upper()}\n",
            f"Timestep:        {timestep:.3f} fs\n",
            f"Total steps:     {total_steps_display}\n",
            f"Simulation time: {total_steps_display * timestep:.2f} fs\n",
            f"Velocity repr:   {self.velocity_representation}\n",
        ])

        if step_offset > 0:
            self.log_main([
                f"Resuming from:   step {step_offset} "
                f"({step_offset * timestep / 1000.0:.3f} ps)\n",
                f"Steps remaining: {n_steps}\n",
            ])

        if self._ensemble in ('nvt', 'npt'):
            self.log_main([f"Target temp:     {temperature:.2f} K\n"])
        if self._ensemble == 'npt' and pressure is not None:
            self.log_main([f"Target pressure: {pressure:.2f} bar\n"])

        self.log_main([
            f"\nSystem:\n",
            f"  Atoms:         {len(atoms)}\n",
            f"  Formula:       {atoms.get_chemical_formula()}\n",
            f"  Charge:        {atoms.info.get('charge', 0)}\n",
            f"  Multiplicity:  {atoms.info.get('mult', 1)}\n",
            "\n" + "="*80 + "\n",
        ])

        if step_offset == 0 and backup_msgs:
            self.log_main(["\nWARNING: Pre-existing output files were backed up:\n"] + backup_msgs + ["\n"])
            for msg in backup_msgs:
                print(f"WARNING: {msg.strip()}")

        # Write thermodynamics header (fresh run only; resume appends a separator instead)
        if step_offset == 0:
            self.thermo_file.write(f"# MD Simulation - {ensemble.upper()} Ensemble\n")
            self.thermo_file.write(f"# Timestep: {timestep} fs\n")
            if self._ensemble == 'npt':
                self.thermo_file.write(
                    f"# {'Step':>8} {'Time(fs)':>12} {'Temp(K)':>12} "
                    f"{'KE(Ha)':>15} {'PE(Ha)':>15} {'TE(Ha)':>15} "
                    f"{'Press(bar)':>12} {'Vol(A^3)':>12}\n"
                )
            elif self._ensemble == 'nvt':
                header = (
                    f"# {'Step':>8} {'Time(fs)':>12} {'Temp(K)':>12} "
                    f"{'KE(Ha)':>15} {'PE(Ha)':>15} {'TE(Ha)':>15}"
                )
                if self._write_sync_thermo:
                    header += (
                        f" {'Temp_sync(K)':>15} {'KE_sync(Ha)':>15} {'TE_sync(Ha)':>15}"
                    )
                elif self._write_conserved_energy:
                    header += f" {'H_cons_ext(Ha)':>15}"
                header += "\n"
                self.thermo_file.write(header)
            else:
                self.thermo_file.write(
                    f"# {'Step':>8} {'Time(fs)':>12} {'Temp(K)':>12} "
                    f"{'KE(Ha)':>15} {'PE(Ha)':>15} {'TE(Ha)':>15}\n"
                )
            self.thermo_file.flush()

        # ------------------------------------------------------------------
        # Progress header  (verbose >= 1)
        # Printed once; the data row below it is refreshed in-place with \r.
        # ------------------------------------------------------------------
        if self.verbose >= 1:
            is_npt = self._ensemble == 'npt'
            total_ps = self._n_steps * self._timestep / 1000.0
            _time_col_w = max(len(f"0.000/{total_ps:.3f}"), len("Time(ps)")) + 1
            hdr = (
                f"\n"
                f"  {'Step':>9}  {'Time(ps)':>{_time_col_w}}  {'Progress':>8}  "
                f"{'T(K)':>8}  {'E_total(Ha)':>15}"
                + (f"  {'H_cons_ext(Ha)':>15}" if self._write_conserved_energy else "")
                + f"  {'Speed(ns/day)':>13}  {'ETA':>10}"
                + (f"  {'P(bar)':>10}" if is_npt else "")
            )
            sep = (
                f"  {'-'*9}  {'-'*_time_col_w}  {'-'*8}  "
                f"{'-'*8}  {'-'*15}"
                + (f"  {'-'*15}" if self._write_conserved_energy else "")
                + f"  {'-'*13}  {'-'*10}"
                + (f"  {'-'*10}" if is_npt else "")
            )
            self._time_col_w = _time_col_w
            self._total_ps   = total_ps
            # Write header to file and print to terminal
            self.log_main([hdr + "\n", sep + "\n"])
            print(hdr)
            print(sep)
            # Reserve the data row — cursor stays on this line for \r updates
            print("", end="", flush=True)

    def log_step(self, step: int, time: float, temperature: float,
                 kinetic_energy: float, potential_energy: float,
                 total_energy: float, atoms: Atoms, velocities: np.ndarray,
                 pressure: float = None, volume: float = None,
                 rng_state: Optional[str] = None,
                 rst_every: Optional[int] = None,
                 conserved_energy: Optional[float] = None,
                 velocity_representation: Optional[str] = None,
                 temperature_sync: Optional[float] = None,
                 kinetic_energy_sync: Optional[float] = None,
                 total_energy_sync: Optional[float] = None):
        """
        Log data for current step.

        Args:
            step: Current step number
            time: Current simulation time (fs)
            temperature: Current temperature (K)
            kinetic_energy: Kinetic energy (Hartree)
            potential_energy: Potential energy (Hartree)
            total_energy: Total energy (Hartree)
            atoms: Current ASE Atoms object
            velocities: Current velocities (atomic units: Bohr/a.u. time)
            pressure: Instantaneous pressure in bar (NPT only)
            volume: Cell volume in Å³ (NPT only)
            rng_state: Hex-encoded RNG state to embed in trajectory frame (NVT/NPT only)
            conserved_energy: V-rescale conserved-energy bookkeeping value
                H̃ = H − ΣΔW_external (Hartree). For pure thermostat dynamics this
                reduces to the Bussi 2007 Eq. 15 form; when runtime COM/angular
                projection is enabled it also includes the projection KE change.
                None for NVE or Langevin thermostat.
            velocity_representation: Label describing the semantics of ``velocities``.
            temperature_sync: Sync-corrected temperature (K) for optional thermo output.
            kinetic_energy_sync: Sync-corrected kinetic energy (Hartree).
            total_energy_sync: Sync-corrected total energy (Hartree).
        """
        velocity_representation = normalize_velocity_representation(
            velocity_representation or self.velocity_representation
        )
        self.velocity_representation = velocity_representation
        set_atoms_velocity_representation(atoms, velocity_representation)

        # PE and KE are both passed in Hartree (UMACalculator already converts)
        potential_energy_hartree = potential_energy
        kinetic_energy_hartree = kinetic_energy
        total_energy_hartree = kinetic_energy_hartree + potential_energy_hartree

        # Store raw values and analysis values separately.
        analysis_temperature = temperature_sync if temperature_sync is not None else temperature
        analysis_kinetic_energy = kinetic_energy_sync if kinetic_energy_sync is not None else kinetic_energy_hartree
        analysis_total_energy = total_energy_sync if total_energy_sync is not None else total_energy_hartree
        if temperature_sync is not None and kinetic_energy_sync is not None and total_energy_sync is not None:
            self.analysis_label = "sync-corrected"

        self.energies.append(total_energy_hartree)
        self.temperatures.append(temperature)
        self.times.append(time)
        self.kinetic_energies.append(kinetic_energy_hartree)
        self.potential_energies.append(potential_energy_hartree)
        self.analysis_energies.append(analysis_total_energy)
        self.analysis_temperatures.append(analysis_temperature)
        self.analysis_kinetic_energies.append(analysis_kinetic_energy)
        if conserved_energy is not None:
            self.conserved_energies.append(conserved_energy)
        if pressure is not None:
            self.pressures.append(pressure)

        # Write thermodynamic data every step
        # H_cons column is included only for NVT with V-rescale (Bussi 2007 Eq. 15)
        if self._ensemble == 'npt' and pressure is not None and volume is not None:
            self.thermo_file.write(
                f"{step:>10} {time:>12.3f} {temperature:>12.2f} "
                f"{kinetic_energy_hartree:>15.8f} {potential_energy_hartree:>15.8f} "
                f"{total_energy_hartree:>15.8f} {pressure:>12.3f} {volume:>12.4f}\n"
            )
        elif conserved_energy is not None:
            self.thermo_file.write(
                f"{step:>10} {time:>12.3f} {temperature:>12.2f} "
                f"{kinetic_energy_hartree:>15.8f} {potential_energy_hartree:>15.8f} "
                f"{total_energy_hartree:>15.8f} {conserved_energy:>15.8f}\n"
            )
        else:
            line = (
                f"{step:>10} {time:>12.3f} {temperature:>12.2f} "
                f"{kinetic_energy_hartree:>15.8f} {potential_energy_hartree:>15.8f} "
                f"{total_energy_hartree:>15.8f}"
            )
            if self._write_sync_thermo and temperature_sync is not None and kinetic_energy_sync is not None and total_energy_sync is not None:
                line += (
                    f" {temperature_sync:>15.2f} {kinetic_energy_sync:>15.8f}"
                    f" {total_energy_sync:>15.8f}"
                )
            line += "\n"
            self.thermo_file.write(line)
        self.thermo_file.flush()

        # ------------------------------------------------------------------
        # Progress line  (verbose >= 1, printed every log_every steps)
        # ------------------------------------------------------------------
        # Terminal progress line  (verbose >= 1, every 100 steps)
        # Uses \r to overwrite the same line — no screen scrolling.
        # The .dat file still receives a record at every log_every interval.
        # ------------------------------------------------------------------
        _PRINT_EVERY = 100
        if self.verbose >= 1 and step % _PRINT_EVERY == 0:
            now          = _time.perf_counter()

            # Rolling speed over the last print window
            delta_steps  = step - self._last_step
            delta_wall   = now  - self._last_wall
            if delta_wall > 0:
                ns_per_day = (delta_steps * self._timestep / delta_wall / 1e6 * 86400)
            else:
                ns_per_day = float('nan')
            self._last_wall = now
            self._last_step = step

            # ETA
            steps_remaining = self._n_steps - step
            if ns_per_day > 0 and ns_per_day == ns_per_day:   # not nan
                remaining_s = steps_remaining * self._timestep / 1e6 / ns_per_day * 86400
            else:
                remaining_s = float('inf')

            progress_pct = 100.0 * step / self._n_steps if self._n_steps > 0 else 0.0
            speed_str    = f"{ns_per_day:.3f}" if ns_per_day == ns_per_day else "---"
            eta_str      = _fmt_eta(remaining_s)

            # Build progress row aligned to the header columns:
            #   Step(9)  Time(fs)(10)  Progress(8)  T(K)(8)  E_total(Ha)(15)
            #   Speed(ns/day)(13)  ETA(10)  [P(bar)(10)]
            progress_str = f"{progress_pct:.1f}%"
            current_ps   = time / 1000.0
            time_str     = f"{current_ps:.3f}/{self._total_ps:.3f}"
            progress_temperature = temperature_sync if temperature_sync is not None else temperature
            progress_total_energy = total_energy_sync if total_energy_sync is not None else total_energy_hartree
            line = (
                f"  {step:>9}  {time_str:>{self._time_col_w}}  {progress_str:>8}  "
                f"{progress_temperature:>8.2f}  {progress_total_energy:>15.6f}"
            )
            if self._write_conserved_energy and conserved_energy is not None:
                line += f"  {conserved_energy:>15.6f}"
            line += f"  {speed_str:>13}  {eta_str:>10}"
            if self._ensemble == 'npt' and pressure is not None:
                line += f"  {pressure:>10.2f}"

            # \r returns cursor to line start; pad with spaces to erase
            # any leftover characters from a longer previous line
            print(f"\r{line:<120}", end='', flush=True)

        # Also write to .dat file at log_every frequency (unchanged)
        if step % self.log_every == 0:
            line = (
                f"  Step {step:>8}  {time:>10.2f} fs  "
                f"T {temperature:>7.2f} K  "
                f"E {total_energy_hartree:>14.6f} Ha"
            )
            if self._write_conserved_energy and conserved_energy is not None:
                line += f"  H_cons {conserved_energy:>14.6f} Ha"
            self.log_main([line + "\n"])

        # Write trajectory at traj_every frequency.
        # frame_number stores the MD *step* number so that restart_simulation()
        # can recover step_offset directly without needing to know traj_every.
        if step % self.traj_every == 0:
            if self.traj_format == 'dcd':
                # DCD writer handles its own writing
                self.traj_file.write_frame(atoms, step=step)
            else:  # xyz
                write_xyz_frame(
                    self.traj_file,
                    atoms,
                    energy=total_energy_hartree,
                    frame_number=step,          # MD step number, NOT sequential frame index
                    velocity=velocities if self.debug else None,
                    include_velocities=self.debug,
                    rng_state=rng_state,
                    velocity_representation=velocity_representation,
                )
                self.traj_file.flush()

        # Write restart checkpoint at rst_every frequency
        if rst_every and step % rst_every == 0:
            rotate_rst_checkpoint(
                rst_path=self.rst_path,
                rst_prev_path=self.rst_prev_path,
                atoms=atoms,
                velocities=velocities,
                step=step,
                timestep=self._timestep,
                ensemble=self._ensemble,
                energy=total_energy_hartree,
                rng_state=rng_state,
                velocity_representation=velocity_representation,
            )

    def restart_simulation(
        self,
        ensemble: str,
        timestep: float,
        n_steps: int,
        temperature: float,
        atoms: Atoms,
        pressure: float = None,
        rst_file: str = None,
        load_state: bool = False,
    ):
        """
        Restore state from a .rst checkpoint file, validate against input atoms,
        open output files, and return (atoms, velocities, step_offset).

        Restore state from a .rst checkpoint file, validate against input atoms,
        and return ``(atoms, velocities, step_offset)``.

        Input selection modes:

        1. **Auto-detect** (rst_file=None):
           Search ``{base}_md.rst`` and ``{base}_md_prev.rst``.

        2. **Explicit checkpoint** (rst_file="/path/to/file.rst"):
           Read the specified checkpoint file (auto-appends ``.rst`` if missing).

        Semantic modes:

        - ``restart=yes`` / ``load_state=no``:
          Resume semantics. Checkpoint coordinates, velocities, and step count
          are restored, so the run continues from the saved step.

        - ``load_state=yes``:
          Load-state semantics. Checkpoint coordinates and velocities are
          restored, but the new run starts from step 0.

        Returns None if a restart checkpoint already completed the requested run.
        Raises RuntimeError on hard failures (mismatch, missing files, etc.).
        """
        from ase.cell import Cell

        # ------------------------------------------------------------------
        # Determine checkpoint source
        # ------------------------------------------------------------------
        has_explicit_rst = rst_file is not None

        if has_explicit_rst:
            # Resolve path: auto-append .rst if missing
            rst = Path(rst_file)
            if rst.suffix == '':
                rst = rst.with_suffix('.rst')
            # If relative path, resolve relative to output file's directory
            if not rst.is_absolute():
                rst = Path(self.main_output).parent / rst
            candidates = [rst]
        else:
            candidates = [self.rst_path, self.rst_prev_path]

        self._protected_restart_inputs = set()

        state = None
        errors = []
        used_path = None
        for path in candidates:
            if not path.exists():
                errors.append(f"missing: {path}")
                continue
            try:
                state = read_rst(path)
                used_path = path
                break
            except ValueError as exc:
                errors.append(f"{path.name}: {exc}")

        if state is None:
            raise RuntimeError("MD restart failed: " + "; ".join(errors))

        # ------------------------------------------------------------------
        # Validation: atoms must always match
        # ------------------------------------------------------------------
        if state["natoms"] != len(atoms):
            raise RuntimeError(
                f"Atom count mismatch: rst has {state['natoms']}, input has {len(atoms)}"
            )
        ref_symbols = atoms.get_chemical_symbols()
        for idx, (rst_sym, input_sym) in enumerate(zip(state["symbols"], ref_symbols), 1):
            if rst_sym != input_sym:
                raise RuntimeError(
                    f"Element mismatch between rst and input at position {idx}"
                )

        if not load_state:
            if state["ensemble"] != ensemble:
                raise RuntimeError(
                    f"Ensemble mismatch: rst has '{state['ensemble']}', "
                    f"input specifies '{ensemble}'"
                )
            if abs(state["timestep"] - timestep) > 1e-12:
                raise RuntimeError(
                    f"Timestep mismatch: rst has {state['timestep']}, "
                    f"input specifies {timestep}"
                )
            if state["step"] >= n_steps:
                self.log_main([
                    f"\nRestart checkpoint {used_path.name} already completed the "
                    f"requested run ({state['step']}/{n_steps} steps).\n"
                ])
                # Even though the run is complete, export final.xyz from
                # the checkpoint so the user always has the last-frame file.
                if not self.final_path.exists():
                    atoms.set_positions(state["positions"])
                    if state["cell"] is not None:
                        atoms.set_cell(Cell.fromcellpar(state["cell"]))
                    if state["pbc"] is not None:
                        atoms.set_pbc(state["pbc"])
                    with open(self.final_path, 'w') as f:
                        write_xyz_frame(
                            f,
                            atoms=atoms,
                            energy=state["energy"],
                            frame_number=state["step"],
                            velocity=state["velocities"] if self.debug else None,
                            include_velocities=self.debug,
                            velocity_representation=state.get("velocity_representation"),
                        )
                    self.log_main([
                        f"  Exported final structure: {self.final_path.name}\n"
                    ])
                return None

        if has_explicit_rst and used_path in (self.rst_path, self.rst_prev_path):
            self._protected_restart_inputs = {self.rst_path, self.rst_prev_path}

        if load_state:
            if self.debug:
                self.log_main([
                    f"\nLoading state from explicit checkpoint: {used_path.name} "
                    f"(start new run from step 0)\n",
                ])
            step_offset = 0
        else:
            if has_explicit_rst and self.debug:
                self.log_main([
                    f"\nRestarting from explicit checkpoint: {used_path.name} "
                    f"(resume from step {state['step']})\n",
                ])
            step_offset = state["step"]

        # Restore atoms state
        atoms.set_positions(state["positions"])
        if state["cell"] is not None:
            atoms.set_cell(Cell.fromcellpar(state["cell"]))
        if state["pbc"] is not None:
            atoms.set_pbc(state["pbc"])

        # Store RNG state for ensemble drivers (NVT/NPT) to restore
        self.resumed_rng_state = state.get("rng_state")
        self.resumed_velocity_representation = normalize_velocity_representation(
            state.get("velocity_representation")
        )
        self.resumed_timestep = state["timestep"]
        self.velocity_representation = self.resumed_velocity_representation
        set_atoms_velocity_representation(atoms, self.velocity_representation)

        self.log_debug_initial_state(
            atoms=atoms,
            velocities=state["velocities"],
            mode="load_state" if load_state else "restart",
            source=str(used_path),
            rst_step=state["step"],
            effective_step=step_offset,
            velocity_representation=self.velocity_representation,
        )

        # ------------------------------------------------------------------
        # Open output files only for resume semantics.
        # start_simulation(step_offset>0) skips file opening, so resume mode
        # must append here. load_state starts a fresh run and therefore relies
        # on start_simulation(step_offset=0) to open clean output files.
        # ------------------------------------------------------------------
        if not load_state:
            self.thermo_file = (open(self.thermo_path, "a") if self.thermo_path.exists()
                                else open(self.thermo_path, "w"))
            if self.traj_format == 'dcd':
                if self.traj_path.exists():
                    self.traj_file = DCDWriter.open_for_append(self.traj_path)
                else:
                    self.traj_file = DCDWriter(
                        path=self.traj_path,
                        natoms=len(atoms),
                        timestep=timestep,
                        is_periodic=any(atoms.pbc),
                        first_step=state["step"],
                    )
            else:
                self.traj_file = (open(self.traj_path, "a") if self.traj_path.exists()
                                  else open(self.traj_path, "w"))
            self.thermo_file.write(
                f"\n# --- RESTARTED from {used_path.name} step {state['step']} ---\n"
            )
            self.thermo_file.flush()

        return atoms, state["velocities"], step_offset

    def end_simulation(self, atoms: Atoms = None, final_velocities: np.ndarray = None,
                       rng_state: str = None,
                       velocity_representation: Optional[str] = None):
        """
        Finalize simulation, compute publication-quality conservation metrics,
        write summary file, and write final restart checkpoint.

        Parameters
        ----------
        atoms : ase.Atoms, optional
            Final atomic configuration.
        final_velocities : np.ndarray, optional
            Final velocities in atomic units (Bohr/a.u. time).
            Written to the restart checkpoint so that a subsequent
            NVE/NVT/NPT run can restart from the exact end state.
            Analogous to GROMACS confout.gro (coordinates + velocities).
        rng_state : str, optional
            Hex-encoded RNG state (for NVT/NPT). Written to RST checkpoint
            for deterministic continuation. NVE does not need this.
        velocity_representation : str, optional
            Label describing the semantics of ``final_velocities``.
        """
        velocity_representation = normalize_velocity_representation(
            velocity_representation or self.velocity_representation
        )
        self.velocity_representation = velocity_representation
        if atoms is not None:
            set_atoms_velocity_representation(atoms, velocity_representation)

        # End the \r progress line with a newline so the summary starts cleanly
        if self.verbose >= 1:
            print(flush=True)

        energies     = np.array(self.analysis_energies)
        temperatures = np.array(self.analysis_temperatures)
        times        = np.array(self.times)          # fs
        ke_arr       = np.array(self.analysis_kinetic_energies)
        pe_arr       = np.array(self.potential_energies)
        has_conserved = len(self.conserved_energies) > 0
        cons_arr     = np.array(self.conserved_energies) if has_conserved else None

        # ------------------------------------------------------------------
        # 1. Basic energy statistics  (full trajectory, no skip)
        # ------------------------------------------------------------------
        energy_mean = np.mean(energies)
        energy_std  = np.std(energies)

        # ------------------------------------------------------------------
        # 2. Total energy drift from a full-trajectory linear fit.
        #    Use all sampled points for robustness, then report the fitted
        #    total change across the full trajectory in kcal/mol.
        # ------------------------------------------------------------------
        analysis_time_ps = (times[-1] - times[0]) * 1e-3 if len(times) >= 2 else 0.0
        total_time_fs = (times[-1] - times[0]) if len(times) >= 2 else 0.0

        if len(times) >= 2 and total_time_fs > 0:
            coef = np.polyfit(times, energies, 1)
            drift_value = coef[0] * total_time_fs * HARTREE_TO_KCAL_PER_MOL
        else:
            drift_value = float('nan')

        # ------------------------------------------------------------------
        # 3. Relative energy fluctuation
        # ------------------------------------------------------------------
        rel_fluctuation = energy_std / abs(energy_mean) if energy_mean != 0 else float('nan')

        # ------------------------------------------------------------------
        # 3b. V-rescale conserved energy H̃ drift (Bussi 2007, Eq. 15)
        #     H̃ = H − Σ ΔW should be constant; measure its drift by fitting the
        #     full trajectory and reporting the fitted total change in kcal/mol.
        # ------------------------------------------------------------------
        if has_conserved and len(cons_arr) == len(times):
            if len(times) >= 2 and total_time_fs > 0:
                cons_coef = np.polyfit(times, cons_arr, 1)
                cons_drift = cons_coef[0] * total_time_fs * HARTREE_TO_KCAL_PER_MOL
            else:
                cons_drift = float('nan')
        else:
            cons_drift = float('nan')

        # ------------------------------------------------------------------
        # 4. KE–PE correlation coefficient
        # ------------------------------------------------------------------
        if len(ke_arr) > 1 and np.std(ke_arr) > 0 and np.std(pe_arr) > 0:
            ke_pe_corr = np.corrcoef(ke_arr, pe_arr)[0, 1]
        else:
            ke_pe_corr = float('nan')

        # ------------------------------------------------------------------
        # 5. Temperature statistics
        # ------------------------------------------------------------------
        temp_mean = np.mean(temperatures)
        temp_std  = np.std(temperatures)

        n_dof = self._n_dof_override
        if n_dof is None:
            n_dof = (3 * self._n_atoms if (self._is_pbc or self._n_atoms == 0)
                     else 3 * self._n_atoms - 3)
        if n_dof <= 0:
            n_dof = 1
        dof_description = self._dof_description or ('legacy fallback: PBC 3N' if self._is_pbc else 'legacy fallback: isolated 3N')

        observed_ratio = temp_std / temp_mean if temp_mean > 0 else float('nan')

        # ------------------------------------------------------------------
        # 6. Assemble summary lines
        # ------------------------------------------------------------------
        is_nve = self._ensemble == 'nve'
        ens_label = self._ensemble.upper()
        drift_metric_label = "H̃ drift" if has_conserved else "Energy drift"
        drift_metric_value = cons_drift if has_conserved else drift_value

        if is_nve:
            energy_section = [
                f"\n{'── [NVE] Energy Conservation ──':^80}\n",
                f"  σ(TE)/|⟨TE⟩|:              {rel_fluctuation:>18.2e}\n",
                f"  {drift_metric_label}:         {drift_metric_value:>+18.6f}  kcal/mol\n",
                f"  Fit window:           {analysis_time_ps:>15.3f}  ps  (full trajectory)\n",
                f"  r(KE,PE):                  {ke_pe_corr:>18.4f}\n",
            ]
            temp_section = [
                f"\n{'── [NVE] Temperature ──':^80}\n",
                f"  ⟨T⟩:                       {temp_mean:>18.2f}  K\n",
                f"  σ(T):                      {temp_std:>18.2f}  K\n",
                f"  σ(T)/<T>:                  {observed_ratio:>18.4f}\n",
                f"  N_dof = {n_dof}  {dof_description}\n",
            ]
        else:
            energy_section = [
                f"\n{'── [' + ens_label + '] Temperature Control ──':^80}\n",
                f"  ⟨T⟩:                       {temp_mean:>18.2f}  K\n",
                f"  σ(T):                      {temp_std:>18.2f}  K\n",
                f"  N_dof = {n_dof}  {dof_description}\n",
                f"  r(KE,PE):                  {ke_pe_corr:>18.4f}\n",
            ]
            temp_section = [
                f"\n{'── [' + ens_label + '] Energy ──':^80}\n",
                f"  Mean TE:                   {energy_mean:>18.8f}  Ha\n",
                f"  σ(TE):                     {energy_std:>18.8f}  Ha\n",
                f"  σ(TE)/|⟨TE⟩|:             {rel_fluctuation:>18.2e}\n",
                f"  {drift_metric_label}:         {drift_metric_value:>+18.6f}  kcal/mol\n",
                f"  Fit window:           {analysis_time_ps:>15.3f}  ps  (full trajectory)\n",
            ]

        if has_conserved:
            energy_label = "vrescale-conserved"
        elif self.analysis_label == "sync-corrected":
            energy_label = "lfmiddle-sync-corrected"
        else:
            energy_label = "vv-raw"
        summary_lines = [
            "\n" + "="*80 + "\n",
            f"{'MD SIMULATION COMPLETED':^80}\n",
            "="*80 + "\n",
            f"  Energy reporting basis:    {energy_label:>18}\n",

            f"\n{'── Energy Statistics ──':^80}\n",
            f"  Mean total energy:         {energy_mean:>18.8f}  Ha\n",
            f"  Std deviation:             {energy_std:>18.8f}  Ha\n",
        ] + energy_section + temp_section

        # ------------------------------------------------------------------
        # Wall-clock performance summary  (always appended)
        # ------------------------------------------------------------------
        total_wall   = _time.perf_counter() - self._wall_start   # seconds
        sim_time_ns  = (times[-1] - times[0]) * FS_TO_NS         # ns of MD
        if total_wall > 0:
            ns_per_day_avg = sim_time_ns / total_wall * 86400
            s_per_ns       = total_wall / sim_time_ns if sim_time_ns > 0 else float('nan')
        else:
            ns_per_day_avg = float('nan')
            s_per_ns       = float('nan')

        perf_lines = [
            f"\n{'── Performance ──':^80}\n",
            f"  Wall time:                  {_fmt_duration(total_wall):>18}\n",
            f"  Simulation time:            {sim_time_ns*1000:>15.3f}  ps\n",
        ]
        if not (ns_per_day_avg != ns_per_day_avg):   # not nan
            perf_lines += [
                f"  Average speed:              {ns_per_day_avg:>15.4f}  ns/day\n",
                f"  Time per ns:                {s_per_ns:>15.1f}  s/ns\n",
            ]
        summary_lines += perf_lines

        self.log_main(summary_lines, echo=True)

        # ------------------------------------------------------------------
        # Write summary file
        # ------------------------------------------------------------------
        with open(self.summary_path, 'w') as f:
            f.write("MD Simulation Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Ensemble:                   {self._ensemble.upper()}\n")
            f.write(f"Total steps:                {len(self.energies)}\n")
            f.write(f"Total time:                 {self.times[-1]:.2f} fs\n")
            f.write(f"Number of atoms:            {self._n_atoms}\n")
            f.write(f"N_dof:                      {n_dof}  "
                    f"{dof_description}\n")
            f.write(f"Energy reporting basis:     {energy_label}\n\n")

            f.write("Energy Statistics:\n")
            f.write(f"  Mean total energy:        {energy_mean:.8f} Ha\n")
            f.write(f"  Std deviation:            {energy_std:.8f} Ha\n\n")

            if is_nve:
                f.write("Energy Conservation Metrics [NVE]:\n")
                f.write(f"  σ(TE)/|⟨TE⟩|:            {rel_fluctuation:.2e}\n")
                f.write(f"  {drift_metric_label}:        {drift_metric_value:+.6f} kcal/mol\n")
                f.write(f"  Fit window:          {analysis_time_ps:.3f} ps  (full trajectory)\n")
                f.write(f"  r(KE,PE):                 {ke_pe_corr:.4f}\n\n")

                f.write("Temperature Statistics:\n")
                f.write(f"  Mean temperature:         {temp_mean:.2f} K\n")
                f.write(f"  Std deviation:            {temp_std:.2f} K\n")
                f.write(f"  σ(T)/<T>:                 {observed_ratio:.4f}\n")
            else:
                f.write(f"Temperature Control Metrics [{self._ensemble.upper()}]:\n")
                f.write(f"  Mean temperature:         {temp_mean:.2f} K\n")
                f.write(f"  Std deviation:            {temp_std:.2f} K\n")
                f.write(f"  r(KE,PE):                 {ke_pe_corr:.4f}\n\n")

                f.write(f"Energy Metrics [{self._ensemble.upper()}]:\n")
                f.write(f"  σ(TE)/|⟨TE⟩|:            {rel_fluctuation:.2e}\n")
                f.write(f"  {drift_metric_label}:        {drift_metric_value:+.6f} kcal/mol\n")
                f.write(f"  Fit window:          {analysis_time_ps:.3f} ps  (full trajectory)\n")

            if self.pressures:
                pressures_arr = np.array(self.pressures)
                f.write(f"\nPressure Statistics:\n")
                f.write(f"  Mean pressure:            {np.mean(pressures_arr):.3f} bar\n")
                f.write(f"  Std deviation:            {np.std(pressures_arr):.3f} bar\n")

            f.write(f"\nPerformance:\n")
            f.write(f"  Wall time:                {_fmt_duration(total_wall)}\n")
            if not (ns_per_day_avg != ns_per_day_avg):
                f.write(f"  Average speed:            {ns_per_day_avg:.4f} ns/day\n")
                f.write(f"  Time per ns:              {s_per_ns:.1f} s/ns\n")

        if self.pressures:
            pressures_log = np.array(self.pressures)
            self.log_main([
                f"\n{'── Pressure Statistics ──':^80}\n",
                f"  Mean pressure:              {np.mean(pressures_log):>18.3f}  bar\n",
                f"  Std deviation:              {np.std(pressures_log):>18.3f}  bar\n",
            ], echo=True)

        # ------------------------------------------------------------------
        # Write final restart checkpoint  (GROMACS state.cpt equivalent)
        # Contains final coordinates + velocities + simulation state
        # so that a subsequent NVE/NVT/NPT run can restart from the exact end state.
        # ------------------------------------------------------------------
        # Also write final structure file (GROMACS confout.gro equivalent)
        # Contains final coordinates + velocities in XYZ format for easy inspection
        # and use as input for the next simulation stage.
        # ------------------------------------------------------------------
        final_written = False
        final_xyz_written = False
        if atoms is not None and final_velocities is not None:
            final_energy = self.energies[-1] if self.energies else float("nan")
            final_step = int(round(self.times[-1] / self._timestep)) if self.times and self._timestep > 0 else 0

            # Write RST checkpoint (complete state for restart)
            # For NVT/NPT, include RNG state for deterministic continuation
            rotate_rst_checkpoint(
                rst_path=self.rst_path,
                rst_prev_path=self.rst_prev_path,
                atoms=atoms,
                velocities=final_velocities,
                step=final_step,
                timestep=self._timestep,
                ensemble=self._ensemble,
                energy=final_energy,
                rng_state=rng_state,
                velocity_representation=velocity_representation,
            )
            final_written = True

            # Write final structure XYZ (confout.gro equivalent)
            # This file contains:
            #   - Coordinates (can be used as input for next stage)
            #   - Velocities (embedded in XYZ, read by InputReader)
            #   - Cell parameters (if PBC)
            with open(self.final_path, 'w') as f:
                write_xyz_frame(
                    f,
                    atoms=atoms,
                    energy=final_energy,
                    frame_number=final_step,
                    velocity=final_velocities if self.debug else None,
                    include_velocities=self.debug,
                    velocity_representation=velocity_representation,
                )
            final_xyz_written = True

        self.log_main([
            f"\n{'── Output Files ──':^80}\n",
            f"  Thermodynamics:             {self.thermo_path.name}\n",
            f"  Trajectory:                 {self.traj_path.name}\n",
            f"  Summary:                    {self.summary_path.name}\n",
            *([f"  Final structure:           {self.final_path.name}\n"
               f"    (Structure handoff only; strict restart state is in {self.rst_path.name})\n"]
              if final_xyz_written else []),
            *([f"  Checkpoint:                 {self.rst_path.name}\n"
               f"  Previous checkpoint:        {self.rst_prev_path.name}\n"]
              if final_written else []),
            "="*80 + "\n",
        ], echo=True)

        # Close files
        if self.thermo_file:
            self.thermo_file.close()
        if self.traj_file:
            self.traj_file.close()

    def log_main(self, messages: list, echo: bool = False):
        """
        Write messages to main output file.
        If echo=True and verbose >= 1, also print to stdout (terminal).

        Args:
            messages: List of message strings
            echo:     Mirror output to stdout (used for progress lines)
        """
        with open(self.main_output, 'a') as f:
            for msg in messages:
                f.write(msg)
        if echo and self.verbose >= 1:
            for msg in messages:
                print(msg, end='', flush=True)
