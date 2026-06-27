"""
NPT (isothermal-isobaric) ensemble implementation.

Supports two combinations:
    thermostat: 'langevin' | 'v-rescale'  (default: v-rescale)
    barostat:   'berendsen' | 'c-rescale' (default: c-rescale)

Recommended combination for production MLP runs:
    thermostat=v-rescale + barostat=c-rescale
    → both produce the correct NPT ensemble.

Berendsen variants are suitable for rapid pre-equilibration but suppress
pressure/temperature fluctuations and do not generate correct ensemble averages.

Integration order each step:
    - Langevin: LFMiddle carried-velocity sequence, then barostat.
    - V-rescale: existing split Velocity Verlet sequence, then barostat.

Requirements:
    - Atoms object must have a periodic cell (atoms.pbc must be True)
    - Calculator should support stress tensor evaluation for accurate pressure

References:
    Berendsen et al., J. Chem. Phys. 81, 3684 (1984).
    Bussi, Donadio & Parrinello, J. Chem. Phys. 126, 014101 (2007).
    Bernetti & Bussi, J. Chem. Phys. 153, 114107 (2020).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from ase import Atoms

from ...jobABC import JobABC
from maple.function.timer import timer

from ..integrator.velocity_verlet import VelocityVerlet
from ..thermostat.langevin import LangevinThermostat
from ..thermostat.vrescale import VRescaleThermostat
from ..barostat.berendsen import BerendsenBarostat
from ..barostat.crescale import CRescaleBarostat
from ..utils import (
    VELOCITY_REPR_LFMIDDLE_CARRIED,
    VELOCITY_REPR_STANDARD,
    apply_runtime_motion_projection,
    calculate_temperature,
    calculate_kinetic_energy,
    compute_instantaneous_pressure,
    describe_dof_policy,
    get_atoms_velocity_representation,
    get_initialization_dof_policy,
    get_n_dof_from_policy,
    get_runtime_dof_policy,
    initialize_velocities,
    HA_PER_ANG_TO_AU,
    lfmiddle_carried_to_standard,
    set_atoms_velocity_representation,
    standard_to_lfmiddle_carried,
    FS_TO_AU,
)
from ..rst_io import get_rng_state_hex, restore_rng_from_hex
from ..logger import MDLogger


@dataclass
class NPTParams:
    """
    Parameters for NPT (isothermal-isobaric) ensemble simulation.

    NPT is used for density equilibration and computing thermodynamic
    properties at constant pressure (e.g., liquid density, compressibility).
    Defaults follow published standards for ML potentials.

    Thermostat default: v-rescale (Bussi et al. 2007 JCP 126, 014101)
      — correct canonical ensemble; less perturbative than Langevin.
    Barostat default: c-rescale (Bernetti & Bussi 2020 JCP 153, 114107)
      — correct isothermal-isobaric ensemble; analogue of v-rescale for pressure.

    Recommended production combination: thermostat=v-rescale + barostat=c-rescale.
    Berendsen variants are suitable for rapid pre-equilibration only.
    """
    # ------------------------------------------------------------------
    # Timestep: 0.1 fs
    # Smaller timestep for ML potentials improves energy conservation.
    # Refs: Zhang et al. (2018) Phys. Rev. Lett. 120, 143001 (DeePMD);
    #       Batatia et al. (2022) NeurIPS 35, 11423 (MACE).
    # ------------------------------------------------------------------
    timestep:        float = 0.1          # fs

    # ------------------------------------------------------------------
    # Total steps: 100000 × 0.1 fs = 10 ps
    # Standard default simulation length for ML-MD runs.
    # Refs: GROMACS Lemkul tutorial; AMBER Tutorial 1.
    # ------------------------------------------------------------------
    steps:           int   = 100000       # steps (= 10 ps at 0.1 fs/step)

    temperature:     float = 300.0        # K
    pressure:        float = 1.0          # bar

    # ------------------------------------------------------------------
    # Thermostat: v-rescale (default for NPT)
    # V-rescale produces correct canonical KE distribution while perturbing
    # dynamics less than Langevin, making it better suited for NPT where
    # both thermostat and barostat act each step.
    # Ref: GROMACS default since v4.5 (Bussi et al. 2007).
    # ------------------------------------------------------------------
    thermostat:      str   = 'v-rescale'  # [GROMACS default; Bussi 2007]

    # ------------------------------------------------------------------
    # Barostat: c-rescale (default for NPT)
    # C-rescale is the correct NPT barostat (Bernetti & Bussi 2020).
    # Unlike Berendsen, it produces the full Gibbs (N,P,T) distribution.
    # ------------------------------------------------------------------
    barostat:        str   = 'c-rescale'  # [Bernetti & Bussi 2020 JCP 153, 114107]

    # Langevin-specific (only used when thermostat='langevin')
    friction:        float = 0.001        # 1/fs = 1 ps⁻¹

    # ------------------------------------------------------------------
    # V-rescale τ_T: 200 fs
    # Slightly larger than NVT default (100 fs) to avoid over-coupling
    # when both thermostat and barostat act each step.
    # Ref: GROMACS NPT tutorial: tau_t = 0.1 ps = 100 fs;
    #      CHARMM-GUI NPT protocol: tau_t = 1 ps (conservative).
    # ------------------------------------------------------------------
    tau_t:           float = 200.0        # fs  [GROMACS NPT tutorial]

    # ------------------------------------------------------------------
    # Barostat τ_P: 2000 fs = 2 ps
    # Larger than classical MD defaults (0.5–1 ps) to account for ML
    # potential noise in instantaneous pressure.  Noisy pressure → large
    # τ_P needed to avoid volume instability.
    # Ref: GROMACS Lemkul NPT tutorial: tau_p = 2.0 ps;
    #      Bernetti & Bussi 2020 §III: τ_P ≥ 1 ps recommended.
    # ------------------------------------------------------------------
    tau_p:           float = 2000.0       # fs  [GROMACS Lemkul; Bernetti 2020]

    # ------------------------------------------------------------------
    # Isothermal compressibility: 4.5e-5 1/bar (liquid water, 300 K, 1 bar)
    # Used by both Berendsen and C-rescale barostats as a scaling prefactor.
    # The barostat dynamics are not very sensitive to this value; using water
    # as a default is standard practice for biomolecular systems.
    # Ref: CRC Handbook of Chemistry and Physics; GROMACS mdp default.
    # ------------------------------------------------------------------
    compressibility: float = 4.5e-5       # 1/bar  [CRC Handbook; GROMACS default]

    # ------------------------------------------------------------------
    # Output frequencies
    #
    # ML potentials are ~1000–3000× slower than classical FFs.
    # A typical ML-NPT run is 10–50 ps; dense output is needed to monitor
    # density convergence and detect volume instabilities early.
    #
    # Target: 100–1000 frames per 10 ps.
    #   traj_every = 100 steps × 0.1 fs/step = 10 fs = 0.01 ps/frame
    #   10 ps → 1000 frames  ✓
    #
    # Refs: Stocker et al. (2022) Mach. Learn.: Sci. Technol. 3, 045010 —
    #         GNN-MD benchmarks, typical run 10–100 ps with dense output.
    #       Kovács et al. (2023) J. Chem. Phys. 159, 044118 — MACE evaluation
    #         with per-step monitoring of thermodynamic convergence.
    # ------------------------------------------------------------------
    traj_every:      int   = 100          # steps (= 10 fs = 0.01 ps at 0.1 fs/step)
    log_every:       int   = 100          # steps (= 10 fs)

    # ------------------------------------------------------------------
    # Trajectory format: xyz (text) or dcd (binary)
    # DCD binary format is ~3-4x smaller than XYZ and faster to read/write.
    # Ref: CHARMM documentation; VMD molfile plugin.
    # ------------------------------------------------------------------
    traj_format:     str   = "xyz"        # "xyz" (text, default) or "dcd" (binary)

    verbose:         int   = 1
    debug:           bool  = False
    init_velocities: bool  = True
    restart:          bool  = False
    load_state:       bool  = False
    rst_file:         str   = ""           # Path to RST checkpoint file (explicit source for restart/load_state)
    rst_every:        int   = 1000
    remove_com:       bool  = True   # initialization-only COM removal
    remove_rotation:  bool  = False  # legacy alias path; prefer remove_angular
    remove_angular:   bool  = False  # initialization-only COM + rotation; parallel to remove_com
    remove_com_every: int   = 100    # runtime-only COM removal
    remove_angular_every: int = 0    # runtime-only COM + rotation; parallel to remove_com_every
    random_seed: Optional[int] = None


class NPT(JobABC):
    """
    NPT (isothermal-isobaric) ensemble simulation.

    Integrates with the MAPLE dispatcher via JobABC.
    """

    _THERMOSTAT_CHOICES = {'langevin', 'v-rescale'}
    _BAROSTAT_CHOICES   = {'berendsen', 'c-rescale'}

    def __init__(self, output: str, atoms: Atoms, paras: Optional[dict] = None):
        super().__init__(output)

        if atoms.calc is None:
            raise ValueError("Atoms object must have a calculator attached")
        if not any(atoms.pbc):
            raise ValueError(
                "NPT ensemble requires a periodic cell (atoms.pbc must be True). "
                "Use NVT or NVE for non-periodic systems."
            )

        self.atoms = atoms
        self.params = self._init_params(NPTParams, paras, ("md", "MD", "npt", "NPT"))

        if self.params.thermostat not in self._THERMOSTAT_CHOICES:
            raise ValueError(
                f"Unknown thermostat '{self.params.thermostat}'. "
                f"Choose from: {self._THERMOSTAT_CHOICES}"
            )
        if self.params.barostat not in self._BAROSTAT_CHOICES:
            raise ValueError(
                f"Unknown barostat '{self.params.barostat}'. "
                f"Choose from: {self._BAROSTAT_CHOICES}"
            )

        # Warn if user set Langevin-specific params but chose v-rescale (or vice versa)
        if self.params.thermostat == 'v-rescale' and paras and 'friction' in (paras or {}):
            self.log_info([
                "\n*** WARNING: 'friction' parameter was specified but thermostat is 'v-rescale'.\n"
                "    The friction parameter is only used by the Langevin thermostat.\n"
                "    If you intended Langevin dynamics, add: thermostat=langevin\n\n"
            ])
        if self.params.thermostat == 'langevin' and paras and 'tau_t' in (paras or {}):
            self.log_info([
                "\n*** WARNING: 'tau_t' parameter was specified but thermostat is 'langevin'.\n"
                "    The tau_t parameter is only used by the V-rescale thermostat.\n\n"
            ])

        self._rng = (np.random.default_rng(self.params.random_seed)
                     if self.params.random_seed is not None
                     else np.random.default_rng())

        runtime_policy = get_runtime_dof_policy(
            atoms,
            remove_com_every=self.params.remove_com_every,
            remove_angular_every=self.params.remove_angular_every,
        )
        for warning in runtime_policy["warnings"]:
            self.log_info([f"\n*** WARNING: {warning}\n"])
        if self.params.remove_angular:
            self.log_info(["\n*** WARNING: remove_angular is ignored for NPT/PBC systems; only initialization COM removal remains active.\n"])
        self._runtime_n_dof = get_n_dof_from_policy(runtime_policy)
        self._runtime_dof_description = describe_dof_policy(runtime_policy)

        if self.params.thermostat == 'langevin':
            self.thermostat = LangevinThermostat(
                atoms,
                temperature=self.params.temperature,
                friction=self.params.friction,
                timestep=self.params.timestep,
                rng=self._rng,
            )
        else:  # v-rescale
            self.thermostat = VRescaleThermostat(
                atoms,
                temperature=self.params.temperature,
                tau_t=self.params.tau_t,
                timestep=self.params.timestep,
                rng=self._rng,
                n_dof=self._runtime_n_dof,
            )

        if self.params.barostat == 'berendsen':
            self.barostat = BerendsenBarostat(
                atoms,
                pressure=self.params.pressure,
                tau_p=self.params.tau_p,
                timestep=self.params.timestep,
                compressibility=self.params.compressibility,
            )
        else:  # c-rescale
            self.barostat = CRescaleBarostat(
                atoms,
                pressure=self.params.pressure,
                temperature=self.params.temperature,
                tau_p=self.params.tau_p,
                timestep=self.params.timestep,
                compressibility=self.params.compressibility,
                rng=self._rng,
            )

        self.logger = MDLogger(
            output_path=output,
            log_every=self.params.log_every,
            traj_every=self.params.traj_every,
            traj_format=self.params.traj_format,
            verbose=self.params.verbose,
            debug=self.params.debug,
        )

    def _prepare_langevin_velocities(
        self,
        velocities: np.ndarray,
        representation: str,
        forces: np.ndarray,
        source_timestep_au: Optional[float] = None,
    ) -> tuple[np.ndarray, str]:
        """Return LF-Middle carried velocities for the Langevin path."""
        if representation == VELOCITY_REPR_LFMIDDLE_CARRIED:
            standard_velocities = lfmiddle_carried_to_standard(
                self.atoms,
                velocities,
                forces,
                source_timestep_au if source_timestep_au is not None else self.thermostat.timestep,
            )
            carried = standard_to_lfmiddle_carried(
                self.atoms,
                standard_velocities,
                forces,
                self.thermostat.timestep,
            )
            return carried, VELOCITY_REPR_LFMIDDLE_CARRIED
        carried = standard_to_lfmiddle_carried(
            self.atoms,
            velocities,
            forces,
            self.thermostat.timestep,
        )
        return carried, VELOCITY_REPR_LFMIDDLE_CARRIED

    def run(self):
        """Execute NPT simulation."""
        with timer("MD Simulation (NPT)"):
            self._log_parameters()

            if self.params.load_state:
                if self.params.init_velocities and self.params.debug:
                    self.log_info([
                        "\nload_state=True: ignoring init_velocities and using coordinates/velocities from RST.\n"
                    ])
                result = self.logger.restart_simulation(
                    ensemble='npt',
                    timestep=self.params.timestep,
                    n_steps=self.params.steps,
                    temperature=self.params.temperature,
                    atoms=self.atoms,
                    pressure=self.params.pressure,
                    rst_file=self.params.rst_file if self.params.rst_file else None,
                    load_state=True,
                )
                self.atoms, velocities, step_offset = result
                velocity_representation = self.logger.resumed_velocity_representation
                resumed_timestep_au = (
                    self.logger.resumed_timestep * FS_TO_AU
                    if self.logger.resumed_timestep is not None else None
                )
                if self.logger.resumed_rng_state is not None:
                    restore_rng_from_hex(self._rng, self.logger.resumed_rng_state)
                remaining = self.params.steps
            elif self.params.restart:
                if self.params.init_velocities and self.params.debug:
                    self.log_info([
                        "\nrestart=True: ignoring init_velocities and using coordinates/velocities from RST.\n"
                    ])
                result = self.logger.restart_simulation(
                    ensemble='npt',
                    timestep=self.params.timestep,
                    n_steps=self.params.steps,
                    temperature=self.params.temperature,
                    atoms=self.atoms,
                    pressure=self.params.pressure,
                    rst_file=self.params.rst_file if self.params.rst_file else None,
                    load_state=False,
                )
                if result is None:   # already completed
                    return
                self.atoms, velocities, step_offset = result
                velocity_representation = self.logger.resumed_velocity_representation
                resumed_timestep_au = None
                # Restore RNG state for deterministic continuation
                if self.logger.resumed_rng_state is not None:
                    restore_rng_from_hex(self._rng, self.logger.resumed_rng_state)
                remaining = self.params.steps - step_offset
            else:
                if 'velocities' in self.atoms.arrays and self.params.init_velocities:
                    velocities = self.atoms.arrays['velocities']
                    velocity_representation = get_atoms_velocity_representation(self.atoms)
                    t_check = calculate_temperature(self.atoms, velocities)
                    self.log_info([
                        f"\nVelocities loaded from input file "
                        f"(T = {t_check:.2f} K); skipping random initialisation.\n"
                    ])
                elif self.params.init_velocities:
                    velocities = self._initialize_velocities()
                    velocity_representation = VELOCITY_REPR_STANDARD
                else:
                    if 'velocities' not in self.atoms.arrays:
                        raise ValueError(
                            "init_velocities=False, "
                            "but no velocities found in atoms.arrays"
                        )
                    velocities = self.atoms.arrays['velocities']
                    velocity_representation = get_atoms_velocity_representation(self.atoms)
                resumed_timestep_au = None
                step_offset = 0
                remaining   = self.params.steps
                source = "input_xyz" if 'velocities' in self.atoms.arrays and not self.params.init_velocities else ("input_xyz" if 'velocities' in self.atoms.arrays and self.params.init_velocities else "init_velocities")
                self.logger.log_debug_initial_state(
                    self.atoms,
                    velocities,
                    mode=source,
                    effective_step=step_offset,
                    velocity_representation=velocity_representation,
                )

            final_velocities, final_representation = self._run_simulation(
                velocities,
                velocity_representation=velocity_representation,
                step_offset=step_offset,
                n_steps=remaining,
                source_timestep_au=resumed_timestep_au,
            )
            self.atoms.arrays['velocities'] = final_velocities
            set_atoms_velocity_representation(self.atoms, final_representation)

    def _log_parameters(self):
        """Log NPT parameters to output."""
        lines = [
            "\n" + "=" * 80 + "\n",
            f"{'NPT MD PARAMETERS':^80}\n",
            "=" * 80 + "\n",
            f"Ensemble:              NPT (isothermal-isobaric)\n",
            f"Thermostat:            {self.params.thermostat}\n",
            f"Barostat:              {self.params.barostat}\n",
            f"Timestep:              {self.params.timestep:.3f} fs\n",
            f"Total steps:           {self.params.steps}\n",
            f"Temperature:           {self.params.temperature:.2f} K\n",
            f"Target pressure:       {self.params.pressure:.2f} bar\n",
        ]
        if self.params.thermostat == 'langevin':
            lines.append(f"Friction (γ):          {self.params.friction:.4f} 1/fs\n")
        else:
            lines.append(f"τ_T:                   {self.params.tau_t:.1f} fs\n")
        lines += [
            f"τ_P:                   {self.params.tau_p:.1f} fs\n",
            f"Compressibility:       {self.params.compressibility:.2e} 1/bar\n",
            f"\nOutput frequencies:\n",
            f"  Log every:           {self.params.log_every} steps\n",
            f"  Traj every:          {self.params.traj_every} steps\n",
            f"\nVelocity init:         {self.params.init_velocities}\n",
            f"Restart mode:          {self.params.restart}\n",
            f"Load-state mode:       {self.params.load_state}\n",
            f"RST every:             {self.params.rst_every} steps\n",
            f"Remove COM:            {self.params.remove_com} (initialization-only)\n",
            f"Remove angular:        {self.params.remove_angular} (initialization-only; ignored under PBC)\n",
            f"Remove COM every:      {self.params.remove_com_every} (runtime-only)\n",
            f"Remove angular ev.:    {self.params.remove_angular_every} (runtime-only; ignored under PBC)\n",
        ]
        if self.params.random_seed is not None:
            lines.append(f"Random seed:           {self.params.random_seed}\n")
        lines.append("=" * 80 + "\n")
        self.log_info(lines)

    def _initialize_velocities(self) -> np.ndarray:
        """Initialize velocities from Maxwell-Boltzmann distribution."""
        self.log_info([f"\nInitializing velocities at {self.params.temperature:.2f} K...\n"])
        velocities = initialize_velocities(
            atoms=self.atoms,
            temperature=self.params.temperature,
            remove_com=self.params.remove_com,
            remove_rotation=self.params.remove_rotation,
            remove_angular=False,
            target_n_dof=self._runtime_n_dof,
            rng=self._rng,
        )
        actual_temp = calculate_temperature(
            self.atoms,
            velocities,
            n_dof=self._runtime_n_dof,
        )
        self.log_info([f"Initial temperature: {actual_temp:.2f} K\n"])
        return velocities

    def _run_simulation(self, velocities: np.ndarray,
                        velocity_representation: str,
                        step_offset: int = 0, n_steps: int = None,
                        source_timestep_au: Optional[float] = None) -> tuple[np.ndarray, str]:
        """
        Run NPT simulation.

        Langevin uses LF-Middle carried velocities followed by the barostat.
        V-rescale keeps the existing split Velocity Verlet + barostat path.
        """
        if n_steps is None:
            n_steps = self.params.steps

        is_langevin = self.params.thermostat == 'langevin'
        force_for_conversion = None
        if velocity_representation == VELOCITY_REPR_LFMIDDLE_CARRIED or is_langevin:
            force_for_conversion = self.atoms.get_forces() * HA_PER_ANG_TO_AU
        conversion_timestep_au = source_timestep_au if source_timestep_au is not None else self.thermostat.timestep
        if is_langevin:
            velocities, velocity_representation = self._prepare_langevin_velocities(
                velocities,
                velocity_representation,
                force_for_conversion,
                source_timestep_au=conversion_timestep_au,
            )
        elif velocity_representation == VELOCITY_REPR_LFMIDDLE_CARRIED:
            velocities = lfmiddle_carried_to_standard(
                self.atoms,
                velocities,
                force_for_conversion,
                conversion_timestep_au,
            )
            velocity_representation = VELOCITY_REPR_STANDARD
        else:
            velocity_representation = VELOCITY_REPR_STANDARD

        write_sync_thermo = bool(
            is_langevin and velocity_representation == VELOCITY_REPR_LFMIDDLE_CARRIED
        )

        self.logger.start_simulation(
            ensemble='npt',
            timestep=self.params.timestep,
            n_steps=n_steps,
            temperature=self.params.temperature,
            atoms=self.atoms,
            pressure=self.params.pressure,
            step_offset=step_offset,
            velocity_representation=velocity_representation,
            n_dof=self._runtime_n_dof,
            dof_description=self._runtime_dof_description,
            write_sync_thermo=write_sync_thermo,
        )
        self.logger.log_main([
            f"\nStarting NPT simulation "
            f"({self.params.thermostat} + {self.params.barostat})...\n\n"
        ])

        integrator = VelocityVerlet(self.atoms, self.params.timestep)
        v = velocities.copy()

        # Cache forces at t=0; the Langevin LFMiddle path reuses the same initial
        # forces for the standard→carried conversion and for the first kick.
        forces = force_for_conversion if force_for_conversion is not None else (
            self.atoms.get_forces() * HA_PER_ANG_TO_AU
        )  # Ha/Å → a.u.
        pressure_stress_warned = False

        for step in range(1, n_steps + 1):
            if is_langevin:
                # LFMiddle sequence with carried velocities, then barostat.
                v = integrator.lfmiddle_full_kick(v, forces)
                integrator.half_step_r(v)
                v = self.thermostat.apply(v)
                v, forces = integrator.lfmiddle_post_thermostat(v)
            else:
                # Keep the existing V-rescale split chain unchanged.
                v_half = integrator.split_step(v, forces)

                # O: thermostat (V-rescale)
                # Note: V-rescale returns (velocities, delta_w) but the thermostat
                # work is not tracked here — Bussi 2007 Eq. 15 conserved quantity
                # is only valid for NVT, not NPT where the barostat also does work.
                v, _delta_w = self.thermostat.apply(v_half)

                # A(half)-B: half-position + force eval + half-kick; returns cached forces
                v, forces = integrator.complete_split_step(v)

            # Barostat: rescale cell after the thermostat/integrator cycle.
            self.barostat.apply(v)
            v, _projection = apply_runtime_motion_projection(
                self.atoms,
                v,
                step=step,
                remove_com_every=self.params.remove_com_every,
                remove_angular_every=self.params.remove_angular_every,
            )
            forces = self.atoms.get_forces() * HA_PER_ANG_TO_AU
            pressure, pressure_stress_warned = compute_instantaneous_pressure(
                self.atoms,
                v,
                stress_warned=pressure_stress_warned,
                class_name=type(self.barostat).__name__,
            )

            abs_step         = step_offset + step
            current_time     = abs_step * self.params.timestep
            temperature      = calculate_temperature(self.atoms, v, n_dof=self._runtime_n_dof)
            kinetic_energy   = calculate_kinetic_energy(self.atoms, v)
            potential_energy = self.atoms.get_potential_energy()  # Ha
            volume           = self.atoms.get_volume()

            temperature_sync = None
            kinetic_energy_sync = None
            total_energy_sync = None
            if write_sync_thermo:
                v_sync = lfmiddle_carried_to_standard(
                    self.atoms,
                    v,
                    forces,
                    integrator.timestep,
                )
                temperature_sync = calculate_temperature(
                    self.atoms,
                    v_sync,
                    n_dof=self._runtime_n_dof,
                )
                kinetic_energy_sync = calculate_kinetic_energy(self.atoms, v_sync)
                total_energy_sync = kinetic_energy_sync + potential_energy

            self.logger.log_step(
                step=abs_step,
                time=current_time,
                temperature=temperature,
                kinetic_energy=kinetic_energy,
                potential_energy=potential_energy,
                total_energy=kinetic_energy + potential_energy,
                atoms=self.atoms,
                velocities=v,
                pressure=pressure,
                volume=volume,
                rng_state=get_rng_state_hex(self._rng),
                rst_every=self.params.rst_every,
                velocity_representation=velocity_representation,
                temperature_sync=temperature_sync,
                kinetic_energy_sync=kinetic_energy_sync,
                total_energy_sync=total_energy_sync,
            )

        self.logger.end_simulation(
            atoms=self.atoms,
            final_velocities=v,
            rng_state=get_rng_state_hex(self._rng),
            velocity_representation=velocity_representation,
        )
        self.logger.log_main(["\nNPT simulation completed successfully.\n"])
        return v, velocity_representation
