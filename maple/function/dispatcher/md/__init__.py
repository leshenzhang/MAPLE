"""
Molecular Dynamics (MD) module for MAPLE.

Provides classical molecular dynamics simulation with machine learning potentials.

Supported ensembles
-------------------
NVE : microcanonical (constant energy)
NVT : canonical (constant temperature) — Langevin or V-rescale thermostat
NPT : isothermal-isobaric — Berendsen or C-rescale barostat

Typical usage
-------------
    from maple.function.dispatcher.md import NVT
    sim = NVT(output='run', atoms=atoms, paras={'timestep': 1.0, 'steps': 100000})
    sim.run()

Components
----------
Integrator  : VelocityVerlet (symplectic, second-order)
Thermostats : LangevinThermostat (BAOAB), VRescaleThermostat (Bussi 2007)
Barostats   : BerendsenBarostat, CRescaleBarostat (Bernetti & Bussi 2020)
Logger      : MDLogger
"""

__version__ = '0.1.4'
__author__ = 'MAPLE Development Team'

# Ensembles (primary public API)
from .ensemble.nve import NVE
from .ensemble.nvt import NVT
from .ensemble.npt import NPT
# Opt-in batched (replica/ensemble) MD; single-structure NVE/NVT path is the oracle.
from .ensemble.batched import BatchedMD
# C3: batched canonical (NVT) kernel — single-system NVT is the degenerate B=1 case.
from .ensemble.nvt_batched import BatchedNVT

# Thermostats
from .thermostat.langevin import LangevinThermostat
from .thermostat.vrescale import VRescaleThermostat

# Barostats
from .barostat.berendsen import BerendsenBarostat
from .barostat.crescale import CRescaleBarostat

# Integrator
from .integrator.velocity_verlet import VelocityVerlet

# Utilities (retained for backwards compatibility and direct use)
from .utils import (
    calculate_temperature,
    calculate_kinetic_energy,
    initialize_velocities,
    KELVIN_TO_HARTREE,
    AMU_TO_AU,
    FS_TO_AU,
)

__all__ = [
    # Ensembles
    'NVE', 'NVT', 'NPT',
    # Batched (replica/ensemble) MD
    'BatchedMD', 'BatchedNVT',
    # Thermostats
    'LangevinThermostat', 'VRescaleThermostat',
    # Barostats
    'BerendsenBarostat', 'CRescaleBarostat',
    # Integrator
    'VelocityVerlet',
    # Utilities
    'calculate_temperature', 'calculate_kinetic_energy', 'initialize_velocities',
    'KELVIN_TO_HARTREE', 'AMU_TO_AU', 'FS_TO_AU',
]
