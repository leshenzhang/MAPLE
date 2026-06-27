"""
Thermostat implementations for NVT simulations.

Provides temperature control methods:
    - Langevin: LFMiddle / middle-scheme Langevin thermostat primitive — correct NVT, strong coupling
    - V-rescale: Stochastic velocity rescaling (Bussi 2007) — correct NVT, weaker perturbation
"""

from .langevin import LangevinThermostat
from .vrescale import VRescaleThermostat

__all__ = ['LangevinThermostat', 'VRescaleThermostat']
