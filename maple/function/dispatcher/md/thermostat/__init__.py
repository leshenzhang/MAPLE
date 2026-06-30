"""
Thermostat implementations for NVT simulations.

Provides temperature control methods:
    - Langevin: LFMiddle / middle-scheme Langevin thermostat primitive — correct NVT, strong coupling
    - V-rescale: Stochastic velocity rescaling (Bussi 2007) — correct NVT, weaker perturbation
    - Nosé-Hoover chains: deterministic, time-reversible canonical map (Martyna 1992/1996) —
      correct NVT and preserves dynamical correlations (VACF/spectra)
"""

from .langevin import LangevinThermostat
from .vrescale import VRescaleThermostat
from .nose_hoover import NoseHooverChain

__all__ = ['LangevinThermostat', 'VRescaleThermostat', 'NoseHooverChain']
