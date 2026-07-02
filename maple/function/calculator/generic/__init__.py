"""Universal ASE-calculator adapter package.

``GenericASECalculator`` wraps any instantiated ``ase.calculators.calculator``
Calculator so it satisfies MAPLE's ``CalcABC`` contract with **zero per-potential
engine code**. The MD/optimization engine is already potential-agnostic — it only
ever touches the ASE interface (``get_potential_energy`` / ``get_forces`` /
``get_stress``) routed through ``CalcABC._finalize_results`` for units. This
package makes that explicit: a new MLIP becomes a few-line builder + one registry
line instead of a bespoke ``CalcABC`` subclass.

See ``_generic_ase_calculator`` for the adapter and the rationale; see
``_mace_mp_generic`` for a worked proof (mace-mp-0 through the adapter, coexisting
with the native ``mace-mp-0`` backend for A/B verification).
"""
from ._generic_ase_calculator import GenericASECalculator
# Import the concrete builder backends so their @register_calculator runs when this
# package is imported (SetCalculator also imports each by _BUILTIN_NAME_TO_MODULE path
# on demand; importing here keeps the registry populated + exports the classes). Each
# builder lazily imports its heavy upstream (mace / fairchem) only inside __init__.
from ._mace_mp_generic import MACEMPGenericCalculator
from ._mace_off_generic import MACEOFFGenericCalculator
from ._uma_generic import UMAGenericCalculator

__all__ = [
    'GenericASECalculator',
    'MACEMPGenericCalculator',
    'MACEOFFGenericCalculator',
    'UMAGenericCalculator',
]
