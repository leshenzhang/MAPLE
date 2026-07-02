"""
MD ensemble implementations.

Provides different statistical ensembles for MD simulations:
    - NVE: Microcanonical (constant N, V, E)
    - NVT: Canonical (constant N, V, T) via Langevin or V-rescale thermostat
    - NPT: Isothermal-isobaric (constant N, P, T) via Langevin/V-rescale + Berendsen/C-rescale

Batched (one MLIP forward over B replicas) + enhanced-sampling ensembles:
    - BatchedMD / BatchedNVT: B-replica canonical MD (one forward/step)
    - BatchedNPT: B-replica isothermal-isobaric MD, per-replica box + barostat
      (one batched force+stress forward/force-eval) -- the PBC Phase-2C kernel
    - BatchedUmbrella: batched umbrella sampling (harmonic windows)
    - BatchedGaMD: batched GaMD + ParGaMD weighted-ensemble
    - BatchedSMD: batched constant-velocity steered MD -> Jarzynski Delta-G
    - REMD: temperature replica-exchange ridden on the batched NVT kernel
    - REST2: node-energy solute-tempering Hamiltonian replica-exchange
    - TPS: transition path sampling (aimless shooting), batch = N shooting trials
"""

from .nve import NVE
from .nvt import NVT
from .npt import NPT
from .batched import BatchedMD
from .nvt_batched import BatchedNVT, BatchedNVTParams
from .npt_batched import BatchedNPT, BatchedNPTParams
from .umbrella_batched import BatchedUmbrella
from .gamd_batched import BatchedGaMD
from .smd_batched import BatchedSMD
from .remd import REMD, REMDParams
from .hremd_rest2 import REST2, REST2Params
from .tps import TPS, TPSParams, OrderParameter, committor_fraction

__all__ = ['NVE', 'NVT', 'NPT', 'BatchedMD', 'BatchedNVT', 'BatchedNVTParams',
           'BatchedNPT', 'BatchedNPTParams',
           'BatchedUmbrella', 'BatchedGaMD', 'BatchedSMD', 'REMD', 'REMDParams',
           'REST2', 'REST2Params',
           'TPS', 'TPSParams', 'OrderParameter', 'committor_fraction']
