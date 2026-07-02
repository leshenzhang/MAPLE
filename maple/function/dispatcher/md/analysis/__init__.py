"""
Trajectory analysis package for MAPLE MD.

Pure post-processing of trajectories MAPLE already writes to disk:
  * single-system DCD (CHARMM/NAMD binary)  -> see ``dcd_writer.py``
  * MAPLE-flavoured XYZ (cell in a NON-standard comment)
  * per-replica XYZ stacks from batched runs (``{stem}.rep{b}.xyz``)

No engine / calculator coupling, no dependency on MAPLE's PBC MD machinery.
Only numpy + ASE are required at runtime (MDAnalysis/cpptraj are used as
*test-time* oracles, never imported here).

Readers yield ``ase.Atoms`` so every downstream observable is ASE-native.

Public API
----------
    from maple.function.dispatcher.md.analysis import (
        DCDTrajReader, MapleXYZReader, MultiReplicaReader, resolve_symbols,
        compute_rdf, compute_msd, compute_rmsd, compute_rmsf,
        total_density, density_profile, count_hbonds,
        mic_distance_matrix,
    )

#1 FOOTGUN: DCD stores NO element symbols. ``DCDTrajReader`` REQUIRES a
``symbols=`` / ``top=`` / ``rst=`` topology source and fails loudly otherwise.
"""

from .reader import (
    DCDTrajReader,
    MapleXYZReader,
    MultiReplicaReader,
    resolve_symbols,
    read_trajectory,
)
from .pbc import mic_distance_matrix, mic_displacements, minimum_image
from .rdf import compute_rdf
from .msd import compute_msd, unwrap_positions
from .rmsf_rmsd import compute_rmsd, compute_rmsf, kabsch_rotate
from .density import total_density, density_profile
from .hbonds import count_hbonds
from .control_variate import (
    control_variate_estimate,
    batched_O_C,
    batched_control_variate,
)

__all__ = [
    "DCDTrajReader",
    "MapleXYZReader",
    "MultiReplicaReader",
    "resolve_symbols",
    "read_trajectory",
    "mic_distance_matrix",
    "mic_displacements",
    "minimum_image",
    "compute_rdf",
    "compute_msd",
    "unwrap_positions",
    "compute_rmsd",
    "compute_rmsf",
    "kabsch_rotate",
    "total_density",
    "density_profile",
    "count_hbonds",
    "control_variate_estimate",
    "batched_O_C",
    "batched_control_variate",
]
