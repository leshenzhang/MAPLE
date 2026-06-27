import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers, covalent_radii, vdw_radii
from scipy.spatial import cKDTree


DATA_DIR = Path(__file__).with_name("data")
AVOGADRO = 6.02214076e23
ANGSTROM3_PER_ML = 1.0e24
WATER_MOLAR_MASS_G_MOL = 18.01528
DEFAULT_VDW_SCALE = 0.57
DEFAULT_VDW_FALLBACK_RADIUS = 1.05
DEFAULT_TEMPLATE_DENSITY_TOLERANCE = 0.20
CUSTOM_TEMPLATE_DENSITY_TOLERANCE = 0.05
CUSTOM_TEMPLATE_BOND_SCALE = 1.25
CUSTOM_TEMPLATE_BOND_TOLERANCE = 0.10
MIN_FINAL_TARGET_RATIO = 0.80

# Standard liquid densities (g/mL) near 298 K for the bundled solvent boxes.
# Used only to derive a target molecule count when neither `number` nor an
# explicit `density` is supplied. Solvents absent here fail closed (the caller
# must pass density= or number=) instead of silently assuming water's 1.0 g/mL,
# which would over/under-fill every non-aqueous solvent.
SOLVENT_DENSITY_G_ML = {
    "water": 1.0,
    "methanol": 0.7918,
    "ethanol": 0.7893,
    "propanol": 0.8035,
    "isopropanol": 0.7809,
    "butanol": 0.8095,
    "acetonitrile": 0.7857,
    "acetone": 0.7845,
    "dimethylsulfoxide": 1.1004,
    "dimethylformamide": 0.9445,
    "dimethylacetamide": 0.9366,
    "chloroform": 1.4892,
    "methylenechloride": 1.3266,
    "dichloroethane": 1.2530,
    "carbontet": 1.5940,
    "tetrahydrofuran": 0.8833,
    "ether": 0.7134,
    "ethylacetate": 0.9006,
    "toluene": 0.8669,
    "benzene": 0.8765,
    "xylene": 0.8600,
    "pyridine": 0.9819,
    "aniline": 1.0217,
    "nitrobenzene": 1.1990,
    "nitromethane": 1.1371,
    "aceticacid": 1.0446,
    "hexane": 0.6606,
    "heptane": 0.6795,
    "octane": 0.7025,
    "cyclohexane": 0.7781,
}


@dataclass(frozen=True)
class SolventTemplate:
    coords: np.ndarray
    symbols: np.ndarray
    atom_names: np.ndarray
    residue_names: np.ndarray
    residue_ids: np.ndarray
    groups: list[list[int]]
    cryst1_cellpar: Optional[tuple[float, float, float, float, float, float]] = None


@dataclass(frozen=True)
class TemplateMetrics:
    """Physical metadata used to decide whether stack-based tiling is valid."""

    period: np.ndarray
    molar_mass_g_mol: float
    density_g_ml: float


def _element_from_pdb_line(line: str) -> str:
    elem = line[76:78].strip() if len(line) >= 78 else ""
    if elem:
        return elem[0].upper() + elem[1:].lower()

    atom_name = line[12:16].strip()
    letters = "".join(ch for ch in atom_name if ch.isalpha())
    if not letters:
        return atom_name[:1].upper()
    if len(letters) >= 2 and letters[:2].upper() in {
        "CL", "BR", "NA", "MG", "AL", "SI", "CA", "FE", "ZN", "CU",
        "MN", "CO", "NI", "LI", "BE", "NE", "AR", "KR", "XE", "HE",
    }:
        return letters[0].upper() + letters[1].lower()
    return letters[0].upper()


def _parse_atom_coords(line: str) -> tuple[float, float, float]:
    try:
        return float(line[30:38]), float(line[38:46]), float(line[46:54])
    except ValueError:
        parts = line.split()
        if len(parts) < 8:
            raise
        return float(parts[5]), float(parts[6]), float(parts[7])


def _parse_residue_key(line: str, ter_index: int) -> tuple:
    resname = line[17:20].strip() if len(line) >= 20 else ""
    chain_id = line[21:22].strip() if len(line) >= 22 else ""
    resseq = line[22:26].strip() if len(line) >= 26 else ""
    insertion_code = line[26:27].strip() if len(line) >= 27 else ""
    if resname or chain_id or resseq or insertion_code:
        return (chain_id, resseq, insertion_code, resname)
    return ("TER", ter_index)


def _parse_cryst1(line: str) -> Optional[tuple[float, float, float, float, float, float]]:
    parts = line.split()
    if len(parts) < 7:
        return None
    try:
        a, b, c = (float(parts[i]) for i in range(1, 4))
        alpha, beta, gamma = (float(parts[i]) for i in range(4, 7))
    except ValueError:
        return None
    return (a, b, c, alpha, beta, gamma)


def _orthorhombic_period_from_cryst1(
    cellpar: Optional[tuple[float, float, float, float, float, float]],
) -> Optional[np.ndarray]:
    if cellpar is None:
        return None
    a, b, c, alpha, beta, gamma = cellpar
    if not (a > 0 and b > 0 and c > 0):
        return None
    if not all(abs(angle - 90.0) < 1.0 for angle in (alpha, beta, gamma)):
        return None
    return np.asarray([a, b, c], dtype=np.float64)


def parse_pdb_template(pdbfile: str | os.PathLike[str]) -> SolventTemplate:
    """Parse a pure-solvent PDB template, grouping one molecule per residue.

    Residue names and atom names are retained so downstream PDB output keeps a
    single, correct residue identity per solvent molecule rather than relabelling
    atoms by element.
    """
    coords: list[list[float]] = []
    symbols: list[str] = []
    atom_names: list[str] = []
    residue_names: list[str] = []
    residue_ids: list[int] = []
    groups: list[list[int]] = []
    residue_to_group: dict[tuple, int] = {}
    cryst1_cellpar = None
    ter_index = 0

    with open(pdbfile, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("CRYST1"):
                cryst1_cellpar = _parse_cryst1(line)
                continue
            if line.startswith(("ATOM", "HETATM")):
                key = _parse_residue_key(line, ter_index)
                group_id = residue_to_group.get(key)
                if group_id is None:
                    group_id = len(groups)
                    residue_to_group[key] = group_id
                    groups.append([])

                coords.append(list(_parse_atom_coords(line)))
                symbols.append(_element_from_pdb_line(line))
                atom_names.append(line[12:16].strip())
                residue_names.append(line[17:20].strip())
                atom_index = len(coords) - 1
                groups[group_id].append(atom_index)
                residue_ids.append(group_id)
                continue
            if line.startswith("TER"):
                ter_index += 1

    if not coords:
        raise ValueError(f"No ATOM/HETATM records found in solvent template: {pdbfile}")

    return SolventTemplate(
        coords=np.asarray(coords, dtype=np.float64),
        symbols=np.asarray(symbols, dtype=object),
        atom_names=np.asarray(atom_names, dtype=object),
        residue_names=np.asarray(residue_names, dtype=object),
        residue_ids=np.asarray(residue_ids, dtype=np.int64),
        groups=groups,
        cryst1_cellpar=cryst1_cellpar,
    )


def _unwrap_template_molecules(template: SolventTemplate) -> SolventTemplate:
    """Unwrap each residue/molecule through the CRYST1 minimum image.

    GROMACS/OpenMM-style PDB boxes may keep a single solvent molecule split
    across the visual box boundary.  MAPLE's cluster builder is coordinate-only,
    so normalize each molecule to a contiguous representation before molecule
    radius, connectivity, clash, and tiling calculations.
    """
    period = _orthorhombic_period_from_cryst1(template.cryst1_cellpar)
    if period is None:
        return template

    coords = template.coords.copy()
    for group in template.groups:
        if len(group) < 2:
            continue
        anchor = coords[group[0]].copy()
        for idx in group[1:]:
            delta = coords[idx] - anchor
            delta -= period * np.round(delta / period)
            coords[idx] = anchor + delta

    return SolventTemplate(
        coords=coords,
        symbols=template.symbols,
        atom_names=template.atom_names,
        residue_names=template.residue_names,
        residue_ids=template.residue_ids,
        groups=template.groups,
        cryst1_cellpar=template.cryst1_cellpar,
    )


def parse_pdb_residue_groups(pdbfile):
    """Backward-compatible parser returning coords, symbols, and molecule groups."""
    template = parse_pdb_template(pdbfile)
    return template.coords, template.symbols, template.groups


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    raise ValueError(f"Expected boolean value, got {value!r}.")


def _uniform_quaternion_rotation(rng: np.random.Generator) -> np.ndarray:
    """Return a uniform 3D rotation matrix sampled through a unit quaternion."""
    u1, u2, u3 = rng.random(3)
    qx = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    qy = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    qz = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    qw = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _molecular_weight(symbols: Sequence[str]) -> float:
    mass = 0.0
    for symbol in symbols:
        atomic_number = atomic_numbers.get(str(symbol), 0)
        if atomic_number <= 0:
            raise ValueError(f"Unknown element symbol in solvent template: {symbol!r}")
        mass += float(atomic_masses[atomic_number])
    return mass


def _formula(symbols: Sequence[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(Counter(str(symbol) for symbol in symbols).items()))


def _format_formula(formula: Sequence[tuple[str, int]]) -> str:
    return "".join(f"{symbol}{count if count != 1 else ''}" for symbol, count in formula)


def _element_covalent_radius(symbol: str) -> float:
    atomic_number = atomic_numbers.get(str(symbol), 0)
    if atomic_number <= 0:
        raise ValueError(f"Unknown element symbol in solvent template: {symbol!r}")
    radius = float(covalent_radii[atomic_number])
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError(f"No covalent radius for solvent template element: {symbol!r}")
    return radius


def _is_single_covalent_component(coords: np.ndarray, symbols: Sequence[str]) -> bool:
    """Heuristic guard that one PDB residue represents one molecule.

    PDB solvent templates do not carry reliable bond records.  For custom
    templates, reject residue groups that split into disconnected covalent
    components under a conservative covalent-radius cutoff; this catches the
    common bad-template case where multiple solvent molecules share one residue.
    """
    natoms = len(symbols)
    if natoms <= 1:
        return True

    radii = np.asarray([_element_covalent_radius(symbol) for symbol in symbols])
    adjacency: list[list[int]] = [[] for _ in range(natoms)]
    for i in range(natoms - 1):
        deltas = coords[i + 1:] - coords[i]
        distances = np.linalg.norm(deltas, axis=1)
        cutoffs = (
            (radii[i] + radii[i + 1:]) * CUSTOM_TEMPLATE_BOND_SCALE
            + CUSTOM_TEMPLATE_BOND_TOLERANCE
        )
        bonded = np.where(distances <= cutoffs)[0] + i + 1
        for j in bonded.tolist():
            adjacency[i].append(j)
            adjacency[j].append(i)

    seen = {0}
    stack = [0]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == natoms


def _molecule_number_density(density_g_ml: float, molar_mass_g_mol: float) -> float:
    return density_g_ml / molar_mass_g_mol * AVOGADRO / ANGSTROM3_PER_ML


def _mass_density_g_ml(
    molecule_count: int, molar_mass_g_mol: float, volume_angstrom3: float
) -> float:
    if volume_angstrom3 <= 0:
        return math.nan
    return molecule_count * molar_mass_g_mol / AVOGADRO * ANGSTROM3_PER_ML / volume_angstrom3


def _default_clash_method(params: dict) -> str:
    method = params.get("clash_method")
    if method is not None:
        return str(method).lower()
    if "tolerance" in params or "clash_cutoff" in params:
        return "distance"
    return "vdw"


def _element_vdw_radius(symbol: str, fallback_radius: float) -> float:
    atomic_number = atomic_numbers.get(str(symbol), 0)
    if atomic_number <= 0:
        return fallback_radius
    radius = float(vdw_radii[atomic_number])
    if not math.isfinite(radius) or radius <= 0:
        return fallback_radius
    return radius


def _element_vdw_radii(symbols: Sequence[str], fallback_radius: float) -> np.ndarray:
    return np.asarray(
        [_element_vdw_radius(symbol, fallback_radius) for symbol in symbols],
        dtype=np.float64,
    )


def _self_clash_atom_pairs(
    coords: np.ndarray,
    radii: Optional[np.ndarray],
    scale: float,
    method: str,
    tolerance: float,
) -> list[tuple[int, int]]:
    """Atom index pairs (i < j) closer than the clash threshold within one set.

    Uses a KD-tree so seam overlaps in a tiled solvent network are found in
    near-linear time instead of an O(N^2) distance matrix.
    """
    if len(coords) < 2:
        return []
    tree = cKDTree(coords)
    if method == "distance":
        return [
            (i, j)
            for i, j in tree.query_pairs(r=tolerance)
            if np.linalg.norm(coords[i] - coords[j]) < tolerance
        ]
    rmax = float(radii.max())
    candidate_pairs = tree.query_pairs(r=2.0 * rmax * scale)
    pairs: list[tuple[int, int]] = []
    for i, j in candidate_pairs:
        if np.linalg.norm(coords[i] - coords[j]) < (radii[i] + radii[j]) * scale:
            pairs.append((i, j))
    return pairs


def _cross_clash_atoms(
    probe_coords: np.ndarray,
    probe_radii: Optional[np.ndarray],
    ref_coords: np.ndarray,
    ref_radii: Optional[np.ndarray],
    scale: float,
    method: str,
    tolerance: float,
) -> set[int]:
    """Indices into ``probe_coords`` whose atom clashes with any ``ref`` atom."""
    if len(probe_coords) == 0 or len(ref_coords) == 0:
        return set()
    tree = cKDTree(ref_coords)
    if method == "distance":
        neighbor_lists = tree.query_ball_point(probe_coords, r=tolerance)
        clashing: set[int] = set()
        for k, neighbors in enumerate(neighbor_lists):
            for c in neighbors:
                if np.linalg.norm(probe_coords[k] - ref_coords[c]) < tolerance:
                    clashing.add(k)
                    break
        return clashing
    rmax = float(ref_radii.max())
    clashing: set[int] = set()
    for k in range(len(probe_coords)):
        point = probe_coords[k]
        radius = probe_radii[k]
        for c in tree.query_ball_point(point, r=(radius + rmax) * scale):
            if np.linalg.norm(point - ref_coords[c]) < (radius + ref_radii[c]) * scale:
                clashing.add(k)
                break
    return clashing


def _write_xyz(path: Path, atoms: Atoms, comment: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(atoms)}\n")
        handle.write(f"{comment}\n")
        for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            handle.write(f"{symbol:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n")


def _write_pdb(path: Path, atoms: Atoms) -> None:
    """Write a visualization PDB, one residue per molecule.

    Residue and atom names are taken from the ``maple_resname``/``maple_atom_name``
    arrays (carried from the solvent template) so a molecule is never split across
    residue names by element. Solute atoms (molecule id < 0) form residue 1.
    """
    natoms = len(atoms)
    molecule_ids = atoms.arrays.get("maple_molecule_id", np.full(natoms, -1, dtype=np.int64))
    resnames = atoms.arrays.get("maple_resname")
    atom_names = atoms.arrays.get("maple_atom_name")
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    residue_numbers: dict[int, int] = {}
    next_residue = 2
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(natoms):
            symbol = symbols[idx]
            mid = int(molecule_ids[idx])
            if mid < 0:
                resseq = 1
            else:
                if mid not in residue_numbers:
                    residue_numbers[mid] = next_residue
                    next_residue += 1
                resseq = residue_numbers[mid]
            resname = (str(resnames[idx]) if resnames is not None else "") or (
                "MOL" if mid < 0 else "SLV"
            )
            atom_name = (str(atom_names[idx]) if atom_names is not None else "") or symbol
            x, y, z = positions[idx]
            handle.write(
                f"HETATM{idx + 1:5d} {atom_name:<4.4s} {resname:>3.3s} A{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {symbol:>2s}\n"
            )
        handle.write("END\n")


class ExplicitSolv():
    def __new__(
        cls,
        atoms: Atoms,
        params: dict,
        device,
        output: str,
        base_dir: str | os.PathLike[str] | None = None,
    ):
        # ``device`` is accepted for call-site compatibility with the engine; the
        # cluster build is pure NumPy/SciPy on CPU and does not use it.
        if not isinstance(atoms, Atoms):
            raise ValueError(
                "Explicit solvation currently supports exactly one ASE Atoms "
                "structure; split multi-structure/trajectory input first."
            )
        if any(bool(flag) for flag in atoms.get_pbc()):
            raise ValueError(
                "Explicit solvent clusters are non-periodic coordinate-only "
                "clusters; remove #pbc or use a periodic solvent backend."
            )

        obj = super().__new__(cls)
        obj.output = output
        obj.atoms = atoms.copy()
        obj.params = dict(params or {})
        obj.base_dir = Path(base_dir).expanduser() if base_dir is not None else None
        obj.solute_count = len(atoms)

        obj.solv_name = str(obj.params.get("explicit", "water")).lower()
        obj.shape = str(obj.params.get("shape", "sphere")).lower()
        if obj.shape == "box":
            obj.shape = "cube"
        obj.radius = float(obj.params.get("radius", 10.0))
        obj.padding = (
            None if "padding" not in obj.params else float(obj.params["padding"])
        )
        obj.box_size = obj.params.get("box_size")
        obj.density = obj._resolve_density()
        obj.density_scale = float(obj.params.get("density_scale", 1.0))
        obj.number = obj.params.get("number")
        obj.clash_method = _default_clash_method(obj.params)
        obj.tolerance = float(obj.params.get("tolerance", obj.params.get("clash_cutoff", 2.0)))
        obj.vdw_scale = float(obj.params.get("vdw_scale", DEFAULT_VDW_SCALE))
        obj.vdw_fallback_radius = float(
            obj.params.get("vdw_fallback_radius", DEFAULT_VDW_FALLBACK_RADIUS)
        )
        obj.randomize = _as_bool(obj.params.get("randomize"), True)
        obj.write_shell = _as_bool(obj.params.get("write_shell"), False)
        obj.shell_cutoff = obj.params.get("shell_cutoff")
        obj.seed = obj.params.get("seed", 0)
        obj.rng = np.random.default_rng(None if obj.seed == -1 else obj.seed)

        obj._validate_options()
        obj._resolve_geometry_from_solute()

        obj.data_path, obj.uses_custom_template = obj._resolve_template_path()
        if not obj.data_path.is_file():
            if obj.uses_custom_template:
                msg = f"Explicit solvent solvent_pdb file is not found: {obj.data_path}"
            else:
                msg = f"Solvent {obj.solv_name} is not found"
            obj.log_error(msg)
            raise FileNotFoundError(msg)

        obj.template = _unwrap_template_molecules(parse_pdb_template(obj.data_path))
        obj._validate_custom_template_molecules()
        obj.template_metrics = obj._template_metrics()
        obj._validate_template_density()
        obj.target_count = obj._target_solvent_count()
        obj._log_setup()
        obj._process()
        obj.log_info(["\n" + "-" * 70 + "\n"])
        return obj.atoms

    def _resolve_density(self) -> Optional[float]:
        """Density (g/mL) for the target-count formula, or None if unavailable.

        Explicit ``density=`` wins; otherwise the tabulated solvent density is
        used. None means the solvent is not tabulated and no density was given —
        the caller must then supply ``number=`` or the build fails closed.
        """
        if "density" in self.params:
            return float(self.params["density"])
        return SOLVENT_DENSITY_G_ML.get(self.solv_name)

    def _resolve_template_path(self) -> tuple[Path, bool]:
        custom_template = self.params.get("solvent_pdb")
        if custom_template is None:
            return DATA_DIR / f"{self.solv_name}.pdb", False

        path = Path(str(custom_template).strip()).expanduser()
        if not path.is_absolute():
            root = self.base_dir if self.base_dir is not None else Path.cwd()
            path = root / path
        return path.resolve(), True

    def _template_label(self) -> str:
        if self.uses_custom_template:
            return f"'{self.solv_name}' at {self.data_path}"
        return f"'{self.solv_name}'"

    def _validate_options(self) -> None:
        if self.solute_count == 0:
            raise ValueError(
                "Explicit solvation requires a non-empty solute structure."
            )
        if "write_cell" in self.params:
            raise ValueError(
                "Explicit solvent clusters are non-periodic; "
                "write_cell/PBC output is not supported."
            )
        if self.shape not in {"sphere", "cube"}:
            raise ValueError("Explicit solvent shape must be 'sphere' or 'cube'.")
        if self.padding is not None and self.padding <= 0:
            raise ValueError("Explicit solvent padding must be > 0.")
        if self.shape == "sphere" and self.radius <= 0:
            raise ValueError("Explicit solvent radius must be > 0 for shape=sphere.")
        if self.shape == "cube":
            if "radius" in self.params:
                raise ValueError(
                    "Explicit solvent radius is only valid for shape=sphere."
                )
            if self.padding is not None and self.box_size is not None:
                raise ValueError(
                    "Explicit solvent padding derives the cube box_size from the "
                    "solute envelope; do not combine padding with box_size."
                )
            if self.padding is None and self.box_size is None:
                raise ValueError("Explicit solvent shape=cube requires box_size.")
            if self.box_size is not None:
                self.box_size = float(self.box_size)
            if self.box_size is not None and self.box_size <= 0:
                raise ValueError("Explicit solvent box_size must be > 0.")
        elif self.padding is not None and "radius" in self.params:
            raise ValueError(
                "Explicit solvent padding derives the sphere radius from the "
                "solute envelope; do not combine padding with radius."
            )
        elif "box_size" in self.params:
            raise ValueError("Explicit solvent box_size is only valid for shape=cube.")
        if self.density is not None and self.density <= 0:
            raise ValueError("Explicit solvent density must be > 0.")
        if self.density_scale <= 0:
            raise ValueError("Explicit solvent density_scale must be > 0.")
        if self.number is not None and (type(self.number) is not int or self.number < 0):
            raise ValueError("Explicit solvent number must be an integer >= 0.")
        if self.clash_method not in {"vdw", "distance"}:
            raise ValueError("Explicit solvent clash_method must be 'vdw' or 'distance'.")
        if self.clash_method == "vdw":
            if "tolerance" in self.params or "clash_cutoff" in self.params:
                raise ValueError(
                    "Explicit solvent tolerance/clash_cutoff are only valid "
                    "with clash_method=distance."
                )
            if self.vdw_scale <= 0:
                raise ValueError("Explicit solvent vdw_scale must be > 0.")
            if self.vdw_fallback_radius <= 0:
                raise ValueError("Explicit solvent vdw_fallback_radius must be > 0.")
        elif self.tolerance <= 0:
            raise ValueError("Explicit solvent tolerance must be > 0.")
        if self.clash_method == "distance" and (
            "vdw_scale" in self.params or "vdw_fallback_radius" in self.params
        ):
            raise ValueError(
                "Explicit solvent vdw_scale/vdw_fallback_radius are only valid "
                "with clash_method=vdw."
            )
        if self.write_shell:
            if self.shell_cutoff is None:
                raise ValueError("Explicit solvent write_shell=true requires shell_cutoff.")
            self.shell_cutoff = float(self.shell_cutoff)
            if self.shell_cutoff <= 0:
                raise ValueError("Explicit solvent shell_cutoff must be > 0.")

    def _clash_method_label(self) -> str:
        if self.clash_method == "vdw":
            return (
                "vdw "
                f"(scale={self.vdw_scale:.4f}, "
                f"fallback_radius={self.vdw_fallback_radius:.3f} Å)"
            )
        return f"distance (tolerance={self.tolerance:.3f} Å)"

    def _resolve_geometry_from_solute(self) -> None:
        """Set the solute centering point and any padding-derived geometry."""
        positions = self.atoms.get_positions()
        if self.padding is None:
            self.solute_center = positions.mean(axis=0)
            self._set_solute_extent_metrics(positions)
            return

        if self.shape == "sphere":
            self.solute_center = positions.mean(axis=0)
            self._set_solute_extent_metrics(positions)
            self.radius = float(self.solute_radial_extent + self.padding)
            return

        lower = positions.min(axis=0)
        upper = positions.max(axis=0)
        self.solute_center = lower + 0.5 * (upper - lower)
        span = upper - lower
        self._set_solute_extent_metrics(positions)
        self.box_size = float(span.max() + 2.0 * self.padding)

    def _set_solute_extent_metrics(self, positions: np.ndarray) -> None:
        centered = positions - self.solute_center
        self.solute_radial_extent = float(np.linalg.norm(centered, axis=1).max())
        self.solute_axis_extent = float(np.abs(centered).max())

    def _geometry_warning_lines(self) -> list[str]:
        if self.padding is not None:
            return []
        if self.shape == "sphere":
            if self.radius >= self.solute_radial_extent:
                return []
            return [
                "WARNING: explicit solvent radius is smaller than the solute "
                "radial extent from the cluster center: "
                f"radius={self.radius:.3f} Å < R_solute={self.solute_radial_extent:.3f} Å. "
                "The requested sphere does not enclose the solute; increase "
                "radius or use padding=<Å> for envelope-based sizing.\n"
            ]

        half_box = float(self.box_size) / 2.0
        if half_box >= self.solute_axis_extent:
            return []
        return [
            "WARNING: explicit solvent cube half-size is smaller than the solute "
            "axis-aligned extent from the cluster center: "
            f"box_size/2={half_box:.3f} Å < R_solute_axis={self.solute_axis_extent:.3f} Å. "
            "The requested cube does not enclose the solute; increase box_size "
            "or use padding=<Å> for envelope-based sizing.\n"
        ]

    def log_error(self, error_message: str) -> None:
        with open(self.output, "a", encoding="utf-8") as file:
            file.write(f"ERROR: {error_message}\n")

    def log_info(self, info_message: Iterable[str]) -> None:
        with open(self.output, "a", encoding="utf-8") as file:
            for info in info_message:
                file.write(str(info))

    def _volume(self) -> float:
        if self.shape == "sphere":
            return 4.0 / 3.0 * math.pi * self.radius ** 3
        return float(self.box_size) ** 3

    def _target_solvent_count(self) -> int:
        if self.number is not None:
            return int(self.number)
        if self.density is None:
            msg = (
                f"No tabulated density for solvent '{self.solv_name}'. "
                "Provide density=<g/mL> in #solv(...), or set number=<int> to "
                "request an explicit molecule count."
            )
            self.log_error(msg)
            raise ValueError(msg)
        number_density = (
            _molecule_number_density(self.density, self.template_metrics.molar_mass_g_mol)
            * self.density_scale
        )
        return max(0, int(round(self._volume() * number_density)))

    def _extent(self) -> float:
        return self.radius if self.shape == "sphere" else float(self.box_size) / 2.0

    def _depth(self, center: np.ndarray) -> float:
        """Geometry-aware distance from the cluster centre.

        Euclidean for spheres, Chebyshev for cubes, so that trimming the
        outermost molecules keeps a compact, uniformly filled cluster of the
        requested shape rather than carving a sphere out of a box.
        """
        if self.shape == "sphere":
            return float(np.linalg.norm(center))
        return float(np.max(np.abs(center)))

    def _template_molecule_radius(self) -> float:
        return max(
            np.linalg.norm(
                self.template.coords[group] - self.template.coords[group].mean(axis=0),
                axis=1,
            ).max()
            for group in self.template.groups
        )

    def _molecule_centers(
        self,
        coords: np.ndarray,
        groups: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        return {tag: coords[indices].mean(axis=0) for tag, indices in groups.items()}

    @staticmethod
    def _groups_from_tags(tags: np.ndarray) -> dict[int, np.ndarray]:
        groups: dict[int, list[int]] = {}
        for idx, tag in enumerate(tags.tolist()):
            groups.setdefault(int(tag), []).append(idx)
        return {tag: np.asarray(indices, dtype=np.int64) for tag, indices in groups.items()}

    def _template_molar_mass(self) -> float:
        first_group_symbols = [self.template.symbols[i] for i in self.template.groups[0]]
        if self.solv_name == "water":
            return WATER_MOLAR_MASS_G_MOL
        return _molecular_weight(first_group_symbols)

    def _validate_custom_template_molecules(self) -> None:
        if not self.uses_custom_template:
            return

        first_group = self.template.groups[0]
        first_formula = _formula(self.template.symbols[i] for i in first_group)
        if self.solv_name == "water":
            expected = (("H", 2), ("O", 1))
            if first_formula != expected:
                msg = (
                    f"Custom solvent_pdb template {self.data_path} is not grouped as "
                    "one water molecule per residue: first residue formula is "
                    f"{_format_formula(first_formula)}, expected H2O. Ensure the PDB is "
                    "a pure solvent box with one residue per water molecule."
                )
                self.log_error(msg)
                raise ValueError(msg)

        for group_index, group in enumerate(self.template.groups, start=1):
            group_coords = self.template.coords[group]
            group_symbols = [self.template.symbols[i] for i in group]
            if not _is_single_covalent_component(group_coords, group_symbols):
                msg = (
                    f"Custom solvent_pdb template {self.data_path} residue group "
                    f"{group_index} is split into multiple covalent components. "
                    "Ensure the PDB contains one residue per solvent molecule; "
                    "do not merge multiple disconnected molecules into one residue."
                )
                self.log_error(msg)
                raise ValueError(msg)

        for group_index, group in enumerate(self.template.groups[1:], start=2):
            formula = _formula(self.template.symbols[i] for i in group)
            if formula != first_formula:
                msg = (
                    f"Custom solvent_pdb template {self.data_path} is not a homogeneous "
                    "pure-solvent template: residue group "
                    f"{group_index} formula {_format_formula(formula)} differs from "
                    f"the first residue formula {_format_formula(first_formula)}. "
                    "Ensure one residue per solvent molecule and do not mix solvent "
                    "species in one template."
                )
                self.log_error(msg)
                raise ValueError(msg)

    def _template_period(self) -> np.ndarray:
        """Orthorhombic CRYST1 period for stack-style bulk-solvent tiling."""
        template_label = self._template_label()
        cellpar = self.template.cryst1_cellpar
        if cellpar is None:
            msg = (
                f"Solvent template {template_label} has no CRYST1 cell. "
                "Stack-based explicit solvation requires a periodic bulk template."
            )
            self.log_error(msg)
            raise ValueError(msg)
        a, b, c, alpha, beta, gamma = cellpar
        if not (a > 0 and b > 0 and c > 0):
            msg = f"Solvent template {template_label} has invalid CRYST1 lengths."
            self.log_error(msg)
            raise ValueError(msg)
        if not all(abs(angle - 90.0) < 1.0 for angle in (alpha, beta, gamma)):
            msg = (
                f"Solvent template {template_label} uses a non-orthogonal CRYST1 cell. "
                "Only orthorhombic stack templates are currently supported."
            )
            self.log_error(msg)
            raise ValueError(msg)
        cell = np.array([a, b, c], dtype=np.float64)
        if np.any(cell < 2.0 * self._template_molecule_radius()):
            msg = (
                f"Solvent template {template_label} CRYST1 cell is smaller than "
                "the solvent molecule diameter; refusing unsafe tiling."
            )
            self.log_error(msg)
            raise ValueError(msg)
        return cell

    def _template_metrics(self) -> TemplateMetrics:
        period = self._template_period()
        molar_mass = self._template_molar_mass()
        density = _mass_density_g_ml(len(self.template.groups), molar_mass, float(np.prod(period)))
        return TemplateMetrics(
            period=period,
            molar_mass_g_mol=molar_mass,
            density_g_ml=density,
        )

    def _validate_template_density(self) -> None:
        """Fail closed when a bulk template cannot support the requested density."""
        if self.number is not None or self.density is None:
            return
        rel_error = abs(self.template_metrics.density_g_ml - self.density) / self.density
        if self.uses_custom_template and "density" not in self.params:
            if rel_error <= CUSTOM_TEMPLATE_DENSITY_TOLERANCE:
                return
            msg = (
                f"Custom solvent_pdb template {self.data_path} density "
                f"({self.template_metrics.density_g_ml:.4f} g/mL from CRYST1) differs "
                f"from the default density for explicit='{self.solv_name}' "
                f"({self.density:.4f} g/mL) by {rel_error:.1%}. "
                "Refusing to infer the target density from the solvent name for a "
                "user-provided template. Use a validated bulk template closer to the "
                f"default density, or add density={self.template_metrics.density_g_ml:.4f} "
                "to #solv(...) to continue knowingly with this template density, "
                "or provide number=<int> for an explicit non-density-targeted count."
            )
            self.log_error(msg)
            raise ValueError(msg)
        if rel_error <= DEFAULT_TEMPLATE_DENSITY_TOLERANCE:
            return
        msg = (
            f"Solvent template {self._template_label()} density "
            f"({self.template_metrics.density_g_ml:.4f} g/mL from CRYST1) differs "
            f"from requested density ({self.density:.4f} g/mL) by "
            f"{rel_error:.1%}; refusing to generate a false-density solvent cluster. "
            "Use a validated bulk template, or provide number=<int> for an explicit "
            "non-density-targeted count."
        )
        self.log_error(msg)
        raise ValueError(msg)

    def _tile_template_network(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        coords = self.template.coords
        symbols = self.template.symbols
        atom_names = self.template.atom_names
        residue_names = self.template.residue_names
        tags = self.template.residue_ids
        span = coords.max(axis=0) - coords.min(axis=0)
        span = np.where(span > 1.0e-6, span, 20.0)
        period = self.template_metrics.period
        origin = coords.min(axis=0) + span / 2.0
        centered = coords - origin
        extent = self._extent()
        max_mol_radius = self._template_molecule_radius()
        tile_radius = max(0, int(math.ceil((extent + max_mol_radius) / float(period.min()))))
        if self.randomize:
            tile_radius = max(tile_radius, 1)
            phase = self.rng.random(3) * period
            rotation = _uniform_quaternion_rotation(self.rng)
        else:
            phase = np.zeros(3, dtype=np.float64)
            rotation = np.eye(3, dtype=np.float64)

        coords_all = []
        symbols_all: list[str] = []
        atom_names_all: list[str] = []
        residue_names_all: list[str] = []
        tags_all = []
        base_count = len(self.template.groups)
        tile_index = 0
        for i in range(-tile_radius, tile_radius + 1):
            for j in range(-tile_radius, tile_radius + 1):
                for k in range(-tile_radius, tile_radius + 1):
                    tile_shift = np.array([i, j, k], dtype=np.float64) * period
                    block = centered + tile_shift - phase
                    block = block @ rotation.T
                    coords_all.append(block)
                    symbols_all.extend(symbols.tolist())
                    atom_names_all.extend(atom_names.tolist())
                    residue_names_all.extend(residue_names.tolist())
                    tags_all.append(tags + base_count * tile_index)
                    tile_index += 1

        return (
            np.vstack(coords_all),
            np.asarray(symbols_all, dtype=object),
            np.asarray(atom_names_all, dtype=object),
            np.asarray(residue_names_all, dtype=object),
            np.concatenate(tags_all).astype(np.int64),
        )

    def _crop_tags(self, coords: np.ndarray, tags: np.ndarray) -> list[int]:
        groups = self._groups_from_tags(tags)
        centers = self._molecule_centers(coords, groups)
        keep: list[int] = []
        # Keep a whole molecule when its centre lies within the requested
        # geometry. Cropping by centre keeps molecules intact (atoms may extend
        # at most one molecular radius past the boundary) and guarantees the
        # cluster never reports a density inflated by out-of-bounds molecules.
        extent = self._extent()
        if self.shape == "sphere":
            radius2 = extent ** 2
            for tag, center in centers.items():
                if float(center @ center) <= radius2:
                    keep.append(tag)
        else:
            for tag, center in centers.items():
                if bool((np.abs(center) <= extent).all()):
                    keep.append(tag)
        return keep

    def _selected_atom_indices(
        self, tags: np.ndarray, molecule_tags: Sequence[int]
    ) -> np.ndarray:
        groups = self._groups_from_tags(tags)
        if not molecule_tags:
            return np.empty(0, dtype=np.int64)
        return np.concatenate([groups[tag] for tag in molecule_tags])

    def _solute_clash_filter(
        self,
        coords: np.ndarray,
        symbols: np.ndarray,
        tags: np.ndarray,
        molecule_tags: list[int],
    ) -> list[int]:
        if not molecule_tags or len(self.atoms) == 0:
            return molecule_tags
        sel = self._selected_atom_indices(tags, molecule_tags)
        probe_coords = coords[sel]
        solute_coords = self.atoms.get_positions()
        if self.clash_method == "vdw":
            probe_radii = _element_vdw_radii(symbols[sel].tolist(), self.vdw_fallback_radius)
            solute_radii = _element_vdw_radii(
                self.atoms.get_chemical_symbols(), self.vdw_fallback_radius
            )
        else:
            probe_radii = solute_radii = None
        clash_atoms = _cross_clash_atoms(
            probe_coords, probe_radii, solute_coords, solute_radii,
            self.vdw_scale, self.clash_method, self.tolerance,
        )
        clash_tags = {int(tags[sel[k]]) for k in clash_atoms}
        return [tag for tag in molecule_tags if tag not in clash_tags]

    def _solvent_self_clash_filter(
        self,
        coords: np.ndarray,
        symbols: np.ndarray,
        tags: np.ndarray,
        molecule_tags: list[int],
    ) -> list[int]:
        """Drop solvent molecules that overlap each other at tiling seams.

        The bundled boxes are not minimum-image periodic, so stacking copies and
        applying a random phase can place atoms of adjacent images on top of one
        another. A greedy pass keeps the inner molecule of every clashing pair and
        removes the outer one, guaranteeing a clash-free network (mirrors the
        solvent-solvent overlap removal in gmx solvate).
        """
        if len(molecule_tags) < 2:
            return molecule_tags
        sel = self._selected_atom_indices(tags, molecule_tags)
        sub_coords = coords[sel]
        sub_tags = tags[sel]
        if self.clash_method == "vdw":
            radii = _element_vdw_radii(symbols[sel].tolist(), self.vdw_fallback_radius)
        else:
            radii = None
        pairs = _self_clash_atom_pairs(
            sub_coords, radii, self.vdw_scale, self.clash_method, self.tolerance
        )
        conflicts: dict[int, set[int]] = defaultdict(set)
        for i, j in pairs:
            ti, tj = int(sub_tags[i]), int(sub_tags[j])
            if ti != tj:
                conflicts[ti].add(tj)
                conflicts[tj].add(ti)
        if not conflicts:
            return molecule_tags
        groups = self._groups_from_tags(tags)
        centers = self._molecule_centers(coords, {tag: groups[tag] for tag in molecule_tags})
        order = sorted(molecule_tags, key=lambda tag: self._depth(centers[tag]))
        accepted: set[int] = set()
        for tag in order:
            if any(neighbor in accepted for neighbor in conflicts.get(tag, ())):
                continue
            accepted.add(tag)
        return [tag for tag in molecule_tags if tag in accepted]

    def _cap_to_target(
        self, coords: np.ndarray, tags: np.ndarray, molecule_tags: list[int]
    ) -> tuple[list[int], int]:
        if self.target_count <= 0:
            return [], len(molecule_tags)
        if len(molecule_tags) <= self.target_count:
            return molecule_tags, 0
        groups = self._groups_from_tags(tags)
        centers = self._molecule_centers(coords, {tag: groups[tag] for tag in molecule_tags})
        # Keep the innermost `target_count` molecules so the trimmed cluster is
        # compact and void-free (deterministic, no random subselection).
        ordered = sorted(molecule_tags, key=lambda tag: self._depth(centers[tag]))
        kept = set(ordered[: self.target_count])
        capped = [tag for tag in molecule_tags if tag in kept]
        return capped, len(molecule_tags) - len(capped)

    def _assemble_solvent(
        self,
        coords: np.ndarray,
        symbols: np.ndarray,
        atom_names: np.ndarray,
        residue_names: np.ndarray,
        tags: np.ndarray,
        molecule_tags: list[int],
    ) -> tuple[np.ndarray, list[str], list[str], list[str], np.ndarray]:
        if not molecule_tags:
            return np.empty((0, 3), dtype=np.float64), [], [], [], np.empty(0, dtype=np.int64)
        order = {tag: i for i, tag in enumerate(molecule_tags)}
        keep = np.isin(tags, molecule_tags)
        indices = np.where(keep)[0]
        indices = sorted(indices.tolist(), key=lambda idx: (order[int(tags[idx])], idx))
        final_indices = np.asarray(indices, dtype=np.int64)
        remap = {tag: i for i, tag in enumerate(molecule_tags)}
        final_tags = np.asarray([remap[int(tags[idx])] for idx in final_indices], dtype=np.int64)
        return (
            coords[final_indices],
            symbols[final_indices].tolist(),
            atom_names[final_indices].tolist(),
            residue_names[final_indices].tolist(),
            final_tags,
        )

    def _set_nonperiodic_metadata(
        self,
        solvent_tags: np.ndarray,
        solvent_atom_names: Sequence[str],
        solvent_res_names: Sequence[str],
    ) -> None:
        self.atoms.set_pbc([False, False, False])
        self.atoms.set_cell(np.zeros((3, 3)))
        natoms = len(self.atoms)
        molecule_ids = np.full(natoms, -1, dtype=np.int64)
        resnames = np.empty(natoms, dtype="U5")
        atom_names = np.empty(natoms, dtype="U5")
        solute_symbols = self.atoms.get_chemical_symbols()[: self.solute_count]
        resnames[: self.solute_count] = "MOL"
        atom_names[: self.solute_count] = [str(sym) for sym in solute_symbols]
        if len(solvent_tags) > 0:
            molecule_ids[self.solute_count:] = solvent_tags
            atom_names[self.solute_count:] = [str(name) for name in solvent_atom_names]
            resnames[self.solute_count:] = [str(name) for name in solvent_res_names]
        for key in ("maple_molecule_id", "maple_resname", "maple_atom_name"):
            if key in self.atoms.arrays:
                del self.atoms.arrays[key]
        self.atoms.new_array("maple_molecule_id", molecule_ids)
        self.atoms.new_array("maple_resname", resnames)
        self.atoms.new_array("maple_atom_name", atom_names)
        self.atoms.info["maple_explicit_solvent"] = "non-periodic coordinate-only cluster"

    def _base_path(self) -> Path:
        return Path(os.path.splitext(self.output)[0])

    def _write_coordinate_outputs(self) -> tuple[Path, Path]:
        base = self._base_path()
        xyz_path = base.with_name(base.name + "_solvated.xyz")
        pdb_path = base.with_name(base.name + "_solvated.pdb")
        comment = "MAPLE explicit solvent cluster; non-periodic coordinate-only model"
        _write_xyz(xyz_path, self.atoms, comment)
        _write_pdb(pdb_path, self.atoms)
        return xyz_path, pdb_path

    def _shell_atom_indices(self) -> np.ndarray:
        if not self.write_shell:
            return np.arange(len(self.atoms), dtype=np.int64)
        molecule_ids = self.atoms.arrays["maple_molecule_id"]
        solute_indices = np.arange(self.solute_count, dtype=np.int64)
        solvent_indices = np.arange(self.solute_count, len(self.atoms), dtype=np.int64)
        if len(solvent_indices) == 0:
            return solute_indices
        solute_positions = self.atoms.positions[solute_indices]
        keep_molecules: set[int] = set()
        for molecule_id in sorted(set(molecule_ids[solvent_indices].tolist())):
            mol_indices = solvent_indices[molecule_ids[solvent_indices] == molecule_id]
            distances = np.linalg.norm(
                self.atoms.positions[mol_indices][:, None, :] - solute_positions[None, :, :],
                axis=-1,
            )
            if float(distances.min()) <= float(self.shell_cutoff):
                keep_molecules.add(int(molecule_id))
        keep_solvent = [idx for idx in solvent_indices if int(molecule_ids[idx]) in keep_molecules]
        return np.asarray(solute_indices.tolist() + keep_solvent, dtype=np.int64)

    def _write_shell_outputs(self) -> Optional[tuple[Path, Path, int]]:
        if not self.write_shell:
            return None
        indices = self._shell_atom_indices()
        cluster = self.atoms[indices]
        cluster.set_pbc([False, False, False])
        cluster.set_cell(np.zeros((3, 3)))
        base = self._base_path()
        xyz_path = base.with_name(base.name + "_cluster.xyz")
        pdb_path = base.with_name(base.name + "_cluster.pdb")
        comment = (
            "MAPLE explicit solvent shell cluster; "
            f"cutoff={float(self.shell_cutoff):.3f} Å; non-periodic"
        )
        _write_xyz(xyz_path, cluster, comment)
        _write_pdb(pdb_path, cluster)
        cluster_ids = cluster.arrays["maple_molecule_id"]
        solvent_molecules = len(set(mid for mid in cluster_ids.tolist() if mid >= 0))
        return xyz_path, pdb_path, solvent_molecules

    def _min_distances(self) -> tuple[float, float]:
        """Min solute-solvent and min intermolecular solvent-solvent distances."""
        positions = self.atoms.get_positions()
        molecule_ids = self.atoms.arrays["maple_molecule_id"]
        solvent_mask = molecule_ids >= 0
        solute = positions[~solvent_mask]
        solvent = positions[solvent_mask]
        solvent_ids = molecule_ids[solvent_mask]
        min_solute = math.nan
        min_solvent = math.nan
        if len(solvent) and len(solute):
            tree = cKDTree(solute)
            distances, _ = tree.query(solvent, k=1)
            min_solute = float(distances.min())
        if len(solvent) > 1:
            tree = cKDTree(solvent)
            _, counts = np.unique(solvent_ids, return_counts=True)
            k = min(len(solvent), int(counts.max()) + 8)
            distances, neighbors = tree.query(solvent, k=k)
            best = math.inf
            for a in range(len(solvent)):
                for col in range(1, k):
                    b = neighbors[a, col]
                    if solvent_ids[a] != solvent_ids[b]:
                        best = min(best, float(distances[a, col]))
                        break
            min_solvent = best if math.isfinite(best) else math.nan
        return min_solute, min_solvent

    def _final_solvent_clashes(self) -> int:
        molecule_ids = self.atoms.arrays["maple_molecule_id"]
        mask = molecule_ids >= 0
        if int(mask.sum()) < 2:
            return 0
        coords = self.atoms.get_positions()[mask]
        ids = molecule_ids[mask]
        symbols = [s for s, keep in zip(self.atoms.get_chemical_symbols(), mask) if keep]
        radii = (
            _element_vdw_radii(symbols, self.vdw_fallback_radius)
            if self.clash_method == "vdw"
            else None
        )
        pairs = _self_clash_atom_pairs(
            coords, radii, self.vdw_scale, self.clash_method, self.tolerance
        )
        return sum(1 for i, j in pairs if ids[i] != ids[j])

    def _final_solute_clashes(self) -> int:
        molecule_ids = self.atoms.arrays["maple_molecule_id"]
        solvent_mask = molecule_ids >= 0
        if not solvent_mask.any() or bool(solvent_mask.all()):
            return 0
        coords = self.atoms.get_positions()
        symbols = self.atoms.get_chemical_symbols()
        solvent_indices = np.where(solvent_mask)[0]
        solute_indices = np.where(~solvent_mask)[0]
        if self.clash_method == "vdw":
            solvent_radii = _element_vdw_radii(
                [symbols[i] for i in solvent_indices], self.vdw_fallback_radius
            )
            solute_radii = _element_vdw_radii(
                [symbols[i] for i in solute_indices], self.vdw_fallback_radius
            )
        else:
            solvent_radii = solute_radii = None
        clashes = _cross_clash_atoms(
            coords[solvent_indices],
            solvent_radii,
            coords[solute_indices],
            solute_radii,
            self.vdw_scale,
            self.clash_method,
            self.tolerance,
        )
        return len(clashes)

    def _validate_final_cluster(self, final_count: int) -> None:
        if not np.isfinite(self.atoms.get_positions()).all():
            msg = "Explicit solvent generation produced non-finite coordinates."
            self.log_error(msg)
            raise ValueError(msg)

        solute_clashes = self._final_solute_clashes()
        if solute_clashes:
            msg = (
                "Explicit solvent generation left "
                f"{solute_clashes} solute-solvent clash atom(s)."
            )
            self.log_error(msg)
            raise ValueError(msg)

        solvent_clashes = self._final_solvent_clashes()
        if solvent_clashes:
            msg = (
                "Explicit solvent generation left "
                f"{solvent_clashes} intermolecular solvent clash pair(s)."
            )
            self.log_error(msg)
            raise ValueError(msg)

        if self.number is None and self.target_count > 0:
            fill_ratio = final_count / self.target_count
            if fill_ratio < MIN_FINAL_TARGET_RATIO:
                self.log_info([
                    "WARNING: explicit solvent cluster is below the full-volume "
                    f"density target: final={final_count}, target={self.target_count}, "
                    f"fill={fill_ratio:.1%}. The target is computed from the full "
                    "sphere/cube volume and does not subtract solute excluded volume; "
                    "for bulky solutes or tight shells this can be physically "
                    "reasonable. Increase the geometry size for a denser shell, or "
                    "use number=<int> for an explicit molecule count.\n"
                ])

    def _log_setup(self) -> None:
        if self.shape == "sphere":
            geometry = f"radius={self.radius:.3f} Å"
        else:
            geometry = f"box_size={float(self.box_size):.3f} Å"
        if self.padding is not None:
            envelope = (
                "radial solute envelope"
                if self.shape == "sphere"
                else "axis-aligned solute envelope"
            )
            geometry += f", padding={self.padding:.3f} Å from {envelope}"
        if self.number is not None:
            target_line = f"• Target source: explicit molecule count (number={self.number})\n"
        elif self.density is not None:
            target_line = (
                f"• Density target: {self.density:.4f} g/mL × {self.density_scale:.4f}\n"
            )
        else:
            target_line = "• Target source: unavailable density (requires number=<int>)\n"
        lines = [
            "\n\n" + "-" * 70 + "\n",
            f"{'Explicit Solvent Cluster Setup'.center(70)}\n\n",
            f"• Solvent type: {self.solv_name}\n",
            f"• Solvent template PDB: {self.data_path}\n",
            f"• Shape: {self.shape} ({geometry})\n",
            "• Model: non-periodic coordinate-only cluster (no PBC/cell metadata)\n",
            target_line,
            f"• Target solvent molecules: {self.target_count}\n",
            f"• Solute-solvent clash method: {self._clash_method_label()}\n",
            f"• Solvent-solvent overlap removal: {self._clash_method_label()}\n",
            f"• Randomize template sampling: {self.randomize}\n",
        ]
        if self.randomize:
            lines.append(
                "• Randomization: template phase shift + uniform-quaternion "
                "rigid rotation; solvent network geometry is not independently "
                "randomized\n"
            )
        if self.write_shell:
            lines.append(f"• Shell sidecar cutoff: {float(self.shell_cutoff):.3f} Å\n")
        lines.extend(self._geometry_warning_lines())
        lines.append("\n")
        self.log_info(lines)

    def _process(self):
        self.atoms.positions -= self.solute_center

        coords, symbols, atom_names, residue_names, tags = self._tile_template_network()
        candidate_count = len(set(tags.tolist()))

        cropped_tags = self._crop_tags(coords, tags)
        cropped_count = len(cropped_tags)

        clash_kept_tags = self._solute_clash_filter(coords, symbols, tags, cropped_tags)
        solute_clash_removed = cropped_count - len(clash_kept_tags)

        declashed_tags = self._solvent_self_clash_filter(coords, symbols, tags, clash_kept_tags)
        solvent_clash_removed = len(clash_kept_tags) - len(declashed_tags)

        final_tags, density_removed = self._cap_to_target(coords, tags, declashed_tags)

        (
            solvent_positions,
            solvent_symbols,
            solvent_atom_names,
            solvent_res_names,
            solvent_tags,
        ) = self._assemble_solvent(coords, symbols, atom_names, residue_names, tags, final_tags)
        if solvent_symbols:
            self.atoms += Atoms(symbols=solvent_symbols, positions=solvent_positions)
        self._set_nonperiodic_metadata(solvent_tags, solvent_atom_names, solvent_res_names)

        final_count = len(final_tags)
        self._validate_final_cluster(final_count)

        xyz_path, pdb_path = self._write_coordinate_outputs()
        shell_result = self._write_shell_outputs()

        volume = self._volume()
        actual_density = final_count / volume if volume > 0 else 0.0
        target_density = self.target_count / volume if volume > 0 else 0.0
        actual_mass_density = _mass_density_g_ml(
            final_count, self.template_metrics.molar_mass_g_mol, volume
        )
        min_solute, min_solvent = self._min_distances()
        lines = [
            "Explicit solvent coordinate generation summary:\n",
            (
                f"candidate={candidate_count} cropped={cropped_count} "
                f"solute_clash_removed={solute_clash_removed} "
                f"solvent_clash_removed={solvent_clash_removed} "
                f"density_trimmed={density_removed} final={final_count}\n"
            ),
            f"target_number_density={target_density:.8f} molecules/Å^3\n",
            f"actual_number_density={actual_density:.8f} molecules/Å^3\n",
            f"template_mass_density={self.template_metrics.density_g_ml:.4f} g/mL\n",
            f"actual_cluster_mass_density={actual_mass_density:.4f} g/mL\n",
            f"min_solute_solvent_distance={min_solute:.3f} Å\n",
            f"min_solvent_solvent_distance={min_solvent:.3f} Å\n",
            f"solute_solvent_clash_method={self._clash_method_label()}\n",
            f"Added {len(solvent_positions)} solvent atoms.\n",
            f"Solvated XYZ written to: {xyz_path}\n",
            f"Solvated PDB written to: {pdb_path}\n",
        ]
        if final_count < self.target_count:
            lines.append(
                "WARNING: fewer non-clashing solvent molecules were available than "
                "the density target after seam-overlap removal. Increase the "
                "geometry size or relax the clash criterion if a denser cluster is "
                "required.\n"
            )
        if shell_result is not None:
            shell_xyz, shell_pdb, shell_molecules = shell_result
            lines.extend([
                f"Shell cluster solvent molecules: {shell_molecules}\n",
                f"Shell XYZ written to: {shell_xyz}\n",
                f"Shell PDB written to: {shell_pdb}\n",
            ])
        self.log_info(lines)
