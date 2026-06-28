import os
from typing import List, Optional

from ase import Atoms
from ase.constraints import FixInternals
from ase.neighborlist import NeighborList, natural_cutoffs

from ..jobABC import JobABC

from maple.function.timer import timer

class Scan(JobABC):
    """
    N-dimensional relaxed/rigid scan (supports 1D, 2D, 3D).
    Uses hierarchical continuous scanning strategy with streaming output.
    """

    def __init__(self, output: str, atoms: Atoms, method: str = "lbfgs", 
                 constraints: Optional[list] = None, params: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        self.output = output
        self.params = params if params is not None else {}
        self.initial_calc = atoms.calc
        self.method = str(method or self.params.get("method") or "lbfgs").lower()
        self.mode = str(self.params.get("mode", "relaxed")).lower()
        if self.mode not in ("relaxed", "rigid"):
            raise ValueError(f"mode must be 'relaxed' or 'rigid', got: {self.mode}")
        self._adj = self._build_connectivity(self.atoms)
        if constraints is None:
            raise ValueError("Constraints must be provided for scan.")
        self.constraints = self._convert_constraints(constraints)
        
        # Save initial custom threshold attributes
        self._threshold_attrs = ["f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th"]
        self._initial_thresholds = {}
        for attr in self._threshold_attrs:
            self._initial_thresholds[attr] = getattr(self.atoms, attr, 1e10)
        
        # Initialize XYZ file handle
        self.xyz_file = None

    def _convert_constraints(self, original_constraints: list) -> list:
        """Normalize constraint definitions."""
        converted = []
        for c in original_constraints:
            if len(c) == 4:     # distance: [a1, a2, step, steps]
                a1, a2, step, steps = c
                converted.append({
                    "type": "distance",
                    "atoms": [a1, a2],
                    "step": step,
                    "steps": steps
                })
            elif len(c) == 5:   # angle: [a1, a2, a3, step, steps]
                a1, a2, a3, step, steps = c
                converted.append({
                    "type": "angle",
                    "atoms": [a1, a2, a3],
                    "step": step,
                    "steps": steps
                })
            elif len(c) == 6:   # dihedral: [a1, a2, a3, a4, step, steps]
                a1, a2, a3, a4, step, steps = c
                converted.append({
                    "type": "dihedral",
                    "atoms": [a1, a2, a3, a4],
                    "step": step,
                    "steps": steps
                })
            else:
                raise ValueError(f"Unsupported constraint format: {c}")
        return converted

    def _generate_scan_values(self) -> List[List[float]]:
        """Generate scan grid values for each dimension."""
        scan_values = []
        for con in self.constraints:
            ctype = con["type"]
            step = con["step"]
            steps = con["steps"]
            # Convert atom indices to 0-based
            atoms_idx = [a - 1 for a in con["atoms"]]

            if ctype == "distance":
                initial = self.atoms.get_distance(*atoms_idx)
            elif ctype == "angle":
                initial = self.atoms.get_angle(*atoms_idx)
            elif ctype == "dihedral":
                initial = self.atoms.get_dihedral(*atoms_idx)
            else:
                raise ValueError(f"Unknown constraint type: {ctype}")

            values = [initial + i * step for i in range(steps + 1)]
            scan_values.append(values)
        return scan_values

    def _build_fix_internals(self, current_values: List[float]) -> FixInternals:
        """Build FixInternals constraint for given values."""
        bonds, angles, dihedrals = [], [], []

        for idx, con in enumerate(self.constraints):
            ctype = con["type"]
            atoms_idx = [a - 1 for a in con["atoms"]]  # convert to 0-based
            val = current_values[idx]

            if ctype == "distance":
                bonds.append([val, atoms_idx])
            elif ctype == "angle":
                angles.append([val, atoms_idx])
            elif ctype == "dihedral":
                dihedrals.append([val, atoms_idx])

        return FixInternals(
            bonds=bonds if bonds else None,
            angles_deg=angles if angles else None,
            dihedrals_deg=dihedrals if dihedrals else None
        )

    def _safe_copy(self, atoms: Atoms) -> Atoms:
        """
        Create a deep copy of Atoms with calculator and custom attributes restored.
        """
        new_atoms = atoms.copy()
        new_atoms.info = dict(atoms.info)
        
        # Restore calculator
        new_atoms.calc = atoms.calc if atoms.calc is not None else self.initial_calc
        
        # Restore custom threshold attributes
        for attr, val in self._initial_thresholds.items():
            setattr(new_atoms, attr, val)

        return new_atoms

    def _print_progress(self, idx: int, total: int, coord: List[float]):
        """Print progress header before calling optimizer."""
        coord_str = "[" + ", ".join(f"{v:.2f}" for v in coord) + "]"
        self.log_info(["\n"])
        self.log_info(["-" * 70])
        self.log_info([f"\n            Scanning combination {idx}/{total}: {coord_str}\n"])

    def _apply_constraints(self, atoms: Atoms, coord: List[float]) -> Atoms:
        """
        Apply FixInternals constraint to atoms (in-place modification).
        Returns the same atoms object with constraint applied.
        """
        constraint = self._build_fix_internals(coord)
        atoms.set_constraint(constraint)  # in-place
        
        # Ensure calculator remains valid
        if atoms.calc is None:
            atoms.calc = self.initial_calc
        
        return atoms

    def _apply_rigid_geometry(self, atoms: Atoms, coord: List[float]) -> Atoms:
        """ 
        Moving rigid body subgroup according to constraints
        Recongnize the moving fragment automatically.
        """
        atoms.set_constraint(None)
    
        for idx, con in enumerate(self.constraints):
            val = coord[idx]
            ctype = con["type"]
            if ctype == "distance":
                mask, a0, a1 = self._get_rigid_mask(con)
                atoms.set_distance(a0, a1, val, fix=0, mask=mask)  # fix a0, move a1 fragment
            elif ctype == "angle":
                mask, a1, a2, a3 = self._get_rigid_mask(con)
                atoms.set_angle(a1, a2, a3, val, mask=mask)
            elif ctype == "dihedral":
                mask, a1, a2, a3, a4 = self._get_rigid_mask(con)
                atoms.set_dihedral(a1, a2, a3, a4, val, mask=mask)
            else:
                raise ValueError(f"Unknown constraint type: {ctype}")
    
        if atoms.calc is None:
            atoms.calc = self.initial_calc
        return atoms

    def _run_optimizer(self, atoms: Atoms) -> Atoms:
        """Run geometry optimization."""
        if self.mode == "rigid":
            return atoms
        from maple.function.dispatcher.optimization import Optimization

        params = dict(self.params)
        params["method"] = self.method
        params["verbose"] = 0  # suppress optimizer output inside each scan point
        params["log_final_paths"] = False  # scan removes per-point optimizer temp files
        return Optimization(params=params, output=self.output, atoms=atoms).run()

    def _record_result(self, atoms: Atoms, coord: List[float],
                       coords_list: list, energies: list, energy=None):
        """Record a scan point result by streaming to file.

        ``energy`` (optional): pre-computed potential energy (Hartree). The serial
        path leaves it None and reads ``atoms.get_potential_energy`` through the
        per-atoms calculator; the OPT-IN batched path passes the energy from the
        single batched ``get_ef_gpu`` forward so no per-structure calc is needed.
        """
        # Get energy and structure info
        e = (float(energy) if energy is not None
             else float(atoms.get_potential_energy(force_consistent=True)))
        pos = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        
        # Write to XYZ file immediately
        self.xyz_file.write(f"{len(symbols)}\n")
        coord_str = "[" + ", ".join(f"{v:.4f}" for v in coord) + "]"
        self.xyz_file.write(
            f"Scanning combination {self._current_index}/{self._total_combinations}: "
            f"{coord_str}  Energy = {e:.10f}\n"
        )
        for s, (x, y, z) in zip(symbols, pos):
            self.xyz_file.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")
        self.xyz_file.flush()  # Ensure data is written
        
        # Store lightweight data
        coords_list.append(coord[:])
        energies.append(e)

        # For rigid scan, log coordinates to output
        if self.mode == "rigid":
            info = []
            info.append(f'\n{"Coordinates".center(70)}\n')
            info.append('-' * 70 + '\n')
            for atom_index, atom in enumerate(atoms):
                x, y, z = atom.position
                info.append(f"{atom_index:<4} {atom.symbol:<2} {x:>20.4f} {y:>20.4f} {z:>20.4f}\n")
            info.append(f"\n\nEnergy:                {e:>12.6f}\n")
            self.log_info(info)

    def _scan_1d(self, scan_values: List[List[float]]):
        """Execute 1D scan."""
        x_values = scan_values[0]
        coords_list, energies = [], []
        
        atoms_current = self._safe_copy(self.atoms)

        for xv in x_values:
            coord = [xv]
            self._current_index += 1
            self._print_progress(self._current_index, self._total_combinations, coord)
            if self.mode == "rigid":
                atoms_current = self._apply_rigid_geometry(atoms_current, coord)
            else:
                atoms_current = self._apply_constraints(atoms_current, coord)
            atoms_current = self._run_optimizer(atoms_current)
            self._record_result(atoms_current, coord, coords_list, energies)

        return coords_list, energies

    def _scan_2d(self, scan_values: List[List[float]]):
        """Execute 2D scan using hierarchical strategy."""
        x_values, y_values = scan_values[0], scan_values[1]
        coords_list, energies = [], []
        grid_xy = {}

        # Step 1: scan along X (y = y0) - keep this initial line
        atoms_current = self._safe_copy(self.atoms)
        for ix, xv in enumerate(x_values):
            coord = [xv, y_values[0]]
            self._current_index += 1
            self._print_progress(self._current_index, self._total_combinations, coord)

            if self.mode == "rigid":
                atoms_current = self._apply_rigid_geometry(atoms_current, coord)
            else:
                atoms_current = self._apply_constraints(atoms_current, coord)
            atoms_current = self._run_optimizer(atoms_current)
            
            grid_xy[(ix, 0)] = self._safe_copy(atoms_current)  # Keep initial line
            self._record_result(atoms_current, coord, coords_list, energies)

        # Step 2: for each X, scan along Y (no need to keep these)
        for ix, xv in enumerate(x_values):
            atoms_current = self._safe_copy(grid_xy[(ix, 0)])
            
            for iy in range(1, len(y_values)):
                coord = [xv, y_values[iy]]
                self._current_index += 1
                self._print_progress(self._current_index, self._total_combinations, coord)
                if self.mode == "rigid":
                    atoms_current = self._apply_rigid_geometry(atoms_current, coord)
                else:
                    atoms_current = self._apply_constraints(atoms_current, coord)
                atoms_current = self._run_optimizer(atoms_current)
                self._record_result(atoms_current, coord, coords_list, energies)

        return coords_list, energies

    def _scan_3d(self, scan_values: List[List[float]]):
        """Execute 3D scan using hierarchical strategy."""
        x_values, y_values, z_values = scan_values[0], scan_values[1], scan_values[2]
        coords_list, energies = [], []
        grid_xy = {}

        # Step 1: scan along X (y=y0, z=z0)
        atoms_current = self._safe_copy(self.atoms)
        for ix, xv in enumerate(x_values):
            coord = [xv, y_values[0], z_values[0]]
            self._current_index += 1
            self._print_progress(self._current_index, self._total_combinations, coord)

            if self.mode == "rigid":
                atoms_current = self._apply_rigid_geometry(atoms_current, coord)
            else:
                atoms_current = self._apply_constraints(atoms_current, coord)
            atoms_current = self._run_optimizer(atoms_current)
            
            grid_xy[(ix, 0)] = self._safe_copy(atoms_current)
            self._record_result(atoms_current, coord, coords_list, energies)

        # Step 2: scan along Y (z=z0) for each X - build the initial plane
        for ix, xv in enumerate(x_values):
            atoms_current = self._safe_copy(grid_xy[(ix, 0)])
            
            for iy in range(1, len(y_values)):
                coord = [xv, y_values[iy], z_values[0]]
                self._current_index += 1
                self._print_progress(self._current_index, self._total_combinations, coord)
                if self.mode == "rigid":
                    atoms_current = self._apply_rigid_geometry(atoms_current, coord)
                else:
                    atoms_current = self._apply_constraints(atoms_current, coord)
                atoms_current = self._run_optimizer(atoms_current)
                
                grid_xy[(ix, iy)] = self._safe_copy(atoms_current)  # Keep initial plane
                self._record_result(atoms_current, coord, coords_list, energies)
            
            # Can delete the first line point now (initial plane is complete)
            del grid_xy[(ix, 0)]

        # Step 3: scan along Z for each (X, Y)
        for ix, xv in enumerate(x_values):
            for iy, yv in enumerate(y_values):
                atoms_current = self._safe_copy(grid_xy[(ix, iy)])
                
                for iz, zv in enumerate(z_values):
                    # Skip z=z0 (already computed)
                    if iz == 0:
                        continue
                    
                    coord = [xv, yv, zv]
                    self._current_index += 1
                    self._print_progress(self._current_index, self._total_combinations, coord)
                    if self.mode == "rigid":
                        atoms_current = self._apply_rigid_geometry(atoms_current, coord)
                    else:
                        atoms_current = self._apply_constraints(atoms_current, coord)
                    atoms_current = self._run_optimizer(atoms_current)
                    self._record_result(atoms_current, coord, coords_list, energies)
                
                # Delete this (x,y) plane point after finishing its z-scan
                del grid_xy[(ix, iy)]

        return coords_list, energies

    # ==============================================================
    # OPT-IN GPU-batched scan path (Task: ai-maple-gpu)
    # ==============================================================
    def _resolve_batched_calc(self):
        """Return the batched calculator implementing the (prepare / get_ef_gpu /
        set_coords_ / step_cart_) contract.

        Priority: an explicit pre-built instance in ``params['batched_calc']``;
        else build a ``UMABatchCalc`` from ``params['batch_model_path']`` (with
        optional ``batch_device``/``batch_dtype``/``batch_task``). Keeping it
        injectable means Scan never hard-depends on UMA and the serial path keeps
        no torch/fairchem import.
        """
        calc = self.params.get("batched_calc")
        if calc is not None:
            return calc
        model_path = self.params.get("batch_model_path")
        if model_path is None:
            raise ValueError(
                "batched=True requires params['batched_calc'] (a prepared batched "
                "calculator) or params['batch_model_path'] to build a UMABatchCalc."
            )
        import torch
        from maple.function.calculator.uma._uma_batch_calculator import UMABatchCalc
        dtype = self.params.get("batch_dtype", torch.float64)
        return UMABatchCalc(
            model_path,
            device=self.params.get("batch_device", "cuda"),
            dtype=dtype,
            task=self.params.get("batch_task", "omol"),
        )

    def _build_rigid_geom(self, coord: List[float]) -> Atoms:
        """Build ONE grid geometry by applying the scan-CV target values as a rigid
        fragment move (no relaxation), starting from a fresh copy of the input
        geometry. Path-independent for disjoint single-fragment CVs, so each grid
        geometry is generated independently (no chain seed)."""
        at = self.atoms.copy()
        at.info = dict(self.atoms.info)
        self._apply_rigid_geometry(at, coord)   # sets constraint None + moves fragments
        at.calc = None
        at.set_constraint(None)
        return at

    def _grid_coords(self, scan_values: List[List[float]]) -> List[List[float]]:
        """All grid points as a flat list (lexicographic: dim0 outermost)."""
        import itertools
        return [list(c) for c in itertools.product(*scan_values)]

    def _scan_batched(self, scan_values: List[List[float]], dim: int):
        """Batched scan over the FULL grid in one (rigid) / one shrinking-batch
        (relaxed) GPU pass. Reproduces the serial PES; serial stays the oracle."""
        if dim not in (1, 2, 3):
            raise ValueError(f"Only 1D, 2D, 3D scans are supported, got {dim}D")
        calc = self._resolve_batched_calc()
        grid = self._grid_coords(scan_values)
        coords_list, energies = [], []

        # ---------------- RIGID: one batched forward over all geometries -------
        if self.mode == "rigid":
            geoms = [self._build_rigid_geom(coord) for coord in grid]
            calc.prepare(geoms)
            E, _F = calc.get_ef_gpu()                 # E (B,) Hartree
            E = [float(x) for x in E.detach().cpu().tolist()]
            for k, coord in enumerate(grid):
                self._current_index = k + 1
                self._record_result(geoms[k], coord, coords_list, energies,
                                     energy=E[k])
            return coords_list, energies

        # ---------------- RELAXED: batched constrained relaxations -------------
        # Two seedings (both batch the expensive UMA forward; default reproduces
        # the serial PES):
        #   'wavefront' (DEFAULT) -- serial-IDENTICAL seeds: a spine chain along
        #       dim0 then per-generation batched relaxations across the lower dims
        #       (the serial hierarchy, just batched across the independent axis).
        #       Same seeds as serial -> same constrained minimum (geometry+energy
        #       parity), still GPU-batched over each generation.
        #   'independent' (OPT-IN, fastest) -- every grid point seeded from its own
        #       rigid-CV geometry and relaxed in ONE shrinking batch. Reproduces
        #       the serial PES *energy* but, on floppy systems, may settle into a
        #       different near-degenerate constrained minimum (geometry differs).
        seed_mode = str(self.params.get("batched_seed", "wavefront")).lower()
        if seed_mode == "independent":
            return self._scan_relaxed_independent(calc, grid)
        return self._scan_relaxed_wavefront(calc, scan_values, dim)

    # ---- relaxed batched: optimizer kwargs (serial LBFGSParams defaults) -----
    def _lbfgs_kwargs(self):
        opt = self.params.get("lbfgs", {})
        opt = opt if isinstance(opt, dict) else {}
        return dict(
            memory=int(self.params.get("memory", opt.get("memory", 5))),
            curvature=float(self.params.get("curvature", opt.get("curvature", 70.0))),
            maxstep=float(self.params.get("max_step", opt.get("max_step", 0.2))),
            maxiter=int(self.params.get("max_iter", opt.get("max_iter", 256))),
            device=self.params.get("batch_device", "cuda"),
        )

    def _seed_atoms(self, positions) -> Atoms:
        """Fresh molecule copy at ``positions`` (no constraint, no calc)."""
        at = self.atoms.copy()
        at.info = dict(self.atoms.info)
        at.set_constraint(None)
        at.set_positions(positions, apply_constraint=False)
        at.calc = None
        return at

    def _relax_batch(self, calc, seed_positions, coords):
        """Relax a generation: build seed atoms (each at ``seed_positions[i]`` with
        FixInternals for ``coords[i]``) and run the batched constrained L-BFGS.
        Returns (final_atoms_list, final_E_list) in input order."""
        seeds = []
        for pos, coord in zip(seed_positions, coords):
            at = self._seed_atoms(pos)
            at.set_constraint(self._build_fix_internals(coord))
            for attr, val in self._initial_thresholds.items():
                setattr(at, attr, val)
            seeds.append(at)
        relaxer = _BatchConstrainedLBFGS(seeds, calc, **self._lbfgs_kwargs())
        final_atoms, final_E, _ = relaxer.run()
        return final_atoms, final_E

    def _scan_relaxed_independent(self, calc, grid):
        """Every grid point relaxed independently from its rigid-CV seed in ONE
        shrinking batch (fastest; energy-faithful, geometry may differ on floppy
        systems)."""
        coords_list, energies = [], []
        seed_positions = [self._build_rigid_geom(coord).get_positions()
                          for coord in grid]
        final_atoms, final_E = self._relax_batch(calc, seed_positions, grid)
        for k, coord in enumerate(grid):
            self._current_index = k + 1
            self._record_result(final_atoms[k], coord, coords_list, energies,
                                 energy=final_E[k])
        return coords_list, energies

    def _scan_relaxed_wavefront(self, calc, scan_values, dim):
        """Serial-identical-seed batched relaxed scan.

        Mirrors the serial hierarchy: a sequential spine chain along dim0 (other
        dims fixed at index 0), then for each higher dim k a sequence of
        generations along k, each generation batched across the FULL lower-dim
        block (dims < k) with dims > k pinned at 0. Each point is seeded from its
        relaxed dim-k neighbour exactly as the serial scan seeds it, so every
        constrained relaxation starts from the same geometry as serial and reaches
        the same minimum -- reproducing the serial PES (geometry AND energy) while
        batching every generation's UMA forwards."""
        import itertools
        shapes = [len(v) for v in scan_values]
        relaxed_pos = {}     # multi-index tuple -> relaxed positions (N,3)
        relaxed_E = {}       # multi-index tuple -> energy (Ha)

        def coord_of(idx):
            return [scan_values[d][idx[d]] for d in range(dim)]

        # ---- spine: chain along dim0 (all other dims = 0) --------------------
        for i0 in range(shapes[0]):
            idx = (i0,) + (0,) * (dim - 1)
            if i0 == 0:
                seed_pos = self.atoms.get_positions()            # input geometry
            else:
                seed_pos = relaxed_pos[(i0 - 1,) + (0,) * (dim - 1)]
            fa, fE = self._relax_batch(calc, [seed_pos], [coord_of(idx)])
            relaxed_pos[idx] = fa[0].get_positions()
            relaxed_E[idx] = fE[0]

        # ---- generations along each higher dim k, batched across dims < k ----
        for k in range(1, dim):
            for ik in range(1, shapes[k]):
                seed_positions, coords, idxs = [], [], []
                for lower in itertools.product(*[range(shapes[d]) for d in range(k)]):
                    idx = lower + (ik,) + (0,) * (dim - 1 - k)
                    prev = lower + (ik - 1,) + (0,) * (dim - 1 - k)
                    seed_positions.append(relaxed_pos[prev])
                    coords.append(coord_of(idx))
                    idxs.append(idx)
                fa, fE = self._relax_batch(calc, seed_positions, coords)
                for j, idx in enumerate(idxs):
                    relaxed_pos[idx] = fa[j].get_positions()
                    relaxed_E[idx] = fE[j]

        # ---- record in grid order (lexicographic, dim0 outermost) ------------
        coords_list, energies = [], []
        k = 0
        for idx in itertools.product(*[range(s) for s in shapes]):
            coord = coord_of(idx)
            k += 1
            self._current_index = k
            rec = self._seed_atoms(relaxed_pos[idx])
            self._record_result(rec, coord, coords_list, energies,
                                 energy=relaxed_E[idx])
        return coords_list, energies

    def run_scan(self):
        """Main scan entry point."""
        scan_values = self._generate_scan_values()
        dim = len(scan_values)

        total = 1
        for values in scan_values:
            total *= len(values)
        self._total_combinations = total
        self._current_index = 0

        # Open output XYZ file for streaming
        base, _ = os.path.splitext(self.output)
        xyz_filename = base + "_scan_final.xyz"
        
        try:
            self.xyz_file = open(xyz_filename, "w")

            if self.params.get("batched"):
                # OPT-IN GPU-batched path (rigid: one forward over all grid
                # geometries; relaxed: all constrained relaxations in one
                # shrinking batched L-BFGS). Serial path (below) is the oracle.
                coords_list, energies = self._scan_batched(scan_values, dim)
            elif dim == 1:
                coords_list, energies = self._scan_1d(scan_values)
            elif dim == 2:
                coords_list, energies = self._scan_2d(scan_values)
            elif dim == 3:
                coords_list, energies = self._scan_3d(scan_values)
            else:
                raise ValueError(f"Only 1D, 2D, 3D scans are supported, got {dim}D")
            
            self.log_info(["\n"])
            self.log_info(["=" * 70])
            self.log_info([f"\nScan completed! Total points: {len(energies)}"])
            self.log_info([f"Results saved to: {xyz_filename}\n"])
            self.log_info([f"Energy range: {min(energies):.6f} to {max(energies):.6f} Hartree\n"])
            
        finally:
            if self.xyz_file is not None:
                self.xyz_file.close()

    def run(self):
        """JobABC interface."""
        with timer("Scan"):
            self.run_scan()
            self._cleanup_opt_files(self.output)  # cleanup opt temp files

  
    @staticmethod
    def _cleanup_opt_files(output_path):
        from pathlib import Path
        base, _ = os.path.splitext(str(output_path))
        for f in (base + "_opt.xyz", base + "_opt_traj.xyz"):
            Path(f).unlink(missing_ok=True)

    def _build_connectivity(self, atoms: Atoms):
        """Build adjacency list(bond graph) based on neighbor list."""
        cutoffs = natural_cutoffs(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)
    
        n = len(atoms)
        adj = [set() for _ in range(n)]
        for i in range(n):
            neigh, _ = nl.get_neighbors(i)
            for j in neigh:
                j = int(j)
                adj[i].add(j)
                adj[j].add(i)
        return adj
    
    def _fragment(self, start: int, blocked_edge=None):
        """Find connected fragment from start, optionally blocking an edge."""
        # blocked_edge: tuple(u, v) meaning forbid traversing u<->v
        u, v = blocked_edge if blocked_edge else (None, None)
        stack = [start]
        seen = {start}
        while stack:
            i = stack.pop()
            for j in self._adj[i]:
                if blocked_edge is not None and ((i == u and j == v) or (i == v and j == u)):
                    continue
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        return seen
    
    def _mask_from_set(self, n, idx_set):
        mask = [False] * n
        for i in idx_set:
            mask[i] = True
        return mask
    
    def _get_rigid_mask(self, con):
        n = len(self.atoms)
        atoms_idx = [a - 1 for a in con["atoms"]]
        ctype = con["type"]
    
        if ctype == "distance":
            a0, a1 = atoms_idx
            blocked = (a0, a1) if a1 in self._adj[a0] else None
            frag = self._fragment(start=a1, blocked_edge=blocked)
            return self._mask_from_set(n, frag), a0, a1
    
        if ctype == "angle":
            a1, a2, a3 = atoms_idx
            blocked = (a2, a3) if a3 in self._adj[a2] else None
            frag = self._fragment(start=a3, blocked_edge=blocked)
            return self._mask_from_set(n, frag), a1, a2, a3
    
        if ctype == "dihedral":
            a1, a2, a3, a4 = atoms_idx
            blocked = (a2, a3) if a3 in self._adj[a2] else None
            frag = self._fragment(start=a4, blocked_edge=blocked)
            return self._mask_from_set(n, frag), a1, a2, a3, a4

        raise ValueError(ctype)


# ==================================================================== #
# Batched constrained L-BFGS for the OPT-IN relaxed batched scan.
#
# Reuses the BatchLBFGS skeleton (nmax-padded (B, nmax) state, batched two-loop
# recursion, dynamic per-structure batch shrinking, per-structure convergence
# masking) but replaces:
#   * the force source -> the SAME ASE FixInternals projection the SERIAL scan
#     uses (constraint.adjust_forces, per structure) so the constrained gradient
#     is byte-identical to the oracle;
#   * the position step -> ASE atoms.set_positions (which calls
#     constraint.adjust_positions: the iterative re-imposition of the scan CVs
#     onto the constraint manifold, identical to the serial relaxed point).
# The expensive part -- the UMA energy/force evaluation -- is the ONLY thing
# batched (one get_ef_gpu over all live grid points per iteration); the cheap
# geometric constraint solve runs per structure on CPU, guaranteeing the batched
# relaxed PES reproduces the serial relaxed PES (each point converges to the same
# constrained local minimum at the same fixed CV values).
# ==================================================================== #
class _BatchConstrainedLBFGS:

    def __init__(self, atoms_list, calc, memory=5, curvature=70.0,
                 maxstep=0.2, maxiter=256, device="cuda"):
        import numpy as np
        import torch
        self.np = np
        self.torch = torch
        self.DTYPE = torch.float64
        self.calc = calc
        self.memory = int(memory)
        self.curvature = float(curvature)
        self.maxstep = float(maxstep)
        self.maxiter = int(maxiter)
        self.device = torch.device(
            "cuda" if (str(device).startswith("cuda") and torch.cuda.is_available())
            else "cpu")

        B0 = len(atoms_list)
        self.B0 = B0
        # orig-order ASE atoms (own copies; we mutate their positions in place).
        self._atoms_orig = [at.copy() for at in atoms_list]
        for a_src, a_dst in zip(atoms_list, self._atoms_orig):
            a_dst.info = dict(a_src.info)              # ASE copy() may drop info
            a_dst.set_constraint(a_src.constraints)    # ASE copy() drops constraints
            for attr in ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th"):
                if hasattr(a_src, attr):
                    setattr(a_dst, attr, getattr(a_src, attr))
        # per-structure thresholds (orig order)
        self._th = [
            (float(getattr(at, "f_max_th", 2e-3)),
             float(getattr(at, "f_rms_th", 1e-3)),
             float(getattr(at, "dp_max_th", 1e-3)),
             float(getattr(at, "dp_rms_th", 5e-4)))
            for at in self._atoms_orig
        ]
        # results (orig order)
        self.final_pos = [None] * B0
        self.final_E = [None] * B0
        self.nsteps = [0] * B0

    # ---- per-structure ASE constraint projections -------------------------
    def _project_forces_pad(self, F_pad, atoms_cur, n_cur, nmax):
        """F_pad (Bc, nmax) torch -> (Bc, nmax) torch with the SAME FixInternals
        force projection ASE applies in atoms.get_forces (per structure)."""
        np = self.np
        Bc = len(atoms_cur)
        F_cpu = F_pad.detach().to("cpu").numpy()
        out = np.zeros((Bc, nmax), dtype=np.float64)
        for l in range(Bc):
            n = n_cur[l]
            f = F_cpu[l, :3 * n].reshape(n, 3).copy()
            for c in atoms_cur[l].constraints:
                c.adjust_forces(atoms_cur[l], f)
            out[l, :3 * n] = f.reshape(-1)
        return self.torch.as_tensor(out, dtype=self.DTYPE, device=self.device)

    # ---- batched L-BFGS two-loop (BatchLBFGS-identical math + SD fallback) --
    def _two_loop(self, g, S, Y, rho, valid):
        torch = self.torch
        q = g.clone()
        nh = len(S)
        alpha = []
        for t in range(nh - 1, -1, -1):
            a = rho[t] * (S[t] * q).sum(-1)
            a = torch.where(valid[:, t], a, torch.zeros_like(a))
            alpha.append(a)
            q = q - a.unsqueeze(-1) * Y[t]
        alpha = list(reversed(alpha))
        if nh > 0:
            ys = (Y[-1] * S[-1]).sum(-1)
            yy = (Y[-1] * Y[-1]).sum(-1)
            gamma = ys / (yy + 1e-20)
            gamma = torch.where(valid[:, -1], gamma,
                                torch.ones_like(gamma) / self.curvature)
        else:
            gamma = torch.full((g.shape[0],), 1.0 / self.curvature,
                               dtype=self.DTYPE, device=self.device)
        z = gamma.unsqueeze(-1) * q
        for t in range(nh):
            b = rho[t] * (Y[t] * z).sum(-1)
            b = torch.where(valid[:, t], b, torch.zeros_like(b))
            z = z + S[t] * (alpha[t] - b).unsqueeze(-1)
        search = -z
        # per-structure steepest-descent fallback (serial LBFGS._two_loop parity):
        # non-finite OR non-descent (search . grad >= 0) -> -(1/curv) grad.
        bad = (~torch.isfinite(search).all(-1)) | ((search * g).sum(-1) >= 0.0)
        sd = -(1.0 / self.curvature) * g
        return torch.where(bad.unsqueeze(-1), sd, search)

    def _clip(self, step):
        md = step.abs().amax(dim=-1)
        scale = self.torch.clamp(self.maxstep / (md + 1e-20), max=1.0)
        return step * scale.unsqueeze(-1)

    def run(self):
        torch, np = self.torch, self.np
        atoms_cur = list(self._atoms_orig)            # local order (shrinks)
        loc2orig = list(range(self.B0))               # local -> orig
        if not atoms_cur:
            return self._atoms_orig, self.final_E, self.nsteps

        calc = self.calc
        calc.prepare(atoms_cur)                        # fixes nmax
        nmax = int(calc.nmax_dof)
        n_cur = [len(at) for at in atoms_cur]

        def _thr_tensors():
            t = torch.tensor([self._th[loc2orig[l]] for l in range(len(atoms_cur))],
                             dtype=self.DTYPE, device=self.device)
            return t[:, 0], t[:, 1], t[:, 2], t[:, 3]

        f_max_th, f_rms_th, dp_max_th, dp_rms_th = _thr_tensors()

        # history (local order), BatchLBFGS layout
        S, Y, rho = [], [], []
        valid = torch.zeros((len(atoms_cur), self.memory), dtype=torch.bool,
                            device=self.device)

        # initial projected gradient at the seed geometry
        E_cur, F_cur = calc.get_ef_gpu()
        F_proj = self._project_forces_pad(F_cur, atoms_cur, n_cur, nmax)
        g = -F_proj

        it = 0
        while it < self.maxiter and atoms_cur:
            it += 1
            search = self._two_loop(g, S, Y, rho, valid)
            step = self._clip(search)                  # (Bc, nmax) clipped LBFGS step

            # apply per structure with ASE constraint re-imposition
            step_cpu = step.detach().to("cpu").numpy()
            s_act = np.zeros((len(atoms_cur), nmax), dtype=np.float64)
            for l in range(len(atoms_cur)):
                n = n_cur[l]
                r_old = atoms_cur[l].get_positions()
                newpos = r_old + step_cpu[l, :3 * n].reshape(n, 3)
                atoms_cur[l].set_positions(newpos)     # adjust_positions -> manifold
                r_new = atoms_cur[l].get_positions()
                s_act[l, :3 * n] = (r_new - r_old).reshape(-1)
            s_vec = torch.as_tensor(s_act, dtype=self.DTYPE, device=self.device)

            # sync calc coords to the projected geometry + single batched forward
            coord_cat = np.concatenate([at.get_positions() for at in atoms_cur], axis=0)
            calc.set_coords_(torch.as_tensor(coord_cat, dtype=calc.dtype,
                                             device=self.device))
            E_new, F_new = calc.get_ef_gpu()
            F_new_proj = self._project_forces_pad(F_new, atoms_cur, n_cur, nmax)
            g_new = -F_new_proj
            y_vec = g_new - g

            # L-BFGS history update (BatchLBFGS-identical curvature gate)
            ys = (y_vec * s_vec).sum(-1)
            rho_new = 1.0 / (ys + 1e-20)
            v_new = torch.isfinite(rho_new) & (ys > 1e-12)
            S.append(s_vec.clone()); Y.append(y_vec.clone()); rho.append(rho_new)
            if len(S) > self.memory:
                S.pop(0); Y.pop(0); rho.pop(0)
                valid = torch.cat([valid[:, 1:], v_new.unsqueeze(-1)], dim=-1)
            else:
                valid[:, len(S) - 1] = v_new

            # convergence metrics over REAL dof (serial compute_metrics semantics:
            # forces = projected NEW forces, displacement = clipped LBFGS step).
            Bc = len(atoms_cur)
            max_f = torch.zeros(Bc, dtype=self.DTYPE, device=self.device)
            rms_f = torch.zeros(Bc, dtype=self.DTYPE, device=self.device)
            max_dp = torch.zeros(Bc, dtype=self.DTYPE, device=self.device)
            rms_dp = torch.zeros(Bc, dtype=self.DTYPE, device=self.device)
            for l in range(Bc):
                d = 3 * n_cur[l]
                fr = F_new_proj[l, :d]
                st = step[l, :d]
                max_f[l] = fr.abs().amax()
                rms_f[l] = torch.sqrt((fr ** 2).sum() / d)
                max_dp[l] = st.abs().amax()
                rms_dp[l] = torch.sqrt((st ** 2).sum() / d)

            done = ((max_f <= f_max_th) & (rms_f <= f_rms_th)
                    & (max_dp <= dp_max_th) & (rms_dp <= dp_rms_th))
            done_l = done.detach().to("cpu").tolist()
            E_new_l = [float(x) for x in E_new.detach().to("cpu").tolist()]

            # record converged structures (final geom + energy at that geom)
            survive = []
            for l in range(Bc):
                self.nsteps[loc2orig[l]] = it
                if done_l[l]:
                    o = loc2orig[l]
                    self.final_pos[o] = atoms_cur[l].get_positions().copy()
                    self.final_E[o] = E_new_l[l]
                    self._atoms_orig[o].set_positions(self.final_pos[o])
                else:
                    survive.append(l)

            if len(survive) < Bc:
                if not survive:
                    atoms_cur = []
                    break
                surv_t = torch.tensor(survive, dtype=torch.long, device=self.device)
                atoms_cur = [atoms_cur[l] for l in survive]
                loc2orig = [loc2orig[l] for l in survive]
                n_cur = [n_cur[l] for l in survive]
                S = [s[surv_t] for s in S]
                Y = [y[surv_t] for y in Y]
                rho = [r[surv_t] for r in rho]
                valid = valid[surv_t]
                f_max_th, f_rms_th, dp_max_th, dp_rms_th = _thr_tensors()
                calc.prepare(atoms_cur, fixed_nmax=nmax)
                # re-sync calc coords to survivors' (already-projected) geometry
                coord_cat = np.concatenate(
                    [at.get_positions() for at in atoms_cur], axis=0)
                calc.set_coords_(torch.as_tensor(coord_cat, dtype=calc.dtype,
                                                 device=self.device))
                F_proj = F_new_proj[surv_t]
                g = -F_proj
                E_cur = E_new[surv_t]
            else:
                g = g_new
                E_cur = E_new

        # any structure still un-converged after maxiter -> record current state
        E_cur_l = ([float(x) for x in E_cur.detach().to("cpu").tolist()]
                   if atoms_cur else [])
        for l in range(len(atoms_cur)):
            o = loc2orig[l]
            self.final_pos[o] = atoms_cur[l].get_positions().copy()
            self.final_E[o] = E_cur_l[l]
            self._atoms_orig[o].set_positions(self.final_pos[o])

        return self._atoms_orig, self.final_E, self.nsteps

