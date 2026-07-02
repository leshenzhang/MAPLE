# -*- coding: utf-8 -*-
"""
AutoNEB: Automated Nudged Elastic Band for multi-step reaction pathway exploration.

Features:
- Adaptive image insertion based on distance threshold
- Local minima detection and path splitting
- Endpoint optimization (detecting lower-energy endpoints)
- Binary tree management for multi-path optimization
"""
from __future__ import annotations
import os
import json
import math
import copy
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
from ase import Atoms

from .neb import (
    NEB, NEBParams, LBFGSDriver,
    neb_forces, rms_force, compute_dynamic_k,
    write_xyz, write_all_images_xyz,
    to_numpy_f64, vec1d, kabsch_align
)
from .logger import log_info
from ...jobABC import JobABC
from maple.function.utility import Molecules


# =============================================================================
# AutoNEB Parameters
# =============================================================================
@dataclass
class AutoNEBParams:
    """Parameters for AutoNEB algorithm."""

    # ===== Basic NEB settings (inherited concept) =====
    n_images: int = 20                     # Initial number of images (excluding endpoints)
    k_min: float = 0.03                    # Minimum spring constant
    k_max: float = 0.3                     # Maximum spring constant
    use_dynamic_k: bool = True             # Enable ORCA-style dynamic spring constants
    k_decay: float = 0.5                   # Dynamic k decay factor
    max_iter: int = 256                    # Maximum iterations per path
    lbfgs_m: int = 10                      # L-BFGS memory size
    step0: float = 2e-2                    # Initial step length
    ifidpp: int = 1                        # 1: use IDPP, 0: linear interpolation

    # ===== Adaptive image insertion =====
    ang_max: float = 0.3                   # Maximum distance between adjacent images (Angstrom)
    ang_iter: int = 20                     # Check insertion every N iterations

    # ===== Local minima detection and path splitting =====
    path_iter: int = 50                    # Check for local minima every N iterations
    min_e_drop: float = 0.001              # Energy drop threshold for minima detection (Eh)

    # ===== Endpoint optimization =====
    ep_iter: int = 30                      # Check endpoint updates every N iterations
    ep_e_drop: float = 0.0005              # Energy drop threshold for endpoint update (Eh)

    # ===== Convergence settings (looser than standard NEB) =====
    autoneb_f_max_th: float = 9.5e-3       # Max force convergence threshold
    autoneb_f_rms_th: float = 5e-3         # RMS force convergence threshold

    # ===== Recursion control =====
    max_depth: int = 5                     # Maximum recursion depth
    max_paths: int = 10                    # Maximum number of paths

    # ===== Final global refinement =====
    do_final_refine: bool = True           # Run final global NEB refinement
    final_refine_factor: float = 2.0       # Convergence threshold multiplier (looser)
    final_refine_max_iter: int = 200       # Max iterations for final refinement
    final_refine_ep_iter: int = 20         # Endpoint check interval in final refinement

    # ===== Output control =====
    verbose: int = 1                       # Verbosity level


# =============================================================================
# PathNode: Data structure for path tree management
# =============================================================================
@dataclass
class PathNode:
    """Node in the path tree for AutoNEB."""
    path_id: str                           # Unique path identifier
    images: List[Atoms] = field(default_factory=list)  # Current images
    energies: List[float] = field(default_factory=list)  # Energies
    parent: Optional[str] = None           # Parent path ID
    children: List[str] = field(default_factory=list)  # Child path IDs
    status: str = 'pending'                # 'pending', 'running', 'converged', 'split', 'needs_reopt'
    depth: int = 0                         # Recursion depth
    start_is_shared: bool = False          # Start point shared with sibling
    end_is_shared: bool = False            # End point shared with sibling
    iteration: int = 0                     # Current iteration count
    hei_idx: int = -1                      # Highest energy image index

    # L-BFGS state (for continuing optimization)
    lbfgs_S: List[np.ndarray] = field(default_factory=list)
    lbfgs_Y: List[np.ndarray] = field(default_factory=list)
    lbfgs_rhos: List[float] = field(default_factory=list)


# =============================================================================
# AutoNEB Class
# =============================================================================
class AutoNEB(JobABC):
    """
    Automated NEB for multi-step reaction pathway exploration.

    This class manages multiple NEB paths in a tree structure, automatically
    detecting intermediate minima and splitting paths as needed.
    """

    def __init__(self,
                 output: str,
                 atoms_or_molecules,
                 paras: Optional[dict] = None):
        super().__init__(output)

        # Handle Molecules input
        if isinstance(atoms_or_molecules, Molecules):
            self.input_images = atoms_or_molecules.multiatoms
        elif isinstance(atoms_or_molecules, list):
            self.input_images = atoms_or_molecules
        else:
            raise ValueError("Please provide Molecules object or list of Atoms")

        if len(self.input_images) < 2:
            raise ValueError("Need at least 2 images (reactant + product)")

        # Initialize parameters
        self.params = self._init_params(AutoNEBParams, paras, ("autoneb", "AutoNEB", "ts"))

        # Path tree management
        self.path_tree: Dict[str, PathNode] = {}
        self.root_path_id: str = "root"

        # Global tracking
        self.global_iteration = 0
        self.all_intermediates: List[Atoms] = []
        self.all_ts: List[Atoms] = []

        # Log parameters
        self._log_params()

    def _log_params(self):
        """Log AutoNEB parameters."""
        p = self.params
        info = [
            "\n" + "=" * 70 + "\n",
            "AutoNEB Parameters\n",
            "=" * 70 + "\n",
            f"n_images:            {p.n_images}\n",
            f"ang_max:             {p.ang_max} Angstrom\n",
            f"ang_iter:            {p.ang_iter}\n",
            f"path_iter:           {p.path_iter}\n",
            f"min_e_drop:          {p.min_e_drop} Eh\n",
            f"ep_iter:             {p.ep_iter}\n",
            f"ep_e_drop:           {p.ep_e_drop} Eh\n",
            f"autoneb_f_max_th:    {p.autoneb_f_max_th}\n",
            f"autoneb_f_rms_th:    {p.autoneb_f_rms_th}\n",
            f"max_depth:           {p.max_depth}\n",
            f"max_paths:           {p.max_paths}\n",
            f"do_final_refine:     {p.do_final_refine}\n",
            f"final_refine_factor: {p.final_refine_factor}\n",
            f"final_refine_max_iter: {p.final_refine_max_iter}\n",
            "=" * 70 + "\n\n",
        ]
        log_info(info, self.output)

    def _copy_atoms_with_calc(self, atoms: Atoms) -> Atoms:
        """Copy Atoms object while preserving the calculator."""
        new_atoms = atoms.copy()
        new_atoms.calc = atoms.calc
        return new_atoms

    # =========================================================================
    # Initialization
    # =========================================================================

    def _init_root_path(self):
        """Initialize the root path from input images."""
        images = [self._copy_atoms_with_calc(img) for img in self.input_images]

        # Align all images to first one
        ref = to_numpy_f64(images[0].get_positions())
        for i in range(1, len(images)):
            Q = to_numpy_f64(images[i].get_positions())
            Q_aligned, _, _, _ = kabsch_align(ref, Q)
            images[i].set_positions(Q_aligned)

        # Interpolate if needed
        n_required = self.params.n_images + 2
        if len(images) < n_required:
            images = self._interpolate_images(images, n_required)

        # Get initial energies
        energies = self._get_energies(images)

        # Create root node
        root = PathNode(
            path_id=self.root_path_id,
            images=images,
            energies=energies,
            parent=None,
            children=[],
            status='pending',
            depth=0,
            start_is_shared=False,
            end_is_shared=False,
        )

        self.path_tree[self.root_path_id] = root

        log_info([
            f"\nInitialized root path with {len(images)} images.\n",
            f"Energy range: {min(energies):.6f} to {max(energies):.6f} Eh\n",
        ], self.output)

    def _interpolate_images(self, images: List[Atoms], n_target: int) -> List[Atoms]:
        """Interpolate images to reach target count."""
        n_current = len(images)
        if n_current >= n_target:
            return images

        n_to_insert = n_target - n_current

        # Compute distances between consecutive images
        distances = self._compute_distances(images)

        # Determine insertion plan (prioritize largest gaps)
        plan = self._determine_insertion_plan(distances, n_to_insert)

        # Execute insertion
        new_images = self._insert_images_by_plan(images, plan)

        # Optional IDPP smoothing
        if self.params.ifidpp == 1 and len(new_images) > 2:
            new_images = self._run_idpp_smoothing(new_images)

        return new_images

    def _compute_distances(self, images: List[Atoms]) -> List[float]:
        """Compute distances between consecutive images."""
        distances = []
        for i in range(len(images) - 1):
            pos1 = to_numpy_f64(images[i].get_positions())
            pos2 = to_numpy_f64(images[i+1].get_positions())
            dist = np.linalg.norm(pos2 - pos1)
            distances.append(dist)
        return distances

    def _determine_insertion_plan(self, distances: List[float], n_to_insert: int) -> List[Tuple[int, int]]:
        """Determine where to insert images based on distances."""
        insertion_counts = [0] * len(distances)

        for _ in range(n_to_insert):
            # Find segment with largest distance/insertions ratio
            max_ratio = -1
            max_idx = 0
            for i, d in enumerate(distances):
                ratio = d / (insertion_counts[i] + 1)
                if ratio > max_ratio:
                    max_ratio = ratio
                    max_idx = i
            insertion_counts[max_idx] += 1

        plan = [(i, count) for i, count in enumerate(insertion_counts) if count > 0]
        return plan

    def _insert_images_by_plan(self, images: List[Atoms], plan: List[Tuple[int, int]]) -> List[Atoms]:
        """Insert interpolated images according to plan."""
        plan_sorted = sorted(plan, key=lambda x: x[0], reverse=True)
        new_images = list(images)

        for seg_idx, count in plan_sorted:
            pos1 = to_numpy_f64(new_images[seg_idx].get_positions())
            pos2 = to_numpy_f64(new_images[seg_idx + 1].get_positions())

            inserted = []
            for k in range(1, count + 1):
                lam = k / (count + 1)
                new_pos = (1.0 - lam) * pos1 + lam * pos2
                new_atom = new_images[seg_idx].copy()
                new_atom.set_positions(new_pos)
                new_atom.calc = new_images[seg_idx].calc
                inserted.append(new_atom)

            for idx, img in enumerate(inserted):
                new_images.insert(seg_idx + 1 + idx, img)

        return new_images

    def _run_idpp_smoothing(self, images: List[Atoms]) -> List[Atoms]:
        """Run IDPP smoothing on internal images."""
        # Simplified IDPP - use NEB's implementation
        from .neb import _pair_indices, _idpp_targets

        n_img = len(images)
        n_inner = n_img - 2
        n_atoms = len(images[0])

        if n_inner <= 0:
            return images

        coords = [to_numpy_f64(img.get_positions()) for img in images]
        R0, R1 = coords[0], coords[-1]

        pairs = _pair_indices(n_atoms)
        targets = _idpp_targets(R0, R1, n_inner)

        # Flatten internal images
        x = np.concatenate([coords[i].reshape(-1) for i in range(1, n_img - 1)])

        # Simple gradient descent for IDPP
        for _ in range(100):
            grad = np.zeros_like(x)
            offset = 0
            for k in range(n_inner):
                Xi = x[offset:offset + n_atoms * 3].reshape(n_atoms, 3)
                Rij = Xi[pairs[:, 0]] - Xi[pairs[:, 1]]
                dij = np.linalg.norm(Rij, axis=1) + 1e-12
                inv = 1.0 / dij
                diff = inv - targets[k]
                dE_dd = -2.0 * diff / (dij**2)
                g_pair = dE_dd[:, None] * (Rij / dij[:, None])

                g_atoms = np.zeros_like(Xi)
                for p in range(len(pairs)):
                    i, j = pairs[p]
                    g_atoms[i] += g_pair[p]
                    g_atoms[j] -= g_pair[p]

                grad[offset:offset + n_atoms * 3] = g_atoms.reshape(-1)
                offset += n_atoms * 3

            x -= 0.01 * grad
            if np.sqrt(np.mean(grad * grad)) < 1e-6:
                break

        # Write back
        offset = 0
        for i in range(1, n_img - 1):
            Xi = x[offset:offset + n_atoms * 3].reshape(n_atoms, 3)
            images[i].set_positions(Xi)
            offset += n_atoms * 3

        return images

    def _get_energies(self, images: List[Atoms]) -> List[float]:
        """Get energies for all images."""
        return [float(at.get_potential_energy(force_consistent=True)) for at in images]

    # =========================================================================
    # Single Path Optimization
    # =========================================================================

    def _run_single_path_iteration(self, path_id: str) -> bool:
        """
        Run one iteration of NEB optimization on a path.
        Returns True if converged.
        """
        node = self.path_tree[path_id]
        images = node.images
        p = self.params

        # Initialize L-BFGS driver if needed
        if not hasattr(node, '_driver') or node._driver is None:
            node._driver = LBFGSDriver(m=p.lbfgs_m, curvature=70.0, maxstep=p.step0)
            # Restore state if available
            if node.lbfgs_S:
                node._driver.S = node.lbfgs_S
                node._driver.Y = node.lbfgs_Y
                node._driver.rhos = node.lbfgs_rhos

        driver = node._driver

        # Compute energies and forces
        energies = self._get_energies(images)
        node.energies = energies

        # Compute NEB forces
        if p.use_dynamic_k:
            k_springs = compute_dynamic_k(energies, p.k_min, p.k_max, p.k_decay)
        else:
            k_springs = [p.k_max] * len(images)

        Fp_list, maxfp, hei_idx = neb_forces(
            images, energies,
            k_spring=None,
            k_springs=k_springs,
            use_dynamic_k=False
        )

        node.hei_idx = hei_idx
        rmsfp = rms_force(Fp_list)

        # Check convergence
        if maxfp < p.autoneb_f_max_th and rmsfp < p.autoneb_f_rms_th:
            return True

        # Pack internal images to gradient vector
        grads = []
        for i in range(1, len(images) - 1):
            grads.append((-Fp_list[i]).reshape(-1))
        g = np.concatenate(grads) if grads else np.zeros(0)

        # L-BFGS step
        step = driver.two_loop(g)
        step = driver.step_limit(step)

        # Pack current positions
        x = np.concatenate([to_numpy_f64(images[i].get_positions()).reshape(-1)
                           for i in range(1, len(images) - 1)])

        # Update positions
        x_new = x + step

        # Unpack to images
        offset = 0
        for i in range(1, len(images) - 1):
            n = len(images[i]) * 3
            Xi = x_new[offset:offset + n].reshape(-1, 3)
            images[i].set_positions(Xi)
            offset += n

        # Compute new gradient for L-BFGS update
        new_energies = self._get_energies(images)
        node.energies = new_energies

        if p.use_dynamic_k:
            k_springs = compute_dynamic_k(new_energies, p.k_min, p.k_max, p.k_decay)

        new_Fp_list, _, _ = neb_forces(images, new_energies, k_springs=k_springs)

        new_grads = []
        for i in range(1, len(images) - 1):
            new_grads.append((-new_Fp_list[i]).reshape(-1))
        g_new = np.concatenate(new_grads) if new_grads else np.zeros(0)

        # Update L-BFGS history
        if len(g) > 0:
            driver.update(x_new - x, g_new - g)

        # Save L-BFGS state
        node.lbfgs_S = list(driver.S)
        node.lbfgs_Y = list(driver.Y)
        node.lbfgs_rhos = list(driver.rhos)

        node.iteration += 1

        return False

    # =========================================================================
    # Adaptive Image Insertion
    # =========================================================================

    def _check_adaptive_insertion(self, path_id: str) -> bool:
        """
        Check and perform adaptive image insertion if needed.
        Returns True if images were inserted.
        """
        node = self.path_tree[path_id]
        images = node.images
        p = self.params

        distances = self._compute_distances(images)

        # Find segments that need insertion
        insertions_needed = []
        for i, dist in enumerate(distances):
            if dist > p.ang_max:
                n_insert = int(math.ceil(dist / p.ang_max)) - 1
                if n_insert > 0:
                    insertions_needed.append((i, n_insert))

        if not insertions_needed:
            return False

        # Log insertion
        total_insert = sum(n for _, n in insertions_needed)
        log_info([
            f"\n[Path {path_id}] Adaptive insertion: adding {total_insert} image(s)\n"
        ], self.output)

        # Execute insertion
        new_images = self._insert_images_by_plan(images, insertions_needed)

        # Optional IDPP
        if p.ifidpp == 1:
            new_images = self._run_idpp_smoothing(new_images)

        # Update node
        node.images = new_images
        node.energies = self._get_energies(new_images)

        # Reset L-BFGS
        node._driver = None
        node.lbfgs_S = []
        node.lbfgs_Y = []
        node.lbfgs_rhos = []

        return True

    # =========================================================================
    # Local Minima Detection and Path Splitting
    # =========================================================================

    def _detect_local_minima(self, path_id: str) -> List[int]:
        """
        Detect local minima in a path (excluding endpoints).
        Returns list of minima indices, sorted by distance from start.
        """
        node = self.path_tree[path_id]
        energies = node.energies
        p = self.params

        minima = []
        # Skip first 2 and last 2 images to avoid false positives near endpoints
        for i in range(2, len(energies) - 2):
            E_prev = energies[i-1]
            E_curr = energies[i]
            E_next = energies[i+1]

            # Local minimum condition with energy drop threshold
            if (E_prev - E_curr > p.min_e_drop) and (E_next - E_curr > p.min_e_drop):
                minima.append(i)

        # Sort by index (closest to start first)
        return sorted(minima)

    def _split_path(self, path_id: str, split_idx: int):
        """Split a path at the given index."""
        parent = self.path_tree[path_id]

        if parent.depth >= self.params.max_depth:
            log_info([f"\n[Path {path_id}] Max depth reached, skipping split.\n"], self.output)
            return

        if len(self.path_tree) >= self.params.max_paths:
            log_info([f"\n[Path {path_id}] Max paths reached, skipping split.\n"], self.output)
            return

        parent.status = 'split'

        # Create left child (start -> split_idx)
        child1_id = f"{path_id}_L"
        child1_images = [self._copy_atoms_with_calc(img) for img in parent.images[:split_idx+1]]
        child1 = PathNode(
            path_id=child1_id,
            images=child1_images,
            energies=parent.energies[:split_idx+1],
            parent=path_id,
            children=[],
            status='pending',
            depth=parent.depth + 1,
            start_is_shared=parent.start_is_shared,
            end_is_shared=True,
        )

        # Create right child (split_idx -> end)
        child2_id = f"{path_id}_R"
        child2_images = [self._copy_atoms_with_calc(img) for img in parent.images[split_idx:]]
        child2 = PathNode(
            path_id=child2_id,
            images=child2_images,
            energies=parent.energies[split_idx:],
            parent=path_id,
            children=[],
            status='pending',
            depth=parent.depth + 1,
            start_is_shared=True,
            end_is_shared=parent.end_is_shared,
        )

        parent.children = [child1_id, child2_id]
        self.path_tree[child1_id] = child1
        self.path_tree[child2_id] = child2

        # Record intermediate
        intermediate = self._copy_atoms_with_calc(parent.images[split_idx])
        self.all_intermediates.append(intermediate)

        log_info([
            f"\n{'='*70}\n",
            f"Path Split: {path_id} at index {split_idx}\n",
            f"{'='*70}\n",
            f"Intermediate energy: {parent.energies[split_idx]:.6f} Eh\n",
            f"Created paths: {child1_id} ({len(child1_images)} images), "
            f"{child2_id} ({len(child2_images)} images)\n",
        ], self.output)

    # =========================================================================
    # Endpoint Optimization
    # =========================================================================

    def _check_endpoint_updates(self, path_id: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Check if endpoints need to be updated based on local monotonic trend.
        Returns (new_start_idx, new_end_idx), None if no update needed.

        Only looks at 3 points near each endpoint (A-X-Y):
        - A = current endpoint
        - X = 1st neighbor toward HEI
        - Y = 2nd neighbor toward HEI

        Update rules:
        - If A → X → Y continuously decreases, update to Y
        - If A → X decreases but Y > A, only update to X
        - If A → X decreases but Y is between A and X, update to X
        - Never cross an energy rise
        """
        node = self.path_tree[path_id]
        energies = node.energies
        p = self.params
        n = len(energies)

        new_start = None
        new_end = None

        if n < 4:  # Need at least 4 points to judge
            return None, None

        # ========== Start side ==========
        # A=0, X=1, Y=2 (looking right toward HEI)
        E_A = energies[0]
        E_X = energies[1]
        E_Y = energies[2]

        if E_X < E_A - p.ep_e_drop:
            # X is lower than A, check Y
            if E_Y < E_X - p.ep_e_drop:
                # Y also decreases, update to Y
                new_start = 2
            elif E_Y > E_A:
                # Y rises above A, only update to X
                new_start = 1
            else:
                # Y is between A and X (or equal), update to X
                new_start = 1

        # ========== End side ==========
        # A=n-1, X=n-2, Y=n-3 (looking left toward HEI)
        E_A = energies[-1]
        E_X = energies[-2]
        E_Y = energies[-3]

        if E_X < E_A - p.ep_e_drop:
            # X is lower than A, check Y
            if E_Y < E_X - p.ep_e_drop:
                # Y also decreases, update to Y
                new_end = n - 3
            elif E_Y > E_A:
                # Y rises above A, only update to X
                new_end = n - 2
            else:
                # Y is between A and X (or equal), update to X
                new_end = n - 2

        return new_start, new_end

    def _update_endpoint(self, path_id: str, which_end: str, new_idx: int):
        """Update an endpoint and handle shared endpoint synchronization."""
        node = self.path_tree[path_id]

        if which_end == 'start':
            # Trim start
            new_images = node.images[new_idx:]
            new_energies = node.energies[new_idx:]
            node.images = new_images
            node.energies = new_energies

            log_info([
                f"\n[Path {path_id}] Updated start point to index {new_idx}, "
                f"energy: {new_energies[0]:.6f} Eh\n"
            ], self.output)

            # If start was shared, update sibling's end
            if node.start_is_shared and node.parent:
                self._sync_shared_endpoint(path_id, 'start', node.images[0])

        elif which_end == 'end':
            # Trim end
            new_images = node.images[:new_idx+1]
            new_energies = node.energies[:new_idx+1]
            node.images = new_images
            node.energies = new_energies

            log_info([
                f"\n[Path {path_id}] Updated end point to index {new_idx}, "
                f"energy: {new_energies[-1]:.6f} Eh\n"
            ], self.output)

            # If end was shared, update sibling's start
            if node.end_is_shared and node.parent:
                self._sync_shared_endpoint(path_id, 'end', node.images[-1])

        # Reset L-BFGS
        node._driver = None
        node.lbfgs_S = []
        node.lbfgs_Y = []
        node.lbfgs_rhos = []

    def _sync_shared_endpoint(self, path_id: str, which_end: str, new_point: Atoms):
        """Synchronize a shared endpoint with sibling path."""
        node = self.path_tree[path_id]
        if not node.parent:
            return

        parent = self.path_tree[node.parent]
        siblings = [c for c in parent.children if c != path_id]

        for sib_id in siblings:
            sib = self.path_tree[sib_id]

            if which_end == 'start' and sib.end_is_shared:
                # This path's start is sibling's end
                sib.images[-1] = self._copy_atoms_with_calc(new_point)
                sib.energies[-1] = float(new_point.get_potential_energy(force_consistent=True))
                sib.status = 'needs_reopt'
                log_info([f"[Path {sib_id}] Marked for re-optimization (shared endpoint updated)\n"], self.output)

            elif which_end == 'end' and sib.start_is_shared:
                # This path's end is sibling's start
                sib.images[0] = self._copy_atoms_with_calc(new_point)
                sib.energies[0] = float(new_point.get_potential_energy(force_consistent=True))
                sib.status = 'needs_reopt'
                log_info([f"[Path {sib_id}] Marked for re-optimization (shared endpoint updated)\n"], self.output)

    # =========================================================================
    # Tree Traversal and Path Selection
    # =========================================================================

    def _select_next_path(self) -> Optional[str]:
        """Select next path to optimize (DFS: leftmost pending/needs_reopt first)."""
        def dfs(path_id: str) -> Optional[str]:
            node = self.path_tree[path_id]

            # If this node needs work
            if node.status in ('pending', 'running', 'needs_reopt'):
                return path_id

            # If split, check children
            if node.status == 'split':
                for child_id in node.children:
                    result = dfs(child_id)
                    if result:
                        return result

            return None

        return dfs(self.root_path_id)

    def _all_leaves_converged(self) -> bool:
        """Check if all leaf paths are converged."""
        def check_leaves(path_id: str) -> bool:
            node = self.path_tree[path_id]

            if node.status == 'split':
                return all(check_leaves(c) for c in node.children)
            else:
                return node.status == 'converged'

        return check_leaves(self.root_path_id)

    # =========================================================================
    # Output Generation
    # =========================================================================

    def _merge_global_mep(self) -> Tuple[List[Atoms], List[float]]:
        """Merge all converged leaf paths into global MEP."""
        def collect_leaves(path_id: str) -> List[str]:
            node = self.path_tree[path_id]
            if node.status == 'split':
                leaves = []
                for c in node.children:
                    leaves.extend(collect_leaves(c))
                return leaves
            else:
                return [path_id]

        leaf_ids = collect_leaves(self.root_path_id)

        # Merge paths in order
        global_images = []
        global_energies = []

        for i, leaf_id in enumerate(leaf_ids):
            node = self.path_tree[leaf_id]
            if i == 0:
                global_images.extend(node.images)
                global_energies.extend(node.energies)
            else:
                # Skip first image (shared with previous path's end)
                global_images.extend(node.images[1:])
                global_energies.extend(node.energies[1:])

        return global_images, global_energies

    def _run_final_refinement(self, images: List[Atoms], energies: List[float]) -> Tuple[List[Atoms], List[float]]:
        """
        Run final global NEB refinement on the merged MEP.

        This phase:
        - Uses looser convergence thresholds (2x the normal thresholds)
        - Performs endpoint optimization (no path splitting)
        - Smooths out the entire path
        """
        p = self.params
        n_images = len(images)

        if n_images < 4:
            return images, energies

        # Copy images to avoid modifying originals
        ref_images = [self._copy_atoms_with_calc(img) for img in images]

        # Looser convergence thresholds
        f_max_th = p.autoneb_f_max_th * p.final_refine_factor
        f_rms_th = p.autoneb_f_rms_th * p.final_refine_factor

        log_info([
            f"Final refinement: {n_images} images\n",
            f"Convergence: f_max < {f_max_th:.6f}, f_rms < {f_rms_th:.6f}\n",
            f"Max iterations: {p.final_refine_max_iter}\n",
        ], self.output)

        # Initialize L-BFGS driver
        driver = LBFGSDriver(m=p.lbfgs_m, curvature=70.0, maxstep=p.step0)

        converged = False
        for iteration in range(1, p.final_refine_max_iter + 1):
            # Get current energies
            energies = self._get_energies(ref_images)
            n_images = len(ref_images)

            # Compute spring constants
            if p.use_dynamic_k:
                k_springs = compute_dynamic_k(energies, p.k_min, p.k_max, p.k_decay)
            else:
                k_springs = [p.k_max] * n_images

            # Compute NEB forces
            Fp_list, maxfp, hei_idx = neb_forces(
                ref_images, energies,
                k_springs=k_springs,
                use_dynamic_k=False
            )
            rmsfp = rms_force(Fp_list)

            # Log progress
            if iteration % 10 == 0 or iteration == 1:
                dE = energies[hei_idx] - energies[0]
                log_info([
                    f"  Refine Iter {iteration:4d}  HEI={hei_idx}  dE={dE:.6f}  "
                    f"max|Fp|={maxfp:.6f}  RMS(Fp)={rmsfp:.6f}\n"
                ], self.output)

            # Check convergence
            if maxfp < f_max_th and rmsfp < f_rms_th:
                log_info([
                    f"\nFinal refinement converged after {iteration} iterations.\n"
                ], self.output)
                converged = True
                break

            # Pack internal images to gradient vector
            grads = []
            for i in range(1, n_images - 1):
                grads.append((-Fp_list[i]).reshape(-1))
            g = np.concatenate(grads) if grads else np.zeros(0)

            # L-BFGS step
            step = driver.two_loop(g)
            step = driver.step_limit(step)

            # Pack current positions
            x = np.concatenate([to_numpy_f64(ref_images[i].get_positions()).reshape(-1)
                               for i in range(1, n_images - 1)])

            # Update positions
            x_new = x + step

            # Unpack to images
            offset = 0
            for i in range(1, n_images - 1):
                n_atoms_3 = len(ref_images[i]) * 3
                Xi = x_new[offset:offset + n_atoms_3].reshape(-1, 3)
                ref_images[i].set_positions(Xi)
                offset += n_atoms_3

            # Compute new gradient for L-BFGS update
            new_energies = self._get_energies(ref_images)

            if p.use_dynamic_k:
                k_springs = compute_dynamic_k(new_energies, p.k_min, p.k_max, p.k_decay)

            new_Fp_list, _, _ = neb_forces(ref_images, new_energies, k_springs=k_springs)

            new_grads = []
            for i in range(1, n_images - 1):
                new_grads.append((-new_Fp_list[i]).reshape(-1))
            g_new = np.concatenate(new_grads) if new_grads else np.zeros(0)

            # Update L-BFGS history
            if len(g) > 0:
                driver.update(x_new - x, g_new - g)

            # Check for endpoint updates periodically
            if iteration % p.final_refine_ep_iter == 0:
                new_start, new_end = self._check_endpoint_updates_global(ref_images, new_energies)
                if new_start is not None:
                    log_info([
                        f"\n[Final Refine] Updated start to index {new_start}, "
                        f"energy: {new_energies[new_start]:.6f} Eh\n"
                    ], self.output)
                    ref_images = ref_images[new_start:]
                    n_images = len(ref_images)
                    driver = LBFGSDriver(m=p.lbfgs_m, curvature=70.0, maxstep=p.step0)

                if new_end is not None:
                    log_info([
                        f"\n[Final Refine] Updated end to index {new_end}, "
                        f"energy: {new_energies[new_end]:.6f} Eh\n"
                    ], self.output)
                    ref_images = ref_images[:new_end + 1]
                    n_images = len(ref_images)
                    driver = LBFGSDriver(m=p.lbfgs_m, curvature=70.0, maxstep=p.step0)

        if not converged:
            log_info([
                f"\nFinal refinement reached max iterations ({p.final_refine_max_iter}).\n"
            ], self.output)

        # Get final energies
        final_energies = self._get_energies(ref_images)

        return ref_images, final_energies

    def _check_endpoint_updates_global(self, images: List[Atoms], energies: List[float]) -> Tuple[Optional[int], Optional[int]]:
        """
        Check endpoint updates for global MEP (used in final refinement).
        Same local 3-point logic as _check_endpoint_updates.
        """
        p = self.params
        n = len(energies)

        new_start = None
        new_end = None

        if n < 4:
            return None, None

        # ========== Start side ==========
        E_A = energies[0]
        E_X = energies[1]
        E_Y = energies[2]

        if E_X < E_A - p.ep_e_drop:
            if E_Y < E_X - p.ep_e_drop:
                new_start = 2
            elif E_Y > E_A:
                new_start = 1
            else:
                new_start = 1

        # ========== End side ==========
        E_A = energies[-1]
        E_X = energies[-2]
        E_Y = energies[-3]

        if E_X < E_A - p.ep_e_drop:
            if E_Y < E_X - p.ep_e_drop:
                new_end = n - 3
            elif E_Y > E_A:
                new_end = n - 2
            else:
                new_end = n - 2

        return new_start, new_end

    def _collect_all_ts(self):
        """Collect all transition states from converged paths."""
        self.all_ts = []

        def collect_from_path(path_id: str):
            node = self.path_tree[path_id]
            if node.status == 'split':
                for c in node.children:
                    collect_from_path(c)
            elif node.status == 'converged' and node.hei_idx > 0:
                self.all_ts.append(self._copy_atoms_with_calc(node.images[node.hei_idx]))

        collect_from_path(self.root_path_id)

    def _write_outputs(self):
        """Write all output files."""
        base, _ = os.path.splitext(self.output)

        # Global MEP (use final refined images if available)
        if hasattr(self, 'final_images') and self.final_images:
            global_images = self.final_images
            global_energies = self.final_energies
        else:
            global_images, global_energies = self._merge_global_mep()

        mep_file = base + "_autoneb_global_mep.xyz"
        write_xyz(mep_file, global_images, energies=global_energies)
        log_info([f"\nWrote global MEP to: {mep_file}\n"], self.output)

        # Intermediates
        if self.all_intermediates:
            int_file = base + "_autoneb_intermediates.xyz"
            int_energies = [float(at.get_potential_energy(force_consistent=True))
                          for at in self.all_intermediates]
            write_xyz(int_file, self.all_intermediates, energies=int_energies)
            log_info([f"Wrote intermediates to: {int_file}\n"], self.output)

        # Transition states
        self._collect_all_ts()
        if self.all_ts:
            ts_file = base + "_autoneb_ts_list.xyz"
            ts_energies = [float(at.get_potential_energy(force_consistent=True))
                         for at in self.all_ts]
            write_xyz(ts_file, self.all_ts, energies=ts_energies)
            log_info([f"Wrote transition states to: {ts_file}\n"], self.output)

        # Path tree (JSON for debugging)
        tree_file = base + "_autoneb_tree.json"
        tree_data = {}
        for pid, node in self.path_tree.items():
            tree_data[pid] = {
                'status': node.status,
                'depth': node.depth,
                'n_images': len(node.images),
                'parent': node.parent,
                'children': node.children,
                'hei_idx': node.hei_idx,
                'iteration': node.iteration,
            }
        with open(tree_file, 'w') as f:
            json.dump(tree_data, f, indent=2)
        log_info([f"Wrote path tree to: {tree_file}\n"], self.output)

        # Individual path MEPs
        def write_path_meps(path_id: str):
            node = self.path_tree[path_id]
            if node.status == 'split':
                for c in node.children:
                    write_path_meps(c)
            elif node.status == 'converged':
                path_file = base + f"_autoneb_path_{path_id}_mep.xyz"
                write_xyz(path_file, node.images, energies=node.energies)

        write_path_meps(self.root_path_id)

    def _print_path_summary(self, path_id: str):
        """Print summary for a path."""
        node = self.path_tree[path_id]
        energies = node.energies

        hei_idx = node.hei_idx if node.hei_idx > 0 else max(range(1, len(energies)-1), key=lambda i: energies[i])

        kcal_per_Eh = 627.509
        barrier = (energies[hei_idx] - energies[0]) * kcal_per_Eh
        reaction_E = (energies[-1] - energies[0]) * kcal_per_Eh

        log_info([
            f"\n[Path {path_id}] Summary:\n",
            f"  Images: {len(node.images)}, Iterations: {node.iteration}\n",
            f"  HEI index: {hei_idx}\n",
            f"  Forward barrier: {barrier:.2f} kcal/mol\n",
            f"  Reaction energy: {reaction_E:.2f} kcal/mol\n",
        ], self.output)

    # =========================================================================
    # Main Run Method
    # =========================================================================

    def run(self):
        """Main entry point for AutoNEB optimization."""
        log_info([
            "\n" + "=" * 70 + "\n",
            "Starting AutoNEB Optimization\n",
            "=" * 70 + "\n",
        ], self.output)

        # Initialize root path
        self._init_root_path()

        p = self.params
        max_global_iter = p.max_iter * p.max_paths  # Safety limit

        while self.global_iteration < max_global_iter:
            self.global_iteration += 1

            # Check if all done
            if self._all_leaves_converged():
                log_info(["\nAll paths converged!\n"], self.output)
                break

            # Select next path to work on
            path_id = self._select_next_path()
            if path_id is None:
                log_info(["\nNo more paths to optimize.\n"], self.output)
                break

            node = self.path_tree[path_id]

            # Mark as running
            if node.status in ('pending', 'needs_reopt'):
                node.status = 'running'
                log_info([
                    f"\n{'-'*70}\n",
                    f"Optimizing path: {path_id} (depth={node.depth}, images={len(node.images)})\n",
                    f"{'-'*70}\n",
                ], self.output)

            # Run one NEB iteration
            converged = self._run_single_path_iteration(path_id)

            # Log progress periodically
            if node.iteration % 10 == 0:
                energies = node.energies
                hei = node.hei_idx if node.hei_idx > 0 else 1

                if p.use_dynamic_k:
                    k_springs = compute_dynamic_k(energies, p.k_min, p.k_max, p.k_decay)
                else:
                    k_springs = [p.k_max] * len(node.images)

                Fp_list, maxfp, _ = neb_forces(node.images, energies, k_springs=k_springs)
                rmsfp = rms_force(Fp_list)
                dE = energies[hei] - energies[0]

                log_info([
                    f"  Iter {node.iteration:4d}  HEI={hei}  dE={dE:.6f}  "
                    f"max|Fp|={maxfp:.6f}  RMS(Fp)={rmsfp:.6f}\n"
                ], self.output)

            if converged:
                node.status = 'converged'
                log_info([f"\n[Path {path_id}] Converged after {node.iteration} iterations.\n"], self.output)
                self._print_path_summary(path_id)
                continue

            # Check for adaptive insertion
            if node.iteration % p.ang_iter == 0 and node.iteration > 0:
                self._check_adaptive_insertion(path_id)

            # Check for local minima
            if node.iteration % p.path_iter == 0 and node.iteration > 0:
                minima = self._detect_local_minima(path_id)
                if minima:
                    # Split at first minimum (closest to start)
                    self._split_path(path_id, minima[0])
                    continue

            # Check for endpoint updates
            if node.iteration % p.ep_iter == 0 and node.iteration > 0:
                new_start, new_end = self._check_endpoint_updates(path_id)
                if new_start is not None:
                    self._update_endpoint(path_id, 'start', new_start)
                if new_end is not None:
                    self._update_endpoint(path_id, 'end', new_end)

            # Check if path reached max iterations
            if node.iteration >= p.max_iter:
                log_info([f"\n[Path {path_id}] Reached max iterations ({p.max_iter}).\n"], self.output)
                node.status = 'converged'  # Mark as done even if not fully converged
                self._print_path_summary(path_id)

        # Final output before refinement
        log_info([
            "\n" + "=" * 70 + "\n",
            "AutoNEB Tree Optimization Complete\n",
            "=" * 70 + "\n",
            f"Total global iterations: {self.global_iteration}\n",
            f"Total paths: {len(self.path_tree)}\n",
            f"Intermediates found: {len(self.all_intermediates)}\n",
        ], self.output)

        # Merge global MEP
        global_images, global_energies = self._merge_global_mep()

        # Final global refinement
        if p.do_final_refine and len(global_images) > 3:
            log_info([
                "\n" + "=" * 70 + "\n",
                "Starting Final Global MEP Refinement\n",
                "=" * 70 + "\n",
            ], self.output)
            global_images, global_energies = self._run_final_refinement(global_images, global_energies)

        # Store final results for output
        self.final_images = global_images
        self.final_energies = global_energies

        self._write_outputs()

        # Print global MEP summary
        kcal = 627.509
        log_info([
            "\nGlobal MEP Summary:\n",
            f"  Total images: {len(global_images)}\n",
            f"  Energy range: {min(global_energies):.6f} to {max(global_energies):.6f} Eh\n",
            f"  Overall barrier: {(max(global_energies) - global_energies[0]) * kcal:.2f} kcal/mol\n",
        ], self.output)


# =============================================================================
# OPT-IN BATCHED AutoNEB  (additive; the single-reaction AutoNEB class above is
# kept BYTE-IDENTICAL as the parity oracle).
#
# AutoNEBBatch runs N INDEPENDENT AutoNEB reactions and evaluates the band
# forces of every LIVE image across ALL reactions in ONE batched GPU forward
# (calc.prepare(atoms_list) + calc.get_ef_gpu()) per optimizer half-step. UMA's
# per-atom mol_idx (and standard multi-graph MACE / ANI / decoupled AIMNet2)
# keeps each reaction's graph block-diagonal, so the batched forward yields each
# image's E/F independently -- identical to evaluating them one at a time
# (CatTSunami / OCPNEB mechanic, Wander et al. arXiv:2405.02078,
# ACS Catal. 2024, DOI 10.1021/acscatal.4c04272).
#
# KNOWN HARD PART -- adaptive image insertion diverges the batch width.
# AutoNEB inserts a DIFFERENT number of images per reaction over time (and splits
# paths at intermediate minima), so the number of live images per reaction --
# hence the batch width -- changes as the run proceeds. We DO NOT lockstep the
# outer control flow: the per-reaction image-insertion / path-splitting /
# endpoint-update / convergence / DFS scheduling stays PER-ITEM (delegated
# verbatim to a private _BatchReaction(AutoNEB) sub-object per reaction). ONLY
# the band force-evaluation is batched: every global half-step re-packs the flat
# image list from each reaction's CURRENT band (so a reaction that just inserted
# an image or split a path simply contributes a different-length block next
# step), and finished reactions are masked out of the batch. This is exactly the
# NEB.run_multiband "image-as-batch over bands + per-band L-BFGS + re-pack on
# membership change" pattern, extended with AutoNEB's adaptive re-pack.
#
# CEILING (documented, per project profiling the band force-eval is 95-99% of
# wall): the batched speedup covers (a) the tree-phase per-iteration band force
# eval and (b) the final global-refinement band force eval -- both cross-reaction
# batched. The per-reaction CONTROL FLOW (which images to insert, where to split,
# whether to trim an endpoint, DFS path selection) is intentionally serial numpy
# (cheap, and diverges per reaction so it cannot be lockstepped). Reactions that
# finish their tree at different global steps are simply dropped from the active
# batch; the batch therefore shrinks toward the end of a campaign (the standard
# straggler tail). Cross-reaction batching is only physically valid for
# calculators whose co-batched molecules stay decoupled (see the backend gate).
# =============================================================================


class _BatchReaction(AutoNEB):
    """One reaction inside an ``AutoNEBBatch`` campaign.

    Identical to ``AutoNEB`` (all adaptive-insertion / path-split / endpoint /
    DFS / merge / final-refine / output machinery inherited verbatim) EXCEPT the
    energy-evaluation entry point ``_get_energies`` is redirected through the
    shared batched calculator when one is bound. This keeps the per-reaction
    control-flow energy re-computes (adaptive-insertion re-energize, endpoint
    checks, final harvest) working for a batch-only calculator (UMA/MACE/etc.)
    that has no per-atom ASE ``get_potential_energy``; when no batched evaluator
    is bound it falls back to the parent per-atom behaviour (the oracle path)."""

    def _bind_batch(self, batch_eval_fn):
        # batch_eval_fn(images) -> (energies_list, raw_forces_list) via the shared
        # AutoNEBBatch calculator (a single-reaction prepare()+get_ef_gpu()).
        self._batch_eval_fn = batch_eval_fn

    def _get_energies(self, images):
        fn = getattr(self, "_batch_eval_fn", None)
        if fn is not None:
            return fn(images)[0]
        return super()._get_energies(images)

    def _batch_energy(self, atoms):
        """Single-structure energy via the shared batched evaluator (falls back
        to the parent per-atom ASE read when no batched evaluator is bound)."""
        fn = getattr(self, "_batch_eval_fn", None)
        if fn is not None:
            return float(fn([atoms])[0][0])
        return float(atoms.get_potential_energy(force_consistent=True))

    def _sync_shared_endpoint(self, path_id, which_end, new_point):
        """Override of ``AutoNEB._sync_shared_endpoint`` that routes the shared-
        endpoint energy read through the batched evaluator instead of
        ``Atoms.get_potential_energy`` (which a BATCH-ONLY calculator -- UMA /
        MACE / ... -- does not provide). Without this, a path split followed by a
        shared-endpoint trim would crash the whole batched campaign. Control-flow
        logic is byte-identical to the parent; only the energy source changes."""
        node = self.path_tree[path_id]
        if not node.parent:
            return
        parent = self.path_tree[node.parent]
        siblings = [c for c in parent.children if c != path_id]
        for sib_id in siblings:
            sib = self.path_tree[sib_id]
            if which_end == 'start' and sib.end_is_shared:
                # this path's start is the sibling's end
                sib.images[-1] = self._copy_atoms_with_calc(new_point)
                sib.energies[-1] = self._batch_energy(new_point)
                sib.status = 'needs_reopt'
                log_info([f"[Path {sib_id}] Marked for re-optimization "
                          f"(shared endpoint updated)\n"], self.output)
            elif which_end == 'end' and sib.start_is_shared:
                # this path's end is the sibling's start
                sib.images[0] = self._copy_atoms_with_calc(new_point)
                sib.energies[0] = self._batch_energy(new_point)
                sib.status = 'needs_reopt'
                log_info([f"[Path {sib_id}] Marked for re-optimization "
                          f"(shared endpoint updated)\n"], self.output)

    def _write_outputs(self):
        """Override of ``AutoNEB._write_outputs`` that sources the intermediates /
        transition-state energies from the batched evaluator (``_batch_energy``)
        instead of ``Atoms.get_potential_energy`` -- which a BATCH-ONLY calculator
        (UMA / MACE / ...) does NOT implement. The parent version's two per-atom
        reads (intermediates + TS list) would raise ``AttributeError`` on a batch-
        only calc, get swallowed by ``run()``'s guard, and SILENTLY drop the TS-
        list / intermediates / path-tree JSON / per-path MEP files while the global
        MEP was already written. This override writes the FULL parent file set on
        the batched path; every non-energy line is byte-for-byte the parent's.
        Parent ``AutoNEB`` stays untouched."""
        base, _ = os.path.splitext(self.output)

        # Global MEP (use final refined images if available)
        if hasattr(self, 'final_images') and self.final_images:
            global_images = self.final_images
            global_energies = self.final_energies
        else:
            global_images, global_energies = self._merge_global_mep()

        mep_file = base + "_autoneb_global_mep.xyz"
        write_xyz(mep_file, global_images, energies=global_energies)
        log_info([f"\nWrote global MEP to: {mep_file}\n"], self.output)

        # Intermediates (energies via batched evaluator, NOT Atoms.get_potential_energy)
        if self.all_intermediates:
            int_file = base + "_autoneb_intermediates.xyz"
            int_energies = [self._batch_energy(at) for at in self.all_intermediates]
            write_xyz(int_file, self.all_intermediates, energies=int_energies)
            log_info([f"Wrote intermediates to: {int_file}\n"], self.output)

        # Transition states (energies via batched evaluator)
        self._collect_all_ts()
        if self.all_ts:
            ts_file = base + "_autoneb_ts_list.xyz"
            ts_energies = [self._batch_energy(at) for at in self.all_ts]
            write_xyz(ts_file, self.all_ts, energies=ts_energies)
            log_info([f"Wrote transition states to: {ts_file}\n"], self.output)

        # Path tree (JSON for debugging)
        tree_file = base + "_autoneb_tree.json"
        tree_data = {}
        for pid, node in self.path_tree.items():
            tree_data[pid] = {
                'status': node.status,
                'depth': node.depth,
                'n_images': len(node.images),
                'parent': node.parent,
                'children': node.children,
                'hei_idx': node.hei_idx,
                'iteration': node.iteration,
            }
        with open(tree_file, 'w') as f:
            json.dump(tree_data, f, indent=2)
        log_info([f"Wrote path tree to: {tree_file}\n"], self.output)

        # Individual path MEPs (energies from stored node.energies; no eval)
        def write_path_meps(path_id):
            node = self.path_tree[path_id]
            if node.status == 'split':
                for c in node.children:
                    write_path_meps(c)
            elif node.status == 'converged':
                path_file = base + f"_autoneb_path_{path_id}_mep.xyz"
                write_xyz(path_file, node.images, energies=node.energies)

        write_path_meps(self.root_path_id)


@dataclass
class AutoNEBBatchParams(AutoNEBParams):
    """AutoNEBBatch shares AutoNEBParams exactly (so a batched reaction follows
    the byte-identical trajectory of the single-reaction oracle). Kept as a named
    subclass purely for the ``('autonebbatch', ...)`` paras alias."""
    pass


class AutoNEBBatch(JobABC):
    """OPT-IN batched AutoNEB over N independent reactions.

    Parameters
    ----------
    output : str
        Base output path; each reaction i logs to ``<base>_rxn{i}<ext>``.
    reactions : List[Molecules | List[Atoms]]
        N reaction inputs, each the reactant/product (or a pre-built band) for
        one AutoNEB reaction. Same per-item form accepted by ``AutoNEB``.
    calc : object, optional
        The SHARED batched calculator exposing the batch contract
        ``prepare(atoms_list, fixed_nmax) -> get_ef_gpu() -> (E(B,), F(B,M))``
        [Hartree, Eh/Angstrom]. If ``None`` it is taken from the first
        ``Molecules`` input that carries a ``.calc``.
    paras : dict, optional
        Same AutoNEB parameter dict; applied identically to every reaction.
    outputs : List[str], optional
        Explicit per-reaction output paths (overrides the ``<base>_rxn{i}``
        default); must have length N.

    Backend gate (``_is_batch_calc`` / ``_is_coupled_calc``):
      * Batchable (UMA-local, standard MACE / MACE-OFF, decoupled AIMNet2, ANI):
        graphs stay block-diagonal per molecule -> FULL cross-reaction batch.
      * Coupled / polarizable (MACE-POL, global-charge-eq AIMNet2 / NSE): a
        single molecular graph couples co-batched molecules (~1e-3 Ha/A leak,
        above the 1e-4 parity gate) -> cross-reaction batching is physically
        WRONG. ``run()`` raises a clear ``NotImplementedError`` (mirrors the
        MACE-POL batch calculator's own ``coupling_mode='raise'`` default); run
        the reactions individually via the single-reaction ``AutoNEB`` instead.
      * Non-batch calculator (no prepare/get_ef_gpu): raises ``ValueError`` (use
        the single-reaction ``AutoNEB``).
    """

    def __init__(self, output, reactions, calc=None, paras=None, outputs=None):
        base_out = output if isinstance(output, str) else (outputs[0] if outputs else "autoneb_batch.out")
        super().__init__(base_out)

        if not isinstance(reactions, (list, tuple)) or len(reactions) < 1:
            raise ValueError("AutoNEBBatch needs a non-empty list of reactions")

        self.params = self._init_params(
            AutoNEBBatchParams, paras, ("autonebbatch", "autoneb", "AutoNEB", "ts"))

        # --- resolve the shared batched calculator ---
        if calc is None:
            for r in reactions:
                if isinstance(r, Molecules) and getattr(r, "calc", None) is not None:
                    calc = r.calc
                    break
        self._calc = calc
        self._is_batch = self._is_batch_calc(calc)
        self._coupled = self._is_coupled_calc(calc)

        # --- diagnostics (batched-forward accounting) ---
        self._forwards = 0            # # of calc.prepare()+get_ef_gpu() calls
        self._image_evals = 0         # total image force-evals across all forwards

        # --- build one _BatchReaction per reaction (control flow inherited) ---
        base, ext = os.path.splitext(base_out)
        if not ext:
            ext = ".out"
        self.reactions: List[_BatchReaction] = []
        for i, r in enumerate(reactions):
            out_i = (outputs[i] if (outputs and i < len(outputs))
                     else f"{base}_rxn{i}{ext}")
            rxn = _BatchReaction(out_i, r, paras)
            rxn._bind_batch(self._batched_eval)
            rxn._done = False
            self.reactions.append(rxn)

        self.results: List[dict] = []

    # ------------------------------------------------------------------ gates
    @staticmethod
    def _is_batch_calc(calc) -> bool:
        """A batch calculator exposes prepare() + get_ef_gpu() (shared predicate)."""
        from ..._batch_calc_utils import is_batch_calc
        return is_batch_calc(calc)

    @staticmethod
    def _is_coupled_calc(calc) -> bool:
        """True for calculators whose co-batched molecules are COUPLED, so a
        cross-reaction (or even intra-band) packed forward does NOT reproduce
        isolated single-molecule physics: MACE-POL (long-range polarization,
        ``coupling_mode`` attr) and the global-charge-equilibration AIMNet2
        (``AIMNet2BatchCalc`` -- as opposed to ``AIMNet2DecoupledBatchCalc``) /
        any NSE variant. Mirrors NEB's reliance on the calc's own coupling
        guard, but detected up-front so a silently-wrong batch is refused."""
        if calc is None:
            return False
        name = type(calc).__name__.lower()
        if "decoupl" in name or getattr(calc, "batch_decoupled", False) is True:
            return False                            # decoupled variants are safe
        if bool(getattr(calc, "SUPPORTS_COUPLING", False)):  # declarative capability (BatchCalcABC)
            return True
        if hasattr(calc, "coupling_mode"):          # MACEPolBatchCalc
            return True
        if "pol" in name or "nse" in name:          # polarizable / global charge-eq
            return True
        if name == "aimnet2batchcalc":              # global charge-eq (coupled)
            return True
        mn = str(getattr(calc, "_model_name", None)
                 or getattr(calc, "model_name", None) or "").lower()
        if "pol" in mn or "nse" in mn:
            return True
        return False

    # -------------------------------------------------------- batched forward
    @staticmethod
    def _to_np(x):
        """Coerce E/F from get_ef_gpu to float64 numpy WITHOUT a hard torch
        dependency (accepts numpy arrays or torch tensors)."""
        if isinstance(x, np.ndarray):
            return x.astype(np.float64, copy=False)
        if hasattr(x, "detach"):                    # torch.Tensor duck-type
            return x.detach().to("cpu").numpy().astype(np.float64, copy=False)
        return np.asarray(x, dtype=np.float64)

    def _batched_eval(self, flat_images: List[Atoms]):
        """ONE ``calc.prepare()`` + ONE ``calc.get_ef_gpu()`` over an ARBITRARY
        flat list of images. Returns ``(Es, Fs)`` aligned to ``flat_images``:
        ``Es`` = list[float] (Hartree), ``Fs`` = list[(N_i,3) float64] (Eh/A).
        Every reaction's band across the whole campaign is packed here."""
        if not flat_images:
            return [], []
        calc = self._calc
        calc.prepare(flat_images, fixed_nmax=None)
        E_raw, F_raw = calc.get_ef_gpu()            # E (B,), F (B, nmax_dof)
        E_np = self._to_np(E_raw)
        F_np = self._to_np(F_raw)
        self._forwards += 1
        self._image_evals += len(flat_images)
        Es, Fs = [], []
        for i, at in enumerate(flat_images):
            n_i = len(at)
            Es.append(float(E_np[i]))
            Fs.append(F_np[i, :3 * n_i].reshape(n_i, 3).astype(np.float64, copy=True))
        return Es, Fs

    def _batched_eval_groups(self, groups: List[List[Atoms]]):
        """ONE batched forward over ALL images of ALL groups (each group = one
        reaction's current band). Re-packs on every call, so a reaction whose
        image count changed (adaptive insertion / path split / endpoint trim)
        just contributes a different-length block. Returns (E_out, F_out) with
        E_out[k]=list[float], F_out[k]=list[(n,3) f64] aligned to ``groups``."""
        flat, counts = [], []
        for g in groups:
            counts.append(len(g))
            flat.extend(g)
        Es, Fs = self._batched_eval(flat)
        E_out, F_out, pos = [], [], 0
        for c in counts:
            E_out.append(Es[pos:pos + c])
            F_out.append(Fs[pos:pos + c])
            pos += c
        return E_out, F_out

    def _ensure_driver(self, node):
        """Lazy L-BFGS driver init + state restore -- verbatim mirror of
        ``AutoNEB._run_single_path_iteration`` (so a batched reaction reproduces
        the oracle's optimizer state exactly)."""
        p = self.params
        if not hasattr(node, "_driver") or node._driver is None:
            node._driver = LBFGSDriver(m=p.lbfgs_m, curvature=70.0, maxstep=p.step0)
            if node.lbfgs_S:
                node._driver.S = node.lbfgs_S
                node._driver.Y = node.lbfgs_Y
                node._driver.rhos = node.lbfgs_rhos

    # ---------------------------------------------- batched tree-phase step
    def _batched_path_step(self, active):
        """Advance every ACTIVE reaction by ONE NEB path-iteration, batching the
        two band force evaluations across ALL active reactions.

        ``active`` : list of (rxn, node). Mirrors ``_run_single_path_iteration``
        exactly per reaction, but the ``_get_energies`` + ``neb_forces`` reads at
        each geometry are replaced by ONE cross-reaction batched forward:
          EVAL-A : full band of every active reaction  -> converge-test / propose
          EVAL-B : full band of every reaction that STEPPED -> L-BFGS curvature.
        Returns a list[bool] (converged-this-step) aligned to ``active``."""
        p = self.params
        for _, node in active:
            self._ensure_driver(node)

        # ---- EVAL-A : ONE batched forward over every active band ----
        E_A, F_A = self._batched_eval_groups([node.images for _, node in active])

        converged = [False] * len(active)
        stepped = []                                    # (slot, rxn, node)
        for slot, (rxn, node) in enumerate(active):
            images = node.images
            energies = E_A[slot]
            node.energies = energies
            if p.use_dynamic_k:
                k_springs = compute_dynamic_k(energies, p.k_min, p.k_max, p.k_decay)
            else:
                k_springs = [p.k_max] * len(images)
            Fp_list, maxfp, hei_idx = neb_forces(
                images, energies, k_spring=None, k_springs=k_springs,
                use_dynamic_k=False, raw_forces=F_A[slot])
            node.hei_idx = hei_idx
            rmsfp = rms_force(Fp_list)
            node._last_maxfp = maxfp
            node._last_rmsfp = rmsfp

            if maxfp < p.autoneb_f_max_th and rmsfp < p.autoneb_f_rms_th:
                converged[slot] = True
                continue

            driver = node._driver
            grads = [(-Fp_list[i]).reshape(-1) for i in range(1, len(images) - 1)]
            g = np.concatenate(grads) if grads else np.zeros(0)
            step = driver.two_loop(g)
            step = driver.step_limit(step)
            x = (np.concatenate([to_numpy_f64(images[i].get_positions()).reshape(-1)
                                 for i in range(1, len(images) - 1)])
                 if len(images) > 2 else np.zeros(0))
            x_new = x + step
            offset = 0
            for i in range(1, len(images) - 1):
                n = len(images[i]) * 3
                images[i].set_positions(x_new[offset:offset + n].reshape(-1, 3))
                offset += n
            node._step_x = x
            node._step_xnew = x_new
            node._step_g = g
            stepped.append((slot, rxn, node))

        # ---- EVAL-B : ONE batched forward over every band that stepped ----
        if stepped:
            E_B, F_B = self._batched_eval_groups([node.images for _, _, node in stepped])
            for k, (slot, rxn, node) in enumerate(stepped):
                images = node.images
                new_energies = E_B[k]
                node.energies = new_energies
                if p.use_dynamic_k:
                    k_springs = compute_dynamic_k(new_energies, p.k_min, p.k_max, p.k_decay)
                else:
                    k_springs = [p.k_max] * len(images)
                new_Fp_list, _, _ = neb_forces(
                    images, new_energies, k_springs=k_springs, raw_forces=F_B[k])
                new_grads = [(-new_Fp_list[i]).reshape(-1) for i in range(1, len(images) - 1)]
                g_new = np.concatenate(new_grads) if new_grads else np.zeros(0)
                driver = node._driver
                g = node._step_g
                if len(g) > 0:
                    driver.update(node._step_xnew - node._step_x, g_new - g)
                node.lbfgs_S = list(driver.S)
                node.lbfgs_Y = list(driver.Y)
                node.lbfgs_rhos = list(driver.rhos)
                node.iteration += 1

        return converged

    def _post_iteration_control(self, rxn, path_id, node, converged):
        """Per-reaction control flow AFTER one batched path-iteration -- a
        verbatim mirror of ``AutoNEB.run()``'s post-iteration block (adaptive
        insertion / local-minima split / endpoint update / max-iter), kept
        PER-ITEM (it diverges per reaction and must not be lockstepped). The only
        change vs the oracle is that periodic logging reuses the force metrics
        already computed in the batched step (no extra per-reaction forward)."""
        p = self.params

        if node.iteration % 10 == 0 and node.iteration > 0:
            energies = node.energies
            hei = node.hei_idx if node.hei_idx > 0 else 1
            dE = energies[hei] - energies[0]
            log_info([
                f"[{rxn.output}] Iter {node.iteration:4d}  HEI={hei}  dE={dE:.6f}  "
                f"max|Fp|={getattr(node, '_last_maxfp', 0.0):.6f}  "
                f"RMS(Fp)={getattr(node, '_last_rmsfp', 0.0):.6f}\n"
            ], self.output)

        if converged:
            node.status = 'converged'
            log_info([f"\n[{rxn.output}][Path {path_id}] Converged after "
                      f"{node.iteration} iterations.\n"], self.output)
            rxn._print_path_summary(path_id)
            return

        # adaptive insertion (changes image count -> batch re-packs next step)
        if node.iteration % p.ang_iter == 0 and node.iteration > 0:
            rxn._check_adaptive_insertion(path_id)

        # local-minima detection + path split (per-item divergence)
        if node.iteration % p.path_iter == 0 and node.iteration > 0:
            minima = rxn._detect_local_minima(path_id)
            if minima:
                rxn._split_path(path_id, minima[0])
                return

        # endpoint updates
        if node.iteration % p.ep_iter == 0 and node.iteration > 0:
            new_start, new_end = rxn._check_endpoint_updates(path_id)
            if new_start is not None:
                rxn._update_endpoint(path_id, 'start', new_start)
            if new_end is not None:
                rxn._update_endpoint(path_id, 'end', new_end)

        # per-path max-iter cap
        if node.iteration >= p.max_iter:
            log_info([f"\n[{rxn.output}][Path {path_id}] Reached max iterations "
                      f"({p.max_iter}).\n"], self.output)
            node.status = 'converged'
            rxn._print_path_summary(path_id)

    # ------------------------------------------ batched final refinement
    def _batched_final_refine(self):
        """Cross-reaction batched final global MEP refinement.

        Each reaction's merged MEP is refined with the SAME looser-threshold
        endpoint-optimizing NEB as ``AutoNEB._run_final_refinement`` (byte-for-
        byte per reaction), but the two per-iteration band force evals are
        batched across all still-refining reactions. Endpoint trimming changes a
        reaction's image count -> the flat batch is re-packed each half-step;
        reactions that converge / hit ``final_refine_max_iter`` are masked out.
        Reactions whose MEP is <=3 images (or do_final_refine off) skip refine
        and keep the merged MEP directly."""
        p = self.params
        states = []
        for rxn in self.reactions:
            gi, ge = rxn._merge_global_mep()
            if not (p.do_final_refine and len(gi) > 3):
                rxn.final_images = gi
                rxn.final_energies = rxn._get_energies(gi) if gi else []
                continue
            ref_images = [rxn._copy_atoms_with_calc(img) for img in gi]
            states.append(dict(
                rxn=rxn, images=ref_images,
                driver=LBFGSDriver(m=p.lbfgs_m, curvature=70.0, maxstep=p.step0),
                f_max_th=p.autoneb_f_max_th * p.final_refine_factor,
                f_rms_th=p.autoneb_f_rms_th * p.final_refine_factor,
                it=0, done=False,
            ))
        if not states:
            return

        max_it = p.final_refine_max_iter
        while any(not s["done"] for s in states):
            active = [s for s in states if not s["done"]]

            # ---- EVAL-A ----
            E_A, F_A = self._batched_eval_groups([s["images"] for s in active])
            stepped = []
            for k, s in enumerate(active):
                imgs = s["images"]
                n = len(imgs)
                energies = E_A[k]
                if p.use_dynamic_k:
                    k_springs = compute_dynamic_k(energies, p.k_min, p.k_max, p.k_decay)
                else:
                    k_springs = [p.k_max] * n
                Fp_list, maxfp, _hei = neb_forces(
                    imgs, energies, k_springs=k_springs,
                    use_dynamic_k=False, raw_forces=F_A[k])
                rmsfp = rms_force(Fp_list)
                if maxfp < s["f_max_th"] and rmsfp < s["f_rms_th"]:
                    s["done"] = True
                    continue
                drv = s["driver"]
                grads = [(-Fp_list[i]).reshape(-1) for i in range(1, n - 1)]
                g = np.concatenate(grads) if grads else np.zeros(0)
                step = drv.step_limit(drv.two_loop(g))
                x = (np.concatenate([to_numpy_f64(imgs[i].get_positions()).reshape(-1)
                                     for i in range(1, n - 1)]) if n > 2 else np.zeros(0))
                x_new = x + step
                off = 0
                for i in range(1, n - 1):
                    m = len(imgs[i]) * 3
                    imgs[i].set_positions(x_new[off:off + m].reshape(-1, 3))
                    off += m
                s["_x"] = x
                s["_xnew"] = x_new
                s["_g"] = g
                stepped.append(s)

            # ---- EVAL-B ----
            if stepped:
                E_B, F_B = self._batched_eval_groups([s["images"] for s in stepped])
                for k, s in enumerate(stepped):
                    imgs = s["images"]
                    n = len(imgs)
                    new_energies = E_B[k]
                    if p.use_dynamic_k:
                        k_springs = compute_dynamic_k(new_energies, p.k_min, p.k_max, p.k_decay)
                    else:
                        k_springs = [p.k_max] * n
                    new_Fp_list = neb_forces(
                        imgs, new_energies, k_springs=k_springs, raw_forces=F_B[k])[0]
                    new_grads = [(-new_Fp_list[i]).reshape(-1) for i in range(1, n - 1)]
                    g_new = np.concatenate(new_grads) if new_grads else np.zeros(0)
                    if len(s["_g"]) > 0:
                        s["driver"].update(s["_xnew"] - s["_x"], g_new - s["_g"])
                    s["it"] += 1

                    # endpoint updates (byte-for-byte mirror of the oracle's
                    # _run_final_refinement, incl. its index-after-front-slice
                    # ordering) -- image count changes -> re-pack next half-step.
                    if s["it"] % p.final_refine_ep_iter == 0:
                        ns, ne = s["rxn"]._check_endpoint_updates_global(imgs, new_energies)
                        if ns is not None:
                            s["images"] = s["images"][ns:]
                            s["driver"] = LBFGSDriver(m=p.lbfgs_m, curvature=70.0, maxstep=p.step0)
                            imgs = s["images"]
                        if ne is not None:
                            s["images"] = imgs[:ne + 1]
                            s["driver"] = LBFGSDriver(m=p.lbfgs_m, curvature=70.0, maxstep=p.step0)

                    if s["it"] >= max_it:
                        s["done"] = True

        for s in states:
            rxn = s["rxn"]
            rxn.final_images = s["images"]
            rxn.final_energies = rxn._get_energies(s["images"])

    # -------------------------------------------------------------- driver
    def _reaction_result(self, rxn) -> dict:
        imgs = getattr(rxn, "final_images", None) or []
        ens = list(getattr(rxn, "final_energies", None) or [])
        if len(ens) >= 3:
            hei = max(range(1, len(ens) - 1), key=lambda i: ens[i])
        elif len(ens) >= 1:
            hei = len(ens) - 1
        else:
            hei = 0
        return dict(
            output=rxn.output,
            n_images=len(imgs),
            energies=ens,
            hei=hei,
            ts_image=(imgs[hei] if imgs else None),
            barrier_Eh=(ens[hei] - ens[0]) if ens else 0.0,
            reaction_energy_Eh=(ens[-1] - ens[0]) if ens else 0.0,
        )

    def run(self):
        """Batched multi-reaction AutoNEB driver.

        Tree phase: every global half-step re-selects each reaction's active path
        (DFS) and advances it by ONE batched path-iteration (all live images of
        all reactions in one forward); per-reaction control flow (insertion /
        split / endpoint / convergence) runs per-item. Then a cross-reaction
        batched final refinement. Returns one result dict per reaction."""
        if not self._is_batch:
            raise ValueError(
                "AutoNEBBatch requires a batched calculator exposing "
                "prepare(atoms_list) + get_ef_gpu(); none was supplied. "
                "Use the single-reaction AutoNEB for a serial ASE calculator.")
        if self._coupled:
            raise NotImplementedError(
                f"AutoNEBBatch cannot cross-reaction batch a COUPLED calculator "
                f"({type(self._calc).__name__}): co-batched molecules are "
                f"physically coupled (MACE-POL long-range polarization / global "
                f"charge-equilibration AIMNet2), so a packed forward does not "
                f"reproduce isolated single-molecule physics. Run each reaction "
                f"with the single-reaction AutoNEB, or use a decoupled backend "
                f"(UMA-local / standard MACE / MACE-OFF / decoupled AIMNet2 / ANI).")

        log_info([
            "\n" + "=" * 70 + "\n",
            f"Starting AutoNEBBatch  (N={len(self.reactions)} reactions, "
            f"calc={type(self._calc).__name__})\n",
            "Cross-reaction batched band force eval; per-reaction adaptive "
            "insertion / split / endpoint control flow kept per-item.\n",
            "=" * 70 + "\n",
        ], self.output)

        # init every reaction's root path (energies via the batched evaluator)
        for rxn in self.reactions:
            rxn._init_root_path()

        p = self.params
        max_global = p.max_iter * p.max_paths
        # global safety bound: each half-step advances >=1 reaction by >=1 of its
        # own path-iters, and every reaction leaves at <= max_global own-iters.
        hard_cap = max_global * len(self.reactions) + 32
        gstep = 0
        self._aborted = False
        while gstep < hard_cap:
            gstep += 1
            active = []
            for rxn in self.reactions:
                if rxn._done:
                    continue
                if rxn.global_iteration >= max_global or rxn._all_leaves_converged():
                    rxn._done = True
                    continue
                path_id = rxn._select_next_path()
                if path_id is None:
                    rxn._done = True
                    continue
                node = rxn.path_tree[path_id]
                rxn.global_iteration += 1
                if node.status in ('pending', 'needs_reopt'):
                    node.status = 'running'
                    log_info([
                        f"\n[{rxn.output}] Optimizing path: {path_id} "
                        f"(depth={node.depth}, images={len(node.images)})\n"
                    ], self.output)
                active.append((rxn, path_id, node))

            if not active:
                break

            # N1/N2 graceful abort: a batched forward can fail on OOM / NaN /
            # calculator error; catch it, stop the tree phase, and emit partial
            # output (below) rather than crashing the whole campaign with none.
            try:
                converged = self._batched_path_step(
                    [(rxn, node) for rxn, _pid, node in active])
            except Exception as _err:                    # noqa: BLE001
                log_info([
                    f"\nAutoNEBBatch: batched forward FAILED at global step "
                    f"{gstep}: {_err!r}\n"
                    "Aborting the tree phase; emitting partial output.\n"
                ], self.output)
                self._aborted = True
                break
            for (rxn, path_id, node), conv in zip(active, converged):
                self._post_iteration_control(rxn, path_id, node, conv)

        log_info([
            "\n" + "=" * 70 + "\n",
            "AutoNEBBatch tree phase complete\n",
            f"  global half-steps: {gstep}\n",
            f"  batched forwards:  {self._forwards}\n",
            f"  image-evals:       {self._image_evals}\n",
            "=" * 70 + "\n",
        ], self.output)

        # cross-reaction batched final global refinement (also a batched forward
        # -> guarded the same way; on failure fall back to the unrefined MEP).
        if not getattr(self, "_aborted", False):
            try:
                self._batched_final_refine()
            except Exception as _err:                    # noqa: BLE001
                log_info([f"\nAutoNEBBatch: batched final refinement FAILED: "
                          f"{_err!r}\nEmitting partial (unrefined) merged MEP.\n"],
                         self.output)
                self._aborted = True

        # partial output: fill any reaction still missing a final MEP from its
        # last-good stored node energies (_merge_global_mep issues NO calc call).
        if getattr(self, "_aborted", False):
            for rxn in self.reactions:
                if not getattr(rxn, "final_images", None):
                    gi, ge = rxn._merge_global_mep()
                    rxn.final_images = gi
                    rxn.final_energies = ge

        # per-reaction outputs (best-effort; output energy reads use the calc's
        # per-atom ASE interface where present -- see ceiling note above)
        for rxn in self.reactions:
            try:
                rxn._write_outputs()
            except Exception as _e:                 # noqa: BLE001 (output only)
                log_info([f"[{rxn.output}] _write_outputs skipped: {_e}\n"], self.output)

        self.results = [self._reaction_result(rxn) for rxn in self.reactions]

        kcal = 627.509
        for r in self.results:
            ens = r["energies"]
            if ens:
                log_info([
                    f"\n[{r['output']}] final MEP: images={r['n_images']}  "
                    f"HEI={r['hei']}  barrier={r['barrier_Eh'] * kcal:.3f} kcal/mol  "
                    f"dErxn={r['reaction_energy_Eh'] * kcal:.3f} kcal/mol\n"
                ], self.output)
        log_info([
            f"\nAutoNEBBatch done: {self._forwards} batched forwards, "
            f"{self._image_evals} image-evals over {len(self.reactions)} reactions.\n"
        ], self.output)
        return self.results
