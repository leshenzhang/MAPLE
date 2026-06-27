# -*- coding: utf-8 -*-
"""
GDIIS (Geometry DIIS) Accelerator

Implements the Geometry DIIS algorithm (Csaszar & Pulay, J. Mol. Struct. 1984)
with improvements from Farkas & Schlegel (PCCP, 2002) for robust geometry
optimization acceleration.

Key difference from naive DIIS:
  - Naive DIIS:  x_new = sum(c_i * x_i)           (wrong for geometry opt)
  - GDIIS:       x_new = sum(c_i * x_tilde_i)     (correct)
    where x_tilde_i = x_i + step_scale * f_i  (corrected coordinates)
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from ase import Atoms


@dataclass
class DIISParams:
    """GDIIS accelerator parameters."""
    memory: int = 6               # Maximum number of historical vectors
    min_vectors: int = 3          # Minimum vectors required for extrapolation
    regularization: float = 1e-10  # Tikhonov regularization for B matrix


class DIISAccelerator:
    """
    GDIIS (Geometry DIIS) accelerator.

    Uses the GDIIS formulation to accelerate geometry optimization:
      1. Error vectors are forces (f_i)
      2. Corrected coordinates: x_tilde_i = x_i + step_scale * f_i
      3. Extrapolated position: x_new = sum(c_i * x_tilde_i)

    The corrected coordinates represent the estimated minimum as seen from
    each historical point. For SD, step_scale approximates H^{-1}.

    Reference:
      Csaszar & Pulay, J. Mol. Struct. (Theochem) 114, 31-34 (1984)
      Farkas & Schlegel, PCCP 4, 11-15 (2002)
    """

    def __init__(self, params: Optional[DIISParams] = None):
        self.params = params if params is not None else DIISParams()
        self.x_vectors: List[np.ndarray] = []
        self.error_vectors: List[np.ndarray] = []
        self._last_reject_reason: Optional[str] = None

    def store(self, x: np.ndarray, error: np.ndarray) -> None:
        """
        Store position and error (force) vectors.

        Args:
            x: Current position vector [N, 3]
            error: Error vector (forces) same shape as x
        """
        self.x_vectors.append(x.copy())
        self.error_vectors.append(error.copy())

        if len(self.x_vectors) > self.params.memory:
            self.x_vectors.pop(0)
            self.error_vectors.pop(0)

    def can_extrapolate(self) -> bool:
        """Check if enough vectors are available for extrapolation."""
        return len(self.error_vectors) >= self.params.min_vectors

    def extrapolate(self, step_scale: float = 0.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Perform GDIIS extrapolation.

        Args:
            step_scale: Scaling factor for corrected coordinates.
                If > 0, uses GDIIS: x_tilde_i = x_i + step_scale * f_i
                If == 0, uses naive DIIS: x_tilde_i = x_i

        Returns:
            Tuple of (new_position, coefficients) or None if extrapolation fails.
        """
        self._last_reject_reason = None
        if not self.can_extrapolate():
            self._last_reject_reason = "insufficient vectors"
            return None

        m = len(self.error_vectors)

        # Build B matrix: B[i,j] = <e_i | e_j>
        B = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                B[i, j] = np.dot(
                    self.error_vectors[i].ravel(),
                    self.error_vectors[j].ravel()
                )

        # Tikhonov regularization to improve conditioning
        reg = self.params.regularization * np.max(np.abs(np.diag(B)))
        B += reg * np.eye(m)

        # Build augmented system: [B, -1; -1, 0] * [c; lambda] = [0; -1]
        A = np.zeros((m + 1, m + 1))
        A[:m, :m] = B
        A[m, :m] = -1.0
        A[:m, m] = -1.0

        b = np.zeros(m + 1)
        b[m] = -1.0

        # Use lstsq for numerical stability instead of solve
        try:
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            self._last_reject_reason = "singular matrix"
            return None

        coeffs = result[:m]

        # GDIIS: extrapolate corrected coordinates
        # x_tilde_i = x_i + step_scale * f_i  (approximate Newton correction)
        new_position = np.zeros_like(self.x_vectors[0])
        for i, c in enumerate(coeffs):
            corrected = self.x_vectors[i] + step_scale * self.error_vectors[i]
            new_position += c * corrected

        return new_position, coeffs

    def drop_oldest(self) -> None:
        """Drop the oldest stored vector pair (instead of full reset)."""
        if self.x_vectors:
            self.x_vectors.pop(0)
            self.error_vectors.pop(0)

    def reset(self) -> None:
        """Reset all stored historical vectors."""
        self.x_vectors.clear()
        self.error_vectors.clear()


# ============================================================================
# Legacy compatibility (deprecated)
# ============================================================================

class OptimizationStorage:
    """
    [DEPRECATED] Use DIISAccelerator instead.
    """
    def __init__(self):
        self.diis_x_vectors = []
        self.diis_error_vectors = []
        self.iteration = 0

    def reset(self):
        self.diis_x_vectors = []
        self.diis_error_vectors = []
        self.iteration = 0


def DIIS(atoms: Atoms, output: str, memory: int = 10,
         max_step_size: float = 0.2, maxiterations: int = 128,
         storage: Optional[OptimizationStorage] = None) -> int:
    """
    [DEPRECATED] Use SD class with diis_enabled=True instead.
    """
    import warnings
    warnings.warn(
        "DIIS() function is deprecated. Use SD class with diis_enabled=True.",
        DeprecationWarning,
        stacklevel=2
    )

    if storage is not None:
        diis_x_vectors = storage.diis_x_vectors
        diis_error_vectors = storage.diis_error_vectors
        iteration = storage.iteration
    else:
        diis_x_vectors = []
        diis_error_vectors = []
        iteration = 0

    m = len(diis_error_vectors)
    if m < 3:
        return iteration

    B = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            B[i, j] = np.dot(diis_error_vectors[i].flat, diis_error_vectors[j].flat)

    A = np.zeros((m + 1, m + 1))
    A[:m, :m] = B
    A[m, :m] = -1.0
    A[:m, m] = -1.0

    b = np.zeros(m + 1)
    b[m] = -1.0

    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return iteration

    coeff = x[:m]

    new_position = np.zeros_like(diis_x_vectors[0])
    for i, c in enumerate(coeff):
        new_position += c * diis_x_vectors[i]

    step = new_position - atoms.get_positions()
    step_length = np.sqrt((step**2).sum())

    if step_length > max_step_size:
        step = step * (max_step_size / step_length)
        new_position = atoms.get_positions() + step

    atoms.set_positions(new_position)
    return iteration
