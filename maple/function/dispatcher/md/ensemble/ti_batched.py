"""Batched Thermodynamic Integration (TI) + MBAR for MAPLE -- N_lambda coupling
windows ride ``BatchedNVT`` as the batch dimension, then the EXISTING alchemical
estimators (``bias/ti_mbar.py``) give the free-energy difference two ways (TI and
MBAR) from the SAME collected data.

This is the FIRST-CLASS ensemble entry for alchemical free energy -- the SAME
sanctioned pattern as ``BatchedUmbrella`` (N harmonic windows on the batch axis ->
WHAM PMF in ``run()``) and ``BatchedEABF``: a ``BatchedNVT`` subclass with a
``*Params(BatchedNVTParams)`` dataclass, replica replication to N windows, a
per-replica ``self._bias``, and ``run()`` = ``super().run()`` + the estimator
post-processing (here TI trapezoid + MBAR fixed-point instead of WHAM).

Algorithm -- linear two-state coupling on the batch axis
--------------------------------------------------------
Each window ``b`` carries its own coupling ``lambda_b`` (grid over
[``ti_lambda_min``, ``ti_lambda_max``], nominally 0..1). The batched forward IS
end-state A (the base MLIP model -> E_A/F_A in the padded buffer); the per-replica
:class:`bias.lambda_mix.BatchedLambdaMix` mixes toward end-state B,

    E_lambda = (1-lambda) E_A + lambda E_B ,   F_lambda = (1-lambda) F_A + lambda F_B ,

so ALL N_lambda windows advance with ONE ``calc.get_ef_gpu()`` per step (the batch
axis). End-state B is the toy analytic :class:`bias.lambda_mix.HarmonicEndState`
(an Einstein-crystal tether, force const ``ti_kappa`` [Ha/Angstrom^2], reference =
each replica's initial geometry) -- so the CORE run needs a SINGLE forward (no
second MLIP). ``dU/dlambda = E_B - E_A`` and the two end-state energies are logged
per window.

Post-processing (both from the same samples, in ``run()``):
  * **TI**   : dF = integral_0^1 <dU/dlambda>_lambda dlambda  (trapezoid over the grid).
  * **MBAR** : cross-evaluate each window's (E_A, E_B) samples at ALL lambda (the
               ``u_kn`` matrix) -> self-consistent MBAR free energies -> dF.
On the same data TI and MBAR agree within statistical error (the in-tree gate).

Units: energies/kT in Hartree (``kT = KELVIN_TO_HARTREE * temperature``); dF in Ha.
``ti_kappa`` is Ha/Angstrom^2 (the MAPLE force convention shared with the batched
restraint / SMD). Isolated replicas (the batch backends carry no cell).

ponytail: reuse ``BatchedNVT`` (one batched forward + thermostats/projection) and
the pure-numpy ``ti_mbar`` estimators verbatim; this wrapper is only the ensemble
entry + param plumbing + the TI/MBAR call in ``run()``. The coupling is the LINEAR
two-state mix only (no soft-core / no alchemical topology change), end-state B is
analytic (no second MLIP forward). See ``bias/wham2d.py`` for why the single-model
lambda-split is NOT portable -- the two-end-state mix makes dU/dlambda explicit.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from ase import Atoms

from maple.function.utility import Molecules
from .nvt_batched import BatchedNVT, BatchedNVTParams
from ..bias.lambda_mix import BatchedLambdaMix, HarmonicEndState
from ..bias.ti_mbar import ti_integrate, mbar_delta_f
from ..utils import KELVIN_TO_HARTREE                 # kB [Ha/K]  (kT = this * T)


@dataclass
class BatchedTIParams(BatchedNVTParams):
    """Batched-TI parameters = the batched-NVT params PLUS the alchemical block.
    All fields are DECLARED (B-51: ``_init_params`` keeps the keys)."""
    ti_nwindows:    int   = 8       # number of lambda windows = batch B
    ti_lambda_min:  float = 0.0     # first coupling value
    ti_lambda_max:  float = 1.0     # last coupling value
    ti_kappa:       float = 2.0     # end-state B tether const [Ha/Angstrom^2]
    ti_discard_frac: float = 0.2    # leading fraction of each window's log dropped
    ti_mbar_tol:    float = 1e-10   # MBAR fixed-point tolerance
    ti_mbar_max_iter: int = 100000  # MBAR fixed-point iteration cap


class BatchedTI(BatchedNVT):
    """N_lambda coupling windows ridden on BatchedNVT (one forward/step) -> TI + MBAR."""

    _ALIASES = ("md", "MD", "nvt", "NVT", "batched", "batchnvt",
                "ti", "TI", "mbar", "MBAR", "ti_batched")

    def __init__(self, output: str,
                 systems: Union[Molecules, List[Atoms], Atoms],
                 calc=None,
                 paras: Optional[dict] = None):
        # 1) parse TI config first (need N before replicating the template).
        tcfg = self._init_params(BatchedTIParams, paras, self._ALIASES)
        n = int(tcfg.ti_nwindows)
        if n < 2:
            raise ValueError("BatchedTI needs ti_nwindows >= 2.")
        if not (tcfg.ti_lambda_max > tcfg.ti_lambda_min):
            raise ValueError("BatchedTI requires ti_lambda_max > ti_lambda_min "
                             f"(got min={tcfg.ti_lambda_min}, max={tcfg.ti_lambda_max}).")

        # 2) resolve the template + replicate to N windows (B = N). A pre-built list
        #    of exactly N systems is used as-is.
        template, calc = self._resolve_template(systems, calc)
        if isinstance(systems, (list, tuple)) and len(systems) == n:
            replicas = [a.copy() for a in systems]
        else:
            replicas = [template.copy() for _ in range(n)]

        # 3) build the batched NVT over the N replicas (calc prepared on B = N).
        super().__init__(output, replicas, calc=calc, paras=paras)
        # re-attach the TI fields (super() parsed only BatchedNVTParams).
        self.params = self._init_params(BatchedTIParams, paras, self._ALIASES)

        # 4) per-window lambda grid + the two-state lambda-mixing bias.
        self.lambdas = np.linspace(float(tcfg.ti_lambda_min),
                                   float(tcfg.ti_lambda_max), n)          # (N,)
        ref = [at.get_positions() for at in self.atoms_list]              # end-state B ref
        self._end_state_B = HarmonicEndState(ref, float(tcfg.ti_kappa))
        self._bias = BatchedLambdaMix(self.atoms_list, self.lambdas, self._end_state_B)

        # free-energy results (filled by run()).
        self.dudl_mean = None
        self.dF_ti = None
        self.dF_mbar = None
        self.mbar_f = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _resolve_template(systems, calc):
        if isinstance(systems, Molecules):
            template = systems.multiatoms[0]
            calc = calc if calc is not None else systems.calc
        elif isinstance(systems, (list, tuple)):
            if not systems:
                raise ValueError("BatchedTI: empty systems list.")
            template = systems[0]
        else:
            template = systems              # single ase.Atoms template
        return template, calc

    # ====================================================================== run
    def run(self):
        super().run()                        # BatchedNVT advances all N lambda windows
        self._compute_free_energy()
        return self

    def _compute_free_energy(self):
        p = self.params
        disc = float(p.ti_discard_frac)
        EA_list, EB_list, dudl_means = [], [], []
        for b in range(self.B):
            eA = np.asarray(self._bias.eA_history[b], np.float64)
            eB = np.asarray(self._bias.eB_history[b], np.float64)
            du = np.asarray(self._bias.dudl_history[b], np.float64)
            if disc > 0.0 and du.size:
                s = int(disc * du.size)
                eA, eB, du = eA[s:], eB[s:], du[s:]
            EA_list.append(eA)
            EB_list.append(eB)
            dudl_means.append(float(du.mean()) if du.size else np.nan)
        self.dudl_mean = np.array(dudl_means)

        kT = KELVIN_TO_HARTREE * float(p.temperature)          # Ha
        # TI: trapezoid of <dU/dlambda> over the lambda grid (EXISTING estimator).
        self.dF_ti = ti_integrate(self.lambdas, self.dudl_mean)
        # MBAR: cross-evaluate all samples at all lambda -> self-consistent dF.
        try:
            self.dF_mbar, self.mbar_f = mbar_delta_f(
                self.lambdas, EA_list, EB_list, kT,
                tol=float(p.ti_mbar_tol), max_iter=int(p.ti_mbar_max_iter))
        except Exception as exc:                               # never crash run()
            self.dF_mbar, self.mbar_f = np.nan, None
            self._mbar_err = f"{type(exc).__name__}: {exc}"
        self._log_ti_summary(kT)

    def _log_ti_summary(self, kT):
        disagree = (abs(self.dF_ti - self.dF_mbar)
                    if np.isfinite(self.dF_mbar) else float("nan"))
        lines = ["\n" + "=" * 72 + "\n",
                 f"{'BATCHED TI + MBAR -> free energy':^72}\n", "=" * 72 + "\n",
                 f"  windows (B):   {self.B}\n",
                 f"  lambda grid:   {self.lambdas[0]:.3f} .. {self.lambdas[-1]:.3f} "
                 f"({self.B} pts)\n",
                 f"  end-state B:   Einstein tether kappa="
                 f"{self.params.ti_kappa:.4f} Ha/Angstrom^2\n",
                 f"  T:             {self.params.temperature:.1f} K   "
                 f"(kT={kT:.6f} Ha)\n",
                 f"  dF (TI):       {self.dF_ti:.6f} Ha\n",
                 f"  dF (MBAR):     {self.dF_mbar:.6f} Ha\n",
                 f"  |TI - MBAR|:   {disagree:.6f} Ha\n",
                 "\n  win   lambda    <dU/dlambda>(Ha)\n"]
        for b in range(self.B):
            lines.append(f"  {b:>3}   {self.lambdas[b]:>6.3f}   "
                         f"{self.dudl_mean[b]:>16.6f}\n")
        lines.append("=" * 72 + "\n")
        self.log_info(lines)
