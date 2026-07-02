# -*- coding: utf-8 -*-
"""Decoupled AIMNet2 batch calculator (cross-reaction batchable backend).

Retrofitted onto :class:`BatchCalcABC` (batch_calculator_base.py): the coord ops,
``(B, nmax_dof)`` pack/scatter, unit conversion, generic numerical FD Hessian,
PBC fail-fast, fixed_nmax validation and the registry now come from the base; this
subclass keeps ONLY the AIMNet2-specific surface (the wrapped ``aimnet2calc``
model, per-structure charge/multiplicity, the ``_forward`` bridge, the energy-only
``get_e_gpu`` single point).

It implements the duck-typed *batch contract* consumed by ``NEB.run_multiband`` /
``BatchPRFO`` / ``BatchLBFGS``::

    prepare(atoms_list, fixed_nmax=None)   # set topology + initial coords
    get_ef_gpu()  -> (E (B,) Ha, F (B, nmax_dof) Ha/Angstrom)
    get_e_gpu()   -> E (B,) Ha                      # energy-only single point

WHY A SEPARATE CALC.  The bundled ``aimnet2.pt`` (SanderAIMNet2_2025) performs a
*global* charge-equilibration over the whole forward, so co-batching independent
molecules COUPLES them (verified: 3 identical far-apart copies disagree by
~0.16 Ha; even NEB images within one band corrupt each other -> garbage ~30 eV
barriers). That model is therefore NOT cross-reaction batchable.

This calc instead wraps the upstream ``aimnet2calc.AIMNet2Calculator`` loaded
with a NON charge-equilibration model (default ``aimnet2`` ->
``aimnet2_wb97m_0.jpt``, cutoff 5.0). Its neighbour graph is segmented by
``mol_idx`` (``radius_graph`` batch=mol_idx) so NO edge ever crosses a molecule
boundary; with a short-range (non-Ewald) energy this makes every structure's
E/F exactly independent of its co-batched peers -- i.e. cross-reaction
batching is decoupled (perturb-one probe: dE_other < 1e-6 Ha). Hence
``SUPPORTS_COUPLING = False`` and ``BATCHABLE = True`` (unlike the charge-eq
sibling ``AIMNet2BatchCalc``).

The wrapped high-level API casts coordinates to float32 internally, so energies
are float32-precision (the decoupling is *structural*, i.e. zero cross terms,
not precision-limited, so the 1e-4 Ha decouple gate still passes by orders of
magnitude). Results are returned as float64 tensors for the MAPLE contract.

Runtime: requires ``aimnet2calc`` importable and ``PYTHONNOUSERSITE=1``.
"""
from __future__ import annotations

from typing import Optional

import torch

from ..batch_calculator_base import (  # noqa: F401
    BatchCalcABC,
    EV2HARTREE,
    register_batch_calculator,
)


@register_batch_calculator
class AIMNet2DecoupledBatchCalc(BatchCalcABC):
    """Batched, cross-reaction-decoupled AIMNet2 calculator.

    Parameters
    ----------
    model : str
        aimnet2calc model name or path. Default ``"aimnet2"`` ->
        ``aimnet2_wb97m_0.jpt`` (the validated decoupled model). Pass an
        explicit ``.jpt`` path to pin a specific checkpoint.
    device : str | None
        Torch device. ``None`` -> the aimnet2calc auto-selected device.
    dtype : torch.dtype
        Output dtype for the returned tensors (default float64 per the MAPLE
        contract). Note the underlying forward runs in float32.
    """

    # ---- capability protocol (BatchCalcABC declarative attrs) --------------
    MODEL_NAMES = ("aimnet2_decoupled",)
    MODEL_ENERGY_UNIT = "eV"                 # AIMNet2 returns eV; base -> Hartree once
    MODEL_DTYPE = torch.float32              # wrapped forward runs in f32 (master coord f64)
    SUPPORTS_PBC = False                     # no-PBC molecular wrapper
    SUPPORTED_HESSIAN_MODES = ("numerical",)  # base generic central-FD via _forward
    HAS_HVP = False
    SUPPORTS_COUPLING = False                # mol_idx-segmented graph -> structurally decoupled
    BATCHABLE = True                         # decoupled => cross-reaction batchable

    def __init__(self, model: str = "aimnet2", device: Optional[str] = "cuda",
                 dtype: torch.dtype = torch.float64):
        from aimnet2calc import AIMNet2Calculator  # lazy: keep maple import light

        self.aim = AIMNet2Calculator(model)
        # Resolve device BEFORE super().__init__: device=None means "use the
        # aimnet2calc auto-selected device" (a torch.device, passed through by the
        # base's _resolve_device unchanged). base.__init__ sets self.device/self.dtype
        # + the generic prepared-state (_prepared/_ptr/numbers/mol_idx/_local_atom/
        # _n_b/_cols/coord/N_atoms/Nmax_atoms/nmax_dof/_coord_backup).
        super().__init__(self.aim.device if device is None else device, dtype)
        self.cutoff = float(self.aim.cutoff)

        # ---- model-specific prepared-state ONLY (built in _build_topology) ----
        self.charge_B = None        # (B,) per-structure total charge
        self.mult_B = None          # (B,) per-structure multiplicity

    # -------------------------------------------------------- topology hook
    def _build_topology(self, atoms_list):
        """Cache AIMNet2 per-structure charge/multiplicity for the fixed batch.

        Called at the END of ``BatchCalcABC.prepare()`` -- which already built
        ptr/numbers/mol_idx/_local_atom/_n_b/_cols/coord/nmax_dof (with the shared
        ``fixed_nmax`` validation + PBC fail-fast) and reset ``_coord_backup``. The
        wrapped aimnet2calc forward is graph-segmented by ``mol_idx``, so the only
        remaining per-batch state is the (B,) charge/mult vectors read from
        ``at.info`` (default charge 0.0, mult 1.0).
        """
        device, dtype = self.device, self.dtype
        charges, mults = [], []
        for at in atoms_list:
            info = getattr(at, "info", {}) or {}
            charges.append(float(info.get("charge", 0.0)))
            mults.append(float(info.get("mult", 1.0)))
        self.charge_B = torch.tensor(charges, dtype=dtype, device=device)
        self.mult_B = torch.tensor(mults, dtype=dtype, device=device)

    # step_cart_ / set_coords_ / backup_coords / restore_coords / _resolve_movable
    # are identical to BatchCalcABC's -> inherited (deleted here).

    # ---------------------------------------------------------------- forward
    def _build_data(self, coord):
        """Assemble the aimnet2calc input dict at the given (N,3) ``coord``."""
        return {
            "coord": coord,
            "numbers": self.numbers,
            "charge": self.charge_B,
            "mult": self.mult_B,
            "mol_idx": self.mol_idx,
        }

    def _reduce_energy(self, out) -> torch.Tensor:
        B = self._atoms_B
        # detach: the high-level API returns ``energy`` still attached to the
        # autograd graph when forces=True; the MAPLE contract expects plain
        # result tensors (NEB/PRFO detach downstream, but be self-contained).
        e = out["energy"].detach().to(self.dtype).reshape(-1)
        if e.numel() == B:
            return e
        if e.numel() == B + 1:        # padded sentinel structure
            return e[:B]
        raise RuntimeError(f"Unexpected energy shape {tuple(e.shape)} (B={B})")

    def _forward(self, coord: torch.Tensor, need_graph: bool = False):
        """ONE batched forward -> BatchCalcABC contract (E_eV (B,), F_eV (N,3),
        leaf|None) in NATIVE eV (the base converts once via MODEL_ENERGY_UNIT; do
        NOT scale by EV2HARTREE here).

        The wrapped ``aimnet2calc`` computes forces internally and returns them
        directly (no user-controlled position leaf), so ``leaf`` is always None and
        ``need_graph`` is ignored -- SUPPORTED_HESSIAN_MODES=('numerical',) so the
        base only ever calls this with need_graph=False (get_ef_gpu / generic FD
        Hessian). Force extraction is byte-identical to the pre-retrofit path
        (``out["forces"]`` detached, reshaped to (N,3)); the base then scatters via
        ``_cols`` and multiplies by EV2HARTREE (the retrofit centralizes the old
        per-class ``/EH2EV`` into the single ``_to_hartree``).
        """
        out = self.aim(self._build_data(coord), forces=True)
        E_eV = self._reduce_energy(out)
        F_atom_eV = out["forces"].detach().to(self.dtype).reshape(self.N_atoms, 3)
        return E_eV, F_atom_eV, None

    # ------------------------------------------------------- energy-only path
    def get_e_gpu(self):
        """Energy-only batched single point -> E (B,) Hartree (model-specific).

        Not part of BatchCalcABC (which is E+F / E+F+H only); kept for the
        energy-only callers (NEB/PRFO/BatchLBFGS line searches). Runs the wrapped
        forward with ``forces=False`` and converts eV->Hartree via the base's
        centralized EV2HARTREE (replaces the pre-retrofit ``/EH2EV``).
        """
        assert self._prepared, "call prepare() first"
        if self._atoms_B == 0:
            return torch.zeros((0,), dtype=self.dtype, device=self.device)
        out = self.aim(self._build_data(self.coord), forces=False)
        return self._reduce_energy(out) * EV2HARTREE
