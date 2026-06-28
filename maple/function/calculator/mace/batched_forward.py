# -*- coding: utf-8 -*-
"""
Batched multi-system MACE forward for MAPLE MD (TASK #10, primary deliverable).

ONE ML forward pass that evaluates ``N`` independent configurations at once
instead of ``N`` serial ``atoms.get_forces()`` calls. The natural consumer is
umbrella / replica / multiple-walker sampling: the K umbrella windows (or M
replicas) share an identical topology and differ only in coordinates, so their
per-step MLIP force evaluations are embarrassingly batchable into a single GPU
call. The bias (PLUMED RESTRAINT, etc.) is still applied per-window on the host
AFTER this returns the bare MLIP forces, so batching here is orthogonal to and
composes with the existing :mod:`..bias` layer.

WHY THE MACE DATA PIPELINE (not the existing batch calculators)
---------------------------------------------------------------
The pre-existing ``_mace_batch_calculator`` / ``_mace_autograd_batch_calculator``
build a *block-diagonal radius graph with zero shifts* -- i.e. NON-periodic, and
they target gas-phase organics (MACE-OFF). Umbrella sampling of condensed-phase
/ metal systems (e.g. Cu108 with MACE-MP-0) is PERIODIC: edges need
minimum-image shifts from ``atoms.cell`` / ``atoms.pbc``. This module instead
reuses MACE's OWN graph builder (``config_from_atoms`` + ``AtomicData.from_config``
+ ``torch_geometric.Batch``) -- exactly the path the upstream single-config
``MACECalculator.calculate`` takes -- so the batched per-config forces equal the
serial per-config forces bit-for-bit (the neighbour list, shifts and model call
are identical; only the batch dimension is added). This works for BOTH periodic
(MACE-MP-0) and gas-phase (MACE-OFF / maceomol) standard MACE models, which are
pure local MLIPs (per-graph ``scatter_sum`` energy pooling) -> each graph in the
batch is exactly isolated.

CONTRACT (driver-facing)
------------------------
    forces_batched(calc, atoms_list, *, units="Ha", return_energy=False)
        calc        : a MAPLE MACE backend (MACEMPCalculator / a bias wrapper
                      around one) OR a raw upstream ``mace.calculators.MACECalculator``.
        atoms_list  : list[ase.Atoms]  (any per-config atom counts; same model).
        units       : "Ha" (default, Hartree/Angstrom -- the MAPLE MD convention,
                      matches a MAPLE backend's ``get_forces()``) or "eV"
                      (eV/Angstrom -- raw MACE units, matches a raw MACECalculator).
        return_energy: also return per-config energies (same unit system).
        -> list[np.ndarray (n_i, 3)]   (forces), or (forces, energies) tuple.

Units: MACE returns energy in eV and forces in eV/Angstrom; EV2HARTREE =
1/27.211386245988 converts to the MAPLE-native Hartree / (Hartree/Angstrom).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from ase import Atoms

EV2HARTREE = 1.0 / 27.211386245988


# --------------------------------------------------------------------------- #
# Resolve a MAPLE wrapper down to the underlying upstream MACECalculator.      #
# --------------------------------------------------------------------------- #
def unwrap_to_mace(calc):
    """Return the upstream ``mace.calculators.MACECalculator`` inside ``calc``.

    Peels MAPLE layers:
      * bias wrappers (``PlumedCalculator`` / ``ColvarsCalculator``) expose the
        wrapped backend as ``.inner`` -> recurse;
      * the MAPLE periodic backend ``MACEMPCalculator`` holds the upstream calc
        as ``._mace``;
      * a raw upstream ``MACECalculator`` (has ``.models`` + ``._atoms_to_batch``)
        is returned as-is.

    Raises ``TypeError`` if no MACE calculator can be reached (the batched path
    is MACE-specific; other backends fall back to serial evaluation upstream).
    """
    seen = set()
    cur = calc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        # raw upstream MACECalculator: has the data pipeline we need.
        if hasattr(cur, "models") and hasattr(cur, "_atoms_to_batch") \
                and hasattr(cur, "z_table") and hasattr(cur, "r_max"):
            return cur
        # MAPLE periodic MACE backend.
        inner = getattr(cur, "_mace", None)
        if inner is not None:
            cur = inner
            continue
        # bias wrappers.
        inner = getattr(cur, "inner", None)
        if inner is not None:
            cur = inner
            continue
        break
    raise TypeError(
        "forces_batched: could not resolve an upstream mace.calculators."
        "MACECalculator from the given calculator. The batched forward path "
        "currently supports standard MACE backends (MACE-MP-0 / MACE-OFF / "
        "maceomol). Use serial evaluation for other backends."
    )


# --------------------------------------------------------------------------- #
# Core: build N graphs the same way MACE does, batch, one forward, split.      #
# --------------------------------------------------------------------------- #
def mace_forces_batched(
    mace_calc,
    atoms_list: Sequence[Atoms],
    *,
    return_energy: bool = False,
):
    """ONE batched MACE forward over ``atoms_list``. Native (eV, eV/Angstrom).

    ``mace_calc`` must be an upstream ``mace.calculators.MACECalculator`` (use
    :func:`unwrap_to_mace` first if you hold a MAPLE wrapper). Returns a list of
    ``(n_i, 3)`` force arrays (eV/Angstrom), one per input config, in the input
    order; with ``return_energy`` also a ``(N,)`` energy array (eV).

    Correctness: graphs are built with the model's own ``z_table`` / ``r_max`` /
    key specification and batched with ``torch_geometric.Batch.from_data_list``,
    so the neighbour list (incl. periodic shifts) and the model call are
    identical to the serial path -- only a batch axis is added. Single-model
    backends only (foundation MACE ships one model); ensembles would need
    per-model batching, which umbrella/replica use cases never require.
    """
    from mace import data as mace_data
    from mace.tools import torch_geometric, torch_tools

    B = len(atoms_list)
    if B == 0:
        return ([], np.zeros((0,))) if return_energy else []

    if len(getattr(mace_calc, "models", [])) != 1:
        raise NotImplementedError(
            "mace_forces_batched supports single-model MACE backends only "
            f"(got {len(mace_calc.models)} committee models)."
        )
    model = mace_calc.models[0]
    device = mace_calc.device

    # Mirror MACECalculator._atoms_to_batch key handling (charges array key).
    arrays_keys = dict(getattr(mace_calc, "arrays_keys", {}) or {})
    arrays_keys[mace_calc.charges_key] = "charges"
    keyspec = mace_data.KeySpecification(
        info_keys=getattr(mace_calc, "info_keys", {}),
        arrays_keys=arrays_keys,
    )

    data_list = []
    with torch_tools.default_dtype(mace_calc.default_dtype):
        for atoms in atoms_list:
            config = mace_data.config_from_atoms(
                atoms, key_specification=keyspec, head_name=mace_calc.head
            )
            graph = mace_data.AtomicData.from_config(
                config,
                z_table=mace_calc.z_table,
                cutoff=mace_calc.r_max,
                heads=mace_calc.available_heads,
            )
            data_list.append(graph)

    batch = torch_geometric.Batch.from_data_list(data_list).to(device)

    # Cast floating tensors to the model dtype (mirrors MACECalculator.calculate).
    model_dtype = next(model.parameters()).dtype
    batch_dict = batch.to_dict()
    for key, value in batch_dict.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            batch_dict[key] = value.to(dtype=model_dtype)

    compute_stress = getattr(mace_calc, "model_type", "MACE") in (
        "MACE", "EnergyDipoleMACE", "PolarMACE",
    )
    out = model(
        batch_dict,
        compute_stress=compute_stress,
        compute_force=True,
        training=False,
    )

    f_to_eV = mace_calc.energy_units_to_eV / mace_calc.length_units_to_A
    e_to_eV = mace_calc.energy_units_to_eV

    forces_all = out["forces"].detach().to(torch.float64).cpu().numpy() * f_to_eV
    ptr = batch.ptr.detach().cpu().tolist()
    forces_list = [forces_all[ptr[i]:ptr[i + 1]] for i in range(B)]

    if return_energy:
        energies = out["energy"].detach().to(torch.float64).cpu().numpy() * e_to_eV
        return forces_list, energies
    return forces_list


# --------------------------------------------------------------------------- #
# Driver-facing wrapper: unwrap + unit selection.                              #
# --------------------------------------------------------------------------- #
def forces_batched(
    calc,
    atoms_list: Sequence[Atoms],
    *,
    units: str = "Ha",
    return_energy: bool = False,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], np.ndarray]]:
    """Batched per-config forces for a list of systems in ONE MACE forward.

    See the module docstring for the full contract. ``units="Ha"`` (default)
    returns Hartree/Angstrom (MAPLE MD convention, == a MAPLE backend's
    ``get_forces()``); ``units="eV"`` returns eV/Angstrom (raw MACE / a raw
    ``MACECalculator.get_forces()``).
    """
    u = str(units).lower()
    if u in ("ha", "hartree", "au"):
        f_scale, e_scale = EV2HARTREE, EV2HARTREE
    elif u in ("ev",):
        f_scale, e_scale = 1.0, 1.0
    else:
        raise ValueError(f"units must be 'Ha' or 'eV', got {units!r}")

    mace_calc = unwrap_to_mace(calc)
    result = mace_forces_batched(mace_calc, atoms_list, return_energy=return_energy)
    if return_energy:
        forces_list, energies = result
        forces_list = [f * f_scale for f in forces_list]
        return forces_list, energies * e_scale
    return [f * f_scale for f in result]


if __name__ == "__main__":
    # Runnable self-test (CPU-friendly): batched forces over N perturbed copies
    # of a small PERIODIC cell MUST equal serial per-config forces. Skips clean
    # if mace / a model download is unavailable.
    import sys

    try:
        from mace.calculators import mace_mp
        from ase.build import bulk
    except Exception as exc:  # pragma: no cover
        print(f"SKIP self-test (mace/ase unavailable: {exc})")
        sys.exit(0)

    rng = np.random.RandomState(0)
    base = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 1, 1))  # 8 atoms, PBC
    atoms_list = []
    for _ in range(4):
        a = base.copy()
        a.positions += 0.1 * rng.randn(*a.positions.shape)
        atoms_list.append(a)

    try:
        calc = mace_mp(model="small", device="cpu", default_dtype="float64")
    except Exception as exc:  # pragma: no cover
        print(f"SKIP self-test (model load failed: {exc})")
        sys.exit(0)

    # serial reference (raw MACE units, eV/A)
    serial = []
    for a in atoms_list:
        a2 = a.copy(); a2.calc = calc
        serial.append(a2.get_forces().copy())

    bat = forces_batched(calc, atoms_list, units="eV")
    maxdiff = max(float(np.abs(b - s).max()) for b, s in zip(bat, serial))
    print(f"OK batched-forward self-test: N={len(atoms_list)} (8-atom periodic Cu) "
          f"max|F_batched - F_serial| = {maxdiff:.3e} eV/A "
          f"(= {maxdiff * EV2HARTREE:.3e} Ha/A)")
    assert maxdiff < 1e-6, f"batched != serial: {maxdiff}"
    print("OK isolation: per-config forces match serial within 1e-6.")
