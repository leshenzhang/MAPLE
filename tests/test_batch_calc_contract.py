# -*- coding: utf-8 -*-
"""In-tree contract guardrail for the batch-calculator layer (CALCULATOR_REVIEW §5).

Model-independent (CPU, no checkpoints): asserts every registered batch backend
declares the BatchCalcABC capability contract, and that the base plumbing
(prepare / (B,nmax_dof) pack-scatter / get_ef_gpu / FD + autograd Hessian /
padding / unit conversion) is correct via a trivial harmonic "model".

Runnable both as pytest and standalone:  python -m tests.test_batch_calc_contract
"""
import numpy as np
import torch

from maple.function.calculator.batch_calculator_base import (
    BatchCalcABC,
    register_batch_calculator,
    get_registered_batch_calculator,
    _BATCH_REGISTRY,
    _import_batch_backends,
    EV2HARTREE,
)

_HESSIAN_MODES = {"numerical", "autograd"}
_UNITS = {"eV", "hartree"}


def test_all_backends_import_and_register():
    """Every shipped backend module imports + self-registers (no missing dep)."""
    failed = _import_batch_backends()
    # optional-dep guard is allowed to skip a backend, but on the dev env (cxtorch)
    # all 7 must register; assert the registry is non-empty and report skips.
    assert _BATCH_REGISTRY, "no batch calculators registered"
    assert not failed, f"backend modules failed to import: {failed}"


def test_capability_contract_declared():
    """Each registered class declares the full BatchCalcABC capability surface."""
    _import_batch_backends()
    seen = set()
    for name, cls in _BATCH_REGISTRY.items():
        if cls in seen:
            continue
        seen.add(cls)
        assert get_registered_batch_calculator(name) is cls
        assert isinstance(cls.MODEL_NAMES, tuple) and cls.MODEL_NAMES, cls
        assert all(isinstance(n, str) for n in cls.MODEL_NAMES), cls
        assert cls.MODEL_ENERGY_UNIT in _UNITS, (cls, cls.MODEL_ENERGY_UNIT)
        assert isinstance(cls.MODEL_DTYPE, torch.dtype), cls
        assert isinstance(cls.SUPPORTS_PBC, bool), cls
        assert isinstance(cls.HAS_HVP, bool), cls
        assert isinstance(cls.SUPPORTS_COUPLING, bool), cls
        assert isinstance(cls.BATCHABLE, bool), cls
        assert isinstance(cls.SUPPORTED_HESSIAN_MODES, tuple), cls
        assert set(cls.SUPPORTED_HESSIAN_MODES) <= _HESSIAN_MODES, cls
        # the batch contract methods exist (inherited or overridden)
        for m in ("prepare", "get_ef_gpu", "get_efh_gpu", "_forward", "step_cart_"):
            assert callable(getattr(cls, m, None)), (cls, m)


_K = 1.7


@register_batch_calculator
class _HarmonicBatch(BatchCalcABC):
    """Trivial E = 0.5*K*sum(r^2) => F = -K*r, H = K*I. Exercises base plumbing."""
    MODEL_NAMES = ("_harmonic_contract_test",)
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES = ("numerical", "autograd")

    def _forward(self, coord, need_graph):
        from ase import Atoms  # noqa: F401 (kept import-light)
        leaf = coord.detach().to(self.MODEL_DTYPE).requires_grad_(need_graph)
        e_atoms = 0.5 * _K * (leaf ** 2).sum(dim=1)
        E = torch.zeros(self._atoms_B, dtype=leaf.dtype, device=leaf.device)
        E.index_add_(0, self.mol_idx, e_atoms)
        if need_graph:
            F_all = -torch.autograd.grad(e_atoms.sum(), leaf, create_graph=True)[0]
        else:
            F_all = -_K * leaf
        return E, F_all, (leaf if need_graph else None)


def _harmonic_mols():
    from ase import Atoms
    return [Atoms("H2O", positions=np.random.RandomState(0).randn(3, 3)),
            Atoms("H2", positions=np.random.RandomState(1).randn(2, 3))]


def test_base_plumbing_harmonic_cpu():
    """prepare/pack/scatter/get_ef_gpu + FD & autograd Hessian on a harmonic PES."""
    c = _HarmonicBatch(device="cpu", dtype=torch.float64)
    c.prepare(_harmonic_mols())
    E, F = c.get_ef_gpu()
    assert E.shape == (2,) and F.shape == (2, c.nmax_dof)
    _, _, H_fd, P = c.get_efh_gpu(mode="numerical")
    _, _, H_ag, _ = c.get_efh_gpu(mode="autograd")
    kHa = _K * EV2HARTREE                                  # H returned in Hartree
    diag0 = torch.diagonal(H_fd[0])[:9]                    # H2O = 9 real DOFs
    assert torch.allclose(diag0, torch.full_like(diag0, kHa), atol=1e-6), (diag0, kHa)
    assert torch.allclose(H_fd[0][:9, :9], H_ag[0][:9, :9], atol=1e-6)
    assert int(P[0]) == 0 and int(P[1]) == 1               # mol1 has 1 padding atom


def test_pbc_fail_fast():
    """A no-PBC backend rejects periodic atoms in prepare()."""
    from ase import Atoms
    at = Atoms("H2", positions=np.zeros((2, 3)), cell=[5, 5, 5], pbc=True)
    c = _HarmonicBatch(device="cpu", dtype=torch.float64)  # SUPPORTS_PBC=False default
    try:
        c.prepare([at])
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError on periodic atoms")


def test_partial_hessian_movable_mask():
    """PHVA path: get_efh_gpu(movable_masks=subset) matches the full Hessian's
    movable block, frozen atoms decouple (rows/cols = 0). FD + autograd."""
    from ase import Atoms
    c = _HarmonicBatch(device="cpu", dtype=torch.float64)
    c.prepare([Atoms("H2O", positions=np.random.RandomState(3).randn(3, 3))])  # 3 atoms
    _, _, H_full, _ = c.get_efh_gpu(mode="numerical")
    for mode in ("numerical", "autograd"):
        _, _, H_p, _ = c.get_efh_gpu(movable_masks=[[0, 1]], mode=mode)  # atoms 0,1 movable
        # movable block (DOFs 0..5) == the full Hessian's same block
        assert torch.allclose(H_p[0][:6, :6], H_full[0][:6, :6], atol=1e-6), (mode, H_p[0][:6, :6])
        # frozen atom 2 (DOFs 6,7,8): rows AND cols zero (constrained-PES block)
        assert H_p[0][6:9, :].abs().max() < 1e-9 and H_p[0][:, 6:9].abs().max() < 1e-9, mode


@register_batch_calculator
class _HarmonicPBC(_HarmonicBatch):
    """PBC-capable variant: SUPPORTS_PBC=True (aligns with ai-maple-md Phase-B)."""
    MODEL_NAMES = ("_harmonic_pbc_test",)
    SUPPORTS_PBC = True


def test_pbc_capable_accepts_and_homogeneous_gate():
    """A SUPPORTS_PBC=True backend accepts periodic atoms + enforces homogeneous pbc."""
    from ase import Atoms
    c = _HarmonicPBC(device="cpu", dtype=torch.float64)
    at_p = Atoms("H2", positions=np.zeros((2, 3)), cell=[5, 5, 5], pbc=True)
    c.prepare([at_p, at_p.copy()])                 # homogeneous periodic -> OK
    assert c._periodic is True
    E, F = c.get_ef_gpu()
    assert E.shape == (2,)
    at_m = Atoms("H2", positions=np.zeros((2, 3)))  # non-periodic
    try:
        c.prepare([at_p, at_m])                     # heterogeneous -> reject
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError on heterogeneous PBC")


if __name__ == "__main__":
    test_all_backends_import_and_register()
    test_capability_contract_declared()
    test_base_plumbing_harmonic_cpu()
    test_partial_hessian_movable_mask()
    test_pbc_fail_fast()
    test_pbc_capable_accepts_and_homogeneous_gate()
    n = len({id(c) for c in _BATCH_REGISTRY.values()})
    print(f"batch-calc contract tests PASS: {n} registered backends contract-compliant "
          f"+ base plumbing + PBC fail-fast")
