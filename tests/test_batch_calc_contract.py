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


def test_uma_hessian_plan_cache_key_covers_fd_step():
    """Regression guard for c3af78d (stale-Hessian-plan bug).

    ``UMABatchCalc`` caches the block-diagonal FD-Hessian plan across get_efh_gpu()
    calls. The plan BAKES IN the FD step (pert_val = s*delta, fac = +-1/(2*delta)),
    the chunk budget (h_max) and the FD discretization (central/forward). If the
    cache key were the movable mask alone, changing any of those on a live
    calculator would silently reuse the OLD plan -> a Hessian computed with the
    WRONG delta (no error, wrong numbers).

    Key-level test (CPU, no checkpoint): the key must change when _delta,
    _h_max_atoms or _fd_mode changes, and must be stable otherwise.
    """
    from maple.function.calculator.uma._uma_batch_calculator import UMABatchCalc

    # Stub instance: _movable_key only needs _atoms_B / _n_b (for _resolve_movable)
    # plus the three plan-defining knobs. No model, no GPU.
    c = object.__new__(UMABatchCalc)
    c._atoms_B = 2
    c._n_b = torch.tensor([3, 2], dtype=torch.int64)
    c._delta = 2e-3
    c._h_max_atoms = 4096
    c._fd_mode = "central"

    k0 = c._movable_key(None)
    assert c._movable_key(None) == k0, "key must be stable when nothing changes"

    c._delta = 1e-3
    k_delta = c._movable_key(None)
    assert k_delta != k0, ("FD step is NOT in the Hessian-plan cache key -> a delta "
                           "change would silently reuse the stale plan (c3af78d)")

    c._delta = 2e-3
    c._h_max_atoms = 2048
    k_hmax = c._movable_key(None)
    assert k_hmax != k0, "chunk budget (h_max) is not in the Hessian-plan cache key"

    c._h_max_atoms = 4096
    c._fd_mode = "forward"
    k_fd = c._movable_key(None)
    assert k_fd != k0, "FD discretization (central/forward) is not in the cache key"

    # movable mask still discriminates (the pre-c3af78d key's only component)
    c._fd_mode = "central"
    assert c._movable_key([[0, 1], [0]]) != c._movable_key(None)


# =========================================================================== #
# BUG-2 regression: _efh_fd must write each Hessian column ONLY to the molecules
# that own+move that DOF (owner mask), else FD noise leaks into frozen/padding.
# =========================================================================== #
class _HarmonicF64(_HarmonicBatch):
    """f64 harmonic stub (NOT registered -- used directly by the BUG-2 tests).

    BatchCalcABC.MODEL_DTYPE defaults to float32, so the plain _HarmonicBatch runs its
    forward in f32: its central-difference Hessian then carries ~1e-6 Ha/A^2 of pure
    rounding noise (f32 force error ~1e-7 eV/A amplified by 1/(2*delta) = 250). That is
    the same order as the leak BUG-2 is about, so the BUG-2 assertions must not be made
    against it. Running the stub in f64 removes the noise source entirely (rather than
    loosening a tolerance to hide it), making the movable block exact to ~1e-12.
    """
    MODEL_DTYPE = torch.float64


class _CrossTalkHarmonic(_HarmonicF64):
    """Harmonic + a deterministic BATCH-GLOBAL force term.

    Stands in for the floating-point reduction noise of a real batched MLIP forward.
    In a block-diagonal batch, perturbing molecule i must not change molecule j's
    forces -- mathematically. Numerically it does: Fp and Fm are two INDEPENDENT
    forwards, so a real backend's untouched molecules differ by ~1e-9, which the
    central-difference 1/(2*delta) then amplifies ~500x into ~1e-6 (measured
    3.29e-6 Ha/A^2 on aimnet2_decoupled).

    Reproducing that fp noise deterministically is impossible, so this stub makes the
    SAME structural leak explicit and large: every atom's force carries EPS*sum(coord),
    so displacing ANY atom by +/-delta shifts EVERY molecule's forces by 2*EPS*delta ->
    FD column entry -EPS for every molecule in the batch, owner or not. The buggy
    ``H[:, :, k] = col_pad`` wrote that into the non-owners' frozen/padding columns;
    the owner-masked write leaves them at exactly 0.
    """
    MODEL_NAMES = ("_crosstalk_harmonic_contract_test",)
    EPS = 1e-3   # >> any tolerance: the assertions below are EXACT-zero assertions

    def _forward(self, coord, need_graph):
        E, F_all, leaf = super()._forward(coord, need_graph)
        src = leaf if (need_graph and leaf is not None) else coord
        return E, F_all + self.EPS * src.sum(), leaf


def _hetero_batch():
    """Heterogeneous batch: different atom counts AND different movable sets.

    mols    = [H2O (3 atoms), H2 (2 atoms)]      -> nmax_dof = 9, mol1 has 1 padding atom
    movable = [[0, 1, 2],     [0]]
    So for mol1 (H2):
      * local atom 1 exists but is FROZEN  -> columns k=3,4,5 are frozen columns
      * local atom 2 does not exist        -> columns k=6,7,8 are padding columns
    Both are swept (mol0 owns+moves atoms 1 and 2), so both are written by _efh_fd --
    exactly the two leak channels the A5 controls isolated.
    """
    from ase import Atoms
    mols = [Atoms("H2O", positions=np.random.RandomState(7).randn(3, 3)),
            Atoms("H2", positions=np.random.RandomState(8).randn(2, 3))]
    return mols, [[0, 1, 2], [0]]


def test_fd_hessian_columns_owner_masked_frozen_exactly_zero():
    """BUG-2: frozen + padding Hessian columns of a NON-OWNER molecule are EXACTLY 0.

    The PHVA contract (and the paper's "the remaining atoms carry infinite masses")
    requires the frozen block to decouple exactly, not approximately. Pre-fix this
    asserted-zero region held _CrossTalkHarmonic.EPS (and ~3.29e-6 Ha/A^2 with a real
    backend); post-fix it is bitwise 0.
    """
    mols, mov = _hetero_batch()
    c = _CrossTalkHarmonic(device="cpu", dtype=torch.float64)
    c.prepare(mols)
    _, _, H, P = c.get_efh_gpu(movable_masks=mov, mode="numerical")
    assert int(P[1]) == 1, "mol1 (H2) should carry exactly 1 padding atom"

    # mol1: only local atom 0 is movable -> everything outside its (3x3) block is
    # frozen (atom 1) or padding (atom 2) and MUST be exactly zero.
    frozen_and_padding_rows = H[1][3:, :].abs().max()
    frozen_and_padding_cols = H[1][:, 3:].abs().max()
    assert float(frozen_and_padding_rows) == 0.0, (
        f"FD noise leaked into mol1's frozen/padding ROWS: {float(frozen_and_padding_rows):.3e} "
        "(_efh_fd wrote column k for a molecule that does not own+move that DOF)")
    assert float(frozen_and_padding_cols) == 0.0, (
        f"FD noise leaked into mol1's frozen/padding COLS: {float(frozen_and_padding_cols):.3e}")

    # ...and the leak is real in this stub: the owner (mol0) DOES see the cross-talk,
    # so the test would catch a fix that simply zeroed the whole Hessian.
    assert float(H[0].abs().max()) > 0.0


def test_fd_hessian_movable_block_unchanged_by_owner_mask():
    """BUG-2 zero-regression: the owner mask must not perturb the MOVABLE block.

    Non-owner columns were noise and become 0; owner columns are untouched. On the
    plain harmonic PES the movable block is analytically K*I, so this pins the
    numbers the owner-mask fix must NOT change.
    """
    mols, mov = _hetero_batch()
    c = _HarmonicF64(device="cpu", dtype=torch.float64)
    c.prepare(mols)
    _, _, H, _ = c.get_efh_gpu(movable_masks=mov, mode="numerical")
    kHa = _K * EV2HARTREE
    # mol0: all 3 atoms movable -> full 9x9 block = K*I
    assert torch.allclose(H[0][:9, :9], kHa * torch.eye(9, dtype=torch.float64), atol=1e-9)
    # mol1: only atom 0 movable -> its 3x3 movable block = K*I, rest exactly 0
    assert torch.allclose(H[1][:3, :3], kHa * torch.eye(3, dtype=torch.float64), atol=1e-9)
    assert float(H[1][3:, :].abs().max()) == 0.0 and float(H[1][:, 3:].abs().max()) == 0.0


# =========================================================================== #
# BUG-1 regression: the batched path must not silently drop implicit solvent.
# =========================================================================== #
def test_batched_implicit_solvent_fails_fast():
    """BUG-1: '#solv(method=gbsa, implicit=water)' + a gas-phase batch calc -> raise.

    Pre-fix, resolve_batched_calc ignored params['solv'] entirely: a batched
    OPT/TS/IRC/SCAN ran in the GAS PHASE (614.34 Ha off the solvated oracle) AND
    bypassed the single-structure path's solvent+derivatives NotImplementedError.
    """
    from maple.function.dispatcher.dispatcher import resolve_batched_calc

    gas_calc = _HarmonicBatch(device="cpu", dtype=torch.float64)
    solvated = {"solv": {"method": "gbsa", "implicit": "water", "experimental": True},
                "batched_calc": gas_calc}

    # THE behavioral assertion (this is what fails on the buggy tree -- keep it first, so
    # the test detects the BUG rather than merely the absence of the new helper module).
    try:
        resolve_batched_calc(solvated, [], attached_calc=None)
    except NotImplementedError as exc:
        assert "gas" in str(exc).lower() or "solvat" in str(exc).lower(), exc
    else:
        raise AssertionError(
            "batched path accepted an implicit-solvent job -> it would silently return "
            "gas-phase numbers (BUG-1)")

    # no #solv -> unchanged (zero regression on every gas-phase batched job)
    assert resolve_batched_calc({"batched_calc": gas_calc}, []) is gas_calc

    from maple.function.dispatcher._batch_calc_utils import (
        implicit_solvent_requested, reject_batched_implicit_solvent)
    assert implicit_solvent_requested(solvated) is True
    assert gas_calc.SUPPORTS_IMPLICIT_SOLVENT is False
    # explicit solvation adds real atoms; it is NOT the implicit path and is not gated
    assert implicit_solvent_requested({"solv": {"explicit": "water"}}) is False
    # 'none' sentinels are not a request
    assert implicit_solvent_requested({"solv": {"method": "none", "implicit": "none"}}) is False

    # a future solvent-capable batch backend passes the gate (capability, not hard-code)
    class _SolvatedBatch(_HarmonicBatch):
        SUPPORTS_IMPLICIT_SOLVENT = True
    ok = _SolvatedBatch(device="cpu", dtype=torch.float64)
    assert reject_batched_implicit_solvent(solvated, ok) is ok

    # calc=None means "fall back to SERIAL", which applies the solvent correctly (and
    # rejects solvent+derivatives). It must pass through ungated -- gating it would break
    # the serial fallback rather than protect it (frequency._resolve_batched_calc can
    # legitimately return None).
    assert reject_batched_implicit_solvent(solvated, None) is None


# =========================================================================== #
# BUG-3 regression: a CUDA-arch/OOM failure must NOT be reported as "trace-locked".
# =========================================================================== #
def test_probe_batch_native_env_error_propagates_not_trace_locked():
    """BUG-3: _probe_batch_native's bare `except Exception` mislabeled hardware faults.

    On a V100 (sm_70) with an sm_80-only torch build, model(...) raises "no kernel image
    is available for execution on the device". The bare except turned that into
    ``_batch_native = False`` and prepare() then blamed the MODEL ("traced single-graph").
    Environment errors must propagate; only genuine model/trace errors set the flag.
    """
    from maple.function.calculator.mace._mace_batch_calculator import MACEBatchCalc

    cuda_arch = RuntimeError(
        "CUDA error: no kernel image is available for execution on the device")
    trace_lock = RuntimeError(
        "The size of tensor a (2) must match the size of tensor b (1) at dimension 0")

    def _probe_with(exc):
        """Drive the real _probe_batch_native with a model that raises `exc`."""
        c = object.__new__(MACEBatchCalc)          # no checkpoint, no GPU
        c._batch_native = None
        c._batch_probe_error = None
        c._two_atom_graph = lambda b: (None,) * 6
        def _boom(*a, **kw):
            raise exc
        c.model = _boom
        return c

    # (a) environment / hardware -> propagates as ITSELF, verdict left unresolved
    c = _probe_with(cuda_arch)
    try:
        c._probe_batch_native()
    except RuntimeError as exc:
        assert "no kernel image" in str(exc)
    else:
        raise AssertionError("CUDA arch failure was swallowed and mislabeled (BUG-3)")
    assert c._batch_native is None, (
        "a hardware failure must NOT be recorded as 'model is trace-locked'")

    # (b) genuine model/trace failure -> classified trace-locked, reason retained
    c = _probe_with(trace_lock)
    c._probe_batch_native()                        # must not raise
    assert c._batch_native is False
    assert "must match the size" in (c._batch_probe_error or "")

    # the shared classifier itself (checked last: the assertions above are the ones that
    # must fail on the buggy tree, and they do so behaviorally, not by ImportError)
    from maple.function.calculator.batch_calculator_base import is_environment_error
    assert is_environment_error(cuda_arch) is True
    assert is_environment_error(trace_lock) is False
    assert is_environment_error(RuntimeError("CUDA out of memory. Tried to allocate...")) is True


# =========================================================================== #
# BUG-4 regression: get_efh_gpu overrides must honor the base mode/delta/chunk_size.
# =========================================================================== #
def test_get_efh_gpu_signature_conformance_all_backends():
    """BUG-4 (Liskov): every backend accepts the documented base signature.

    MACEBatchCalc / MACEPolBatchCalc (and ANI / MACE-autograd / UMA) had narrowed the
    override to ``get_efh_gpu(self, movable_masks=None)``, so ``get_efh_gpu(mode=...)``
    -- the documented dispatch of BatchCalcABC -- raised TypeError.
    """
    import inspect
    _import_batch_backends()
    required = {"movable_masks", "mode", "delta", "chunk_size"}
    for cls in {id(c): c for c in _BATCH_REGISTRY.values()}.values():
        params = inspect.signature(cls.get_efh_gpu).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue                                # **kw absorbs the contract
        missing = required - set(params)
        assert not missing, (
            f"{cls.__name__}.get_efh_gpu drops {sorted(missing)} from the BatchCalcABC "
            "contract -> get_efh_gpu(mode=...) raises TypeError (BUG-4)")


def test_hessian_helper_overrides_accept_the_kwargs_get_efh_gpu_passes():
    """The Hessian bodies must accept what get_efh_gpu hands them -- by KEYWORD.

    Two live hazards this pins down:

    1. A backend whose ``_efh_analytic`` lacks ``chunk_size`` will raise TypeError if
       get_efh_gpu forwards it -- and because the analytic call sits inside
       ``except Exception: self._hess_mode = 'fd'``, that TypeError is SWALLOWED and the
       Hessian silently degrades from analytic to finite-difference. That is a silent
       numerical change (9.06e-07 Ha/A^2 on toy_maceomol, 3.96e-04 on macepols), not a
       crash. It happened during this very bugfix round and was caught only by the
       pre/post zero-regression gate. MACE / MACE-POL therefore reject chunk_size in
       get_efh_gpu instead of forwarding it.

    2. The ``_efh_fd`` overrides declare (delta, movable_masks) -- the REVERSE positional
       order of the base's (movable_masks, delta). Every caller must therefore pass these
       BY KEYWORD; a positional call silently swaps the FD step with the movable mask.
    """
    import inspect
    _import_batch_backends()
    for cls in {id(c): c for c in _BATCH_REGISTRY.values()}.values():
        fd = inspect.signature(cls._efh_fd).parameters
        assert {"movable_masks", "delta"} <= set(fd), (
            f"{cls.__name__}._efh_fd must accept movable_masks + delta by keyword")

        ag = inspect.signature(cls._efh_analytic).parameters
        assert "movable_masks" in ag, f"{cls.__name__}._efh_analytic needs movable_masks"

        # If the analytic body has no chunk budget, get_efh_gpu must NOT forward one.
        if "chunk_size" not in ag and not any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in ag.values()):
            c = object.__new__(cls)
            c._atoms_B = 1
            c._hess_mode = "fd"
            c.coupling_mode = "approx"          # macepol-only knob; harmless elsewhere
            c._efh_fd = lambda **kw: "FD"
            c._efh_analytic = lambda *a, **kw: "AG"
            try:
                cls.get_efh_gpu(c, chunk_size=8)
            except ValueError:
                pass                             # rejected loudly: correct
            except TypeError as exc:
                raise AssertionError(
                    f"{cls.__name__}: get_efh_gpu forwards chunk_size to an _efh_analytic "
                    f"that cannot take it ({exc}); inside the analytic try/except this "
                    "silently downgrades the Hessian to FD") from None
            else:
                raise AssertionError(
                    f"{cls.__name__}: get_efh_gpu accepted chunk_size although its "
                    "_efh_analytic has no chunk budget -> the knob is silently ignored")


def test_mace_get_efh_gpu_forwards_mode_and_delta():
    """BUG-4: mode/delta/chunk_size are FORWARDED, not accepted-and-ignored.

    Accepting the kwargs but ignoring them would silently return an analytic Hessian to
    a caller who asked for numerical (or the default delta to one who asked for 1e-4) --
    the same silent-divergence class as the bugs above. Driven on bare instances (no
    checkpoint) with the two Hessian bodies stubbed out.
    """
    from maple.function.calculator.mace._mace_batch_calculator import MACEBatchCalc
    from maple.function.calculator.mace._macepol_batch_calculator import MACEPolBatchCalc

    for cls, extra in ((MACEBatchCalc, {}),
                       (MACEPolBatchCalc, {"coupling_mode": "approx"})):
        c = object.__new__(cls)
        c._atoms_B = 1
        c._hess_mode = "fd"
        for k, v in extra.items():
            setattr(c, k, v)
        seen = {}
        c._efh_fd = lambda movable_masks=None, delta=None: seen.update(
            path="fd", movable_masks=movable_masks, delta=delta) or "FD"
        c._efh_analytic = lambda movable_masks=None, chunk_size=None: seen.update(
            path="analytic", movable_masks=movable_masks, chunk_size=chunk_size) or "AG"

        # explicit numerical + custom delta -> FD body, delta threaded through
        assert cls.get_efh_gpu(c, movable_masks=[[0]], mode="numerical", delta=7e-4) == "FD"
        assert seen == {"path": "fd", "movable_masks": [[0]], "delta": 7e-4}, (cls, seen)

        # explicit autograd -> analytic body (no silent downgrade to FD)
        seen.clear()
        assert cls.get_efh_gpu(c, mode="autograd") == "AG"
        assert seen["path"] == "analytic", (cls, seen)

        # default (mode=None) still routes by the internal knob -> unchanged behavior
        seen.clear()
        assert cls.get_efh_gpu(c) == "FD"
        assert seen["path"] == "fd" and seen["delta"] == 2e-3, (cls, seen)

        # chunk_size: these two backends' analytic Hessian is a hand-written seeded
        # double-backward with NO vmap chunking, so _efh_analytic does not accept it.
        # It must be REJECTED LOUDLY. Passing it down would raise TypeError inside the
        # analytic try/except and silently downgrade the Hessian to FD -- a real
        # regression (9.06e-07 Ha/A^2 on toy_maceomol, 3.96e-04 on macepols) that the
        # pre/post zero-regression gate caught during this bugfix round.
        try:
            cls.get_efh_gpu(c, chunk_size=8)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"{cls.__name__}: chunk_size silently accepted -> risks a silent "
                "analytic->FD downgrade")

        # a typo must not silently pick a path
        try:
            cls.get_efh_gpu(c, mode="analytical")
        except ValueError:
            pass
        else:
            raise AssertionError(f"{cls.__name__}: unknown mode silently accepted")


if __name__ == "__main__":
    test_all_backends_import_and_register()
    test_capability_contract_declared()
    test_base_plumbing_harmonic_cpu()
    test_partial_hessian_movable_mask()
    test_pbc_fail_fast()
    test_pbc_capable_accepts_and_homogeneous_gate()
    test_uma_hessian_plan_cache_key_covers_fd_step()
    test_fd_hessian_columns_owner_masked_frozen_exactly_zero()
    test_fd_hessian_movable_block_unchanged_by_owner_mask()
    test_batched_implicit_solvent_fails_fast()
    test_probe_batch_native_env_error_propagates_not_trace_locked()
    test_get_efh_gpu_signature_conformance_all_backends()
    test_hessian_helper_overrides_accept_the_kwargs_get_efh_gpu_passes()
    test_mace_get_efh_gpu_forwards_mode_and_delta()
    n = len({id(c) for c in _BATCH_REGISTRY.values()})
    print(f"batch-calc contract tests PASS: {n} registered backends contract-compliant "
          f"+ base plumbing + PBC fail-fast + UMA Hessian-plan cache key "
          f"+ BUG-1 solvent gate + BUG-2 owner-masked FD Hessian columns "
          f"+ BUG-3 env-vs-trace probe triage + BUG-4 get_efh_gpu signature/forwarding")
