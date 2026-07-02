"""
Batched NPT (isothermal-isobaric) molecular dynamics for MAPLE -- Phase-2C.

The last PBC gap: batch axis = replica, ONE batched MLIP+stress forward
(``calc.get_efs_gpu()`` over all B replicas) drives B independent periodic systems
per force evaluation, each with its OWN box + barostat. Single-system NPT is the
DEGENERATE B=1 case of this same kernel.

Authoritative physics (B's ``ensemble/npt.py``) is preserved EXACTLY and, where it
is force-free, DELEGATED per replica to B's authoritative classes rather than
re-derived:
  * Integration = the same V-rescale "middle" (split Velocity-Verlet) scheme as the
    single-system NPT (Swope 1982): B(dt/2) A(dt/2) O A(dt/2) [force] B(dt/2), with
    the ONE expensive force+stress evaluation batched over all replicas. The
    constrained path uses the same monolithic VV+RATTLE (END-thermostat) scheme the
    single-system constrained NPT uses.
  * Thermostat (V-rescale, Bussi 2007 A7) is delegated per replica to
    ``thermostat/vrescale.py`` -- force-free, so looping the small B axis costs
    nothing and guarantees bit-for-bit single<->batched parity of the thermostat math.
  * Barostat (Berendsen / C-rescale, Bernetti & Bussi 2020) is delegated per replica
    to ``barostat/berendsen.py`` / ``barostat/crescale.py`` operating on a per-replica
    ASE ``Atoms`` whose calculator returns the batched forward's already-computed
    per-replica stress (eV/Ang^3). The barostat's ``atoms.set_cell(scale_atoms=True)``
    (ASE fp path) rescales that replica's box + positions; the rescaled geometry is
    pushed back into the batched calc via ``set_coords_`` / ``set_cells_``.
  * Pressure = the same ``compute_instantaneous_pressure`` (virial from the stress
    tensor + kinetic term), the same GROMACS-grompp box guard, the same runtime
    COM/angular projection -- all per replica.

★CRITICAL (memory feedback_npt_barostat_constraint_reprojection): after the barostat
rescales positions by mu it STRETCHES every constrained bond by mu; the constraint set
is RE-PROJECTED (``sync_cell`` -> RATTLE ``project_positions`` -> ``project_velocities``)
per replica AFTER every barostat rescale, exactly as the single-system NPT does. Without
this a constrained condensed-phase NPT run silently violates its bonds and blows up.

PARITY: with the SAME force+stress engine (an ASE bridge wrapping THIS batched calc,
B=1) and the SAME seed, B=1 through this kernel reproduces the single-system ``NPT``
to fp64 machine precision -- velocities / potential energy / cell (volume) are
bit-identical; positions are identical up to the cosmetic periodic wrap the
single-system integrator applies (min-image-equivalent through the isotropic barostat).
See ``_test_npt_batched.py``.

BACKEND gate: requires a PBC-capable, batch-ISOLATED calculator that returns a real
configurational stress (``SUPPORTS_PBC`` and "stress" in ``implemented_properties`` --
MaceOffBatchCalc Phase-2C). A gas-phase / stress-less batch calc is rejected, exactly
as the single-system NPT refuses pressure coupling without a virial.

ponytail: thermostat = V-rescale only (the recommended NPT production thermostat);
Langevin-NPT (LF-Middle carried-velocity barostat coupling) is a DELIBERATE ceiling --
run those on the single-system NPT. GaMD/SMD/PLUMED/Colvars/posres likewise.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from ...jobABC import JobABC
from maple.function.timer import timer
from maple.function.utility import Molecules

from ..thermostat.vrescale import VRescaleThermostat
from ..barostat.berendsen import BerendsenBarostat
from ..barostat.crescale import CRescaleBarostat
from ..constraints import build_constraint_manager, maybe_repartition_masses
from ..anneal import make_anneal_fn
from ..box_guard import check_box_size
from ..utils import (
    AMU_TO_AU,
    BOHR_TO_ANGSTROM,
    FS_TO_AU,
    KELVIN_TO_HARTREE,
    HA_PER_ANG_TO_AU,
    apply_runtime_motion_projection,
    compute_instantaneous_pressure,
    initialize_velocities,
)

# amu / Ang^3 -> g/cm^3 : m[amu]*u[g] / (V[Ang^3]*1e-24 cm^3) with u=1.66053906660e-24 g
_AMU_PER_ANG3_TO_G_CM3 = 1.66053906660


class _InjectedResultsCalc(Calculator):
    """Minimal ASE calculator that returns pre-injected energy/forces/stress (the
    batched forward already computed them). Lets the per-replica AUTHORITATIVE barostat
    + ``compute_instantaneous_pressure`` run UNCHANGED on a per-replica ``Atoms`` proxy:
    they call ``atoms.get_stress()`` and get exactly the batched calc's stress for that
    replica (eV/Ang^3 Voigt), so the barostat/pressure math is bit-identical to the
    single-system path. ``calculate`` always returns the stored results (independent of
    ASE system_changes), so a set_positions / set_cell before the barostat never
    triggers a real recompute."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]
    SUPPORTS_PBC = True

    def __init__(self, natoms):
        super().__init__()
        self._store = {"energy": 0.0, "free_energy": 0.0,
                       "forces": np.zeros((natoms, 3)), "stress": np.zeros(6)}

    def set_results(self, energy=None, stress=None, forces=None):
        if energy is not None:
            self._store["energy"] = float(energy)
            self._store["free_energy"] = float(energy)
        if stress is not None:
            self._store["stress"] = np.asarray(stress, dtype=float).reshape(6)
        if forces is not None:
            self._store["forces"] = np.asarray(forces, dtype=float)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties or ["energy"], system_changes)
        self.results = dict(self._store)


@dataclass
class BatchedNPTParams:
    """Parameters for batched NPT. Honored fields mirror the single-system
    ``NPTParams`` so a B=1 replica reproduces a single-structure NPT run.

    Trailing ceiling fields (gamd/smd/plumed/colvars/posres) are DECLARED so
    ``_init_params`` keeps the keys (B-51) but are rejected in the batched kernel."""
    timestep:        float = 0.5          # fs
    steps:           int   = 500          # MD steps
    temperature:     float = 300.0        # K
    pressure:        float = 1.0          # bar
    thermostat:      str   = "v-rescale"  # only 'v-rescale' (langevin-NPT is a ceiling)
    barostat:        str   = "c-rescale"  # 'berendsen' | 'c-rescale'
    tau_t:           float = 200.0        # fs  (v-rescale coupling)
    tau_p:           float = 2000.0       # fs  (barostat coupling)
    compressibility: float = 4.5e-5       # 1/bar
    anneal:          str   = ""           # thermostat T schedule (K); "" = constant
    traj_every:      int   = 50           # steps between recorded frames
    log_every:       int   = 50           # steps between main-log lines
    remove_com_every: int  = 100          # runtime per-replica COM removal cadence
    remove_angular_every: int = 0         # (ignored under PBC; parity with single NPT)
    remove_com:      bool  = True         # initialization-only COM removal
    init_velocities: bool  = True
    verbose:         int   = 1
    random_seed:     Optional[int] = None
    box_check:       str   = "strict"
    # constraints (RATTLE): supported, with the ★CRITICAL post-barostat reprojection.
    constraints:          str = "none"    # none|h-bonds|all-bonds|h-angles
    constraint_algorithm: str = "lincs"
    # HMR (mass-only; supported).
    hmr:           str = ""
    hmr_factor:    Optional[float] = None
    hmr_bond_mult: float = 1.2
    # ponytail: declared-but-ceiling in the batched kernel (use single-system NPT).
    gamd:   str = ""
    smd:    str = ""
    plumed: str = ""
    colvars: str = ""
    posres: str = ""


class BatchedNPT(JobABC):
    """Batched isothermal-isobaric (NPT) MD over B replicas, one batched
    force+stress forward per force evaluation, each replica with its own box."""

    _THERMOSTAT_CHOICES = {"v-rescale"}
    _BAROSTAT_CHOICES = {"berendsen", "c-rescale"}
    _COUPLED_CALC_NAMES = {"AIMNet2BatchCalc", "MACEPolBatchCalc"}

    def __init__(self, output: str,
                 systems: Union[Molecules, List[Atoms]],
                 calc=None,
                 paras: Optional[dict] = None):
        super().__init__(output)

        if isinstance(systems, Molecules):
            atoms_list = list(systems.multiatoms)
            calc = calc if calc is not None else systems.calc
        else:
            atoms_list = list(systems)
            if calc is None and atoms_list:
                calc = getattr(atoms_list[0], "calc", None)
        if not atoms_list:
            raise ValueError("BatchedNPT requires at least one replica/system.")
        if calc is None:
            raise ValueError("BatchedNPT requires a batched calculator (Molecules.calc, "
                             "atoms[0].calc, or the calc= argument).")
        for m in ("prepare", "get_ef_gpu", "get_efs_gpu", "step_cart_", "set_coords_",
                  "set_cells_", "rescale_isotropic_"):
            if not callable(getattr(calc, m, None)):
                raise TypeError(f"BatchedNPT needs a batch calculator exposing {m}(). "
                                "Use MaceOffBatchCalc (Phase-2C) or the single-system NPT.")

        self.atoms_list = atoms_list
        self.calc = calc
        self.B = len(atoms_list)

        self._assert_batch_isolated(calc, self.B)
        self.params = self._init_params(
            BatchedNPTParams, paras, ("md", "MD", "npt", "NPT", "batched", "batchnpt"))

        if self.params.thermostat not in self._THERMOSTAT_CHOICES:
            raise NotImplementedError(
                f"BatchedNPT thermostat='{self.params.thermostat}' unsupported; only "
                f"'v-rescale' (the recommended NPT thermostat). Langevin-NPT is a "
                f"deliberate ceiling -> use the single-system NPT.")
        if self.params.barostat not in self._BAROSTAT_CHOICES:
            raise ValueError(f"Unknown barostat '{self.params.barostat}'. "
                             f"Choose from: {self._BAROSTAT_CHOICES}")

        self._require_pbc_and_stress(calc, atoms_list)
        self._reject_ceiling_features()

        # HMR (mass-only) BEFORE the mass buffer / thermostats are built.
        for at in self.atoms_list:
            maybe_repartition_masses(at, self.params, log=False)

        import torch
        self._torch = torch
        dev = getattr(calc, "device", None)
        self.device = dev if dev is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(calc, "dtype", torch.float64)

    # --------------------------------------------------------------- gates
    @classmethod
    def _assert_batch_isolated(cls, calc, B):
        if B <= 1:
            return
        name = type(calc).__name__
        flag = getattr(calc, "batch_isolated", None)
        coupled = (flag is False) or (name in cls._COUPLED_CALC_NAMES) or (
            "AIMNet2" in name and "Decoupled" not in name)
        if coupled:
            raise ValueError(
                f"BatchedNPT B={B}>1 requires a batch-ISOLATED calculator; '{name}' "
                f"couples replicas across the co-batched graph. Use a local/decoupled "
                f"batch calc or the single-system NPT.")

    def _require_pbc_and_stress(self, calc, atoms_list):
        """NPT is meaningless without a periodic cell AND a real configurational virial
        (mirrors the single-system NPT capability gate). Every replica must be periodic."""
        pbc_flags = [bool(np.any(np.asarray(at.pbc))) for at in atoms_list]
        if not all(pbc_flags):
            raise ValueError(
                "BatchedNPT requires a periodic cell for EVERY replica (atoms.pbc True). "
                "Use batched NVE/NVT for non-periodic systems.")
        supports_pbc = bool(getattr(calc, "SUPPORTS_PBC", False))
        impl = tuple(getattr(calc, "implemented_properties", ()) or ())
        if not (supports_pbc and "stress" in impl):
            raise ValueError(
                "BatchedNPT requires a calculator that supports periodic boundaries AND "
                "returns a stress tensor (real configurational virial). Calculator "
                f"'{type(calc).__name__}' reports SUPPORTS_PBC={supports_pbc}, "
                f"implemented_properties={impl}. Use MaceOffBatchCalc (Phase-2C).")
        box_mode = getattr(self.params, "box_check", "strict")
        for b, at in enumerate(atoms_list):
            check_box_size(at, calc=calc, mode=box_mode,
                           context=f"BatchedNPT preflight replica {b}")

    def _reject_ceiling_features(self):
        p = self.params
        bad = []
        for nm in ("gamd", "smd", "plumed", "colvars", "posres"):
            val = getattr(p, nm, "")
            if str(val or "").strip().lower() not in ("", "off", "none", "no", "false", "0"):
                bad.append(f"{nm}={val}")
        if bad:
            raise NotImplementedError(
                "ponytail: the batched NPT kernel does not implement "
                f"{', '.join(bad)} (deliberate ceiling). Run those on the single-system NPT.")

    # ====================================================================== run
    def run(self):
        with timer(f"Batched MD (NPT, B={self.B})"):
            self._log_parameters()
            self._prepare_buffers()
            self._run_npt()
            self._finalize()
        return self

    # ------------------------------------------------------------ setup buffers
    def _prepare_buffers(self):
        torch = self._torch
        B = self.B
        self.calc.prepare(self.atoms_list, fixed_nmax=None)
        self.nmax_dof = int(self.calc.nmax_dof)
        nmax_a = self.nmax_dof // 3
        self._ptr = np.concatenate([[0], np.cumsum([len(a) for a in self.atoms_list])]).astype(int)

        self.n_b = np.array([len(at) for at in self.atoms_list], dtype=int)
        # PBC + runtime COM removal -> 3N-3 active DOF (angular ignored under PBC),
        # matching the single-system NPT DOF policy.
        self.n_dof = np.maximum(3 * self.n_b - 3, 1)

        dev, dt = self.device, self.dtype
        mass = torch.ones((B, self.nmax_dof), dtype=dt, device=dev)
        mask = torch.zeros((B, self.nmax_dof), dtype=dt, device=dev)
        for b, at in enumerate(self.atoms_list):
            m_au = torch.tensor(at.get_masses(), dtype=dt, device=dev) * AMU_TO_AU
            n = m_au.shape[0]
            mass[b, :3 * n] = m_au.repeat_interleave(3)
            mask[b, :3 * n] = 1.0
        self.mass = mass
        self.mask = mask
        self._mass_amu = [np.asarray(at.get_masses(), float) for at in self.atoms_list]

        seed = self.params.random_seed
        # one RNG per replica, SHARED by that replica's velocity init + thermostat +
        # barostat -> the b=0 draw order (init, then per step: thermostat then barostat)
        # matches the single-system NPT's single shared rng, so B=1 is bit-identical.
        self._rngs = [np.random.default_rng(seed + b) if seed is not None
                      else np.random.default_rng() for b in range(B)]

        # velocities (B, nmax_dof) a.u.
        v = torch.zeros((B, self.nmax_dof), dtype=dt, device=dev)
        if self.params.init_velocities:
            for b, at in enumerate(self.atoms_list):
                vb = initialize_velocities(
                    atoms=at, temperature=self.params.temperature,
                    remove_com=self.params.remove_com, remove_rotation=False,
                    remove_angular=False, target_n_dof=int(self.n_dof[b]),
                    rng=self._rngs[b])
                v[b, :vb.size] = torch.tensor(vb.reshape(-1), dtype=dt, device=dev)
        else:
            for b, at in enumerate(self.atoms_list):
                if "velocities" not in at.arrays:
                    raise ValueError(f"init_velocities=False but replica {b} has no velocities")
                vb = np.asarray(at.arrays["velocities"]).reshape(-1)
                v[b, :vb.size] = torch.tensor(vb, dtype=dt, device=dev)
        self.v = v

        # per-replica ASE proxy (holds current pos+cell; injected-stress calc) that the
        # AUTHORITATIVE thermostat / barostat / pressure / constraint code runs on.
        self._atoms_b = []
        self._thermostats = []
        self._barostats = []
        self._constraints = []
        for b, at in enumerate(self.atoms_list):
            proxy = at.copy()
            proxy.calc = _InjectedResultsCalc(len(at))
            self._atoms_b.append(proxy)
            self._thermostats.append(VRescaleThermostat(
                proxy, temperature=self.params.temperature, tau_t=self.params.tau_t,
                timestep=self.params.timestep, rng=self._rngs[b], n_dof=int(self.n_dof[b])))
            if self.params.barostat == "berendsen":
                self._barostats.append(BerendsenBarostat(
                    proxy, pressure=self.params.pressure, tau_p=self.params.tau_p,
                    timestep=self.params.timestep, compressibility=self.params.compressibility))
            else:
                self._barostats.append(CRescaleBarostat(
                    proxy, pressure=self.params.pressure, temperature=self.params.temperature,
                    tau_p=self.params.tau_p, timestep=self.params.timestep,
                    compressibility=self.params.compressibility, rng=self._rngs[b]))
            cm = build_constraint_manager(proxy, self.params)
            self._constraints.append(cm)
        self._has_constraints = any(c is not None for c in self._constraints)
        # constraints remove DOF -> keep the thermostat target consistent.
        if self._has_constraints:
            for b in range(B):
                cm = self._constraints[b]
                if cm is not None:
                    self.n_dof[b] = max(int(self.n_dof[b]) - cm.n_dof_removed, 1)
                    self._thermostats[b]._n_dof = int(self.n_dof[b])
                    self._thermostats[b]._ke_target = (
                        0.5 * self.n_dof[b] * self._thermostats[b]._kT_target)

        self.dt_au = self.params.timestep * FS_TO_AU
        self._anneal_fn = make_anneal_fn(self.params.anneal, self.params.steps)
        self._hist_T, self._hist_PE, self._hist_P, self._hist_V, self._hist_rho = [], [], [], [], []
        self._steps_done = 0

    # ------------------------------------------------------------- force helper
    def _efs(self):
        """ONE batched forward -> (E (B,) Ha, F (B, nmax_dof) a.u., stress (B,6) eV/Ang^3)."""
        E_Ha, F_Ha, stress = self.calc.get_efs_gpu()
        F = F_Ha.to(self.device, self.dtype) * HA_PER_ANG_TO_AU
        E = E_Ha.to(self.device, self.dtype).reshape(-1)
        return E, F, stress.to(self.device, self.dtype)

    def _displace(self, v, frac_dt):
        disp = v * (frac_dt * self.dt_au) * BOHR_TO_ANGSTROM
        self.calc.step_cart_(disp)

    # --- per-replica numpy <-> padded-buffer slicing ------------------------
    def _v_real(self, v, b):
        n = int(self.n_b[b])
        return v[b, :3 * n].detach().to("cpu").numpy().reshape(n, 3)

    def _set_v_real(self, v, b, vb_np):
        n = int(self.n_b[b])
        v[b, :3 * n] = self._torch.tensor(vb_np.reshape(-1), dtype=self.dtype, device=self.device)

    def _calc_positions(self):
        return self.calc.coord.detach().to("cpu").numpy()          # (N_atoms,3)

    def _calc_cells(self):
        return self.calc._cell.detach().to("cpu").numpy()          # (B,3,3)

    def _sync_proxies_from_calc(self):
        """Load each proxy's positions (from calc master coords) + cell (from calc box)."""
        pos = self._calc_positions()
        cells = self._calc_cells()
        for b in range(self.B):
            self._atoms_b[b].set_cell(cells[b])
            self._atoms_b[b].set_positions(pos[self._ptr[b]:self._ptr[b + 1]])

    def _push_proxies_to_calc(self):
        """Write proxy positions + cells BACK into the batched calc (after the barostat/
        constraint reprojection modified them). Absolute set (not scale) -- the proxies
        hold the authoritative post-rescale geometry."""
        torch = self._torch
        pos = np.concatenate([self._atoms_b[b].get_positions() for b in range(self.B)], axis=0)
        cells = np.stack([np.asarray(self._atoms_b[b].get_cell()) for b in range(self.B)])
        self.calc.set_coords_(torch.tensor(pos, dtype=self.dtype, device=self.device))
        self.calc.set_cells_(torch.tensor(cells, dtype=self.dtype, device=self.device))

    def _inject_stress(self, E, stress):
        E_np = E.detach().to("cpu").numpy()
        s_np = stress.detach().to("cpu").numpy()
        for b in range(self.B):
            self._atoms_b[b].calc.set_results(energy=float(E_np[b]), stress=s_np[b])

    def _set_anneal_T(self, step):
        if self._anneal_fn is None:
            return
        T = self._anneal_fn(step)
        for th in self._thermostats:
            th.set_temperature(T)

    def _apply_thermostat(self, v):
        for b in range(self.B):
            vb = self._v_real(v, b)
            vb, _dw = self._thermostats[b].apply(vb)
            self._set_v_real(v, b, vb)
        return v

    def _apply_projection(self, v, step):
        ce = int(self.params.remove_com_every or 0)
        if not (ce > 0 and step % ce == 0):
            return v
        for b in range(self.B):
            vb = self._v_real(v, b)
            vb, _proj = apply_runtime_motion_projection(
                self._atoms_b[b], vb, step=step,
                remove_com_every=self.params.remove_com_every,
                remove_angular_every=self.params.remove_angular_every)
            self._set_v_real(v, b, vb)
        return v

    # ------------------------------------------------------ barostat (per replica)
    def _apply_barostat(self, v):
        """Per-replica AUTHORITATIVE barostat step. Proxies already hold the current
        (post-VV) positions + cell + injected stress. Each ``barostat.apply(v_b)``
        computes the replica's pressure (virial from injected stress + kinetic term)
        and rescales its box + positions via ase ``set_cell(scale_atoms=True)``. When
        constrained, RE-PROJECT the constraints AFTER the rescale (★CRITICAL). Then the
        rescaled geometry is pushed back into the batched calc. Returns (v, pressures)."""
        pressures = np.zeros(self.B)
        ref_pre = [None] * self.B
        if self._has_constraints:
            ref_pre = [self._atoms_b[b].get_positions().copy() for b in range(self.B)]
        for b in range(self.B):
            vb = self._v_real(v, b)
            pressures[b] = self._barostats[b].apply(vb)     # bar; scales proxy cell+pos
            # ★CRITICAL: barostat stretched every constrained bond by mu -> re-satisfy.
            cm = self._constraints[b]
            if cm is not None:
                cm.sync_cell(self._atoms_b[b])
                vb = cm.project_positions(self._atoms_b[b], ref_pre[b], vb, self.dt_au)
                vb = cm.project_velocities(self._atoms_b[b], vb)
                self._set_v_real(v, b, vb)
        self._push_proxies_to_calc()
        return v, pressures

    # --------------------------------------------------------- constraint hooks
    def _project_positions_after_drift(self, v, ref_positions):
        """RATTLE position stage per replica (constrained path only): restore bond
        lengths on the post-drift proxy geometry + correct the half-step velocities,
        then push the constraint-satisfied positions back into the calc."""
        pos = self._calc_positions()
        cells = self._calc_cells()
        for b in range(self.B):
            cm = self._constraints[b]
            self._atoms_b[b].set_cell(cells[b])
            self._atoms_b[b].set_positions(pos[self._ptr[b]:self._ptr[b + 1]])
            if cm is None:
                continue
            vb = self._v_real(v, b)
            vb = cm.project_positions(self._atoms_b[b], ref_positions[b], vb, self.dt_au)
            self._set_v_real(v, b, vb)
        self._push_proxies_to_calc()
        return v

    def _project_velocities(self, v):
        """RATTLE velocity stage per replica (constrained path only)."""
        for b in range(self.B):
            cm = self._constraints[b]
            if cm is None:
                continue
            vb = self._v_real(v, b)
            vb = cm.project_velocities(self._atoms_b[b], vb)
            self._set_v_real(v, b, vb)
        return v

    # ------------------------------------------------------------------- record
    def _record(self, v, E, stress, step):
        le = max(1, int(self.params.log_every or 1))
        if (step % le != 0) and (step != self.params.steps):
            return
        torch = self._torch
        # COM-subtracted (internal, 3N-3-consistent) kinetic T per replica.
        B = self.B
        mreal = (self.mass * self.mask).view(B, -1, 3)
        vr = v.view(B, -1, 3)
        mom = (mreal * vr).sum(dim=1)
        m_tot = mreal[:, :, 0].sum(dim=1).clamp_min(1e-30)
        v_com = mom / m_tot[:, None]
        vr_int = vr - v_com[:, None, :]
        ke_int = 0.5 * (mreal * vr_int * vr_int).sum(dim=(1, 2))
        ndof = torch.tensor(self.n_dof, dtype=self.dtype, device=self.device)
        T = (2.0 * ke_int / (ndof * KELVIN_TO_HARTREE)).detach().to("cpu").numpy()
        # logging pressure at the rescaled geometry (stress2 injected on proxies).
        self._inject_stress(E, stress)
        P = np.zeros(B)
        for b in range(B):
            vb = self._v_real(v, b)
            P[b], _w = compute_instantaneous_pressure(
                self._atoms_b[b], vb, stress_warned=True, class_name="BatchedNPT")
        vols = self.calc.volumes().detach().to("cpu").numpy()
        rho = np.array([self._mass_amu[b].sum() * _AMU_PER_ANG3_TO_G_CM3 / max(vols[b], 1e-30)
                        for b in range(B)])
        self._hist_T.append(T)
        self._hist_PE.append(E.detach().to("cpu").numpy())
        self._hist_P.append(P)
        self._hist_V.append(vols)
        self._hist_rho.append(rho)

    # ==================================================================== NPT run
    def _run_npt(self):
        if self._has_constraints:
            self._run_npt_constrained()
        else:
            self._run_npt_vrescale_middle()

    def _run_npt_vrescale_middle(self):
        """Unconstrained V-rescale "middle" scheme, mirroring single-system npt.py:
        B(dt/2) A(dt/2) O A(dt/2) [forward#1: F+stress] B(dt/2) [barostat] [box/COM]
        [forward#2: cache F + logging stress]."""
        v = self.v
        E, forces, stress = self._efs()                         # forward #0
        for step in range(1, self.params.steps + 1):
            self._set_anneal_T(step)
            v = v + 0.5 * forces / self.mass * self.dt_au        # B1 (cached F)
            self._displace(v, 0.5)                               # A half
            v = self._apply_thermostat(v)                        # O
            self._displace(v, 0.5)                               # A half
            E, forces, stress = self._efs()                      # forward #1 (F + stress)
            v = v + 0.5 * forces / self.mass * self.dt_au        # B2
            self._sync_proxies_from_calc()
            self._inject_stress(E, stress)                       # stress @ post-VV geom
            v, _P = self._apply_barostat(v)                      # rescale box+pos per replica
            self._box_guard(step)
            v = self._apply_projection(v, step)                  # runtime COM removal
            E, forces, stress = self._efs()                      # forward #2 (cache F, log stress)
            self._record(v, E, stress, step)
            self._steps_done = step
        self.v = v

    def _run_npt_constrained(self):
        """Constrained (RATTLE) path, mirroring the single-system npt.py constrained
        branch: monolithic VV + RATTLE (END-thermostat) then barostat + ★reprojection.
        B1 -> A(full)+RATTLE-pos -> forward#1 -> B2 -> RATTLE-vel -> O -> barostat
        (+reproject) -> box/COM -> forward#2."""
        v = self.v
        E, forces, stress = self._efs()
        for step in range(1, self.params.steps + 1):
            self._set_anneal_T(step)
            # RATTLE position-stage reference = the constraint-satisfied geometry at the
            # START of the step (calc's current positions, before B1/drift).
            ref_pos = self._ref_positions_current()
            v = v + 0.5 * forces / self.mass * self.dt_au        # B1
            self._displace(v, 1.0)                               # A full drift
            v = self._project_positions_after_drift(v, ref_pos)  # RATTLE pos (+ push to calc)
            E, forces, stress = self._efs()                      # forward #1
            v = v + 0.5 * forces / self.mass * self.dt_au        # B2
            v = self._project_velocities(v)                      # RATTLE vel
            v = self._apply_thermostat(v)                        # O (END)
            self._sync_proxies_from_calc()
            self._inject_stress(E, stress)
            v, _P = self._apply_barostat(v)                      # rescale + ★reproject
            self._box_guard(step)
            v = self._apply_projection(v, step)
            E, forces, stress = self._efs()                      # forward #2
            self._record(v, E, stress, step)
            self._steps_done = step
        self.v = v

    def _ref_positions_current(self):
        """RATTLE position-stage reference = the constraint-satisfied geometry at the
        START of the step (= the calc's current positions before the B1/drift)."""
        pos = self._calc_positions()
        return [pos[self._ptr[b]:self._ptr[b + 1]].copy() for b in range(self.B)]

    def _box_guard(self, step):
        cells = self._calc_cells()
        for b in range(self.B):
            self._atoms_b[b].set_cell(cells[b])
            check_box_size(self._atoms_b[b], calc=self.calc, mode=self.params.box_check,
                           context=f"BatchedNPT runtime step {step} replica {b} (after barostat)")

    # ----------------------------------------------------------------- finalize
    def _finalize(self):
        if not self._hist_T:
            self.results = []
            return
        T = np.asarray(self._hist_T)             # (nrec, B)
        PE = np.asarray(self._hist_PE)
        P = np.asarray(self._hist_P)
        V = np.asarray(self._hist_V)
        RHO = np.asarray(self._hist_rho)
        nrec = T.shape[0]
        tail = max(1, nrec // 5)
        self.results = []
        lines = ["\n" + "=" * 84 + "\n",
                 f"{'BATCHED NPT PER-REPLICA SUMMARY':^84}\n", "=" * 84 + "\n",
                 f"  B={self.B}  steps={self.params.steps}  dt={self.params.timestep} fs  "
                 f"T*={self.params.temperature:.1f} K  P*={self.params.pressure:.1f} bar  "
                 f"{self.params.thermostat}+{self.params.barostat}\n",
                 "\n  rep  natoms  <T>tail(K)  <P>tail(bar)  <V>tail(A^3)  <rho>tail(g/cm^3)\n"]
        for b in range(self.B):
            self.results.append(dict(
                replica=b, natoms=int(self.n_b[b]), n_dof=int(self.n_dof[b]),
                T_K=T[:, b].copy(), PE_Ha=PE[:, b].copy(), P_bar=P[:, b].copy(),
                V_A3=V[:, b].copy(), rho_g_cm3=RHO[:, b].copy(),
                T_tail_mean=float(np.mean(T[-tail:, b])),
                P_tail_mean=float(np.mean(P[-tail:, b])),
                V_tail_mean=float(np.mean(V[-tail:, b])),
                rho_tail_mean=float(np.mean(RHO[-tail:, b]))))
            lines.append(f"  {b:>3}  {int(self.n_b[b]):>5}   {np.mean(T[-tail:, b]):>8.2f}   "
                         f"{np.mean(P[-tail:, b]):>10.1f}   {np.mean(V[-tail:, b]):>10.2f}   "
                         f"{np.mean(RHO[-tail:, b]):>14.4f}\n")
        lines.append("=" * 84 + "\n")
        self.log_info(lines)

    # ----------------------------------------------------------------- logging
    def _log_parameters(self):
        p = self.params
        lines = ["\n" + "=" * 80 + "\n", f"{'BATCHED NPT PARAMETERS':^80}\n", "=" * 80 + "\n",
                 f"Replicas (B):    {self.B}\n",
                 f"Calculator:      {type(self.calc).__name__} (one get_efs_gpu/force-eval)\n",
                 f"Thermostat:      {p.thermostat}   tau_t={p.tau_t:.1f} fs\n",
                 f"Barostat:        {p.barostat}   tau_p={p.tau_p:.1f} fs  "
                 f"compressibility={p.compressibility:.2e} 1/bar\n",
                 f"Timestep:        {p.timestep:.3f} fs   steps={p.steps}\n",
                 f"Temperature:     {p.temperature:.2f} K   Pressure: {p.pressure:.2f} bar\n",
                 f"Constraints:     {p.constraints} (algorithm={p.constraint_algorithm})\n",
                 f"Remove COM ev.:  {p.remove_com_every} steps (runtime, per replica)\n"]
        if p.random_seed is not None:
            lines.append(f"Random seed:     {p.random_seed}\n")
        lines.append("=" * 80 + "\n")
        self.log_info(lines)
