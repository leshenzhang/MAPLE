"""
Batched NVT (canonical) molecular dynamics for MAPLE -- C3.

ONE batched MLIP forward (``calc.get_ef_gpu()`` over all B replicas) advances B
independent systems per step, replacing the B separate ``atoms.get_forces()``
calls of the single-system path. Single-system NVT is the DEGENERATE B=1 case of
this same kernel -- there is NO B==1 special branch (DRY: one code path).

Authoritative physics (B's ``ensemble/nvt.py``) is preserved exactly:
  * Velocity-Verlet (Swope 1982) integration is vectorized over the padded
    ``(B, nmax_dof)`` buffer -- the ONE force-coupled, expensive part is batched.
  * The thermostat substep itself is delegated, per replica, to B's AUTHORITATIVE
    thermostat classes (``thermostat/vrescale.py`` Bussi 2007 A7;
    ``thermostat/langevin.py`` LF-Middle OU) -- NOT re-derived here. It is force
    free (no get_ef_gpu), so looping the small B axis for the per-replica
    chi²/OU draw costs nothing and guarantees the batched run reproduces the
    single-system thermostat math bit-for-bit (parity gate below).
  * Runtime COM/angular projection is delegated, per replica, to B's
    ``apply_runtime_motion_projection``.
  * HMR is applied per replica (mass-only; ``constraints.maybe_repartition_masses``)
    before the mass buffer / thermostats are built.

PARITY: with the same force engine + seed, B=1 through this kernel reproduces the
legacy single-system ``NVT`` class to fp64 machine precision (forces bit-identical,
VV arithmetic order matched, thermostat/projection RNG streams shared). See
``_test_nvt_batched.py``.

BACKEND gate: B>1 requires a batch-ISOLATED calculator (per-system energies
independent). AIMNet2-native (global charge equilibration) is REJECTED because it
leaks across the co-batched graph; MACE / MACE-OFF / UMA / AIMNet2-decoupled are
local/decoupled and allowed.

ponytail: DELIBERATE C3 ceilings (declared params so jobABC._init_params does NOT
strip them -- B-51 -- but rejected with a clear error in the batched kernel; use
the single-system ``NVT`` for these): RATTLE constraints, GaMD, steered MD,
PLUMED, Colvars, posres. NPT/NVE are separate ensembles. Isolated (non-periodic)
replicas only (the batch backends here carry no cell).
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from ase import Atoms

from ...jobABC import JobABC
from maple.function.timer import timer
from maple.function.utility import Molecules

from ..thermostat.langevin import LangevinThermostat
from ..thermostat.vrescale import VRescaleThermostat
from ..constraints import maybe_repartition_masses
from ..anneal import make_anneal_fn
from ..utils import (
    AMU_TO_AU,
    BOHR_TO_ANGSTROM,
    FS_TO_AU,
    KELVIN_TO_HARTREE,
    HA_PER_ANG_TO_AU,
    apply_runtime_motion_projection,
    calculate_kinetic_energy,
    calculate_temperature,
    initialize_velocities,
    standard_to_lfmiddle_carried,
)


@dataclass
class BatchedNVTParams:
    """Parameters for batched NVT. Honored fields mirror the single-system
    ``NVTParams`` so a B=1 replica reproduces a single-structure NVT run.

    The trailing block (constraints/gamd/smd/plumed/colvars/posres) is DECLARED
    (so _init_params keeps the keys -- B-51) but is a DELIBERATE ceiling in the
    batched kernel (rejected at setup; use the single-system NVT)."""
    timestep:        float = 0.5            # fs
    steps:           int   = 500            # MD steps
    temperature:     float = 300.0          # K (canonical target)
    thermostat:      str   = "langevin"     # 'langevin' | 'v-rescale'
    friction:        float = 0.01           # 1/fs   (Langevin)
    tau_t:           float = 100.0          # fs     (v-rescale)
    anneal:          str   = ""             # T schedule (K); "" = constant T
    traj_every:      int   = 50             # steps between recorded frames
    log_every:       int   = 50             # steps between main-log lines
    remove_com_every: int  = 100            # runtime per-replica COM removal cadence
    remove_angular_every: int = 0           # runtime per-replica COM+rotation cadence
    remove_com:      bool  = True           # initialization-only COM removal
    init_velocities: bool  = True
    verbose:         int   = 1
    random_seed:     Optional[int] = None
    # --- HMR (mass-only; supported) ---
    hmr:           str   = ""               # ""/off => no-op; on/true => 3.0; or a number
    hmr_factor:    Optional[float] = None
    hmr_bond_mult: float = 1.2
    # --- ponytail: declared-but-ceiling in the batched kernel (use single NVT) ---
    constraints: str = "none"               # none|h-bonds|all-bonds|h-angles
    constraint_algorithm: str = "lincs"
    gamd:   str = ""
    smd:    str = ""
    plumed: str = ""
    colvars: str = ""
    posres: str = ""


class BatchedNVT(JobABC):
    """Batched canonical (NVT) MD over B replicas with one forward per step."""

    _THERMOSTAT_CHOICES = {"langevin", "v-rescale"}
    # class names whose batch couples systems (per-system energies NOT independent)
    _COUPLED_CALC_NAMES = {"AIMNet2BatchCalc", "MACEPolBatchCalc"}

    def __init__(self, output: str,
                 systems: Union[Molecules, List[Atoms]],
                 calc=None,
                 paras: Optional[dict] = None):
        super().__init__(output)

        # ---- resolve replica list + batched calculator (mirror BatchedMD) -----
        if isinstance(systems, Molecules):
            atoms_list = list(systems.multiatoms)
            calc = calc if calc is not None else systems.calc
        else:
            atoms_list = list(systems)
            if calc is None and atoms_list:
                calc = getattr(atoms_list[0], "calc", None)
        if not atoms_list:
            raise ValueError("BatchedNVT requires at least one replica/system.")
        if calc is None:
            raise ValueError("BatchedNVT requires a batched calculator (Molecules.calc, "
                             "atoms[0].calc, or the calc= argument).")
        if not (callable(getattr(calc, "prepare", None))
                and callable(getattr(calc, "get_ef_gpu", None))
                and callable(getattr(calc, "step_cart_", None))):
            raise TypeError("BatchedNVT needs a batch calculator exposing "
                            "prepare()/get_ef_gpu()/step_cart_(). A plain ASE "
                            "single-structure calculator -> use the NVT class instead.")

        self.atoms_list = atoms_list
        self.calc = calc
        self.B = len(atoms_list)

        # BACKEND gate (B>1 must be batch-isolated).
        self._assert_batch_isolated(calc, self.B)

        self.params = self._init_params(
            BatchedNVTParams, paras, ("md", "MD", "nvt", "NVT", "batched", "batchnvt"))

        if self.params.thermostat not in self._THERMOSTAT_CHOICES:
            raise ValueError(f"Unknown thermostat '{self.params.thermostat}'. "
                             f"Choose from: {self._THERMOSTAT_CHOICES}")
        if any(any(at.pbc) for at in atoms_list):
            raise NotImplementedError(
                "BatchedNVT currently supports isolated (non-periodic) replicas only.")
        self._reject_ceiling_features()

        # HMR (mass-only) -- BEFORE building the mass buffer / thermostats.
        for at in self.atoms_list:
            maybe_repartition_masses(at, self.params, log=False)

        # torch is imported lazily (single-system MD stays torch-free at import).
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
            return                                  # one system: coupling is moot
        name = type(calc).__name__
        flag = getattr(calc, "batch_isolated", None)
        coupled = (flag is False) or (name in cls._COUPLED_CALC_NAMES) or (
            "AIMNet2" in name and "Decoupled" not in name)
        if coupled:
            raise ValueError(
                f"BatchedNVT B={B}>1 requires a batch-ISOLATED calculator (per-system "
                f"energies independent). Calculator '{name}' performs GLOBAL charge "
                f"equilibration across the co-batched graph, so replica energies/forces "
                f"LEAK into each other (perturbing one system changes another). Use a "
                f"local/decoupled batch calc (MACE / MACE-OFF / UMA / AIMNet2-decoupled), "
                f"or run replicas one at a time via the single-system NVT.")

    def _reject_ceiling_features(self):
        p = self.params
        bad = []
        if str(p.constraints or "none").strip().lower() not in ("", "none"):
            bad.append(f"constraints={p.constraints}")
        for nm in ("gamd", "smd", "plumed", "colvars", "posres"):
            val = getattr(p, nm, "")
            if str(val or "").strip().lower() not in ("", "off", "none", "no", "false", "0"):
                bad.append(f"{nm}={val}")
        if bad:
            raise NotImplementedError(
                "ponytail: the batched NVT kernel does not implement "
                f"{', '.join(bad)} (deliberate C3 ceiling). RATTLE constraints, GaMD, "
                "steered MD, PLUMED, Colvars and posres are available on the "
                "single-system NVT path -- run those systems one at a time.")

    # ====================================================================== run
    def run(self):
        with timer(f"Batched MD (NVT, B={self.B})"):
            self._log_parameters()
            self._prepare_buffers()
            if self.params.thermostat == "v-rescale":
                self._run_vrescale()
            else:
                self._run_langevin()
            self._finalize()
        return self

    # ------------------------------------------------------------ setup buffers
    def _prepare_buffers(self):
        torch = self._torch
        B = self.B
        self.calc.prepare(self.atoms_list, fixed_nmax=None)
        self.nmax_dof = int(self.calc.nmax_dof)
        nmax_a = self.nmax_dof // 3

        self.n_b = np.array([len(at) for at in self.atoms_list], dtype=int)     # (B,)
        # isolated -> 3N-3 (COM removed at init/runtime), matching the single-system
        # NVT default (remove_com_every>0 => linear_active => 3N-3).
        self.n_dof = np.maximum(3 * self.n_b - 3, 1)                            # (B,)

        dev, dt = self.device, self.dtype
        # mass / mask / inverse-mass buffers (B, nmax_dof) a.u.; padding mass = 1.
        mass = torch.ones((B, self.nmax_dof), dtype=dt, device=dev)
        mask = torch.zeros((B, self.nmax_dof), dtype=dt, device=dev)
        for b, at in enumerate(self.atoms_list):
            m_au = torch.tensor(at.get_masses(), dtype=dt, device=dev) * AMU_TO_AU
            n = m_au.shape[0]
            mass[b, :3 * n] = m_au.repeat_interleave(3)
            mask[b, :3 * n] = 1.0
        self.mass = mass
        self.mask = mask

        # per-replica RNG: rngs[b] drives BOTH replica b's velocity init AND its
        # thermostat (exactly as the single-system NVT shares one rng) -> the b=0
        # stream reproduces a single-system seeded run. (B-int seed; None => fresh.)
        seed = self.params.random_seed
        self._rngs = [np.random.default_rng(seed + b) if seed is not None
                      else np.random.default_rng() for b in range(B)]

        # positions buffer is kept inside the calc (master coords); velocities here.
        v = torch.zeros((B, self.nmax_dof), dtype=dt, device=dev)
        if self.params.init_velocities:
            for b, at in enumerate(self.atoms_list):
                vb = initialize_velocities(
                    atoms=at, temperature=self.params.temperature,
                    remove_com=self.params.remove_com, remove_rotation=False,
                    remove_angular=False, target_n_dof=int(self.n_dof[b]),
                    rng=self._rngs[b])                                          # (n,3) a.u.
                v[b, :vb.size] = torch.tensor(vb.reshape(-1), dtype=dt, device=dev)
        else:
            for b, at in enumerate(self.atoms_list):
                if "velocities" not in at.arrays:
                    raise ValueError(f"init_velocities=False but replica {b} has no "
                                     "velocities in atoms.arrays")
                vb = np.asarray(at.arrays["velocities"]).reshape(-1)
                v[b, :vb.size] = torch.tensor(vb, dtype=dt, device=dev)
        self.v = v

        # per-replica AUTHORITATIVE thermostat objects (B's classes, reused verbatim).
        self._thermostats = []
        for b, at in enumerate(self.atoms_list):
            if self.params.thermostat == "v-rescale":
                th = VRescaleThermostat(at, temperature=self.params.temperature,
                                        tau_t=self.params.tau_t,
                                        timestep=self.params.timestep,
                                        rng=self._rngs[b], n_dof=int(self.n_dof[b]))
            else:
                th = LangevinThermostat(at, temperature=self.params.temperature,
                                        friction=self.params.friction,
                                        timestep=self.params.timestep,
                                        rng=self._rngs[b])
            self._thermostats.append(th)

        self.dt_au = self.params.timestep * FS_TO_AU
        self._anneal_fn = make_anneal_fn(self.params.anneal, self.params.steps)
        self._hist_T, self._hist_KE, self._hist_PE = [], [], []
        self._steps_done = 0

    # ------------------------------------------------------------- force helper
    def _forces_au(self):
        """ONE batched forward -> (E (B,) Ha, F (B, nmax_dof) a.u. = Ha/Bohr)."""
        E_Ha, F_Ha = self.calc.get_ef_gpu()
        F = F_Ha.to(self.device, self.dtype) * HA_PER_ANG_TO_AU
        return E_Ha.to(self.device, self.dtype).reshape(-1), F

    def _displace(self, v, frac_dt):
        """Drift positions by ``v * frac_dt * dt`` (a.u. -> Angstrom) in the calc."""
        disp = v * (frac_dt * self.dt_au) * BOHR_TO_ANGSTROM
        self.calc.step_cart_(disp)

    # --- per-replica numpy <-> padded-buffer slicing helpers ------------------
    def _v_real(self, v, b):
        n = int(self.n_b[b])
        return v[b, :3 * n].detach().to("cpu").numpy().reshape(n, 3)

    def _set_v_real(self, v, b, vb_np):
        n = int(self.n_b[b])
        v[b, :3 * n] = self._torch.tensor(vb_np.reshape(-1), dtype=self.dtype,
                                          device=self.device)

    def _apply_thermostat(self, v, vrescale: bool):
        """Per-replica AUTHORITATIVE thermostat substep (force-free; reuses B's
        thermostat classes). ponytail: loops the small B axis -- vectorizing the
        chi²/OU draw would re-derive B's math and risk divergence."""
        for b in range(self.B):
            vb = self._v_real(v, b)
            if vrescale:
                vb, _dw = self._thermostats[b].apply(vb)     # Bussi A7 (+ work)
            else:
                vb = self._thermostats[b].apply(vb)          # LF-Middle OU
            self._set_v_real(v, b, vb)
        return v

    def _apply_projection(self, v, step):
        """Per-replica runtime COM/angular projection (reuses B's util)."""
        if not (self.params.remove_com_every or self.params.remove_angular_every):
            return v
        for b in range(self.B):
            vb = self._v_real(v, b)
            vb, _proj = apply_runtime_motion_projection(
                self.atoms_list[b], vb, step=step,
                remove_com_every=self.params.remove_com_every,
                remove_angular_every=self.params.remove_angular_every)
            self._set_v_real(v, b, vb)
        return v

    def _set_anneal_T(self, step):
        if self._anneal_fn is None:
            return
        T = self._anneal_fn(step)
        for th in self._thermostats:
            th.set_temperature(T)

    def _record(self, ke, E):
        self._hist_KE.append(ke)
        self._hist_PE.append(E)
        ndof = self._torch.tensor(self.n_dof, dtype=self.dtype, device=self.device)
        self._hist_T.append(2.0 * ke / (ndof * KELVIN_TO_HARTREE))

    # =================================================================== v-rescale
    def _run_vrescale(self):
        """Vectorized VV (Bussi 2007 post-step rescale). Mirrors nvt._run_simulation
        v-rescale branch: B(dt/2) -> A(dt) -> force -> B(dt/2) -> per-replica rescale."""
        torch = self._torch
        v = self.v
        E, F = self._forces_au()                              # cache F at t=0
        for step in range(1, self.params.steps + 1):
            self._set_anneal_T(step)
            v = v + 0.5 * F / self.mass * self.dt_au          # B1 half kick
            self._displace(v, 1.0)                            # A full drift
            E, F = self._forces_au()                          # ONE forward
            v = v + 0.5 * F / self.mass * self.dt_au          # B2 half kick
            v = self._apply_thermostat(v, vrescale=True)      # per-replica Bussi A7
            v = self._apply_projection(v, step)               # per-replica COM/angular
            ke = 0.5 * (self.mass * v * v).sum(dim=1)         # (B,) Ha
            self._record(ke, E)
            self._steps_done = step
        self.v = v

    # =================================================================== langevin
    def _run_langevin(self):
        """Vectorized LF-Middle Langevin (Leimkuhler & Matthews 2013). Mirrors
        nvt._run_simulation langevin branch in the CARRIED-velocity representation:
        full kick -> half drift -> OU thermostat -> half drift -> recompute forces.
        T/KE are reported from the SYNC (standard) velocity, matching nvt.py."""
        torch = self._torch
        v = self.v
        E, F = self._forces_au()
        # standard -> LF-Middle carried at t=0 (v_carried = v - 0.5*(F/m)*dt).
        v = v - 0.5 * F / self.mass * self.dt_au
        for step in range(1, self.params.steps + 1):
            self._set_anneal_T(step)
            v = v + F / self.mass * self.dt_au                # full kick (carried)
            self._displace(v, 0.5)                            # half drift
            v = self._apply_thermostat(v, vrescale=False)     # per-replica OU
            self._displace(v, 0.5)                            # half drift
            E, F = self._forces_au()                          # post-thermostat forward
            v = self._apply_projection(v, step)               # per-replica COM/angular
            # report SYNC (standard) KE/T: v_std = v_carried + 0.5*(F/m)*dt
            v_sync = v + 0.5 * F / self.mass * self.dt_au
            ke = 0.5 * (self.mass * v_sync * v_sync).sum(dim=1)
            self._record(ke, E)
            self._steps_done = step
        self.v = v

    # ----------------------------------------------------------------- finalize
    def _finalize(self):
        torch = self._torch
        if not self._hist_T:
            self.results = []
            return
        T = torch.stack(self._hist_T, 0).detach().to("cpu").numpy()    # (nstep, B)
        KE = torch.stack(self._hist_KE, 0).detach().to("cpu").numpy()
        PE = torch.stack(self._hist_PE, 0).detach().to("cpu").numpy()
        TE = KE + PE
        nstep = T.shape[0]
        tail = max(1, nstep // 5)                                       # last 20%
        self.results = []
        lines = ["\n" + "=" * 72 + "\n",
                 f"{'BATCHED NVT PER-REPLICA SUMMARY':^72}\n", "=" * 72 + "\n",
                 f"  B={self.B}  steps={self.params.steps}  dt={self.params.timestep} fs"
                 f"  thermostat={self.params.thermostat}  T*={self.params.temperature:.1f} K\n",
                 "\n  rep  natoms  ndof   <T>tail(K)  sig(T)(K)   <PE>(Ha)\n"]
        for b in range(self.B):
            t_b, pe_b = T[:, b], PE[:, b]
            t_tail_mean = float(np.mean(t_b[-tail:]))
            self.results.append(dict(
                replica=b, natoms=int(self.n_b[b]), n_dof=int(self.n_dof[b]),
                T_K=t_b.copy(), KE_Ha=KE[:, b].copy(), PE_Ha=pe_b.copy(), TE_Ha=TE[:, b].copy(),
                T_tail_mean=t_tail_mean, T_mean=float(np.mean(t_b)), T_std=float(np.std(t_b)),
                PE_mean=float(np.mean(pe_b))))
            lines.append(f"  {b:>3}  {int(self.n_b[b]):>5}  {int(self.n_dof[b]):>4}   "
                         f"{t_tail_mean:>9.2f}   {np.std(t_b):>8.2f}   {np.mean(pe_b):>12.6f}\n")
        lines.append("=" * 72 + "\n")
        self.log_info(lines)

    # ----------------------------------------------------------------- logging
    def _log_parameters(self):
        p = self.params
        lines = ["\n" + "=" * 72 + "\n", f"{'BATCHED NVT PARAMETERS':^72}\n", "=" * 72 + "\n",
                 f"Replicas (B):    {self.B}\n",
                 f"Calculator:      {type(self.calc).__name__} (one get_ef_gpu/step)\n",
                 f"Thermostat:      {p.thermostat}\n",
                 f"Timestep:        {p.timestep:.3f} fs\n",
                 f"Total steps:     {p.steps}\n",
                 f"Temperature:     {p.temperature:.2f} K\n"]
        if p.thermostat == "langevin":
            lines.append(f"Friction:        {p.friction:.4f} 1/fs\n")
        else:
            lines.append(f"tau_t:           {p.tau_t:.1f} fs\n")
        lines.append(f"Remove COM ev.:  {p.remove_com_every} steps (runtime, per replica)\n")
        if p.random_seed is not None:
            lines.append(f"Random seed:     {p.random_seed}\n")
        lines.append("=" * 72 + "\n")
        self.log_info(lines)
