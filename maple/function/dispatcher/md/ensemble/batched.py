"""
Batched (replica / ensemble) molecular dynamics for MAPLE.

OPT-IN path: integrate ``B`` independent systems/replicas TOGETHER, with ONE
batched MLIP forward per step (``calc.get_ef_gpu()`` over all ``B``) instead of
``B`` single-structure ASE ``atoms.get_forces()`` calls. UMA is a LOCAL potential,
so the ``B`` molecules packed into one block-diagonal graph do not couple
(verified by the calculator's perturb-one byte-isolation parity gate); a batched
forward therefore returns exactly ``B`` independent energies/forces.

The single-structure NVE/NVT path (``ensemble/nve.py``, ``ensemble/nvt.py``) is
UNCHANGED and remains the oracle. This module reuses the SAME velocity-Verlet +
thermostat MATH, only vectorized over a padded ``(B, nmax_dof)`` buffer:
``atoms.get_forces()`` is replaced by ONE ``calc.get_ef_gpu()`` and the
per-atom (N,3) buffers become per-replica (B, nmax_dof) tensors. The masses/DOF
handling mirrors the single-system path.

Calculator contract (must be a LOCAL potential — UMABatchCalc / AIMNet2 *local*
build / MACE*BatchCalc; NOT the AIMNet2 charge-equilibration build, which couples
across the batch):
    prepare(atoms_list, fixed_nmax=None)
    step_cart_(s_cart: (B, nmax_dof))    in-place Cartesian displacement (Angstrom)
    get_ef_gpu() -> (E (B,) Ha, F (B, nmax_dof) Ha/Angstrom)
Padded layout (same as sp.py / scan.py / neb.py unpadding): replica ``b``, atom
``a``, axis ``c`` -> flat column ``3*a + c``; columns ``3*n_b .. nmax_dof`` are
zero padding.

Two known single-system bugs (NOT touched here — this is an independent path):
  * NPT v-rescale half-step  (ensemble/npt.py) — out of scope (NPT not added here).
  * isolated-NVE n_dof over-count (ensemble/nve.py default ``remove_com_every=0``
    leaves the runtime policy at ``n_dof = 3N`` even for an isolated molecule).
    This module computes per-replica ``n_dof = 3*n_b - 3`` for isolated systems
    (COM removed), matching the NVT default, so the reported temperature is
    canonical. (Energy-conservation drift is ``n_dof``-independent.)

References (same math as the single-system path):
  Velocity Verlet — Swope et al. (1982) J. Chem. Phys. 76, 637.
  Langevin LF-Middle (BAOAB-type OU substep) — Leimkuhler & Matthews (2013)
    Appl. Math. Res. eXpress 2013, 34; Zhang et al. (2019) J. Phys. Chem. A 123, 6056.
  Stochastic velocity rescaling (v-rescale) — Bussi, Donadio & Parrinello (2007)
    J. Chem. Phys. 126, 014101.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from ase import Atoms

from ...jobABC import JobABC
from maple.function.timer import timer
from maple.function.utility import Molecules

from ..utils import (
    AMU_TO_AU,
    BOHR_TO_ANGSTROM,
    FS_TO_AU,
    HA_PER_ANG_TO_AU,
    KELVIN_TO_HARTREE,
    initialize_velocities,
    write_xyz_frame,
)


@dataclass
class BatchedMDParams:
    """Parameters for batched (replica/ensemble) MD.

    Defaults mirror the single-system NVE/NVT params where they apply, so a
    batched replica reproduces a single-structure run with the same settings.
    """
    ensemble:    str   = "nve"          # 'nve' | 'nvt'
    timestep:    float = 0.5            # fs
    steps:       int   = 500            # MD steps
    temperature: float = 300.0          # K (NVT target; NVE velocity-init only)
    thermostat:  str   = "langevin"     # 'langevin' | 'v-rescale'  (NVT only)
    friction:    float = 0.01           # 1/fs   (Langevin)
    tau_t:       float = 100.0          # fs     (v-rescale)
    traj_every:  int   = 50             # steps between trajectory frames
    log_every:   int   = 50             # steps between main-log lines (thermo is per-step)
    traj_format: str   = "xyz"          # 'xyz' (only xyz implemented for the batched path)
    remove_com_every: int = 0           # runtime per-replica COM removal cadence (0=off; NVT default set in __init__)
    remove_com:  bool  = True           # initialization-only COM removal
    write_traj:  bool  = True           # write per-replica xyz trajectory + csv
    init_velocities: bool = True
    verbose:     int   = 1
    random_seed: Optional[int] = None


class BatchedMD(JobABC):
    """Batched velocity-Verlet MD over ``B`` replicas with one forward per step.

    Direct-API usage (the dispatcher routing is owned by a sibling wave and is
    NOT touched here; this is invoked like the other ``Batch*`` classes):

        from maple.function.utility import Molecules
        from maple.function.calculator.uma._uma_batch_calculator import UMABatchCalc
        calc = UMABatchCalc(model_path, device="cuda", dtype=torch.float64, task="omol")
        m = Molecules(atoms_list); m.calc = calc          # B independent systems
        sim = BatchedMD("run", m, paras={"ensemble": "nve", "steps": 500})
        sim.run()
        # per-replica histories in sim.results[b] -> dict(step,time_fs,T_K,KE_Ha,PE_Ha,TE_Ha)

    ``systems`` may also be a plain ``list[Atoms]`` (then ``calc`` is taken from
    ``atoms_list[0].calc`` or the explicit ``calc=`` argument).
    """

    _THERMOSTAT_CHOICES = {"langevin", "v-rescale"}

    def __init__(self, output: str,
                 systems: Union[Molecules, List[Atoms]],
                 calc=None,
                 paras: Optional[dict] = None):
        super().__init__(output)

        # ---- resolve the replica list + the batched calculator -----------------
        if isinstance(systems, Molecules):
            atoms_list = list(systems.multiatoms)
            calc = calc if calc is not None else systems.calc
        else:
            atoms_list = list(systems)
            if calc is None and atoms_list:
                calc = getattr(atoms_list[0], "calc", None)

        if not atoms_list:
            raise ValueError("BatchedMD requires at least one replica/system.")
        if calc is None:
            raise ValueError("BatchedMD requires a batched calculator (Molecules.calc, "
                             "atoms[0].calc, or the calc= argument).")
        if not (callable(getattr(calc, "prepare", None))
                and callable(getattr(calc, "get_ef_gpu", None))
                and callable(getattr(calc, "step_cart_", None))):
            raise TypeError("BatchedMD needs a batch calculator exposing "
                            "prepare()/get_ef_gpu()/step_cart_() (e.g. UMABatchCalc). "
                            "A plain ASE single-structure calculator -> use NVE/NVT instead.")

        # The batched calc holds one block-diagonal graph; PBC/cell handling is not
        # part of the UMA-omol batched contract, so this path is for isolated systems.
        if any(any(at.pbc) for at in atoms_list):
            raise NotImplementedError(
                "BatchedMD currently supports isolated (non-periodic) replicas only "
                "(UMA omol batched contract). Use the single-structure NVT/NPT path "
                "for periodic systems.")

        self.atoms_list = atoms_list
        self.calc = calc
        self.B = len(atoms_list)

        self.params = self._init_params(
            BatchedMDParams, paras, ("md", "MD", "batched", "BATCHED", "batchmd"))

        if self.params.ensemble.lower() not in ("nve", "nvt"):
            raise ValueError(f"BatchedMD ensemble must be 'nve' or 'nvt', "
                             f"got '{self.params.ensemble}'.")
        if self.params.thermostat not in self._THERMOSTAT_CHOICES:
            raise ValueError(f"Unknown thermostat '{self.params.thermostat}'. "
                             f"Choose from: {self._THERMOSTAT_CHOICES}")

        # NVT default: periodic per-replica COM removal (mirrors single-system NVT
        # remove_com_every=100) unless the caller set it explicitly via paras.
        if (self.params.ensemble.lower() == "nvt"
                and (paras is None or "remove_com_every" not in (paras or {}))):
            self.params.remove_com_every = 100

        # torch imported lazily so the single-system MD path stays torch-free at
        # package import; the batched path always runs under the torch/UMA env.
        import torch
        self._torch = torch
        dev = getattr(calc, "device", None)
        self.device = dev if dev is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(calc, "dtype", torch.float64)

    # ====================================================================== run
    def run(self):
        ens = self.params.ensemble.lower()
        with timer(f"Batched MD ({ens.upper()}, B={self.B})"):
            self._log_parameters()
            self._prepare_buffers()
            self._open_outputs()
            try:
                if ens == "nve":
                    self._run_nve()
                else:
                    self._run_nvt()
            finally:
                self._close_outputs()
            self._finalize()
        return self

    # ------------------------------------------------------------ setup buffers
    def _prepare_buffers(self):
        torch = self._torch
        B = self.B
        # ONE topology fix for all replicas (block-diagonal graph).
        self.calc.prepare(self.atoms_list, fixed_nmax=None)
        self.nmax_dof = int(self.calc.nmax_dof)
        nmax_a = self.nmax_dof // 3
        self.nmax_atoms = nmax_a

        self.n_b = np.array([len(at) for at in self.atoms_list], dtype=int)   # (B,)
        # Per-replica active DOF: isolated -> 3N-3 (COM removed at init/runtime),
        # matching the single-system NVT default and fixing the isolated-NVE
        # n_dof over-count noted in the module docstring.
        self.n_dof = np.maximum(3 * self.n_b - 3, 1)                          # (B,)

        dev, dt = self.device, self.dtype

        # mass (B, nmax_dof) a.u.; padding mass = 1.0 (safe denom, v=0 there).
        mass = torch.ones((B, self.nmax_dof), dtype=dt, device=dev)
        mask = torch.zeros((B, self.nmax_dof), dtype=dt, device=dev)
        atom_mass = torch.zeros((B, nmax_a), dtype=dt, device=dev)
        for b, at in enumerate(self.atoms_list):
            m_au = torch.tensor(at.get_masses(), dtype=dt, device=dev) * AMU_TO_AU  # (n,)
            n = m_au.shape[0]
            atom_mass[b, :n] = m_au
            mdof = m_au.repeat_interleave(3)                                  # (3n,)
            mass[b, :3 * n] = mdof
            mask[b, :3 * n] = 1.0
        self.mass = mass
        self.mask = mask
        self.atom_mass = atom_mass                                            # (B, nmax_a)
        self.atom_present = (atom_mass > 0).to(dt)                            # (B, nmax_a)
        self.minv = (1.0 / mass) * mask                                       # padding -> 0

        # positions buffer (B, nmax_dof) in Angstrom (tracked in parallel with the
        # calc master coords; identical displacements are applied to both).
        x = torch.zeros((B, self.nmax_dof), dtype=dt, device=dev)
        for b, at in enumerate(self.atoms_list):
            pos = torch.tensor(at.get_positions().reshape(-1), dtype=dt, device=dev)
            x[b, :pos.shape[0]] = pos
        self.x = x

        # velocities (B, nmax_dof) a.u., padding = 0.
        v = torch.zeros((B, self.nmax_dof), dtype=dt, device=dev)
        if self.params.init_velocities:
            rng_base = self.params.random_seed
            for b, at in enumerate(self.atoms_list):
                rng = (np.random.default_rng(rng_base + b)
                       if rng_base is not None else np.random.default_rng())
                vb = initialize_velocities(
                    atoms=at,
                    temperature=self.params.temperature,
                    remove_com=self.params.remove_com,
                    remove_angular=False,
                    target_n_dof=int(self.n_dof[b]),
                    rng=rng,
                )                                                            # (n,3) a.u.
                v[b, :vb.size] = torch.tensor(vb.reshape(-1), dtype=dt, device=dev)
        else:
            for b, at in enumerate(self.atoms_list):
                if "velocities" not in at.arrays:
                    raise ValueError(f"init_velocities=False but replica {b} has no "
                                     "velocities in atoms.arrays")
                vb = np.asarray(at.arrays["velocities"]).reshape(-1)
                v[b, :vb.size] = torch.tensor(vb, dtype=dt, device=dev)
        self.v = v

        # torch RNG for thermostat noise (Langevin).
        self._gen = torch.Generator(device=dev)
        if self.params.random_seed is not None:
            self._gen.manual_seed(int(self.params.random_seed))
        # numpy RNG for the per-replica v-rescale chi-squared draws.
        self._nrng = (np.random.default_rng(self.params.random_seed)
                      if self.params.random_seed is not None
                      else np.random.default_rng())

        # per-step history (kept on device, synced once at the end).
        self._hist_T, self._hist_KE, self._hist_PE = [], [], []
        self._steps_done = 0

        self.dt_au = self.params.timestep * FS_TO_AU

    # ------------------------------------------------------------- force helper
    def _forces_au(self):
        """ONE batched forward -> (E (B,) Ha, F (B, nmax_dof) a.u. = Ha/Bohr)."""
        E_Ha, F_Ha = self.calc.get_ef_gpu()          # F in Ha/Angstrom, padded
        F = F_Ha.to(self.device, self.dtype) * HA_PER_ANG_TO_AU
        return E_Ha.to(self.device, self.dtype).reshape(-1), F

    def _kinetic(self, v):
        """Per-replica KE (B,) [Ha] = 0.5 * sum(m * v^2) over DOFs (padding v=0)."""
        return 0.5 * (self.mass * v * v).sum(dim=1)

    def _temperature(self, ke):
        """Per-replica T (B,) [K] from KE and per-replica active n_dof."""
        torch = self._torch
        ndof = torch.tensor(self.n_dof, dtype=self.dtype, device=self.device)
        return 2.0 * ke / (ndof * KELVIN_TO_HARTREE)

    def _remove_com(self, v):
        """Project out per-replica COM velocity (real atoms only)."""
        B = self.B
        v3 = v.reshape(B, self.nmax_atoms, 3)
        p = (self.atom_mass[..., None] * v3).sum(dim=1)        # (B,3)
        M = self.atom_mass.sum(dim=1, keepdim=True)            # (B,1)
        com_v = p / M                                          # (B,3)
        v3 = (v3 - com_v[:, None, :]) * self.atom_present[..., None]
        return v3.reshape(B, self.nmax_dof)

    def _displace(self, v, frac_dt):
        """Drift positions by ``v * frac_dt`` (a.u. -> Angstrom) in calc + buffer."""
        disp = v * (frac_dt * self.dt_au) * BOHR_TO_ANGSTROM   # (B, nmax_dof) Angstrom
        self.calc.step_cart_(disp)
        self.x = self.x + disp

    def _record(self, ke, E):
        self._hist_KE.append(ke)
        self._hist_PE.append(E)
        self._hist_T.append(self._temperature(ke))

    # =================================================================== NVE run
    def _run_nve(self):
        """Vectorized Velocity Verlet with force caching (mirror of nve._run_simulation)."""
        v = self.v
        E, F = self._forces_au()                                # cache F at t=0
        for step in range(1, self.params.steps + 1):
            v = v + 0.5 * F * self.minv * self.dt_au            # B1 half kick
            self._displace(v, 1.0)                              # A full drift
            E, F = self._forces_au()                            # new forces
            v = v + 0.5 * F * self.minv * self.dt_au            # B2 half kick
            ke = self._kinetic(v)
            self._record(ke, E)
            self._maybe_write(step, E, ke)
            self._steps_done = step
        self.v = v

    # =================================================================== NVT run
    def _run_nvt(self):
        if self.params.thermostat == "langevin":
            self._run_nvt_langevin()
        else:
            self._run_nvt_vrescale()

    def _run_nvt_langevin(self):
        """Vectorized LF-Middle Langevin (mirror of the NVT langevin loop).

        Carried-velocity representation. OU substep coefficients are per-DOF:
            c1 = exp(-gamma*dt)                     (scalar)
            c2 = sqrt((1 - c1^2) * kB*T / m)        (per DOF; 0 on padding)
        Sequence per step: full kick -> half drift -> OU thermostat -> half drift
        -> recompute forces. T/KE are reported from the carried velocities, exactly
        as the single-system langevin path logs its raw thermo columns.
        """
        torch = self._torch
        v = self.v
        gamma_au = self.params.friction / FS_TO_AU
        c1 = float(np.exp(-gamma_au * self.dt_au))
        kT = self.params.temperature * KELVIN_TO_HARTREE
        c2 = torch.sqrt(torch.clamp((1.0 - c1 * c1) * kT / self.mass, min=0.0)) * self.mask

        E, F = self._forces_au()
        for step in range(1, self.params.steps + 1):
            v = v + F * self.minv * self.dt_au                 # full kick
            self._displace(v, 0.5)                             # half drift
            noise = torch.randn(v.shape, generator=self._gen,
                                dtype=self.dtype, device=self.device)
            v = c1 * v + c2 * noise                            # OU thermostat (padding c2=0)
            self._displace(v, 0.5)                             # half drift
            E, F = self._forces_au()                           # post-thermostat forces
            if self.params.remove_com_every and step % self.params.remove_com_every == 0:
                v = self._remove_com(v)
            # Report the SYNC-corrected (standard-velocity) kinetic temperature:
            # the LF-Middle carried velocity is offset from the standard velocity by
            # a half force-kick, v_std = v_carried + 0.5*(F/m)*dt
            # (lfmiddle_carried_to_standard); the standard-velocity KE is the
            # physically canonical kinetic temperature, matching the single-system
            # NVT langevin analysis (Temp_sync) column.
            v_sync = v + 0.5 * F * self.minv * self.dt_au
            ke = self._kinetic(v_sync)
            self._record(ke, E)
            self._maybe_write(step, E, ke)
            self._steps_done = step
        self.v = v

    def _run_nvt_vrescale(self):
        """Vectorized stochastic velocity rescaling (Bussi 2007), per replica.

        Standard VV step, then each replica is globally rescaled with its own
        kinetic energy / n_dof: same Eq. A7 as VRescaleThermostat, looped over the
        (small) batch for the per-replica chi-squared draw.
        """
        torch = self._torch
        v = self.v
        tau_au = self.params.tau_t * FS_TO_AU
        f = float(np.exp(-self.dt_au / tau_au))
        ke_target = 0.5 * self.n_dof.astype(float) * (self.params.temperature * KELVIN_TO_HARTREE)

        E, F = self._forces_au()
        for step in range(1, self.params.steps + 1):
            v = v + 0.5 * F * self.minv * self.dt_au           # standard VV
            self._displace(v, 1.0)
            E, F = self._forces_au()
            v = v + 0.5 * F * self.minv * self.dt_au
            # per-replica Bussi A7 rescale
            ke = self._kinetic(v)
            ke_np = ke.detach().to("cpu").numpy()
            alpha = np.ones(self.B, dtype=float)
            for b in range(self.B):
                K = float(ke_np[b])
                if K < 1e-30:
                    continue
                ndof = int(self.n_dof[b])
                c = ke_target[b] / (ndof * K)
                r1 = self._nrng.standard_normal()
                sum_r2 = (float(np.dot(w := self._nrng.standard_normal(ndof - 1), w))
                          if ndof > 1 else 0.0)
                a2 = (f + c * (1.0 - f) * (r1 * r1 + sum_r2)
                      + 2.0 * np.sqrt(f) * np.sqrt(c * (1.0 - f)) * r1)
                alpha[b] = np.sqrt(max(a2, 0.0))
            v = v * torch.tensor(alpha, dtype=self.dtype, device=self.device)[:, None]
            if self.params.remove_com_every and step % self.params.remove_com_every == 0:
                v = self._remove_com(v)
            ke = self._kinetic(v)
            self._record(ke, E)
            self._maybe_write(step, E, ke)
            self._steps_done = step
        self.v = v

    # ================================================================= outputs
    def _open_outputs(self):
        self._traj_files = None
        self._csv_files = None
        if not self.params.write_traj:
            return
        # JobABC stores the main output path in self.output.
        base = Path(getattr(self, "output", None) or "batched_md")
        stem, parent = base.stem, base.parent
        self._traj_files, self._csv_files = [], []
        for b in range(self.B):
            tf = open(parent / f"{stem}.rep{b}.xyz", "w")
            cf = open(parent / f"{stem}.rep{b}.csv", "w")
            cf.write("step,time_fs,T_K,KE_Ha,PE_Ha,TE_Ha\n")
            self._traj_files.append(tf)
            self._csv_files.append(cf)

    def _maybe_write(self, step, E, ke):
        if not self.params.write_traj:
            return
        write_csv = (step % self.params.log_every == 0) or (step == self.params.steps)
        write_xyz = (step % self.params.traj_every == 0) or (step == self.params.steps)
        if not (write_csv or write_xyz):
            return
        E_cpu = E.detach().to("cpu").numpy()
        ke_cpu = ke.detach().to("cpu").numpy()
        T_cpu = (self._temperature(ke)).detach().to("cpu").numpy()
        t_fs = step * self.params.timestep
        for b in range(self.B):
            te = float(ke_cpu[b] + E_cpu[b])
            if write_csv:
                self._csv_files[b].write(
                    f"{step},{t_fs:.4f},{T_cpu[b]:.4f},{ke_cpu[b]:.8f},"
                    f"{float(E_cpu[b]):.8f},{te:.8f}\n")
                self._csv_files[b].flush()
            if write_xyz:
                n = int(self.n_b[b])
                pos = self.x[b, :3 * n].detach().to("cpu").numpy().reshape(n, 3)
                at = self.atoms_list[b].copy()
                at.set_positions(pos)
                write_xyz_frame(self._traj_files[b], at, energy=te, frame_number=step)
                self._traj_files[b].flush()

    def _close_outputs(self):
        for handles in (getattr(self, "_traj_files", None), getattr(self, "_csv_files", None)):
            if handles:
                for h in handles:
                    try:
                        h.close()
                    except Exception:
                        pass

    # ---------------------------------------------------------------- finalize
    def _finalize(self):
        torch = self._torch
        if not self._hist_T:
            self.results = []
            return
        T = torch.stack(self._hist_T, dim=0).detach().to("cpu").numpy()   # (n, B)
        KE = torch.stack(self._hist_KE, dim=0).detach().to("cpu").numpy()
        PE = torch.stack(self._hist_PE, dim=0).detach().to("cpu").numpy()
        TE = KE + PE
        steps = np.arange(1, T.shape[0] + 1)
        time_fs = steps * self.params.timestep

        self.results = []
        lines = ["\n" + "=" * 78 + "\n",
                 f"{'BATCHED MD PER-REPLICA SUMMARY':^78}\n",
                 "=" * 78 + "\n",
                 f"  ensemble={self.params.ensemble.upper()}  B={self.B}  "
                 f"steps={self.params.steps}  dt={self.params.timestep} fs\n"]
        if self.params.ensemble.lower() != "nve":
            lines.append(f"  thermostat={self.params.thermostat}  T_target={self.params.temperature:.1f} K\n")
        lines.append("\n  rep  natoms   <T>(K)   sig(T)(K)   <TE>(Ha)        sig(TE)/|<TE>|\n")
        for b in range(self.B):
            te_b = TE[:, b]
            t_b = T[:, b]
            te_mean = float(np.mean(te_b))
            te_std = float(np.std(te_b))
            rel = te_std / abs(te_mean) if te_mean != 0 else float("nan")
            self.results.append(dict(
                replica=b, natoms=int(self.n_b[b]), n_dof=int(self.n_dof[b]),
                step=steps.copy(), time_fs=time_fs.copy(),
                T_K=t_b.copy(), KE_Ha=KE[:, b].copy(), PE_Ha=PE[:, b].copy(), TE_Ha=te_b.copy(),
                T_mean=float(np.mean(t_b)), T_std=float(np.std(t_b)),
                TE_mean=te_mean, TE_std=te_std, TE_rel_fluct=rel,
            ))
            lines.append(f"  {b:>3}  {int(self.n_b[b]):>5}   {np.mean(t_b):>7.2f}   "
                         f"{np.std(t_b):>8.2f}   {te_mean:>14.6f}   {rel:>14.3e}\n")
        lines.append("=" * 78 + "\n")
        self.log_info(lines)

    # ----------------------------------------------------------------- logging
    def _log_parameters(self):
        p = self.params
        lines = ["\n" + "=" * 78 + "\n",
                 f"{'BATCHED MD PARAMETERS':^78}\n",
                 "=" * 78 + "\n",
                 f"Ensemble:        {p.ensemble.upper()} (batched, B={self.B} replicas)\n",
                 f"Calculator:      {type(self.calc).__name__} (one get_ef_gpu per step)\n",
                 f"Timestep:        {p.timestep:.3f} fs\n",
                 f"Total steps:     {p.steps}\n",
                 f"Temperature:     {p.temperature:.2f} K"
                 f"{' (target)' if p.ensemble.lower()=='nvt' else ' (velocity init only)'}\n"]
        if p.ensemble.lower() == "nvt":
            lines.append(f"Thermostat:      {p.thermostat}\n")
            if p.thermostat == "langevin":
                lines.append(f"Friction:        {p.friction:.4f} 1/fs\n")
            else:
                lines.append(f"tau_t:           {p.tau_t:.1f} fs\n")
            lines.append(f"Remove COM ev.:  {p.remove_com_every} steps (runtime, per replica)\n")
        if p.random_seed is not None:
            lines.append(f"Random seed:     {p.random_seed}\n")
        lines.append("=" * 78 + "\n")
        self.log_info(lines)
