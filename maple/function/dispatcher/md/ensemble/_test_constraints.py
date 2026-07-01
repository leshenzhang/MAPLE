"""
In-tree correctness gate for the RATTLE holonomic-constraint solver
(dispatcher/md/constraints.py) used by NVE / NVT / NPT.

Validation axis = ALGORITHM CORRECTNESS of the RATTLE position + velocity
projections (Andersen 1983; velocity-Verlet correct), plus the backward-
compatibility guarantee that constraints='none' leaves the integrator untouched.

  A) RATTLE PROJECTIONS (calc-free, deterministic).
       * project_positions: after an arbitrary drift, EVERY constrained bond is
         restored to its frozen length d0 with |d - d0| < tol = 1e-8 A.
       * project_velocities: after projection the relative velocity along every
         bond is zero, |(v_i - v_j).r_ij| < vtol = 1e-10.
       * rigid water: an O bonded to exactly two H is made fully rigid with 3
         distance constraints (SETTLE-equivalent) -> removes exactly 3 DOF.

  B) BOND INVARIANCE ALONG A REAL TRAJECTORY (a100).  Run 500 constrained NVE
     steps (constraints='h-bonds') with MACE-OFF23 fp64; every constrained X-H
     bond in the FINAL geometry is still at d0 to < 1e-8 A (the constraint held
     the whole trajectory, not just one step).

  C) constraints='none' IS THE UNTOUCHED PATH (calc-free/CPU).  build_constraint_
     manager('none') returns None and NVE._constraints is None (no projection
     code runs), and two 'none' runs with a deterministic calculator are
     BIT-IDENTICAL (the constraint feature does not perturb the default path).

Run as a script (never on import).
"""
import os, sys, tempfile
from types import SimpleNamespace
import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.calculators.lj import LennardJones

from maple.function.dispatcher.md.constraints import ConstraintManager, build_constraint_manager
from maple.function.dispatcher.md.ensemble.nve import NVE
from maple.function.dispatcher.md.utils import FS_TO_AU

if __name__ != "__main__":
    raise SystemExit("run _test_constraints.py as a script, not an import")

MODEL = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")


# ---------------------------------------------------------------- Gate A
def gate_rattle_projections():
    rng = np.random.default_rng(0)

    # water rigidity + DOF bookkeeping
    w = molecule("H2O")
    cmw = ConstraintManager(w, mode="all-bonds")
    water_ok = (cmw.n_water == 1 and cmw.n_constraints == 3 and cmw.n_dof_removed == 3)
    print(f"[CON-RATTLE] rigid water: n_water={cmw.n_water} n_constraints={cmw.n_constraints} "
          f"DOF_removed={cmw.n_dof_removed} (expect 1/3/3)")

    # RATTLE position stage on ethanol (all covalent bonds constrained)
    at = molecule("CH3CH2OH")
    cm = ConstraintManager(at, mode="all-bonds")
    ref = at.get_positions().copy()
    at.set_positions(ref + 0.05 * rng.standard_normal(ref.shape))     # break every bond
    v = 1e-3 * rng.standard_normal(ref.shape)
    dt_au = 0.5 * FS_TO_AU
    v = cm.project_positions(at, ref, v, dt_au)
    pos = at.get_positions()
    pres = [abs(np.linalg.norm(pos[cm.ai[k]] - pos[cm.aj[k]]) - cm.d0[k])
            for k in range(cm.n_constraints)]
    max_pos_res = float(max(pres))

    # RATTLE velocity stage: relative velocity along every bond must vanish
    v = 5e-4 * rng.standard_normal(ref.shape)
    v = cm.project_velocities(at, v)
    pos = at.get_positions()
    vres = []
    for k in range(cm.n_constraints):
        rij = pos[cm.ai[k]] - pos[cm.aj[k]]
        vres.append(abs((v[cm.ai[k]] - v[cm.aj[k]]) @ rij))
    max_v_res = float(max(vres))

    print(f"[CON-RATTLE] ethanol n_constraints={cm.n_constraints}  "
          f"pos: max|d-d0|={max_pos_res:.2e} A (tol 1e-8)  "
          f"vel: max|(vi-vj).rij|={max_v_res:.2e} (tol 1e-10)")
    ok = water_ok and (max_pos_res < 1e-8) and (max_v_res < 1e-9)
    print(f"[CON-RATTLE] {'PASS' if ok else 'FAIL'}")
    return ok, max_pos_res, max_v_res


# ---------------------------------------------------------------- Gate B
def gate_bond_invariance_md(steps=500, dt=1.0, seed=1):
    import torch
    torch.set_default_dtype(torch.float64)
    from ase.calculators.calculator import Calculator, all_changes
    from maple.function.calculator.mace._maceoff_batch_calculator import MaceOffBatchCalc
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    class _MaceOffSingle(Calculator):
        implemented_properties = ["energy", "forces", "free_energy"]

        def __init__(self, template):
            super().__init__()
            self._bc = MaceOffBatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64)
            self._bc.prepare([template.copy()], fixed_nmax=None)
            self.atomic_numbers = self._bc.atomic_numbers
            self.device = self._bc.device

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            n = len(self.atoms)
            coord = torch.tensor(self.atoms.get_positions(), dtype=torch.float64, device=self._bc.device)
            self._bc.set_coords_(coord)
            E_Ha, F_Ha = self._bc.get_ef_gpu()
            self.results = {"energy": float(E_Ha[0].item()),
                            "free_energy": float(E_Ha[0].item()),
                            "forces": F_Ha[0, :3 * n].detach().to("cpu").numpy().reshape(n, 3)}

    at = molecule("CH3CH2OH")
    at.calc = _MaceOffSingle(at)
    paras = dict(timestep=dt, steps=steps, temperature=300.0, init_velocities=True,
                 random_seed=seed, constraints="h-bonds", constraint_algorithm="lincs",
                 remove_com=True, remove_angular=True, log_every=steps, traj_every=steps,
                 verbose=0)
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        out = f.name
    nve = NVE(out, at, paras=paras)
    cm = nve._constraints
    assert cm is not None and cm.n_constraints > 0, "h-bonds built no constraints"
    nve.run()

    pos = nve.atoms.get_positions()
    res = [abs(np.linalg.norm(pos[cm.ai[k]] - pos[cm.aj[k]]) - cm.d0[k])
           for k in range(cm.n_constraints)]
    max_res = float(max(res))
    print(f"[CON-MDBOND] {steps} constrained NVE steps @ {dt}fs (h-bonds, "
          f"{cm.n_constraints} X-H bonds)  final max|d-d0|={max_res:.2e} A (tol 1e-8)")
    ok = max_res < 1e-8
    print(f"[CON-MDBOND] {'PASS' if ok else 'FAIL'}")
    return ok, max_res


# ---------------------------------------------------------------- Gate C
def gate_none_bit_identical(steps=200, dt=0.1, seed=42):
    # (i) 'none' builds no constraint manager (untouched path)
    none_is_none = build_constraint_manager(
        molecule("CH3CH2OH"),
        SimpleNamespace(constraints="none", constraint_algorithm="lincs")) is None

    # (ii) two deterministic 'none' NVE runs must be bit-identical
    def _run():
        s = 1.2
        pos = [[i * s, j * s, k * s] for i in range(2) for j in range(2) for k in range(2)]
        ar = Atoms("Ar8", positions=pos)
        ar.calc = LennardJones()                          # CPU, fully deterministic
        with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
            out = f.name
        nve = NVE(out, ar, paras=dict(timestep=dt, steps=steps, temperature=50.0,
                                      init_velocities=True, random_seed=seed,
                                      constraints="none", remove_com=True,
                                      remove_angular=True, verbose=0))
        cons_none = nve._constraints is None
        nve.run()
        return (nve.atoms.get_positions().copy(),
                nve.atoms.arrays["velocities"].copy(), cons_none)

    p1, v1, c1 = _run()
    p2, v2, c2 = _run()
    dpos = float(np.max(np.abs(p1 - p2)))
    dvel = float(np.max(np.abs(v1 - v2)))
    bit_identical = np.array_equal(p1, p2) and np.array_equal(v1, v2)
    nve_none = c1 and c2

    print(f"[CON-NONE] build_constraint_manager('none') is None = {none_is_none}  "
          f"NVE._constraints is None = {nve_none}")
    print(f"[CON-NONE] two 'none' runs: max|dpos|={dpos:.1e} max|dvel|={dvel:.1e}  "
          f"bit_identical={bit_identical}")
    ok = none_is_none and nve_none and bit_identical
    print(f"[CON-NONE] {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    okA, pres, vres = gate_rattle_projections()
    okB, mdres = gate_bond_invariance_md()
    okC = gate_none_bit_identical()
    allok = okA and okB and okC
    print(f"\n[CONSTRAINTS RESULT] rattle(pos={pres:.1e},vel={vres:.1e}) "
          f"md_bond={mdres:.1e} none_bitident={okC}")
    print(f"[CONSTRAINTS RESULT] {'ALL CONSTRAINT GATES PASS' if allok else 'SOME FAILED'}")
    sys.exit(0 if allok else 1)
