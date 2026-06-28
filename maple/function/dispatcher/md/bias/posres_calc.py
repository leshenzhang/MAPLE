"""GROMACS-style position restraints (posres) for MAPLE MD.

Same single-injection-point architecture as the enhanced-sampling bias layer
(:mod:`plumed_calc` / :mod:`colvars_calc`): a calculator wrapper recomputes a
harmonic tether force once per MD step at ``atoms.get_forces`` and folds it into
the MLIP force, so NVE/NVT/NPT all gain position restraints with **no integrator
change** and the restraint composes for free with any active PLUMED/Colvars bias.

This reproduces GROMACS ``define = -DPOSRES`` + ``[ position_restraints ]`` +
``grompp -r ref.gro``: a chosen set of atoms (heavy atoms / an explicit index
list) is harmonically tethered to a *reference* geometry so that, during
equilibration, the solute is pinned near its crystal/initial coordinates while
solvent and hydrogens relax; the force constant is then ramped down to zero.

Restraint potential (conservative — F = -dV/dx, so within a constant-k segment
it does not break energy-conservation diagnostics)::

    V       = 1/2 * k * Σ_i |Δr_i|²          [Ha]
    F_i     = -k * Δr_i                        [Ha/Å]
    Δr_i    = minimum_image(x_i - x_ref,i)     [Å]   (per-atom, restrained set)

Units
-----
MAPLE works in Hartree / Å / fs (energy Ha, force Ha/Å, positions Å — see
:mod:`colvars_calc`). The force constant ``posres_fc`` (and every entry of
``posres_ramp``) is therefore in **Ha/Å²** (energy per length²), so the force
comes out in Ha/Å — exactly what the integrator consumes. Convert from a GROMACS
force constant (kJ/mol/nm²)::

    k[Ha/Å²] = k[kJ/mol/nm²] / 627.5094740631 / 4.184 / 100
             = k[kJ/mol/nm²] * 3.808788e-6

e.g. a typical GROMACS 1000 kJ/mol/nm² ≈ 3.809e-3 Ha/Å².

Selection (``posres_group``)
----------------------------
* ``"all"``   — every atom.
* ``"heavy"`` — every non-hydrogen atom (Z != 1); the usual equilibration set.
* explicit 0-based index list, comma-separated, inclusive ranges allowed,
  e.g. ``"0,1,2,5-10"``.

Reference geometry (``posres``)
-------------------------------
* falsey / ``"off"``                                — restraint disabled.
* ``True`` / ``"on"`` / ``"initial"`` / ``"self"``  — reference = the structure at
  MD start (the initial coordinates; like ``grompp -r`` with the run input).
* a path to a structure file (anything ASE can read: .xyz/.gro/.pdb/...) —
  reference read from that file (like ``grompp -r ref.gro``); atom count must
  match the simulated system.

Ramp (``posres_ramp``)
----------------------
Optional descending schedule of force constants, e.g. ``"1000,500,100,0"`` — the
total run (``steps``) is split into N equal segments; segment j uses the j-th
value. Values are absolute k in Ha/Å² (same unit as ``posres_fc``) and *override*
``posres_fc`` when set. Segmentation counts force evaluations, which track MD
steps 1:1 under velocity Verlet (a few extra pre-run force calls at step 0 are
negligible for an equilibration schedule). At a segment boundary k jumps, so the
restraint PE has a small expected discontinuity there (same as a GROMACS staged
equilibration); within a segment H = KE + PE(incl. restraint) is conserved.
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

# 1 kJ/mol/nm² → Ha/Å²  (documented conversion; not used internally — user supplies Ha/Å²)
KJ_PER_MOL_NM2_TO_HA_ANG2 = 1.0 / 627.5094740631 / 4.184 / 100.0  # ≈ 3.808788e-6


def posres_enabled(posres):
    """True if a posres setting requests restraints (falsey/'off' ⇒ disabled)."""
    if posres is None:
        return False
    if isinstance(posres, bool):
        return posres
    if isinstance(posres, str):
        return posres.strip().lower() not in ("", "off", "no", "false", "none", "0")
    return bool(posres)


def _parse_index_list(spec, natoms):
    """Parse '0,1,2,5-10' → sorted 0-based index list; validate bounds."""
    idx = set()
    for tok in str(spec).replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok[1:]:  # range, but allow leading '-' nowhere (indices >=0)
            a, _, b = tok.partition("-")
            lo, hi = int(a), int(b)
            if lo > hi:
                lo, hi = hi, lo
            idx.update(range(lo, hi + 1))
        else:
            idx.add(int(tok))
    bad = sorted(i for i in idx if not (0 <= i < natoms))
    if bad:
        raise ValueError(
            f"posres_group index out of range [0,{natoms}): {bad}")
    return sorted(idx)


def _build_selection(group, atoms):
    """Return an int index array of restrained atoms."""
    natoms = len(atoms)
    if isinstance(group, str):
        gl = group.strip().lower()
        if gl in ("all", "*", ""):
            return np.arange(natoms)
        if gl in ("heavy", "non-hydrogen", "noh", "heavy-atoms", "heavyatoms"):
            z = np.asarray(atoms.get_atomic_numbers())
            return np.where(z != 1)[0]
        return np.asarray(_parse_index_list(group, natoms), dtype=int)
    # list/tuple/array of explicit indices
    return np.asarray(
        _parse_index_list(",".join(str(int(x)) for x in group), natoms), dtype=int)


def _parse_ramp(spec):
    """Return list[k] or None (constant-fc) from a ramp spec."""
    if spec is None:
        return None
    if isinstance(spec, bool):
        return None
    if isinstance(spec, (int, float)):
        return [float(spec)]
    if isinstance(spec, (list, tuple)):
        vals = [float(x) for x in spec]
        return vals or None
    s = str(spec).strip().lower()
    if s in ("", "off", "none", "no", "false"):
        return None
    vals = [float(x) for x in str(spec).replace(";", ",").split(",") if x.strip()]
    return vals or None


def _load_reference(posres, atoms):
    """Return (N,3) reference positions in Å (file like grompp -r, or initial)."""
    import os
    if isinstance(posres, str):
        sl = posres.strip().lower()
        if sl in ("on", "yes", "true", "initial", "init", "self"):
            return np.asarray(atoms.get_positions(), np.float64).copy()
        if os.path.exists(posres):
            from ase.io import read
            ref = read(posres)
            if len(ref) != len(atoms):
                raise ValueError(
                    f"posres reference '{posres}' has {len(ref)} atoms but the "
                    f"system has {len(atoms)}; atom count must match (grompp -r).")
            return np.asarray(ref.get_positions(), np.float64)
        raise FileNotFoundError(
            f"posres reference file not found: {posres!r} (use 'initial' to "
            "tether to the starting coordinates, or 'off' to disable).")
    # True / non-str truthy ⇒ initial coordinates
    return np.asarray(atoms.get_positions(), np.float64).copy()


class PosresCalculator(Calculator):
    """Wrap a MAPLE MLIP (or bias) calculator with GROMACS-style posres."""

    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

    def __init__(self, inner, posres, fc, group, ramp, total_steps,
                 *, atoms=None, output='run', restart_step=0):
        Calculator.__init__(self)
        if atoms is None:
            raise ValueError("PosresCalculator needs the initial atoms to build "
                             "the reference geometry and atom selection.")
        self.inner = inner
        self._istep = int(restart_step)
        self._total_steps = max(int(total_steps or 0), 1)
        self._output = output
        self.SUPPORTS_PBC = getattr(inner, 'SUPPORTS_PBC', False)

        self._ref = _load_reference(posres, atoms)        # (N,3) Å
        self._sel = _build_selection(group, atoms)        # int index array
        self._ramp = _parse_ramp(ramp)                    # list[k] or None
        self._fc = float(fc)
        self.atoms = atoms.copy()
        self._last_k = None
        self._last_pe = 0.0

    def _k_now(self):
        if not self._ramp:
            return self._fc
        n = len(self._ramp)
        seg = int(self._istep * n / self._total_steps)
        seg = max(0, min(n - 1, seg))
        return float(self._ramp[seg])

    @staticmethod
    def _min_image(disp, cell, pbc):
        """Minimum-image of cartesian displacements (M,3) for the periodic dirs."""
        pbc = np.asarray(pbc, bool)
        if not np.any(pbc):
            return disp
        cell = np.asarray(cell, np.float64)
        if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-12:
            return disp
        frac = disp @ np.linalg.inv(cell)
        shift = np.round(frac)
        shift[:, ~pbc] = 0.0
        return disp - shift @ cell

    def calculate(self, atoms=None, properties=('energy', 'forces'),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        atoms = self.atoms

        self.inner.calculate(atoms, list(properties), system_changes)
        energy = float(self.inner.results['energy'])                 # Ha
        forces = np.array(self.inner.results['forces'], np.float64)  # Ha/Å (copy)

        k = self._k_now()
        sel = self._sel
        e_res = 0.0
        if k != 0.0 and len(sel) > 0:
            pos = np.asarray(atoms.get_positions(), np.float64)      # Å
            disp = pos[sel] - self._ref[sel]                         # Å
            disp = self._min_image(disp, atoms.get_cell()[:], atoms.pbc)
            forces[sel] = forces[sel] - k * disp                     # F = -k Δr
            e_res = 0.5 * k * float(np.sum(disp * disp))             # Ha
        self._istep += 1
        self._last_k = k
        self._last_pe = e_res

        self.results = {
            'energy': energy + e_res,
            'free_energy': energy + e_res,
            'forces': forces,
            'posres_energy': e_res,
            'posres_k': k,
        }
        if 'stress' in self.inner.results:
            self.results['stress'] = np.asarray(self.inner.results['stress'])


# --------------------------------------------------------------------------- #
# Standalone numpy/ASE self-check (no maple import; no GPU). Run:
#   PYTHONSAFEPATH=1 python posres_calc.py
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    from ase import Atoms

    class _ZeroCalc(Calculator):
        implemented_properties = ['energy', 'forces']

        def calculate(self, atoms=None, properties=('energy', 'forces'),
                      system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            n = len(self.atoms)
            self.results = {'energy': 0.0, 'forces': np.zeros((n, 3))}

    ok = True

    # (1) F = -k Δr and E = 1/2 k Δr², select 'all', non-periodic.
    a = Atoms('H3', positions=[[0., 0, 0], [1, 0, 0], [2, 0, 0]])
    a.calc = _ZeroCalc()
    w = PosresCalculator(a.calc, 'initial', 1.0, 'all', '', 100, atoms=a)
    a.calc = w
    a.positions[0] = [0.3, 0.0, 0.0]
    f = a.get_forces()
    e = a.get_potential_energy()
    ok &= abs(f[0, 0] + 0.3) < 1e-12 and abs(e - 0.045) < 1e-12
    print(f"(1) F0x={f[0,0]:+.6f} (exp -0.300000)  E={e:.6f} (exp 0.045000)  "
          f"F1x={f[1,0]:+.6f} (exp 0)")

    # (2) k=0 ⇒ zero restraint force/energy (regression invariance).
    a2 = Atoms('H3', positions=[[0., 0, 0], [1, 0, 0], [2, 0, 0]])
    a2.calc = _ZeroCalc()
    w2 = PosresCalculator(a2.calc, 'initial', 0.0, 'all', '', 100, atoms=a2)
    a2.calc = w2
    a2.positions[0] = [0.5, 0.0, 0.0]
    f2 = a2.get_forces()
    e2 = a2.get_potential_energy()
    ok &= np.allclose(f2, 0.0) and e2 == 0.0
    print(f"(2) k=0  max|F|={np.abs(f2).max():.1e}  E={e2:.1e} (exp 0)")

    # (3) 'heavy' selects O, not H; restraint only on heavy atom.
    a3 = Atoms('OH2', positions=[[0., 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    a3.calc = _ZeroCalc()
    w3 = PosresCalculator(a3.calc, 'initial', 2.0, 'heavy', '', 100, atoms=a3)
    a3.calc = w3
    a3.positions[0] = [0.1, 0.0, 0.0]   # move O
    a3.positions[1] = [1.20, 0.0, 0.0]  # move an H
    f3 = a3.get_forces()
    ok &= abs(f3[0, 0] + 0.2) < 1e-12 and np.allclose(f3[1], 0.0)
    print(f"(3) heavy: F_O_x={f3[0,0]:+.6f} (exp -0.200000)  "
          f"F_H={f3[1]}  (exp [0 0 0])")

    # (4) ramp: k descends per segment over total_steps.
    a4 = Atoms('H', positions=[[0., 0, 0]])
    a4.calc = _ZeroCalc()
    w4 = PosresCalculator(a4.calc, 'initial', 0.0, 'all', '4,2,1,0', 100,
                          atoms=a4)
    ks = []
    for st in (0, 30, 60, 99):
        w4._istep = st
        ks.append(w4._k_now())
    ok &= ks == [4.0, 2.0, 1.0, 0.0]
    print(f"(4) ramp k at steps[0,30,60,99]={ks} (exp [4.0, 2.0, 1.0, 0.0])")

    # (5) minimum image: ref across a periodic boundary ⇒ tiny disp, tiny force.
    cell = 10.0 * np.eye(3)
    a5 = Atoms('H', positions=[[0.1, 0, 0]], cell=cell, pbc=True)
    a5.calc = _ZeroCalc()
    w5 = PosresCalculator(a5.calc, 'initial', 1.0, 'all', '', 100, atoms=a5)
    a5.calc = w5
    a5.positions[0] = [9.9, 0.0, 0.0]   # 9.8 Å naive, but MIC ⇒ -0.2 Å
    f5 = a5.get_forces()
    ok &= abs(f5[0, 0] - 0.2) < 1e-9   # F = -k*(-0.2) = +0.2
    print(f"(5) MIC: F_x={f5[0,0]:+.6f} (exp +0.200000, not -9.8)")

    print("SELF-CHECK:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
