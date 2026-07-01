"""
In-tree correctness gate for the NPT (isothermal-isobaric) ensemble.

Validation axis = ALGORITHM CORRECTNESS of the coupled v-rescale thermostat +
C-rescale (Bernetti & Bussi 2020) barostat in ensemble/npt.py.  The full
<P> -> 1 bar / rho -> 1 g/cc convergence over many ps is the EXTERNAL capstone;
this in-tree gate asserts the fast, deterministic correctness cores:

  A) C-RESCALE BETA-UNIT (calc-free, EXACT).  The deterministic volume update is
       dV/V = beta*(dt/tau_P)*(P - P_target),  beta in 1/bar, P in bar.
     With the noise switched off (T=0) the barostat.apply() must scale the volume
     by EXACTLY (1 + beta*(dt/tau_P)*(P - P_target)).  This catches any pressure-
     unit / compressibility-unit bug (the REMD-base-bug class): a beta given in
     1/Pa instead of 1/bar would make the response ~1e5x too small.  Sign
     convention (P>target -> expand) is also asserted.

  B) BOX-GUARD (calc-free).  The GROMACS-style minimum-image guard must FATALLY
     abort (RuntimeError) when a periodic width < 2*r_max, warn-and-continue in
     'warn' mode, and pass a large box / skip a non-PBC calculator.

  C) SINGLE-SYSTEM NPT WATER (a100).  Run 64-water at ~1 g/cc, T=300 K, P=1 bar
     with v-rescale + c-rescale through a real PBC+stress calculator
     (MACEOFFGenericCalculator).  Assert the run is STABLE and PHYSICAL:
       * density stays in a physical band (no collapse / blow-up),
       * <P>_tail is finite and bar-scale (units sane; not GPa nonsense),
       * shortest O-O distance > 1.8 A (no collapse).

Deterministic (fixed seeds).  Run as a script (never on import).
"""
import os, sys, tempfile
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from ase import Atoms

from maple.function.dispatcher.md.barostat.crescale import CRescaleBarostat
from maple.function.dispatcher.md.box_guard import check_box_size, evaluate_box
from maple.function.dispatcher.md.utils import (
    KELVIN_TO_HARTREE, EV_PER_ANG3_TO_BAR, HARTREE_TO_EV,
)

if __name__ != "__main__":
    raise SystemExit("run _test_npt.py as a script, not an import")

MODEL = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMU_G = 1.66053906660e-24        # amu -> g


def water_box(n=4, spacing=3.10, jitter=0.10, seed=1):
    rng = np.random.default_rng(seed)
    dOH, ang = 0.9572, np.deg2rad(104.52)
    base = np.array([[0, 0, 0], [dOH, 0, 0],
                     [dOH * np.cos(ang), dOH * np.sin(ang), 0.0]])
    pos, sym = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                q = rng.standard_normal(4); q /= np.linalg.norm(q); w, x, y, z = q
                R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                              [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                              [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
                c = (np.array([i, j, k]) + 0.5) * spacing + jitter * rng.standard_normal(3)
                pos.extend(base @ R.T + c); sym.extend(["O", "H", "H"])
    L = n * spacing
    return Atoms(symbols=sym, positions=np.array(pos), cell=[L, L, L], pbc=True)


def _density_gcc(atoms):
    mass_g = float(np.sum(atoms.get_masses())) * AMU_G
    vol_cm3 = float(atoms.get_volume()) * 1e-24
    return mass_g / vol_cm3


def _min_OO(atoms):
    O = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "O"]
    d = atoms[O].get_all_distances(mic=True)
    iu = np.triu_indices(len(O), 1)
    return float(np.min(d[iu]))


# ---------------------------------------------------------------- Gate A
def gate_crescale_beta_unit():
    box = water_box(n=2, spacing=4.0)                 # any periodic cell
    v = np.zeros((len(box), 3))
    comp = 4.5e-5        # 1/bar
    dt, tau = 0.5, 2000.0
    target = 1.0

    baro0 = CRescaleBarostat(box.copy(), pressure=target, temperature=0.0,
                             tau_p=tau, timestep=dt, compressibility=comp,
                             rng=np.random.default_rng(0))
    det_expect = comp * dt / tau
    ok_det = np.isclose(baro0._det_prefactor, det_expect, rtol=0, atol=1e-300 + 1e-15 * det_expect)
    ok_noise0 = (baro0._noise_prefactor == 0.0)       # T=0 kills the stochastic term
    print(f"[NPT-BETA] det_prefactor={baro0._det_prefactor:.6e} expect={det_expect:.6e} "
          f"(beta*dt/tau_P, dimensionless)  noise@T=0={baro0._noise_prefactor:.3e}")

    # deterministic volume response: V1/V0 must equal 1 + beta*(dt/tau)*(P - target)
    exact_ok = True
    sign_ok = True
    for P in (1.0e6, -1.0e6, 1.0):
        b = CRescaleBarostat(box.copy(), pressure=target, temperature=0.0,
                             tau_p=tau, timestep=dt, compressibility=comp,
                             rng=np.random.default_rng(0))
        b.get_pressure = (lambda _v, _P=P: _P)         # inject a known instantaneous P
        V0 = b.atoms.get_volume()
        b.apply(v)
        ratio = b.atoms.get_volume() / V0
        expect = 1.0 + det_expect * (P - target)
        exact_ok &= bool(np.isclose(ratio, expect, rtol=1e-10, atol=0.0))
        if P > target:
            sign_ok &= (ratio > 1.0)                   # over-pressure expands
        elif P < target:
            sign_ok &= (ratio < 1.0)                   # under-pressure shrinks
        print(f"[NPT-BETA]  P={P:>+.3e} bar  V1/V0={ratio:.9f}  expect={expect:.9f}")

    # noise-term unit check at T=300 K: sqrt(2 kT[eV] beta[A^3/eV] dt/tau) in sqrt(A^3)
    T = 300.0
    b300 = CRescaleBarostat(box.copy(), pressure=target, temperature=T,
                            tau_p=tau, timestep=dt, compressibility=comp,
                            rng=np.random.default_rng(0))
    kT_ev = T * KELVIN_TO_HARTREE * HARTREE_TO_EV
    beta_ang3_ev = comp * EV_PER_ANG3_TO_BAR
    noise_expect = np.sqrt(2.0 * kT_ev * beta_ang3_ev * (dt / tau))
    ok_noise = np.isclose(b300._noise_prefactor, noise_expect, rtol=1e-12, atol=0.0)
    print(f"[NPT-BETA] noise_prefactor@300K={b300._noise_prefactor:.6e} "
          f"expect={noise_expect:.6e} sqrt(A^3)")

    ok = ok_det and ok_noise0 and exact_ok and sign_ok and ok_noise
    print(f"[NPT-BETA] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- Gate B
class _MockPBCCalc:
    SUPPORTS_PBC = True
    r_max = 5.0                      # => required min width 2*r_max = 10 A


class _MockNoPBCCalc:
    SUPPORTS_PBC = False
    r_max = 5.0


def gate_box_guard():
    small = Atoms("Ar", positions=[[0, 0, 0]], cell=[8.0, 8.0, 8.0], pbc=True)   # 8 < 10
    big = Atoms("Ar", positions=[[0, 0, 0]], cell=[12.0, 12.0, 12.0], pbc=True)  # 12 > 10
    calc = _MockPBCCalc()

    ok_small, info = evaluate_box(small, calc.r_max)
    ok_big, _ = evaluate_box(big, calc.r_max)

    raised = False
    try:
        check_box_size(small, calc, "strict", context="NPT box-guard gate")
    except RuntimeError as e:
        raised = "too small" in str(e) or "VIOLATION" in str(e)
    passed_big = check_box_size(big, calc, "strict", context="NPT box-guard gate")
    warned = (check_box_size(small, calc, "warn", context="warn-mode") is False)   # no raise
    skipped_nonpbc = check_box_size(small, _MockNoPBCCalc(), "strict")             # r_max None-path -> skip

    print(f"[NPT-BOXGUARD] evaluate small(8A)ok={ok_small} big(12A)ok={ok_big} "
          f"required={info['required']:.1f}A")
    print(f"[NPT-BOXGUARD] strict-raise-on-small={raised}  strict-pass-big={passed_big}  "
          f"warn-continue={warned}  nonPBC-skip={skipped_nonpbc}")
    ok = (not ok_small) and ok_big and raised and passed_big and warned and skipped_nonpbc
    print(f"[NPT-BOXGUARD] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- Gate C
def gate_npt_water(steps=1500, dt=0.5, seed=3):
    from maple.function.calculator.generic._mace_off_generic import MACEOFFGenericCalculator
    from maple.function.dispatcher.md.ensemble.npt import NPT

    at = water_box(n=4, spacing=3.10, jitter=0.10, seed=1)      # 64 waters, side 12.4 A > 10
    rho0 = _density_gcc(at)
    side = float(at.cell.lengths()[0])
    at.calc = MACEOFFGenericCalculator(device=DEV, model_path=MODEL)
    paras = dict(timestep=dt, steps=steps, temperature=300.0, pressure=1.0,
                 thermostat="v-rescale", barostat="c-rescale",
                 tau_t=100.0, tau_p=2000.0, compressibility=4.5e-5,
                 init_velocities=True, random_seed=seed,
                 remove_com=True, remove_com_every=100,
                 log_every=1, traj_every=steps, verbose=0, box_check="strict")
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        out = f.name
    npt = NPT(out, at, paras=paras)
    npt.run()                                                    # RuntimeError here => box-guard tripped

    P = np.asarray(npt.logger.pressures, dtype=np.float64)       # bar, per step
    Ptail = P[int(0.5 * P.size):]
    Pmean = float(np.mean(Ptail)); Pstd = float(np.std(Ptail))
    rhoN = _density_gcc(npt.atoms)
    minOO = _min_OO(npt.atoms)

    print(f"[NPT-WATER] n_atoms={len(at)} side0={side:.2f}A rho0={rho0:.4f} g/cc  "
          f"steps={steps} dt={dt}fs")
    print(f"[NPT-WATER] <P>_tail={Pmean:.1f} +/- {Pstd:.1f} bar (target 1 bar; "
          f"tiny-box fluctuation-limited)  rho_final={rhoN:.4f} g/cc  minO-O={minOO:.2f}A")
    ok = (np.isfinite(Pmean) and abs(Pmean) < 3000.0
          and 0.70 < rhoN < 1.30 and minOO > 1.8)
    print(f"[NPT-WATER] {'PASS' if ok else 'FAIL'}")
    return ok, Pmean, rhoN, minOO


if __name__ == "__main__":
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    a = gate_crescale_beta_unit()
    b = gate_box_guard()
    c, Pmean, rhoN, minOO = gate_npt_water()
    allok = a and b and c
    print(f"\n[NPT RESULT] beta_unit={a} box_guard={b} npt_water={c} "
          f"(<P>={Pmean:.1f}bar rho={rhoN:.4f}g/cc minOO={minOO:.2f}A)")
    print(f"[NPT RESULT] {'ALL NPT GATES PASS' if allok else 'SOME NPT GATES FAILED'}")
    sys.exit(0 if allok else 1)
