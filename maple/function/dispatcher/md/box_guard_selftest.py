#!/usr/bin/env python3
"""CPU self-test for the MD box-size guard (no GPU, no maple package import).

Run:  python maple/function/dispatcher/md/box_guard_selftest.py
Loads the sibling box_guard.py standalone (numpy-only) via importlib so we
exercise the guard logic without pulling torch/MACE. Uses real ase.Atoms.
"""
import importlib.util
import os
import sys

import numpy as np
from ase import Atoms

HERE = os.path.dirname(os.path.abspath(__file__))
BG_PATH = os.path.join(HERE, "box_guard.py")
spec = importlib.util.spec_from_file_location("box_guard", BG_PATH)
bg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bg)

fails = []


def check(cond, label):
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        fails.append(label)


class PBCCalc:
    SUPPORTS_PBC = True
    def __init__(self, r_max):
        self.r_max = r_max


class NoPBCCalc:
    SUPPORTS_PBC = False
    r_max = 6.0


R_MAX = 6.0
REQ = 2.0 * R_MAX  # 12.0


def cubic(L, pbc=True):
    a = Atoms("Cu", positions=[[0, 0, 0]], cell=[L, L, L], pbc=pbc)
    return a


# -- 1. perpendicular widths: orthorhombic == vector norms --
w = bg.box_perpendicular_widths([[15, 0, 0], [0, 12, 0], [0, 0, 20]])
check(np.allclose(w, [15, 12, 20]), f"perp widths orthorhombic = {w}")

# -- 2. perpendicular width < vector norm for a skewed (triclinic) cell --
# vector b is long but skewed toward a -> small perpendicular separation.
cell = [[10, 0, 0], [9, 4, 0], [0, 0, 20]]
w2 = bg.box_perpendicular_widths(cell)
norms = np.linalg.norm(np.array(cell, float), axis=1)
check(w2[1] < norms[1] and abs(w2[1] - 4.0) < 1e-9,
      f"triclinic perp width b={w2[1]:.3f} < norm b={norms[1]:.3f} (stricter)")

# -- 3. r_max only read for PBC calculators --
check(bg.get_calculator_r_max(PBCCalc(6.0)) == 6.0, "get_calculator_r_max PBC -> 6.0")
check(bg.get_calculator_r_max(NoPBCCalc()) is None, "get_calculator_r_max non-PBC -> None")

# -- 4. good box passes (strict) --
ok = bg.check_box_size(cubic(15.0), PBCCalc(R_MAX), "strict", context="unit good")
check(ok is True, "strict: 15 A box (>=12) passes")

# -- 5. small box -> strict FATAL with named-vector / r_max / required in msg --
raised = None
try:
    bg.check_box_size(cubic(9.0), PBCCalc(R_MAX), "strict", context="unit small")
except RuntimeError as e:
    raised = str(e)
msg_ok = (raised is not None
          and "6.000" in raised               # r_max
          and "12.000" in raised              # required = 2*r_max
          and "9.000" in raised               # offending width
          and "VIOLATION" in raised
          and "FATAL" in raised)
check(msg_ok, "strict: 9 A box raises RuntimeError with r_max+width+required")
if raised:
    print("---- fatal message ----")
    print(raised)
    print("-----------------------")

# -- 6. warn mode: returns False, emits warning, does NOT raise --
captured = []
ok = bg.check_box_size(cubic(9.0), PBCCalc(R_MAX), "warn",
                       context="unit warn", warn=captured.append)
check(ok is False and captured and "WARNING" in captured[0], "warn: returns False + emits warning, no raise")

# -- 7. off mode: skip even tiny box --
ok = bg.check_box_size(cubic(3.0), PBCCalc(R_MAX), "off", context="unit off")
check(ok is True, "off: tiny box skipped")

# -- 8. non-PBC calc: skip even tiny box (no minimum image) --
ok = bg.check_box_size(cubic(3.0), NoPBCCalc(), "strict", context="unit nopbc")
check(ok is True, "non-PBC calc: tiny box skipped (no image self-interaction)")

# -- 9. non-periodic system (pbc=False): skip --
ok = bg.check_box_size(cubic(3.0, pbc=False), PBCCalc(R_MAX), "strict", context="unit nonperiodic")
check(ok is True, "non-periodic atoms: skipped")

# -- 10. RUNTIME guard analogue: shrink cell across threshold -> raises at crossing --
atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=[13.0, 13.0, 13.0], pbc=True)
calc = PBCCalc(R_MAX)
crossed_step = None
for step in range(1, 40):
    atoms.set_cell(atoms.get_cell() * 0.98, scale_atoms=True)  # ~2% shrink/step
    try:
        bg.check_box_size(atoms, calc, "strict",
                          context=f"NPT runtime step {step} (after barostat rescale)")
    except RuntimeError as e:
        crossed_step = step
        runtime_width = float(min(np.linalg.norm(np.array(atoms.get_cell()), axis=1)))
        check("after barostat rescale" in str(e), "runtime: context label present in message")
        break
check(crossed_step is not None, f"runtime: abort fired at step {crossed_step} (width crossed below 12 A)")

# -- 11. composition_sanity: overlapping atoms -> warning --
clash = Atoms("Cu2", positions=[[0, 0, 0], [0.2, 0, 0]], cell=[20, 20, 20], pbc=True)
cap = []
bg.composition_sanity(clash, PBCCalc(R_MAX), "strict", context="unit clash", warn=cap.append)
check(any("interatomic distance" in m for m in cap), "composition_sanity: overlapping atoms warned")

print()
if fails:
    print(f"RESULT: {len(fails)} FAILED -> {fails}")
    sys.exit(1)
print("RESULT: ALL UNIT TESTS PASS")
