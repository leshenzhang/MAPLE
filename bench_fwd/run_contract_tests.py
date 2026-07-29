#!/usr/bin/env python
"""Dependency-free runner for tests/test_batch_calc_contract.py (cxtorch has no pytest).

Imports the module and calls every top-level ``test_*`` that takes no required args.
Prints PASS/FAIL per test + a summary; exits nonzero on any failure.
"""
import inspect
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import importlib.util  # noqa: E402

path = os.path.join(ROOT, "tests", "test_batch_calc_contract.py")
spec = importlib.util.spec_from_file_location("test_batch_calc_contract", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

names = [n for n in dir(mod) if n.startswith("test_")]
npass = nfail = nskip = 0
fails = []
for n in sorted(names):
    fn = getattr(mod, n)
    if not callable(fn):
        continue
    sig = inspect.signature(fn)
    if any(p.default is inspect.Parameter.empty
           and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
           for p in sig.parameters.values()):
        print(f"SKIP {n} (needs fixtures)")
        nskip += 1
        continue
    try:
        fn()
        print(f"PASS {n}")
        npass += 1
    except BaseException as exc:          # SystemExit from pytest.skip-alikes too
        if type(exc).__name__ in ("Skipped", "SystemExit"):
            print(f"SKIP {n} ({exc})")
            nskip += 1
            continue
        print(f"FAIL {n}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        fails.append(n)
        nfail += 1

print(f"\nSUMMARY pass={npass} fail={nfail} skip={nskip}")
if npass + nfail == 0:
    print("ERROR: ZERO tests executed -- collection failed (a green exit here would be "
          "a silent no-op). Failing loudly.")
    print("CONTRACT_TESTS_FAILED")
    sys.exit(2)
if fails:
    print("FAILED: " + ", ".join(fails))
print("CONTRACT_TESTS_" + ("OK" if nfail == 0 else "FAILED"))
sys.exit(1 if nfail else 0)
