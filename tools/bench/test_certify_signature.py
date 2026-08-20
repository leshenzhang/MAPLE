"""Runnable check for the _certify call sites: the probe-mode parameter and the
calculator kwargs must never collide (job 50722551 died with 'got multiple values
for keyword argument hessian_mode' on the UMA arm only)."""
import importlib.util, inspect, os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "MAPLE", "tools"))
spec = importlib.util.spec_from_file_location(
    "rm", os.path.join(here, "..", "MAPLE", "tools", "bench", "run_matrix.py"))
rm = importlib.util.module_from_spec(spec); spec.loader.exec_module(rm)
sig = inspect.signature(rm._certify)
for backend in ("uma", "mace_traced", "mace_autograd"):
    kw = dict(hessian_mode="numerical") if backend == "uma" else {}
    sig.bind(object(), needs_hessian=True, probe_mode="numerical", **kw)   # must not raise
sig.bind(object(), needs_hessian=True, fd_mode="forward", fast_inference=1)  # pipeline call site
print("certify signature check OK: probe_mode + calc kwargs bind for uma/mace_traced/mace_autograd + pipeline")
