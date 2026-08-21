"""GPU-free check on build_backend's UMA-only kwarg policy (D-309).

The pipeline dispatcher forwards fd_mode and fast_inference to every backend, and the
non-UMA branches refuse leftover kwargs so that a configured knob can never be silently
ignored. Refusing them at their DEFAULT value protects nothing and made the whole
second-backend arm unrunnable (job 50752795 died in 12 s). The rule is therefore:
a default is not a configuration, anything else still gets refused.
"""
from bench.core import _UMA_ONLY_DEFAULTS


def surviving(kw):
    """The kwargs build_backend would hand to a non-UMA backend."""
    kw = dict(kw)
    for k, default in _UMA_ONLY_DEFAULTS.items():
        if k in kw and bool(kw[k]) == bool(default) and kw[k] == default:
            kw.pop(k)
    return sorted(kw)


def test_defaults_are_dropped():
    assert surviving({"fd_mode": "central", "fast_inference": 0}) == []
    assert surviving({"fd_mode": "central", "fast_inference": False}) == []
    assert surviving({"hessian_mode": None}) == []


def test_every_non_default_is_still_refused():
    assert surviving({"fd_mode": "forward", "fast_inference": False}) == ["fd_mode"]
    assert surviving({"fd_mode": "central", "fast_inference": True}) == ["fast_inference"]
    assert surviving({"hessian_mode": "fd"}) == ["hessian_mode"]
    assert surviving({"fd_mode": "forward", "fast_inference": 1}) == ["fast_inference", "fd_mode"]


if __name__ == "__main__":
    test_defaults_are_dropped()
    test_every_non_default_is_still_refused()
    print("backend kwarg policy OK: defaults dropped, every non-default refused")
