from .calculator_base import (
    CalcABC,
    register_calculator,
    get_registered_calculator,
    import_calculator_plugin,
    load_calculator_plugins_from_env,
)
from .set_calculator import SetCalculator, SetClaculator

# Batch layer (BatchCalcABC / registry / factory) lives in batch_calculator_base,
# which imports torch at module top. Expose it lazily via PEP 562 __getattr__ so
# `import maple.function.calculator` stays torch-free (the dispatcher's lazy
# checkpoint-derive path relies on that) while these names remain importable as
# package attributes on first access.
_BATCH_EXPORTS = frozenset({
    "BatchCalcABC",
    "register_batch_calculator",
    "get_registered_batch_calculator",
    "make_batch_calc",
})


def __getattr__(name):
    if name in _BATCH_EXPORTS:
        from . import batch_calculator_base as _b
        return getattr(_b, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_BATCH_EXPORTS))
