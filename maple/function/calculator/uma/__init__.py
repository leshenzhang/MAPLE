# -*- coding: utf-8 -*-
"""UMA (fairchem) calculator package.

Shared fairchem env-compat shim (runs once, before ANY submodule below imports
fairchem). Python imports this package ``__init__`` before importing either
``_uma_calculator`` (single) or ``_uma_batch_calculator`` (batch), so placing the
guard here fixes BOTH calculator paths from one spot. The batch module keeps its
own idempotent copy of the same two steps for standalone-import safety; both are
no-ops once applied.

Why this is needed (compatibility shim, NOT a behavior change):
  (1) torch>=2.6 flips ``torch.load`` to ``weights_only=True`` by default. e3nn's
      ``o3/_wigner.py`` loads a ``constants.pt`` that pickles a ``slice`` object,
      and the UMA checkpoint load hits the same allowlist gate, so both raise
      ``UnpicklingError: GLOBAL slice ... was not an allowed global`` without an
      explicit allowlist. These are trusted local model assets ->
      ``add_safe_globals([slice])``.
  (2) ``import fairchem.core`` pulls a ``ray.serve`` batch-serve shim
      (``_batch_serve`` / ``InferenceBatcher``) whose fastapi dependency needs
      pydantic v2 (``IncEx`` from ``pydantic.main``); on a pydantic-v1 env the
      import raises ``ImportError: cannot import name 'IncEx'`` even though direct
      single/batch inference never touches ray.serve. A ``ray.serve`` stub in
      ``sys.modules`` short-circuits that import chain.
"""

import sys as _sys
import types as _types

import torch as _torch

# (1) Allowlist `slice` for torch.load weights_only=True (e3nn / UMA checkpoint).
try:
    _torch.serialization.add_safe_globals([slice])
except Exception:
    pass


# (2) Stub ray.serve so fairchem.core import does not pull a pydantic-v2 fastapi
#     chain. Direct UMA inference (single or batch) never uses ray.serve.
def _install_ray_serve_stub() -> None:
    try:
        import ray  # noqa: F401
    except Exception:
        return  # ray not present -> fairchem import path differs; nothing to do
    existing = _sys.modules.get("ray.serve")
    if existing is not None and getattr(existing, "_maple_stub", False):
        return
    try:
        import ray.serve  # noqa: F401  -- imports cleanly -> leave it alone
        return
    except Exception:
        pass
    serve = _types.ModuleType("ray.serve")
    serve._maple_stub = True
    serve.deployment = lambda *a, **k: (lambda cls: cls)
    serve.batch = lambda *a, **k: (lambda fn: fn)
    serve.handle = None
    serve.run = lambda *a, **k: None
    serve.start = lambda *a, **k: None
    schema = _types.ModuleType("ray.serve.schema")
    schema.LoggingConfig = lambda *a, **k: None
    serve.schema = schema
    import ray
    ray.serve = serve
    _sys.modules["ray.serve"] = serve
    _sys.modules["ray.serve.schema"] = schema


_install_ray_serve_stub()
