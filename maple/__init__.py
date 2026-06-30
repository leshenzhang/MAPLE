import os as _os

# R2-F2: prefer CUDA's expandable_segments allocator to curb fragmentation under
# MAPLE's inherently variable batch sizes (auto_chunk adaptive budget,
# dynamic-shrink optimizer re-prepare, NEB band shrink/refill). The env var is
# only read at CUDA allocator init, so it MUST be set before torch creates a CUDA
# context. This package module is imported before any maple submodule's
# `import torch`, making it the earliest in-process hook. setdefault keeps any
# user/sbatch override (e.g. the bench sbatch `export PYTORCH_CUDA_ALLOC_CONF`).
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.1.4"
