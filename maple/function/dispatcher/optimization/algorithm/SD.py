# -*- coding: utf-8 -*-
"""
Backward-compatibility stub for SD optimizer.

The SD optimizer has been superseded by the SDCG fusion optimizer.
Use SDCG with sd_enabled=True, cg_enabled=False for SD-only behavior,
or use the default SDCG mode for SD+CG fusion.
"""
from .SDCG import SDCG as SD, SDCGParams as SDParams

__all__ = ["SD", "SDParams"]
