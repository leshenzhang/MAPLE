# -*- coding: utf-8 -*-
"""
maple.workflow -- higher-level orchestrations built on the MAPLE dispatchers.

These consume the core engine (single-point, EM pre-stage, MD ensembles) as a
Python library to assemble multi-structure scientific workflows:

* ``docking``  -- bridge an external docked pose into a MAPLE EM -> MD run.
* ``mmpbsa``   -- single-trajectory endpoint binding free energy with an MLIP
                  energy engine (the MAPLE-native half of MM/GBSA).
* ``amber_io`` -- dependency-light AMBER prmtop + NetCDF readers used by both.
"""
