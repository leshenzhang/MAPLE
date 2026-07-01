# colvars_ext — host-less Colvars Python binding for MAPLE MD

`bias/colvars_calc.py` drives the Colvars CV/bias engine (harmonic restraint /
ABF / eABF / metadynamics) through a `colvars.Colvars()` object API. Upstream
`github.com/Colvars/colvars` ships **no** importable `colvars` package with that
API — only a ctypes scripting shim linked into a host NAMD/VMD executable. This
directory builds the missing binding: a pybind11 wrapper (`colvarsmodule.cpp`)
around Colvars' `colvarproxy_stub` (its host-less standalone proxy), driving
`colvarmodule` per-step from Python exactly as the upstream functional-test
harness (`tests/functional/run_colvars_test.cpp`) drives it from an XYZ file.

## What the binding exposes (`colvars.Colvars`)
`set_unit_system(str)`, `read_config_string(str)`, `set_output_prefix(str)`,
`set_timestep(dt_fs)`, `set_temperature(T_K)`, `set_step(int)`,
`set_positions((N,3) float64)`, `calc()` / `update()`,
`get_energy() -> float`, `get_forces() -> (N,3) float64`,
`write_output_files()`, `num_biases()` — the exact names/signatures
`colvars_calc.py::_init_colvars` probes.

- **Config format:** Colvars config is newline-delimited — a keyword's value
  runs to end-of-line, so top-level statements must be on separate lines (a
  single space-joined line makes `name` swallow the rest → `read_config_string`
  error code 5). See `_test_colvars.py::_restraint_cfg` for the layout.
- **eABF / extended-Lagrangian:** `set_timestep`/`set_temperature` set the
  stub proxy's `dt` (fs) and target `T` (K) — 0 by default — so eABF and
  extended-system metadynamics can integrate their fictitious DOF + thermostat.
  eABF is the ABF-family method that fits this position-only seam: the physical
  system feels only the bounded ξ↔λ harmonic coupling (applied exactly), and the
  CZAR free-energy estimate (`<prefix>.czar.grad`) needs no system total force.
  Plain force-based ABF is unstable here (no system force is pushed → ungrounded
  mean-force estimate → runaway); use eABF instead.
- **`set_output_prefix`:** syncs BOTH the proxy prefix and the module prefix
  (`cvmodule->output_prefix()`, which ABF/metaD re-read each step for grid
  filenames), so it works whether called before or after `read_config_string`.
  For a single PMF-computing bias, grid files are `<prefix>.count/.grad/.pmf`
  (no `<biasname>` infix).

- **Units:** Colvars runs in its `"real"` system (kcal/mol, Å). Positions are Å
  (no conversion). `get_energy`/`get_forces` are kcal/mol and kcal/mol/Å;
  `colvars_calc.py` converts to Hartree/(Ha·Å⁻¹) at the seam via `HA_TO_KCAL`.
- **Atom map:** only atoms referenced by `atomNumbers` (1-indexed) become Colvars
  slots. `set_positions` fills each slot from the full ASE array by global atom
  id; `get_forces` scatters per-slot applied forces back to a full (N,3) array.
- **Sign:** the returned applied force is `-dU_bias/dr` (the term to *add* to the
  MLIP force), matching `colvars_calc.py`'s additive fold.
- **PBC:** the stub proxy is non-periodic; `set_cell` is intentionally not
  exposed (periodic minimum-image is out of scope). Target = non-periodic /
  molecular / implicit-solvent CVs (restraint / metaD / (e)ABF).

## Build (HEAVY compile → compute node, not login head)
```bash
COLVARS_SRC=/ibex/user/xiaox/zls/ai-maple-md/build_colvars/colvars-master \
PY=/ibex/user/xiaox/zls/ai-maple-md/envs/plumed/bin/python \
OUT=/ibex/user/xiaox/zls/ai-maple-md/build_colvars/out \
MODULE_CPP=$(pwd)/colvarsmodule.cpp \
JOBS=16 bash build_colvars.sh
```
`build_colvars.sh` compiles `colvars-master/src/*.cpp` → `libcolvars.a` (no CUDA:
the `*_gpu.cpp` files self-guard on `COLVARS_CUDA`/`COLVARS_HIP` and reduce to
empty TUs; no Lepton: `customFunction` CVs unavailable, not needed here), then
links `colvarsmodule.cpp` → `colvars<EXT_SUFFIX>.so` in `$OUT`. Put `$OUT` on
`PYTHONPATH` to `import colvars`.

Colvars source (`build_colvars/`) and the compiled `.so` are **not** committed
(outside the repo tree / gitignored). Only `colvarsmodule.cpp` + `build_colvars.sh`
+ this note are versioned.

## Test
`bias/_test_colvars.py` auto-detects the binding: with it built (+ GPU + MACE) it
runs the REAL gates (restraint force vs analytic, ABF histogram accumulation,
cross-backend MACE-OFF/mace-mp-0); without it, only the lib-free seam gate runs.
Env: `PYTHONPATH=$OUT:<MAPLE root>`; run on a GPU node for the MLIP gates.

Re-clone Colvars (login node has network):
`git clone --depth 1 https://github.com/Colvars/colvars` (or fetch the
`master.tar.gz` tarball) into `build_colvars/colvars-master`.
