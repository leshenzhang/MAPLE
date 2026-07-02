#!/usr/bin/env bash
# Build libcolvars.a (Colvars core, no CUDA / no Lepton / no host) and compile
# the pybind11 `colvars` module (colvarsmodule.cpp) against it, producing a
# `colvars<EXT_SUFFIX>.so` importable by MAPLE's bias/colvars_calc.py.
#
# HEAVY compile -> run on a compute node (sbatch/srun), NOT the login head.
#
# Env in (with defaults):
#   COLVARS_SRC   Colvars checkout root (contains src/)               [required]
#   PY            target python interpreter (its headers/ext-suffix used)
#   OUT           output dir for objects + libcolvars.a + colvars*.so
#   MODULE_CPP    path to colvarsmodule.cpp
#   JOBS          parallel compile jobs
set -euo pipefail

COLVARS_SRC="${COLVARS_SRC:?set COLVARS_SRC to the colvars checkout root}"
PY="${PY:?set PY to the target python}"
OUT="${OUT:?set OUT to an output dir}"
MODULE_CPP="${MODULE_CPP:?set MODULE_CPP to colvarsmodule.cpp}"
JOBS="${JOBS:-8}"
CXX="${CXX:-g++}"

SRC="${COLVARS_SRC}/src"
STUB="${COLVARS_SRC}/misc_interfaces/stubs"   # colvarproxy_stub.{h,cpp} (host-less proxy)
OBJ="${OUT}/obj"
LIB="${OUT}/libcolvars.a"
mkdir -p "${OBJ}"

echo "[build] CXX=$(${CXX} --version | head -1)"
echo "[build] Colvars src: ${SRC}"
echo "[build] stub proxy : ${STUB}"
echo "[build] compiling core objects (no CUDA, no Lepton) -> ${OBJ}"

# Compile every src/*.cpp PLUS the host-less stub proxy. The *_gpu.cpp files
# self-guard on COLVARS_CUDA/HIP (undefined here) and reduce to empty translation
# units. Lepton is disabled (no COLVARS_LEPTON), so customFunction CVs are
# unavailable (not needed for distance/harmonic/ABF/eABF/metadynamics).
CXXFLAGS="-O2 -std=c++17 -fPIC -fvisibility=hidden -I${SRC} -I${STUB}"
{ ls "${SRC}"/*.cpp; echo "${STUB}/colvarproxy_stub.cpp"; } | \
  xargs -P "${JOBS}" -I{} bash -c '
  f="{}"; o="'"${OBJ}"'/$(basename "$f" .cpp).o"
  '"${CXX}"' '"${CXXFLAGS}"' -c "$f" -o "$o"
'
echo "[build] archiving -> ${LIB}"
rm -f "${LIB}"
ar rcs "${LIB}" "${OBJ}"/*.o

# --- pybind11 module ---------------------------------------------------------
PYINC=$(${PY} -c 'import sysconfig; print(sysconfig.get_path("include"))')
PYBINDINC=$(${PY} -c 'import pybind11; print(pybind11.get_include())')
EXTSUF=$(${PY} -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
SO="${OUT}/colvars${EXTSUF}"

echo "[build] compiling pybind module -> ${SO}"
${CXX} -O2 -std=c++17 -fPIC -fvisibility=hidden -shared \
  -I"${SRC}" -I"${STUB}" -I"${PYINC}" -I"${PYBINDINC}" \
  "${MODULE_CPP}" "${LIB}" \
  -o "${SO}"

echo "[build] DONE: ${SO}"
${PY} -c "import sys; sys.path.insert(0, '${OUT}'); import colvars; c=colvars.Colvars(); print('[build] import OK, methods:', [m for m in dir(c) if not m.startswith('_')])"
