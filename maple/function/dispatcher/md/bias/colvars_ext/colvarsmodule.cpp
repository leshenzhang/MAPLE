// colvarsmodule.cpp -- pybind11 wrapper exposing a minimal `colvars.Colvars`
// object API around Colvars' host-less `colvarproxy_stub`, so MAPLE's
// `bias/colvars_calc.py` can drive the REAL Colvars CV/bias engine (harmonic
// restraint / ABF / eABF / metadynamics) with no NAMD/VMD/LAMMPS host.
//
// Upstream github.com/Colvars/colvars ships NO importable `colvars` Python
// package exposing this object API (only a ctypes scripting shim linked into a
// host executable). This file BUILDS that binding: it drives `colvarmodule`
// through a `colvarproxy_stub` instance exactly as the standalone functional
// test harness (tests/functional/run_colvars_test.cpp) does, but per-step from
// Python instead of from an XYZ trajectory file.
//
// Unit system: run Colvars in its "real" system (kcal/mol, Angstrom). Positions
// are Angstrom (no conversion); bias energy is kcal/mol; applied forces are
// kcal/mol/Angstrom. colvars_calc.py converts energy/force back to Hartree at
// the seam (HA_TO_KCAL). This binding never sees the inner MLIP energy.
//
// Atom mapping: Colvars registers only the atoms referenced by the config
// (via atomNumbers, 1-indexed) as internal "slots". `set_positions` receives the
// FULL (N,3) ASE coordinate array and fills each slot from the position of its
// recorded global atom id; `get_forces` scatters the per-slot applied colvar
// forces back into a full (N,3) array (zero on unreferenced atoms). This makes
// the mapping robust to arbitrary atomNumbers ordering.
//
// PBC: the stub proxy runs non-periodic (boundaries_non_periodic). This binding
// therefore targets non-periodic CVs (gas-phase / molecular / implicit-solvent
// enzyme targets -- the restraint/metaD/eABF use case). `set_cell` is
// intentionally NOT exposed, so colvars_calc.py's periodic branch (_maybe)
// no-ops; a periodic minimum-image distance is out of scope for this stub build.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "colvarmodule.h"
#include "colvarproxy.h"
#include "colvarproxy_stub.h"
#include "colvartypes.h"

namespace py = pybind11;

namespace {

// Colvars returns 0 (COLVARS_OK) on success; any non-zero is an error code.
inline void check(int err, const char *what) {
  if (err != 0) {
    throw std::runtime_error(std::string("colvars: ") + what +
                             " failed (code " + std::to_string(err) + ")");
  }
}

}  // namespace

class Colvars {
public:
  Colvars() {
    // The stub ctor creates the colvarmodule, wires setup_input/setup_output
    // and calls setup() (non-periodic boundaries, it=0). See colvarproxy_stub.cpp.
    proxy_ = new colvarproxy_stub();
    cvm_ = proxy_->cvmodule;
    // Default to "real" (kcal/mol, Angstrom) so a caller that forgets to call
    // set_unit_system still gets the unit contract colvars_calc.py assumes.
    check(proxy_->set_unit_system("real", false), "set_unit_system(real)");
  }

  // Not deleting proxy_/cvm_ on purpose: colvarproxy ownership of the module is
  // subtle and this object lives for the whole (short) MD process; a clean
  // teardown is not worth a double-free risk here.
  ~Colvars() = default;

  // 1-arg form matching colvars_calc.py's probe: proxy.set_unit_system("real").
  void set_unit_system(const std::string &units) {
    check(proxy_->set_unit_system(units, false),
          ("set_unit_system(" + units + ")").c_str());
  }

  // Parse a complete Colvars configuration. Referenced atoms are registered as
  // internal slots here (atom groups call proxy->init_atom during parsing).
  void read_config_string(const std::string &conf) {
    check(cvm_->read_config_string(conf), "read_config_string");
    config_read_ = true;
  }

  void set_output_prefix(const std::string &prefix) {
    check(proxy_->set_output_prefix(prefix), "set_output_prefix");
    // The colvarmodule keeps its OWN prefix string (cvm_output_prefix), synced
    // from the proxy only inside setup_output(); it is NOT auto-refreshed when
    // the proxy prefix changes after setup. ABF/metadynamics build their grid
    // filenames from cvmodule->output_prefix() (re-read every step), so without
    // this line a prefix set after read_config_string is ignored and grids land
    // in the CWD with an empty prefix (".count"/".grad"/".pmf"). Sync it here so
    // set_output_prefix works whether called before OR after config read.
    cvm_->output_prefix() = prefix;
  }

  // Integration timestep (fs) and target temperature (K). Extended-Lagrangian
  // methods (eABF, extended-system metadynamics) integrate a fictitious DOF and
  // its thermostat, which need dt>0 and T>0; the stub proxy leaves both 0 until
  // set here. Units are fixed (fs / K) in Colvars regardless of unit_system, so
  // no conversion. No-ops for pure position-space biases (restraint / plain ABF).
  void set_timestep(double dt_fs) {
    check(proxy_->set_integration_timestep(dt_fs), "set_integration_timestep");
  }

  void set_temperature(double temperature_K) {
    check(proxy_->set_target_temperature(temperature_K), "set_target_temperature");
  }

  // MD step index (drives hill/output frequencies). colvars_calc.py owns the
  // counter and calls this once per step before calc().
  void set_step(long long step) { cvm_->it = static_cast<colvarmodule::step_number>(step); }

  // Fill each Colvars atom slot from the full (N,3) ASE coordinate array, using
  // the slot's recorded global atom id (0-based).
  void set_positions(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
      throw std::runtime_error("colvars.set_positions expects an (N,3) array");
    }
    natoms_ = static_cast<int>(buf.shape[0]);
    const double *p = static_cast<const double *>(buf.ptr);

    auto *pos = proxy_->modify_atom_positions();  // size == #referenced atoms
    const auto *ids = proxy_->get_atom_ids();
    for (std::size_t i = 0; i < pos->size(); ++i) {
      const int aid = (*ids)[i];  // 0-based global id
      if (aid < 0 || aid >= natoms_) {
        throw std::runtime_error(
            "colvars config references atom " + std::to_string(aid + 1) +
            " but only " + std::to_string(natoms_) + " atoms were provided");
      }
      (*pos)[i] = cvm::rvector(p[3 * aid + 0], p[3 * aid + 1], p[3 * aid + 2]);
    }
  }

  // Zero the applied-force buffer, then run one Colvars step (colvars + biases).
  void calc() {
    if (!config_read_) {
      throw std::runtime_error("colvars.calc() called before read_config_string()");
    }
    auto *nf = proxy_->modify_atom_applied_forces();
    std::fill(nf->begin(), nf->end(), cvm::rvector(0.0, 0.0, 0.0));
    check(cvm_->calc(), "calc");
  }

  // Alias so colvars_calc.py's ("calc","compute","update") probe always hits.
  void update() { calc(); }

  // Summed bias energy in current units (kcal/mol under "real"). Set by calc().
  double get_energy() const { return cvm_->total_bias_energy; }

  // Scatter per-slot applied colvar forces to a full (N,3) array (Ha-free;
  // kcal/mol/Angstrom under "real"). Applied force == -dU_bias/dr, the term to
  // add to the MD/MLIP force -- matching colvars_calc.py's additive fold.
  py::array_t<double> get_forces() const {
    py::array_t<double> out({static_cast<py::ssize_t>(natoms_),
                             static_cast<py::ssize_t>(3)});
    auto buf = out.request();
    double *o = static_cast<double *>(buf.ptr);
    std::fill(o, o + static_cast<std::size_t>(natoms_) * 3, 0.0);

    const auto *nf = proxy_->get_atom_applied_forces();
    const auto *ids = proxy_->get_atom_ids();
    for (std::size_t i = 0; i < nf->size(); ++i) {
      const int aid = (*ids)[i];
      const cvm::rvector &f = (*nf)[i];
      o[3 * aid + 0] += f.x;
      o[3 * aid + 1] += f.y;
      o[3 * aid + 2] += f.z;
    }
    return out;
  }

  // Flush ABF/metadynamics grids (.grad/.count/.pmf/.hills) to output_prefix.
  void write_output_files() { check(cvm_->write_output_files(), "write_output_files"); }

  int num_biases() const { return static_cast<int>(cvm_->biases.size()); }

private:
  colvarproxy_stub *proxy_ = nullptr;
  colvarmodule *cvm_ = nullptr;
  int natoms_ = 0;
  bool config_read_ = false;
};

PYBIND11_MODULE(colvars, m) {
  m.doc() =
      "Host-less Colvars object binding (pybind11 around colvarproxy_stub) for "
      "MAPLE MD enhanced sampling. Run Colvars in 'real' units (kcal/mol, "
      "Angstrom); non-periodic CVs only.";

  py::class_<Colvars>(m, "Colvars")
      .def(py::init<>())
      .def("set_unit_system", &Colvars::set_unit_system, py::arg("units"))
      .def("read_config_string", &Colvars::read_config_string, py::arg("conf"))
      .def("set_output_prefix", &Colvars::set_output_prefix, py::arg("prefix"))
      .def("set_timestep", &Colvars::set_timestep, py::arg("dt_fs"))
      .def("set_temperature", &Colvars::set_temperature, py::arg("temperature_K"))
      .def("set_step", &Colvars::set_step, py::arg("step"))
      .def("set_positions", &Colvars::set_positions, py::arg("positions"))
      .def("calc", &Colvars::calc)
      .def("update", &Colvars::update)
      .def("get_energy", &Colvars::get_energy)
      .def("get_forces", &Colvars::get_forces)
      .def("write_output_files", &Colvars::write_output_files)
      .def("num_biases", &Colvars::num_biases);
}
