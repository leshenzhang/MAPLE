"""
C3 gates for the batched NVT kernel (run on an a100 node; pure MLIP MACE-OFF).

  PARITY  : B=1 BatchedNVT  ===  legacy single-system NVT  (v-rescale, fp64).
  SMOKE   : B=2 two-replica NVT (langevin), per-replica T -> ~300 K, plus the
            perturb-one ENERGY isolation gate (rep1 unchanged < 1e-6 Ha).
  BACKEND : B>1 with a charge-coupled (AIMNet2-native-like) calc is REJECTED.

The PARITY comparison drives BOTH paths from the SAME MACE-OFF force engine
(MaceOffBatchCalc, fp64) so forces are bit-identical; with matched VV arithmetic
and a shared RNG stream, B=1 reproduces the legacy NVT to machine precision.
"""
import os, sys, tempfile
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator.mace._maceoff_batch_calculator import MaceOffBatchCalc, EV2HARTREE
from maple.function.dispatcher.md.ensemble.nvt import NVT
from maple.function.dispatcher.md.ensemble.nvt_batched import BatchedNVT

MODEL = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
DEV = "cuda"


def ethanol(shift=(0.0, 0.0, 0.0)):
    at = molecule("CH3CH2OH")
    at.positions = at.positions + np.asarray(shift)
    return at


# ---- single-system ASE bridge: Hartree-returning MACE-OFF via MaceOffBatchCalc(B=1).
# Used ONLY to drive the legacy NVT with the IDENTICAL force engine the batched
# kernel uses, so the parity test isolates the integrator/thermostat math.
class _MaceOffSingle(Calculator):
    implemented_properties = ["energy", "forces", "free_energy"]

    def __init__(self, template_atoms, model_path=MODEL, device=DEV):
        super().__init__()
        self._bc = MaceOffBatchCalc(model_path=model_path, device=device, dtype=torch.float64)
        self._bc.prepare([template_atoms.copy()], fixed_nmax=None)
        self.atomic_numbers = self._bc.atomic_numbers
        self.device = self._bc.device

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(self.atoms)
        coord = torch.tensor(self.atoms.get_positions(), dtype=torch.float64, device=self._bc.device)
        self._bc.set_coords_(coord)
        E_Ha, F_Ha = self._bc.get_ef_gpu()                  # (1,), (1, nmax_dof) Ha, Ha/A
        E = float(E_Ha[0].item())
        F = F_Ha[0, :3 * n].detach().to("cpu").numpy().reshape(n, 3)   # Ha/A
        self.results = {"energy": E, "free_energy": E, "forces": F}


def run_parity(steps=200, seed=42):
    paras = dict(timestep=0.5, steps=steps, temperature=300.0, thermostat="v-rescale",
                 tau_t=100.0, remove_com_every=100, random_seed=seed, verbose=0)
    # legacy single-system NVT
    at_leg = ethanol()
    at_leg.calc = _MaceOffSingle(at_leg, device=DEV)
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        out_leg = f.name
    NVT(out_leg, at_leg, paras=paras).run()
    pos_leg = at_leg.get_positions()
    vel_leg = np.asarray(at_leg.arrays["velocities"])
    pe_leg = float(at_leg.get_potential_energy())           # Ha

    # batched B=1 through the kernel, same engine + seed
    at_bat = ethanol()
    bc = MaceOffBatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64)
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        out_bat = f.name
    sim = BatchedNVT(out_bat, [at_bat], calc=bc, paras=paras).run()
    n = len(at_bat)
    pos_bat = bc.coord.detach().to("cpu").numpy()
    vel_bat = sim.v[0, :3 * n].detach().to("cpu").numpy().reshape(n, 3)
    pe_bat = float(sim.results[0]["PE_Ha"][-1])

    dpos = float(np.max(np.abs(pos_leg - pos_bat)))
    dvel = float(np.max(np.abs(vel_leg - vel_bat)))
    dpe_eV = abs(pe_leg - pe_bat) * 27.211386245988
    print(f"[PARITY] steps={steps} seed={seed}  max|dpos|={dpos:.3e} A  "
          f"max|dvel|={dvel:.3e} au  |dPE|={dpe_eV:.3e} eV  (PE_leg={pe_leg:.8f} Ha)")
    assert dpos < 1e-6 and dvel < 1e-6 and dpe_eV < 1e-6, "PARITY FAIL"
    print("[PARITY] PASS")
    return dpe_eV


def run_smoke_b2(steps=1000, seed=7):
    a0 = ethanol(shift=(0, 0, 0))
    a1 = ethanol(shift=(60.0, 0, 0))          # far apart; block-diagonal anyway
    bc = MaceOffBatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64)
    # log_every=1: dense per-step T history (tight tail-mean estimate for a tiny
    # 9-atom system) -- also exercises the B-81 fix#3 cadence path at every-step.
    paras = dict(timestep=0.5, steps=steps, temperature=300.0, thermostat="langevin",
                 friction=0.02, remove_com_every=100, log_every=1,
                 random_seed=seed, verbose=0)
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        out = f.name
    sim = BatchedNVT(out, [a0, a1], calc=bc, paras=paras).run()
    # last-50% mean T (tight estimate for a tiny 9-atom system; per-step T has
    # huge canonical fluctuations at n_dof=24, so a wide window is needed).
    # B-81 fix#3: the recorded history is cadence-gated by log_every, so index the
    # tail by the RECORDED-frame count (len // 2), not the raw step count.
    nrec = len(sim.results[0]["T_K"])
    half = nrec // 2
    T0 = float(np.mean(sim.results[0]["T_K"][half:]))
    T1 = float(np.mean(sim.results[1]["T_K"][half:]))
    # perturb-one ENERGY isolation on the post-run batch state.
    leak = bc.isolation_check(perturb=0.1)    # max |dE| in OTHER replicas, Ha
    print(f"[SMOKE B=2] final T_tail: rep0={T0:.2f} K  rep1={T1:.2f} K  "
          f"(target 300 K)  perturb-one isolation dE={leak:.3e} Ha")
    assert leak < 1e-6, "ISOLATION FAIL (replicas coupled)"
    assert 150.0 < T0 < 500.0 and 150.0 < T1 < 500.0, "THERMOSTAT FAIL (T off target)"
    print("[SMOKE B=2] PASS")
    return T0, T1, leak


def run_backend_gate():
    class _FakeCoupled:                       # mimics a charge-coupled batch calc
        batch_isolated = False
        device = torch.device(DEV); dtype = torch.float64
        def prepare(self, *a, **k): pass
        def get_ef_gpu(self): pass
        def step_cart_(self, *a, **k): pass
    a0, a1 = ethanol(), ethanol(shift=(60, 0, 0))
    try:
        BatchedNVT("/dev/null", [a0, a1], calc=_FakeCoupled(), paras=dict(steps=1))
    except ValueError as e:
        assert "ISOLATED" in str(e)
        print("[BACKEND] B=2 charge-coupled calc REJECTED (ok):", str(e).split(".")[0])
        return True
    raise AssertionError("BACKEND gate did not reject a coupled calc")


if __name__ == "__main__":
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    run_backend_gate()
    dpe = run_parity()
    T0, T1, leak = run_smoke_b2()
    print(f"\n[C3 RESULT] parity_dPE_eV={dpe:.3e}  B2_T0={T0:.2f}K  B2_T1={T1:.2f}K  "
          f"isolation_dE_Ha={leak:.3e}")
    print("[C3 RESULT] ALL GATES PASS")
