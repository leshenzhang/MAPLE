"""
REMD gates (a100, pure MLIP MACE-OFF). Mirrors _test_nvt_batched.py methodology.

  MODE-GATE : REST2 / Hamiltonian-REMD rejected (temperature-REMD only).
  PARITY    : exchange_every=0 (no swaps) REMD === independent N-replica
              BatchedNVT, to fp64 machine precision (same B=N force engine,
              same RNG, same per-replica thermostat targets). Proves the REMD
              wrapper reduces to N independent NVT replicas when swaps are off
              (and the calc's perturb-one isolation == 0 Ha proves the N replicas
              in that batch are mutually independent).
  SMOKE     : N=4 geometric ladder ~300/354/418/493 K, langevin, exchange_every
              =100. Reports per-pair acceptance, per-slot measured T vs target,
              detailed-balance sanity (symmetric sweeps), 300 K replica ran,
              perturb-one isolation.
"""
import os, sys, tempfile
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from ase.build import molecule

from maple.function.calculator.mace._maceoff_batch_calculator import MaceOffBatchCalc
from maple.function.dispatcher.md.ensemble.nvt_batched import BatchedNVT
from maple.function.dispatcher.md.ensemble.remd import REMD

MODEL = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
DEV = "cuda"


def ethanol(shift=(0.0, 0.0, 0.0)):
    at = molecule("CH3CH2OH")
    at.positions = at.positions + np.asarray(shift)
    return at


def _tmp():
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
        return f.name


def run_mode_gate():
    a = ethanol()
    bc = MaceOffBatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64)
    for bad in ("rest2", "hamiltonian", "alchemical"):
        try:
            REMD(_tmp(), a, calc=bc,
                 paras=dict(n_replicas=4, temp_min=300, temp_max=493, mode=bad, steps=1))
        except NotImplementedError as e:
            assert "TEMPERATURE-REMD" in str(e)
        else:
            raise AssertionError(f"mode '{bad}' was NOT rejected")
    print("[MODE-GATE] REST2 / Hamiltonian / alchemical REMD rejected (ok)")
    return True


def run_parity(steps=120, seed=42, N=4, t_min=300.0, t_max=493.0):
    ladder = t_min * (t_max / t_min) ** (np.arange(N) / (N - 1))
    paras = dict(timestep=0.5, steps=steps, thermostat="langevin", friction=0.02,
                 remove_com_every=100, random_seed=seed, verbose=0)
    bc = MaceOffBatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64)

    # (1) REMD with swaps OFF (exchange_every=0): the pure independent path.
    a = ethanol()
    rp = dict(paras); rp.update(n_replicas=N, temp_min=t_min, temp_max=t_max,
                                exchange_every=0)
    sim = REMD(_tmp(), a, calc=bc, paras=rp).run()
    pos_remd = bc.coord.detach().to("cpu").numpy().copy()
    v_remd = sim.v.detach().to("cpu").numpy().copy()

    # (2) reference: plain BatchedNVT(B=N) with the SAME ladder applied by hand
    #     (init at t_min then rescale per replica) and NO swap code at all.
    bn_paras = dict(paras); bn_paras.update(temperature=t_min)
    reps = [ethanol() for _ in range(N)]
    bn = BatchedNVT(_tmp(), reps, calc=bc, paras=bn_paras)
    bn._log_parameters(); bn._prepare_buffers()
    for b in range(N):
        bn.v[b] = bn.v[b] * (float(ladder[b]) / t_min) ** 0.5
        bn._thermostats[b].set_temperature(float(ladder[b]))
    # mirror REMD._apply_ladder: with fused_loop default ON (B-156), the per-replica
    # ladder retuned each thermostat's ._c2, so the on-device _c2_dev (built at t_min in
    # _prepare_buffers) must be refreshed here too -- REMD does this inside _apply_ladder;
    # this hand-rolled reference must match it or it runs the stale t_min noise amplitude.
    if getattr(bn, "_fused", False) and bn.params.thermostat == "langevin":
        bn._refresh_c2_dev()
    bn._run_langevin(); bn._finalize()
    pos_ref = bc.coord.detach().to("cpu").numpy().copy()
    v_ref = bn.v.detach().to("cpu").numpy().copy()

    dpos = float(np.max(np.abs(pos_remd - pos_ref)))
    dvel = float(np.max(np.abs(v_remd - v_ref)))
    print(f"[PARITY] no-swap REMD vs independent BatchedNVT: steps={steps} N={N} "
          f"seed={seed}  max|dpos|={dpos:.3e} A  max|dvel|={dvel:.3e} au")
    assert dpos < 1e-10 and dvel < 1e-10, "PARITY FAIL (no-swap REMD != independent NVT)"
    print("[PARITY] PASS  (no-swap REMD === N independent BatchedNVT replicas)")
    return dpos, dvel


def run_smoke(steps=6000, seed=7, N=4, t_min=300.0, t_max=493.0, exchange_every=100):
    a = ethanol()
    bc = MaceOffBatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64)
    paras = dict(timestep=0.5, steps=steps, thermostat="langevin", friction=0.02,
                 remove_com_every=100, random_seed=seed, verbose=0,
                 n_replicas=N, temp_min=t_min, temp_max=t_max,
                 exchange_every=exchange_every)
    sim = REMD(_tmp(), a, calc=bc, paras=paras).run()

    leak = bc.isolation_check(perturb=0.1)               # cross-replica energy leak [Ha]
    ladder = sim.ladder
    print(f"\n[SMOKE] N={N}  ladder(K)={'/'.join(f'{t:.0f}' for t in ladder)}  "
          f"steps={steps}  dt=0.5 fs  exchange_every={exchange_every}  "
          f"isolation_dE={leak:.3e} Ha")

    print("  -- per ladder slot: measured kinetic T vs target (tail 50%) --")
    okT = True
    for s in sim.slot_results:
        d = s['T_measured'] - s['T_target']
        okT = okT and abs(d) < 0.20 * s['T_target']
        print(f"    T_target={s['T_target']:7.1f} K   <T>meas={s['T_measured']:7.2f} K "
              f"(sig {s['T_std']:5.1f})   d={d:+7.2f} K   steps_occ={s['steps_occupied']}")

    print("  -- per ladder-adjacent pair: swap acceptance --")
    rates = []
    for e in sim.exchange_stats:
        rates.append(e['rate'])
        print(f"    {e['T_lo']:6.1f}->{e['T_hi']:6.1f} K   attempts={e['attempts']:4d}  "
              f"accepts={e['accepts']:4d}  rate={e['rate']:.3f}")

    # detailed-balance sanity: symmetric even/odd sweeps, every boundary attempted.
    tot_sweeps = sim._swap_round
    even = (tot_sweeps + 1) // 2; odd = tot_sweeps // 2
    all_attempted = all(e['attempts'] > 0 for e in sim.exchange_stats)
    print(f"  -- detailed-balance sanity: swap sweeps={tot_sweeps} "
          f"(even={even}, odd={odd}), all {N-1} boundaries attempted={all_attempted}, "
          f"symmetric Metropolis (delta antisymmetric under i<->j) --")

    target_ran = (sim.target_slot_steps == sim.nstep)
    print(f"  -- 300 K (lowest rung) occupied every step: "
          f"{sim.target_slot_steps}/{sim.nstep} -> {target_ran}")

    assert leak < 1e-6, "ISOLATION FAIL"
    assert all_attempted and even > 0 and odd > 0, "SWEEP/DB SANITY FAIL"
    assert target_ran, "300 K replica did not run every step"
    assert okT, "PER-SLOT T off target (>20%)"
    assert all(r > 0.0 for r in rates), "a swap boundary never accepted"
    print("[SMOKE] PASS")
    return rates, [s['T_measured'] for s in sim.slot_results], leak


if __name__ == "__main__":
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    run_mode_gate()
    dpos, dvel = run_parity()
    rates, Tmeas, leak = run_smoke()
    print(f"\n[REMD RESULT] parity max|dpos|={dpos:.3e} A  max|dvel|={dvel:.3e} au")
    print(f"[REMD RESULT] per-pair acceptance={[round(r,3) for r in rates]}")
    print(f"[REMD RESULT] per-slot measured T(K)={[round(t,1) for t in Tmeas]}")
    print(f"[REMD RESULT] isolation_dE_Ha={leak:.3e}")
    print("[REMD RESULT] ALL GATES PASS")
