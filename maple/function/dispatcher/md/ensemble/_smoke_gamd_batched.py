"""Self-consistency smoke for batched GaMD + ParGaMD WE (run on an a100).

System: propane (C3H8) on MACE-OFF; 1-D CV = COM-COM distance between the two
terminal methyl CARBONS (tracks the soft C-C-C bend -> a single broad well that
plain MD samples fully, so the GaMD-reweight-vs-plain-MD agreement is a clean
internal-consistency gate, NOT a literature comparison; validation axis B-64).

GATE: reweighted GaMD PMF ~= plain-MD PMF on the SAME potential (max|dPMF|).
Also reports: boost force-factor in [0,1]; WE total-weight conservation +
walkers-per-bin flattening.
"""
import os, sys, tempfile, itertools
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from ase.build import molecule

from maple.function.calculator.mace._maceoff_batch_calculator import MaceOffBatchCalc
from maple.function.dispatcher.md.ensemble.gamd_batched import BatchedGaMD
from maple.function.dispatcher.md.ensemble.nvt_batched import BatchedNVT
from maple.function.dispatcher.md.bias.gamd import (
    gamd_reweight_1d, KB_HA_PER_K, HARTREE_PER_KCAL)

MODEL = os.path.expanduser("~/.cache/mace/MACE-OFF23_medium.model")
DEV = "cuda"

NWALK = int(os.environ.get("NWALK", "16"))
STEPS = int(os.environ.get("STEPS", "30000"))
PREP = int(os.environ.get("PREP", "5000"))
WE_STEPS = int(os.environ.get("WE_STEPS", "12000"))
WE_EVERY = int(os.environ.get("WE_EVERY", "1000"))
SEED = int(os.environ.get("SEED", "20260629"))
T = 300.0


def propane_cv_groups():
    at = molecule("C3H8")
    Z = at.get_atomic_numbers()
    pos = at.get_positions()
    carbons = [i for i, z in enumerate(Z) if z == 6]
    # the two terminal methyl carbons = the carbon pair with the LARGEST separation
    pairs = list(itertools.combinations(carbons, 2))
    d = [np.linalg.norm(pos[i] - pos[j]) for i, j in pairs]
    i, j = pairs[int(np.argmax(d))]
    return at, str(i), str(j)


def pooled_cv(bias, eq_frac=0.2):
    """Pool ALL logged CV frames across walkers (reference run; drop leading eq)."""
    cv = []
    for b in range(len(bias.cv_history)):
        s = np.asarray(bias.cv_history[b], dtype=np.float64)
        if s.size:
            s = s[int(eq_frac * s.size):]
        cv.append(s)
    return np.concatenate(cv)


def main():
    print(f"torch {torch.__version__} cuda {torch.cuda.is_available()} "
          f"dev {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"NWALK={NWALK} STEPS={STEPS} PREP={PREP} WE_STEPS={WE_STEPS} "
          f"WE_EVERY={WE_EVERY} SEED={SEED}")
    at, g1, g2 = propane_cv_groups()
    print(f"propane CV = COM-COM distance between atoms {g1} and {g2} (terminal C)")
    calc = MaceOffBatchCalc(model_path=MODEL, device=DEV, dtype=torch.float64)
    tmp = tempfile.mkdtemp(prefix="gamdsmoke_")

    common = dict(timestep=0.5, temperature=T, thermostat="langevin",
                  friction=0.02, remove_com_every=100, random_seed=SEED,
                  verbose=0, traj_every=10**9, log_every=10**9,
                  nwalkers=NWALK, cv_group1=g1, cv_group2=g2,
                  gamd_prep_steps=PREP, gamd_sigma0=6.0, gamd_mode="lower",
                  gamd_nbins=40)

    # ---- (A) plain reference: BatchedGaMD with boost OFF (passive CV logger) ----
    refp = dict(common, steps=STEPS, gamd="off", we="off")
    ref = BatchedGaMD(os.path.join(tmp, "ref.out"), at, calc=calc, paras=refp).run()
    cv_ref = pooled_cv(ref._bias)
    print(f"[REF]  frames={cv_ref.size}  CV mean={cv_ref.mean():.4f} "
          f"std={cv_ref.std():.4f}  range=[{cv_ref.min():.3f},{cv_ref.max():.3f}] A")

    # ---- (B) GaMD: boost ON, WE OFF -> reweight with EXISTING gamd_reweight_1d ----
    gp = dict(common, steps=STEPS, gamd="on", we="off")
    g = BatchedGaMD(os.path.join(tmp, "gamd.out"), at, calc=calc, paras=gp).run()
    cv_g, dv_g = g.production_samples()
    bp = g._bias.params
    print(f"[GaMD] params: mode={bp['mode']} k0={bp['k0']:.4f} k={bp['k']:.6g}/Ha "
          f"E={bp['E']:.6f} Ha  prod_frames={cv_g.size}")
    # boost force-factor range over production frames: factor = 1 - sqrt(2 k dV).
    kk = bp["k"]
    fac = 1.0 - np.sqrt(np.maximum(2.0 * kk * dv_g, 0.0))
    frac_boosted = float(np.mean(dv_g > 0.0))
    dV_kcal = dv_g / HARTREE_PER_KCAL
    print(f"[GaMD] boost factor range=[{fac.min():.4f},{fac.max():.4f}] "
          f"(must be in [0,1])  frames_boosted={frac_boosted*100:.1f}%  "
          f"<dV>={dV_kcal.mean():.3f} kcal/mol max={dV_kcal.max():.3f}")

    # ---- GATE: shared bins; compare reweighted GaMD PMF vs plain-MD PMF ----
    lo = float(np.percentile(cv_ref, 1.0))
    hi = float(np.percentile(cv_ref, 99.0))
    edges = np.linspace(lo, hi, common["gamd_nbins"] + 1)
    cen_ref, pmf_ref = gamd_reweight_1d(cv_ref, np.zeros_like(cv_ref),
                                        temperature=T, bins=edges, mode="ce2")
    cen_ce2, pmf_ce2 = gamd_reweight_1d(cv_g, dv_g, temperature=T, bins=edges,
                                        mode="ce2")
    cen_mac, pmf_mac = gamd_reweight_1d(cv_g, dv_g, temperature=T, bins=edges,
                                        mode="maclaurin")
    # well-sampled bins: ref has enough counts AND both PMFs finite.
    cnt_ref, _ = np.histogram(cv_ref, bins=edges)
    cnt_g, _ = np.histogram(cv_g, bins=edges)
    good = (cnt_ref >= max(50, cv_ref.size // (10 * common["gamd_nbins"]))) & \
           (cnt_g >= 20) & np.isfinite(pmf_ref) & np.isfinite(pmf_ce2) & \
           np.isfinite(pmf_mac)
    # re-zero each PMF on the common well-sampled support for a fair max|d|.
    def rezero(p):
        q = p.copy(); q[good] -= np.nanmin(q[good]); return q
    pr, p2, pm = rezero(pmf_ref), rezero(pmf_ce2), rezero(pmf_mac)
    d_ce2 = float(np.nanmax(np.abs(p2[good] - pr[good])))
    d_mac = float(np.nanmax(np.abs(pm[good] - pr[good])))
    print(f"[GATE] well-sampled bins={int(good.sum())}/{common['gamd_nbins']}  "
          f"PMF_ref range={float(np.nanmax(pr[good])):.3f} kcal/mol")
    print(f"[GATE] max|dPMF| GaMD(CE2)-vs-plainMD = {d_ce2:.4f} kcal/mol")
    print(f"[GATE] max|dPMF| GaMD(Maclaurin)-vs-plainMD = {d_mac:.4f} kcal/mol")
    for c, a, b2, c2 in zip(cen_ref[good], pr[good], p2[good], pm[good]):
        print(f"        CV={c:.3f}  plain={a:.3f}  ce2={b2:.3f}  mac={c2:.3f}")

    # ---- (C) ParGaMD WE: boost ON + WE ON -> weight conservation + flattening ----
    we_lo = float(np.percentile(cv_ref, 2.0))
    we_hi = float(np.percentile(cv_ref, 98.0))
    wp = dict(common, steps=WE_STEPS, gamd="on", we="on",
              we_resample_every=WE_EVERY, we_nbins=5,
              we_cv_min=we_lo, we_cv_max=we_hi)
    we = BatchedGaMD(os.path.join(tmp, "we.out"), at, calc=calc, paras=wp).run()
    print(f"[WE]   resamples={len(we._we_log)} over {WE_STEPS} steps  "
          f"bins=5 range=[{we_lo:.3f},{we_hi:.3f}] A")
    wmax = 0.0
    sflat = True
    for ev in we._we_log:
        dw = abs(ev["weight_after"] - 1.0)
        wmax = max(wmax, dw, abs(ev["weight_before"] - 1.0))
        sflat &= (ev["occ_std_after"] <= ev["occ_std_before"] + 1e-9)
        print(f"        step {ev['step']:>6}: n_occ={ev['n_occ']}  "
              f"Wtot {ev['weight_before']:.8f}->{ev['weight_after']:.8f}  "
              f"occ {ev['occ_before'].tolist()}->{ev['occ_after'].tolist()}  "
              f"occ_std {ev['occ_std_before']:.3f}->{ev['occ_std_after']:.3f}")
    print(f"[WE]   max|totalWeight-1|={wmax:.2e}  occupancy_flattened={sflat}")

    # ---- verdict ----
    gate_pass = (d_ce2 < 0.5) and (fac.min() >= -1e-9) and (fac.max() <= 1.0 + 1e-9)
    we_pass = (wmax < 1e-9) and sflat
    print(f"\n[RESULT] GATE(GaMD reweight self-consistency) "
          f"{'PASS' if gate_pass else 'FAIL'}  max|dPMF|_CE2={d_ce2:.4f} kcal/mol")
    print(f"[RESULT] WE(weight conservation + flattening) "
          f"{'PASS' if we_pass else 'FAIL'}  max|W-1|={wmax:.2e}")
    print("[RESULT] " + ("ALL PASS" if (gate_pass and we_pass) else "CHECK ABOVE"))


if __name__ == "__main__":
    main()
