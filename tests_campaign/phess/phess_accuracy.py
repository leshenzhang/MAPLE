#!/usr/bin/env python
"""A2-phess accuracy harness: how much Hessian fidelity does each cheap
partial/approximate-Hessian scheme cost, measured against the EXACT central-FD
Hessian (the production oracle)?

Four independent probes (select with --probe, default all):

  fd      forward-FD vs central-FD Hessian at the reference TS geometries.
          cost: 3m+1 vs 6m force evaluations per structure.
  update  Bofill / PSB / SR1 / BFGS quasi-Newton updates: start from the exact
          Hessian at frame 0 of a REAL update-only P-RFO trajectory, apply k
          updates with that trajectory's own (s, dg), compare to the exact
          Hessian at frame k.  cost: 0 force evaluations per update.
          Bofill: J. Comput. Chem. 1994, 15, 1  (DOI 10.1002/jcc.540150102)
  lanczos matrix-free leftmost eigenpair (Lanczos on FD HVPs) vs the exact
          Hessian's leftmost eigenpair, cold start vs warm start (previous
          geometry's exact mode).  cost: 1 batched forward per Lanczos step.
          Sella: JCTC 2019, 15, 6536 (DOI 10.1021/acs.jctc.9b00869)
  core    core-region (movable-subset) partial Hessian as a function of the
          selection radius around the reaction centre, vs the full Hessian.
          cost: 6*m_core+1 vs 6*n+1.

Every metric is reported in the MASS-WEIGHTED frame (that is where n_imag and
the reaction mode live).  Nothing here changes the production math: the probes
call the shipped calculator / BPRFO code paths.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from ase import Atoms

HA = 1.0
NEG_CUT = 1e-3          # mass-weighted eigenvalue cut separating imag mode from ~0 rot/trans


# --------------------------------------------------------------------------- utils
def mass_weight(H, masses):
    D = 1.0 / np.sqrt(np.repeat(masses, 3))
    Hm = H * np.outer(D, D)
    return 0.5 * (Hm + Hm.T)


def spectrum(Hmw):
    ev, V = np.linalg.eigh(Hmw)
    return ev, V


def n_imag(ev):
    return int((ev < -NEG_CUT).sum())


def compare(Href_mw, Htst_mw, k_low=6):
    """Fidelity metrics of a test Hessian vs the reference, both mass-weighted."""
    ev_r, V_r = spectrum(Href_mw)
    ev_t, V_t = spectrum(Htst_mw)
    ov = abs(float(V_r[:, 0] @ V_t[:, 0]))
    n = min(k_low, len(ev_r))
    return dict(
        lam0_ref=float(ev_r[0]), lam0_tst=float(ev_t[0]),
        d_lam0=float(ev_t[0] - ev_r[0]),
        rel_d_lam0=float(abs(ev_t[0] - ev_r[0]) / max(abs(ev_r[0]), 1e-12)),
        max_d_ev_low=float(np.abs(ev_t[:n] - ev_r[:n]).max()),
        mode0_overlap=ov,
        nimag_ref=n_imag(ev_r), nimag_tst=n_imag(ev_t),
        fro_rel=float(np.linalg.norm(Htst_mw - Href_mw) / max(np.linalg.norm(Href_mw), 1e-12)),
    )


def load_cases(pkl, k, min_atoms=0):
    import pickle
    data = pickle.load(open(pkl, "rb"))
    out = []
    for d in data:
        Z = np.asarray(d["atomic_numbers"], dtype=int)
        if len(Z) < min_atoms:
            continue
        out.append(dict(
            idx=int(d.get("index", -1)),
            Z=Z,
            ts=np.asarray(d["transition_state"]["positions"], dtype=float),
            react=np.asarray(d["reactant"]["positions"], dtype=float),
        ))
        if len(out) >= k:
            break
    return out


def make_calc(model, dev, fd_mode="central", delta=2e-3):
    from maple.function.calculator.uma._uma_batch_calculator import UMABatchCalc
    return UMABatchCalc(model, device=dev, dtype=torch.float64, task="omol",
                        fd_mode=fd_mode, hessian_delta=delta)


def exact_H(calc, atoms_list, movable_masks=None):
    """Batched exact FD Hessian -> list of (3n_b, 3n_b) numpy float64 (Ha/A^2)."""
    calc.prepare(atoms_list)
    E, F, H, P = calc.get_efh_gpu(movable_masks=movable_masks)
    out = []
    for b, at in enumerate(atoms_list):
        n = 3 * len(at)
        Hb = H[b, :n, :n].detach().cpu().numpy().astype(float)
        out.append(0.5 * (Hb + Hb.T))
    g = []
    for b, at in enumerate(atoms_list):
        n = 3 * len(at)
        g.append(-F[b, :n].detach().cpu().numpy().astype(float))
    return out, g, [float(x) for x in E.detach().cpu().numpy()]


def ge_hessian(natoms_list, fd_mode="central", movable=None):
    """gradient-equivalents of one batched FD Hessian (1 GE = 1 structure force eval)."""
    mult = 3 if fd_mode == "forward" else 6
    if movable is None:
        movable = natoms_list
    return int(sum(mult * m for m in movable) + len(natoms_list))


# --------------------------------------------------------------------------- probe: fd
def probe_fd(args, cases, out):
    """forward-FD vs central-FD Hessian at the reference TS."""
    dev = args.device
    atoms = [Atoms(numbers=c["Z"], positions=c["ts"]) for c in cases]
    masses = [a.get_masses() for a in atoms]

    rows = []
    for delta in args.fd_deltas:
        calc_c = make_calc(args.model, dev, "central", delta)
        t0 = time.time()
        Hc, _, _ = exact_H(calc_c, atoms)
        t_c = time.time() - t0
        del calc_c
        torch.cuda.empty_cache() if dev == "cuda" else None

        calc_f = make_calc(args.model, dev, "forward", delta)
        t0 = time.time()
        Hf, _, _ = exact_H(calc_f, atoms)
        t_f = time.time() - t0
        del calc_f
        torch.cuda.empty_cache() if dev == "cuda" else None

        for b, c in enumerate(cases):
            m = compare(mass_weight(Hc[b], masses[b]), mass_weight(Hf[b], masses[b]))
            m.update(case=c["idx"], natoms=len(c["Z"]), delta=delta)
            rows.append(m)
        nat = [len(c["Z"]) for c in cases]
        out.setdefault("fd_cost", []).append(dict(
            delta=delta, wall_central_s=t_c, wall_forward_s=t_f,
            ge_central=ge_hessian(nat, "central"), ge_forward=ge_hessian(nat, "forward"),
        ))
        print(f"[fd] delta={delta}: central {t_c:.1f}s  forward {t_f:.1f}s "
              f"({t_c / max(t_f, 1e-9):.2f}x)", flush=True)
    out["fd"] = rows
    _summ_fd(rows)


def _summ_fd(rows):
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[r["delta"]].append(r)
    print(f"{'delta':>8} {'n':>4} {'|dlam0|max':>11} {'rel_dlam0':>10} "
          f"{'ovl_min':>8} {'nimag_ok':>9} {'fro_rel_med':>12}", flush=True)
    for d, rs in sorted(by.items()):
        print(f"{d:>8} {len(rs):>4} {max(abs(r['d_lam0']) for r in rs):>11.3e} "
              f"{np.median([r['rel_d_lam0'] for r in rs]):>10.3e} "
              f"{min(r['mode0_overlap'] for r in rs):>8.5f} "
              f"{sum(r['nimag_ref'] == r['nimag_tst'] for r in rs)}/{len(rs):<7} "
              f"{np.median([r['fro_rel'] for r in rs]):>12.3e}", flush=True)


# --------------------------------------------------------------------------- probe: update
def _apply_update(kind, H, s, y):
    """One quasi-Newton update on a single (n,n) numpy Hessian. Calls the SHIPPED
    batched kernels with B=1 so the tested formula is the production one."""
    from maple.function.dispatcher.ts.algorithm.BPRFO import BatchPRFO
    n = H.shape[0]
    dev = "cpu"
    Ht = torch.tensor(H, dtype=torch.float64, device=dev).unsqueeze(0)
    st = torch.tensor(s, dtype=torch.float64, device=dev).unsqueeze(0)
    gp = torch.zeros((1, n), dtype=torch.float64, device=dev)
    gn = torch.tensor(y, dtype=torch.float64, device=dev).unsqueeze(0)  # g_new-g_prev = y
    rm = torch.ones((1, n), dtype=torch.bool, device=dev)
    acc = torch.ones((1,), dtype=torch.bool, device=dev)
    if kind == "bfgs":
        Hn = BatchPRFO._bfgs_update_batched(H=Ht, s_cart=st, g_prev=gp, g_new=gn,
                                            real_mask=rm, step_accepted=acc)
    else:
        phi = {"bofill": None, "psb": 1.0, "sr1": 0.0}[kind]
        Hn = BatchPRFO._bofill_update_batched(H=Ht, s_cart=st, g_prev=gp, g_new=gn,
                                              real_mask=rm, step_accepted=acc,
                                              phi_override=phi)
    return Hn[0].cpu().numpy()


def _read_traj(path):
    if not os.path.exists(path):
        return []
    lines = open(path).read().split("\n")
    frames, i = [], 0
    while i < len(lines):
        if lines[i].strip() == "":
            i += 1
            continue
        try:
            nat = int(lines[i].strip())
        except ValueError:
            break
        if i + 2 + nat > len(lines):
            break
        c = [[float(x) for x in lines[i + 2 + k].split()[1:4]] for k in range(nat)]
        frames.append(np.asarray(c))
        i += 2 + nat
    return frames


def probe_update(args, cases, out):
    """Fidelity of quasi-Newton updates along a REAL update-only P-RFO trajectory."""
    from maple.function.dispatcher.ts.algorithm import BatchPRFO
    from maple.function.utility.molecules import Molecules
    dev = args.device
    calc = make_calc(args.model, dev, "central", 2e-3)

    rng = np.random.RandomState(args.seed)
    starts = [c["ts"] + rng.randn(*c["ts"].shape) * args.sigma for c in cases]
    atoms0 = [Atoms(numbers=c["Z"], positions=starts[b].copy())
              for b, c in enumerate(cases)]

    # Generate the step sequence with an UPDATE-ONLY run (exact Hessian only at
    # iter 1) -- exactly the trajectory a long-recalc production run would take.
    odir = os.path.join(args.out_dir, "update_traj")
    os.makedirs(odir, exist_ok=True)
    mols = Molecules(atoms0)
    mols.calc = calc
    bp = BatchPRFO(output=os.path.join(odir, "run.out"), device=dev,
                   max_outer_iter=args.update_steps, recalc=10 ** 6,
                   hessian_update="bofill")
    bp.run(mols)

    rows = []
    for b, c in enumerate(cases):
        frames = _read_traj(os.path.join(odir, f"ts_batch{b + 1}.xyz"))
        frames = frames[:args.update_steps + 1]
        if len(frames) < 3:
            print(f"[update] case {c['idx']}: only {len(frames)} frames, skip", flush=True)
            continue
        al = [Atoms(numbers=c["Z"], positions=f) for f in frames]
        Hs, gs, _ = exact_H(calc, al)                    # exact H + g at every frame
        masses = al[0].get_masses()
        H_upd = {k: Hs[0].copy() for k in args.updates}
        for k in range(1, len(frames)):
            s = (frames[k] - frames[k - 1]).reshape(-1)
            y = gs[k] - gs[k - 1]
            ref_mw = mass_weight(Hs[k], masses)
            for kind in args.updates:
                # secant residual q = ||y - H s|| / ||y|| measured with the PRE-update
                # Hessian: the zero-cost staleness signal that drives the
                # hessian_recalc_quality trigger. Recorded next to the TRUE error so
                # the gate can be calibrated (and falsified) on real data.
                q = float(np.linalg.norm(y - H_upd[kind] @ s)
                          / max(np.linalg.norm(y), 1e-12))
                H_upd[kind] = _apply_update(kind, H_upd[kind], s, y)
                m = compare(ref_mw, mass_weight(H_upd[kind], masses))
                m.update(case=c["idx"], natoms=len(c["Z"]), update=kind, k=k,
                         step_norm=float(np.linalg.norm(s)), secant_resid=q)
                rows.append(m)
        print(f"[update] case {c['idx']} n={len(c['Z'])} frames={len(frames)}", flush=True)
    out["update"] = rows
    _summ_update(rows, args.updates)


def _summ_update(rows, kinds):
    # correlation between the zero-cost staleness signal and the true eigen error
    for kind in kinds:
        rs = [r for r in rows if r["update"] == kind]
        if len(rs) > 3:
            q = np.array([r["secant_resid"] for r in rs])
            e = np.array([r["rel_d_lam0"] for r in rs])
            o = np.array([1.0 - r["mode0_overlap"] for r in rs])
            with np.errstate(invalid="ignore"):
                cq = float(np.corrcoef(q, e)[0, 1])
                co = float(np.corrcoef(q, o)[0, 1])
            print(f"[calib] {kind}: corr(secant_resid, rel_dlam0)={cq:.3f}  "
                  f"corr(secant_resid, 1-overlap)={co:.3f}  "
                  f"q med={np.median(q):.3f} p90={np.percentile(q, 90):.3f}", flush=True)
    print(f"{'update':>8} {'k':>3} {'n':>4} {'med rel_dlam0':>14} {'min ovl':>9} "
          f"{'nimag_ok':>9} {'med fro_rel':>12} {'med q':>8}", flush=True)
    ks = sorted(set(r["k"] for r in rows))
    for kind in kinds:
        for k in ks:
            rs = [r for r in rows if r["update"] == kind and r["k"] == k]
            if not rs:
                continue
            print(f"{kind:>8} {k:>3} {len(rs):>4} "
                  f"{np.median([r['rel_d_lam0'] for r in rs]):>14.3e} "
                  f"{min(r['mode0_overlap'] for r in rs):>9.5f} "
                  f"{sum(r['nimag_ref'] == r['nimag_tst'] for r in rs)}/{len(rs):<7} "
                  f"{np.median([r['fro_rel'] for r in rs]):>12.3e} "
                  f"{np.median([r['secant_resid'] for r in rs]):>8.3f}", flush=True)


# --------------------------------------------------------------------------- probe: lanczos
def probe_lanczos(args, cases, out):
    """Matrix-free leftmost eigenpair: accuracy + HVP cost, cold vs warm start."""
    from maple.function.dispatcher.ts.algorithm import BatchPRFO
    dev = args.device
    calc = make_calc(args.model, dev, "central", 2e-3)

    rng = np.random.RandomState(args.seed)
    # x0 = perturbed TS (previous iteration), x1 = x0 + a typical P-RFO step
    x0 = [c["ts"] + rng.randn(*c["ts"].shape) * args.sigma for c in cases]
    x1 = [p + rng.randn(*p.shape) * args.lanczos_step for p in x0]
    a0 = [Atoms(numbers=c["Z"], positions=x0[b]) for b, c in enumerate(cases)]
    a1 = [Atoms(numbers=c["Z"], positions=x1[b]) for b, c in enumerate(cases)]

    H0, _, _ = exact_H(calc, a0)
    H1, g1, _ = exact_H(calc, a1)
    masses = [a.get_masses() for a in a1]
    ev1 = [spectrum(mass_weight(H1[b], masses[b])) for b in range(len(cases))]
    ev0 = [spectrum(mass_weight(H0[b], masses[b])) for b in range(len(cases))]

    # Root-cause metric for the Krylov path: is the leftmost eigenvalue extremal in
    # MAGNITUDE?  spread = lam_max / |lam_min| >> 1 means it is NOT, and a short
    # Krylov space will resolve the stiff end first.
    out["spectrum_spread"] = [
        dict(case=cases[b]["idx"], natoms=len(cases[b]["Z"]),
             lam_min=float(ev1[b][0][0]), lam_max=float(ev1[b][0][-1]),
             spread=float(ev1[b][0][-1] / max(abs(ev1[b][0][0]), 1e-12)),
             gap_lam1=float(ev1[b][0][1] - ev1[b][0][0]))
        for b in range(len(cases))]
    sp = [r["spread"] for r in out["spectrum_spread"]]
    print(f"[lanczos] |lam_max/lam_min| median={np.median(sp):.1f} "
          f"min={min(sp):.1f} max={max(sp):.1f}", flush=True)

    rows = []
    solvers = [("lanczos", m) for m in args.lanczos_m] + \
              [("lobpcg", m) for m in args.lobpcg_iters]
    for solver, m_max in solvers:
        for warm in (False, True):
            calc.prepare(a1)
            E, F = calc.get_ef_gpu()
            nmax = int(F.shape[1])
            bp = BatchPRFO(output=os.path.join(args.out_dir, "lanczos.log"), device=dev,
                           hessian_mode="iterative", iter_lanczos_m=m_max,
                           iter_solver=solver, iter_lobpcg_max=m_max,
                           iter_precond=args.precond,
                           iter_gamma=args.lanczos_gamma, iter_fd_eta=args.lanczos_eta,
                           iter_warm_start=warm)
            bp._nmax = nmax
            bp._arange_n = torch.arange(nmax, device=bp.device)
            bp._orig_index = torch.arange(len(a1), device=bp.device)
            bp._rebuild_topology(a1)
            bp._open_log()
            real_mask = bp._real_mask
            g_cart = -F.to(torch.float64) * real_mask.to(torch.float64)
            g_mw = bp._D * g_cart
            if warm:
                # previous geometry's EXACT leftmost mass-weighted mode, padded
                tv = torch.zeros((len(a1), nmax), dtype=torch.float64, device=bp.device)
                for b in range(len(a1)):
                    n = 3 * len(a1[b])
                    tv[b, :n] = torch.tensor(ev0[b][1][:, 0], dtype=torch.float64,
                                             device=bp.device)
                bp.tracked_mode_vec_mw = tv
            t0 = time.time()
            if solver == "lobpcg":
                # preconditioner = the model Hessian a production run would carry.
                # Realistic surrogate: the EXACT Hessian at the PREVIOUS geometry x0
                # (that is what a fresh recalc / a Bofill-updated H_work approximates).
                Hm = torch.zeros((len(a1), nmax, nmax), dtype=torch.float64,
                                 device=bp.device)
                for b in range(len(a1)):
                    n = 3 * len(a1[b])
                    Hm[b, :n, :n] = torch.tensor(mass_weight(H0[b], masses[b]),
                                                 dtype=torch.float64, device=bp.device)
                lam, vec = bp._leftmost_eigpair_lobpcg(calc, g_mw, g_cart, real_mask,
                                                       H_mw_model=Hm)
            else:
                lam, vec = bp._leftmost_eigpairs_mw(calc, g_mw, g_cart, real_mask, 1)
            torch.cuda.synchronize() if dev == "cuda" else None
            dt = time.time() - t0
            bp._close_log()
            for b, c in enumerate(cases):
                n = 3 * len(c["Z"])
                v = vec[b, :n, 0].detach().cpu().numpy()
                lam_t = float(lam[b, 0].item())
                lam_r = float(ev1[b][0][0])
                v_r = ev1[b][1][:, 0]
                rows.append(dict(
                    case=c["idx"], natoms=len(c["Z"]), solver=solver,
                    m_max=m_max, warm=warm,
                    lam_ref=lam_r, lam_tst=lam_t, d_lam=lam_t - lam_r,
                    rel_d_lam=abs(lam_t - lam_r) / max(abs(lam_r), 1e-12),
                    mode0_overlap=abs(float(v @ v_r)),
                ))
            steps = bp._lanczos_steps
            ge = bp._ge_hvp
            out.setdefault("lanczos_cost", []).append(dict(
                solver=solver, m_max=m_max, warm=warm, lanczos_steps=steps, ge_hvp=ge,
                unconverged=bp._lanczos_unconverged, wall_s=dt, B=len(a1),
                ge_full_hessian=ge_hessian([len(c["Z"]) for c in cases], "central"),
            ))
            print(f"[{solver}] m<={m_max} warm={warm}: steps={steps} ge_hvp={ge} "
                  f"unconv={bp._lanczos_unconverged} {dt:.1f}s", flush=True)
    out["lanczos"] = rows
    _summ_lanczos(rows)


def _summ_lanczos(rows):
    print(f"{'solver':>8} {'m_max':>6} {'warm':>5} {'n':>4} {'med rel_dlam':>13} "
          f"{'med ovl':>9} {'min ovl':>9} {'ovl>=0.99':>10}", flush=True)
    for solv in sorted(set(r["solver"] for r in rows)):
        for m in sorted(set(r["m_max"] for r in rows if r["solver"] == solv)):
            for w in (False, True):
                rs = [r for r in rows
                      if r["solver"] == solv and r["m_max"] == m and r["warm"] == w]
                if not rs:
                    continue
                print(f"{solv:>8} {m:>6} {str(w):>5} {len(rs):>4} "
                      f"{np.median([r['rel_d_lam'] for r in rs]):>13.3e} "
                      f"{np.median([r['mode0_overlap'] for r in rs]):>9.5f} "
                      f"{min(r['mode0_overlap'] for r in rs):>9.5f} "
                      f"{sum(r['mode0_overlap'] >= 0.99 for r in rs)}/{len(rs):<8}",
                      flush=True)


# --------------------------------------------------------------------------- probe: synth
def probe_synth(args, cases, out):
    """Solver-only unit check: EXACT matrix-vector products, no calculator.

    Separates 'the eigensolver algebra is wrong' from 'the FD HVP is noisy' and
    from 'the leftmost eigenvalue is hard for a Krylov space'. The synthetic
    spectra mimic a mass-weighted molecular Hessian: ONE small negative
    eigenvalue plus a positive bulk spanning ~3 decades.
    """
    from maple.function.dispatcher.ts.algorithm import BatchPRFO
    dev = "cpu"
    torch.manual_seed(7)
    B, n = 8, 30
    rows = []
    for spread in args.synth_spreads:
        lam = torch.zeros(B, n, dtype=torch.float64)
        lam[:, 0] = -0.05
        bulk = torch.logspace(np.log10(0.05), np.log10(0.05 * spread), n - 1,
                              dtype=torch.float64)
        lam[:, 1:] = bulk.unsqueeze(0)
        Q = torch.linalg.qr(torch.randn(B, n, n, dtype=torch.float64))[0]
        A = Q @ torch.diag_embed(lam) @ Q.transpose(-1, -2)
        A = 0.5 * (A + A.transpose(-1, -2))
        ev, V = torch.linalg.eigh(A)

        for solver, m in ([("lanczos", m) for m in args.lanczos_m]
                          + [("lobpcg", m) for m in args.lobpcg_iters]):
            bp = BatchPRFO(output=os.path.join(args.out_dir, "synth.log"), device=dev,
                           hessian_mode="iterative", iter_lanczos_m=m,
                           iter_solver=solver, iter_lobpcg_max=m,
                           iter_precond=args.precond, iter_gamma=args.lanczos_gamma)
            bp._nmax = n
            bp._arange_n = torch.arange(n)
            bp._D = torch.ones(B, n, dtype=torch.float64)   # already mass-weighted
            bp._real_mask = torch.ones(B, n, dtype=torch.bool)
            bp._real_mask_f = bp._real_mask.to(torch.float64)
            bp._B = B
            bp._open_log()
            # exact operator in place of the FD HVP
            bp._hvp_cart = lambda calc, u, g0, rm, _A=A: torch.einsum("bij,bj->bi", _A, u)
            rm = bp._real_mask
            g = torch.zeros(B, n, dtype=torch.float64)
            if solver == "lobpcg":
                # preconditioner = a PERTURBED copy of A (what a stale model Hessian is)
                Ap = A + 0.2 * torch.diag_embed(torch.rand(B, n, dtype=torch.float64))
                lamx, vec = bp._leftmost_eigpair_lobpcg(None, g, g, rm, H_mw_model=Ap)
            else:
                lamx, vec = bp._leftmost_eigpairs_mw(None, g, g, rm, 1)
            bp._close_log()
            ovl = (vec[:, :, 0] * V[:, :, 0]).sum(-1).abs()
            rows.append(dict(
                solver=solver, m=m, spread=spread,
                lam_ref=float(ev[0, 0]), lam_med=float(lamx[:, 0].median()),
                rel_d_lam=float(((lamx[:, 0] - ev[:, 0]).abs()
                                 / ev[:, 0].abs()).median()),
                ovl_med=float(ovl.median()), ovl_min=float(ovl.min()),
                hvps=int(bp._ge_hvp) if bp._ge_hvp else int(bp._lanczos_steps * B),
            ))
            print(f"[synth] spread={spread:>6.0f} {solver:>8} m={m:>2} "
                  f"rel_dlam={rows[-1]['rel_d_lam']:.3e} ovl_med={rows[-1]['ovl_med']:.5f} "
                  f"ovl_min={rows[-1]['ovl_min']:.5f}", flush=True)
    out["synth"] = rows


# --------------------------------------------------------------------------- probe: core
def probe_core(args, cases, out):
    """Core-region (movable subset) partial Hessian vs the full Hessian, by radius."""
    dev = args.device
    calc = make_calc(args.model, dev, "central", 2e-3)
    atoms = [Atoms(numbers=c["Z"], positions=c["ts"]) for c in cases]
    masses = [a.get_masses() for a in atoms]
    Hfull, _, _ = exact_H(calc, atoms)
    full_mw = [mass_weight(Hfull[b], masses[b]) for b in range(len(cases))]
    full_sp = [spectrum(H) for H in full_mw]

    # reaction centre = atom with the largest reactant->TS displacement
    centres = []
    for c in cases:
        d = np.linalg.norm(c["ts"] - c["react"], axis=1)
        centres.append(int(np.argmax(d)))

    rows = []
    for R in args.core_radii:
        movable = []
        for b, c in enumerate(cases):
            d = np.linalg.norm(c["ts"] - c["ts"][centres[b]], axis=1)
            mv = [int(i) for i in np.nonzero(d <= R)[0]]
            if len(mv) < 2:
                mv = list(np.argsort(d)[:2].astype(int))
            movable.append(mv)
        t0 = time.time()
        calc.prepare(atoms)
        E, F, H, P = calc.get_efh_gpu(movable_masks=movable)
        torch.cuda.synchronize() if dev == "cuda" else None
        dt = time.time() - t0
        for b, c in enumerate(cases):
            mv = movable[b]
            cols = np.concatenate([[3 * a, 3 * a + 1, 3 * a + 2] for a in mv])
            n = 3 * len(c["Z"])
            Hb = H[b, :n, :n].detach().cpu().numpy().astype(float)
            Hb = 0.5 * (Hb + Hb.T)
            Hsub = Hb[np.ix_(cols, cols)]
            msub = masses[b][mv]
            Hsub_mw = mass_weight(Hsub, msub)
            ev_s, V_s = spectrum(Hsub_mw)
            # embed the partial leftmost mode into the full space (zeros on frozen DOFs)
            v_emb = np.zeros(n)
            v_emb[cols] = V_s[:, 0]
            v_emb /= max(np.linalg.norm(v_emb), 1e-30)
            ev_f, V_f = full_sp[b]
            rows.append(dict(
                case=c["idx"], natoms=len(c["Z"]), radius=R, m_core=len(mv),
                frac_core=len(mv) / len(c["Z"]),
                lam_full=float(ev_f[0]), lam_core=float(ev_s[0]),
                rel_d_lam=abs(float(ev_s[0]) - float(ev_f[0])) / max(abs(float(ev_f[0])), 1e-12),
                mode0_overlap=abs(float(v_emb @ V_f[:, 0])),
                nimag_full=n_imag(ev_f), nimag_core=n_imag(ev_s),
                ge=int(6 * len(mv) + 1), ge_full=int(6 * len(c["Z"]) + 1),
            ))
        print(f"[core] R={R}: wall {dt:.1f}s  <m_core>="
              f"{np.mean([len(m) for m in movable]):.1f}", flush=True)
    out["core"] = rows
    _summ_core(rows)


def _summ_core(rows):
    # Cases whose FULL Hessian is not a clean index-1 saddle cannot judge a partial
    # Hessian -- they are SKIPPED from the accuracy columns (a gate that fires on an
    # inapplicable case is a broken gate), and counted separately.
    print(f"{'R(A)':>6} {'n':>4} {'skip':>5} {'<m_core>':>9} {'<frac>':>7} "
          f"{'med rel_dlam':>13} {'med ovl':>9} {'min ovl':>9} {'ovl>=0.99':>10} "
          f"{'nimag_ok':>9} {'GE ratio':>9}", flush=True)
    for R in sorted(set(r["radius"] for r in rows)):
        allr = [r for r in rows if r["radius"] == R]
        rs = [r for r in allr if r["nimag_full"] == 1]
        if not rs:
            continue
        print(f"{R:>6} {len(rs):>4} {len(allr) - len(rs):>5} "
              f"{np.mean([r['m_core'] for r in rs]):>9.1f} "
              f"{np.mean([r['frac_core'] for r in rs]):>7.2f} "
              f"{np.median([r['rel_d_lam'] for r in rs]):>13.3e} "
              f"{np.median([r['mode0_overlap'] for r in rs]):>9.5f} "
              f"{min(r['mode0_overlap'] for r in rs):>9.5f} "
              f"{sum(r['mode0_overlap'] >= 0.99 for r in rs)}/{len(rs):<9} "
              f"{sum(r['nimag_core'] == 1 for r in rs)}/{len(rs):<7} "
              f"{sum(r['ge'] for r in rs) / sum(r['ge_full'] for r in rs):>9.3f}",
              flush=True)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="phess_acc")
    ap.add_argument("--probe", nargs="+",
                    default=["fd", "update", "lanczos", "core"])
    ap.add_argument("--n-cases", type=int, default=20)
    ap.add_argument("--min-atoms", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--fd-deltas", nargs="+", type=float, default=[2e-3])
    ap.add_argument("--updates", nargs="+", default=["bofill", "psb", "sr1", "bfgs"])
    ap.add_argument("--update-steps", type=int, default=8)
    ap.add_argument("--lanczos-m", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--lobpcg-iters", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--precond", default="model", choices=["model", "none"])
    ap.add_argument("--synth-spreads", nargs="+", type=float, default=[10, 100, 1000])
    ap.add_argument("--lanczos-gamma", type=float, default=0.4)
    ap.add_argument("--lanczos-eta", type=float, default=1e-3)
    ap.add_argument("--lanczos-step", type=float, default=0.05)
    ap.add_argument("--core-radii", nargs="+", type=float,
                    default=[1.5, 2.0, 2.5, 3.0, 4.0, 99.0])
    args = ap.parse_args()

    sys.path.insert(0, args.fork)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    cases = load_cases(args.pkl, args.n_cases, args.min_atoms)
    print(f"[acc] device={args.device} torch={torch.__version__} "
          f"cases={len(cases)} natoms={[len(c['Z']) for c in cases]}", flush=True)

    out = dict(meta=dict(tag=args.tag, device=args.device, torch=torch.__version__,
                         n_cases=len(cases), natoms=[len(c["Z"]) for c in cases],
                         seed=args.seed, sigma=args.sigma,
                         gpu=(torch.cuda.get_device_name(0)
                              if args.device == "cuda" else "cpu"),
                         argv=vars(args)))
    for p in args.probe:
        print(f"\n================= PROBE {p} =================", flush=True)
        {"fd": probe_fd, "update": probe_update, "lanczos": probe_lanczos,
         "core": probe_core, "synth": probe_synth}[p](args, cases, out)

    path = os.path.join(args.out_dir, f"{args.tag}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\n[acc] wrote {path}", flush=True)
    print("PHESS_ACC_DONE", flush=True)


if __name__ == "__main__":
    main()
