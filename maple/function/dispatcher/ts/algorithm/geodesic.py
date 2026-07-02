# -*- coding: utf-8 -*-
"""
Geodesic TS-guess generator (training-free, batchable, self-contained).

A drop-in front-end that produces a transition-state initial guess WITHOUT any
model training, ready to feed MAPLE's batched P-RFO (``BatchPRFO``) for exact
refinement + frequency validation.

Algorithm
---------
1. Geodesic interpolation in a Morse-scaled interatomic-distance metric
   (Zhu, Thompson & Martinez, "Geodesic interpolation for reaction pathways",
   J. Chem. Phys. 150, 164103 (2019), DOI 10.1063/1.5090303). The path between
   reactant R and product P is built by minimising the path LENGTH measured with
   redundant internal coordinates -- the pairwise distances passed through a
   Morse-like scaler q_ij = exp(-a (r_ij - r_e)/r_e) + b r_e/r_ij with
   r_e = covalent_radii[i] + covalent_radii[j]. Because the metric is dominated
   by the short (bonding) distances, the geodesic naturally curves AROUND atom
   clashes that a straight Cartesian / linear interpolation drives through -- a
   far better starting path than linear or IDPP-inverse-distance interpolation.
   This step is purely GEOMETRIC (no energy/force evaluations) and trivially
   batches across reactions.

2. (Optional) FIRE relaxation of the geodesic path on the MLIP surface with a
   climbing node (Morse-geodesic guess relaxed on an ML potential,
   arXiv:2507.17968). The interior nodes follow the climbing-image-NEB force
   (perpendicular MLIP force + spring along the tangent); the current
   highest-energy node CLIMBS (F_mlp - 2(F_mlp . tau) tau) so it slides up to the
   saddle. One batched ``calc.prepare(...) + get_ef_gpu()`` over every image of
   every band per FIRE step (image-as-batch, exactly like NEB.run_multiband).

3. The highest-energy node along the (relaxed) path is the TS GUESS. Wrap the
   guesses in a ``Molecules`` and hand them to ``BatchPRFO`` for the exact
   first-order-saddle refinement + ``get_efh_gpu`` n_imag check.

Self-contained: numpy + ASE only at import time (torch imported lazily inside the
batched relaxer). No new heavy dependencies; the geodesic core is reimplemented
in MAPLE's style (mirrors neb.py's IDPP smoother), not taken from the external
``geodesic-interpolate`` package.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

from .logger import log_info
from ...jobABC import JobABC

from maple.function.utility import Molecules


# =============================================================================
# ------------------------------ small helpers --------------------------------
# =============================================================================
def _to_f64(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, float]:
    """Rigidly superpose Q onto P (both (N,3)); return (Q_aligned, rmsd)."""
    P = _to_f64(P); Q = _to_f64(Q)
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Qc.T @ Pc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    Qa = Qc @ R + P.mean(0)
    rmsd = float(np.sqrt(((Qa - P) ** 2).sum(1).mean()))
    return Qa, rmsd


def _pair_indices(n_atoms: int, pos_list: Optional[List[np.ndarray]] = None,
                  cutoff: Optional[float] = None) -> np.ndarray:
    """Upper-triangular (i<j) pairs (M,2). If ``cutoff`` + ``pos_list`` given,
    keep only pairs whose distance is < cutoff in AT LEAST ONE structure of the
    path (redundant-internal sparsification for larger systems)."""
    iu = np.triu_indices(n_atoms, k=1)
    pairs = np.stack(iu, axis=1).astype(np.int64)
    if cutoff is not None and pos_list:
        keep = np.zeros(len(pairs), dtype=bool)
        for X in pos_list:
            d = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)
            keep |= (d < cutoff)
        pairs = pairs[keep]
    return pairs


# =============================================================================
# ------------------------- Morse-scaled geodesic core ------------------------
# =============================================================================
@dataclass
class GeodesicParams:
    n_images: int = 10          # interior images (excludes the 2 endpoints)
    alpha: float = 1.7          # Morse decay (geodesic-interpolate default)
    beta: float = 0.01          # long-range 1/r term weight
    cutoff: Optional[float] = None   # None -> all pairs (fine for <~25 atoms)
    max_steps: int = 200        # L-BFGS sweeps for the geodesic minimisation
    maxstep: float = 0.10       # per-iteration Cartesian step cap (Angstrom)
    rms_tol: float = 1e-4       # stop when RMS geodesic gradient < tol
    lbfgs_m: int = 10


def _morse_scaled(X: np.ndarray, pairs: np.ndarray, re: np.ndarray,
                  alpha: float, beta: float):
    """Return (q, dqdr, unit) for one geometry.
       q_ij  = exp(-alpha (r-re)/re) + beta re/r
       dqdr  = -(alpha/re) exp(-alpha (r-re)/re) - beta re/r^2
       unit  = (X_i - X_j)/r           (M,3)
    """
    diff = X[pairs[:, 0]] - X[pairs[:, 1]]          # (M,3)
    r = np.linalg.norm(diff, axis=1) + 1e-12        # (M,)
    em = np.exp(-alpha * (r - re) / re)
    q = em + beta * re / r
    dqdr = -(alpha / re) * em - beta * re / (r * r)
    unit = diff / r[:, None]
    return q, dqdr, unit


def _jacT(coef: np.ndarray, unit: np.ndarray, pairs: np.ndarray,
          n_atoms: int) -> np.ndarray:
    """Scatter per-pair scalar ``coef`` along ``unit`` to atomic gradient (N,3).
    coef = (dE/dq)_p * dqdr_p  (J^T v)."""
    g = np.zeros((n_atoms, 3), dtype=np.float64)
    cv = coef[:, None] * unit                       # (M,3)
    np.add.at(g, pairs[:, 0], cv)
    np.add.at(g, pairs[:, 1], -cv)
    return g


def geodesic_interpolate(R_pos: np.ndarray, P_pos: np.ndarray, Z: np.ndarray,
                         params: GeodesicParams = GeodesicParams(),
                         output: Optional[str] = None) -> List[np.ndarray]:
    """Geodesic path R -> P in the Morse-scaled interatomic-distance metric.

    Minimises the discrete path length  E = sum_k ||q(X_{k+1}) - q(X_k)||^2  over
    the interior Cartesian nodes (endpoints fixed) by L-BFGS with the ANALYTIC
    gradient  dE/dX_m = J_m^T (2 (2 q_m - q_{m-1} - q_{m+1})).  The sum-of-squared
    segment-length objective yields a near-uniformly-spaced geodesic (the path
    that is *short in the Morse metric* hence clash-avoiding).

    Returns a list of (N,3) arrays of length n_images+2 (endpoints included).
    Pure geometry: NO energy/force calls.
    """
    R_pos = _to_f64(R_pos)
    P_aligned, _ = kabsch_align(R_pos, _to_f64(P_pos))
    Z = np.asarray(Z)
    n_atoms = R_pos.shape[0]
    n_inner = int(params.n_images)
    n_img = n_inner + 2

    rc = covalent_radii[Z]
    pairs = _pair_indices(n_atoms, [R_pos, P_aligned], params.cutoff)
    re = rc[pairs[:, 0]] + rc[pairs[:, 1]]
    re = np.maximum(re, 0.3)                          # guard tiny covalent radii

    # --- initial path: linear Cartesian interpolation (interior nodes) ---
    coords = [R_pos.copy()]
    for k in range(1, n_inner + 1):
        lam = k / (n_inner + 1)
        coords.append((1.0 - lam) * R_pos + lam * P_aligned)
    coords.append(P_aligned.copy())

    if n_inner == 0:
        return coords

    # pre-compute endpoint scaled coords (fixed)
    qR, _, _ = _morse_scaled(coords[0], pairs, re, params.alpha, params.beta)
    qP, _, _ = _morse_scaled(coords[-1], pairs, re, params.alpha, params.beta)

    def pack(c):  return np.concatenate([c[i].reshape(-1) for i in range(1, n_img - 1)])
    def unpack(x):
        out = []
        off = 0
        for _ in range(n_inner):
            out.append(x[off:off + n_atoms * 3].reshape(n_atoms, 3)); off += n_atoms * 3
        return out

    def grad_and_qs(x):
        inner = unpack(x)
        qs = [qR]
        sc = []
        for Xi in inner:
            q, dqdr, unit = _morse_scaled(Xi, pairs, re, params.alpha, params.beta)
            qs.append(q); sc.append((dqdr, unit))
        qs.append(qP)
        g = np.zeros_like(x)
        off = 0
        E = 0.0
        for m in range(1, n_img - 1):
            seg = qs[m] - qs[m - 1]
            E += float(np.dot(seg, seg))
            res = 2.0 * qs[m] - qs[m - 1] - qs[m + 1]   # (M,)
            dqdr, unit = sc[m - 1]
            coef = 2.0 * res * dqdr                      # dE/dq * dqdr
            gm = _jacT(coef, unit, pairs, n_atoms)
            g[off:off + n_atoms * 3] = gm.reshape(-1); off += n_atoms * 3
        return g, E

    # --- L-BFGS (two-loop), mirrors neb.py::_run_idpp_smoothing ---
    x = pack(coords)
    m = int(params.lbfgs_m)
    S, Y, rho = [], [], []
    g, E = grad_and_qs(x)
    for it in range(int(params.max_steps)):
        # two-loop direction
        q = g.copy(); alpha_s = []
        for s, y, r in zip(reversed(S), reversed(Y), reversed(rho)):
            a = r * np.dot(s, q); alpha_s.append(a); q = q - a * y
        gamma = (np.dot(S[-1], Y[-1]) / (np.dot(Y[-1], Y[-1]) + 1e-20)) if Y else 1.0
        z = gamma * q
        for (s, y, r), a in zip(zip(S, Y, rho), reversed(alpha_s)):
            b = r * np.dot(y, z); z = z + s * (a - b)
        step = -z
        md = np.max(np.abs(step))
        if md > params.maxstep:
            step *= params.maxstep / md
        x_new = x + step
        g_new, E = grad_and_qs(x_new)
        s = x_new - x; yv = g_new - g; sy = float(np.dot(s, yv))
        if sy > 1e-12:
            if len(S) == m:
                S.pop(0); Y.pop(0); rho.pop(0)
            S.append(s); Y.append(yv); rho.append(1.0 / sy)
        x, g = x_new, g_new
        if np.sqrt(np.mean(g * g)) < params.rms_tol:
            break
    if output is not None:
        log_info([f"[geodesic] converged path-length E={E:.5e} in {it+1} L-BFGS sweeps "
                  f"(rms grad {np.sqrt(np.mean(g*g)):.2e})\n"], output)

    inner = unpack(x)
    return [coords[0]] + inner + [coords[-1]]


# =============================================================================
# ------------------- batched MLIP forward over a flat band -------------------
# =============================================================================
def _eval_flat(calc, atoms_list: List[Atoms]):
    """ONE prepare + ONE get_ef_gpu over an arbitrary flat list. Returns
    (E[list float, Hartree], F[list (N,3) np.f64, Ha/Angstrom])."""
    import torch
    calc.prepare(atoms_list)
    E, F = calc.get_ef_gpu()
    E = (E.detach().to("cpu", torch.float64).numpy() if isinstance(E, torch.Tensor)
         else np.asarray(E, dtype=np.float64))
    F = (F.detach().to("cpu", torch.float64).numpy() if isinstance(F, torch.Tensor)
         else np.asarray(F, dtype=np.float64))
    Es, Fs = [], []
    for i, at in enumerate(atoms_list):
        n = len(at)
        Es.append(float(E[i]))
        Fs.append(F[i, :3 * n].reshape(n, 3).astype(np.float64, copy=True))
    return Es, Fs


def _improved_tangent(Rm, R, Rp, Em, E, Ep):
    """Henkelman-Jonsson improved tangent (JCP 113, 9978)."""
    tp = Rp - R; tm = R - Rm
    if Ep > E < Em or Ep < E > Em:        # not strictly monotonic
        dmax = max(abs(Ep - E), abs(Em - E)); dmin = min(abs(Ep - E), abs(Em - E))
        if Ep > Em:
            t = tp * dmax + tm * dmin
        else:
            t = tp * dmin + tm * dmax
    elif Ep >= Em:
        t = tp
    else:
        t = tm
    nrm = np.linalg.norm(t)
    return t / nrm if nrm > 1e-12 else t


# =============================================================================
# ----------------------------- the dispatcher --------------------------------
# =============================================================================
class GeodesicTSGuess(JobABC):
    """Geodesic TS-guess front-end. Mirrors NEB's construction style; consumes a
    Molecules([R, P]) (single reaction) or a list of [R, P] bands (batched)."""

    def __init__(self, output: str, atoms_or_molecules=None, paras: Optional[dict] = None):
        super().__init__(output)
        self._mol_calc = None
        if isinstance(atoms_or_molecules, Molecules):
            self.input_images = atoms_or_molecules.multiatoms
            self._mol_calc = getattr(atoms_or_molecules, "calc", None)
        elif isinstance(atoms_or_molecules, list):
            self.input_images = atoms_or_molecules
        else:
            self.input_images = None
        self.params = self._init_geo_params(paras)

    @staticmethod
    def _init_geo_params(paras: Optional[dict]) -> GeodesicParams:
        gp = GeodesicParams()
        if paras:
            for f in GeodesicParams().__dataclass_fields__:
                if f in paras:
                    setattr(gp, f, paras[f])
        return gp

    @staticmethod
    def _is_batch_calc(calc) -> bool:
        from ..._batch_calc_utils import is_batch_calc
        return is_batch_calc(calc)

    # ----------------------------- band construction ------------------------
    def build_band(self, atoms_R: Atoms, atoms_P: Atoms) -> List[Atoms]:
        """Geodesic band (n_images+2 Atoms) for one reaction. Pure geometry."""
        Z = atoms_R.get_atomic_numbers()
        coords = geodesic_interpolate(atoms_R.get_positions(), atoms_P.get_positions(),
                                      Z, self.params, output=self.output)
        return [Atoms(numbers=Z, positions=c) for c in coords]

    def build_bands(self, bands: List[List[Atoms]]) -> List[List[Atoms]]:
        return [self.build_band(b[0], b[-1]) for b in bands]

    # ----------------------------- TS-guess picking -------------------------
    def pick_hei(self, geo_bands: List[List[Atoms]], calc) -> Tuple[List[int], List[List[float]]]:
        """ONE batched forward over every node of every band -> per-band index of
        the highest-energy node + per-band node energies (Hartree)."""
        flat, spans = [], []
        for b in geo_bands:
            spans.append((len(flat), len(flat) + len(b))); flat.extend(b)
        Es, _ = _eval_flat(calc, flat)
        heis, ens = [], []
        for (a, z) in spans:
            e = Es[a:z]; ens.append(e); heis.append(int(np.argmax(e)))
        return heis, ens

    # --------------------- optional FIRE relax + climb ----------------------
    def relax_bands_fire(self, geo_bands: List[List[Atoms]], calc,
                         max_iter: int = 60, k_spring: float = 0.10,
                         dt0: float = 0.10, dt_max: float = 0.30, climb: bool = True,
                         fmax_tol: float = 0.05):
        """Batched climbing-image FIRE relaxation of the geodesic bands on the MLIP.

        One ``calc.prepare + get_ef_gpu`` over all images of all bands per step.
        Endpoints frozen. The current highest-energy interior node of each band
        climbs. Returns (heis, energies) after relaxation (geometries updated
        in-place in ``geo_bands``). FIRE per Bitzek et al., PRL 97, 170201 (2006).
        """
        # FIRE constants
        N_MIN, F_INC, F_DEC, A_START, F_A = 5, 1.1, 0.5, 0.1, 0.99
        nb = len(geo_bands)
        # per-band velocity / dt / alpha / npos
        vel = [[np.zeros((len(im), 3)) for im in b] for b in geo_bands]
        dt = [dt0] * nb; alpha = [A_START] * nb; npos = [0] * nb

        for step in range(int(max_iter)):
            # one flat forward
            flat, spans = [], []
            for b in geo_bands:
                spans.append((len(flat), len(flat) + len(b))); flat.extend(b)
            Es, Fs = _eval_flat(calc, flat)
            band_max_f = 0.0
            for bi, (a, z) in enumerate(spans):
                b = geo_bands[bi]
                E = Es[a:z]; F = Fs[a:z]
                n_img = len(b)
                hei = int(np.argmax(E))
                pos = [im.get_positions() for im in b]
                # NEB / climb forces on interior nodes
                for i in range(1, n_img - 1):
                    tau = _improved_tangent(pos[i - 1], pos[i], pos[i + 1],
                                            E[i - 1], E[i], E[i + 1])
                    Fmlp = F[i]
                    if climb and i == hei:
                        Fi = Fmlp - 2.0 * np.vdot(Fmlp, tau) * tau
                    else:
                        Fperp = Fmlp - np.vdot(Fmlp, tau) * tau
                        sp = (np.linalg.norm(pos[i + 1] - pos[i])
                              - np.linalg.norm(pos[i] - pos[i - 1]))
                        Fi = Fperp + k_spring * sp * tau
                    band_max_f = max(band_max_f, float(np.abs(Fi).max()))
                    # FIRE update (per band)
                    v = vel[bi][i]
                    P = float(np.vdot(Fi, v))
                    fhat = Fi / (np.linalg.norm(Fi) + 1e-12)
                    v = (1.0 - alpha[bi]) * v + alpha[bi] * np.linalg.norm(v) * fhat
                    if P > 0:
                        npos[bi] += 1
                        if npos[bi] > N_MIN:
                            dt[bi] = min(dt[bi] * F_INC, dt_max); alpha[bi] *= F_A
                    else:
                        npos[bi] = 0; dt[bi] *= F_DEC; alpha[bi] = A_START; v[:] = 0.0
                    v = v + dt[bi] * Fi
                    vel[bi][i] = v
                    dx = dt[bi] * v
                    md = np.max(np.abs(dx))
                    if md > 0.2:
                        dx *= 0.2 / md
                    b[i].set_positions(pos[i] + dx)
            if band_max_f < fmax_tol:
                break
        return self.pick_hei(geo_bands, calc)

    # ------------------------------- run_multiband --------------------------
    def run_multiband(self, bands: List[List[Atoms]], calc=None,
                      relax_iters: int = 0, climb: bool = True, **kw):
        """Batched geodesic TS-guess over B reactions.

        bands : list of [R, P] (or pre-built bands). calc : batched MLIP calc
        (defaults to the one carried on the Molecules). relax_iters>0 -> FIRE
        relax+climb on the MLIP; ==0 -> pure geometric geodesic + ONE forward to
        pick the highest-energy node.

        Returns (results, stats) with results[i] = {images, energies, hei,
        ts_guess(Atoms)} -- ts_guess feeds straight into Molecules -> BatchPRFO.
        """
        calc = calc or self._mol_calc
        if not self._is_batch_calc(calc):
            raise ValueError("GeodesicTSGuess needs a batched calc (prepare + get_ef_gpu).")
        geo_bands = self.build_bands(bands)
        if relax_iters and relax_iters > 0:
            heis, ens = self.relax_bands_fire(geo_bands, calc, max_iter=relax_iters,
                                              climb=climb, **{k: v for k, v in kw.items()
                                              if k in ("k_spring", "dt0", "dt_max", "fmax_tol")})
        else:
            heis, ens = self.pick_hei(geo_bands, calc)
        results = []
        for b, hei, e in zip(geo_bands, heis, ens):
            results.append(dict(images=b, energies=e, hei=hei, ts_guess=b[hei].copy()))
        return results, dict(reactions=len(results), relax_iters=int(relax_iters))

    # ------------------------------- single run -----------------------------
    def run(self):
        """Single-reaction entry (Molecules([R,P]) in -> writes the TS guess)."""
        if self.input_images is None or len(self.input_images) < 2:
            raise ValueError("GeodesicTSGuess needs reactant + product structures.")
        R, P = self.input_images[0], self.input_images[-1]
        band = self.build_band(R, P)
        if self._is_batch_calc(self._mol_calc):
            heis, ens = self.pick_hei([band], self._mol_calc)
            hei = heis[0]
        else:
            hei = len(band) // 2     # no calc -> geometric midpoint of the geodesic
        guess = band[hei]
        base, _ = os.path.splitext(self.output)
        out = base + "_geodesic_tsguess.xyz"
        from ase.io import write as _ase_write
        _ase_write(out, guess)
        log_info([f"[geodesic] TS guess (node {hei}/{len(band)-1}) written to {out}\n"], self.output)
        self.ts_guess = guess
        return guess
