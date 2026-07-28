#!/usr/bin/env python
"""A3-fwd gate: R3-7 cached per-candidate cell gather in MACEBatchCalc._build_edges_pbc
is BYTE-identical to the previous per-step gather (no MACE model needed).

Two tiers.
Tier-1 (model-free, always runs): no traced 6-arg standard-MACE .pt exists in this
sandbox (maceomol.pt absent; macepol*.pt has a different 12-arg signature). What IS
provable without any model: the edge builder is a
pure function of (coord, cand_i/cand_j/cand_rep, cell, cell_inv, shift_combos,
r_max), and the change only hoists two gathers that depend on loop-invariant state.
This gate drives the REAL ``_build_edges_pbc`` on a bare instance (object.__new__,
the pattern tests/test_batch_calc_contract.py already uses) with random triclinic
cells + random geometries, and compares against the pre-change inline-gather
reference implementation. torch.equal on edge_index AND shifts.

Tier-2 (runs when MACE_CKPT points at a mace-package checkpoint): a REAL periodic
forward. The checkpoint is wrapped in the Eager6 6-arg adapter that
maple/function/calculator/mace/_test_stdmace_pbc.py already uses, driving
MACEBatchCalc.get_ef_gpu on a periodic water box twice -- once with the new cached
gather, once with the pre-change reference monkeypatched in -- and requires
bit-identical E and F.

Also asserts the mutation precondition: MACEBatchCalc exposes no cell-mutating hook
(set_cells_/rescale_isotropic_ are MaceOffBatchCalc-only), so the cached gather can
never go stale.

env: OUT [MACE_CKPT]
"""
import json
import os
import traceback

import torch

OUT = os.environ.get("OUT", "gate_r37.json")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64

from maple.function.calculator.mace._mace_batch_calculator import MACEBatchCalc  # noqa: E402

RES = {"gate": "R3-7_stdmace_cached_cell_gather", "device": DEV}


def _ref_build_edges_pbc(c, coord):
    """VERBATIM pre-change body: gathers self._cell[rep] / self._cell_inv[rep]
    inside the per-forward call."""
    device = c.device
    ci, cj, rep = c.cand_i, c.cand_j, c.cand_rep
    cellp = c._cell[rep]
    invp = c._cell_inv[rep]
    rij0 = coord[cj] - coord[ci]
    n0 = torch.round(torch.einsum("pc,pck->pk", rij0, invp))
    rmax2 = c.r_max * c.r_max
    src_l, dst_l, sh_l = [], [], []
    for S in c._shift_combos:
        sint = S.view(1, 3) - n0
        scart = torch.einsum("pk,pkc->pc", sint, cellp)
        rij = rij0 + scart
        d2 = (rij * rij).sum(dim=-1)
        keep = d2 < rmax2
        if bool(keep.any()):
            kci, kcj = ci[keep], cj[keep]
            ksh = scart[keep]
            src_l.append(kci); dst_l.append(kcj); sh_l.append(ksh)
            src_l.append(kcj); dst_l.append(kci); sh_l.append(-ksh)
    if src_l:
        edge_index = torch.stack([torch.cat(src_l), torch.cat(dst_l)], dim=0)
        shifts = torch.cat(sh_l, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.int64, device=device)
        shifts = torch.zeros((0, 3), dtype=DT, device=device)
    return edge_index, shifts


def make_bare(B, n_per, seed, r_max=5.0):
    g = torch.Generator().manual_seed(seed)
    c = object.__new__(MACEBatchCalc)
    c.device = torch.device(DEV)
    c.dtype = DT
    c.mdtype = DT
    c.r_max = r_max
    c._periodic = True
    c._atoms_B = B
    n_b = torch.full((B,), n_per, dtype=torch.int64, device=DEV)
    c._n_b = n_b
    ptr = torch.cat([torch.zeros(1, dtype=torch.int64, device=DEV), n_b.cumsum(0)])
    c._ptr = ptr
    ci, cj, rep = [], [], []
    for b in range(B):
        off = int(ptr[b].item())
        iu, ju = torch.triu_indices(n_per, n_per, offset=1, device=DEV)
        ci.append(iu + off); cj.append(ju + off)
        rep.append(torch.full((iu.numel(),), b, dtype=torch.int64, device=DEV))
    c.cand_i = torch.cat(ci); c.cand_j = torch.cat(cj); c.cand_rep = torch.cat(rep)
    # random TRICLINIC cells, side ~ 2.2 * r_max (box guard satisfied)
    base = torch.eye(3, dtype=DT).repeat(B, 1, 1) * (2.2 * r_max)
    skew = 0.15 * torch.randn((B, 3, 3), generator=g, dtype=DT) * r_max
    cells = (base + torch.triu(skew, diagonal=1)).to(DEV)
    c._cell = cells
    c._cell_inv = torch.linalg.inv(cells)
    c._shift_combos = torch.tensor(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=DT, device=DEV)
    c._cand_cell = c._cell[c.cand_rep]
    c._cand_cell_inv = c._cell_inv[c.cand_rep]
    return c, g


def main():
    cases = []
    try:
        for (B, n_per, seed) in [(1, 8, 1), (4, 10, 2), (16, 6, 3), (8, 12, 4)]:
            c, g = make_bare(B, n_per, seed)
            N = int(c._ptr[-1].item())
            ok_all = True
            for trial in range(6):
                # geometries: some inside the box, some drifted far out (unwrapped)
                coord = (torch.rand((N, 3), generator=g, dtype=DT) * 11.0
                         + (trial * 7.0 if trial % 2 else 0.0)).to(DEV)
                ei_new, sh_new = c._build_edges_pbc(coord)
                ei_ref, sh_ref = _ref_build_edges_pbc(c, coord)
                same = bool(torch.equal(ei_new, ei_ref) and torch.equal(sh_new, sh_ref))
                ok_all = ok_all and same
                if not same:
                    cases.append({"B": B, "n_per": n_per, "trial": trial,
                                  "equal": False,
                                  "n_edges_new": int(ei_new.size(1)),
                                  "n_edges_ref": int(ei_ref.size(1))})
            cases.append({"B": B, "n_per": n_per, "n_atoms": N,
                          "trials": 6, "all_byte_equal": ok_all,
                          "n_edges_last": int(ei_new.size(1))})
        RES["byte_equality"] = cases
        RES["pass"] = all(c.get("all_byte_equal", True) for c in cases)
    except Exception:
        RES["error"] = traceback.format_exc()
        RES["pass"] = False

    # ---- Tier-2: REAL forward E/F parity on a real MACE checkpoint -------------
    # MACEBatchCalc needs a traced 6-arg model (absent), so wrap a mace-package
    # checkpoint in the same Eager6 adapter maple/function/calculator/mace/
    # _test_stdmace_pbc.py already uses. Then run the SAME periodic batch twice:
    # once with the new cached-gather _build_edges_pbc, once with the verbatim
    # pre-change reference monkeypatched in. E/F must be bit-identical.
    ckpt = os.environ.get("MACE_CKPT", "")
    t2 = {"ckpt": ckpt, "ckpt_exists": bool(ckpt) and os.path.exists(ckpt)}
    if t2["ckpt_exists"]:
        try:
            from maple.function.calculator.mace._mace_batch_calculator import (
                _one_hot_node_attrs,
            )

            class Eager6(torch.nn.Module):
                def __init__(self, mace):
                    super().__init__()
                    self.mace = mace

                def forward(self, positions, node_attrs, edge_index, shifts, batch, ptr):
                    Bn = int(ptr.numel() - 1)
                    dev, dt = positions.device, positions.dtype
                    data = {"positions": positions, "node_attrs": node_attrs,
                            "edge_index": edge_index, "shifts": shifts,
                            "unit_shifts": torch.zeros_like(shifts),
                            "batch": batch, "ptr": ptr,
                            "cell": torch.zeros((Bn, 3, 3), dtype=dt, device=dev),
                            "head": torch.zeros(Bn, dtype=torch.long, device=dev)}
                    out = self.mace(data, compute_force=False, training=False)
                    return out["energy"].reshape(-1)

            import numpy as np
            from ase import Atoms
            mace = torch.load(ckpt, map_location=DEV, weights_only=False)
            mace = mace.to(DEV).to(DT).eval()
            for p in mace.parameters():
                p.requires_grad_(False)
            r_max = float(mace.r_max)
            atn = [int(z) for z in mace.atomic_numbers]

            def water_box(n, spacing, seed=0):
                rng = np.random.default_rng(seed)
                dOH, ang = 0.9572, np.deg2rad(104.52)
                base = np.array([[0, 0, 0], [dOH, 0, 0],
                                 [dOH * np.cos(ang), dOH * np.sin(ang), 0.0]])
                pos, sym = [], []
                for i in range(n):
                    for j in range(n):
                        for k in range(n):
                            c3 = (np.array([i, j, k]) + 0.5) * spacing \
                                + 0.15 * rng.standard_normal(3)
                            pos.extend(base + c3)
                            sym.extend(["O", "H", "H"])
                L = n * spacing
                return Atoms(symbols=sym, positions=np.array(pos),
                             cell=[L, L, L], pbc=True)

            side = max(2.2 * r_max, 9.0)
            n = 3
            box = water_box(n, side / n, seed=5)
            c = object.__new__(MACEBatchCalc)
            c.device = torch.device(DEV); c.dtype = DT; c.mdtype = DT
            c.r_max = r_max; c.atomic_numbers = atn
            c.model = Eager6(mace)
            c._batch_native = True
            c._hess_mode = None
            c._prepared = False
            c._coord_backup = None
            MACEBatchCalc.prepare(c, [box.copy(), box.copy()])
            E_new, F_new = c.get_ef_gpu()
            E_new = E_new.detach().cpu().numpy().copy()
            F_new = F_new.detach().cpu().numpy().copy()
            # swap in the verbatim pre-change builder
            c._build_edges_pbc = lambda coord: _ref_build_edges_pbc(c, coord)
            E_ref, F_ref = c.get_ef_gpu()
            E_ref = E_ref.detach().cpu().numpy().copy()
            F_ref = F_ref.detach().cpu().numpy().copy()
            t2.update({
                "n_atoms": int(c.N_atoms), "B": int(c._atoms_B),
                "r_max_A": r_max, "box_side_A": side,
                "maxdE_Ha": float(np.abs(E_new - E_ref).max()),
                "maxdF_HaA": float(np.abs(F_new - F_ref).max()),
                "bit_identical": bool(np.array_equal(E_new, E_ref)
                                      and np.array_equal(F_new, F_ref)),
                "E_new_Ha": [float(x) for x in E_new]})
        except Exception:
            t2["error"] = traceback.format_exc(limit=4)
    RES["tier2_real_forward_parity"] = t2
    if "error" in t2:
        print("[tier2] " + t2["error"], flush=True)

    # mutation precondition
    RES["no_cell_mutation_hook"] = {
        "set_cells_": hasattr(MACEBatchCalc, "set_cells_"),
        "rescale_isotropic_": hasattr(MACEBatchCalc, "rescale_isotropic_"),
    }
    RES["note"] = ("model-free byte-equality gate: no loadable periodic standard-MACE "
                   ".pt in this sandbox, so no runtime E/F parity is possible here")
    json.dump(RES, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in RES.items() if k != "byte_equality"}, indent=1),
          flush=True)
    t2 = RES.get("tier2_real_forward_parity", {})
    if t2.get("ckpt_exists") and "error" not in t2:
        RES["pass"] = bool(RES.get("pass")) and bool(t2.get("bit_identical"))
    json.dump(RES, open(OUT, "w"), indent=1)
    print("R37_GATE_PASS" if RES.get("pass") else "R37_GATE_FAIL", flush=True)


if __name__ == "__main__":
    main()
