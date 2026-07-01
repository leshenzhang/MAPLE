"""
Thin CLI for MAPLE trajectory analysis — usable WITHOUT the MD engine.

Examples
--------
    python -m maple.function.dispatcher.md.analysis.analyze \
        run.dcd --rst run_md.rst --rdf --type-a O --type-b H --r-max 6.0

    python -m maple.function.dispatcher.md.analysis.analyze \
        run.xyz --msd --dt 10.0 --rmsd

Topology (DCD only): one of --symbols / --top / --rst is REQUIRED because DCD
stores no element identity.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .reader import DCDTrajReader, MapleXYZReader, MultiReplicaReader
from .rdf import compute_rdf
from .msd import compute_msd
from .rmsf_rmsd import compute_rmsd, compute_rmsf
from .density import total_density, density_profile
from .hbonds import hbond_timeseries


def _open_reader(args):
    path = args.traj
    if "*" in path or ".rep" in path:
        return MultiReplicaReader(path)
    if path.endswith(".dcd"):
        symbols = args.symbols.split() if args.symbols else None
        return DCDTrajReader(path, symbols=symbols, top=args.top, rst=args.rst)
    return MapleXYZReader(path)


def _frames(reader):
    if isinstance(reader, MultiReplicaReader):
        return reader.replica(0).read_all()      # CLI analyses replica 0 by default
    return list(reader)


def main(argv=None):
    p = argparse.ArgumentParser(description="MAPLE trajectory analysis (post-processing).")
    p.add_argument("traj", help="DCD / MAPLE-XYZ / replica-glob trajectory path")
    p.add_argument("--symbols", help="space-separated element symbols (DCD topology)")
    p.add_argument("--top", help="topology sidecar (ASE-readable or symbol tokens)")
    p.add_argument("--rst", help="MAPLE .rst checkpoint for the topology")

    p.add_argument("--rdf", action="store_true")
    p.add_argument("--type-a", default=None)
    p.add_argument("--type-b", default=None)
    p.add_argument("--r-max", type=float, default=6.0)
    p.add_argument("--nbins", type=int, default=200)

    p.add_argument("--msd", action="store_true")
    p.add_argument("--dt", type=float, default=1.0, help="time between frames")
    p.add_argument("--no-unwrap", action="store_true")

    p.add_argument("--rmsd", action="store_true")
    p.add_argument("--rmsf", action="store_true")
    p.add_argument("--density", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--axis", type=int, default=2)
    p.add_argument("--hbonds", action="store_true")

    p.add_argument("--out", default=None, help="optional .npz to save arrays")
    args = p.parse_args(argv)

    reader = _open_reader(args)
    frames = _frames(reader)
    print(f"# loaded {len(frames)} frames, {len(frames[0])} atoms from {args.traj}")
    results = {}

    if args.rdf:
        r, g = compute_rdf(frames, r_max=args.r_max, nbins=args.nbins,
                           type_a=args.type_a, type_b=args.type_b)
        results["rdf_r"] = r
        results["rdf_g"] = g
        peak = r[np.argmax(g)]
        print(f"# RDF({args.type_a or 'all'}-{args.type_b or 'all'}): "
              f"first peak g_max={g.max():.3f} at r={peak:.3f} A")

    if args.msd:
        res = compute_msd(frames, dt=args.dt, unwrap=not args.no_unwrap)
        results["msd_t"] = res["t"]
        results["msd"] = res["msd"]
        print(f"# MSD: D={res['D']:.6g} (length^2/time of dt units), "
              f"slope={res['slope']:.6g}")

    if args.rmsd:
        rmsd = compute_rmsd(frames)
        results["rmsd"] = rmsd
        print(f"# RMSD vs frame0: mean={rmsd.mean():.4f} max={rmsd.max():.4f} A "
              f"(self={rmsd[0]:.2e})")

    if args.rmsf:
        rmsf = compute_rmsf(frames)
        results["rmsf"] = rmsf
        print(f"# RMSF: mean={rmsf.mean():.4f} max={rmsf.max():.4f} A")

    if args.density:
        d = total_density(frames[0])
        print(f"# density: {d['mass_density']:.4f} g/cm^3, "
              f"{d['number_density']:.5f} atoms/A^3, V={d['volume']:.2f} A^3")

    if args.profile:
        c, prof = density_profile(frames, axis=args.axis)
        results["profile_x"] = c
        results["profile"] = prof
        print(f"# density profile along axis {args.axis}: "
              f"{len(c)} bins, peak={prof.max():.4f}")

    if args.hbonds:
        hb = hbond_timeseries(frames)
        results["hbonds"] = hb
        print(f"# H-bonds (geometric): mean={hb.mean():.2f} per frame "
              f"(min={hb.min()}, max={hb.max()})")

    if args.out:
        np.savez(args.out, **results)
        print(f"# saved arrays -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
