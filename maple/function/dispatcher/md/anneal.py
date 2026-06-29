"""Simulated annealing — a piecewise-linear target-temperature schedule for the
NVT/NPT thermostat (GROMACS `annealing = single`, AMBER `nmropt` tgtmd-style
temperature ramp). Pure schedule math: the thermostat's per-step target T is set
from this schedule each step, so it composes with any MLIP force engine and with
HMR/constraints/GaMD (none of which touch the thermostat target).

Schedule spec (``params.anneal``) — comma-separated control points:
  ""/off/none            -> disabled (constant temperature, no-op)
  "100,300"              -> linear ramp 100 K -> 300 K across the run
  "300,500,300"          -> equally-spaced points: heat to 500 then cool to 300
  "0:100,0.5:300,1:300"  -> explicit ``frac:T`` points (frac in [0,1] of the run)
  "300"                  -> constant 300 K (degenerate; same as no annealing)

Bare tokens are placed at equally-spaced fractions [0, 1/(n-1), ..., 1]; ``frac:T``
tokens carry an explicit fraction. The two forms may be mixed. The fraction is of
*this run's* step budget (``step / n_steps``), so for a restart you normally leave
annealing off and run constant-T production.
"""
from typing import Callable, List, Optional, Tuple

_OFF = ("", "off", "no", "none", "false")


def parse_anneal(spec) -> Optional[List[Tuple[float, float]]]:
    """Parse an anneal spec into sorted (frac, T_kelvin) control points, or None.

    Returns None when annealing is disabled. A single control point is promoted
    to a flat two-point schedule so callers can always interpolate.
    """
    if spec is None or isinstance(spec, bool):
        return None
    if isinstance(spec, (int, float)):
        return [(0.0, float(spec)), (1.0, float(spec))]
    s = str(spec).strip()
    if s.lower() in _OFF:
        return None

    tokens = [t.strip() for t in s.split(",") if t.strip()]
    if not tokens:
        return None

    explicit: List[Tuple[float, float]] = []
    bare: List[Tuple[int, float]] = []          # (token_index, T) for equal spacing
    for i, tok in enumerate(tokens):
        if ":" in tok:
            f_str, t_str = tok.split(":", 1)
            explicit.append((float(f_str), float(t_str)))
        else:
            bare.append((i, float(tok)))

    points: List[Tuple[float, float]] = list(explicit)
    if bare:
        n = len(tokens)
        denom = (n - 1) if n > 1 else 1
        for i, T in bare:
            frac = i / denom if n > 1 else 0.0
            points.append((frac, T))

    # clamp fractions to [0,1], sort, and pin a point at each end so the schedule
    # is defined over the whole run
    points = [(min(1.0, max(0.0, f)), T) for f, T in points]
    points.sort(key=lambda p: p[0])
    if points[0][0] > 0.0:
        points.insert(0, (0.0, points[0][1]))
    if points[-1][0] < 1.0:
        points.append((1.0, points[-1][1]))
    return points


def anneal_temperature(points: List[Tuple[float, float]], frac: float) -> float:
    """Piecewise-linear target temperature at ``frac`` in [0,1] of the run."""
    frac = min(1.0, max(0.0, frac))
    for (f0, t0), (f1, t1) in zip(points, points[1:]):
        if frac <= f1:
            if f1 == f0:
                return t1
            w = (frac - f0) / (f1 - f0)
            return t0 + w * (t1 - t0)
    return points[-1][1]


def make_anneal_fn(spec, n_steps: int) -> Optional[Callable[[int], float]]:
    """Return a callable mapping a 1-based loop step -> target T (K), or None.

    ``frac = step / n_steps`` so step ``n_steps`` lands exactly on the final
    control point. None when annealing is disabled (caller keeps constant T).
    """
    points = parse_anneal(spec)
    if points is None:
        return None
    span = float(max(1, n_steps))
    return lambda step: anneal_temperature(points, step / span)


if __name__ == "__main__":
    # Self-test: parse forms, interpolate midpoints, disabled + constant.
    assert parse_anneal("") is None and parse_anneal("off") is None
    assert parse_anneal(None) is None

    # linear 100 -> 300
    fn = make_anneal_fn("100,300", 100)
    assert abs(fn(1) - (100 + 200 * 0.01)) < 1e-9, fn(1)
    assert abs(fn(50) - 200.0) < 1e-9, fn(50)
    assert abs(fn(100) - 300.0) < 1e-9, fn(100)
    print(f"(1) linear 100->300 OK: mid={fn(50):.1f} end={fn(100):.1f}")

    # heat-then-cool 300 -> 500 -> 300 (equally spaced)
    p = parse_anneal("300,500,300")
    assert p == [(0.0, 300.0), (0.5, 500.0), (1.0, 300.0)], p
    assert abs(anneal_temperature(p, 0.25) - 400.0) < 1e-9
    assert abs(anneal_temperature(p, 0.75) - 400.0) < 1e-9
    print(f"(2) heat/cool OK: q1={anneal_temperature(p,0.25):.0f} q3={anneal_temperature(p,0.75):.0f}")

    # explicit frac:T points, hold at the end
    p2 = parse_anneal("0:100,0.5:300,1:300")
    assert abs(anneal_temperature(p2, 0.25) - 200.0) < 1e-9
    assert abs(anneal_temperature(p2, 0.9) - 300.0) < 1e-9
    print(f"(3) explicit frac:T OK: f0.25={anneal_temperature(p2,0.25):.0f}")

    # constant single value
    p3 = parse_anneal("300")
    assert anneal_temperature(p3, 0.0) == 300.0 and anneal_temperature(p3, 1.0) == 300.0
    print("(4) constant OK")

    # endpoints pinned when first/last frac are interior
    p4 = parse_anneal("0.5:400")
    assert p4[0] == (0.0, 400.0) and p4[-1] == (1.0, 400.0), p4
    print("(5) endpoint pinning OK")

    print("ANNEAL SELF-CHECK PASS")
