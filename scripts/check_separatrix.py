"""High-precision check of the basin boundaries found in double precision.

The basin bisection in ``analyze_basins_precision.py`` leaves some boundaries
with an ``unresolved`` upper side: just above the separatrix the orbit crawls,
and 60 000 float64 rounds are not enough to land on a fixed point.  That is
exactly the situation section 11 of the task warns about -- a tiny step size
that is NOT convergence.

This script takes each bisected boundary eps*, steps a finite distance to
either side, and integrates the reduced map at mpmath dps = 60 as well as in
float64, long enough for the two sides to separate unambiguously.  It reports
where each side actually goes and whether the two precisions agree.

Output: results/data/repeated_dynamics_independent/separatrix_check.csv
"""

from __future__ import annotations

import csv

import mpmath as mp
import numpy as np

from _bootstrap import DATA_DIR  # noqa: E402
from pqec_distill.repeated_dynamics import isotropic_pops, one_round  # noqa: E402
from analyze_repeated_dynamics import (  # noqa: E402
    all_fixed_points, fidelity, jacobian, reduced_map, spectral_radius,
)

OUT = DATA_DIR / "repeated_dynamics_independent"
# must match analyze_repeated_dynamics.DPS: all_fixed_points() accepts a
# candidate only when its residual is below 10^(-DPS+10), so running the
# orbits at a lower working precision silently drops the main branches.
DPS = 60
ROUNDS = 20_000
DELTAS = ("1e-3", "1e-6")


def mp_run(circuit, eps, q, rounds):
    qb = mp.mpf(1) - mp.mpf(str(q))
    t0 = mp.mpf(1) - mp.mpf(eps)
    s, t = t0, t0
    for _ in range(rounds):
        s, t = reduced_map(s, t, qb, circuit)
    return s, t, fidelity(s, t, circuit)


def f64_run(circuit, eps, q, rounds):
    p = isotropic_pops(float(eps))
    for _ in range(rounds):
        p = one_round(p, float(q), circuit)
    return p


def nearest(fps, s, t):
    best, bd = None, np.inf
    for fp in fps:
        d = max(abs(float(s) - fp["coord1"]), abs(float(t) - fp["coord2"]))
        if d < bd:
            best, bd = fp, d
    return best, bd


def main() -> int:
    mp.mp.dps = DPS
    bnd = [r for r in csv.DictReader(open(OUT / "basin_boundaries.csv"))
           if r["method"].startswith("bisection")]
    rows = []
    for b in bnd:
        circ, q = b["circuit"], b["q"]
        eps_star = mp.mpf(b["eps_below"])
        fps = all_fixed_points(mp.mpf(1) - mp.mpf(q), circ)
        fp_info = []
        for s, t in fps:
            rho = spectral_radius(jacobian(s, t, mp.mpf(1) - mp.mpf(q), circ))
            fp_info.append({"coord1": float(s), "coord2": float(t),
                            "fidelity": float(fidelity(s, t, circ)),
                            "spectral_radius": float(rho),
                            "stable": bool(rho < 1)})
        for d in DELTAS:
            for sgn, side in ((-1, "below"), (+1, "above")):
                eps = eps_star + sgn * mp.mpf(d)
                if eps <= 0 or eps >= 1:
                    continue
                s, t, f = mp_run(circ, mp.nstr(eps, 30), q, ROUNDS)
                fp, dist = nearest(fp_info, s, t)
                p64 = f64_run(circ, float(eps), q, ROUNDS)
                x64, z64 = float(p64[0] - p64[1] + p64[2] - p64[3]), \
                           float(p64[0] + p64[1] - p64[2] - p64[3])
                rows.append({
                    "circuit": circ, "q": q,
                    "eps_star": mp.nstr(eps_star, 18), "delta": d, "side": side,
                    "eps": mp.nstr(eps, 18), "rounds": ROUNDS,
                    "fid_highprec": mp.nstr(f, 16),
                    "fid_float64": repr(float(p64[0])),
                    "abs_diff_fidelity": mp.nstr(abs(f - mp.mpf(repr(float(p64[0])))), 6),
                    "nearest_fixed_point_fidelity": fp["fidelity"] if fp else "",
                    "nearest_fixed_point_stable": fp["stable"] if fp else "",
                    "dist_to_nearest": mp.nstr(mp.mpf(repr(dist)), 6),
                    # coord1 is x and coord2 is z in BOTH reduced charts
                    "float64_coord1": repr(x64), "float64_coord2": repr(z64),
                    "float64_matches_highprec": bool(
                        abs(x64 - float(s)) < 1e-8 and abs(z64 - float(t)) < 1e-8),
                })
                print(f"{circ:6s} q={q:8s} d={d:5s} {side:5s} "
                      f"F={mp.nstr(f, 12):>16}  -> nearest fp F="
                      f"{fp['fidelity'] if fp else float('nan'):.9f} "
                      f"(stable={fp['stable'] if fp else '?'}) dist={dist:.3e}  "
                      f"|f64-mp|={mp.nstr(abs(f - mp.mpf(repr(float(p64[0])))), 4)}")
    with open(OUT / "separatrix_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nseparatrix_check.csv: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
