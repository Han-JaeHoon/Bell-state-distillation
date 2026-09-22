"""How long the orbit is trapped near the saddle node (task sections 4.7, 11).

Just above the saddle-node q the target branch no longer exists, but the map
still has a narrow bottleneck where the two merged fixed points used to be.
An orbit entering it crawls for a long time before escaping -- a "ghost".  If
the bifurcation really is a saddle node, the passage time scales as

    N  ~  C (q - q_SN)^(-1/2).

This script measures the passage time in float64, fits the exponent, and
re-measures one point per circuit at mpmath dps = 60 to show the crawl is a
property of the map and not of double precision.

Below q_SN it measures the opposite quantity: the number of rounds needed to
settle onto the target branch, which diverges as q_SN is approached.

Output: results/data/repeated_dynamics_independent/transient_scaling.csv
"""

from __future__ import annotations

import csv
import json

import mpmath as mp
import numpy as np

from _bootstrap import DATA_DIR  # noqa: E402
from pqec_distill.repeated_dynamics import (  # noqa: E402
    CIRCUITS, isotropic_pops, one_round,
)
from analyze_repeated_dynamics import DPS, fidelity, reduced_map  # noqa: E402

OUT = DATA_DIR / "repeated_dynamics_independent"
EPS0 = 0.1
CAP = 2_000_000
DELTAS = ["1e-2", "3e-3", "1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6", "1e-6"]


def passage_rounds_f64(circuit, q, cap=CAP):
    """Rounds until the orbit has clearly left the bottleneck (F < 0.3) or
    has settled (step < 1e-15).  Returns (rounds, fidelity, why)."""
    p = isotropic_pops(EPS0)
    for n in range(1, cap + 1):
        nxt = one_round(p, q, circuit)
        step = float(np.max(np.abs(nxt - p)))
        p = nxt
        if p[0] < 0.3:
            return n, float(p[0]), "escaped to F<0.3"
        if step < 1e-15:
            return n, float(p[0]), "settled"
    return cap, float(p[0]), "hit cap"


def passage_rounds_mp(circuit, q, cap):
    qb = mp.mpf(1) - mp.mpf(q)
    t0 = mp.mpf(1) - mp.mpf(str(EPS0))
    s, t = t0, t0
    for n in range(1, cap + 1):
        s, t = reduced_map(s, t, qb, circuit)
        f = fidelity(s, t, circuit)
        if f < mp.mpf("0.3"):
            return n, f, "escaped to F<0.3"
    return cap, fidelity(s, t, circuit), "hit cap"


def main() -> int:
    mp.mp.dps = DPS
    crit = json.loads((OUT / "critical_points.json").read_text())
    rows = []
    fits = {}
    for circ in CIRCUITS:
        q_sn = mp.mpf(crit[circ]["q_saddle_node_discriminant_zero"])
        xs, ys = [], []
        for d in DELTAS:
            for sgn, side in ((+1, "above"), (-1, "below")):
                q = q_sn + sgn * mp.mpf(d)
                n, f, why = passage_rounds_f64(circ, float(q))
                rows.append({
                    "circuit": circ, "q_saddle_node": mp.nstr(q_sn, 20),
                    "delta": d, "side": side, "q": mp.nstr(q, 20),
                    "eps": EPS0, "rounds": n, "fidelity_at_stop": f,
                    "stop_reason": why, "precision": "float64",
                })
                if side == "above" and why == "escaped to F<0.3":
                    xs.append(float(mp.log(mp.mpf(d))))
                    ys.append(float(np.log(n)))
        if len(xs) >= 3:
            slope, intercept = np.polyfit(xs, ys, 1)
            fits[circ] = {"exponent": float(slope),
                          "prefactor": float(np.exp(intercept)),
                          "points": len(xs)}
            print(f"{circ:6s} passage time ~ C (q-q_SN)^p : "
                  f"p = {slope:+.6f}  C = {np.exp(intercept):.4f}  "
                  f"({len(xs)} points)")
        # one high-precision confirmation, at the tightest delta that fits
        d = "1e-5"
        q = q_sn + mp.mpf(d)
        n64, f64_, _ = passage_rounds_f64(circ, float(q))
        nmp, fmp, why = passage_rounds_mp(circ, mp.nstr(q, 30), min(4 * n64 + 100, 400_000))
        rows.append({
            "circuit": circ, "q_saddle_node": mp.nstr(q_sn, 20),
            "delta": d, "side": "above", "q": mp.nstr(q, 20),
            "eps": EPS0, "rounds": nmp, "fidelity_at_stop": mp.nstr(fmp, 16),
            "stop_reason": why, "precision": f"mpmath dps={DPS}",
        })
        print(f"   {circ}: delta=1e-5  float64 {n64} rounds vs "
              f"dps{DPS} {nmp} rounds (difference {nmp - n64})")

    with open(OUT / "transient_scaling.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "transient_scaling_fit.json").write_text(json.dumps(fits, indent=2))
    print(f"\ntransient_scaling.csv: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
