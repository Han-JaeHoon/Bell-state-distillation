"""Broad exploratory scan of the repeated dynamics (task section 4).

For each (circuit, eps, q) on a coarse grid, iterate the verified one-round map
from the Bell-isotropic initial state and record, WITHOUT assuming anything
about where it should go:

  * whether the orbit settles (final step size) and how fast,
  * whether a short cycle (period 2..8) is present in the tail,
  * whether the target fidelity is monotone, and its largest overshoot,
  * which non-target Bell component grows and which shrinks,
  * the exact equalities that hold along the orbit (invariant-plane check),
  * the endpoint, its residual under the full map, and its fidelity.

Output: results/data/repeated_dynamics_independent/broad_scan.csv
"""

from __future__ import annotations

import csv

import numpy as np

from _bootstrap import DATA_DIR  # noqa: E402
from pqec_distill.analytic_exact_map import pauli_from_pops  # noqa: E402
from pqec_distill.repeated_dynamics import (  # noqa: E402
    CIRCUITS, classify_trajectory, isotropic_pops, one_round,
    success_probability,
)

OUT = DATA_DIR / "repeated_dynamics_independent"
EPS = [round(0.05 * i, 3) for i in range(1, 21)]          # 0.05 .. 1.00
QS = ["0.0", "0.005", "0.01", "0.02", "0.05", "0.08", "0.10", "0.12", "0.13",
      "0.14", "0.15", "0.17", "0.18", "0.19", "0.20", "0.25", "0.30", "0.40"]
ROUNDS = 4000


def plane_defects(circuit, eps, q, rounds=200):
    """Largest |x+y| and |y+z| seen along the orbit (exact invariants are 0)."""
    p = isotropic_pops(eps)
    d_xy = d_yz = 0.0
    for _ in range(rounds):
        p = one_round(p, q, circuit)
        x, y, z = pauli_from_pops(p)
        d_xy = max(d_xy, abs(x + y))
        d_yz = max(d_yz, abs(y + z))
    return d_xy, d_yz


def main() -> int:
    rows = []
    for circ in CIRCUITS:
        for qs in QS:
            q = float(qs)
            for eps in EPS:
                c = classify_trajectory(circ, eps, q, max_rounds=ROUNDS, tol=0.0)
                end = c["endpoint"]
                p0 = isotropic_pops(eps)
                d_xy, d_yz = plane_defects(circ, eps, q)
                names = ("Phi+", "Phi-", "Psi+", "Psi-")
                grew = [names[i] for i in range(1, 4) if end[i] > p0[i] + 1e-12]
                shrank = [names[i] for i in range(1, 4) if end[i] < p0[i] - 1e-12]
                dom = names[1 + int(np.argmax(end[1:]))]
                ps = success_probability(end, q, circ)
                rows.append({
                    "circuit": circ, "q": qs, "eps": eps, "rounds": ROUNDS,
                    "fid_start": c["fid_start"], "fid_end": c["fid_end"],
                    "fid_max": c["fid_max"], "n_fid_max": c["fid_argmax"],
                    "overshoot": c["overshoot"],
                    "monotone_increasing": c["monotone_increasing"],
                    "monotone_decreasing": c["monotone_decreasing"],
                    "final_step": c["final_step"],
                    "residual_at_endpoint": c["residual_at_endpoint"],
                    "cycle_period": "" if c["cycle_period"] is None else c["cycle_period"],
                    "tail_spread": c["tail_spread"],
                    "p_end_PhiP": end[0], "p_end_PhiM": end[1],
                    "p_end_PsiP": end[2], "p_end_PsiM": end[3],
                    "dominant_nontarget": dom,
                    "grew": "|".join(grew), "shrank": "|".join(shrank),
                    "max_abs_x_plus_y": d_xy, "max_abs_y_plus_z": d_yz,
                    "P_succ_end": "" if ps is None else ps,
                })
    with open(OUT / "broad_scan.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"broad_scan.csv: {len(rows)} rows")

    per = [r for r in rows if r["cycle_period"] not in ("", 1)]
    print(f"orbits with a tail cycle of period > 1: {len(per)}")
    slow = [r for r in rows if r["final_step"] > 1e-12]
    print(f"orbits still moving after {ROUNDS} rounds (step > 1e-12): {len(slow)}")
    for r in sorted(slow, key=lambda r: -r["final_step"])[:12]:
        print(f"   {r['circuit']:6s} q={r['q']:6s} eps={r['eps']:<5} "
              f"step={r['final_step']:.3e} F={r['fid_end']:.9f}")
    print("\nlargest invariant-plane defect over the whole grid:")
    for circ in CIRCUITS:
        sub = [r for r in rows if r["circuit"] == circ]
        key = "max_abs_x_plus_y" if circ != "4Q" else "max_abs_y_plus_z"
        other = "max_abs_y_plus_z" if circ != "4Q" else "max_abs_x_plus_y"
        print(f"   {circ:6s} preserved combination: {max(r[key] for r in sub):.3e}   "
              f"other combination: {max(r[other] for r in sub):.3e}")
    nonmono = [r for r in rows if not r["monotone_increasing"]
               and not r["monotone_decreasing"]]
    print(f"\nnon-monotone fidelity histories: {len(nonmono)} / {len(rows)}")
    big = sorted(rows, key=lambda r: -r["overshoot"])[:8]
    print("largest fidelity overshoot (transient max above the endpoint):")
    for r in big:
        print(f"   {r['circuit']:6s} q={r['q']:6s} eps={r['eps']:<5} "
              f"F_max={r['fid_max']:.6f}@{r['n_fid_max']} F_end={r['fid_end']:.6f} "
              f"overshoot={r['overshoot']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
