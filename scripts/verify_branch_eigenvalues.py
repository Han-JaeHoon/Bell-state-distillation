"""Closed-form eigenvalues on the off-plane branch, and the bifurcation it
undergoes with the maximally mixed state.

Derived here (not taken from any existing document):

On the off-plane branch the reduced Jacobian is triangular, because the
vanishing coordinate enters its own update equation only linearly:

  Type 3/4 (u = 0):   u' = 2 qbar^(k+2) u (1+v)/D   is proportional to u
  4Q      (b = 0):    b' = qbar^4 b [(1+qbar) + 2 qbar a]/D  is proportional to b

so the two eigenvalues are the two diagonal derivatives.  Writing

  M  = qbar^(k+1) (1 + qbar)      (Type 3/4)
  M  = qbar^3     (1 + qbar)      (4Q)

the branch exists exactly when M > 1, its fixed-point condition is D = M, and

  along-branch  eigenvalue  lambda_par = (2 - M) / M      (both families)
  transverse    eigenvalue  lambda_perp = 2 qbar (1+v*)/(1+qbar)   (Type 3/4)
                            lambda_perp = qbar + 2 qbar^5 a* / M   (4Q)

Two consequences, both checked numerically below:

1. M is ALSO one of the two closed-form eigenvalues of the maximally mixed
   fixed point.  So at M = 1 the off-plane branch is born exactly where the
   origin loses stability in that direction, and lambda_par = 1 there:
   a TRANSCRITICAL bifurcation, not a saddle node.
2. Type 4 has k = 2, so its M equals 4Q's M identically.  The two circuits
   therefore share the same off-plane existence condition and the same
   lambda_par at every q, although the branch coordinate and lambda_perp
   differ.

Output: results/data/repeated_dynamics_independent/offplane_eigenvalues.csv
"""

from __future__ import annotations

import csv

import mpmath as mp

from _bootstrap import DATA_DIR  # noqa: E402
from pqec_distill.repeated_dynamics import CIRCUITS  # noqa: E402
from analyze_repeated_dynamics import (  # noqa: E402
    DPS, K, branch_offplane, eigvals2, jacobian, origin_eigs,
)

OUT = DATA_DIR / "repeated_dynamics_independent"
QS = ["0.001", "0.01", "0.05", "0.08", "0.10", "0.12", "0.14", "0.15",
      "0.16", "0.17", "0.175", "0.178", "0.180", "0.1808"]


def bigM(qbar, circuit):
    k = 3 if circuit == "4Q" else K[circuit] + 1
    return qbar**k * (1 + qbar)


def closed_form(qbar, circuit, fp):
    M = bigM(qbar, circuit)
    lam_par = (2 - M) / M
    if circuit == "4Q":
        lam_perp = qbar + 2 * qbar**5 * fp[0] / M
    else:
        lam_perp = 2 * qbar * (1 + fp[1]) / (1 + qbar)
    return lam_par, lam_perp


def main() -> int:
    mp.mp.dps = DPS
    rows = []
    for circ in CIRCUITS:
        for qs in QS:
            qb = 1 - mp.mpf(qs)
            fp = branch_offplane(qb, circ)
            if fp is None:
                continue
            num = sorted((abs(e) for e in eigvals2(jacobian(fp[0], fp[1], qb, circ))))
            par, perp = closed_form(qb, circ, fp)
            pred = sorted((abs(par), abs(perp)))
            M = bigM(qb, circ)
            rows.append({
                "circuit": circ, "q": qs,
                "M": mp.nstr(M, 20),
                "coord1": mp.nstr(fp[0], 20), "coord2": mp.nstr(fp[1], 20),
                "lambda_parallel_closed_form": mp.nstr(par, 20),
                "lambda_perp_closed_form": mp.nstr(perp, 20),
                "eig_numeric_small": mp.nstr(num[0], 20),
                "eig_numeric_large": mp.nstr(num[1], 20),
                "max_abs_error": mp.nstr(max(abs(a - b) for a, b in zip(num, pred)), 6),
                "origin_eig_equals_M": mp.nstr(
                    min(abs(e - M) for e in origin_eigs(qb, circ)), 6),
            })
    with open(OUT / "offplane_eigenvalues.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    worst = max(mp.mpf(r["max_abs_error"]) for r in rows)
    worst_M = max(mp.mpf(r["origin_eig_equals_M"]) for r in rows)
    print(f"offplane_eigenvalues.csv: {len(rows)} rows")
    print(f"worst |closed form - numeric Jacobian eigenvalue| = {mp.nstr(worst, 6)}")
    print(f"worst |M - nearest origin eigenvalue|             = {mp.nstr(worst_M, 6)}")

    # Type 4 and 4Q share M, hence lambda_parallel, at every q
    by = {}
    for r in rows:
        by.setdefault(r["q"], {})[r["circuit"]] = r
    print("\nType4 vs 4Q, same q:")
    for q, d in by.items():
        if "Type4" in d and "4Q" in d:
            dM = abs(mp.mpf(d["Type4"]["M"]) - mp.mpf(d["4Q"]["M"]))
            dl = abs(mp.mpf(d["Type4"]["lambda_parallel_closed_form"])
                     - mp.mpf(d["4Q"]["lambda_parallel_closed_form"]))
            print(f"  q={q:8s} |M_T4 - M_4Q|={mp.nstr(dM, 4):>10}  "
                  f"|lam_par difference|={mp.nstr(dl, 4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
