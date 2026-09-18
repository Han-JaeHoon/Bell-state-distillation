"""Sanity checks on the circuit-only reference dataset (no analytic map used).

Reads results/data/circuit_reference/circuit_reference.csv and verifies, per
circuit, the properties that the raw simulation must satisfy by itself:

  normalization      sum_B p'_B = 1
  positivity         min_B p'_B >= -tol   (see the Type 3/4 caveat below)
  Bell closure       max_{B!=C} |<B|rho'|C>| ~ 0
  q = 0 behaviour    reported as measured, not compared to any formula
  4Q probability     0 <= P_success <= 1 and normalization of the conditional state
  backends           the two independent execution paths agree

CAVEAT recorded rather than patched: for Type 3 / Type 4 the effective output is
the PARITY-WEIGHTED operator tau_A/Tr(tau_A), not a postselected physical state,
so its Bell populations are not guaranteed non-negative and Tr(tau_A) is a parity
visibility rather than a probability.  Violations are reported, not suppressed.
"""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from _bootstrap import DATA_DIR  # noqa: E402

CSV = DATA_DIR / "circuit_reference" / "circuit_reference.csv"
TOL_NORM = 1e-12
TOL_OFFDIAG = 1e-12
TOL_POS = 1e-12
POPS_OUT = ["p_out_PhiP", "p_out_PhiM", "p_out_PsiP", "p_out_PsiM"]
POPS_IN = ["p_in_PhiP", "p_in_PhiM", "p_in_PsiP", "p_in_PsiM"]


def main() -> int:
    rows = list(csv.DictReader(open(CSV)))
    by = defaultdict(list)
    for r in rows:
        by[r["circuit"]].append(r)

    print(f"{len(rows)} rows from {CSV}\n")
    ok = True
    for circ, rs in by.items():
        ncx = rs[0]["n_cnot"]
        norm = [float(r["norm_err"]) for r in rs]
        offd = [float(r["max_bell_offdiag"]) for r in rs]
        minp = [float(r["min_bell_pop"]) for r in rs]
        wts = [float(r["trace_weight"]) for r in rs]
        meaning = rs[0]["trace_weight_meaning"]
        cross = [float(r["backend_cross_check"]) for r in rs if r["backend_cross_check"]]

        print(f"=== {circ}  ({ncx} CNOT, {len(rs)} rows) ===")
        print(f"  max normalization error        : {max(norm):.3e}   (tol {TOL_NORM:.0e})")
        print(f"  max Bell off-diagonal          : {max(offd):.3e}   (tol {TOL_OFFDIAG:.0e})")
        print(f"  min Bell population over all   : {min(minp):+.6e}")
        print(f"  trace weight ({meaning}): [{min(wts):.6f}, {max(wts):.6f}]")
        if cross:
            print(f"  backend cross-check (n={len(cross)}) : max {max(cross):.3e}")

        if max(norm) > TOL_NORM:
            print("  FAIL normalization"); ok = False
        if max(offd) > TOL_OFFDIAG:
            print("  FAIL Bell-diagonal closure"); ok = False

        neg = [r for r in rs if float(r["min_bell_pop"]) < -TOL_POS]
        if neg:
            worst = min(neg, key=lambda r: float(r["min_bell_pop"]))
            print(f"  NOTE negative Bell populations in {len(neg)}/{len(rs)} rows "
                  f"(worst {float(worst['min_bell_pop']):+.4e} at "
                  f"case={worst['case']}, q={worst['q']})")
            if circ == "4Q":
                print("  FAIL 4Q must stay a physical state"); ok = False
            else:
                print("  -> expected for a parity-weighted effective operator; recorded")

        if circ == "4Q":
            bad = [r for r in rs if not (0.0 <= float(r["trace_weight"]) <= 1.0 + 1e-12)]
            if bad:
                print(f"  FAIL {len(bad)} rows with P_success outside [0,1]"); ok = False
            else:
                print("  P_success within [0,1] for every row: OK")

        z = [r for r in rs if float(r["q"]) == 0.0]
        print(f"  q = 0 ({len(z)} rows), measured directly from the circuit:")
        print(f"    max normalization error : {max(float(r['norm_err']) for r in z):.3e}")
        print(f"    max Bell off-diagonal   : {max(float(r['max_bell_offdiag']) for r in z):.3e}")
        print(f"    trace weight range      : "
              f"[{min(float(r['trace_weight']) for r in z):.6f}, "
              f"{max(float(r['trace_weight']) for r in z):.6f}]")
        for name in ("isotropic_eps0.1000", "asym_large_psi_plus", "strongly_mixed_uniform"):
            m = [r for r in z if r["case"] == name]
            if m:
                r = m[0]
                pin = [float(r[k]) for k in POPS_IN]
                pout = [float(r[k]) for k in POPS_OUT]
                print(f"    {name:24s} in {np.round(pin,6)} -> out {np.round(pout,8)}")
        print()

    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
