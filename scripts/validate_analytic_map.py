"""Compare the analytic exact maps against the FROZEN circuit-only reference.

Path:  frozen CSV  ->  analytic evaluation  ->  row-by-row comparison.

The CSV is opened read-only and never regenerated; no circuit simulation runs
here.  The analytic side comes from pqec_distill.analytic_exact_map, which
imports nothing from this package.

Nothing is tuned to make the comparison succeed: discrepancies are reported
with their structure (q dependence, Bell component, input family), not fixed.
"""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from _bootstrap import DATA_DIR  # noqa: E402
from pqec_distill.analytic_exact_map import (  # noqa: E402
    K_TYPE3, K_TYPE4, analytic_map_4q, analytic_map_type3, analytic_map_type4,
    analytic_map_type34_population_form, analytic_normalization,
    analytic_success_4q, pauli_from_pops,
)

CSV = DATA_DIR / "circuit_reference" / "circuit_reference.csv"
PIN = ["p_in_PhiP", "p_in_PhiM", "p_in_PsiP", "p_in_PsiM"]
POUT = ["p_out_PhiP", "p_out_PhiM", "p_out_PsiP", "p_out_PsiM"]
BELL = ["Phi+", "Phi-", "Psi+", "Psi-"]
MAPS = {"Type3": analytic_map_type3, "Type4": analytic_map_type4, "4Q": analytic_map_4q}
PASS_TOL = 1e-12


def fmt(v):
    return np.array2string(np.asarray(v), precision=12, floatmode="fixed",
                           max_line_width=200)


def main() -> int:
    rows = list(csv.DictReader(open(CSV)))
    print(f"frozen reference: {CSV}")
    print(f"{len(rows)} rows (read-only; no circuit simulation in this script)\n")

    per = defaultdict(list)
    for r in rows:
        per[r["circuit"]].append(r)

    verdict = {}
    print("=" * 78)
    print(" A. Bell-population comparison   Delta_B = circuit - analytic")
    print("=" * 78)
    for circ in ("Type3", "Type4", "4Q"):
        rs = per[circ]
        errs, worst, worst_row = [], -1.0, None
        for r in rs:
            pin = np.array([float(r[k]) for k in PIN])
            q = float(r["q"])
            got = np.array([float(r[k]) for k in POUT])
            ana = MAPS[circ](pin, q)
            d = got - ana
            errs.append(d)
            m = float(np.max(np.abs(d)))
            if m > worst:
                worst, worst_row = m, (r, pin, q, got, ana, d)
        e = np.array(errs)
        mx, mae, rms = np.max(np.abs(e)), np.mean(np.abs(e)), np.sqrt(np.mean(e ** 2))
        verdict[circ] = mx < PASS_TOL
        print(f"\n--- {circ}  ({rs[0]['n_cnot']} CNOT) ---")
        print(f"  total rows          : {len(rs)}")
        print(f"  max |Delta_B|       : {mx:.3e}")
        print(f"  mean |Delta_B|      : {mae:.3e}")
        print(f"  RMS  Delta_B        : {rms:.3e}")
        r, pin, q, got, ana, d = worst_row
        print(f"  worst row: case={r['case']} ({r['case_kind']}), q={q}")
        print(f"    input    p = {fmt(pin)}")
        print(f"    circuit  p'= {fmt(got)}")
        print(f"    analytic p'= {fmt(ana)}")
        print(f"    diff       = {fmt(d)}")
        print(f"    per-component: " + ", ".join(f"{b}={v:+.2e}" for b, v in zip(BELL, d)))

    print("\n" + "=" * 78)
    print(" B. 4Q success probability   Delta P = P_circuit - N^(4Q)/4")
    print("=" * 78)
    dps, wr, wv = [], None, -1.0
    for r in per["4Q"]:
        pin = np.array([float(r[k]) for k in PIN])
        q = float(r["q"])
        dp = float(r["trace_weight"]) - analytic_success_4q(pin, q)
        dps.append(dp)
        if abs(dp) > wv:
            wv, wr = abs(dp), (r, pin, q, float(r["trace_weight"]), analytic_success_4q(pin, q))
    dps = np.array(dps)
    ok_p = float(np.max(np.abs(dps))) < PASS_TOL
    verdict["4Q_Psucc"] = ok_p
    print(f"  rows                     : {len(dps)}")
    print(f"  max |Delta P_succ|       : {np.max(np.abs(dps)):.3e}")
    print(f"  mean |Delta P_succ|      : {np.mean(np.abs(dps)):.3e}")
    r, pin, q, pc, pa = wr
    print(f"  worst row: case={r['case']}, q={q}")
    print(f"    input p    = {fmt(pin)}")
    print(f"    circuit P  = {pc:.15f}")
    print(f"    analytic P = {pa:.15f}   (diff {pc-pa:+.3e})")

    print("\n" + "=" * 78)
    print(" C. Type 3 / Type 4 trace_weight vs the analytic normalization N")
    print("=" * 78)
    print("  (the document does NOT identify these; tested, not assumed)")
    for circ in ("Type3", "Type4"):
        rs = per[circ]
        d_quarter, ratios, expo, d_pow = [], [], [], []
        for r in rs:
            pin = np.array([float(r[k]) for k in PIN])
            q = float(r["q"])
            n = analytic_normalization(pin, q, circ)
            w = float(r["trace_weight"])
            d_quarter.append(w - n / 4.0)
            if n != 0:
                ratios.append(w / n)
            if q > 0:
                expo.append(np.log(4.0 * w / n) / np.log(1.0 - q))
            d_pow.append(w - (1.0 - q) ** 10 * n / 4.0)
        print(f"  {circ}: max|w - N/4| = {np.max(np.abs(d_quarter)):.3e}, "
              f"w/N in [{min(ratios):.6f}, {max(ratios):.6f}]")
        print(f"    -> w != N/4 for q > 0.  Fitting w = qbar^m N/4 gives "
              f"m in [{min(expo):.10f}, {max(expo):.10f}]")
        print(f"    -> with m = 10 fixed: max |w - qbar^10 N/4| = "
              f"{np.max(np.abs(d_pow)):.3e}")
    # 4Q: the same test, where the document DOES assert P_succ = N/4 (i.e. m = 0)
    rs = per["4Q"]
    expo = [np.log(4.0 * float(r["trace_weight"])
                   / analytic_normalization(np.array([float(r[k]) for k in PIN]),
                                            float(r["q"]), "4Q")) / np.log(1.0 - float(r["q"]))
            for r in rs if float(r["q"]) > 0]
    print(f"  4Q: same fit gives m in [{min(expo):.3e}, {max(expo):.3e}] (i.e. m = 0, "
          f"no attenuation) -- consistent with P_succ = N/4")
    print("  -> EMPIRICAL relation, found from the frozen data in this step.  The")
    print("     supplied document states no Type 3/4 normalization identity, and the")
    print("     origin of the qbar^10 prefactor is NOT established here.")

    print("\n" + "=" * 78)
    print(" D. q = 0")
    print("=" * 78)
    for circ in ("Type3", "Type4", "4Q"):
        z = [r for r in per[circ] if float(r["q"]) == 0.0]
        e, esq = [], []
        for r in z:
            pin = np.array([float(r[k]) for k in PIN])
            got = np.array([float(r[k]) for k in POUT])
            e.append(np.max(np.abs(got - MAPS[circ](pin, 0.0))))
            sq = pin ** 2
            esq.append(np.max(np.abs(got - sq / sq.sum())))
        print(f"  {circ}: {len(z)} rows | vs analytic map {max(e):.3e} "
              f"| vs p_B^2/sum p_C^2 {max(esq):.3e}")

    print("\n" + "=" * 78)
    print(" E. Invariant planes (analytic map, symbolic)")
    print("=" * 78)
    import sympy as sp
    X, Y, Z, QB = sp.symbols("x y z qbar")

    def sym34(k):
        n = 1 + QB ** k * (X ** 2 + Y ** 2 + Z ** 2)
        return (2 * QB ** (k + 2) * (X - Y * Z) / n,
                2 * QB ** (k + 2) * (Y - X * Z) / n,
                QB ** (k + 1) * (1 + QB) * (Z - X * Y) / n)

    def sym4q():
        n = 1 + QB ** 5 * (X ** 2 + Y ** 2) + QB ** 3 * Z ** 2
        return (QB ** 3 * ((1 + QB) * X - 2 * QB ** 2 * Y * Z) / n,
                QB ** 4 * ((1 + QB) * Y - 2 * QB * X * Z) / n,
                QB ** 4 * ((1 + QB) * Z - 2 * QB * X * Y) / n)

    for name, syms in (("Type3", sym34(K_TYPE3)), ("Type4", sym34(K_TYPE4)), ("4Q", sym4q())):
        xp, yp, zp = syms
        on_yx = sp.simplify((yp + xp).subs(Y, -X))       # input y = -x  -> output y = -x ?
        on_yz = sp.simplify((yp + zp).subs(Y, -Z))       # input y = -z  -> output y = -z ?
        print(f"  {name}: (y'+x')|_(y=-x) = {on_yx}    (y'+z')|_(y=-z) = {on_yz}")
    print("  -> a vanishing entry means that plane is exactly invariant under the")
    print("     analytic map; this is a statement about those planes only, NOT about")
    print("     general Bell-diagonal states.")

    print("\n" + "=" * 78)
    print(" F. Analytic output is a physical Bell probability vector")
    print("=" * 78)
    for circ in ("Type3", "Type4", "4Q"):
        worst_sum, worst_neg = 0.0, 0.0
        for r in per[circ]:
            pin = np.array([float(r[k]) for k in PIN])
            ana = MAPS[circ](pin, float(r["q"]))
            worst_sum = max(worst_sum, abs(ana.sum() - 1.0))
            worst_neg = min(worst_neg, ana.min())
        print(f"  {circ}: max |sum p' - 1| = {worst_sum:.3e}, min p'_B = {worst_neg:+.3e}")

    print("\n" + "=" * 78)
    print(" G. Cross-check of the two transcriptions of Type 3 / Type 4")
    print("=" * 78)
    for circ, k in (("Type3", K_TYPE3), ("Type4", K_TYPE4)):
        w = 0.0
        for r in per[circ]:
            pin = np.array([float(r[k2]) for k2 in PIN])
            q = float(r["q"])
            w = max(w, float(np.max(np.abs(
                MAPS[circ](pin, q) - analytic_map_type34_population_form(pin, q, k)))))
        print(f"  {circ}: max |Pauli form - Bell-population form| = {w:.3e}")

    print("\n" + "=" * 78)
    print(" VERDICT   (PASS = agreement at floating-point level, tol "
          f"{PASS_TOL:.0e})")
    print("=" * 78)
    labels = [("Type 3 exact map", "Type3"), ("Type 4 exact map", "Type4"),
              ("4Q exact map", "4Q"), ("4Q success probability formula", "4Q_Psucc")]
    for lab, key in labels:
        print(f"  {lab:34s}: {'PASS' if verdict[key] else 'FAIL'}")
    return 0 if all(verdict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
