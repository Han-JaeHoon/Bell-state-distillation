"""Independent repeated-dynamics analysis, from the verified one-round map only.

Pipeline:  verified one-round map -> invariant manifold -> reduced map ->
fixed points -> Jacobian -> bifurcations -> basins -> frozen result files.

Excluded on purpose (these already contain repeated-dynamics conclusions):
  pqec_distill.noisy_analytics, pqec_distill.repeated_noisy,
  pqec_distill.swap_test_reference, pqec_distill.analytics,
  scripts/repeated_noisy_dynamics.py, scripts/analyze_repeated_rounds.py,
  scripts/compare_with_swap_test.py, results/data/repeated_*.csv,
  results/data/compare_fixed_points.csv, docs/derivation.md ss13-14,
  docs/comparison_swap_test.md, results/verification_report.md ssJ.6-J.7,
  tests/test_repeated_noisy.py, tests/test_repeated_rounds.py

REDUCED MAPS, derived here from the one-round map (see analyze steps below).

Type 3 (k=4) and Type 4 (k=2), on the invariant plane y = -x, with
u = x = -y, v = z,  F = (1 + 2u + v)/4:

    D   = 1 + qbar^k (2u^2 + v^2)
    u'  = 2 qbar^(k+2) u (1 + v) / D
    v'  = qbar^(k+1) (1 + qbar) (u^2 + v) / D

4Q, on the invariant plane y = -z, with a = x, b = z = -y,
F = (1 + a + 2b)/4:

    D   = 1 + qbar^5 a^2 + (qbar^5 + qbar^3) b^2
    a'  = qbar^3 [ (1 + qbar) a + 2 qbar^2 b^2 ] / D
    b'  = qbar^4 b [ (1 + qbar) + 2 qbar a ] / D
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone

import mpmath as mp
import numpy as np

from _bootstrap import DATA_DIR  # noqa: E402
from pqec_distill.repeated_dynamics import CIRCUITS, isotropic_pops, trajectory  # noqa: E402

OUT = DATA_DIR / "repeated_dynamics_independent"
K = {"Type3": 4, "Type4": 2}
DPS = 60


# ===========================================================================
# reduced maps (mpmath, arbitrary precision)
# ===========================================================================

def reduced_map(s, t, qbar, circuit):
    """(u,v) -> (u',v') for Type 3/4; (a,b) -> (a',b') for 4Q."""
    if circuit == "4Q":
        d = 1 + qbar**5 * s**2 + (qbar**5 + qbar**3) * t**2
        return (qbar**3 * ((1 + qbar) * s + 2 * qbar**2 * t**2) / d,
                qbar**4 * t * ((1 + qbar) + 2 * qbar * s) / d)
    k = K[circuit]
    d = 1 + qbar**k * (2 * s**2 + t**2)
    return (2 * qbar**(k + 2) * s * (1 + t) / d,
            qbar**(k + 1) * (1 + qbar) * (s**2 + t) / d)


def fidelity(s, t, circuit):
    return (1 + s + 2 * t) / 4 if circuit == "4Q" else (1 + 2 * s + t) / 4


def initial_reduced(eps, circuit):
    """Bell-isotropic start has (x,y,z) = (t,-t,t), t = 1-eps, which lies on
    BOTH invariant planes; in either reduced chart it is (t, t)."""
    t = 1 - mp.mpf(eps)
    return (t, t)


def jacobian(s, t, qbar, circuit, h=None):
    """2x2 Jacobian of the reduced map by high-precision central differences."""
    if h is None:
        h = mp.mpf(10) ** (-DPS // 3)
    j = mp.zeros(2, 2)
    for col, (ds, dt) in enumerate(((h, 0), (0, h))):
        pp = reduced_map(s + ds, t + dt, qbar, circuit)
        pm = reduced_map(s - ds, t - dt, qbar, circuit)
        j[0, col] = (pp[0] - pm[0]) / (2 * h)
        j[1, col] = (pp[1] - pm[1]) / (2 * h)
    return j


def eigvals2(j):
    tr, det = j[0, 0] + j[1, 1], j[0, 0] * j[1, 1] - j[0, 1] * j[1, 0]
    disc = mp.sqrt(tr * tr - 4 * det)
    return ((tr + disc) / 2, (tr - disc) / 2)


def spectral_radius(j):
    return max(abs(e) for e in eigvals2(j))


# ===========================================================================
# fixed-point branches, from the quadratics derived symbolically
# ===========================================================================

def branch_offplane(qbar, circuit):
    """The branch with the second coordinate zero (Type 3/4: u=0; 4Q: b=0).

    Type 3/4:  v^2 = [qbar^(k+1)(1+qbar) - 1] / qbar^k     (u = 0)
    4Q:        a^2 = [qbar^3 (1+qbar)      - 1] / qbar^5   (b = 0)
    Returns the positive root, or None when it does not exist.
    """
    if circuit == "4Q":
        num, den = qbar**3 * (1 + qbar) - 1, qbar**5
        sq = num / den
        return (mp.sqrt(sq), mp.mpf(0)) if sq > 0 else None
    k = K[circuit]
    sq = (qbar**(k + 1) * (1 + qbar) - 1) / qbar**k
    return (mp.mpf(0), mp.sqrt(sq)) if sq > 0 else None


def branch_quadratic_coeffs(qbar, circuit):
    """Coefficients (A, B, C) of the fixed-point quadratic on the main branch.

    Type 3/4 (variable v):
        qbar^k (5 qbar + 1) v^2 - 2 qbar^k (qbar^3+qbar^2-qbar+1) v
            - (qbar+1)(2 qbar^(k+2) - 1) = 0
        with u^2 = v (2 qbar v + qbar - 1) / (1 + qbar)

    4Q (variable a):
        2 qbar^3 (2 qbar^2 + 1) a^2 - qbar (3 qbar^4 + 1) a
            - 2 (qbar^5 + qbar^4 - 1) = 0
        with b^2 = a (2 a qbar^2 + qbar^2 - 1) / (2 qbar^2)
    """
    if circuit == "4Q":
        return (2 * qbar**3 * (2 * qbar**2 + 1),
                -qbar * (3 * qbar**4 + 1),
                -2 * (qbar**5 + qbar**4 - 1))
    k = K[circuit]
    return (qbar**k * (5 * qbar + 1),
            -2 * qbar**k * (qbar**3 + qbar**2 - qbar + 1),
            -(qbar + 1) * (2 * qbar**(k + 2) - 1))


def branch_discriminant(qbar, circuit):
    A, B, C = branch_quadratic_coeffs(qbar, circuit)
    return B * B - 4 * A * C


def branches_main(qbar, circuit):
    """Both roots of the quadratic, kept only when the partner coordinate
    squared is non-negative.  Returns a list of (s, t) fixed points."""
    A, B, C = branch_quadratic_coeffs(qbar, circuit)
    disc = B * B - 4 * A * C
    if disc < 0:
        return []
    out = []
    for root in ((-B + mp.sqrt(disc)) / (2 * A), (-B - mp.sqrt(disc)) / (2 * A)):
        if circuit == "4Q":
            a = root
            sq = a * (2 * a * qbar**2 + qbar**2 - 1) / (2 * qbar**2)
            if sq >= 0:
                out.append((a, mp.sqrt(sq)))
        else:
            v = root
            sq = v * (2 * qbar * v + qbar - 1) / (1 + qbar)
            if sq >= 0:
                out.append((mp.sqrt(sq), v))
    return out


def all_fixed_points(qbar, circuit):
    """Origin + off-plane branch + both main-branch roots, verified."""
    cands = [(mp.mpf(0), mp.mpf(0))]
    off = branch_offplane(qbar, circuit)
    if off is not None:
        cands.append(off)
    cands += branches_main(qbar, circuit)
    good = []
    for s, t in cands:
        sp_, tp = reduced_map(s, t, qbar, circuit)
        if max(abs(sp_ - s), abs(tp - t)) < mp.mpf(10) ** (-DPS + 10):
            good.append((s, t))
    return good


def target_branch(qbar, circuit):
    """The fixed point continuously connected to Phi+ as q -> 0 (highest F
    among the verified ones with the off-plane coordinate non-zero)."""
    fps = [fp for fp in branches_main(qbar, circuit)]
    if not fps:
        return None
    return max(fps, key=lambda p: fidelity(p[0], p[1], circuit))


# ===========================================================================
# critical q from analytic conditions (high-precision root finding)
# ===========================================================================

def origin_eigs(qbar, circuit):
    """Jacobian eigenvalues at the maximally mixed state, in closed form."""
    if circuit == "4Q":
        return (qbar**3 * (1 + qbar), qbar**4 * (1 + qbar))
    k = K[circuit]
    return (2 * qbar**(k + 2), qbar**(k + 1) * (1 + qbar))


def _root(f, lo, hi):
    try:
        return mp.findroot(f, (mp.mpf(lo), mp.mpf(hi)), solver="anderson",
                           tol=mp.mpf(10) ** (-2 * DPS // 3))
    except Exception:
        return None


def find_critical(circuit):
    """Locate the critical q values from analytic conditions, not grid search."""
    out = {}

    # (1) saddle node: discriminant of the fixed-point quadratic vanishes
    r = _root(lambda q: branch_discriminant(1 - q, circuit), 0.05, 0.45)
    if r is not None and branch_discriminant(1 - r, circuit) is not None:
        out["q_saddle_node_discriminant_zero"] = r

    # (2) off-plane branch appears/disappears (its squared coordinate hits 0)
    if circuit == "4Q":
        f = lambda q: (1 - q)**3 * (2 - q) - 1
    else:
        k = K[circuit]
        f = lambda q: (1 - q)**(k + 1) * (2 - q) - 1
    r = _root(f, 0.05, 0.45)
    if r is not None:
        out["q_offplane_branch_vanishes"] = r

    # (3) origin eigenvalues crossing 1
    for i, lab in enumerate(("lambda1", "lambda2")):
        r = _root(lambda q: origin_eigs(1 - q, circuit)[i] - 1, 0.01, 0.6)
        if r is not None:
            out[f"q_origin_{lab}_eq_1"] = r

    # (4) main (target) branch: spectral radius crossing 1
    def rho_target(q):
        qb = 1 - q
        fp = target_branch(qb, circuit)
        if fp is None:
            return mp.mpf(1e3)
        return spectral_radius(jacobian(fp[0], fp[1], qb, circuit)) - 1
    lo = mp.mpf("0.02")
    hi = out.get("q_saddle_node_discriminant_zero", mp.mpf("0.3")) - mp.mpf("1e-9")
    if rho_target(lo) * rho_target(hi) < 0:
        out["q_target_branch_spectral_radius_eq_1"] = _root(rho_target, lo, hi)

    # (5) off-plane branch: spectral radius crossing 1
    def rho_off(q):
        qb = 1 - q
        fp = branch_offplane(qb, circuit)
        if fp is None:
            return mp.mpf(1e3)
        return spectral_radius(jacobian(fp[0], fp[1], qb, circuit)) - 1
    lim = out.get("q_offplane_branch_vanishes", mp.mpf("0.3"))
    lo2 = mp.mpf("0.02")
    if rho_off(lo2) * rho_off(lim - mp.mpf("1e-9")) < 0:
        out["q_offplane_branch_spectral_radius_eq_1"] = _root(
            rho_off, lo2, lim - mp.mpf("1e-9"))

    # (6) target-branch fidelity crossing 1/2 (entanglement of the Bell-diagonal state)
    def fid_half(q):
        qb = 1 - q
        fp = target_branch(qb, circuit)
        if fp is None:
            return mp.mpf(-1)
        return fidelity(fp[0], fp[1], circuit) - mp.mpf("0.5")
    hi3 = out.get("q_saddle_node_discriminant_zero", mp.mpf("0.3")) - mp.mpf("1e-9")
    if fid_half(mp.mpf("0.02")) * fid_half(hi3) < 0:
        out["q_target_branch_fidelity_eq_half"] = _root(fid_half, 0.02, hi3)
    return out


# ===========================================================================
# driver
# ===========================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dps", type=int, default=DPS)
    args = ap.parse_args()
    mp.mp.dps = args.dps
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- fixed-point branches vs q --------------------------------
    qgrid = [mp.mpf(s) for s in
             ("0.001", "0.005", "0.01", "0.02", "0.03", "0.05", "0.07", "0.10",
              "0.12", "0.13", "0.14", "0.15", "0.16", "0.17", "0.175", "0.18",
              "0.1805", "0.181", "0.185", "0.19", "0.20", "0.22", "0.25", "0.30")]
    fp_rows = []
    for circ in CIRCUITS:
        for q in qgrid:
            qb = 1 - q
            for kind, fp in (
                    [("origin", (mp.mpf(0), mp.mpf(0)))]
                    + ([("offplane", branch_offplane(qb, circ))]
                       if branch_offplane(qb, circ) else [])
                    + [(f"main{i}", f) for i, f in enumerate(branches_main(qb, circ))]):
                if fp is None:
                    continue
                s, t = fp
                j = jacobian(s, t, qb, circ)
                e1, e2 = eigvals2(j)
                res = max(abs(x - y) for x, y in zip(reduced_map(s, t, qb, circ), (s, t)))
                fp_rows.append({
                    "circuit": circ, "q": mp.nstr(q, 12), "branch": kind,
                    "coord1": mp.nstr(s, 20), "coord2": mp.nstr(t, 20),
                    "fidelity": mp.nstr(fidelity(s, t, circ), 20),
                    "lambda1": mp.nstr(e1, 16), "lambda2": mp.nstr(e2, 16),
                    "spectral_radius": mp.nstr(max(abs(e1), abs(e2)), 16),
                    "stable": bool(max(abs(e1), abs(e2)) < 1),
                    "residual": mp.nstr(res, 6),
                })
    with open(OUT / "fixed_points.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fp_rows[0])); w.writeheader(); w.writerows(fp_rows)
    print(f"fixed_points.csv: {len(fp_rows)} rows")

    # ---------- critical points -------------------------------------------
    crit = {}
    for circ in CIRCUITS:
        c = find_critical(circ)
        crit[circ] = {k2: mp.nstr(v2, 20) for k2, v2 in c.items() if v2 is not None}
        print(f"\n{circ} critical q:")
        for k2, v2 in crit[circ].items():
            print(f"   {k2:46s} {v2}")
    (OUT / "critical_points.json").write_text(json.dumps(crit, indent=2))

    # ---------- representative trajectories --------------------------------
    reps = {
        "Type3": [(0.1, 0.01, "low q, fast convergence"), (0.1, 0.12, "near saddle node"),
                  (0.5, 0.13, "basin A"), (0.9, 0.13, "basin B"),
                  (0.1, 0.132, "beyond saddle node"), (0.3, 0.1305, "long transient")],
        "Type4": [(0.1, 0.01, "low q, fast convergence"), (0.1, 0.17, "near saddle node"),
                  (0.5, 0.185, "basin A"), (0.95, 0.185, "basin B"),
                  (0.1, 0.20, "beyond saddle node"), (0.3, 0.189, "long transient")],
        "4Q": [(0.1, 0.01, "low q, fast convergence"), (0.1, 0.17, "near saddle node"),
               (0.5, 0.178, "basin A"), (0.95, 0.178, "basin B"),
               (0.1, 0.19, "beyond saddle node"), (0.3, 0.1806, "long transient")],
    }
    tr_rows = []
    for circ, cases in reps.items():
        for eps, q, label in cases:
            t = trajectory(circ, eps, q, max_rounds=3000, tol=1e-15)
            for r in t["rows"]:
                if r["n"] > 400 and r["n"] % 25:
                    continue
                row = {"circuit": circ, "label": label, "eps": eps, "q": q, **r}
                tr_rows.append(row)
    keys = sorted({k for r in tr_rows for k in r})
    with open(OUT / "representative_trajectories.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(tr_rows)
    print(f"\nrepresentative_trajectories.csv: {len(tr_rows)} rows")

    # ---------- metadata ---------------------------------------------------
    meta = {
        "purpose": "repeated dynamics derived independently from the verified one-round map",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, check=True).stdout.strip(),
        "precision_dps": args.dps,
        "root_solver": "mpmath.findroot (anderson), analytic conditions, not grid search",
        "invariant_planes": {"Type3": "y = -x", "Type4": "y = -x", "4Q": "y = -z"},
        "reduced_maps": {
            "Type3/Type4": ("D = 1 + qbar^k (2u^2+v^2); "
                            "u' = 2 qbar^(k+2) u (1+v)/D; "
                            "v' = qbar^(k+1)(1+qbar)(u^2+v)/D;  k=4 (Type3), 2 (Type4); "
                            "F = (1+2u+v)/4"),
            "4Q": ("D = 1 + qbar^5 a^2 + (qbar^5+qbar^3) b^2; "
                   "a' = qbar^3[(1+qbar)a + 2 qbar^2 b^2]/D; "
                   "b' = qbar^4 b[(1+qbar) + 2 qbar a]/D;  F = (1+a+2b)/4"),
        },
        "fixed_point_quadratics": {
            "Type3/Type4": ("qbar^k(5qbar+1) v^2 - 2 qbar^k(qbar^3+qbar^2-qbar+1) v "
                            "- (qbar+1)(2 qbar^(k+2)-1) = 0, "
                            "u^2 = v(2 qbar v + qbar - 1)/(1+qbar)"),
            "4Q": ("2 qbar^3(2qbar^2+1) a^2 - qbar(3qbar^4+1) a - 2(qbar^5+qbar^4-1) = 0, "
                   "b^2 = a(2 a qbar^2 + qbar^2 - 1)/(2 qbar^2)"),
        },
        "offplane_branches": {
            "Type3/Type4": "u = 0, v^2 = [qbar^(k+1)(1+qbar) - 1]/qbar^k",
            "4Q": "b = 0, a^2 = [qbar^3(1+qbar) - 1]/qbar^5",
        },
        "origin_eigenvalues": {
            "Type3/Type4": "2 qbar^(k+2)  and  qbar^(k+1)(1+qbar)",
            "4Q": "qbar^3(1+qbar)  and  qbar^4(1+qbar)",
        },
        "excluded": ["pqec_distill.noisy_analytics", "pqec_distill.repeated_noisy",
                     "pqec_distill.swap_test_reference", "pqec_distill.analytics",
                     "scripts/repeated_noisy_dynamics.py", "scripts/analyze_repeated_rounds.py",
                     "scripts/compare_with_swap_test.py", "results/data/repeated_*.csv",
                     "results/data/compare_fixed_points.csv", "docs/derivation.md ss13-14",
                     "docs/comparison_swap_test.md", "results/verification_report.md ssJ.6-J.7",
                     "tests/test_repeated_noisy.py", "tests/test_repeated_rounds.py"],
        "disclosure": ("Prior repeated-dynamics numbers from earlier in the same working "
                       "session were present in the analyst's context; the derivation here "
                       "was redone from the one-round map and no stored result file was read, "
                       "but this is an independent derivation path, not a blind replication."),
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
