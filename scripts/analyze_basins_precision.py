"""Basin structure, precision cross-check and endpoint verification.

Companion to ``analyze_repeated_dynamics.py``; same rule set -- everything is
derived from the verified one-round map (``pqec_distill.analytic_exact_map``)
and from the reduced maps re-derived in the companion script.  No stored
repeated-dynamics result file, document or module is read.

Three questions, three frozen output files:

  basin_scan.csv / basin_boundaries.csv
      Along the Bell-isotropic initial line rho_0(eps), at a fixed q, which
      attractor does the orbit reach?  Endpoints are NOT classified by a
      convergence tolerance alone: each endpoint is matched against the
      analytically computed fixed points of the reduced map, and the match is
      accepted only when the one-round residual at the endpoint is small AND
      the local spectral radius there is < 1.

  precision_check.csv
      float64 orbit vs mpmath orbit at dps = 60, for the cases where the
      double-precision run was slow or ambiguous.

  endpoint_verification.csv
      For every endpoint reported anywhere here: residual of the FULL
      4-component one-round map (not the reduced one), distance to the nearest
      analytic fixed point, and the local spectral radius.
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone

import mpmath as mp
import numpy as np

from _bootstrap import DATA_DIR  # noqa: E402
from pqec_distill.repeated_dynamics import (  # noqa: E402
    CIRCUITS, isotropic_pops, one_round,
)
from pqec_distill.analytic_exact_map import pauli_from_pops  # noqa: E402

from analyze_repeated_dynamics import (  # noqa: E402
    all_fixed_points, fidelity, jacobian, reduced_map, spectral_radius,
)

OUT = DATA_DIR / "repeated_dynamics_independent"
DPS = 60
mp.mp.dps = DPS

# how close an endpoint must sit to an analytic fixed point to be called that
# fixed point, and how small the one-round residual there must be
MATCH_DIST = 1e-6
MATCH_RESID = 1e-9


# ---------------------------------------------------------------------------
# reduced coordinates
# ---------------------------------------------------------------------------

def to_reduced(p, circuit):
    """(u,v) for Type 3/4 [u = x = -y, v = z]; (a,b) for 4Q [a = x, b = z = -y]."""
    x, y, z = pauli_from_pops(p)
    return (float(x), float(z))


def offplane_defect(p, circuit):
    """How far the state is off its invariant plane (should stay ~0)."""
    x, y, z = pauli_from_pops(p)
    return float(abs(y + x)) if circuit != "4Q" else float(abs(y + z))


def analytic_fixed_points(q, circuit):
    """Verified fixed points of the reduced map at this q, as floats, with
    their fidelity and local spectral radius."""
    qb = mp.mpf(1) - mp.mpf(str(q))
    out = []
    for s, t in all_fixed_points(qb, circuit):
        rho = spectral_radius(jacobian(s, t, qb, circuit))
        out.append({
            "coord1": float(s), "coord2": float(t),
            "fidelity": float(fidelity(s, t, circuit)),
            "spectral_radius": float(rho),
            "stable": bool(rho < 1),
        })
    return out


def match_endpoint(p, q, circuit, fps):
    """Name the attractor an orbit landed on, using the analytic fixed points.

    Returns (label, distance, residual_full, spectral_radius_or_None).
    ``label`` is "unresolved" whenever the endpoint is not a verified stable
    fixed point -- a small step size alone never earns a label.
    """
    s, t = to_reduced(p, circuit)
    resid_full = float(np.max(np.abs(one_round(p, q, circuit) - np.asarray(p))))
    best, bestd = None, np.inf
    for fp in fps:
        d = max(abs(s - fp["coord1"]), abs(t - fp["coord2"]))
        if d < bestd:
            best, bestd = fp, d
    if best is None:
        return "unresolved", np.inf, resid_full, None
    if bestd < MATCH_DIST and resid_full < MATCH_RESID and best["stable"]:
        label = f"F={best['fidelity']:.12f}"
    else:
        label = "unresolved"
    return label, float(bestd), resid_full, best["spectral_radius"]


def run_to_end(circuit, eps, q, max_rounds=200_000, tol=1e-16):
    """float64 orbit from the Bell-isotropic start; returns endpoint + stats."""
    p = isotropic_pops(eps)
    fid_max, n_max, step = float(p[0]), 0, np.inf
    defect = 0.0
    for n in range(max_rounds):
        nxt = one_round(p, q, circuit)
        step = float(np.max(np.abs(nxt - p)))
        p = nxt
        defect = max(defect, offplane_defect(p, circuit))
        if p[0] > fid_max:
            fid_max, n_max = float(p[0]), n + 1
        if step < tol:
            return p, n + 1, step, fid_max, n_max, defect
    return p, max_rounds, step, fid_max, n_max, defect


# ---------------------------------------------------------------------------
# mpmath orbit (independent precision)
# ---------------------------------------------------------------------------

def mp_orbit(circuit, eps, q, rounds):
    """Same dynamics at dps=60, straight from the reduced map."""
    qb = mp.mpf(1) - mp.mpf(str(q))
    t0 = mp.mpf(1) - mp.mpf(str(eps))
    s, t = t0, t0  # isotropic start is (t,-t,t): (u,v) = (a,b) = (t,t)
    for _ in range(rounds):
        s, t = reduced_map(s, t, qb, circuit)
    return s, t, fidelity(s, t, circuit)


# ---------------------------------------------------------------------------
# basin scan
# ---------------------------------------------------------------------------

# q values chosen per circuit to straddle the saddle node found in the
# companion script (Type3 ~0.1306, Type4 ~0.1894, 4Q ~0.1807)
SCAN_Q = {
    "Type3": ["0.05", "0.10", "0.12", "0.125", "0.128", "0.129", "0.130",
              "0.1305", "0.13057", "0.131", "0.14"],
    "Type4": ["0.05", "0.14", "0.17", "0.18", "0.185", "0.188", "0.189",
              "0.1894", "0.18941", "0.190", "0.20"],
    "4Q":    ["0.05", "0.14", "0.17", "0.175", "0.178", "0.180", "0.1805",
              "0.18066", "0.1807", "0.182", "0.19"],
}
SCAN_EPS = [round(0.02 * i, 3) for i in range(1, 51)]  # 0.02 .. 1.00


def basin_scan():
    rows = []
    for circ in CIRCUITS:
        for qs in SCAN_Q[circ]:
            q = float(qs)
            fps = analytic_fixed_points(q, circ)
            for eps in SCAN_EPS:
                p, n, step, fmax, nmax, defect = run_to_end(circ, eps, q)
                label, dist, resid, rho = match_endpoint(p, q, circ, fps)
                rows.append({
                    "circuit": circ, "q": qs, "eps": eps,
                    "attractor": label,
                    "fid_end": float(p[0]),
                    "fid_max": fmax, "n_fid_max": nmax,
                    "rounds": n, "final_step": step,
                    "residual_full_map": resid,
                    "dist_to_fixed_point": dist,
                    "spectral_radius_at_match": rho,
                    "offplane_defect_max": defect,
                    "n_stable_fixed_points": sum(1 for f in fps if f["stable"]),
                })
    return rows


def basin_boundaries(scan_rows, halvings=50, max_rounds=60_000):
    """Bisect the eps at which the attractor label changes, at fixed q.

    A transition whose upper side is 'unresolved' is NOT bisected: there the
    upper endpoint is eps = 1, i.e. the maximally mixed state itself, which is
    an exact fixed point of the map and is reached only from exactly eps = 1
    while it is unstable.  Bisecting towards it would report a boundary that
    is an artefact of the initial condition, not of the basin.
    """
    out = []
    by = {}
    for r in scan_rows:
        by.setdefault((r["circuit"], r["q"]), []).append(r)
    for (circ, qs), rows in sorted(by.items()):
        q = float(qs)
        fps = analytic_fixed_points(q, circ)
        rows.sort(key=lambda r: r["eps"])
        for a, b in zip(rows, rows[1:]):
            if a["attractor"] == b["attractor"]:
                continue
            if "unresolved" in (a["attractor"], b["attractor"]):
                out.append({
                    "circuit": circ, "q": qs,
                    "eps_below": repr(a["eps"]), "attractor_below": a["attractor"],
                    "eps_above": repr(b["eps"]), "attractor_above": b["attractor"],
                    "bracket_width": b["eps"] - a["eps"],
                    "fid_below": a["fid_end"], "fid_above": b["fid_end"],
                    "method": "not bisected (one side is the exact maximally mixed fixed point)",
                })
                continue
            lo, hi = a["eps"], b["eps"]
            lab_lo, lab_hi = a["attractor"], b["attractor"]
            f_lo, f_hi = a["fid_end"], b["fid_end"]
            for _ in range(halvings):
                mid = (lo + hi) / 2
                if mid <= lo or mid >= hi:
                    break
                p_, *_rest = run_to_end(circ, mid, q, max_rounds=max_rounds)
                lab, _d, _r, _s = match_endpoint(p_, q, circ, fps)
                if lab == lab_lo:
                    lo, f_lo = mid, float(p_[0])
                else:
                    hi, lab_hi, f_hi = mid, lab, float(p_[0])
            out.append({
                "circuit": circ, "q": qs,
                "eps_below": repr(lo), "attractor_below": lab_lo,
                "eps_above": repr(hi), "attractor_above": lab_hi,
                "bracket_width": hi - lo,
                "fid_below": f_lo, "fid_above": f_hi,
                "method": f"bisection, {halvings} halvings, max_rounds={max_rounds}",
            })
    return out


# ---------------------------------------------------------------------------
# precision cross-check
# ---------------------------------------------------------------------------

PRECISION_CASES = [
    # (circuit, eps, q, rounds)  -- the slow / near-critical cases
    ("Type3", 0.1, 0.13057, 4000), ("Type3", 0.3, 0.1305, 4000),
    ("Type3", 0.1, 0.131, 4000), ("Type3", 0.9, 0.13, 4000),
    ("Type4", 0.1, 0.18941, 4000), ("Type4", 0.3, 0.189, 4000),
    ("Type4", 0.1, 0.18, 4000), ("Type4", 0.95, 0.185, 4000),
    ("4Q", 0.1, 0.18066, 4000), ("4Q", 0.3, 0.1806, 4000),
    ("4Q", 0.1, 0.18, 4000), ("4Q", 0.95, 0.178, 4000),
]


def precision_check():
    rows = []
    for circ, eps, q, rounds in PRECISION_CASES:
        # float64, same number of rounds, no early stop
        p = isotropic_pops(eps)
        for _ in range(rounds):
            p = one_round(p, q, circ)
        s64, t64 = to_reduced(p, circ)
        f64 = float(p[0])
        s60, t60, f60 = mp_orbit(circ, eps, q, rounds)
        rows.append({
            "circuit": circ, "eps": eps, "q": q, "rounds": rounds,
            "fid_float64": repr(f64), "fid_dps60": mp.nstr(f60, 20),
            "abs_diff_fidelity": mp.nstr(abs(mp.mpf(repr(f64)) - f60), 6),
            "coord1_float64": repr(s64), "coord1_dps60": mp.nstr(s60, 20),
            "coord2_float64": repr(t64), "coord2_dps60": mp.nstr(t60, 20),
            "max_coord_diff": mp.nstr(max(abs(mp.mpf(repr(s64)) - s60),
                                          abs(mp.mpf(repr(t64)) - t60)), 6),
        })
    return rows


# ---------------------------------------------------------------------------
# endpoint verification
# ---------------------------------------------------------------------------

def endpoint_verification(scan_rows):
    """Every distinct (circuit, q, attractor) endpoint, checked against the
    FULL 4-component map and against the analytic fixed point."""
    seen, rows = set(), []
    for r in scan_rows:
        key = (r["circuit"], r["q"], r["attractor"])
        if key in seen:
            continue
        seen.add(key)
        circ, q, eps = r["circuit"], float(r["q"]), r["eps"]
        p, n, step, *_ = run_to_end(circ, eps, q)
        fps = analytic_fixed_points(q, circ)
        label, dist, resid, rho = match_endpoint(p, q, circ, fps)
        # also confirm the populations are a valid probability vector
        rows.append({
            "circuit": circ, "q": r["q"], "eps_used": eps,
            "attractor": label,
            "p_PhiP": repr(float(p[0])), "p_PhiM": repr(float(p[1])),
            "p_PsiP": repr(float(p[2])), "p_PsiM": repr(float(p[3])),
            "sum_minus_1": float(p.sum() - 1.0),
            "min_population": float(p.min()),
            "residual_full_map": resid,
            "dist_to_analytic_fixed_point": dist,
            "spectral_radius": rho,
            "rounds": n, "final_step": step,
        })
    return rows


def write(name, rows):
    if not rows:
        return
    keys = list(rows[0])
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    scan = basin_scan()
    write("basin_scan.csv", scan)

    bnd = basin_boundaries(scan)
    write("basin_boundaries.csv", bnd)

    prec = precision_check()
    write("precision_check.csv", prec)

    ver = endpoint_verification(scan)
    write("endpoint_verification.csv", ver)

    # short console digest
    print("\n--- distinct attractors per (circuit, q) ---")
    agg = {}
    for r in scan:
        agg.setdefault((r["circuit"], r["q"]), {}).setdefault(r["attractor"], []).append(r["eps"])
    for (circ, q), d in sorted(agg.items()):
        if len(d) > 1:
            parts = [f"{lab}: {len(e)} eps [{min(e)}..{max(e)}]" for lab, e in d.items()]
            print(f"  {circ:6s} q={q:9s}  MULTI  " + " | ".join(parts))
    print("\n--- basin boundaries ---")
    for r in bnd:
        print(f"  {r['circuit']:6s} q={r['q']:9s}  eps* in "
              f"[{float(r['eps_below']):.15f}, {float(r['eps_above']):.15f}]  "
              f"{r['attractor_below']} -> {r['attractor_above']}   ({r['method']})")
    print("\n--- precision check (max |float64 - dps60|) ---")
    for r in prec:
        print(f"  {r['circuit']:6s} eps={r['eps']:<5} q={r['q']:<8} "
              f"dF={r['abs_diff_fidelity']:>12}  dcoord={r['max_coord_diff']}")

    meta = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, check=True).stdout.strip(),
        "precision_dps": DPS,
        "attractor_match_rule": (
            f"labelled only when dist_to_fixed_point < {MATCH_DIST} AND "
            f"residual of the FULL 4-component one-round map < {MATCH_RESID} AND "
            "the analytic spectral radius there is < 1; otherwise 'unresolved'"),
        "float64_orbit_max_rounds": 200000,
        "float64_orbit_stop_tol": 1e-16,
        "boundary_method": "bisection on eps at fixed q, 50 halvings, max_rounds=60000; transitions whose upper side is the exact maximally mixed fixed point are not bisected",
    }
    (OUT / "basins_precision_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
