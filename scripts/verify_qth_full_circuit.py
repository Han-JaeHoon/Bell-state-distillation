#!/usr/bin/env python3
"""Full-circuit numerical study of the fidelity-improvement threshold q_th.

THE QUANTITY (not assumed, measured)
------------------------------------
With ``F_n = <Phi+| rho_n |Phi+>`` read off the effective two-qubit state of
round ``n``,

    q_th_full(eps) = sup { q : F_(n+1) > F_n  for EVERY n >= 0 }

which is deliberately NOT the one-round break-even threshold

    q_th_first(eps) = sup { q : F_1 > F_0 }.

Whether the two coincide is the central question and is an OUTPUT.  q_th is
also kept distinct from q_abs, q_rel, q_fixed, q_target, saddle-node or
F* = 1/2 crossing thresholds: only ``F_(n+1) > F_n`` enters.

THREE PROTOCOLS, all with the frozen repository circuits and the shared
per-CNOT two-qubit replacement depolarizing channel: 5Q Type 3 (16 CNOTs),
5Q Type 4 (14 CNOTs), 4Q postselected (5 CNOTs).

NO ANALYTIC SHORTCUTS IN THE PRIMARY PATH.  Nothing is imported from
``pqec_distill``; no analytic recurrence, no closed-form map, no stored
threshold of any kind, and no q_abs / q_rel value is used as a hint or a
bracket.  Those comparisons happen once, in clearly separated SECONDARY steps,
after the primary tables are complete.

THREE-VALUED PREDICATE.  Type 3 and Type 4 amplify the Bell-off-diagonal
direction, so a double-precision orbit eventually stops representing the exact
Bell-diagonal initial-value problem.  A trajectory that becomes
transverse-contaminated before it either converges or genuinely violates is
reported as AMBIGUOUS, never as FAIL, and is then resolved in ball arithmetic
by the optional ``python-flint`` route (``pip install python-flint``).

Usage::

    python scripts/verify_qth_full_circuit.py
    python scripts/verify_qth_full_circuit.py --quick
    python scripts/verify_qth_full_circuit.py --no-highprec
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _qth_core as C                                             # noqa: E402
import _qth_highprec as HP                                        # noqa: E402

OUT = ROOT / "results" / "qth"

# ===========================================================================
# tolerances -- recorded, never used to hide a case
# ===========================================================================
STEP_TOL = 1e-14        # full-state convergence
FID_TOL = 1e-14         # fidelity convergence (|Delta F| below this ...)
CONV_RUN = 10           # ... sustained for this many consecutive rounds
VIOL_TOL = 1e-12        # a round violates only if Delta F < -VIOL_TOL
N_MAX = 400             # maximum rounds, double precision
N_MAX_HP = 600          # maximum rounds, high precision
HP_CHECK_EVERY = 25     # sample the expensive ball diagnostics this often
OFFDIAG_TRUST = 1e-18   # Bell off-diagonal weight above which we stop trusting
DENOM_FLOOR = 1e-09     # Type 3/4 normalisation denominator safety floor
BISECT_TOL = 1e-10
Q_SCAN_HI = 0.9
N_COARSE = 90
HP_PREC_BITS = 400

EPS_TABLE = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60,
             2.0 / 3.0, 0.75, 0.90)
EPS_DEGENERATE = (0.0, 1.0)
NEAR_EPS = (0.05, 0.15, 0.40, 2.0 / 3.0)
PROTO_LABEL = {"type3": "5Q Type 3 (16 CNOT)", "type4": "5Q Type 4 (14 CNOT)",
               "4q": "4Q postselected (5 CNOT)"}

PASS, FAIL, AMBIG, UNSAFE, MAXR = "PASS", "FAIL", "AMBIGUOUS", "UNSAFE", "MAXROUNDS"


# ===========================================================================
# 1. the double-precision trajectory and predicate
# ===========================================================================

def run_trajectory(protocol: str, eps: float, q: float, record=False,
                   n_max=N_MAX, viol_tol=VIOL_TOL, step_tol=STEP_TOL,
                   fid_tol=FID_TOL, conv_run=CONV_RUN,
                   offdiag_trust=OFFDIAG_TRUST, program=None, **kw):
    """Iterate the real circuit; look for the first FIDELITY violation.

    The failure condition is ``Delta F_n = F_(n+1) - F_n < -viol_tol`` while
    the orbit is still trusted.  Convergence is declared either when the full
    state stops moving (``step < step_tol``) or when ``|Delta F|`` stays below
    ``fid_tol`` for ``conv_run`` consecutive rounds -- the second criterion is
    needed because for Type 3/4 the growing transverse mode keeps the full
    state moving long after the fidelity has settled.
    """
    rho = C.isotropic_rho(eps)
    f = C.fidelity(rho)
    p = C.bell_populations(rho)
    c_off = C.bell_offdiagonal_weight(rho)
    rows = [{"n": 0, "F": f, "dF": float("nan"), "p": p.copy(),
             "c_off": c_off, "step": float("nan"), "denominator": float("nan"),
             "p_succ": float("nan"), "min_eig": float(np.linalg.eigvalsh(rho).min()),
             "trace_error": 0.0, "herm_T": 0.0, "herm_rho": 0.0}] if record else []
    worst = {"c_off_max": c_off, "min_eig": float("inf"), "max_trace_err": 0.0,
             "max_herm_T": 0.0, "max_herm_rho": 0.0, "denom_min": float("inf"),
             "p_succ_min": float("inf"), "min_dF": float("inf"),
             "max_dF": -float("inf")}
    settled = 0
    n = 0
    step = float("nan")
    f_hist = [f]
    while n < n_max:
        try:
            rho2, info = C.one_round(rho, q, protocol, program=program, **kw)
        except FloatingPointError as exc:
            return dict(status=UNSAFE, reason=f"round failed: {exc}", round=None,
                        rounds_run=n, rows=rows, f_hist=f_hist, **worst)
        if not np.all(np.isfinite(rho2)):
            return dict(status=UNSAFE, reason="non-finite state", round=None,
                        rounds_run=n, rows=rows, f_hist=f_hist, **worst)
        f2 = C.fidelity(rho2)
        p2 = C.bell_populations(rho2)
        d_f = f2 - f
        step = float(np.max(np.abs(rho2 - rho)))
        c_off = C.bell_offdiagonal_weight(rho2)
        worst["c_off_max"] = max(worst["c_off_max"], c_off)
        worst["min_eig"] = min(worst["min_eig"], info["min_eig"])
        worst["max_trace_err"] = max(worst["max_trace_err"], info["trace_error"])
        worst["max_herm_T"] = max(worst["max_herm_T"], info["herm_residue_T"])
        worst["max_herm_rho"] = max(worst["max_herm_rho"], info["herm_residue_rho"])
        worst["denom_min"] = min(worst["denom_min"], abs(info["denominator"]))
        if np.isfinite(info["p_succ"]):
            worst["p_succ_min"] = min(worst["p_succ_min"], info["p_succ"])
        worst["min_dF"] = min(worst["min_dF"], d_f)
        worst["max_dF"] = max(worst["max_dF"], d_f)
        if record:
            rows.append({"n": n + 1, "F": f2, "dF": d_f, "p": p2.copy(),
                         "c_off": c_off, "step": step,
                         "denominator": info["denominator"],
                         "p_succ": info["p_succ"], "min_eig": info["min_eig"],
                         "trace_error": info["trace_error"],
                         "herm_T": info["herm_residue_T"],
                         "herm_rho": info["herm_residue_rho"]})
        f_hist.append(f2)
        if abs(info["denominator"]) < DENOM_FLOOR:
            return dict(status=UNSAFE, round=None, rounds_run=n + 1,
                        reason=f"normalisation denominator {info['denominator']:.3e} "
                               f"below floor {DENOM_FLOOR:g}",
                        rows=rows, f_hist=f_hist, **worst)
        if d_f < -viol_tol:
            return dict(status=FAIL, round=n, rounds_run=n + 1, reason="",
                        F_at_bind=f, F_next_at_bind=f2, dF_at_bind=d_f,
                        rows=rows, f_hist=f_hist, **worst)
        rho, f, p = rho2, f2, p2
        n += 1
        settled = settled + 1 if abs(d_f) <= fid_tol else 0
        if step < step_tol or settled >= conv_run:
            return dict(status=PASS, round=None, rounds_run=n, reason="",
                        converged_by=("step" if step < step_tol else "fidelity"),
                        rows=rows, f_hist=f_hist, **worst)
        if c_off > offdiag_trust:
            return dict(status=AMBIG, round=None, rounds_run=n,
                        reason=f"Bell off-diagonal weight {c_off:.3e} exceeded "
                               f"the trust threshold {offdiag_trust:g} before "
                               f"convergence or a violation",
                        rows=rows, f_hist=f_hist, **worst)
    return dict(status=MAXR, round=None, rounds_run=n,
                reason=f"no decision within {n_max} rounds",
                rows=rows, f_hist=f_hist, **worst)


def first_round_step(protocol: str, eps: float, q: float):
    rho0 = C.isotropic_rho(eps)
    rho1, info = C.one_round(rho0, q, protocol)
    return C.fidelity(rho0), C.fidelity(rho1), info


def first_round_violates(protocol: str, eps: float, q: float,
                         viol_tol=VIOL_TOL) -> bool:
    f0, f1, _ = first_round_step(protocol, eps, q)
    return (f1 - f0) < -viol_tol


# ===========================================================================
# 2. the high-precision trajectory (ball arithmetic)
# ===========================================================================

_HP_PROGRAM: dict = {}


def _hp_program(protocol: str):
    if protocol not in _HP_PROGRAM:
        _HP_PROGRAM[protocol] = HP.build_program_hp(protocol, HP_PREC_BITS)
    return _HP_PROGRAM[protocol]


def run_trajectory_hp(protocol: str, eps: Fraction, q: Fraction,
                      n_max=N_MAX_HP, viol_tol=VIOL_TOL, fid_tol=FID_TOL,
                      conv_run=CONV_RUN, record=False):
    """The same physical round in ball arithmetic at ``HP_PREC_BITS`` bits."""
    if not HP.AVAILABLE:
        raise RuntimeError("python-flint is not installed")
    prog = _hp_program(protocol)
    rho = HP.isotropic_rho_hp(eps.numerator, eps.denominator, HP_PREC_BITS)
    f_hp = HP.fidelity_hp(rho)
    f = float(f_hp)
    rows = [{"n": 0, "F": f, "dF": float("nan"), "c_off": 0.0}] if record else []
    settled, n = 0, 0
    worst = {"c_off_max": 0.0, "min_dF": float("inf"), "max_dF": -float("inf"),
             "denom_min": float("inf"), "max_ball_radius": 0.0}
    while n < n_max:
        rho2, denom = HP.one_round_hp(rho, q.numerator, q.denominator,
                                      protocol, prog, HP_PREC_BITS)
        f2_hp = HP.fidelity_hp(rho2)
        f2 = float(f2_hp)
        # Delta F is formed IN high precision: taking the difference of two
        # doubles would floor it at ~1e-16 and make the convergence test
        # meaningless once the orbit is close to its fixed point.
        d_f = float(f2_hp - f_hp)
        # the transverse weight and the ball radius are the expensive
        # diagnostics; in ball arithmetic they stay at the radius scale, so
        # they are sampled rather than evaluated every round
        c_off = float("nan")
        if record or n % HP_CHECK_EVERY == 0 or n + 1 == n_max:
            c_off = float(HP.bell_offdiagonal_weight_hp(rho2))
            worst["c_off_max"] = max(worst["c_off_max"], c_off)
        worst["min_dF"] = min(worst["min_dF"], d_f)
        worst["max_dF"] = max(worst["max_dF"], d_f)
        worst["denom_min"] = min(worst["denom_min"], abs(float(denom.real)))
        worst["max_ball_radius"] = max(worst["max_ball_radius"],
                                       HP.max_abs_radius(rho2))
        rho2 = HP.tighten(rho2)
        if record:
            rows.append({"n": n + 1, "F": f2, "dF": d_f, "c_off": c_off})
        if d_f < -viol_tol:
            worst["c_off_max"] = max(
                worst["c_off_max"], float(HP.bell_offdiagonal_weight_hp(rho2)))
            worst["max_ball_radius"] = max(worst["max_ball_radius"],
                                           HP.max_abs_radius(rho2))
            return dict(status=FAIL, round=n, rounds_run=n + 1,
                        F_at_bind=f, F_next_at_bind=f2, dF_at_bind=d_f,
                        rows=rows, reason="", **worst)
        rho, f, f_hp = rho2, f2, f2_hp
        n += 1
        settled = settled + 1 if abs(d_f) <= fid_tol else 0
        if settled >= conv_run:
            worst["c_off_max"] = max(
                worst["c_off_max"], float(HP.bell_offdiagonal_weight_hp(rho)))
            worst["max_ball_radius"] = max(worst["max_ball_radius"],
                                           HP.max_abs_radius(rho))
            return dict(status=PASS, round=None, rounds_run=n, rows=rows,
                        reason="", converged_by="fidelity", **worst)
    return dict(status=MAXR, round=None, rounds_run=n, rows=rows,
                reason=f"no decision within {n_max} high-precision rounds",
                **worst)


def _frac(x: float, limit=10 ** 15) -> Fraction:
    return Fraction(x).limit_denominator(limit)


def hybrid_status(protocol: str, eps: float, q: float, use_hp=True,
                  viol_tol=VIOL_TOL) -> dict:
    """Double precision first; ball arithmetic only where it is undecided."""
    res = run_trajectory(protocol, eps, q, viol_tol=viol_tol)
    res["route"] = "float64"
    if res["status"] in (PASS, FAIL) or not use_hp or not HP.AVAILABLE:
        return res
    hp = run_trajectory_hp(protocol, _frac(eps), _frac(q), viol_tol=viol_tol)
    hp["route"] = f"flint-{HP_PREC_BITS}bit"
    hp["float_status"] = res["status"]
    hp["float_reason"] = res.get("reason", "")
    return hp


# ===========================================================================
# 3. bracket and bisect on the three-valued predicate
# ===========================================================================

def coarse_scan(protocol: str, eps: float, use_hp=True, q_hi=Q_SCAN_HI,
                n_coarse=N_COARSE) -> dict:
    grid = np.linspace(0.0, q_hi, n_coarse + 1)
    labels = [hybrid_status(protocol, eps, float(q), use_hp)["status"]
              for q in grid]
    transitions = [k for k in range(len(labels) - 1)
                   if labels[k] != labels[k + 1]]
    pass_fail = [k for k in transitions
                 if labels[k] == PASS and labels[k + 1] == FAIL]
    return {"grid": [float(x) for x in grid], "labels": labels,
            "n_transitions": len(transitions),
            "n_pass_to_fail": len(pass_fail),
            "label_counts": {s: labels.count(s)
                             for s in (PASS, FAIL, AMBIG, UNSAFE, MAXR)},
            "grid_step": float(grid[1] - grid[0])}


def _last_pass_first_fail(grid, labels):
    last_pass = max((k for k, s in enumerate(labels) if s == PASS), default=None)
    first_fail = min((k for k, s in enumerate(labels) if s == FAIL), default=None)
    if last_pass is None or first_fail is None or first_fail <= last_pass:
        return None, None
    return float(grid[last_pass]), float(grid[first_fail])


def bisect_threshold(protocol: str, eps: float, lo: float, hi: float,
                     use_hp=True, tol=BISECT_TOL, max_iter=200) -> dict:
    """``lo`` must PASS and ``hi`` must FAIL; shrink the interval.

    An AMBIGUOUS midpoint neither passes nor fails, so it is recorded and the
    interval is NOT shrunk through it: the result is then an interval, not a
    number.
    """
    assert hybrid_status(protocol, eps, lo, use_hp)["status"] == PASS
    assert hybrid_status(protocol, eps, hi, use_hp)["status"] == FAIL
    it, ambiguous = 0, []
    while hi - lo > tol and it < max_iter:
        mid = 0.5 * (lo + hi)
        s = hybrid_status(protocol, eps, mid, use_hp)["status"]
        if s == PASS:
            lo = mid
        elif s == FAIL:
            hi = mid
        else:
            ambiguous.append({"q": mid, "status": s})
            break
        it += 1
    return {"q_lower_pass": lo, "q_upper_fail": hi, "bracket_width": hi - lo,
            "iterations": it, "ambiguous_midpoints": ambiguous,
            "decisive": not ambiguous and (hi - lo) <= tol}


def q_th_first(protocol: str, eps: float, tol=BISECT_TOL, q_hi=Q_SCAN_HI,
               n_coarse=N_COARSE) -> dict:
    """The ONE-ROUND threshold, computed independently of q_th_full."""
    pred = lambda q: first_round_violates(protocol, eps, q)           # noqa: E731
    grid = np.linspace(0.0, q_hi, n_coarse + 1)
    flags = [pred(float(q)) for q in grid]
    changes = [k for k in range(len(flags) - 1) if flags[k] != flags[k + 1]]
    if not changes:
        return {"q_threshold": None, "q_upper_fail": None, "has_root": False,
                "n_sign_changes": 0, "all_violating": bool(all(flags)),
                "none_violating": bool(not any(flags))}
    k = changes[0]
    lo, hi = float(grid[k]), float(grid[k + 1])
    it = 0
    while hi - lo > tol and it < 200:
        mid = 0.5 * (lo + hi)
        if pred(mid):
            hi = mid
        else:
            lo = mid
        it += 1
    return {"q_threshold": lo, "q_upper_fail": hi, "has_root": True,
            "n_sign_changes": len(changes), "bracket_width": hi - lo,
            "iterations": it}


# ===========================================================================
# 4. the main per-protocol table
# ===========================================================================

def _binding_record(protocol: str, eps: float, q_fail: float) -> dict:
    """Just above the upper failing bracket: which round actually binds?"""
    r = run_trajectory(protocol, eps, q_fail, record=True)
    out = {"binding_round": r.get("round"),
           "F_at_binding_round": r.get("F_at_bind"),
           "F_next_at_binding_round": r.get("F_next_at_bind"),
           "DeltaF_at_upper_fail": r.get("dF_at_bind"),
           "status_at_upper_fail": r["status"]}
    if r["status"] != FAIL and HP.AVAILABLE:
        rh = run_trajectory_hp(protocol, _frac(eps), _frac(q_fail), record=True)
        out.update({"binding_round": rh.get("round"),
                    "F_at_binding_round": rh.get("F_at_bind"),
                    "F_next_at_binding_round": rh.get("F_next_at_bind"),
                    "DeltaF_at_upper_fail": rh.get("dF_at_bind"),
                    "status_at_upper_fail": rh["status"],
                    "binding_round_route": f"flint-{HP_PREC_BITS}bit"})
    else:
        out["binding_round_route"] = "float64"
    return out


def protocol_table(protocol: str, eps_values, use_hp=True, verbose=True,
                   time_budget=240.0, n_coarse=N_COARSE) -> list:
    rows = []
    for eps in eps_values:
        t0 = time.time()
        # --- the coarse scan is run in double precision: it is cheap and its
        #     endpoints are re-verified with the hybrid predicate below ------
        grid = np.linspace(0.0, Q_SCAN_HI, n_coarse + 1)
        labels = [run_trajectory(protocol, eps, float(q))["status"] for q in grid]
        transitions = [k for k in range(len(labels) - 1)
                       if labels[k] != labels[k + 1]]
        lo, hi = _last_pass_first_fail([float(x) for x in grid], labels)
        bis = {"q_lower_pass": None, "q_upper_fail": None,
               "bracket_width": None, "iterations": 0,
               "ambiguous_midpoints": [], "decisive": False}
        if lo is not None:
            # re-verify the bracket with the hybrid predicate before bisecting
            s_lo = hybrid_status(protocol, eps, lo, use_hp)["status"]
            s_hi = hybrid_status(protocol, eps, hi, use_hp)["status"]
            if s_lo == PASS and s_hi == FAIL:
                t_start = time.time()
                it = 0
                amb = []

                def _left(t_start=t_start):
                    return time.time() - t_start < time_budget

                while hi - lo > BISECT_TOL and it < 200 and _left():
                    mid = 0.5 * (lo + hi)
                    r_mid = hybrid_status(protocol, eps, mid, use_hp)
                    s = r_mid["status"]
                    if s == PASS:
                        lo = mid
                    elif s == FAIL:
                        hi = mid
                    else:
                        # A non-decisive midpoint does not shrink the
                        # interval, but the two EDGES of the non-decisive
                        # band can still be located: bisect separately for
                        # the largest certified PASS below it and the
                        # smallest certified FAIL above it.  The result is
                        # an honest interval, not an invented number.
                        amb.append({"q": mid, "status": s,
                                    "rounds_run": r_mid["rounds_run"],
                                    "min_dF": r_mid["min_dF"]})
                        a, b = lo, mid
                        while b - a > BISECT_TOL and _left():
                            m2 = 0.5 * (a + b)
                            r2 = hybrid_status(protocol, eps, m2, use_hp)
                            it += 1
                            if r2["status"] == PASS:
                                a = m2
                            else:
                                b = m2
                                if r2["status"] != FAIL:
                                    amb.append({"q": m2,
                                                "status": r2["status"],
                                                "rounds_run": r2["rounds_run"],
                                                "min_dF": r2["min_dF"]})
                        lo = a
                        a, b = mid, hi
                        while b - a > BISECT_TOL and _left():
                            m2 = 0.5 * (a + b)
                            r2 = hybrid_status(protocol, eps, m2, use_hp)
                            it += 1
                            if r2["status"] == FAIL:
                                b = m2
                            else:
                                a = m2
                                if r2["status"] != PASS:
                                    amb.append({"q": m2,
                                                "status": r2["status"],
                                                "rounds_run": r2["rounds_run"],
                                                "min_dF": r2["min_dF"]})
                        hi = b
                        break
                    it += 1
                bis = {"q_lower_pass": lo, "q_upper_fail": hi,
                       "bracket_width": hi - lo, "iterations": it,
                       "ambiguous_midpoints": amb,
                       "decisive": not amb and (hi - lo) <= BISECT_TOL}
            else:
                bis["ambiguous_midpoints"] = [
                    {"q": lo, "status": s_lo}, {"q": hi, "status": s_hi}]
        first = q_th_first(protocol, eps)
        bind = _binding_record(protocol, eps, bis["q_upper_fail"]) \
            if bis["q_upper_fail"] is not None else {}
        below = run_trajectory(protocol, eps, bis["q_lower_pass"]) \
            if bis["q_lower_pass"] is not None else {}
        if below.get("status") not in (PASS, None) and HP.AVAILABLE and use_hp:
            below = run_trajectory_hp(protocol, _frac(eps),
                                      _frac(bis["q_lower_pass"]))
        above = run_trajectory(protocol, eps, bis["q_upper_fail"]) \
            if bis["q_upper_fail"] is not None else {}
        est = (None if bis["q_lower_pass"] is None
               else 0.5 * (bis["q_lower_pass"] + bis["q_upper_fail"]))
        ambiguity = ""
        trusted = bis["decisive"]
        if bis["ambiguous_midpoints"]:
            nd_min_df = min(a.get("min_dF", float("nan"))
                            for a in bis["ambiguous_midpoints"])
            nd_rounds = max(a.get("rounds_run", 0)
                            for a in bis["ambiguous_midpoints"])
            ambiguity = (
                f"{len(bis['ambiguous_midpoints'])} non-decisive q values "
                f"inside the bracket (no violation and no convergence within "
                f"{nd_rounds} high-precision rounds; smallest Delta F seen "
                f"there {nd_min_df:+.3e}); the reported interval is "
                f"[largest certified PASS, smallest certified FAIL]")
        elif bis["bracket_width"] is not None and bis["bracket_width"] > BISECT_TOL:
            ambiguity = (f"bisection stopped at width "
                         f"{bis['bracket_width']:.3e} (time budget "
                         f"{time_budget:g} s)")
        elif lo is None:
            ambiguity = "no PASS -> FAIL transition found in the scanned range"
        row = {
            "protocol": protocol, "epsilon": eps,
            "F_in": C.fidelity(C.isotropic_rho(eps)),
            "q_th_full_lower_pass": bis["q_lower_pass"],
            "q_th_full_upper_fail": bis["q_upper_fail"],
            "q_th_full_estimate": est,
            "q_th_first": first["q_threshold"],
            "q_th_full_minus_first": (
                None if est is None or first["q_threshold"] is None
                else est - first["q_threshold"]),
            "binding_round": bind.get("binding_round"),
            "F_at_binding_round": bind.get("F_at_binding_round"),
            "F_next_at_binding_round": bind.get("F_next_at_binding_round"),
            "DeltaF_at_upper_fail": bind.get("DeltaF_at_upper_fail"),
            "binding_round_route": bind.get("binding_round_route", ""),
            "rounds_to_convergence": below.get("rounds_run"),
            "min_DeltaF_below_threshold": below.get("min_dF"),
            "max_DeltaF_above_threshold": above.get("max_dF"),
            "predicate_sign_changes": len(transitions),
            "coarse_label_counts": " ".join(
                f"{s}:{labels.count(s)}" for s in (PASS, FAIL, AMBIG, UNSAFE, MAXR)
                if labels.count(s)),
            "threshold_bracket_width": bis["bracket_width"],
            "bisect_iterations": bis["iterations"],
            "n_nondecisive_q_in_bracket": len(bis["ambiguous_midpoints"]),
            "min_DeltaF_in_nondecisive_band": (
                min(a.get("min_dF", float("nan"))
                    for a in bis["ambiguous_midpoints"])
                if bis["ambiguous_midpoints"] else None),
            "max_rounds_in_nondecisive_band": (
                max(a.get("rounds_run", 0)
                    for a in bis["ambiguous_midpoints"])
                if bis["ambiguous_midpoints"] else None),
            "max_trace_error": max(below.get("max_trace_err", 0.0) or 0.0,
                                   above.get("max_trace_err", 0.0) or 0.0),
            "max_hermiticity_residual": max(
                below.get("max_herm_T", 0.0) or 0.0,
                below.get("max_herm_rho", 0.0) or 0.0,
                above.get("max_herm_T", 0.0) or 0.0,
                above.get("max_herm_rho", 0.0) or 0.0),
            "min_eigenvalue": min(
                v for v in (below.get("min_eig"), above.get("min_eig"))
                if v is not None and np.isfinite(v)) if (
                    below or above) else None,
            "max_Bell_offdiagonal_weight": max(
                below.get("c_off_max", 0.0) or 0.0,
                above.get("c_off_max", 0.0) or 0.0),
            "normalization_denominator_min": (
                None if protocol == "4q" else
                min(v for v in (below.get("denom_min"), above.get("denom_min"))
                    if v is not None and np.isfinite(v))),
            "success_probability_min": (
                None if protocol != "4q" else
                min(v for v in (below.get("p_succ_min"), above.get("p_succ_min"))
                    if v is not None and np.isfinite(v))),
            "below_route": below.get("route", "float64"),
            "trusted_flag": bool(trusted),
            "ambiguity_reason": ambiguity,
            "seconds": time.time() - t0,
        }
        rows.append(row)
        if verbose:
            print(f"      {protocol:6s} eps={eps:<9.6f} "
                  f"q_th in [{row['q_th_full_lower_pass']!r}, "
                  f"{row['q_th_full_upper_fail']!r}] "
                  f"first={first['q_threshold']!r} "
                  f"bind={row['binding_round']} "
                  f"trusted={trusted} ({row['seconds']:.0f} s)", flush=True)
    return rows


# ===========================================================================
# 5. non-monotone fidelity trajectories
# ===========================================================================

def non_monotone_search(protocol: str, eps_values, table, use_hp=True) -> dict:
    """``F_1 > F_0`` but ``F_(m+1) < F_m`` later -- and the opposite pattern."""
    late, recover = [], []
    for row in table:
        eps = row["epsilon"]
        if eps not in eps_values or row["q_th_full_upper_fail"] is None:
            continue
        q_lo = row["q_th_full_lower_pass"]
        q_first = row["q_th_first"]
        probes = [row["q_th_full_upper_fail"]]
        if q_first is not None:
            probes += [q_first - d for d in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
                       if q_first - d > (q_lo or 0.0)]
            probes += [q_first + 1e-6]
        for q in sorted(set(probes)):
            r = run_trajectory(protocol, eps, q, record=True)
            if r["status"] != FAIL:
                continue
            if r["round"] is None or r["round"] == 0:
                continue
            f = [x["F"] for x in r["rows"]]
            d_f = [x["dF"] for x in r["rows"][1:]]
            if f[1] <= f[0]:
                continue
            late.append({
                "protocol": protocol, "epsilon": eps, "q": q,
                "first_violating_round": r["round"],
                "F_series": " ".join(f"{x:.12f}" for x in f[:r["round"] + 2]),
                "F_0": f[0], "F_1": f[1],
                "F_m": r["F_at_bind"], "F_m_plus_1": r["F_next_at_bind"],
                "min_DeltaF": min(d_f), "DeltaF_at_violation": r["dF_at_bind"],
                "p_at_violation": " ".join(
                    f"{x:.12f}" for x in r["rows"][r["round"]]["p"]),
                "p_after_violation": " ".join(
                    f"{x:.12f}" for x in r["rows"][r["round"] + 1]["p"]),
                "Bell_offdiagonal_at_violation":
                    r["rows"][r["round"] + 1]["c_off"]})
        # the opposite pattern: an initial drop followed by recovery
        for q in sorted(set(probes)):
            r = run_trajectory(protocol, eps, q, record=True, n_max=120)
            f = [x["F"] for x in r["rows"]]
            if len(f) < 3 or f[1] >= f[0]:
                continue
            if max(f[2:]) > f[0] + VIOL_TOL:
                recover.append({
                    "protocol": protocol, "epsilon": eps, "q": q,
                    "F_0": f[0], "F_1": f[1], "F_max_later": max(f[2:]),
                    "round_of_max": int(np.argmax(f[2:]) + 2),
                    "status": r["status"]})
    return {"late_violations": late, "initial_drop_then_recovery": recover}


# ===========================================================================
# 6. degenerate endpoints
# ===========================================================================

def degenerate_endpoints(protocols=C.PROTOCOLS) -> list:
    out = []
    q_probe = (0.0, 1e-6, 1e-3, 0.01, 0.05, 0.2, 0.5)
    for protocol in protocols:
        for eps in EPS_DEGENERATE:
            rho0 = C.isotropic_rho(eps)
            probes = []
            for q in q_probe:
                try:
                    f0, f1, info = first_round_step(protocol, eps, q)
                    r = run_trajectory(protocol, eps, q)
                    probes.append({"q": q, "F0": f0, "F1": f1, "dF1": f1 - f0,
                                   "status": r["status"],
                                   "first_violating_round": r.get("round"),
                                   "rounds_run": r["rounds_run"],
                                   "denominator": info["denominator"],
                                   "p_succ": info["p_succ"]})
                except FloatingPointError as exc:
                    probes.append({"q": q, "F0": C.fidelity(rho0), "F1": None,
                                   "dF1": None, "status": UNSAFE,
                                   "first_violating_round": None,
                                   "rounds_run": 0, "denominator": None,
                                   "p_succ": None, "error": str(exc)})
            out.append({"protocol": protocol, "epsilon": eps,
                        "F_0": C.fidelity(rho0),
                        "strict_increase_possible": bool(eps > 0.0),
                        "probes": probes,
                        "note": ("F_0 = 1 and fidelity cannot exceed one, so "
                                 "F_1 > F_0 is impossible: degenerate"
                                 if eps == 0.0 else
                                 "maximally mixed input; the behaviour is "
                                 "measured, not assumed")})
    return out


# ===========================================================================
# 7. tolerance sensitivity, route comparison, negative controls
# ===========================================================================

def tolerance_sensitivity(protocols, eps_values) -> list:
    rows = []
    for protocol in protocols:
        for eps in eps_values:
            row = {"protocol": protocol, "epsilon": eps}
            for vt in (1e-10, 1e-12, 1e-14):
                r = q_th_first(protocol, eps)
                pred = lambda q: first_round_violates(protocol, eps, q, vt)  # noqa: E731
                lo, hi = 0.0, Q_SCAN_HI
                if pred(hi) and not pred(lo):
                    for _ in range(45):
                        mid = 0.5 * (lo + hi)
                        if pred(mid):
                            hi = mid
                        else:
                            lo = mid
                    row[f"q_th_first_viol_tol_{vt:g}"] = lo
                else:
                    row[f"q_th_first_viol_tol_{vt:g}"] = None
                row["q_th_first_reference"] = r["q_threshold"]
            vals = [row[k] for k in row if k.startswith("q_th_first_viol")
                    and row[k] is not None]
            row["spread"] = (max(vals) - min(vals)) if len(vals) > 1 else None
            rows.append(row)
    return rows


def dip_depth_scan(protocol: str, eps: float, lo: float, hi: float,
                   n_points: int = 15, use_hp=True) -> dict:
    """How deep is the worst fidelity dip, as a function of q?

    ``q_th_full`` for Type 3/4 is located where a LATE-round dip first becomes
    deeper than ``VIOL_TOL``, so unlike ``q_th_first`` it inherits a
    dependence on that tolerance.  This scan measures the dip depth directly
    (the trajectory is run with the violation test switched off) so the
    tolerance dependence can be read off instead of guessed.
    """
    width = max(hi - lo, 1e-06)
    grid = np.linspace(max(0.0, lo - 2 * width), hi + 2 * width, n_points)
    rows = []
    for q in grid:
        r = run_trajectory(protocol, eps, float(q), viol_tol=float("inf"))
        route = "float64"
        if r["status"] in (AMBIG, MAXR) and use_hp and HP.AVAILABLE:
            r = run_trajectory_hp(protocol, _frac(eps), _frac(float(q)),
                                  viol_tol=float("inf"))
            route = f"flint-{HP_PREC_BITS}bit"
        rows.append({"protocol": protocol, "epsilon": eps, "q": float(q),
                     "min_DeltaF": r["min_dF"], "status_without_viol_test":
                     r["status"], "rounds_run": r["rounds_run"],
                     "route": route})
    return {"rows": rows, "grid_spacing": float(grid[1] - grid[0])}


def qth_full_tolerance_sensitivity(protocol: str, eps: float,
                                   tols=(1e-10, 1e-12, 1e-14), use_hp=True,
                                   time_budget=300.0) -> dict:
    """Re-measure q_th_full at several violation tolerances, by BISECTION.

    For Type 3 and Type 4 the all-round threshold sits where a late-round dip
    first becomes deeper than ``VIOL_TOL``.  The dip deepens very slowly with
    q, so q_th_full inherits a tolerance dependence that q_th_first does not,
    and it can be far larger than the bisection width.  Measuring it by
    bisecting the predicate at each tolerance avoids the resolution limit of a
    grid scan.
    """
    out = {"protocol": protocol, "epsilon": eps}
    for tol in tols:
        key = f"q_th_full_viol_tol_{tol:g}"
        grid = np.linspace(0.0, Q_SCAN_HI, N_COARSE + 1)
        labels = [run_trajectory(protocol, eps, float(q),
                                 viol_tol=tol)["status"] for q in grid]
        lo, hi = _last_pass_first_fail([float(x) for x in grid], labels)
        if lo is None:
            out[key] = None
            continue
        if hybrid_status(protocol, eps, lo, use_hp, tol)["status"] != PASS or \
                hybrid_status(protocol, eps, hi, use_hp, tol)["status"] != FAIL:
            out[key] = None
            continue
        t0 = time.time()
        while hi - lo > BISECT_TOL and time.time() - t0 < time_budget:
            mid = 0.5 * (lo + hi)
            st = hybrid_status(protocol, eps, mid, use_hp, tol)["status"]
            if st == PASS:
                lo = mid
            elif st == FAIL:
                hi = mid
            else:
                break
        out[key] = lo
        out[f"{key}_upper"] = hi
    vals = [v for k, v in out.items()
            if k.startswith("q_th_full_viol_tol_") and not k.endswith("_upper")
            and v is not None]
    out["spread"] = (max(vals) - min(vals)) if len(vals) > 1 else None
    return out


def route_comparison(points) -> dict:
    rows, worst = [], {"rho": 0.0, "F": 0.0, "dF": 0.0, "populations": 0.0,
                       "denominator": 0.0}
    for protocol, eps, q in points:
        rho_a = rho_b = C.isotropic_rho(eps)
        f_a_prev, f_b_prev = C.fidelity(rho_a), C.fidelity(rho_b)
        for n in range(1, 6):
            rho_a, ia = C.one_round(rho_a, q, protocol)
            rho_b, ib = C.one_round_b(rho_b, q, protocol)
            f_a, f_b = C.fidelity(rho_a), C.fidelity(rho_b)
            d = {"rho": float(np.max(np.abs(rho_a - rho_b))),
                 "F": abs(f_a - f_b),
                 "dF": abs((f_a - f_a_prev) - (f_b - f_b_prev)),
                 "populations": float(np.max(np.abs(
                     C.bell_populations(rho_a) - C.bell_populations(rho_b)))),
                 "denominator": abs(ia["denominator"] - ib["denominator"])}
            for k in worst:
                worst[k] = max(worst[k], d[k])
            rows.append({"protocol": protocol, "epsilon": eps, "q": q,
                         "round": n, **{f"max_abs_{k}_diff": v
                                        for k, v in d.items()},
                         "F_routeA": f_a, "F_routeB": f_b})
            f_a_prev, f_b_prev = f_a, f_b
    return {"rows": rows, "worst": worst}


def route_comparison_hp(points) -> dict:
    """ROUTE A in double precision against ROUTE A in ball arithmetic."""
    if not HP.AVAILABLE:
        return {"available": False, "rows": [], "worst": {}}
    rows, worst = [], 0.0
    for protocol, eps, q in points:
        rho = C.isotropic_rho(eps)
        ef, qf = _frac(eps), _frac(q)
        prog = _hp_program(protocol)
        rhh = HP.isotropic_rho_hp(ef.numerator, ef.denominator, HP_PREC_BITS)
        for n in range(1, 8):
            rho, _ = C.one_round(rho, q, protocol)
            rhh, _ = HP.one_round_hp(rhh, qf.numerator, qf.denominator,
                                     protocol, prog, HP_PREC_BITS)
            rhh = HP.tighten(rhh)
            d = float(np.max(np.abs(rho - HP.to_numpy(rhh))))
            worst = max(worst, d)
            rows.append({"protocol": protocol, "epsilon": eps, "q": q,
                         "round": n, "max_abs_rho_diff": d,
                         "F_float": C.fidelity(rho),
                         "F_highprec": float(HP.fidelity_hp(rhh))})
    return {"available": True, "rows": rows, "worst": {"rho": worst}}


def _perturbation_scan(protocol: str, eps: float, q: float) -> dict:
    """Which single perturbations actually change the effective state?

    This is a measurement, not an assumption.  For Type 3 and Type 4 the
    effective state is ``T / Tr T``, so a perturbation that merely rescales
    ``T`` is annihilated by the normalisation, and the replacement channel on
    some wire pairs turns out to leave the parity-weighted read-out untouched.
    The scan finds, for each control family, a perturbation that does bite,
    and records the positions where the same perturbation does nothing.
    """
    base, _ = C.one_round(C.isotropic_rho(eps), q, protocol)

    def _diff(program=None, **kw):
        """Infinity marks a perturbation that destroys the read-out entirely
        (the parity weight collapses to zero), which is a detection too."""
        try:
            m, _ = C.one_round(C.isotropic_rho(eps), q, protocol,
                               program=program, **kw)
        except FloatingPointError:
            return float("inf")
        return float(np.max(np.abs(m - base)))

    n_c = C.n_cnot(protocol)
    noise = {k: _diff(C.build_program(protocol, q_offsets={k: 1e-03}))
             for k in range(1, n_c + 1)}
    removed = {k: _diff(C.build_program(protocol, skip_noise_after=(k,)))
               for k in range(1, n_c + 1)}
    ops = list(C.OPS[protocol])
    reversed_ = {}
    for i, (kind, arg) in enumerate(ops):
        if kind != "CNOT":
            continue
        alt = list(ops)
        alt[i] = ("CNOT", (arg[1], arg[0]))
        reversed_[i] = _diff(C.build_program(protocol, ops=alt))
    angles = {}
    for i, (kind, arg) in enumerate(ops):
        alt = list(ops)
        if kind == "U3":
            w, th, ph, la = arg
            alt[i] = ("U3", (w, th + 1e-03, ph, la))
        elif kind in ("T", "Tdg"):
            sgn = 1.0 if kind == "T" else -1.0
            alt[i] = ("U3", (arg, 0.0, 0.0, sgn * np.pi / 4 + 1e-03))
        elif kind == "H":
            alt[i] = ("U3", (arg, np.pi / 2 + 1e-03, 0.0, np.pi))
        else:
            continue
        angles[i] = _diff(C.build_program(protocol, ops=alt))

    def _pick(d):
        """Prefer a position with a large FINITE effect, so the recorded
        control still produces a usable modified state."""
        finite = {k: v for k, v in d.items() if 1e-09 < v < np.inf}
        if finite:
            chosen = max(finite, key=lambda x: finite[x])
        else:
            infinite = [k for k, v in d.items() if v == np.inf]
            chosen = infinite[0] if infinite else None
        return chosen, sorted(k for k, v in d.items() if v <= 1e-09)

    a_k, a_dead = _pick(noise)
    b_k, b_dead = _pick(removed)
    c_k, c_dead = _pick(reversed_)
    d_k, d_dead = _pick(angles)
    return {"q": q, "noise_offset": noise, "channel_removed": removed,
            "cnot_reversed": reversed_, "angle_perturbed": angles,
            "chosen": {"A": a_k, "B": b_k, "C": c_k, "D": d_k},
            "ineffective": {"A_noise_offset_cnots": a_dead,
                            "B_channel_removed_cnots": b_dead,
                            "C_cnot_reversed_op_indices": c_dead,
                            "D_angle_perturbed_op_indices": d_dead}}


def negative_controls(protocols=C.PROTOCOLS, eps=0.40) -> dict:
    """Controls A-F of the task.  Never used to estimate q_th.

    Each control is instantiated at a position the perturbation scan has
    measured to be effective; the positions where the same perturbation is
    annihilated are recorded alongside, because that annihilation is itself a
    property of the protocol worth reporting.
    """
    rows, scans = [], {}
    for protocol in protocols:
        base_q = q_th_first(protocol, eps)["q_threshold"]
        q = max(0.0, base_q * 0.5)
        scan = _perturbation_scan(protocol, eps, q)
        scans[protocol] = scan
        base_rho, base_info = C.one_round(C.isotropic_rho(eps), q, protocol)
        base_f = C.fidelity(base_rho)
        ops = list(C.OPS[protocol])

        def _rec(name, desc, rho_mod, info_mod, first_thr=None, position=None):
            rows.append({
                "protocol": protocol, "control": name, "description": desc,
                "position": position, "epsilon": eps, "q": q,
                "max_abs_rho_diff": float(np.max(np.abs(rho_mod - base_rho))),
                "abs_F_diff": abs(C.fidelity(rho_mod) - base_f),
                "abs_denominator_diff": abs(info_mod["denominator"]
                                            - base_info["denominator"]),
                "q_th_first_modified": first_thr,
                "q_th_first_shift": (None if first_thr is None or base_q is None
                                     else first_thr - base_q)})

        def _first_thr(round_fn):
            f0 = C.fidelity(C.isotropic_rho(eps))

            def pred(qq):
                return (C.fidelity(round_fn(qq)) - f0) < -VIOL_TOL

            lo, hi = 0.0, Q_SCAN_HI
            try:
                if pred(lo) or not pred(hi):
                    return None
                for _ in range(45):
                    mid = 0.5 * (lo + hi)
                    if pred(mid):
                        hi = mid
                    else:
                        lo = mid
            except FloatingPointError:
                return None
            return lo

        k = scan["chosen"]["A"]
        if k is not None:
            prog = C.build_program(protocol, q_offsets={k: 1e-03})
            rho_m, info_m = C.one_round(C.isotropic_rho(eps), q, protocol,
                                        program=prog)
            _rec("A_one_cnot_noise_plus_1e-3",
                 f"replacement-depolarizing strength after CNOT #{k} set to "
                 f"q + 1e-3", rho_m, info_m,
                 _first_thr(lambda qq, kk=k: C.one_round(
                     C.isotropic_rho(eps), qq, protocol,
                     program=C.build_program(protocol,
                                             q_offsets={kk: 1e-03}))[0]), k)
        k = scan["chosen"]["B"]
        if k is not None:
            prog = C.build_program(protocol, skip_noise_after=(k,))
            rho_m, info_m = C.one_round(C.isotropic_rho(eps), q, protocol,
                                        program=prog)
            _rec("B_one_channel_removed",
                 f"depolarizing channel after CNOT #{k} removed",
                 rho_m, info_m,
                 _first_thr(lambda qq, kk=k: C.one_round(
                     C.isotropic_rho(eps), qq, protocol,
                     program=C.build_program(protocol,
                                             skip_noise_after=(kk,)))[0]), k)
        i = scan["chosen"]["C"]
        if i is not None:
            alt_c = list(ops)
            alt_c[i] = ("CNOT", (ops[i][1][1], ops[i][1][0]))
            prog = C.build_program(protocol, ops=alt_c)
            rho_m, info_m = C.one_round(C.isotropic_rho(eps), q, protocol,
                                        program=prog)
            _rec("C_one_cnot_reversed",
                 f"operation {i} reversed: {ops[i][1]} -> {alt_c[i][1]}",
                 rho_m, info_m,
                 _first_thr(lambda qq, a=alt_c: C.one_round(
                     C.isotropic_rho(eps), qq, protocol,
                     program=C.build_program(protocol, ops=a))[0]), i)
        i = scan["chosen"]["D"]
        if i is not None:
            alt_d = list(ops)
            kind, arg = ops[i]
            if kind == "U3":
                w, th, ph, la = arg
                alt_d[i] = ("U3", (w, th + 1e-03, ph, la))
                desc = f"U3 theta of operation {i} (wire {w}) perturbed by +1e-3"
            elif kind in ("T", "Tdg"):
                sgn = 1.0 if kind == "T" else -1.0
                alt_d[i] = ("U3", (arg, 0.0, 0.0, sgn * np.pi / 4 + 1e-03))
                desc = (f"{kind} of operation {i} (wire {arg}) replaced by a "
                        f"phase {'+' if sgn > 0 else '-'}pi/4 + 1e-3")
            else:
                alt_d[i] = ("U3", (arg, np.pi / 2 + 1e-03, 0.0, np.pi))
                desc = (f"H of operation {i} (wire {arg}) replaced by "
                        f"U3(pi/2 + 1e-3, 0, pi)")
            prog = C.build_program(protocol, ops=alt_d)
            rho_m, info_m = C.one_round(C.isotropic_rho(eps), q, protocol,
                                        program=prog)
            _rec("D_one_qubit_angle_perturbed", desc, rho_m, info_m,
                 _first_thr(lambda qq, a=alt_d: C.one_round(
                     C.isotropic_rho(eps), qq, protocol,
                     program=C.build_program(protocol, ops=a))[0]), i)
        if protocol == "4q":
            rho_m, info_m = C.one_round(C.isotropic_rho(eps), q, protocol,
                                        outcome=(0, 1))
            _rec("E_wrong_postselection_branch",
                 "postselect (q3,q4) = (0,1) instead of (0,0)", rho_m, info_m,
                 _first_thr(lambda qq: C.one_round(
                     C.isotropic_rho(eps), qq, protocol, outcome=(0, 1))[0]),
                 "(0,1)")
        else:
            rho_m, info_m = C.one_round_b(C.isotropic_rho(eps), q, protocol,
                                          parity_signs=(+1.0, +1.0))
            _rec("F_wrong_ancilla_parity_sign",
                 "parity weights (+1, +1) instead of (+1, -1)", rho_m, info_m,
                 _first_thr(lambda qq: C.one_round_b(
                     C.isotropic_rho(eps), qq, protocol,
                     parity_signs=(+1.0, +1.0))[0]), "(+1,+1)")
    for r in rows:
        r["detected"] = bool(r["max_abs_rho_diff"] > 1e-09
                             or r["abs_F_diff"] > 1e-09
                             or (r["q_th_first_shift"] is not None
                                 and abs(r["q_th_first_shift"]) > 1e-09)
                             or r["q_th_first_modified"] is None)
    return {"rows": rows, "perturbation_scan": scans}


# ===========================================================================
# 8. SECONDARY comparisons (never inputs to the q_th search)
# ===========================================================================

_QABS_FILES = {"type3": ("results/type3_qabs/type3_qabs_circuit.csv", "q_abs_full"),
               "type4": ("results/type4_qabs/type4_qabs_circuit.csv", "q_abs_full"),
               "4q": ("results/4q_qabs/4q_qabs_circuit.csv", "q_abs_full")}
_QREL_FILES = {"type3": ("results/type3_qrel/type3_qrel_circuit.csv", "q_rel_full"),
               "type4": ("results/type4_qrel/type4_qrel_circuit.csv", "q_rel_full")}
#: the 4Q q_rel study lives on another feature branch; it is read READ-ONLY
#: with ``git show``, and that branch is never merged or modified.
_QREL_GIT = {"4q": ("feature/4q-qrel-circuit-study",
                    "results/4q_qrel/4q_qrel_circuit.csv", "q_rel_full")}


def _read_threshold_csv(text: str, column: str) -> dict:
    out = {}
    for rec in csv.DictReader(text.splitlines()):
        try:
            out[round(float(rec["epsilon"]), 10)] = float(rec[column])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def secondary_hierarchy(table) -> dict:
    """Compare q_abs <= q_rel <= q_th using the already frozen results."""
    q_abs, q_rel, sources = {}, {}, {}
    for proto, (rel_path, col) in _QABS_FILES.items():
        p = ROOT / rel_path
        if p.exists():
            q_abs[proto] = _read_threshold_csv(p.read_text(), col)
            sources[f"q_abs_{proto}"] = rel_path
    for proto, (rel_path, col) in _QREL_FILES.items():
        p = ROOT / rel_path
        if p.exists():
            q_rel[proto] = _read_threshold_csv(p.read_text(), col)
            sources[f"q_rel_{proto}"] = rel_path
    for proto, (branch, rel_path, col) in _QREL_GIT.items():
        try:
            txt = subprocess.run(["git", "-C", str(ROOT), "show",
                                  f"{branch}:{rel_path}"], capture_output=True,
                                 text=True, check=True).stdout
            q_rel[proto] = _read_threshold_csv(txt, col)
            sources[f"q_rel_{proto}"] = f"git show {branch}:{rel_path} (read-only)"
        except Exception as exc:                                  # pragma: no cover
            sources[f"q_rel_{proto}"] = f"unavailable: {exc}"
    rows, violations = [], []
    for r in table:
        proto, eps = r["protocol"], round(float(r["epsilon"]), 10)
        qa = q_abs.get(proto, {}).get(eps)
        qr = q_rel.get(proto, {}).get(eps)
        qt = r["q_th_full_estimate"]
        if qt is None:
            continue
        row = {"protocol": proto, "epsilon": r["epsilon"], "q_abs": qa,
               "q_rel": qr, "q_th": qt,
               "q_th_minus_q_rel": None if qr is None else qt - qr,
               "q_rel_minus_q_abs": None if (qa is None or qr is None)
               else qr - qa,
               "q_abs_le_q_rel": None if (qa is None or qr is None)
               else bool(qa <= qr + 1e-12),
               "q_rel_le_q_th": None if qr is None
               else bool(qr <= qt + 1e-12)}
        rows.append(row)
        if row["q_rel_le_q_th"] is False:
            violations.append(row)
        if row["q_abs_le_q_rel"] is False:
            violations.append(row)
    return {"rows": rows, "sources": sources, "violations": violations,
            "hierarchy_holds_everywhere": not violations,
            "n_compared": len(rows)}


def protocol_comparison(table) -> dict:
    by_eps: dict = {}
    for r in table:
        if r["q_th_full_estimate"] is None:
            continue
        by_eps.setdefault(round(float(r["epsilon"]), 10), {})[r["protocol"]] = \
            r["q_th_full_estimate"]
    rows, crossings = [], []
    prev = None
    for eps in sorted(by_eps):
        vals = by_eps[eps]
        if len(vals) < 3:
            continue
        order = tuple(sorted(vals, key=lambda p: -vals[p]))
        rows.append({"epsilon": eps, **{f"q_th_{p}": vals[p] for p in vals},
                     "ordering_high_to_low": " > ".join(order)})
        if prev is not None and prev[1] != order:
            crossings.append({"between_epsilon": prev[0], "and_epsilon": eps,
                              "ordering_before": " > ".join(prev[1]),
                              "ordering_after": " > ".join(order)})
        prev = (eps, order)
    maxima = {}
    for p in C.PROTOCOLS:
        pts = [(r["epsilon"], r["q_th_full_estimate"]) for r in table
               if r["protocol"] == p and r["q_th_full_estimate"] is not None]
        if pts:
            e, v = max(pts, key=lambda x: x[1])
            maxima[p] = {"epsilon_of_max": e, "q_th_max": v}
    return {"rows": rows, "crossings": crossings, "maxima": maxima}


# ===========================================================================
# 9. near-threshold trajectories
# ===========================================================================

def near_threshold_trajectories(table, eps_values=NEAR_EPS, delta=None,
                                n_rounds=60) -> dict:
    rows, blocks = [], []
    for r in table:
        if r["epsilon"] not in eps_values or r["q_th_full_estimate"] is None:
            continue
        est = r["q_th_full_estimate"]
        d = delta if delta is not None else max(5e-04, 50 * (
            r["threshold_bracket_width"] or 0.0))
        runs = []
        for tag, q in (("below", max(0.0, est - d)), ("near", est),
                       ("above", est + d)):
            tr = run_trajectory(r["protocol"], r["epsilon"], q, record=True,
                                n_max=n_rounds, viol_tol=float("inf"))
            runs.append((tag, q, tr["rows"]))
            for x in tr["rows"]:
                rows.append({
                    "protocol": r["protocol"], "epsilon": r["epsilon"],
                    "q": q, "tag": tag, "n": x["n"], "F": x["F"],
                    "dF": x["dF"],
                    **{f"p_{C.BELL_ORDER[k]}": float(x["p"][k]) for k in range(4)},
                    "bell_offdiagonal_weight": x["c_off"],
                    "denominator": x["denominator"], "p_succ": x["p_succ"],
                    "min_eig": x["min_eig"]})
        blocks.append({"protocol": r["protocol"], "epsilon": r["epsilon"],
                       "q_th": est, "delta": d, "runs": runs})
    return {"rows": rows, "blocks": blocks}


# ===========================================================================
# 10. figures
# ===========================================================================

PROTO_COLOR = {"type3": "#2a78d6", "type4": "#eb6834", "4q": "#1baf7a"}
PROTO_MARKER = {"type3": "o", "type4": "s", "4q": "^"}
BELL_COLOR = {"Phi+": "#2a78d6", "Phi-": "#eb6834",
              "Psi+": "#1baf7a", "Psi-": "#eda100"}
BELL_MARKER = {"Phi+": "o", "Phi-": "s", "Psi+": "^", "Psi-": "D"}
BELL_TEX = {"Phi+": r"$\Phi^{+}$", "Phi-": r"$\Phi^{-}$",
            "Psi+": r"$\Psi^{+}$", "Psi-": r"$\Psi^{-}$"}
INK, INK2, GRID = "#101010", "#4a4a4a", "#d9d9d9"


def use_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.bbox": "tight", "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11,
        "axes.titlesize": 11.5, "axes.labelsize": 11.5,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "axes.facecolor": "white", "axes.edgecolor": GRID,
        "axes.labelcolor": INK, "axes.linewidth": 0.9,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.8,
        "xtick.color": INK2, "ytick.color": INK2,
        "lines.linewidth": 1.8, "lines.markersize": 4.5,
        "legend.frameon": False})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def save(fig, stem):
    import matplotlib.pyplot as plt
    out = []
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        out.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return out


def plot_curves(table):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))
    for p in C.PROTOCOLS:
        rs = [r for r in table if r["protocol"] == p
              and r["q_th_full_estimate"] is not None]
        if not rs:
            continue
        e = [r["epsilon"] for r in rs]
        axes[0].plot(e, [r["q_th_full_estimate"] for r in rs],
                     color=PROTO_COLOR[p], marker=PROTO_MARKER[p],
                     markersize=4.0, markerfacecolor="white",
                     markeredgewidth=1.3, label=PROTO_LABEL[p])
        axes[1].plot(e, [(r["q_th_full_minus_first"] or 0.0) for r in rs],
                     color=PROTO_COLOR[p], marker=PROTO_MARKER[p],
                     markersize=4.0, markerfacecolor="white",
                     markeredgewidth=1.3, label=PROTO_LABEL[p])
    axes[0].set_ylabel(r"$q_{\mathrm{th}}$")
    axes[0].set_title("all-round fidelity-improvement threshold", color=INK)
    axes[1].axhline(0.0, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
    axes[1].set_ylabel(r"$q_{\mathrm{th}}^{\mathrm{full}}"
                       r" - q_{\mathrm{th}}^{\mathrm{first}}$")
    axes[1].set_title("all-round minus one-round (measured)", color=INK)
    axes[1].set_yscale("symlog", linthresh=1e-11)
    for ax in axes:
        ax.set_xlabel(r"input noise $\varepsilon$")
        ax.legend(loc="best")
        _despine(ax)
    fig.tight_layout()
    return save(fig, "qth_curves")


def plot_binding_round(table):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    for p in C.PROTOCOLS:
        rs = [r for r in table if r["protocol"] == p
              and r["binding_round"] is not None]
        if not rs:
            continue
        ax.plot([r["epsilon"] for r in rs], [r["binding_round"] for r in rs],
                color=PROTO_COLOR[p], marker=PROTO_MARKER[p], markersize=4.4,
                markerfacecolor="white", markeredgewidth=1.3,
                label=PROTO_LABEL[p])
    ax.set_xlabel(r"input noise $\varepsilon$")
    ax.set_ylabel("binding round just above threshold")
    ax.set_title("which round actually binds (measured, not assumed)", color=INK)
    ax.legend(loc="best")
    _despine(ax)
    fig.tight_layout()
    return save(fig, "qth_binding_round")


def plot_trajectories(blocks):
    import matplotlib.pyplot as plt
    protos = [p for p in C.PROTOCOLS if any(b["protocol"] == p for b in blocks)]
    eps_list = sorted({b["epsilon"] for b in blocks})
    out = []
    for quantity, stem, ylab in (("F", "qth_trajectories_F", r"$F_n$"),
                                 ("dF", "qth_trajectories_dF",
                                  r"$\Delta F_n$")):
        fig, axes = plt.subplots(len(protos), len(eps_list),
                                 figsize=(3.3 * len(eps_list),
                                          2.9 * len(protos)), squeeze=False)
        for i, p in enumerate(protos):
            for j, eps in enumerate(eps_list):
                ax = axes[i][j]
                blk = next((b for b in blocks if b["protocol"] == p
                            and b["epsilon"] == eps), None)
                if blk is None:
                    ax.axis("off")
                    continue
                for (tag, q, rws), style in zip(blk["runs"],
                                                ["-", (0, (5, 2)), (0, (1, 1))]):
                    n = [x["n"] for x in rws]
                    y = [x[quantity] for x in rws]
                    ax.plot(n, y, linestyle=style, color=PROTO_COLOR[p],
                            linewidth=1.6,
                            label=f"{tag}  q={q:.6f}")
                if quantity == "dF":
                    ax.axhline(0.0, color=INK2, linewidth=0.8,
                               linestyle=(0, (4, 3)))
                    ax.set_yscale("symlog", linthresh=1e-14)
                if i == 0:
                    ax.set_title(rf"$\varepsilon = {eps:.4f}$", color=INK)
                if j == 0:
                    ax.set_ylabel(f"{PROTO_LABEL[p].split(' (')[0]}\n{ylab}")
                if i == len(protos) - 1:
                    ax.set_xlabel("round $n$")
                ax.legend(loc="best", fontsize=7.5)
                _despine(ax)
        fig.tight_layout()
        out += save(fig, stem)
    return out


def plot_hierarchy(hier):
    import matplotlib.pyplot as plt
    protos = sorted({r["protocol"] for r in hier["rows"]})
    fig, axes = plt.subplots(1, len(protos), figsize=(4.0 * len(protos), 4.0),
                             squeeze=False)
    for j, p in enumerate(protos):
        ax = axes[0][j]
        rs = [r for r in hier["rows"] if r["protocol"] == p]
        e = [r["epsilon"] for r in rs]
        for key, col, mk, lab in (("q_abs", BELL_COLOR["Phi-"], "s",
                                   r"$q_{\mathrm{abs}}$"),
                                  ("q_rel", BELL_COLOR["Psi+"], "^",
                                   r"$q_{\mathrm{rel}}$"),
                                  ("q_th", BELL_COLOR["Phi+"], "o",
                                   r"$q_{\mathrm{th}}$")):
            ys = [r[key] for r in rs]
            if all(y is None for y in ys):
                continue
            ax.plot([x for x, y in zip(e, ys) if y is not None],
                    [y for y in ys if y is not None], color=col, marker=mk,
                    markersize=4.0, markerfacecolor="white",
                    markeredgewidth=1.3, label=lab)
        ax.set_title(PROTO_LABEL[p], color=INK)
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel("threshold")
        ax.legend(loc="best")
        _despine(ax)
    fig.suptitle(r"secondary check: $q_{\mathrm{abs}} \leq q_{\mathrm{rel}}"
                 r" \leq q_{\mathrm{th}}$", color=INK, y=1.02)
    fig.tight_layout()
    return save(fig, "qth_hierarchy")


# ===========================================================================
# 11. driver
# ===========================================================================

def write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    keys: list = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT)] + list(args),
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-highprec", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--protocols", default=",".join(C.PROTOCOLS))
    ap.add_argument("--time-budget", type=float, default=240.0)
    args = ap.parse_args()

    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    use_hp = HP.AVAILABLE and not args.no_highprec
    protocols = [p for p in args.protocols.split(",") if p in C.PROTOCOLS]
    eps_values = NEAR_EPS if args.quick else EPS_TABLE

    print("=" * 78)
    print("q_th : the ALL-ROUND fidelity-improvement threshold, from the "
          "full circuits")
    print("=" * 78)
    print("  q_th_full(eps) = sup { q : F_(n+1) > F_n for EVERY n >= 0 }")
    print("  q_th_first(eps) = sup { q : F_1 > F_0 }      (diagnostic only)")
    print("  failure condition: Delta F_n < -VIOL_TOL while the orbit is "
          "trusted")
    print(f"  high-precision route: "
          f"{'python-flint @ %d bits' % HP_PREC_BITS if use_hp else 'DISABLED'}")

    print("\n[1] frozen circuit provenance")
    prov = C.provenance_checks()
    for k, v in prov.items():
        print(f"      {k:52s} {v}")
    for p in protocols:
        print(f"\n  --- {PROTO_LABEL[p]} --- source: {C.CIRCUIT_SOURCE[p]}")
        print(C.format_sequence(p))

    print("\n[2] route A vs route B (independent implementations)")
    pts = [(p, e, q) for p in protocols for e in (0.05, 0.40, 2.0 / 3.0)
           for q in (0.0, 0.05, 0.12)]
    routes = route_comparison(pts)
    for k, v in routes["worst"].items():
        print(f"      max {k:32s} {v:.6e}")
    routes_hp = route_comparison_hp([(p, 0.40, 0.08) for p in protocols]) \
        if use_hp else {"available": False, "rows": [], "worst": {}}
    if routes_hp["available"]:
        print(f"      max float64 vs {HP_PREC_BITS}-bit ball arithmetic "
              f"(7 rounds)   {routes_hp['worst']['rho']:.6e}")
    snap = HP.angle_snap_report()
    print(f"      Type-4 angle snap to multiples of pi/4, max residual "
          f"{snap['max_abs_residual']:.3e} over {snap['n_angles']} angles")

    print(f"\n[3] tolerances: STEP_TOL={STEP_TOL:g} FID_TOL={FID_TOL:g} "
          f"CONV_RUN={CONV_RUN} VIOL_TOL={VIOL_TOL:g} N_MAX={N_MAX} "
          f"N_MAX_HP={N_MAX_HP}")
    print(f"      OFFDIAG_TRUST={OFFDIAG_TRUST:g} DENOM_FLOOR={DENOM_FLOOR:g} "
          f"BISECT_TOL={BISECT_TOL:g} q scan [0, {Q_SCAN_HI}] "
          f"with {N_COARSE + 1} points")

    print("\n[4] main q_th tables")
    table = []
    for p in protocols:
        print(f"    --- {PROTO_LABEL[p]}")
        table += protocol_table(p, eps_values, use_hp=use_hp,
                                time_budget=args.time_budget)

    print("\n[5] non-monotone fidelity trajectories")
    nonmono = {"late_violations": [], "initial_drop_then_recovery": []}
    for p in protocols:
        r = non_monotone_search(p, set(eps_values), table, use_hp)
        nonmono["late_violations"] += r["late_violations"]
        nonmono["initial_drop_then_recovery"] += r["initial_drop_then_recovery"]
    print(f"      F_1 > F_0 but a later round violates: "
          f"{len(nonmono['late_violations'])} cases")
    if nonmono["late_violations"]:
        ex = max(nonmono["late_violations"], key=lambda x: x["first_violating_round"])
        print(f"      deepest example: {ex['protocol']} eps={ex['epsilon']:.6f} "
              f"q={ex['q']:.10f} first violating round {ex['first_violating_round']}"
              f", Delta F = {ex['DeltaF_at_violation']:+.6e}")
    print(f"      initial drop then recovery: "
          f"{len(nonmono['initial_drop_then_recovery'])} cases")

    print("\n[6] near-threshold trajectories")
    traj = near_threshold_trajectories(table)
    print(f"      {len(traj['blocks'])} (protocol, epsilon) blocks, "
          f"{len(traj['rows'])} recorded rounds")

    print("\n[7] degenerate endpoints")
    degen = degenerate_endpoints(protocols)
    for d in degen:
        best = max((pr["dF1"] for pr in d["probes"]
                    if pr["dF1"] is not None), default=None)
        print(f"      {d['protocol']:6s} eps={d['epsilon']}: F_0={d['F_0']:.6f} "
              f"max Delta F_1 over probes = "
              f"{'n/a' if best is None else f'{best:+.3e}'}   {d['note']}")

    print("\n[8] violation-tolerance sensitivity")
    dips, full_sens = [], []
    for p in protocols:
        for eps in (0.40,):
            row = next((r for r in table if r["protocol"] == p
                        and r["epsilon"] == eps), None)
            if row is None or row["q_th_full_lower_pass"] is None:
                continue
            d = dip_depth_scan(p, eps, row["q_th_full_lower_pass"],
                               row["q_th_full_upper_fail"],
                               n_points=13, use_hp=use_hp)
            dips += d["rows"]
            t = qth_full_tolerance_sensitivity(p, eps, use_hp=use_hp)
            full_sens.append(t)
            print(f"      {p:6s} eps={eps:<6.3f} q_th_FULL at "
                  + "  ".join(
                      f"{k.split('viol_tol_')[1]}: "
                      + (f"{t[k]:.10f}" if t[k] is not None else "None")
                      for k in t if k.startswith("q_th_full_viol_tol_")
                      and not k.endswith("_upper"))
                  + f"   spread={t['spread']!r}")
    sens = tolerance_sensitivity(protocols, NEAR_EPS)
    for r in sens:
        print(f"      {r['protocol']:6s} eps={r['epsilon']:<9.6f} "
              + "  ".join(f"{k.split('viol_tol_')[1]}: "
                          f"{r[k]:.12f}" if r[k] is not None else f"{k}: None"
                          for k in r if k.startswith("q_th_first_viol"))
              + f"   spread={r['spread']!r}")

    print("\n[9] negative controls")
    nc = negative_controls(protocols)
    for r in nc["rows"]:
        print(f"      {r['protocol']:6s} {r['control']:30s} at {r['position']!r:<7} "
              f"d|rho|={r['max_abs_rho_diff']:.3e} dF={r['abs_F_diff']:.3e} "
              f"q_th_first shift={r['q_th_first_shift']!r:<26} "
              f"detected={r['detected']}")
    print("      perturbations the effective state ANNIHILATES "
          "(measured property, not a bug):")
    for proto, sc in nc["perturbation_scan"].items():
        for fam, dead in sc["ineffective"].items():
            if dead:
                print(f"          {proto:6s} {fam:34s} {dead}")

    print("\n[10] SECONDARY: q_abs <= q_rel <= q_th")
    hier = secondary_hierarchy(table)
    for k, v in hier["sources"].items():
        print(f"      {k:16s} {v}")
    print(f"      compared at {hier['n_compared']} (protocol, epsilon) points; "
          f"hierarchy holds everywhere: {hier['hierarchy_holds_everywhere']}")
    for r in hier["rows"]:
        print(f"          {r['protocol']:6s} eps={r['epsilon']:<9.6f} "
              f"q_abs={r['q_abs']!r:<22} q_rel={r['q_rel']!r:<22} "
              f"q_th={r['q_th']:.10f} "
              f"q_th-q_rel={r['q_th_minus_q_rel']!r}")
    if hier["violations"]:
        print("      *** APPARENT HIERARCHY VIOLATIONS ***")
        for v in hier["violations"]:
            print(f"          {v}")

    print("\n[11] SECONDARY: protocol comparison")
    comp = protocol_comparison(table)
    for r in comp["rows"]:
        print(f"      eps={r['epsilon']:<9.6f} {r['ordering_high_to_low']}")
    print(f"      crossings: {comp['crossings'] or 'none'}")
    for p, m in comp["maxima"].items():
        print(f"      max q_th {p:6s} = {m['q_th_max']:.10f} at eps="
              f"{m['epsilon_of_max']}")

    # ---- files --------------------------------------------------------
    write_csv(OUT / "qth_summary.csv", table)
    for p in protocols:
        write_csv(OUT / f"qth_{p}.csv", [r for r in table if r["protocol"] == p])
    write_csv(OUT / "qth_first_vs_full.csv",
              [{"protocol": r["protocol"], "epsilon": r["epsilon"],
                "q_th_full_estimate": r["q_th_full_estimate"],
                "q_th_full_lower_pass": r["q_th_full_lower_pass"],
                "q_th_full_upper_fail": r["q_th_full_upper_fail"],
                "q_th_first": r["q_th_first"],
                "q_th_full_minus_first": r["q_th_full_minus_first"],
                "binding_round": r["binding_round"],
                "trusted_flag": r["trusted_flag"]} for r in table])
    write_csv(OUT / "qth_trajectories.csv", traj["rows"])
    write_csv(OUT / "qth_nonmonotone_cases.csv", nonmono["late_violations"])
    write_csv(OUT / "qth_initial_drop_recovery.csv",
              nonmono["initial_drop_then_recovery"])
    write_csv(OUT / "qth_hierarchy_comparison.csv", hier["rows"])
    write_csv(OUT / "qth_protocol_crossings.csv",
              comp["rows"] + [{"epsilon": "crossing", **c}
                              for c in comp["crossings"]])
    write_csv(OUT / "qth_tolerance_sensitivity.csv", sens)
    write_csv(OUT / "qth_dip_depth_scan.csv", dips)
    write_csv(OUT / "qth_full_tolerance_sensitivity.csv", full_sens)
    write_csv(OUT / "qth_route_comparison.csv",
              routes["rows"] + routes_hp["rows"])
    write_csv(OUT / "qth_negative_controls.csv", nc["rows"])
    write_csv(OUT / "qth_degenerate_endpoints.csv",
              [{"protocol": d["protocol"], "epsilon": d["epsilon"],
                "F_0": d["F_0"], "note": d["note"], **pr}
               for d in degen for pr in d["probes"]])

    figures = []
    if not args.no_figures:
        print("\n[12] figures")
        try:
            import matplotlib
            matplotlib.use("Agg")
            use_style()
            figures += plot_curves(table)
            figures += plot_binding_round(table)
            figures += plot_trajectories(traj["blocks"])
            if hier["rows"]:
                figures += plot_hierarchy(hier)
            for f in figures:
                print(f"      {f}")
        except Exception as exc:                                  # pragma: no cover
            print(f"      figures skipped: {exc}")

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/verify_qth_full_circuit.py",
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_head": _git("rev-parse", "HEAD"),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "python_flint_available": HP.AVAILABLE,
        "high_precision_used": use_hp, "high_precision_bits": HP_PREC_BITS,
        "platform": platform.platform(), "quick_mode": bool(args.quick),
        "definition": ("q_th_full(eps) = sup { q : F_(n+1) > F_n for every "
                       "n >= 0 }, F_n = <Phi+|rho_n|Phi+>"),
        "diagnostic_definition": "q_th_first(eps) = sup { q : F_1 > F_0 }",
        "failure_condition": "Delta F_n < -VIOL_TOL while the orbit is trusted",
        "circuit_source": C.CIRCUIT_SOURCE,
        "noise_convention": C.NOISE_CONVENTION,
        "gate_sequences": {p: C.format_sequence(p) for p in protocols},
        "cnot_counts": {p: C.n_cnot(p) for p in C.PROTOCOLS},
        "not_used_in_primary_path": [
            "pqec_distill.analytic_exact_map", "analytic_map_4q",
            "any closed-form Bell-diagonal recurrence",
            "symbolic recurrence equations",
            "previously derived one-round q_th equations",
            "q_abs / q_rel values as root hints or brackets"],
        "tolerances": {"STEP_TOL": STEP_TOL, "FID_TOL": FID_TOL,
                       "CONV_RUN": CONV_RUN, "VIOL_TOL": VIOL_TOL,
                       "N_MAX": N_MAX, "N_MAX_HP": N_MAX_HP,
                       "OFFDIAG_TRUST": OFFDIAG_TRUST,
                       "DENOM_FLOOR": DENOM_FLOOR, "BISECT_TOL": BISECT_TOL,
                       "Q_SCAN_HI": Q_SCAN_HI, "N_COARSE": N_COARSE,
                       "time_budget_seconds": args.time_budget},
        "epsilon_table": list(eps_values),
        "epsilon_degenerate": list(EPS_DEGENERATE),
        "provenance": prov,
        "route_comparison": routes["worst"],
        "route_comparison_highprec": routes_hp.get("worst", {}),
        "type4_angle_snap": {k: v for k, v in snap.items() if k != "rows"},
        "table": table,
        "non_monotone": {
            "n_late_violations": len(nonmono["late_violations"]),
            "n_initial_drop_then_recovery":
                len(nonmono["initial_drop_then_recovery"]),
            "late_violations": nonmono["late_violations"][:50],
            "initial_drop_then_recovery":
                nonmono["initial_drop_then_recovery"][:50]},
        "degenerate_endpoints": degen,
        "tolerance_sensitivity": sens,
        "dip_depth_scan": dips,
        "qth_full_tolerance_sensitivity": full_sens,
        "negative_controls": nc["rows"],
        "perturbation_scan": {
            pp: {"q": v["q"], "chosen": v["chosen"],
                 "ineffective": v["ineffective"]}
            for pp, v in nc["perturbation_scan"].items()},
        "secondary_hierarchy": hier,
        "protocol_comparison": comp,
        "figures": figures,
        "max_rounds_actually_used": max(
            (r["rounds_to_convergence"] or 0) for r in table) if table else 0,
        "total_seconds": time.time() - t_start,
    }
    with (OUT / "qth_verification.json").open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    print(f"\n  written to {OUT.relative_to(ROOT)}")
    print(f"  total {time.time() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
