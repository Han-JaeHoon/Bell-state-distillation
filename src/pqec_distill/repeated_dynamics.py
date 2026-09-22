"""Repeated application of the VERIFIED one-round Bell-diagonal maps.

Starting point and only dependency inside this package:
``pqec_distill.analytic_exact_map`` -- the one-round exact maps that were
validated row-by-row against the frozen circuit-only reference.

Deliberately NOT imported (their stored repeated-dynamics conclusions are the
thing this module is meant to rediscover independently):
    pqec_distill.noisy_analytics       fixed points, saddle node, Jacobian
    pqec_distill.repeated_noisy        dense repeated map + Jacobian
    pqec_distill.swap_test_reference   Step 3/4 (u,v) recursions, q_SN/q_ent
    pqec_distill.analytics             ideal-case closed forms
    scripts/repeated_noisy_dynamics.py, scripts/analyze_repeated_rounds.py
    scripts/compare_with_swap_test.py, results/data/repeated_*.csv
    docs/derivation.md ss13-14, docs/comparison_swap_test.md,
    results/verification_report.md ssJ.6-J.7

Everything here is computed from the one-round map alone.
"""

from __future__ import annotations

import numpy as np

from .analytic_exact_map import (
    analytic_map_4q, analytic_map_type3, analytic_map_type4,
    analytic_success_4q, pauli_from_pops,
)

__all__ = [
    "CIRCUITS", "one_round", "success_probability", "isotropic_pops",
    "trajectory", "classify_trajectory",
]

CIRCUITS = ("Type3", "Type4", "4Q")
_MAPS = {"Type3": analytic_map_type3, "Type4": analytic_map_type4, "4Q": analytic_map_4q}


def isotropic_pops(eps: float) -> np.ndarray:
    """rho_0(eps) = (1-eps) Phi+ + eps I/4 in Bell populations."""
    e = float(eps)
    return np.array([1.0 - 3.0 * e / 4.0, e / 4.0, e / 4.0, e / 4.0])


def one_round(p, q: float, circuit: str) -> np.ndarray:
    return _MAPS[circuit](p, q)


def success_probability(p, q: float, circuit: str):
    """Per-round postselection probability; only defined for 4Q."""
    return analytic_success_4q(p, q) if circuit == "4Q" else None


def trajectory(circuit: str, eps: float, q: float, max_rounds: int = 2000,
               tol: float = 1e-14, record: bool = True) -> dict:
    """Iterate the one-round map from the Bell-isotropic initial state.

    Returns a dict with per-round Bell populations, Pauli coordinates,
    sup-norm step size, target fidelity, non-target/target ratios and (4Q)
    the per-round success probability.  Convergence is reported but NOT used
    on its own to decide what the trajectory converged to.
    """
    p = isotropic_pops(eps)
    rows = []
    converged_at = None
    for n in range(max_rounds + 1):
        x, y, z = pauli_from_pops(p)
        row = {
            "n": n,
            "p_PhiP": p[0], "p_PhiM": p[1], "p_PsiP": p[2], "p_PsiM": p[3],
            "x": x, "y": y, "z": z,
            "fidelity": p[0],
            "ratio_PhiM": p[1] / p[0] if p[0] > 0 else np.inf,
            "ratio_PsiP": p[2] / p[0] if p[0] > 0 else np.inf,
            "ratio_PsiM": p[3] / p[0] if p[0] > 0 else np.inf,
        }
        if circuit == "4Q":
            row["P_succ"] = analytic_success_4q(p, q)
        if n == max_rounds:
            row["step_inf"] = np.nan
            if record:
                rows.append(row)
            break
        nxt = one_round(p, q, circuit)
        step = float(np.max(np.abs(nxt - p)))
        row["step_inf"] = step
        if record:
            rows.append(row)
        p = nxt
        if step < tol and converged_at is None:
            converged_at = n + 1
            if record:
                x, y, z = pauli_from_pops(p)
                last = {"n": n + 1, "p_PhiP": p[0], "p_PhiM": p[1],
                        "p_PsiP": p[2], "p_PsiM": p[3], "x": x, "y": y, "z": z,
                        "fidelity": p[0], "step_inf": 0.0,
                        "ratio_PhiM": p[1] / p[0] if p[0] > 0 else np.inf,
                        "ratio_PsiP": p[2] / p[0] if p[0] > 0 else np.inf,
                        "ratio_PsiM": p[3] / p[0] if p[0] > 0 else np.inf}
                if circuit == "4Q":
                    last["P_succ"] = analytic_success_4q(p, q)
                rows.append(last)
            break
    return {"circuit": circuit, "eps": eps, "q": q, "rows": rows,
            "final": p, "converged_at": converged_at,
            "n_rounds_run": rows[-1]["n"] if rows else 0}


def classify_trajectory(circuit: str, eps: float, q: float,
                        max_rounds: int = 4000, tol: float = 1e-14) -> dict:
    """Run long, then characterise the endpoint WITHOUT assuming convergence.

    Reports the endpoint, the residual of the one-round map at the endpoint,
    whether the last stretch is monotone in fidelity, whether a period-2 (or
    short-cycle) pattern is present, and the largest fidelity overshoot seen
    along the way.
    """
    p = isotropic_pops(eps)
    hist = [p.copy()]
    for _ in range(max_rounds):
        p = one_round(p, q, circuit)
        hist.append(p.copy())
    hist = np.array(hist)
    fid = hist[:, 0]
    end = hist[-1]
    resid = float(np.max(np.abs(one_round(end, q, circuit) - end)))

    # short-cycle detection on the tail
    tail = hist[-64:]
    cycle = None
    for period in range(1, 9):
        if np.max(np.abs(tail[-1] - tail[-1 - period])) < 1e-13:
            cycle = period
            break

    tail_fid = fid[-200:]
    return {
        "circuit": circuit, "eps": eps, "q": q,
        "endpoint": end,
        "residual_at_endpoint": resid,
        "cycle_period": cycle,
        "fid_start": float(fid[0]), "fid_end": float(fid[-1]),
        "fid_max": float(fid.max()), "fid_argmax": int(fid.argmax()),
        "overshoot": float(fid.max() - fid[-1]),
        "monotone_increasing": bool(np.all(np.diff(fid) >= -1e-15)),
        "monotone_decreasing": bool(np.all(np.diff(fid) <= 1e-15)),
        "tail_spread": float(tail_fid.max() - tail_fid.min()),
        "final_step": float(np.max(np.abs(hist[-1] - hist[-2]))),
    }
