"""Reference formulas for the parent project's 5-qubit SWAP-test gadgets.

These are the exact Bell-sector recursions and fixed points derived in the
parent project's notes "Step 3: Analytic Repeated Dynamics" and "Step 4:
Analytic Repeated Dynamics" (Jaehun Han, Aug 2026) for per-CNOT replacement
depolarizing noise ``q`` with ideal single-qubit gates -- the same noise
convention as :mod:`pqec_distill.noise` (``"replace"``).  They are
re-implemented here only to draw the comparison in
``docs/comparison_swap_test.md``; ``tests/test_comparison.py`` checks them
against the frozen trajectory CSV in ``results/data/external/``.

Step 3/4 state family:  rho(u, v) = 1/4 [II + u (XX - YY) + v ZZ],
u = <XX> = -<YY>, v = <ZZ>,  F = (1 + 2u + v)/4   (NOTE: different plane and
different fidelity formula from the 4-qubit circuit's rho(u, v)).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "step3_map", "step4_map", "fidelity_34", "fixed_point_34",
    "Q_SN_STEP3", "Q_SN_STEP4", "Q_ENT_STEP3", "Q_ENT_STEP4", "STEP5_PLATEAU",
]

#: from the parent notes (saddle-node and entanglement limits of the Phi+ branch)
Q_SN_STEP3, Q_ENT_STEP3 = 0.130579267969, 0.129137470776
Q_SN_STEP4, Q_ENT_STEP4 = 0.189416655885, 0.184735805365

#: Step 5 has no fixed point on the Phi+ branch; these are the metastable
#: plateau values from the parent repository's results/iterated_pqec/fixed_points.csv
STEP5_PLATEAU = {0.001: 0.998750, 0.01: 0.987507, 0.05: 0.937304}


def step3_map(u, v, qb):
    d = 1 + qb ** 4 * (2 * u * u + v * v)
    return 2 * qb ** 6 * u * (1 + v) / d, qb ** 5 * (1 + qb) * (v + u * u) / d


def step4_map(u, v, qb):
    d = 1 + qb ** 2 * (2 * u * u + v * v)
    return 2 * qb ** 4 * u * (1 + v) / d, qb ** 3 * (1 + qb) * (v + u * u) / d


def fidelity_34(u, v):
    return (1 + 2 * u + v) / 4


def fixed_point_34(q, step: int):
    """Closed-form Phi+ branch fixed point (u*, v*); (nan, nan) beyond the saddle node."""
    qb = 1 - q
    a = qb ** 3 + qb ** 2 - qb + 1
    if step == 3:
        disc = qb ** 4 * a ** 2 + (5 * qb + 1) * (1 + qb) * (2 * qb ** 6 - 1)
        if disc < 0:
            return np.nan, np.nan
        v = (qb ** 2 * a + np.sqrt(disc)) / (qb ** 2 * (5 * qb + 1))
    elif step == 4:
        disc = qb ** 2 * a ** 2 + (5 * qb + 1) * (1 + qb) * (2 * qb ** 4 - 1)
        if disc < 0:
            return np.nan, np.nan
        v = (qb * a + np.sqrt(disc)) / (qb * (5 * qb + 1))
    else:
        raise ValueError("step must be 3 or 4")
    u = np.sqrt(max(v * (qb - 1 + 2 * qb * v) / (1 + qb), 0.0))
    return float(u), float(v)
