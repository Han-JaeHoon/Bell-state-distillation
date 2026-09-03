"""Exact analytic one-round map of the NOISY circuit on Bell-diagonal states,
and everything that follows from it: the operational CNOT-noise threshold,
the repeated-round fixed points, and their Bell-sector stability.

Derivation (docs/derivation.md, section 11).  Every CNOT is followed by the
two-qubit REPLACEMENT depolarizing channel with no-replacement weight
qbar = 1 - p (the Pauli convention is the same family with
qbar = 1 - 16 p_pauli / 15, see noise.py).  Propagating the four success
observables  P_A (x) Pi_00  backwards through H_3 and the five noisy CNOTs
gives, for a Bell-diagonal input  rho = 1/4 (II + x XX + y YY + z ZZ):

    D(x,y,z)   = 1 + qbar^5 (x^2 + y^2) + qbar^3 z^2
    P_success  = D / 4
    x' = qbar^3 [ (1+qbar) x - 2 qbar^2 y z ] / D
    y' = qbar^4 [ (1+qbar) y - 2 qbar   x z ] / D
    z' = qbar^4 [ (1+qbar) z - 2 qbar   x y ] / D

At qbar = 1 this is exactly  rho -> rho^2 / Tr(rho^2).

The plane  y = -z  is invariant, so a Bell-isotropic input
(x, y, z) = (eb, -eb, eb) stays on the two-parameter family

    rho(u, v) = 1/4 [ II + u XX + v (ZZ - YY) ],   u = <XX>,  v = <ZZ> = -<YY>

with Bell fidelity  F = <Phi+|rho|Phi+> = (1 + u + 2v) / 4  and reduced map

    D(u, v) = 1 + qbar^5 u^2 + (qbar^5 + qbar^3) v^2
    u' = qbar^3 [ (1+qbar) u + 2 qbar^2 v^2 ] / D
    v' = qbar^4 v [ (1+qbar) + 2 qbar u ] / D

This module is deliberately independent of the dense simulator (it imports no
simulator module); tests/test_noisy_closed_form.py compares the two.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

__all__ = [
    "qbar_replace", "qbar_pauli",
    "noisy_map_xyz", "p_success_xyz", "noisy_map_uv", "fidelity_uv",
    "isotropic_uv", "one_round_fidelity", "one_round_slope_pure_bell",
    "threshold_p", "fixed_point_uv", "fixed_point_branch", "jacobian_uv",
    "iterate_uv", "saddle_node_p", "entanglement_limit_p",
    "v0_fixed_point_u", "v0_family_limit_p", "v0_transverse_eigenvalue",
    "bistability_onset_p", "asymptotic_threshold_p",
]


# --------------------------------------------------------------------------
# noise-parameter conventions
# --------------------------------------------------------------------------

def qbar_replace(p: float) -> float:
    """No-replacement weight for the replacement convention."""
    return 1.0 - p


def qbar_pauli(p: float) -> float:
    """Same channel family: D_Pauli,p = D_replace,16p/15."""
    return 1.0 - 16.0 * p / 15.0


# --------------------------------------------------------------------------
# full Bell-diagonal map (x, y, z)
# --------------------------------------------------------------------------

def p_success_xyz(x, y, z, qb):
    return (1.0 + qb ** 5 * (x * x + y * y) + qb ** 3 * z * z) / 4.0


def noisy_map_xyz(x, y, z, qb):
    """One noisy round on 1/4(II + xXX + yYY + zZZ); returns (x', y', z')."""
    d = 1.0 + qb ** 5 * (x * x + y * y) + qb ** 3 * z * z
    xn = qb ** 3 * ((1.0 + qb) * x - 2.0 * qb ** 2 * y * z) / d
    yn = qb ** 4 * ((1.0 + qb) * y - 2.0 * qb * x * z) / d
    zn = qb ** 4 * ((1.0 + qb) * z - 2.0 * qb * x * y) / d
    return xn, yn, zn


# --------------------------------------------------------------------------
# reduced map on the invariant plane y = -z
# --------------------------------------------------------------------------

def isotropic_uv(eps):
    """Bell-isotropic input as (u, v) = (eb, eb), eb = 1 - eps."""
    eb = 1.0 - float(eps)
    return eb, eb


def fidelity_uv(u, v):
    return (1.0 + u + 2.0 * v) / 4.0


def noisy_map_uv(u, v, qb):
    d = 1.0 + qb ** 5 * u * u + (qb ** 5 + qb ** 3) * v * v
    un = qb ** 3 * ((1.0 + qb) * u + 2.0 * qb ** 2 * v * v) / d
    vn = qb ** 4 * v * ((1.0 + qb) + 2.0 * qb * u) / d
    return un, vn


def iterate_uv(u, v, qb, n):
    out = [(u, v)]
    for _ in range(n):
        u, v = noisy_map_uv(u, v, qb)
        out.append((u, v))
    return out


# --------------------------------------------------------------------------
# one-round threshold
# --------------------------------------------------------------------------

def one_round_fidelity(eps, qb):
    u, v = isotropic_uv(eps)
    return fidelity_uv(*noisy_map_uv(u, v, qb))


def one_round_slope_pure_bell(qb_eps: float = 1e-7):
    """Numerical dF_1(1, q)/dq at q = 0 (closed form: exactly 1, see tests)."""
    f0 = one_round_fidelity(0.0, 1.0)
    f1 = one_round_fidelity(0.0, 1.0 - qb_eps)
    return (f0 - f1) / qb_eps


def threshold_p(eps, convention: str = "replace"):
    """Largest per-CNOT noise p at which one round still improves the Bell
    fidelity,  F_out(eps, p*) = F_in(eps).  NaN if eps = 0 (nothing to gain)."""
    eps = float(eps)
    if eps <= 0.0:
        return float("nan")
    f_in = (1.0 + 3.0 * (1.0 - eps)) / 4.0

    def gap(p):
        qb = qbar_replace(p)
        return one_round_fidelity(eps, qb) - f_in

    # gap(0) > 0 for 0 < eps < 1; gap -> negative as p grows
    p_star = brentq(gap, 0.0, 0.999, xtol=1e-14, rtol=1e-14)
    if convention == "pauli":
        return 15.0 * p_star / 16.0
    return p_star


# --------------------------------------------------------------------------
# fixed points of the reduced map
# --------------------------------------------------------------------------

def _fixed_point_polynomial_coeffs(qb):
    """For v* != 0 the v-equation gives  D* = qbar^4 [(1+qbar) + 2 qbar u*].
    Eliminating v*^2 from the u-equation leaves a quadratic in u*; returns its
    coefficients (a, b, c) with a u^2 + b u + c = 0, plus the map u -> v^2."""
    q1 = 1.0 + qb
    # from D* = 1 + qb^5 u^2 + (qb^5+qb^3) v^2 = qb^4 (q1 + 2 qb u):
    #   v^2 = [qb^4 (q1 + 2 qb u) - 1 - qb^5 u^2] / (qb^5 + qb^3)
    # u-equation: u D* = qb^3 [q1 u + 2 qb^2 v^2]
    #   u qb^4 (q1 + 2 qb u) = qb^3 q1 u + 2 qb^5 v^2
    k = qb ** 5 + qb ** 3
    # substitute v^2:
    #   qb^4 q1 u + 2 qb^5 u^2 - qb^3 q1 u = 2 qb^5 [qb^4 (q1 + 2 qb u) - 1 - qb^5 u^2] / k
    # multiply by k:
    a = 2.0 * qb ** 5 * k + 2.0 * qb ** 10
    b = (qb ** 4 * q1 - qb ** 3 * q1) * k - 4.0 * qb ** 10
    c = -2.0 * qb ** 5 * (qb ** 4 * q1 - 1.0)
    return a, b, c


def fixed_point_uv(qb):
    """All real fixed points with v* > 0 of the reduced map, as a list of
    (u*, v*) sorted by descending fidelity.  Empty beyond the saddle node."""
    if qb >= 1.0:
        return [(1.0, 1.0)]
    a, b, c = _fixed_point_polynomial_coeffs(qb)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return []
    k = qb ** 5 + qb ** 3
    q1 = 1.0 + qb
    out = []
    for u in ((-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)):
        v2 = (qb ** 4 * (q1 + 2.0 * qb * u) - 1.0 - qb ** 5 * u * u) / k
        if v2 > 0.0 and -1.0 <= u <= 1.0:
            v = float(np.sqrt(v2))
            un, vn = noisy_map_uv(u, v, qb)
            if abs(un - u) < 1e-9 and abs(vn - v) < 1e-9:
                out.append((float(u), v))
    return sorted(out, key=lambda t: -fidelity_uv(*t))


def fixed_point_branch(qb):
    """The fixed point on the branch that tends to Phi+ as p -> 0, or None."""
    fps = fixed_point_uv(qb)
    return fps[0] if fps else None


def jacobian_uv(u, v, qb, h: float = 1e-7):
    """2x2 Jacobian of the reduced map at (u, v) by central differences."""
    j = np.zeros((2, 2))
    for col, (du, dv) in enumerate(((h, 0.0), (0.0, h))):
        up = noisy_map_uv(u + du, v + dv, qb)
        um = noisy_map_uv(u - du, v - dv, qb)
        j[:, col] = [(up[0] - um[0]) / (2 * h), (up[1] - um[1]) / (2 * h)]
    return j


def saddle_node_p():
    """Noise p at which the Phi+ branch ceases to exist (discriminant = 0)."""
    def disc(p):
        a, b, c = _fixed_point_polynomial_coeffs(1.0 - p)
        return b * b - 4.0 * a * c
    return brentq(disc, 1e-6, 0.9, xtol=1e-14)


def entanglement_limit_p():
    """Noise p at which the fixed-point Bell fidelity crosses 1/2."""
    def gap(p):
        fp = fixed_point_branch(1.0 - p)
        return (fidelity_uv(*fp) - 0.5) if fp else -1.0
    return brentq(gap, 1e-6, saddle_node_p() - 1e-12, xtol=1e-14)


# --------------------------------------------------------------------------
# the v = 0 family (classically correlated separable fixed points)
# --------------------------------------------------------------------------

def v0_fixed_point_u(qb):
    """u0 > 0 with  1/4 (II + u0 XX)  a fixed point:  u0^2 = (qbar^3(1+qbar) - 1)/qbar^5.
    Returns 0.0 when the nonzero solution does not exist (then I/4 is the point)."""
    a = (qb ** 3 * (1.0 + qb) - 1.0) / qb ** 5
    return float(np.sqrt(a)) if a > 0.0 else 0.0


def v0_family_limit_p():
    """p0: the nonzero v = 0 fixed point exists for p < p0, where qbar^3(1+qbar) = 1.
    For p > p0 the maximally mixed state I/4 is the (only) stable fixed point."""
    return brentq(lambda p: (1 - p) ** 3 * (2 - p) - 1.0, 0.05, 0.5, xtol=1e-15)


def v0_transverse_eigenvalue(qb):
    """d v'/d v at (u0, 0):  qbar + 2 qbar^2 u0 / (1 + qbar)  (using 1 + qbar^5 u0^2 = qbar^3(1+qbar))."""
    u0 = v0_fixed_point_u(qb)
    return qb + 2.0 * qb * qb * u0 / (1.0 + qb)


def bistability_onset_p():
    """p_B: the v = 0 fixed point becomes transversally stable (pitchfork in v).
    For p_B < p < p_SN both the Phi+ branch and the v = 0 point are attracting."""
    return brentq(lambda p: v0_transverse_eigenvalue(1 - p) - 1.0, 0.05, 0.18, xtol=1e-15)


def asymptotic_threshold_p(eps):
    """Largest p at which the FIXED POINT still beats the input:  F*(p) = F_in(eps).
    Slightly above the one-round threshold, because the anisotropic output keeps
    improving after a break-even first round."""
    f_in = (1.0 + 3.0 * (1.0 - float(eps))) / 4.0

    def gap(p):
        fp = fixed_point_branch(1.0 - p)
        return (fidelity_uv(*fp) - f_in) if fp else -1.0
    return brentq(gap, 1e-6, saddle_node_p() - 1e-9, xtol=1e-13)
