"""Analytic exact Bell-diagonal input-output maps (Type 3 / Type 4 / 4Q).

This module is a STANDALONE transcription of externally supplied analytic
results.  It deliberately imports nothing from this package: no circuit, no
gate, no noise, no measurement code, and in particular not
``pqec_distill.noisy_analytics``.  Its only dependency is numpy.  It exists so
that the comparison against the frozen circuit-only reference
(``results/data/circuit_reference/circuit_reference.csv``) runs through a code
path that shares nothing with the simulation that produced that reference.

Source: "Bell-Diagonal Input-Output Maps of Noisy Purification Circuits --
Type 3, Type 4, and the 4-Qubit Protocol" (Jaehun Han, September 2026).

Parametrization
---------------
    p = (p_Phi+, p_Phi-, p_Psi+, p_Psi-),   sum p = 1
    x = p_Phi+ - p_Phi- + p_Psi+ - p_Psi-
    y = -p_Phi+ + p_Phi- + p_Psi+ - p_Psi-
    z = p_Phi+ + p_Phi- - p_Psi+ - p_Psi-
    p_Phi+ = (1 + x - y + z)/4      p_Phi- = (1 - x + y + z)/4
    p_Psi+ = (1 + x + y - z)/4      p_Psi- = (1 - x - y - z)/4
    qbar = 1 - q

Type 3 (k = 4) and Type 4 (k = 2) share one algebraic family:

    N_k = 1 + qbar^k (x^2 + y^2 + z^2)
    x'  = 2 qbar^(k+2) (x - y z) / N_k
    y'  = 2 qbar^(k+2) (y - x z) / N_k
    z'  = qbar^(k+1) (1 + qbar) (z - x y) / N_k

4Q has a different structure:

    N_4Q = 1 + qbar^5 (x^2 + y^2) + qbar^3 z^2
    x'   = qbar^3 [(1 + qbar) x - 2 qbar^2 y z] / N_4Q
    y'   = qbar^4 [(1 + qbar) y - 2 qbar   x z] / N_4Q
    z'   = qbar^4 [(1 + qbar) z - 2 qbar   x y] / N_4Q
    P_succ = N_4Q / 4

The same document also states Type 3 / Type 4 directly in Bell populations,
with Delta_PhiPsi = p_Phi+^2 + p_Phi-^2 - p_Psi+^2 - p_Psi-^2 and
N^(k) = 1 - qbar^k + 4 qbar^k sum_B p_B^2.  Both forms are transcribed
separately below (``*_population_form``) and are compared against each other,
so a transcription slip in either one shows up rather than propagating.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "pauli_from_pops", "pops_from_pauli",
    "analytic_map_type3", "analytic_map_type4", "analytic_map_4q",
    "analytic_success_4q", "analytic_normalization",
    "analytic_map_type34_population_form",
]

K_TYPE3 = 4
K_TYPE4 = 2


# --------------------------------------------------------------------------
# coordinate changes
# --------------------------------------------------------------------------

def pauli_from_pops(p) -> tuple[float, float, float]:
    """(p_Phi+, p_Phi-, p_Psi+, p_Psi-) -> (x, y, z)."""
    a, b, c, d = (float(v) for v in p)
    return (a - b + c - d, -a + b + c - d, a + b - c - d)


def pops_from_pauli(x: float, y: float, z: float) -> np.ndarray:
    """(x, y, z) -> (p_Phi+, p_Phi-, p_Psi+, p_Psi-)."""
    return np.array([(1 + x - y + z) / 4, (1 - x + y + z) / 4,
                     (1 + x + y - z) / 4, (1 - x - y - z) / 4], dtype=float)


# --------------------------------------------------------------------------
# Type 3 / Type 4 family, Pauli form
# --------------------------------------------------------------------------

def _map_type34_pauli(p, q: float, k: int) -> tuple[np.ndarray, float]:
    x, y, z = pauli_from_pops(p)
    qb = 1.0 - float(q)
    n = 1.0 + qb ** k * (x * x + y * y + z * z)
    xp = 2.0 * qb ** (k + 2) * (x - y * z) / n
    yp = 2.0 * qb ** (k + 2) * (y - x * z) / n
    zp = qb ** (k + 1) * (1.0 + qb) * (z - x * y) / n
    return pops_from_pauli(xp, yp, zp), n


def analytic_map_type3(p, q: float) -> np.ndarray:
    """Type 3 (16 CNOT), k = 4.  Returns output Bell populations."""
    return _map_type34_pauli(p, q, K_TYPE3)[0]


def analytic_map_type4(p, q: float) -> np.ndarray:
    """Type 4 (14 CNOT), k = 2.  Returns output Bell populations."""
    return _map_type34_pauli(p, q, K_TYPE4)[0]


# --------------------------------------------------------------------------
# Type 3 / Type 4 family, Bell-population form (independent transcription)
# --------------------------------------------------------------------------

def analytic_map_type34_population_form(p, q: float, k: int) -> np.ndarray:
    """Same maps written directly in Bell populations, as a second transcription.

        N^(k)   = 1 - qbar^k + 4 qbar^k sum_B p_B^2
        Delta   = p_Phi+^2 + p_Phi-^2 - p_Psi+^2 - p_Psi-^2
        p'_Phi+ = [N + 8 qbar^(k+2) (p_Phi+^2 - p_Phi-^2) + 2 qbar^(k+1)(1+qbar) Delta] / (4N)
        p'_Phi- = [N - 8 qbar^(k+2) (p_Phi+^2 - p_Phi-^2) + 2 qbar^(k+1)(1+qbar) Delta] / (4N)
        p'_Psi+ = [N + 8 qbar^(k+2) (p_Psi+^2 - p_Psi-^2) - 2 qbar^(k+1)(1+qbar) Delta] / (4N)
        p'_Psi- = [N - 8 qbar^(k+2) (p_Psi+^2 - p_Psi-^2) - 2 qbar^(k+1)(1+qbar) Delta] / (4N)
    """
    a, b, c, d = (float(v) for v in p)
    qb = 1.0 - float(q)
    n = 1.0 - qb ** k + 4.0 * qb ** k * (a * a + b * b + c * c + d * d)
    delta = a * a + b * b - c * c - d * d
    u = 8.0 * qb ** (k + 2) * (a * a - b * b)
    v = 8.0 * qb ** (k + 2) * (c * c - d * d)
    w = 2.0 * qb ** (k + 1) * (1.0 + qb) * delta
    return np.array([(n + u + w), (n - u + w), (n + v - w), (n - v - w)],
                    dtype=float) / (4.0 * n)


# --------------------------------------------------------------------------
# 4Q
# --------------------------------------------------------------------------

def _normalization_4q(x: float, y: float, z: float, qb: float) -> float:
    return 1.0 + qb ** 5 * (x * x + y * y) + qb ** 3 * z * z


def analytic_map_4q(p, q: float) -> np.ndarray:
    """4Q (5 CNOT, postselected).  Returns output Bell populations."""
    x, y, z = pauli_from_pops(p)
    qb = 1.0 - float(q)
    n = _normalization_4q(x, y, z, qb)
    xp = qb ** 3 * ((1.0 + qb) * x - 2.0 * qb ** 2 * y * z) / n
    yp = qb ** 4 * ((1.0 + qb) * y - 2.0 * qb * x * z) / n
    zp = qb ** 4 * ((1.0 + qb) * z - 2.0 * qb * x * y) / n
    return pops_from_pauli(xp, yp, zp)


def analytic_success_4q(p, q: float) -> float:
    """P_succ^(4Q) = N^(4Q) / 4."""
    x, y, z = pauli_from_pops(p)
    return _normalization_4q(x, y, z, 1.0 - float(q)) / 4.0


def analytic_normalization(p, q: float, circuit: str) -> float:
    """The normalization factor N of the stated map, for reference only.

    For 4Q the document identifies N/4 with the postselection probability.  For
    Type 3 / Type 4 the document makes no such identification, and this function
    makes none either -- it just returns N.
    """
    x, y, z = pauli_from_pops(p)
    qb = 1.0 - float(q)
    if circuit == "4Q":
        return _normalization_4q(x, y, z, qb)
    k = K_TYPE3 if circuit == "Type3" else K_TYPE4
    return 1.0 + qb ** k * (x * x + y * y + z * z)
