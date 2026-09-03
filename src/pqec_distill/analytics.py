"""Analytic targets, written INDEPENDENTLY of the circuit simulator.

Anti-confirmation-bias rule: nothing in this module imports
:mod:`pqec_distill.circuit`, :mod:`pqec_distill.measurement`,
:mod:`pqec_distill.gates` or :mod:`pqec_distill.bell_states`.  The Bell states
are re-declared here from their explicit closed forms so that a shared bug in
the simulator's Bell construction cannot propagate into the analytic target.
A dedicated test asserts that the two independent constructions agree.

Bell ordering is the same convention (index = 2a + b):
    0 -> Phi+ = (|00>+|11>)/sqrt2
    1 -> Phi- = (|00>-|11>)/sqrt2
    2 -> Psi+ = (|01>+|10>)/sqrt2
    3 -> Psi- = (|01>-|10>)/sqrt2
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "BELL_VECTORS_LITERAL", "bell_projectors_literal", "bell_basis_literal",
    "schur_square_bell", "matrix_square_normalized",
    "predicted_measurement_bits", "predicted_branch_populations",
    "predicted_branch_probability", "p_success_bell_diagonal",
    "F_of_eps", "q_of_eps", "isotropic_populations",
    "p_success_isotropic", "F_out_isotropic", "eps_prime",
    "rho_out_isotropic_computational", "eps_prime_series_leading",
    "EPS_INPUT_SEPARABLE", "EPS_OUTPUT_SEPARABLE",
    "input_concurrence_isotropic", "output_concurrence_isotropic",
    "repeated_populations", "repeated_eps", "p_total_repeated",
    "per_level_success_probabilities", "repeated_populations_closed_form",
]

_S = 1.0 / np.sqrt(2.0)

#: Bell vectors written out literally, independent of bell_states.py.
BELL_VECTORS_LITERAL = np.array(
    [
        [_S, 0.0, 0.0, _S],    # Phi+
        [_S, 0.0, 0.0, -_S],   # Phi-
        [0.0, _S, _S, 0.0],    # Psi+
        [0.0, _S, -_S, 0.0],   # Psi-
    ],
    dtype=complex,
)


def bell_basis_literal() -> np.ndarray:
    """4x4 matrix whose columns are the literal Bell vectors."""
    return BELL_VECTORS_LITERAL.T.copy()


def bell_projectors_literal() -> list[np.ndarray]:
    out = []
    for v in BELL_VECTORS_LITERAL:
        col = v.reshape(4, 1)
        out.append(col @ col.conj().T)
    return out


# --------------------------------------------------------------------------
# General Bell-basis predictions
# --------------------------------------------------------------------------

def schur_square_bell(rho: np.ndarray) -> np.ndarray:
    """Elementwise (Schur) square of rho in the Bell basis, back in the
    computational basis.  This is the predicted UNNORMALIZED 00-branch output
    for an arbitrary two-qubit input.
    """
    u = bell_basis_literal()
    rho_bell = u.conj().T @ rho @ u
    return u @ (rho_bell * rho_bell) @ u.conj().T


def matrix_square_normalized(rho: np.ndarray) -> np.ndarray:
    """rho^2 / Tr(rho^2) -- the ideal virtual-distillation target."""
    sq = rho @ rho
    return sq / np.trace(sq)


def predicted_measurement_bits(a: int, b: int, c: int, d: int) -> tuple[int, int]:
    """Predicted (m3, m4) for input |B_ab>|B_cd>:  m3 = b xor d, m4 = a xor c."""
    return (b ^ d, a ^ c)


def predicted_branch_populations(p, mu: int, nu: int) -> np.ndarray:
    """Predicted UNNORMALIZED Bell populations of branch (m3,m4) = (mu,nu).

    rho_tilde_{mu,nu} = sum_{a,b} p_ab * p_{a xor nu, b xor mu} B_ab
    """
    p = np.asarray(p, dtype=float)
    out = np.zeros(4, dtype=float)
    for a in (0, 1):
        for b in (0, 1):
            i = 2 * a + b
            j = 2 * (a ^ nu) + (b ^ mu)
            out[i] = p[i] * p[j]
    return out


def predicted_branch_probability(p, mu: int, nu: int) -> float:
    return float(np.sum(predicted_branch_populations(p, mu, nu)))


def p_success_bell_diagonal(p) -> float:
    """P_success = sum_i p_i^2 = Tr(rho^2) for Bell-diagonal rho."""
    p = np.asarray(p, dtype=float)
    return float(np.sum(p ** 2))


# --------------------------------------------------------------------------
# Bell-isotropic family
# --------------------------------------------------------------------------

def F_of_eps(eps):
    """Input Bell fidelity F = 1 - 3 eps / 4."""
    return 1.0 - 3.0 * np.asarray(eps, dtype=float) / 4.0


def q_of_eps(eps):
    """Off-Bell population q = eps / 4 (each of the three other Bell states)."""
    return np.asarray(eps, dtype=float) / 4.0


def isotropic_populations(eps) -> np.ndarray:
    f = float(F_of_eps(eps))
    q = float(q_of_eps(eps))
    return np.array([f, q, q, q], dtype=float)


def p_success_isotropic(eps):
    """P_success = F^2 + 3 q^2 = (4 - 6 eps + 3 eps^2) / 4."""
    eps = np.asarray(eps, dtype=float)
    return (4.0 - 6.0 * eps + 3.0 * eps ** 2) / 4.0


def F_out_isotropic(eps):
    """F_out = F^2 / (F^2 + 3 q^2)."""
    f = F_of_eps(eps)
    q = q_of_eps(eps)
    return f ** 2 / (f ** 2 + 3.0 * q ** 2)


def eps_prime(eps):
    """eps' = eps^2 / (4 - 6 eps + 3 eps^2)."""
    eps = np.asarray(eps, dtype=float)
    return eps ** 2 / (4.0 - 6.0 * eps + 3.0 * eps ** 2)


def eps_prime_series_leading(eps):
    """Leading small-eps behaviour eps^2 / 4."""
    eps = np.asarray(eps, dtype=float)
    return eps ** 2 / 4.0


def rho_out_isotropic_computational(eps) -> np.ndarray:
    """Expected postselected output in the computational basis, written out
    explicitly as a matrix in terms of eps' (independent of the simulator).
    """
    ep = float(eps_prime(eps))
    return np.array(
        [
            [0.5 - ep / 4.0, 0.0, 0.0, (1.0 - ep) / 2.0],
            [0.0, ep / 4.0, 0.0, 0.0],
            [0.0, 0.0, ep / 4.0, 0.0],
            [(1.0 - ep) / 2.0, 0.0, 0.0, 0.5 - ep / 4.0],
        ],
        dtype=complex,
    )


EPS_INPUT_SEPARABLE = 2.0 / 3.0
#: Output concurrence vanishes above this eps: solves F_out(eps) = 1/2,
#: giving eps = 2 - 2*sqrt(3)/3 = 4/(3+sqrt(3)) ~= 0.8452994616207484.
EPS_OUTPUT_SEPARABLE = 2.0 - 2.0 * np.sqrt(3.0) / 3.0


def output_concurrence_isotropic(eps):
    """C_out = max(0, 2 F_out - 1) for the Bell-diagonal output."""
    return np.maximum(0.0, 2.0 * F_out_isotropic(eps) - 1.0)


def input_concurrence_isotropic(eps):
    """C_in = max(0, 2 F_in - 1) = max(0, 1 - 3 eps / 2)."""
    return np.maximum(0.0, 2.0 * F_of_eps(eps) - 1.0)


# --------------------------------------------------------------------------
# Repeated ideal rounds
# --------------------------------------------------------------------------

def repeated_populations(p, levels: int) -> np.ndarray:
    """Bell populations after ``levels`` ideal rounds, by explicit recurrence
    p_i -> p_i^2 / sum_j p_j^2 applied ``levels`` times.
    """
    p = np.asarray(p, dtype=float).copy()
    for _ in range(levels):
        sq = p ** 2
        p = sq / sq.sum()
    return p


def repeated_populations_closed_form(p, levels: int) -> np.ndarray:
    """Closed form p_i^(2^l) / sum_j p_j^(2^l)."""
    p = np.asarray(p, dtype=float)
    powered = p ** (2 ** levels)
    return powered / powered.sum()


def per_level_success_probabilities(p, levels: int) -> list[float]:
    """Success probability of ONE node at each level of the purification tree."""
    p = np.asarray(p, dtype=float).copy()
    out = []
    for _ in range(levels):
        sq = p ** 2
        s = float(sq.sum())
        out.append(s)
        p = sq / s
    return out


def p_total_repeated(p, levels: int) -> float:
    """Total postselection success probability of the FULL binary tree.

    A depth-``levels`` purification tree consumes 2**levels input copies.  Level
    k (1-indexed, counting from the leaves) contains 2**(levels-k) nodes, and
    every one of them must succeed, so

        P_total = prod_k  P_k ** (2 ** (levels - k))

    where P_k is the per-node success probability at level k.  This telescopes
    to Tr[rho^(2**levels)]; see ``tests/test_repeated_rounds.py``.

    NOTE: the naive product ``prod_k P_k`` (one node per level) is a DIFFERENT
    quantity and does *not* equal Tr[rho^(2**levels)].
    """
    per_level = per_level_success_probabilities(p, levels)
    total = 1.0
    for k, prob in enumerate(per_level, start=1):
        total *= prob ** (2 ** (levels - k))
    return float(total)


def repeated_eps(eps, levels: int) -> float:
    """Apply the isotropic eps -> eps' recursion ``levels`` times."""
    e = float(eps)
    for _ in range(levels):
        e = float(eps_prime(e))
    return e
