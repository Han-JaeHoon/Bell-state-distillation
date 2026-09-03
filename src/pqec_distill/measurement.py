"""Projective measurement of (q3, q4), postselection and partial trace.

Qubit ordering as in :mod:`pqec_distill.gates`: |q1 q2 q3 q4>, q1 most
significant, 0-based indices q1->0 .. q4->3.
"""

from __future__ import annotations

import numpy as np

from .circuit import N_QUBITS, Q3, Q4, apply_unitary_dm, full_unitary
from .gates import I2, PROJ0, PROJ1, kron_list

__all__ = [
    "measurement_projector", "partial_trace", "postselect_branch",
    "run_circuit_branches", "run_circuit_success",
]


def measurement_projector(m3: int, m4: int) -> np.ndarray:
    """I (x) I (x) |m3><m3| (x) |m4><m4| on the 4-qubit space."""
    if m3 not in (0, 1) or m4 not in (0, 1):
        raise ValueError("measurement bits must be 0 or 1")
    p3 = PROJ0 if m3 == 0 else PROJ1
    p4 = PROJ0 if m4 == 0 else PROJ1
    factors = [I2] * N_QUBITS
    factors[Q3] = p3
    factors[Q4] = p4
    return kron_list(factors)


def partial_trace(rho: np.ndarray, n_qubits: int, keep: list[int]) -> np.ndarray:
    """Partial trace of a density matrix, keeping the listed qubit indices.

    Implemented directly from the tensor definition (reshape + einsum trace)
    rather than via any library helper.
    """
    keep = sorted(keep)
    traced = [k for k in range(n_qubits) if k not in keep]
    tensor = rho.reshape([2] * (2 * n_qubits))
    # Trace out from the highest index down so earlier axes stay valid.
    for offset, q in enumerate(reversed(traced)):
        n_now = n_qubits - offset
        tensor = np.trace(tensor, axis1=q, axis2=q + n_now)
    d = 2 ** len(keep)
    return tensor.reshape(d, d)


def postselect_branch(rho4: np.ndarray, m3: int, m4: int) -> tuple[np.ndarray, float]:
    """Project the 4-qubit state onto (q3,q4)=(m3,m4) and trace them out.

    Returns ``(rho_tilde, probability)`` where ``rho_tilde`` is the
    UNNORMALIZED retained two-qubit operator on (q1,q2) and ``probability``
    is its trace.
    """
    proj = measurement_projector(m3, m4)
    projected = proj @ rho4 @ proj
    rho_tilde = partial_trace(projected, N_QUBITS, keep=[0, 1])
    prob = float(np.real(np.trace(rho_tilde)))
    return rho_tilde, prob


def run_circuit_branches(rho_a: np.ndarray, rho_b: np.ndarray) -> dict:
    """Run the full physical circuit on rho_a (x) rho_b and return all branches.

    Returns a dict mapping (m3, m4) -> {'rho_tilde', 'prob', 'rho_out'}.
    ``rho_out`` is the normalized conditional state (None if prob == 0).
    """
    rho_in = np.kron(rho_a, rho_b)
    rho_out4 = apply_unitary_dm(rho_in, full_unitary())
    branches = {}
    for m3 in (0, 1):
        for m4 in (0, 1):
            rho_tilde, prob = postselect_branch(rho_out4, m3, m4)
            normalized = rho_tilde / prob if prob > 1e-15 else None
            branches[(m3, m4)] = {
                "rho_tilde": rho_tilde,
                "prob": prob,
                "rho_out": normalized,
            }
    return branches


def run_circuit_success(rho: np.ndarray) -> tuple[np.ndarray, float]:
    """Two identical copies in, (normalized rho_out, P_success) for outcome 00."""
    branch = run_circuit_branches(rho, rho)[(0, 0)]
    return branch["rho_out"], branch["prob"]
