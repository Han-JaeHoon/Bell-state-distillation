"""Dense n-qubit gate construction, from first principles.

QUBIT ORDERING CONVENTION (used everywhere in this package)
-----------------------------------------------------------
The mathematical computational basis is written

    |q1 q2 q3 q4>

with ``q1`` as the LEFTMOST / MOST-SIGNIFICANT tensor factor.

Internally qubits are addressed by 0-based indices::

    q1 -> 0   (most significant)
    q2 -> 1
    q3 -> 2
    q4 -> 3   (least significant)

so that the basis index of |b0 b1 b2 b3> is  8*b0 + 4*b1 + 2*b2 + b3.

This is *not* Qiskit's little-endian convention.  Any Qiskit cross-check must
convert explicitly (see :mod:`pqec_distill.pauli_propagation` for the
independent verification route actually used here).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "I2", "PAULI_X", "PAULI_Y", "PAULI_Z", "HADAMARD",
    "kron_list", "single_qubit_gate", "cnot", "is_unitary",
    "PROJ0", "PROJ1",
]

I2 = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)

PROJ0 = np.array([[1, 0], [0, 0]], dtype=complex)
PROJ1 = np.array([[0, 0], [0, 1]], dtype=complex)


def kron_list(mats):
    """Kronecker product of a list of matrices, left factor = most significant."""
    out = np.array([[1.0 + 0.0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def single_qubit_gate(n_qubits: int, target: int, u: np.ndarray) -> np.ndarray:
    """Embed a one-qubit gate ``u`` acting on ``target`` into ``n_qubits``."""
    if not 0 <= target < n_qubits:
        raise ValueError(f"target {target} out of range for {n_qubits} qubits")
    factors = [u if k == target else I2 for k in range(n_qubits)]
    return kron_list(factors)


def cnot(n_qubits: int, control: int, target: int) -> np.ndarray:
    """CNOT with the given control and target, as a dense 2^n x 2^n matrix.

    Built as  P0_c (x) I  +  P1_c (x) X_t , i.e. straight from the definition
    rather than from any library's gate set.
    """
    if control == target:
        raise ValueError("control and target must differ")
    if not (0 <= control < n_qubits and 0 <= target < n_qubits):
        raise ValueError("qubit index out of range")
    term0 = kron_list([PROJ0 if k == control else I2 for k in range(n_qubits)])
    term1 = kron_list(
        [PROJ1 if k == control else (PAULI_X if k == target else I2) for k in range(n_qubits)]
    )
    return term0 + term1


def is_unitary(u: np.ndarray, atol: float = 1e-12) -> bool:
    n = u.shape[0]
    return bool(np.allclose(u.conj().T @ u, np.eye(n), atol=atol))
