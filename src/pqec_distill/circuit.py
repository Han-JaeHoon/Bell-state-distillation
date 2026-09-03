"""The proposed 4-qubit / 5-CNOT Bell-label comparator circuit.

QUBIT ORDERING (see :mod:`pqec_distill.gates`)
----------------------------------------------
    |q1 q2 q3 q4>,  q1 most significant.
    0-based indices:  q1->0, q2->1, q3->2, q4->3.

    retained pair A     = (q1, q2)
    sacrificial pair B  = (q3, q4)   -- measured and discarded

CIRCUIT (exactly as specified; do not reorder or flip directions)
------------------------------------------------------------------
    1. CNOT q3 -> q4
    2. CNOT q2 -> q4
    3. CNOT q1 -> q4
    4. CNOT q3 -> q2
    5. CNOT q3 -> q1
    then H on q3, then measure q3 and q4 in the computational basis.

``CNOT_SEQUENCE`` stores these as (control, target) 0-based index pairs in
circuit time order.  It is the single source of truth for the gate list.
"""

from __future__ import annotations

import numpy as np

from .gates import HADAMARD, cnot, single_qubit_gate

__all__ = [
    "N_QUBITS", "Q1", "Q2", "Q3", "Q4", "CNOT_SEQUENCE", "CNOT_SEQUENCE_LABELS",
    "five_cnot_unitary", "full_unitary", "apply_unitary_dm",
]

N_QUBITS = 4
Q1, Q2, Q3, Q4 = 0, 1, 2, 3

#: (control, target) in circuit time order, 0-based indices.
CNOT_SEQUENCE: list[tuple[int, int]] = [
    (Q3, Q4),
    (Q2, Q4),
    (Q1, Q4),
    (Q3, Q2),
    (Q3, Q1),
]

#: Human-readable form of the same list, for reports.
CNOT_SEQUENCE_LABELS = [("q3", "q4"), ("q2", "q4"), ("q1", "q4"), ("q3", "q2"), ("q3", "q1")]


def five_cnot_unitary() -> np.ndarray:
    """U_CNOT: the five CNOTs only (no final H), as a 16x16 unitary.

    Gates are applied in circuit time order, so the matrix product is
    C5 @ C4 @ C3 @ C2 @ C1.
    """
    u = np.eye(2 ** N_QUBITS, dtype=complex)
    for control, target in CNOT_SEQUENCE:
        u = cnot(N_QUBITS, control, target) @ u
    return u


def full_unitary() -> np.ndarray:
    """V = H_3 @ U_CNOT, the complete unitary before measurement."""
    h3 = single_qubit_gate(N_QUBITS, Q3, HADAMARD)
    return h3 @ five_cnot_unitary()


def apply_unitary_dm(rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    return u @ rho @ u.conj().T
