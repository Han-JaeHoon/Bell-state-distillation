"""Independent stabilizer / Heisenberg verification of the measured observables.

Route 1: dense matrix conjugation  V^dag Z_q V.
Route 2: rule-based Pauli propagation (no dense matrices).
Both must give  V^dag Z_3 V = XXXX  and  V^dag Z_4 V = ZZZZ.
"""

import numpy as np

from pqec_distill.bell_states import BELL_LABELS, bell_state
from pqec_distill.circuit import CNOT_SEQUENCE, N_QUBITS, Q3, Q4, full_unitary
from pqec_distill.gates import PAULI_X, PAULI_Z, kron_list, single_qubit_gate
from pqec_distill.pauli_propagation import (
    PauliString, conjugate_by_circuit, heisenberg_measured_observables,
)

TOL = 1e-12
XXXX = kron_list([PAULI_X] * 4)
ZZZZ = kron_list([PAULI_Z] * 4)


def _conjugate_dense(pauli_letter, qubit):
    v = full_unitary()
    op = single_qubit_gate(N_QUBITS, qubit, PAULI_Z if pauli_letter == "Z" else PAULI_X)
    return v.conj().T @ op @ v


def test_dense_conjugation_z3_is_xxxx():
    assert np.allclose(_conjugate_dense("Z", Q3), XXXX, atol=TOL)


def test_dense_conjugation_z4_is_zzzz():
    assert np.allclose(_conjugate_dense("Z", Q4), ZZZZ, atol=TOL)


def test_rule_based_propagation_matches():
    """Independent route: symbolic Pauli propagation, no matrices involved."""
    obs = heisenberg_measured_observables(CNOT_SEQUENCE, Q3)
    assert str(obs[Q3]) == "+XXXX"
    assert str(obs[Q4]) == "+ZZZZ"


def test_two_routes_agree():
    """Dense conjugation and rule-based propagation must give the same operator."""
    letter_to_matrix = {"I": np.eye(2, dtype=complex), "X": PAULI_X, "Z": PAULI_Z}
    for qubit in (Q3, Q4):
        symbolic = conjugate_by_circuit(
            PauliString.single(N_QUBITS, qubit, "Z"), CNOT_SEQUENCE, Q3
        )
        dense_from_symbolic = ((-1.0) ** symbolic.sign) * kron_list(
            [letter_to_matrix[c] for c in symbolic.letters()]
        )
        assert np.allclose(dense_from_symbolic, _conjugate_dense("Z", qubit), atol=TOL)


def test_observables_commute():
    assert np.allclose(XXXX @ ZZZZ, ZZZZ @ XXXX, atol=TOL)


def test_observables_are_hermitian_and_square_to_identity():
    for op in (XXXX, ZZZZ):
        assert np.allclose(op, op.conj().T, atol=TOL)
        assert np.allclose(op @ op, np.eye(16), atol=TOL)


def test_bell_products_are_stabilizer_eigenstates():
    """|B_ab>|B_cd> has XXXX eigenvalue (-1)^(b xor d), ZZZZ eigenvalue (-1)^(a xor c)."""
    for (a, b) in BELL_LABELS:
        for (c, d) in BELL_LABELS:
            psi = np.kron(bell_state(a, b), bell_state(c, d))
            assert np.allclose(XXXX @ psi, ((-1.0) ** (b ^ d)) * psi, atol=TOL)
            assert np.allclose(ZZZZ @ psi, ((-1.0) ** (a ^ c)) * psi, atol=TOL)
