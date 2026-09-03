"""The 5-CNOT network and the full unitary V = H_3 U_CNOT."""

import numpy as np

from pqec_distill.circuit import (
    CNOT_SEQUENCE, CNOT_SEQUENCE_LABELS, N_QUBITS, Q3, five_cnot_unitary,
    full_unitary,
)
from pqec_distill.gates import HADAMARD, cnot, is_unitary, single_qubit_gate

TOL = 1e-13


def test_gate_list_is_exactly_as_specified():
    """Guard against silent reordering or direction flips of the circuit."""
    assert CNOT_SEQUENCE == [(2, 3), (1, 3), (0, 3), (2, 1), (2, 0)]
    assert CNOT_SEQUENCE_LABELS == [
        ("q3", "q4"), ("q2", "q4"), ("q1", "q4"), ("q3", "q2"), ("q3", "q1"),
    ]


def test_five_cnots_present():
    assert len(CNOT_SEQUENCE) == 5


def test_unitarity():
    assert is_unitary(five_cnot_unitary())
    assert is_unitary(full_unitary())


def test_cnot_network_is_a_permutation_matrix():
    """A CNOT-only network is a real permutation of computational basis states."""
    u = five_cnot_unitary()
    assert np.allclose(u.imag, 0.0, atol=TOL)
    assert np.allclose(np.sort(u.real, axis=0)[-1], np.ones(16), atol=TOL)
    assert np.allclose(u.real.sum(axis=0), np.ones(16), atol=TOL)
    assert np.allclose(u.real.sum(axis=1), np.ones(16), atol=TOL)


def test_full_unitary_composition_order():
    """V must be H_3 applied AFTER the CNOTs."""
    h3 = single_qubit_gate(N_QUBITS, Q3, HADAMARD)
    assert np.allclose(full_unitary(), h3 @ five_cnot_unitary(), atol=TOL)


def test_time_order_is_c5_c4_c3_c2_c1():
    u = np.eye(16, dtype=complex)
    for c, t in CNOT_SEQUENCE:
        u = cnot(4, c, t) @ u
    assert np.allclose(u, five_cnot_unitary(), atol=TOL)


def test_cnot_network_is_involution_free_check():
    """Sanity: the network is not accidentally the identity."""
    assert not np.allclose(five_cnot_unitary(), np.eye(16), atol=1e-8)
