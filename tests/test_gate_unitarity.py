"""Gate construction: unitarity, CNOT action on basis states, H involution."""

import itertools

import numpy as np
import pytest

from pqec_distill.gates import (
    HADAMARD, I2, PAULI_X, PAULI_Y, PAULI_Z, cnot, is_unitary, kron_list,
    single_qubit_gate,
)

TOL = 1e-13


@pytest.mark.parametrize("u", [PAULI_X, PAULI_Y, PAULI_Z, HADAMARD, I2])
def test_single_qubit_unitary(u):
    assert is_unitary(u)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_cnot_unitary_all_pairs(n):
    for c, t in itertools.permutations(range(n), 2):
        assert is_unitary(cnot(n, c, t))


def test_cnot_is_involution():
    for c, t in itertools.permutations(range(4), 2):
        u = cnot(4, c, t)
        assert np.allclose(u @ u, np.eye(16), atol=TOL)


def test_cnot_basis_action_explicit():
    """CNOT(c,t) must map |...b_c...b_t...> to |...b_c...(b_t xor b_c)...>.

    Basis index of |b0 b1 b2 b3> is 8*b0+4*b1+2*b2+b3 (q1 most significant).
    """
    n = 4
    for c, t in itertools.permutations(range(n), 2):
        u = cnot(n, c, t)
        for bits in itertools.product((0, 1), repeat=n):
            idx = sum(b << (n - 1 - k) for k, b in enumerate(bits))
            out_bits = list(bits)
            out_bits[t] = bits[t] ^ bits[c]
            out_idx = sum(b << (n - 1 - k) for k, b in enumerate(out_bits))
            col = u[:, idx]
            assert abs(col[out_idx] - 1.0) < TOL
            assert abs(np.linalg.norm(col) - 1.0) < TOL


def test_single_qubit_placement():
    """single_qubit_gate must place the operator at the right tensor slot."""
    for target in range(4):
        got = single_qubit_gate(4, target, PAULI_X)
        expected = kron_list([PAULI_X if k == target else I2 for k in range(4)])
        assert np.allclose(got, expected, atol=TOL)


def test_hadamard_involution():
    assert np.allclose(HADAMARD @ HADAMARD, np.eye(2), atol=TOL)


def test_cnot_rejects_same_qubit():
    with pytest.raises(ValueError):
        cnot(4, 1, 1)
