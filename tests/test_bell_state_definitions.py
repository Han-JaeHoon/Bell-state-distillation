"""Bell-state definitions: normalization, orthogonality, ZZ/XX eigenvalues,
and agreement between the two INDEPENDENT constructions (defining formula in
bell_states.py vs literal vectors in analytics.py)."""

import numpy as np
import pytest

from pqec_distill.analytics import bell_basis_literal
from pqec_distill.bell_states import (
    BELL_LABELS, BELL_NAMES, bell_basis_matrix, bell_index, bell_projector,
    bell_state, from_bell_basis, to_bell_basis,
)
from pqec_distill.gates import PAULI_X, PAULI_Z, kron_list

TOL = 1e-13
ZZ = kron_list([PAULI_Z, PAULI_Z])
XX = kron_list([PAULI_X, PAULI_X])


@pytest.mark.parametrize("a,b", BELL_LABELS)
def test_normalized(a, b):
    assert abs(np.linalg.norm(bell_state(a, b)) - 1.0) < TOL


def test_orthonormal_basis():
    u = bell_basis_matrix()
    assert np.allclose(u.conj().T @ u, np.eye(4), atol=TOL)


@pytest.mark.parametrize("a,b", BELL_LABELS)
def test_zz_eigenvalue(a, b):
    v = bell_state(a, b)
    assert np.allclose(ZZ @ v, ((-1.0) ** a) * v, atol=TOL)


@pytest.mark.parametrize("a,b", BELL_LABELS)
def test_xx_eigenvalue(a, b):
    v = bell_state(a, b)
    assert np.allclose(XX @ v, ((-1.0) ** b) * v, atol=TOL)


def test_naming_convention():
    assert BELL_NAMES[bell_index(0, 0)] == "Phi+"
    assert BELL_NAMES[bell_index(0, 1)] == "Phi-"
    assert BELL_NAMES[bell_index(1, 0)] == "Psi+"
    assert BELL_NAMES[bell_index(1, 1)] == "Psi-"


def test_independent_constructions_agree():
    """The defining-formula basis and the literal analytic basis must match."""
    assert np.allclose(bell_basis_matrix(), bell_basis_literal(), atol=TOL)


def test_bell_basis_roundtrip():
    rng = np.random.default_rng(11)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = a @ a.conj().T
    assert np.allclose(from_bell_basis(to_bell_basis(rho)), rho, atol=TOL)


def test_projectors_orthogonal_and_complete():
    projs = [bell_projector(a, b) for (a, b) in BELL_LABELS]
    total = sum(projs)
    assert np.allclose(total, np.eye(4), atol=TOL)
    for i in range(4):
        for j in range(4):
            prod = projs[i] @ projs[j]
            expected = projs[i] if i == j else np.zeros((4, 4))
            assert np.allclose(prod, expected, atol=TOL)
