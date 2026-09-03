"""The success projector Pi_00 = 1/4 (I + XXXX)(I + ZZZZ)."""

import numpy as np

from pqec_distill.bell_states import BELL_LABELS, bell_state
from pqec_distill.circuit import full_unitary
from pqec_distill.gates import PAULI_X, PAULI_Z, kron_list
from pqec_distill.measurement import measurement_projector

TOL = 1e-12
I16 = np.eye(16, dtype=complex)
XXXX = kron_list([PAULI_X] * 4)
ZZZZ = kron_list([PAULI_Z] * 4)
PI_00 = 0.25 * (I16 + XXXX) @ (I16 + ZZZZ)


def test_pi00_is_a_projector():
    assert np.allclose(PI_00 @ PI_00, PI_00, atol=TOL)
    assert np.allclose(PI_00, PI_00.conj().T, atol=TOL)


def test_pi00_rank_is_four():
    assert abs(np.real(np.trace(PI_00)) - 4.0) < TOL


def test_pi00_equals_conjugated_measurement_projector():
    """V^dag (|00><00| on q3,q4) V must equal Pi_00."""
    v = full_unitary()
    conj = v.conj().T @ measurement_projector(0, 0) @ v
    assert np.allclose(conj, PI_00, atol=TOL)


def test_plus_plus_sector_iff_labels_match():
    """A Bell-product state is in the (+1,+1) sector iff its labels match."""
    for (a, b) in BELL_LABELS:
        for (c, d) in BELL_LABELS:
            psi = np.kron(bell_state(a, b), bell_state(c, d))
            in_sector = np.linalg.norm(PI_00 @ psi - psi) < TOL
            assert in_sector == ((a, b) == (c, d))


def test_all_four_projectors_resolve_identity():
    total = sum(
        full_unitary().conj().T @ measurement_projector(m3, m4) @ full_unitary()
        for m3 in (0, 1) for m4 in (0, 1)
    )
    assert np.allclose(total, I16, atol=TOL)
