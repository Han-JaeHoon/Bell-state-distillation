"""All 16 Bell-product inputs through the exact circuit.

Verifies  V |B_ab>|B_cd> = |B_ab> |b xor d>_3 |a xor c>_4  up to global phase,
and that the retained pair is the FIRST input Bell state.
"""

import numpy as np
import pytest

from pqec_distill.analytics import predicted_measurement_bits
from pqec_distill.bell_states import BELL_LABELS, BELL_NAMES, bell_index, bell_state
from pqec_distill.circuit import full_unitary

TOL = 1e-12
CASES = [(a, b, c, d) for (a, b) in BELL_LABELS for (c, d) in BELL_LABELS]


def _computational_basis_vector(bit3, bit4):
    v = np.zeros(4, dtype=complex)
    v[2 * bit3 + bit4] = 1.0
    return v


def _phase_aligned_error(got, expected):
    """Frobenius error after removing an irrelevant global phase."""
    overlap = np.vdot(expected, got)
    phase = overlap / abs(overlap) if abs(overlap) > 1e-14 else 1.0
    return float(np.linalg.norm(got - phase * expected))


def test_there_are_16_cases():
    assert len(CASES) == 16


@pytest.mark.parametrize("a,b,c,d", CASES)
def test_bell_product_case(a, b, c, d):
    v = full_unitary()
    psi_in = np.kron(bell_state(a, b), bell_state(c, d))
    got = v @ psi_in

    m3, m4 = predicted_measurement_bits(a, b, c, d)
    expected = np.kron(bell_state(a, b), _computational_basis_vector(m3, m4))

    assert _phase_aligned_error(got, expected) < TOL


@pytest.mark.parametrize("a,b,c,d", CASES)
def test_measurement_bits_are_deterministic(a, b, c, d):
    """The output must be supported on a single (q3,q4) computational branch."""
    v = full_unitary()
    got = v @ np.kron(bell_state(a, b), bell_state(c, d))
    probs = np.zeros((2, 2))
    for i, amp in enumerate(got):
        b3 = (i >> 1) & 1
        b4 = i & 1
        probs[b3, b4] += abs(amp) ** 2
    m3, m4 = predicted_measurement_bits(a, b, c, d)
    assert abs(probs[m3, m4] - 1.0) < TOL
    assert probs.sum() - probs[m3, m4] < TOL


def test_success_iff_labels_match():
    """(m3,m4) = (0,0) exactly when the two Bell labels are identical."""
    for (a, b, c, d) in CASES:
        m3, m4 = predicted_measurement_bits(a, b, c, d)
        assert ((m3, m4) == (0, 0)) == ((a, b) == (c, d))


def test_expected_table_matches_prompt():
    """Reproduce the manually derived (m3,m4) table exactly."""
    expected_table = {
        ("Phi+", "Phi+"): "00", ("Phi+", "Phi-"): "10", ("Phi+", "Psi+"): "01", ("Phi+", "Psi-"): "11",
        ("Phi-", "Phi+"): "10", ("Phi-", "Phi-"): "00", ("Phi-", "Psi+"): "11", ("Phi-", "Psi-"): "01",
        ("Psi+", "Phi+"): "01", ("Psi+", "Phi-"): "11", ("Psi+", "Psi+"): "00", ("Psi+", "Psi-"): "10",
        ("Psi-", "Phi+"): "11", ("Psi-", "Phi-"): "01", ("Psi-", "Psi+"): "10", ("Psi-", "Psi-"): "00",
    }
    v = full_unitary()
    for (a, b, c, d) in CASES:
        got = v @ np.kron(bell_state(a, b), bell_state(c, d))
        idx = int(np.argmax(np.abs(got)))
        m3, m4 = (idx >> 1) & 1, idx & 1
        key = (BELL_NAMES[bell_index(a, b)], BELL_NAMES[bell_index(c, d)])
        assert f"{m3}{m4}" == expected_table[key], key
