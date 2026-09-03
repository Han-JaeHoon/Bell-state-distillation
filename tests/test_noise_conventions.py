"""Noise channels: CPTP properties, p=0 reduction, and the exact relation
D_Pauli,p = D_replace,(16p/15)."""

import numpy as np
import pytest

from _helpers import random_density_matrix
from pqec_distill.analytics import isotropic_populations
from pqec_distill.bell_states import bell_diagonal_state
from pqec_distill.circuit import apply_unitary_dm, full_unitary
from pqec_distill.gates import I2, kron_list
from pqec_distill.noise import (
    NOISE_CONVENTIONS, noisy_full_channel_dm, pauli_depolarizing,
    replacement_depolarizing,
)

TOL = 1e-12
RNG = np.random.default_rng(555)


@pytest.mark.parametrize("convention", NOISE_CONVENTIONS)
def test_p_zero_reduces_to_ideal(convention):
    rho4 = random_density_matrix(RNG, 16)
    got = noisy_full_channel_dm(rho4, 0.0, convention)
    ideal = apply_unitary_dm(rho4, full_unitary())
    assert np.linalg.norm(got - ideal, "fro") < TOL


@pytest.mark.parametrize("convention", NOISE_CONVENTIONS)
@pytest.mark.parametrize("p", [0.0, 0.01, 0.1, 0.5, 0.9])
def test_channel_is_trace_preserving_and_positive(convention, p):
    rho4 = random_density_matrix(RNG, 16)
    out = noisy_full_channel_dm(rho4, p, convention)
    assert abs(np.real(np.trace(out)) - 1.0) < TOL
    assert np.linalg.norm(out - out.conj().T, "fro") < TOL
    assert np.linalg.eigvalsh((out + out.conj().T) / 2).min() > -TOL


def test_replacement_places_identity_on_the_right_qubits():
    rng = np.random.default_rng(88)
    factors = [random_density_matrix(rng, 2) for _ in range(4)]
    rho = kron_list(factors)
    for pair in [(0, 3), (1, 2), (2, 0), (2, 1)]:
        got = replacement_depolarizing(rho, pair, 1.0)
        expected_factors = [
            I2 / 2 if k in pair else factors[k] for k in range(4)
        ]
        assert np.linalg.norm(got - kron_list(expected_factors), "fro") < TOL


@pytest.mark.parametrize("p", [0.0, 0.05, 0.13, 0.5, 15.0 / 16.0])
def test_pauli_equals_rescaled_replacement(p):
    """D_Pauli,p == D_replace,(16p/15) exactly."""
    rho = random_density_matrix(RNG, 16)
    for pair in [(0, 3), (2, 1)]:
        lhs = pauli_depolarizing(rho, pair, p)
        rhs = replacement_depolarizing(rho, pair, 16.0 * p / 15.0)
        assert np.linalg.norm(lhs - rhs, "fro") < TOL


def test_conventions_differ_at_the_same_p():
    """Guard against treating the two convention labels as interchangeable."""
    rho = bell_diagonal_state(isotropic_populations(0.2))
    rho4 = np.kron(rho, rho)
    a = noisy_full_channel_dm(rho4, 0.1, "replace")
    b = noisy_full_channel_dm(rho4, 0.1, "pauli")
    assert np.linalg.norm(a - b, "fro") > 1e-6


def test_unknown_convention_rejected():
    rho4 = random_density_matrix(RNG, 16)
    with pytest.raises(ValueError):
        noisy_full_channel_dm(rho4, 0.1, "depolarizing")
