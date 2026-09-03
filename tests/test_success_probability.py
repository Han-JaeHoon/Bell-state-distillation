"""P_success = Tr(rho^2) = sum_i p_i^2 for Bell-diagonal inputs."""

import numpy as np
import pytest

from _helpers import random_bell_diagonal_populations
from pqec_distill.analytics import (
    isotropic_populations, p_success_bell_diagonal, p_success_isotropic,
)
from pqec_distill.bell_states import bell_diagonal_state
from pqec_distill.measurement import run_circuit_success

TOL = 1e-12
RNG = np.random.default_rng(4242)


@pytest.mark.parametrize("eps", [0.0, 0.01, 0.1, 0.4, 2.0 / 3.0, 0.7, 0.8, 0.9, 1.0])
def test_isotropic_success_probability(eps):
    rho = bell_diagonal_state(isotropic_populations(eps))
    _, prob = run_circuit_success(rho)
    assert abs(prob - float(p_success_isotropic(eps))) < TOL


def test_eps_point_one_reference_value():
    """Manually derived reference: P_success(0.1) = 0.925^2 + 3*0.025^2 = 0.8575."""
    rho = bell_diagonal_state(isotropic_populations(0.1))
    _, prob = run_circuit_success(rho)
    assert abs(prob - 0.8575) < 1e-12


@pytest.mark.parametrize("p", random_bell_diagonal_populations(RNG, 40))
def test_random_success_probability(p):
    rho = bell_diagonal_state(p)
    _, prob = run_circuit_success(rho)
    assert abs(prob - p_success_bell_diagonal(p)) < TOL


def test_pure_bell_input_succeeds_with_certainty():
    rho = bell_diagonal_state([1.0, 0.0, 0.0, 0.0])
    _, prob = run_circuit_success(rho)
    assert abs(prob - 1.0) < TOL


def test_maximally_mixed_input_success_is_one_quarter():
    rho = bell_diagonal_state([0.25] * 4)
    _, prob = run_circuit_success(rho)
    assert abs(prob - 0.25) < TOL
