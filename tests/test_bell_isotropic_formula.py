"""Bell-isotropic family: F_out, P_success and the output state."""

import numpy as np
import pytest

from pqec_distill.analytics import (
    F_of_eps, F_out_isotropic, eps_prime, isotropic_populations,
    p_success_isotropic, q_of_eps,
)
from pqec_distill.bell_states import bell_diagonal_state, bell_populations
from pqec_distill.measurement import run_circuit_success

TOL = 1e-12
EPS_POINTS = [0.0, 0.01, 0.1, 0.4, 2.0 / 3.0, 0.7, 0.8, 0.9, 1.0]


@pytest.mark.parametrize("eps", EPS_POINTS)
def test_input_fidelity(eps):
    rho = bell_diagonal_state(isotropic_populations(eps))
    assert abs(bell_populations(rho)[0] - float(F_of_eps(eps))) < TOL


@pytest.mark.parametrize("eps", EPS_POINTS)
def test_output_fidelity_matches_closed_form(eps):
    rho = bell_diagonal_state(isotropic_populations(eps))
    rho_out, _ = run_circuit_success(rho)
    assert abs(bell_populations(rho_out)[0] - float(F_out_isotropic(eps))) < TOL


@pytest.mark.parametrize("eps", EPS_POINTS)
def test_output_is_again_isotropic(eps):
    """rho_out = (1-eps')Phi+ + eps' I/4 for some eps'."""
    rho = bell_diagonal_state(isotropic_populations(eps))
    rho_out, _ = run_circuit_success(rho)
    got = bell_populations(rho_out)
    expected = isotropic_populations(float(eps_prime(eps)))
    assert np.max(np.abs(got - expected)) < TOL


@pytest.mark.parametrize("eps", EPS_POINTS)
def test_success_probability_closed_form(eps):
    f, q = float(F_of_eps(eps)), float(q_of_eps(eps))
    rho = bell_diagonal_state(isotropic_populations(eps))
    _, prob = run_circuit_success(rho)
    assert abs(prob - (f ** 2 + 3 * q ** 2)) < TOL
    assert abs(prob - float(p_success_isotropic(eps))) < TOL


def test_reference_values_at_eps_0p1():
    """Manually derived reference point, reproduced by the circuit."""
    rho = bell_diagonal_state(isotropic_populations(0.1))
    rho_out, prob = run_circuit_success(rho)
    assert abs(float(F_of_eps(0.1)) - 0.925) < 1e-15
    assert abs(prob - 0.8575) < 1e-12
    assert abs(float(eps_prime(0.1)) - 0.002915451895043732) < 1e-15
    assert abs(bell_populations(rho_out)[0] - 0.9978134110787172) < 1e-12


def test_maximally_mixed_is_a_fixed_point():
    rho = bell_diagonal_state(isotropic_populations(1.0))
    rho_out, prob = run_circuit_success(rho)
    assert abs(prob - 0.25) < TOL
    assert np.linalg.norm(rho_out - rho, "fro") < TOL


def test_pure_bell_is_a_fixed_point():
    rho = bell_diagonal_state(isotropic_populations(0.0))
    rho_out, prob = run_circuit_success(rho)
    assert abs(prob - 1.0) < TOL
    assert np.linalg.norm(rho_out - rho, "fro") < TOL
