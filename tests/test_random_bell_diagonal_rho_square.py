"""Random Bell-diagonal inputs: the circuit's 00 branch must equal rho^2.

The circuit output is computed independently from
    rho (x) rho -> V -> project |00> on (q3,q4) -> partial trace,
and only then compared with the analytic target.
"""

import numpy as np
import pytest

from _helpers import random_bell_diagonal_populations
from pqec_distill.analytics import matrix_square_normalized, p_success_bell_diagonal
from pqec_distill.bell_states import bell_diagonal_state, bell_populations
from pqec_distill.measurement import run_circuit_branches

TOL = 1e-12
N_SAMPLES = 120
RNG = np.random.default_rng(20260903)
POPULATIONS = random_bell_diagonal_populations(RNG, N_SAMPLES)


def test_sample_count():
    assert len(POPULATIONS) >= 100


@pytest.mark.parametrize("p", POPULATIONS)
def test_unnormalized_output_is_rho_squared(p):
    rho = bell_diagonal_state(p)
    branch = run_circuit_branches(rho, rho)[(0, 0)]
    assert np.linalg.norm(branch["rho_tilde"] - rho @ rho, "fro") < TOL


@pytest.mark.parametrize("p", POPULATIONS)
def test_conditional_output_matches_normalized_square(p):
    rho = bell_diagonal_state(p)
    branch = run_circuit_branches(rho, rho)[(0, 0)]
    target = matrix_square_normalized(rho)
    assert np.linalg.norm(branch["rho_out"] - target, "fro") < TOL


@pytest.mark.parametrize("p", POPULATIONS)
def test_output_populations_are_squared_and_renormalized(p):
    rho = bell_diagonal_state(p)
    branch = run_circuit_branches(rho, rho)[(0, 0)]
    got = bell_populations(branch["rho_out"])
    expected = p ** 2 / np.sum(p ** 2)
    assert np.max(np.abs(got - expected)) < TOL


@pytest.mark.parametrize("p", POPULATIONS[:40])
def test_output_stays_bell_diagonal(p):
    from pqec_distill.bell_states import bell_offdiagonal_norm
    rho = bell_diagonal_state(p)
    branch = run_circuit_branches(rho, rho)[(0, 0)]
    assert bell_offdiagonal_norm(branch["rho_out"]) < TOL


@pytest.mark.parametrize("p", POPULATIONS)
def test_success_probability_is_purity(p):
    rho = bell_diagonal_state(p)
    branch = run_circuit_branches(rho, rho)[(0, 0)]
    assert abs(branch["prob"] - float(np.real(np.trace(rho @ rho)))) < TOL
    assert abs(branch["prob"] - p_success_bell_diagonal(p)) < TOL
