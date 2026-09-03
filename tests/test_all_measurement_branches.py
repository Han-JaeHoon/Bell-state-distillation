"""All four measurement branches, their probabilities and normalization."""

import numpy as np
import pytest

from _helpers import random_bell_diagonal_populations
from pqec_distill.analytics import (
    predicted_branch_populations, predicted_branch_probability,
)
from pqec_distill.bell_states import bell_diagonal_state, bell_populations
from pqec_distill.measurement import run_circuit_branches

TOL = 1e-12
RNG = np.random.default_rng(777)
POPULATIONS = random_bell_diagonal_populations(RNG, 30)
BRANCHES = [(0, 0), (0, 1), (1, 0), (1, 1)]


@pytest.mark.parametrize("p", POPULATIONS)
@pytest.mark.parametrize("mu,nu", BRANCHES)
def test_branch_unnormalized_populations(p, mu, nu):
    """rho_tilde_{mu,nu} = sum_ab p_ab p_{a xor nu, b xor mu} B_ab."""
    rho = bell_diagonal_state(p)
    branch = run_circuit_branches(rho, rho)[(mu, nu)]
    got = bell_populations(branch["rho_tilde"])
    expected = predicted_branch_populations(p, mu, nu)
    assert np.max(np.abs(got - expected)) < TOL


@pytest.mark.parametrize("p", POPULATIONS)
@pytest.mark.parametrize("mu,nu", BRANCHES)
def test_branch_probability(p, mu, nu):
    rho = bell_diagonal_state(p)
    branch = run_circuit_branches(rho, rho)[(mu, nu)]
    assert abs(branch["prob"] - predicted_branch_probability(p, mu, nu)) < TOL


@pytest.mark.parametrize("p", POPULATIONS)
@pytest.mark.parametrize("mu,nu", BRANCHES)
def test_branches_stay_bell_diagonal(p, mu, nu):
    from pqec_distill.bell_states import bell_offdiagonal_norm
    rho = bell_diagonal_state(p)
    branch = run_circuit_branches(rho, rho)[(mu, nu)]
    assert bell_offdiagonal_norm(branch["rho_tilde"]) < TOL


@pytest.mark.parametrize("p", POPULATIONS)
def test_branch_probabilities_sum_to_one(p):
    rho = bell_diagonal_state(p)
    branches = run_circuit_branches(rho, rho)
    total = sum(branches[b]["prob"] for b in BRANCHES)
    assert abs(total - 1.0) < TOL
