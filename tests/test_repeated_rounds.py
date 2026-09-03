"""Repeated ideal purification: p_i -> p_i^2 / sum p_j^2 and the closed form."""

import numpy as np
import pytest

from _helpers import random_bell_diagonal_populations
from pqec_distill.analytics import (
    isotropic_populations, p_total_repeated, per_level_success_probabilities,
    repeated_eps, repeated_populations,
)
from pqec_distill.analytics import repeated_populations_closed_form
from pqec_distill.bell_states import bell_diagonal_state, bell_populations
from pqec_distill.measurement import run_circuit_success

TOL = 1e-12
RNG = np.random.default_rng(606)
POPULATIONS = random_bell_diagonal_populations(RNG, 20)
LEVELS = [1, 2, 3, 4]


@pytest.mark.parametrize("p", POPULATIONS)
@pytest.mark.parametrize("levels", LEVELS)
def test_circuit_iteration_matches_recurrence(p, levels):
    """Iterating the PHYSICAL circuit must match the analytic recurrence."""
    rho = bell_diagonal_state(p)
    for _ in range(levels):
        rho, _ = run_circuit_success(rho)
    got = bell_populations(rho)
    expected = repeated_populations(p, levels)
    assert np.max(np.abs(got - expected)) < TOL


@pytest.mark.parametrize("p", POPULATIONS)
@pytest.mark.parametrize("levels", LEVELS)
def test_closed_form_power_law(p, levels):
    """p_i^(l) = p_i^(2^l) / sum_j p_j^(2^l)."""
    got = repeated_populations(p, levels)
    expected = repeated_populations_closed_form(p, levels)
    assert np.max(np.abs(got - expected)) < 1e-10


@pytest.mark.parametrize("p", POPULATIONS)
@pytest.mark.parametrize("levels", LEVELS)
def test_state_is_normalized_matrix_power(p, levels):
    """rho_l = rho^(2^l) / Tr[rho^(2^l)] for Bell-diagonal rho."""
    rho = bell_diagonal_state(p)
    iterated = rho.copy()
    for _ in range(levels):
        iterated, _ = run_circuit_success(iterated)
    power = np.linalg.matrix_power(rho, 2 ** levels)
    expected = power / np.trace(power)
    assert np.linalg.norm(iterated - expected, "fro") < 1e-10


@pytest.mark.parametrize("p", POPULATIONS)
@pytest.mark.parametrize("levels", LEVELS)
def test_total_success_probability_telescopes(p, levels):
    """Full-tree telescoping:  P_total(l) = prod_k P_k^(2^(l-k)) = Tr[rho^(2^l)].

    A depth-l tree has 2^(l-k) nodes at level k and ALL of them must succeed,
    so the per-level probabilities enter with those multiplicities.
    """
    rho = bell_diagonal_state(p)
    iterated = rho.copy()
    per_level = []
    for _ in range(levels):
        iterated, prob = run_circuit_success(iterated)
        per_level.append(prob)

    tree_total = 1.0
    for k, prob in enumerate(per_level, start=1):
        tree_total *= prob ** (2 ** (levels - k))

    expected = float(np.real(np.trace(np.linalg.matrix_power(rho, 2 ** levels))))
    assert abs(tree_total - expected) < 1e-12 * max(1.0, abs(expected)) + 1e-15
    assert abs(tree_total - p_total_repeated(p, levels)) < 1e-12


@pytest.mark.parametrize("p", POPULATIONS[:6])
@pytest.mark.parametrize("levels", [2, 3, 4])
def test_naive_per_level_product_is_not_the_tree_total(p, levels):
    """Guard against the (wrong) 'one node per level' reading of P_total."""
    naive = 1.0
    for prob in per_level_success_probabilities(p, levels):
        naive *= prob
    tree = p_total_repeated(p, levels)
    # They coincide only at levels == 1.
    assert naive > tree


@pytest.mark.parametrize("eps", [0.05, 0.1, 0.3, 0.6, 0.9])
def test_isotropic_converges_to_bell_state(eps):
    """Convergence is doubly exponential but slower for noisier inputs:
    eps = 0.9 needs ~7 rounds to reach 1 - 1e-9, whereas eps = 0.05 needs 2."""
    rho = bell_diagonal_state(isotropic_populations(eps))
    fidelities = []
    for _ in range(8):
        rho, _ = run_circuit_success(rho)
        fidelities.append(bell_populations(rho)[0])
    assert fidelities[-1] > 1.0 - 1e-9
    # fidelity must increase monotonically along the way
    assert all(b >= a - 1e-15 for a, b in zip(fidelities, fidelities[1:]))


@pytest.mark.parametrize("eps", [0.05, 0.1, 0.3, 0.6, 0.9])
def test_isotropic_eps_recursion_matches_circuit(eps):
    rho = bell_diagonal_state(isotropic_populations(eps))
    for _ in range(3):
        rho, _ = run_circuit_success(rho)
    eps_out = (1.0 - bell_populations(rho)[0]) * 4.0 / 3.0
    assert abs(eps_out - repeated_eps(eps, 3)) < 1e-12


def test_maximally_mixed_does_not_purify():
    rho = bell_diagonal_state([0.25] * 4)
    for _ in range(4):
        rho, _ = run_circuit_success(rho)
    assert np.max(np.abs(bell_populations(rho) - 0.25)) < TOL
