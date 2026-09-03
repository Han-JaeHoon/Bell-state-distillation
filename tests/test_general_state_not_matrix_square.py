"""Counterexample search: for non-Bell-diagonal input the circuit does NOT give
rho^2 / Tr(rho^2).  This is the central limitation of the protocol."""

import numpy as np
import pytest

from _helpers import random_density_matrix
from pqec_distill.analytics import matrix_square_normalized, schur_square_bell
from pqec_distill.bell_states import (
    bell_diagonal_state, bell_offdiagonal_norm, to_bell_basis,
)
from pqec_distill.measurement import run_circuit_branches

TOL = 1e-12
RNG = np.random.default_rng(1414213)
STATES = [random_density_matrix(RNG) for _ in range(40)]


@pytest.mark.parametrize("rho", STATES)
def test_random_states_are_not_bell_diagonal(rho):
    """Guard: the counterexample search must actually use non-diagonal states."""
    assert bell_offdiagonal_norm(rho) > 1e-3


@pytest.mark.parametrize("rho", STATES)
def test_circuit_differs_from_matrix_square(rho):
    branch = run_circuit_branches(rho, rho)[(0, 0)]
    got = branch["rho_out"]
    matrix_square = matrix_square_normalized(rho)
    assert np.linalg.norm(got - matrix_square, "fro") > 1e-6


@pytest.mark.parametrize("rho", STATES)
def test_circuit_agrees_with_schur_square(rho):
    """...while it always agrees with the Bell-basis elementwise square."""
    branch = run_circuit_branches(rho, rho)[(0, 0)]
    assert np.linalg.norm(branch["rho_tilde"] - schur_square_bell(rho), "fro") < TOL


def test_explicit_minimal_counterexample():
    """A concrete, reproducible non-Bell-diagonal counterexample."""
    rb = np.diag([0.5, 0.3, 0.15, 0.05]).astype(complex)
    rb[0, 1] = rb[1, 0] = 0.2
    from pqec_distill.bell_states import from_bell_basis
    rho = from_bell_basis(rb)
    assert np.linalg.eigvalsh(rho).min() > -1e-12
    branch = run_circuit_branches(rho, rho)[(0, 0)]
    schur = schur_square_bell(rho)
    assert np.linalg.norm(branch["rho_tilde"] - schur, "fro") < TOL
    diff = np.linalg.norm(branch["rho_out"] - matrix_square_normalized(rho), "fro")
    assert diff > 1e-3
    # The discrepancy is exactly the off-diagonal (coherence) contribution.
    bell_sq = to_bell_basis(rho @ rho)
    bell_schur = to_bell_basis(schur)
    assert abs(bell_sq[0, 0] - bell_schur[0, 0]) > 1e-3


def test_equivalence_is_restored_for_bell_diagonal_states():
    """Sanity: the same comparison passes when coherences are removed."""
    rho = bell_diagonal_state([0.5, 0.3, 0.15, 0.05])
    branch = run_circuit_branches(rho, rho)[(0, 0)]
    assert np.linalg.norm(
        branch["rho_out"] - matrix_square_normalized(rho), "fro"
    ) < TOL
