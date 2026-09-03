"""General (non-Bell-diagonal) inputs: the 00 branch is a Bell-basis SCHUR square.

The Kraus operator of the successful branch is
    K_00 = sum_i |B_i>_12 <B_i|_12 <B_i|_34
so the unnormalized output has Bell-basis entries (rho_ij)^2.
"""

import numpy as np
import pytest

from _helpers import random_density_matrix
from pqec_distill.analytics import schur_square_bell
from pqec_distill.bell_states import bell_basis_matrix, bell_state, to_bell_basis
from pqec_distill.circuit import full_unitary
from pqec_distill.measurement import measurement_projector, partial_trace, run_circuit_branches

TOL = 1e-12
RNG = np.random.default_rng(2718281)
STATES = [random_density_matrix(RNG) for _ in range(40)]


def _kraus_00():
    """Build K_00 = sum_i |B_i><B_i| (x) <B_i| explicitly (4 x 16 matrix)."""
    k = np.zeros((4, 16), dtype=complex)
    u = bell_basis_matrix()
    for i in range(4):
        bi = u[:, i]
        k += np.outer(bi, np.kron(bi.conj(), bi.conj()))
    return k


@pytest.mark.parametrize("rho", STATES)
def test_circuit_output_is_bell_schur_square(rho):
    branch = run_circuit_branches(rho, rho)[(0, 0)]
    expected = schur_square_bell(rho)
    assert np.linalg.norm(branch["rho_tilde"] - expected, "fro") < TOL


@pytest.mark.parametrize("rho", STATES[:15])
def test_bell_basis_entries_are_squared(rho):
    branch = run_circuit_branches(rho, rho)[(0, 0)]
    got = to_bell_basis(branch["rho_tilde"])
    expected = to_bell_basis(rho) ** 2
    assert np.max(np.abs(got - expected)) < TOL


@pytest.mark.parametrize("rho", STATES[:15])
def test_explicit_kraus_operator_reproduces_circuit(rho):
    """Independent route: apply the analytic K_00 directly."""
    k = _kraus_00()
    predicted = k @ np.kron(rho, rho) @ k.conj().T
    branch = run_circuit_branches(rho, rho)[(0, 0)]
    assert np.linalg.norm(predicted - branch["rho_tilde"], "fro") < TOL


def test_kraus_operator_matches_projected_circuit_definition():
    """K_00 must equal <00|_{q3q4} V, up to the retained-pair basis."""
    v = full_unitary()
    proj = measurement_projector(0, 0)
    # Deriving K_00 from the circuit: <00| V acting on the (q3,q4) slots.
    k_circuit = np.zeros((4, 16), dtype=complex)
    for col in range(16):
        out = (proj @ v)[:, col]
        # rows with q3=q4=0 are indices 4*i + 0 for i = q1q2
        k_circuit[:, col] = out[[0, 4, 8, 12]]
    k_analytic = _kraus_00()
    assert np.linalg.norm(k_circuit - k_analytic, "fro") < TOL
