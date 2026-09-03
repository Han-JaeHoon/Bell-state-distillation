"""Output density matrix in the computational basis, and its state properties."""

import numpy as np
import pytest

from pqec_distill.analytics import (
    eps_prime, isotropic_populations, rho_out_isotropic_computational,
)
from pqec_distill.bell_states import bell_diagonal_state, bell_populations
from pqec_distill.measurement import run_circuit_success

TOL = 1e-12
EPS_POINTS = [0.0, 0.01, 0.1, 0.4, 2.0 / 3.0, 0.7, 0.8, 0.9, 1.0]


@pytest.mark.parametrize("eps", EPS_POINTS)
def test_matches_explicit_matrix(eps):
    rho = bell_diagonal_state(isotropic_populations(eps))
    rho_out, _ = run_circuit_success(rho)
    expected = rho_out_isotropic_computational(eps)
    assert np.linalg.norm(rho_out - expected, "fro") < TOL


@pytest.mark.parametrize("eps", EPS_POINTS)
def test_is_a_valid_state(eps):
    rho = bell_diagonal_state(isotropic_populations(eps))
    rho_out, _ = run_circuit_success(rho)
    assert abs(np.real(np.trace(rho_out)) - 1.0) < TOL
    assert np.linalg.norm(rho_out - rho_out.conj().T, "fro") < TOL
    eigs = np.linalg.eigvalsh((rho_out + rho_out.conj().T) / 2)
    assert eigs.min() > -TOL


@pytest.mark.parametrize("eps", EPS_POINTS)
def test_purity(eps):
    """Purity of an isotropic state with parameter eps' is (1-3e/4)^2 + 3(e/4)^2."""
    rho = bell_diagonal_state(isotropic_populations(eps))
    rho_out, _ = run_circuit_success(rho)
    ep = float(eps_prime(eps))
    expected = (1 - 3 * ep / 4) ** 2 + 3 * (ep / 4) ** 2
    assert abs(float(np.real(np.trace(rho_out @ rho_out))) - expected) < TOL


@pytest.mark.parametrize("eps", EPS_POINTS)
def test_bell_fidelity_from_matrix_elements(eps):
    """F = <Phi+|rho|Phi+> = rho_00 + rho_03 for this matrix shape."""
    rho = bell_diagonal_state(isotropic_populations(eps))
    rho_out, _ = run_circuit_success(rho)
    f_from_elements = float(np.real(rho_out[0, 0] + rho_out[0, 3]))
    assert abs(f_from_elements - bell_populations(rho_out)[0]) < TOL
