"""Does the NOISY 5-CNOT circuit preserve the Bell-diagonal sector?

This matters for the wider research programme: in the 5-qubit SWAP-test study
some circuit realizations of the same ideal unitary leak out of the
Bell-diagonal sector once per-CNOT noise is inserted, which changes the
repeated-round dynamics qualitatively.  Here we test the question directly for
this 4-qubit circuit instead of assuming either answer.
"""

import numpy as np
import pytest

from _helpers import random_bell_diagonal_populations, random_density_matrix
from pqec_distill.bell_states import bell_diagonal_state, bell_offdiagonal_norm
from pqec_distill.measurement import postselect_branch
from pqec_distill.noise import NOISE_CONVENTIONS, noisy_full_channel_dm

TOL = 1e-12
RNG = np.random.default_rng(19937)
POPULATIONS = random_bell_diagonal_populations(RNG, 12)
BRANCHES = [(0, 0), (0, 1), (1, 0), (1, 1)]
P_VALUES = [0.0, 0.01, 0.05, 0.15, 0.4]


@pytest.mark.parametrize("convention", NOISE_CONVENTIONS)
@pytest.mark.parametrize("p", P_VALUES)
@pytest.mark.parametrize("pop", POPULATIONS[:6])
@pytest.mark.parametrize("branch", BRANCHES)
def test_bell_diagonal_input_stays_bell_diagonal(convention, p, pop, branch):
    """Bell-diagonal in -> Bell-diagonal out, in EVERY branch, at every p."""
    rho = bell_diagonal_state(pop)
    out4 = noisy_full_channel_dm(np.kron(rho, rho), p, convention)
    rho_tilde, prob = postselect_branch(out4, *branch)
    assert bell_offdiagonal_norm(rho_tilde) < TOL
    if prob > 1e-9:
        assert bell_offdiagonal_norm(rho_tilde / prob) < TOL


@pytest.mark.parametrize("convention", NOISE_CONVENTIONS)
@pytest.mark.parametrize("p", [0.02, 0.1, 0.3])
def test_two_different_bell_diagonal_copies(convention, p):
    """Closure does not rely on the two input copies being identical."""
    rng = np.random.default_rng(2024)
    for _ in range(8):
        rho_a = bell_diagonal_state(rng.dirichlet(np.ones(4)))
        rho_b = bell_diagonal_state(rng.dirichlet(np.ones(4)))
        out4 = noisy_full_channel_dm(np.kron(rho_a, rho_b), p, convention)
        for branch in BRANCHES:
            rho_tilde, _ = postselect_branch(out4, *branch)
            assert bell_offdiagonal_norm(rho_tilde) < TOL


@pytest.mark.parametrize("convention", NOISE_CONVENTIONS)
def test_non_bell_diagonal_input_does_leak(convention):
    """Control: the closure is a property of the INPUT sector, not of the noise.

    A non-Bell-diagonal input must still produce off-diagonal output, otherwise
    the test above would be vacuous (e.g. if the noise simply dephased
    everything).
    """
    rng = np.random.default_rng(4321)
    leaked = 0
    for _ in range(10):
        rho = random_density_matrix(rng)
        out4 = noisy_full_channel_dm(np.kron(rho, rho), 0.05, convention)
        rho_tilde, _ = postselect_branch(out4, 0, 0)
        if bell_offdiagonal_norm(rho_tilde) > 1e-6:
            leaked += 1
    assert leaked == 10


@pytest.mark.parametrize("convention", NOISE_CONVENTIONS)
@pytest.mark.parametrize("p", [0.0, 0.05, 0.2])
def test_noisy_output_is_a_valid_state(convention, p):
    rho = bell_diagonal_state([0.7, 0.1, 0.1, 0.1])
    out4 = noisy_full_channel_dm(np.kron(rho, rho), p, convention)
    total = 0.0
    for branch in BRANCHES:
        rho_tilde, prob = postselect_branch(out4, *branch)
        total += prob
        assert np.linalg.norm(rho_tilde - rho_tilde.conj().T, "fro") < TOL
        assert np.linalg.eigvalsh((rho_tilde + rho_tilde.conj().T) / 2).min() > -TOL
    assert abs(total - 1.0) < TOL


@pytest.mark.parametrize("p", [0.03, 0.12])
def test_noisy_conventions_related_by_rescaling(p):
    """Circuit-level consequence of D_Pauli,p = D_replace,16p/15."""
    rho = bell_diagonal_state([0.8, 0.1, 0.05, 0.05])
    rho4 = np.kron(rho, rho)
    a = noisy_full_channel_dm(rho4, p, "pauli")
    b = noisy_full_channel_dm(rho4, 16.0 * p / 15.0, "replace")
    assert np.linalg.norm(a - b, "fro") < TOL
