"""Without postselection the retained pair must reduce back to the input rho.

sum_{mu,nu} rho_tilde_{mu,nu} = rho   (physical sanity check)
"""

import numpy as np
import pytest

from _helpers import random_bell_diagonal_populations, random_density_matrix
from pqec_distill.analytics import isotropic_populations
from pqec_distill.bell_states import bell_diagonal_state
from pqec_distill.measurement import run_circuit_branches

TOL = 1e-12
RNG = np.random.default_rng(31337)
BRANCHES = [(0, 0), (0, 1), (1, 0), (1, 1)]


@pytest.mark.parametrize("p", random_bell_diagonal_populations(RNG, 30))
def test_bell_diagonal_no_postselection(p):
    rho = bell_diagonal_state(p)
    branches = run_circuit_branches(rho, rho)
    total = sum(branches[b]["rho_tilde"] for b in BRANCHES)
    assert np.linalg.norm(total - rho, "fro") < TOL


@pytest.mark.parametrize("eps", [0.0, 0.1, 0.5, 2.0 / 3.0, 1.0])
def test_isotropic_no_postselection(eps):
    rho = bell_diagonal_state(isotropic_populations(eps))
    branches = run_circuit_branches(rho, rho)
    total = sum(branches[b]["rho_tilde"] for b in BRANCHES)
    assert np.linalg.norm(total - rho, "fro") < TOL


def test_holds_for_bell_diagonal_A_and_arbitrary_B():
    """The identity needs only the RETAINED pair to be Bell diagonal.

    rho_B may be an arbitrary (non-Bell-diagonal) state, and rho_A need not
    equal rho_B.
    """
    rng = np.random.default_rng(9001)
    for _ in range(20):
        rho_a = bell_diagonal_state(rng.dirichlet(np.ones(4)))
        rho_b = random_density_matrix(rng)
        branches = run_circuit_branches(rho_a, rho_b)
        total = sum(branches[b]["rho_tilde"] for b in BRANCHES)
        assert np.linalg.norm(total - rho_a, "fro") < TOL


def test_fails_for_non_bell_diagonal_retained_pair():
    """COUNTEREXAMPLE: if rho_A is NOT Bell diagonal the identity breaks.

    The circuit's CNOTs q3->q1 and q3->q2 act on the retained pair, so the
    retained marginal is only preserved on the Bell-diagonal operator sector
    (see test_retained_marginal_preserved_only_on_bell_sector below).
    """
    rng = np.random.default_rng(4004)
    failures = 0
    for _ in range(20):
        rho_a = random_density_matrix(rng)
        rho_b = random_density_matrix(rng)
        branches = run_circuit_branches(rho_a, rho_b)
        total = sum(branches[b]["rho_tilde"] for b in BRANCHES)
        if np.linalg.norm(total - rho_a, "fro") > 1e-6:
            failures += 1
    assert failures == 20


def test_retained_marginal_preserved_only_on_bell_sector():
    """Explain the above: V^dag (P_A (x) II) V is supported on A alone exactly
    for P_A in {XX, YY, ZZ}, i.e. on span{II, XX, YY, ZZ} = S_BD.

    Verified by the independent rule-based Pauli propagation route.
    """
    import itertools

    from pqec_distill.circuit import CNOT_SEQUENCE, Q3
    from pqec_distill.pauli_propagation import PauliString, conjugate_by_circuit

    def pauli_on_A(c1, c2):
        ps = PauliString(4)
        for qubit, letter in ((0, c1), (1, c2)):
            if letter == "X":
                ps.x[qubit] = 1
            elif letter == "Z":
                ps.z[qubit] = 1
            elif letter == "Y":
                ps.x[qubit] = ps.z[qubit] = 1
        return ps

    a_local = set()
    for c1, c2 in itertools.product("IXYZ", repeat=2):
        if (c1, c2) == ("I", "I"):
            continue
        out = conjugate_by_circuit(pauli_on_A(c1, c2), CNOT_SEQUENCE, Q3)
        letters = out.letters()
        if letters[2:] == "II":
            a_local.add(c1 + c2)
            # and on that sector it must act as the IDENTITY
            assert letters[:2] == c1 + c2
            assert out.sign == 0

    assert a_local == {"XX", "YY", "ZZ"}
