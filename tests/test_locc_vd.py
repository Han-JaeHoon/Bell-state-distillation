"""Blind circuit-level tests of the six-qubit distributed (LOCC) SWAP-test
estimator against the ordinary five-qubit SWAP test.

Nothing here computes a matrix square or imports an analytic estimator: both
sides come from density-matrix circuits built out of gates.
"""

import numpy as np
import pytest

from pqec_distill.locc_vd import (
    A, A1, A2, A1_5, A2_5, ALICE_WIRES_6Q, B, B1, B2, B1_5, B2_5,
    BOB_WIRES_6Q, C5, N_QUBITS_5Q, N_QUBITS_6Q, PAULI, PAULI_LABELS,
    WIRES_5Q, WIRES_6Q, bell_fidelity_numerator_from_deltas, data_input,
    delta_5q, delta_5q_all_paulis, delta_6q, delta_6q_all_paulis,
    embed_three_qubit_gate, final_state_5q, final_state_6q, fredkin_matrix,
    normalised_bell_fidelity, pauli_product, plus_state_dm,
    reconstruct_operator, s_a_6q, s_b_6q, swap_on, unitary_5q,
    unitary_5q_permutation, unitary_6q_route_a, unitary_6q_route_b,
)

TOL = 1e-12


# ---------------------------------------------------------------------------
# helpers (local to the tests, so the tests do not depend on project helpers)
# ---------------------------------------------------------------------------

def _basis_index(bits) -> int:
    n = len(bits)
    return sum(b << (n - 1 - w) for w, b in enumerate(bits))


def _basis_bits(idx: int, n: int) -> list:
    return [(idx >> (n - 1 - w)) & 1 for w in range(n)]


def _random_mixed(rng, rank=4):
    m = (rng.normal(size=(4, rank)) + 1j * rng.normal(size=(4, rank)))
    rho = m @ m.conj().T
    return rho / np.trace(rho)


def _random_pure(rng):
    v = rng.normal(size=4) + 1j * rng.normal(size=4)
    v = v / np.linalg.norm(v)
    return np.outer(v, v.conj())


def _bell_diagonal(p):
    phi_p = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    phi_m = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
    psi_p = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_m = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    rho = np.zeros((4, 4), dtype=complex)
    for w, v in zip(p, (phi_p, phi_m, psi_p, psi_m)):
        rho = rho + w * np.outer(v, v.conj())
    return rho


def _sample_states(n_random=12):
    rng = np.random.default_rng(20261007)
    states = [("Phi+", _bell_diagonal([1, 0, 0, 0])),
              ("Psi-", _bell_diagonal([0, 0, 0, 1])),
              ("isotropic_0.3", _bell_diagonal([1 - 3 * 0.3 / 4] + [0.3 / 4] * 3)),
              ("maximally_mixed", np.eye(4, dtype=complex) / 4)]
    for k in range(n_random):
        states.append((f"pure_{k}", _random_pure(rng)))
        states.append((f"mixed_{k}", _random_mixed(rng)))
        states.append((f"rank2_{k}", _random_mixed(rng, rank=2)))
    return states


# ---------------------------------------------------------------------------
# 1. the ordering is exactly the specified one, and mistakes are detectable
# ---------------------------------------------------------------------------

def test_six_qubit_ordering_is_exactly_as_specified():
    assert WIRES_6Q == ("a", "b", "A1", "B1", "A2", "B2")
    assert (A, B, A1, B1, A2, B2) == (0, 1, 2, 3, 4, 5)
    assert N_QUBITS_6Q == 6


def test_five_qubit_ordering_is_exactly_as_specified():
    assert WIRES_5Q == ("c", "A1", "B1", "A2", "B2")
    assert (C5, A1_5, B1_5, A2_5, B2_5) == (0, 1, 2, 3, 4)
    assert N_QUBITS_5Q == 5


def test_alice_fredkin_moves_exactly_the_A_bits():
    """A tensor-ordering slip would move the wrong bits; this pins them down.

    |a=1, b=0, A1=1, B1=0, A2=0, B2=0>  ->  |1, 0, 0, 0, 1, 0>
    (A1 and A2 exchanged, every other wire untouched).
    """
    u = embed_three_qubit_gate(fredkin_matrix(), (A, A1, A2), N_QUBITS_6Q)
    src = _basis_index([1, 0, 1, 0, 0, 0])
    out = u @ np.eye(64, dtype=complex)[:, src]
    hits = np.nonzero(np.abs(out) > TOL)[0]
    assert len(hits) == 1
    assert _basis_bits(int(hits[0]), 6) == [1, 0, 0, 0, 1, 0]


def test_bob_fredkin_moves_exactly_the_B_bits():
    u = embed_three_qubit_gate(fredkin_matrix(), (B, B1, B2), N_QUBITS_6Q)
    src = _basis_index([0, 1, 0, 1, 0, 0])
    out = u @ np.eye(64, dtype=complex)[:, src]
    hits = np.nonzero(np.abs(out) > TOL)[0]
    assert len(hits) == 1
    assert _basis_bits(int(hits[0]), 6) == [0, 1, 0, 0, 0, 1]


def test_fredkin_is_inert_when_its_control_is_zero():
    u = unitary_6q_route_a()
    for bits in ([0, 0, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 0]):
        src = _basis_index(bits)
        out = u @ np.eye(64, dtype=complex)[:, src]
        assert np.allclose(out, np.eye(64)[:, src], atol=TOL)


def test_data_input_places_the_first_copy_on_A1B1():
    """rho (x) sigma must put rho on (A1,B1); a swapped kron would not."""
    rho = _bell_diagonal([1, 0, 0, 0])
    sigma = np.diag([1.0, 0, 0, 0]).astype(complex)
    joint = np.kron(rho, sigma)
    # marginal on the second copy must be |00><00|
    t = joint.reshape(2, 2, 2, 2, 2, 2, 2, 2)
    second = np.einsum("ijklijmn->klmn", t).reshape(4, 4)
    assert np.allclose(second, sigma, atol=TOL)
    assert np.allclose(data_input(rho), np.kron(rho, rho), atol=TOL)


# ---------------------------------------------------------------------------
# 2. unitarity, locality and the branch structure
# ---------------------------------------------------------------------------

def test_fredkin_embeddings_are_unitary():
    g = fredkin_matrix()
    assert np.allclose(g.conj().T @ g, np.eye(8), atol=TOL)
    for wires in ((A, A1, A2), (B, B1, B2)):
        u = embed_three_qubit_gate(g, wires, N_QUBITS_6Q)
        assert np.allclose(u.conj().T @ u, np.eye(64), atol=TOL)
        assert np.allclose(u @ u.conj().T, np.eye(64), atol=TOL)


def test_route_unitaries_are_unitary():
    for u in (unitary_6q_route_a(), unitary_6q_route_b()):
        assert np.allclose(u.conj().T @ u, np.eye(64), atol=TOL)
    for u in (unitary_5q(), unitary_5q_permutation()):
        assert np.allclose(u.conj().T @ u, np.eye(32), atol=TOL)


def _single_qubit_on(mat, wire, n):
    factors = [np.eye(2, dtype=complex)] * n
    factors[wire] = mat
    out = np.array([[1.0 + 0j]])
    for f in factors:
        out = np.kron(out, f)
    return out


def test_alice_gate_touches_only_alice_wires():
    """Alice's Fredkin commutes with every Pauli on Bob's wires."""
    u = embed_three_qubit_gate(fredkin_matrix(), (A, A1, A2), N_QUBITS_6Q)
    for wire in BOB_WIRES_6Q:
        for name in ("X", "Y", "Z"):
            p = _single_qubit_on(PAULI[name], wire, N_QUBITS_6Q)
            assert np.allclose(u @ p, p @ u, atol=TOL), (wire, name)


def test_bob_gate_touches_only_bob_wires():
    u = embed_three_qubit_gate(fredkin_matrix(), (B, B1, B2), N_QUBITS_6Q)
    for wire in ALICE_WIRES_6Q:
        for name in ("X", "Y", "Z"):
            p = _single_qubit_on(PAULI[name], wire, N_QUBITS_6Q)
            assert np.allclose(u @ p, p @ u, atol=TOL), (wire, name)


def test_local_fredkins_commute_so_the_order_is_immaterial():
    g = fredkin_matrix()
    ua = embed_three_qubit_gate(g, (A, A1, A2), N_QUBITS_6Q)
    ub = embed_three_qubit_gate(g, (B, B1, B2), N_QUBITS_6Q)
    assert np.allclose(ua @ ub, ub @ ua, atol=TOL)


def test_s_a_and_s_b_commute():
    sa, sb = s_a_6q(), s_b_6q()
    assert np.allclose(sa @ sb, sb @ sa, atol=TOL)


def test_s_a_s_b_exchanges_complete_bipartite_copies():
    """S_A S_B maps (A1,B1) <-> (A2,B2) on every computational-basis state."""
    prod = s_a_6q() @ s_b_6q()
    for idx in range(64):
        bits = _basis_bits(idx, 6)
        want = [bits[0], bits[1], bits[4], bits[5], bits[2], bits[3]]
        out = prod @ np.eye(64, dtype=complex)[:, idx]
        hits = np.nonzero(np.abs(out) > TOL)[0]
        assert len(hits) == 1
        assert _basis_bits(int(hits[0]), 6) == want


def test_four_control_branches_match_the_structure_table():
    """ab = 00 -> I, 01 -> S_B, 10 -> S_A, 11 -> S_A S_B, as a structure check."""
    u = unitary_6q_route_a()
    sa, sb = s_a_6q(), s_b_6q()
    expected = {(0, 0): np.eye(64, dtype=complex), (0, 1): sb,
                (1, 0): sa, (1, 1): sa @ sb}
    for (a_bit, b_bit), branch in expected.items():
        for data in range(16):
            bits = [a_bit, b_bit] + _basis_bits(data, 4)
            idx = _basis_index(bits)
            col = np.eye(64, dtype=complex)[:, idx]
            assert np.allclose(u @ col, branch @ col, atol=TOL), (a_bit, b_bit, data)


# ---------------------------------------------------------------------------
# 3. the two independent six-qubit routes agree
# ---------------------------------------------------------------------------

def test_route_a_equals_route_b_as_matrices():
    assert np.allclose(unitary_6q_route_a(), unitary_6q_route_b(), atol=TOL)


def test_route_a_equals_route_b_on_random_states():
    ua, ub = unitary_6q_route_a(), unitary_6q_route_b()
    for _name, rho in _sample_states(4):
        sa = final_state_6q(rho, ua)
        sb = final_state_6q(rho, ub)
        assert np.max(np.abs(sa - sb)) < TOL


def test_five_qubit_gate_route_equals_permutation_route():
    assert np.allclose(unitary_5q(), unitary_5q_permutation(), atol=TOL)


# ---------------------------------------------------------------------------
# 4. final states are valid (no postselection: the trace stays 1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("builder,dim", [(final_state_6q, 64), (final_state_5q, 32)])
def test_final_states_are_valid_density_matrices(builder, dim):
    for name, rho in _sample_states(6):
        sigma = builder(rho)
        assert sigma.shape == (dim, dim)
        assert abs(np.trace(sigma).real - 1.0) < TOL, name
        assert np.linalg.norm(sigma - sigma.conj().T) < TOL, name
        assert np.linalg.eigvalsh(0.5 * (sigma + sigma.conj().T)).min() > -TOL, name


def test_ancillas_start_in_plus():
    p = plus_state_dm()
    assert np.allclose(p, 0.5 * np.ones((2, 2)), atol=TOL)
    assert abs(np.trace(p).real - 1.0) < TOL


# ---------------------------------------------------------------------------
# 5. the blind question: Delta_6Q == Delta_5Q
# ---------------------------------------------------------------------------

def test_delta_6q_equals_delta_5q_on_all_paulis_and_many_states():
    worst = 0.0
    for name, rho in _sample_states(12):
        d6 = delta_6q_all_paulis(rho)
        d5 = delta_5q_all_paulis(rho)
        for lab in PAULI_LABELS:
            err = abs(d6[lab][0] - d5[lab])
            worst = max(worst, err)
            assert err < TOL, (name, lab, err)
    assert worst < TOL


def test_subtraction_does_real_work():
    """C_XX alone and C_YY alone must each fail to reproduce Delta_5Q somewhere."""
    rng = np.random.default_rng(4242)
    worst_xx = worst_yy = 0.0
    for _ in range(20):
        rho = _random_mixed(rng)
        sigma6, sigma5 = final_state_6q(rho), final_state_5q(rho)
        for lab in PAULI_LABELS:
            o = pauli_product(lab)
            _d, c_xx, c_yy = delta_6q(sigma6, o)
            ref = delta_5q(sigma5, o)
            worst_xx = max(worst_xx, abs(c_xx - ref))
            worst_yy = max(worst_yy, abs(c_yy - ref))
    assert worst_xx > 1e-3
    assert worst_yy > 1e-3


# ---------------------------------------------------------------------------
# 6. the reconstructed operator
# ---------------------------------------------------------------------------

def test_reconstructed_operators_agree_and_are_hermitian():
    for name, rho in _sample_states(6):
        t6 = reconstruct_operator({k: v[0] for k, v in
                                   delta_6q_all_paulis(rho).items()})
        t5 = reconstruct_operator(delta_5q_all_paulis(rho))
        assert np.linalg.norm(t6 - t6.conj().T) < TOL, name
        assert np.linalg.norm(t6 - t5) < TOL, name


def test_reconstruction_round_trips_the_pauli_coefficients():
    """T = (1/4) sum_P d_P P must give back d_P = Tr(P T)."""
    rng = np.random.default_rng(7)
    deltas = {lab: float(rng.normal()) for lab in PAULI_LABELS}
    t = reconstruct_operator(deltas)
    for lab in PAULI_LABELS:
        back = float(np.real(np.trace(pauli_product(lab) @ t)))
        assert abs(back - deltas[lab]) < TOL, lab


# ---------------------------------------------------------------------------
# 7. the Bell-fidelity example
# ---------------------------------------------------------------------------

def test_bell_fidelity_estimators_agree():
    for name, rho in _sample_states(6):
        d6 = {k: v[0] for k, v in delta_6q_all_paulis(rho).items()}
        d5 = delta_5q_all_paulis(rho)
        assert abs(bell_fidelity_numerator_from_deltas(d6)
                   - bell_fidelity_numerator_from_deltas(d5)) < TOL, name
        assert abs(normalised_bell_fidelity(d6)
                   - normalised_bell_fidelity(d5)) < TOL, name


def test_bell_fidelity_uses_only_locally_measurable_products():
    """Phi+ = (II + XX - YY + ZZ)/4 -- each term is a product of local Paulis."""
    phi_p = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    target = np.outer(phi_p, phi_p.conj())
    built = 0.25 * (pauli_product("II") + pauli_product("XX")
                    - pauli_product("YY") + pauli_product("ZZ"))
    assert np.allclose(built, target, atol=TOL)


def test_swap_on_is_an_involution():
    for n, (i, j) in ((6, (A1, A2)), (6, (B1, B2)), (5, (1, 3))):
        s = swap_on(i, j, n)
        assert np.allclose(s @ s, np.eye(2 ** n), atol=TOL)
