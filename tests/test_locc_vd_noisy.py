"""Tests for the noisy 6Q-L16 circuit.

These check the structure of the decomposed circuit, that the q = 0 limit
reproduces the frozen ideal 6Q implementation, that the noise channel is the
repository's existing replacement-depolarizing convention, and that the
Bell-diagonal / isotropy / fidelity diagnostics are computed from the raw
reconstruction rather than imposed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from pqec_distill.locc_vd import (
    PAULI_LABELS, delta_6q_all_paulis, pauli_product, reconstruct_operator,
)
from pqec_distill.locc_vd_noisy import (
    ALICE_TRIPLE, ALICE_WIRES, BOB_TRIPLE, BOB_WIRES, DIM, L16_OPS, N_CNOT,
    N_CNOT_ALICE, N_CNOT_BOB, N_WIRES, PROGRAM, bell_diagonal_rho,
    bell_matrix, bell_offdiagonal_frobenius, bell_populations,
    circuit_unitary, cnot_wire_pairs, crosses_party_cut,
    deltas_all_paulis_noisy, effective_operator, final_state_6q_noisy,
    fredkin_ops, ideal_local_fredkin, isotropic_rho, local_block_unitary,
    phi_plus_fidelity,
)
from pqec_distill.noise import replacement_depolarizing

TOL = 1e-12
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _rand_rho(rng, dim=4):
    m = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    r = m @ m.conj().T
    return r / np.trace(r)


# --- A, B: each local block is the ideal Fredkin ---------------------------

@pytest.mark.parametrize("party", ["alice", "bob"])
def test_local_block_equals_ideal_fredkin(party):
    assert np.allclose(local_block_unitary(party), ideal_local_fredkin(party),
                       atol=TOL)


# --- C: CNOT counts --------------------------------------------------------

def test_cnot_counts_are_eight_eight_sixteen():
    assert N_CNOT_ALICE == 8
    assert N_CNOT_BOB == 8
    assert N_CNOT == 16
    assert len(cnot_wire_pairs()) == 16


def test_fredkin_decomposition_matches_the_canonical_type3_sequence():
    """The same logical sequence as scripts/verify_type3_qabs_circuit.py."""
    sys.path.insert(0, str(SCRIPTS))
    import verify_type3_qabs_circuit as t3          # noqa: E402

    # the Type-3 script's wires are (a, A1, A2, B1, B2) = (0, 1, 2, 3, 4) and
    # its Fredkins are (0;1,3) and (0;2,4); ours are (0;2,4) and (1;3,5).
    assert t3.fredkin_ops(0, 1, 3) == [
        (k, a) for k, a in fredkin_ops(0, 1, 3)]
    assert sum(1 for k, _ in t3.fredkin_ops(0, 1, 3) if k == "CNOT") == 8


# --- D: no gate crosses the Alice/Bob cut ----------------------------------

def test_no_two_qubit_gate_crosses_the_party_cut():
    assert crosses_party_cut() == []
    for pair in cnot_wire_pairs():
        assert (set(pair) <= set(ALICE_WIRES)) or (set(pair) <= set(BOB_WIRES))


def test_every_gate_touches_one_party_only():
    for kind, arg in L16_OPS:
        wires = set(arg) if kind == "CNOT" else {arg}
        assert wires <= set(ALICE_WIRES) or wires <= set(BOB_WIRES)


def test_local_blocks_commute():
    ua, ub = local_block_unitary("alice"), local_block_unitary("bob")
    assert np.allclose(ua @ ub, ub @ ua, atol=TOL)


# --- E: q = 0 matches the frozen ideal 6Q ----------------------------------

def test_q0_matches_frozen_ideal_implementation():
    rng = np.random.default_rng(12345)
    states = [bell_diagonal_rho([1, 0, 0, 0]), isotropic_rho(0.3),
              bell_diagonal_rho(rng.dirichlet(np.ones(4))), _rand_rho(rng)]
    for rho in states:
        noisy0 = deltas_all_paulis_noisy(rho, 0.0)
        ideal = delta_6q_all_paulis(rho)
        for lab in PAULI_LABELS:
            assert abs(noisy0[lab][0] - ideal[lab][0]) < TOL, lab


# --- F, G: the noise channel is the repository's existing convention --------

def test_noise_channel_is_the_repository_replacement_convention():
    """Equal to the parent repository's eps2 Kraus channel on a wire pair."""
    rng = np.random.default_rng(99)
    paulis = [np.eye(2, dtype=complex),
              np.array([[0, 1], [1, 0]], dtype=complex),
              np.array([[0, -1j], [1j, 0]], dtype=complex),
              np.array([[1, 0], [0, -1]], dtype=complex)]
    sigma = _rand_rho(rng, 16)                    # 4 qubits is enough
    n = 4
    for pair in ((0, 2), (1, 3)):
        for q in (0.1, 0.4):
            acc = (1 - q) * sigma
            for p1 in paulis:
                for p2 in paulis:
                    factors = [np.eye(2, dtype=complex)] * n
                    factors[pair[0]], factors[pair[1]] = p1, p2
                    e = factors[0]
                    for f in factors[1:]:
                        e = np.kron(e, f)
                    acc = acc + (q / 16.0) * (e @ sigma @ e.conj().T)
            ours = replacement_depolarizing(sigma, pair, q, n_qubits=n)
            assert np.allclose(acc, ours, atol=TOL)


def test_every_noisy_cnot_channel_is_trace_preserving():
    rng = np.random.default_rng(7)
    sigma = _rand_rho(rng, DIM)
    for pair in cnot_wire_pairs():
        for q in (0.0, 0.05, 0.5, 1.0):
            out = replacement_depolarizing(sigma, pair, q, n_qubits=N_WIRES)
            assert abs(np.trace(out) - 1.0) < TOL
            assert np.linalg.norm(out - out.conj().T) < TOL


# --- H: the final six-qubit state is a valid density matrix ----------------

@pytest.mark.parametrize("q", [0.0, 0.02, 0.1, 0.3])
def test_final_state_is_a_valid_density_matrix(q):
    rho = isotropic_rho(0.3)
    sigma = final_state_6q_noisy(rho, q)
    assert abs(float(np.real(np.trace(sigma))) - 1.0) < TOL
    assert np.linalg.norm(sigma - sigma.conj().T) < TOL
    assert np.linalg.eigvalsh(0.5 * (sigma + sigma.conj().T)).min() > -TOL


# --- I: the reconstructed operator is Hermitian ----------------------------

@pytest.mark.parametrize("q", [0.0, 0.05, 0.2])
def test_reconstructed_operator_is_hermitian(q):
    rng = np.random.default_rng(3)
    for p in ([1, 0, 0, 0], [0.6, 0.2, 0.15, 0.05],
              list(rng.dirichlet(np.ones(4)))):
        info = effective_operator(bell_diagonal_rho(p), q)
        assert info["T_hermiticity"] < TOL
        assert info["D_equals_d_II"] < TOL


# --- J: Bell-diagonality is measured, not imposed --------------------------

def test_bell_offdiagonal_metric_detects_a_planted_off_diagonal_term():
    """The metric must be able to see off-diagonal weight; the near-zero
    result for the circuit is therefore a measurement, not a definition."""
    base = bell_diagonal_rho([0.6, 0.2, 0.1, 0.1])
    assert bell_offdiagonal_frobenius(base) < TOL
    from pqec_distill.locc_vd_noisy import BELL
    u, v = BELL["Phi+"], BELL["Phi-"]
    seeded = base + 0.01 * (np.outer(u, v.conj()) + np.outer(v, u.conj()))
    assert bell_offdiagonal_frobenius(seeded) > 1e-3


def test_noisy_output_keeps_bell_diagonal_inputs_bell_diagonal_numerically():
    rng = np.random.default_rng(21)
    worst = 0.0
    for _ in range(4):
        rho = bell_diagonal_rho(rng.dirichlet(np.ones(4)))
        for q in (0.0, 0.02, 0.2):
            info = effective_operator(rho, q)
            worst = max(worst, bell_offdiagonal_frobenius(info["R"]))
    assert worst < 1e-12          # numerically observed, not imposed


# --- K: the isotropy metric comes from the raw populations -----------------

def test_isotropy_metric_uses_raw_reconstructed_populations():
    info = effective_operator(isotropic_rho(0.3), 0.05)
    p = bell_populations(info["R"])
    # populations read back from the Bell-basis matrix of R, not stored
    diag = np.real(np.diag(bell_matrix(info["R"])))
    assert np.allclose(p, diag, atol=TOL)
    assert abs(p.sum() - 1.0) < TOL
    # and at q > 0 isotropy is genuinely broken, so the metric is not trivially 0
    assert max(abs(p[1] - p[2]), abs(p[1] - p[3])) > 1e-4


def test_psi_symmetry_is_measured_not_imposed():
    """p_Psi+ and p_Psi- are computed independently; their equality is a
    measurement.  Perturbing the input breaks it, which proves the code is not
    forcing it."""
    info = effective_operator(isotropic_rho(0.3), 0.05)
    p = bell_populations(info["R"])
    assert abs(p[2] - p[3]) < TOL
    asym = effective_operator(bell_diagonal_rho([0.7, 0.1, 0.15, 0.05]), 0.05)
    pa = bell_populations(asym["R"])
    assert abs(pa[2] - pa[3]) > 1e-3


# --- L: F_out comes from the reconstructed operator ------------------------

def test_fidelity_out_comes_from_the_reconstructed_operator():
    rho = isotropic_rho(0.2)
    info = effective_operator(rho, 0.03)
    direct = phi_plus_fidelity(info["R"])
    d = info["deltas"]
    from_paulis = 0.25 * (d["II"] + d["XX"] - d["YY"] + d["ZZ"]) / d["II"]
    assert abs(direct - from_paulis) < TOL


def test_one_round_fidelity_gain_exists_and_vanishes_at_high_noise():
    rho = isotropic_rho(0.3)
    f_in = 1.0 - 3.0 * 0.3 / 4.0
    assert phi_plus_fidelity(effective_operator(rho, 0.0)["R"]) > f_in
    assert phi_plus_fidelity(effective_operator(rho, 0.3)["R"]) < f_in


def test_program_has_one_channel_per_cnot_and_none_elsewhere():
    assert sum(1 for kind, _ in PROGRAM if kind == "D") == N_CNOT
    assert [p for k, p in PROGRAM if k == "D"] == cnot_wire_pairs()
