"""Regression tests for the analytic cross-check of the 6Q LOCC estimator.

These test the analytic side and the steps of its derivation.  The frozen
blind circuit implementation and its result files are only read, never
rewritten.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from pqec_distill.locc_vd import (
    PAULI_LABELS, delta_5q, delta_6q, final_state_5q, final_state_6q,
    pauli_product, plus_state_dm, reconstruct_operator, s_a_6q, s_b_6q,
    unitary_5q, unitary_6q_route_a,
)

TOL = 1e-12
FROZEN = Path(__file__).resolve().parents[1] / "results" / "data" / "locc_vd"

PHI_P = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
PHI_M = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
PSI_P = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
PSI_M = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
BELL_VECS = (PHI_P, PHI_M, PSI_P, PSI_M)

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)


def _random_mixed(rng, rank=4):
    m = rng.normal(size=(4, rank)) + 1j * rng.normal(size=(4, rank))
    rho = m @ m.conj().T
    return rho / np.trace(rho)


def _copy_swap(dim=4):
    eye = np.eye(dim, dtype=complex)
    s = np.zeros((dim * dim, dim * dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            s += np.kron(np.outer(eye[i], eye[j]), np.outer(eye[j], eye[i]))
    return s


def _partial_trace_second(m, dim=4):
    return np.einsum("ikjk->ij", m.reshape(dim, dim, dim, dim))


# ---------------------------------------------------------------------------
# the algebraic steps of the derivation
# ---------------------------------------------------------------------------

def test_xx_minus_yy_is_twice_the_outer_coherence():
    d = np.kron(_X, _X) - np.kron(_Y, _Y)
    e = np.eye(4, dtype=complex)
    want = 2 * (np.outer(e[0], e[3]) + np.outer(e[3], e[0]))
    assert np.allclose(d, want, atol=TOL)


def test_yy_flips_only_the_outer_branch_pair():
    """<y|XX|x> is +1 on all four complement pairs; <y|YY|x> is -1 on the
    00<->11 pair and +1 on the 01<->10 pair."""
    xx, yy = np.kron(_X, _X), np.kron(_Y, _Y)
    for x, y, want_yy in ((0, 3, -1.0), (3, 0, -1.0), (1, 2, +1.0), (2, 1, +1.0)):
        assert abs(xx[y, x] - 1.0) < TOL
        assert abs(yy[y, x] - want_yy) < TOL


def test_copy_swap_equals_S_A_S_B_on_the_data_register():
    """S_A S_B on wires (A1,B1,A2,B2) is the SWAP of the two 4-dim copies."""
    big = s_a_6q() @ s_b_6q()           # 64x64, wires [a,b,A1,B1,A2,B2]
    data = big.reshape([2] * 12)
    # ancilla wires are untouched: pick the |00><00| ancilla block
    block = data[0, 0, :, :, :, :, 0, 0, :, :, :, :].reshape(16, 16)
    assert np.allclose(block, _copy_swap(), atol=TOL)


def test_swap_identity_partial_trace():
    rng = np.random.default_rng(3)
    s = _copy_swap()
    for _ in range(20):
        a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        b = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        ab = np.kron(a, b)
        assert np.allclose(_partial_trace_second(ab @ s), a @ b, atol=TOL)
        assert np.allclose(_partial_trace_second(s @ ab), b @ a, atol=TOL)


def test_identical_copies_commute_with_the_copy_swap():
    rng = np.random.default_rng(5)
    s = _copy_swap()
    for _ in range(10):
        rho = _random_mixed(rng)
        r = np.kron(rho, rho)
        assert np.allclose(s @ r @ s, r, atol=TOL)
        assert np.allclose(s @ r, r @ s, atol=TOL)


def test_pauli_orthogonality():
    for lp in PAULI_LABELS:
        for lq in PAULI_LABELS:
            val = complex(np.trace(pauli_product(lp) @ pauli_product(lq)))
            assert abs(val - (4.0 if lp == lq else 0.0)) < TOL


# ---------------------------------------------------------------------------
# the analytic claims against the circuits
# ---------------------------------------------------------------------------

def test_delta_equals_trace_O_rho_squared():
    rng = np.random.default_rng(31415)
    for _ in range(8):
        rho = _random_mixed(rng, rank=rng.integers(1, 5))
        sigma6, sigma5 = final_state_6q(rho), final_state_5q(rho)
        rho2 = rho @ rho
        for lab in PAULI_LABELS:
            o = pauli_product(lab)
            want = float(np.real(np.trace(o @ rho2)))
            assert abs(delta_6q(sigma6, o)[0] - want) < TOL, lab
            assert abs(delta_5q(sigma5, o) - want) < TOL, lab


def test_reconstructed_operator_equals_rho_squared():
    rng = np.random.default_rng(2718)
    for _ in range(8):
        rho = _random_mixed(rng, rank=rng.integers(1, 5))
        sigma6 = final_state_6q(rho)
        deltas = {lab: delta_6q(sigma6, pauli_product(lab))[0]
                  for lab in PAULI_LABELS}
        assert np.linalg.norm(reconstruct_operator(deltas) - rho @ rho) < TOL


def test_denominator_is_the_purity():
    rng = np.random.default_rng(99)
    for _ in range(8):
        rho = _random_mixed(rng)
        d = delta_6q(final_state_6q(rho), np.eye(4, dtype=complex))[0]
        assert abs(d - float(np.real(np.trace(rho @ rho)))) < TOL


def test_bell_diagonal_virtual_populations():
    rng = np.random.default_rng(7)
    for _ in range(8):
        p = rng.dirichlet(np.ones(4))
        rho = sum(w * np.outer(v, v.conj()) for w, v in zip(p, BELL_VECS))
        want = p ** 2 / np.sum(p ** 2)
        rho2 = rho @ rho
        got = np.array([float(np.real(v.conj() @ rho2 @ v)) for v in BELL_VECS])
        assert np.allclose(got / got.sum(), want, atol=TOL)


@pytest.mark.parametrize("eps", [0.0, 0.01, 0.1, 0.3, 2.0 / 3.0, 0.8, 1.0])
def test_bell_isotropic_fidelity_formula(eps):
    rho = (1 - eps) * np.outer(PHI_P, PHI_P.conj()) + eps * np.eye(4) / 4
    sigma6 = final_state_6q(rho)
    d = {lab: delta_6q(sigma6, pauli_product(lab))[0] for lab in PAULI_LABELS}
    circuit = 0.25 * (d["II"] + d["XX"] - d["YY"] + d["ZZ"]) / d["II"]
    p0, p1 = 1 - 3 * eps / 4, eps / 4
    formula = p0 ** 2 / (p0 ** 2 + 3 * p1 ** 2)
    assert abs(circuit - formula) < TOL


def test_nonidentical_inputs_give_the_symmetrised_product():
    """Section 12: rho (x) sigma -> (1/2) Tr[O (rho sigma + sigma rho)]."""
    rng = np.random.default_rng(909)
    u6, u5, plus = unitary_6q_route_a(), unitary_5q(), plus_state_dm()
    gap = 0.0
    for _ in range(6):
        rho, sigma = _random_mixed(rng), _random_mixed(rng, rank=3)
        data = np.kron(rho, sigma)
        s6 = u6 @ np.kron(plus, np.kron(plus, data)) @ u6.conj().T
        s5 = u5 @ np.kron(plus, data) @ u5.conj().T
        for lab in PAULI_LABELS:
            o = pauli_product(lab)
            want = 0.5 * float(np.real(np.trace(o @ (rho @ sigma + sigma @ rho))))
            assert abs(delta_6q(s6, o)[0] - want) < TOL, lab
            assert abs(delta_5q(s5, o) - want) < TOL, lab
            gap = max(gap, abs(delta_6q(s6, o)[0]
                               - float(np.real(np.trace(o @ rho @ rho)))))
    # the identical-copy formula must NOT hold here, or the test is vacuous
    assert gap > 1e-3


# ---------------------------------------------------------------------------
# the frozen data must stay what it was
# ---------------------------------------------------------------------------

def test_frozen_blind_files_are_present_and_unchanged_in_shape():
    rows = list(csv.DictReader((FROZEN / "blind_verification.csv").open()))
    assert len(rows) == 2416
    assert set(rows[0]) >= {"state", "observable", "Delta_6Q", "Delta_5Q"}
    npz = np.load(FROZEN / "representative_operators.npz", allow_pickle=False)
    assert sum(1 for k in npz.files if k.startswith("T_6Q__")) == 19


def test_frozen_deltas_match_the_analytic_formula():
    """The headline cross-check, as a test rather than only a report."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from verify_locc_vd import build_states          # noqa: E402

    by_name = {n: r for n, _f, r in build_states()}
    rows = list(csv.DictReader((FROZEN / "blind_verification.csv").open()))
    worst = 0.0
    for r in rows:
        rho = by_name[r["state"]]
        want = float(np.real(np.trace(pauli_product(r["observable"]) @ rho @ rho)))
        worst = max(worst, abs(float(r["Delta_6Q"]) - want))
    assert worst < 1e-13
