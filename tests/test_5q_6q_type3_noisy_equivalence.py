"""Raw noisy estimator comparison: 5Q Type-3 against 6Q-L16.

These tests exercise the question numerically; they assert equality only after
the residuals have been observed to sit at floating-point scale in
``scripts/verify_5q_6q_type3_noisy_equivalence.py``.  Nothing here uses the
analytic Type-3 recurrence, a closed-form noisy map, or any Bell-diagonal or
isotropy assumption.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

from pqec_distill import locc_vd_noisy as SIX
from pqec_distill import type3_5q_raw_noisy as FIVE
from pqec_distill.locc_vd import (
    PAULI_LABELS, delta_5q, delta_6q, final_state_5q, final_state_6q,
    pauli_product, reconstruct_operator,
)
from pqec_distill.noise import replacement_depolarizing

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "verify_5q_6q_type3_noisy_equivalence.py"
MODULE_PATH = ROOT / "src" / "pqec_distill" / "type3_5q_raw_noisy.py"

TOL = 1e-12
NC_MIN = 1e-06
Q_SMALL = (0.0, 1e-3, 0.05, 0.2, 0.6, 1.0)


def _load_script():
    spec = importlib.util.spec_from_file_location("_eq_script", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCRIPT = _load_script()


# ---------------------------------------------------------------------------
# state helpers (local, so the tests do not depend on the script's test sets)
# ---------------------------------------------------------------------------

def _dm(vec):
    v = np.asarray(vec, dtype=complex)
    return np.outer(v, v.conj()) / float(np.vdot(v, v).real)


def _random_mixed(rng, dim=4, rank=None):
    r = dim if rank is None else rank
    a = rng.normal(size=(dim, r)) + 1j * rng.normal(size=(dim, r))
    m = a @ a.conj().T
    return m / np.trace(m).real


@pytest.fixture(scope="module")
def generic_states():
    """Deliberately generic, non-Bell-diagonal, complex-coherence states."""
    rng = np.random.default_rng(4242)
    out = [_dm([1, 1, 0, 0]),                      # non-Bell-diagonal pure
           _dm([1, 0, 0, 1j]),                     # imaginary coherence
           _dm([np.cos(0.4), 0.3j, -0.6, 0.2]),    # generic complex pure
           _random_mixed(rng),                     # full rank
           _random_mixed(rng, rank=2)]             # rank deficient
    return out


def _delta5(rho, o, q, route="A"):
    return FIVE.delta_5q_type3_noisy(rho, o, q, route)


def _delta6(rho, o, q):
    s6 = SIX.final_state_6q_noisy(rho, q)
    return delta_6q(s6, o)[0]


# ===========================================================================
# A, B: CNOT counts and locality
# ===========================================================================

def test_a_five_qubit_type3_has_exactly_16_cnots():
    assert FIVE.N_CNOT_5Q == 16
    assert len(FIVE.cnot_wire_pairs()) == 16
    assert FIVE.WIRES_5Q_ORDER == ("c", "A1", "B1", "A2", "B2")


def test_b_six_qubit_l16_has_exactly_16_cnots_split_8_8():
    assert SIX.N_CNOT == 16
    assert SIX.N_CNOT_ALICE == 8
    assert SIX.N_CNOT_BOB == 8
    assert SIX.crosses_party_cut() == []


def test_five_qubit_sequence_is_the_canonical_type3_wire_list():
    """The 5Q gate list must be the frozen Type-3 one: Fredkin(0;1,3), (0;2,4)."""
    expected = (FIVE.fredkin_ops(0, 1, 3) + FIVE.fredkin_ops(0, 2, 4))
    assert [(k, tuple(a) if k == "CNOT" else a)
            for k, a in FIVE.TYPE3_5Q_OPS[1:]] == \
           [(k, tuple(a) if k == "CNOT" else a) for k, a in expected]


# ===========================================================================
# C: identical noise convention
# ===========================================================================

def test_c_same_replacement_depolarizing_convention():
    assert FIVE.replacement_depolarizing is replacement_depolarizing
    assert SIX.replacement_depolarizing is replacement_depolarizing
    assert FIVE.NOISE_CONVENTION == SIX.NOISE_CONVENTION
    # the channel locations coincide with the CNOT list in both programs
    assert [tuple(p) for k, p in FIVE.PROGRAM if k == "D"] == \
        FIVE.cnot_wire_pairs()
    assert [tuple(p) for k, p in SIX.PROGRAM if k == "D"] == \
        SIX.cnot_wire_pairs()
    # and the package channel equals an independently coded one on both sizes
    rng = np.random.default_rng(5)
    for n in (5, 6):
        d = 2 ** n
        m = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        sigma = m @ m.conj().T
        sigma = sigma / np.trace(sigma).real
        for q in (0.0, 0.17, 1.0):
            a = replacement_depolarizing(sigma, (0, n - 1), q, n_qubits=n)
            b = SCRIPT.ref_depol(sigma, (0, n - 1), q, n)
            assert np.max(np.abs(a - b)) < TOL


# ===========================================================================
# D, E: q = 0 must reproduce the frozen ideal estimators
# ===========================================================================

def test_d_q0_five_qubit_raw_estimator_matches_frozen_ideal(generic_states):
    worst = 0.0
    for rho in generic_states:
        s5 = final_state_5q(rho)
        for lab in PAULI_LABELS:
            o = pauli_product(lab)
            worst = max(worst, abs(_delta5(rho, o, 0.0) - delta_5q(s5, o)))
    assert worst < TOL


def test_e_q0_six_qubit_raw_estimator_matches_frozen_ideal(generic_states):
    worst = 0.0
    for rho in generic_states:
        s6_ideal = final_state_6q(rho)
        for lab in PAULI_LABELS:
            o = pauli_product(lab)
            worst = max(worst, abs(_delta6(rho, o, 0.0) - delta_6q(s6_ideal, o)[0]))
    assert worst < TOL


# ===========================================================================
# F: the two read-out conventions agree
# ===========================================================================

def test_f_route_a_x_readout_equals_route_b_z_readout(generic_states):
    worst = 0.0
    for rho in generic_states:
        for q in Q_SMALL:
            for lab in PAULI_LABELS:
                o = pauli_product(lab)
                worst = max(worst, abs(_delta5(rho, o, q, "A")
                                       - _delta5(rho, o, q, "B")))
    assert worst < TOL


# ===========================================================================
# G, H: the primary equivalence on Paulis and on random Hermitian observables
# ===========================================================================

def test_g_all_sixteen_pauli_estimators_agree(generic_states):
    worst = 0.0
    for rho in generic_states:
        data_r = np.kron(rho, rho)
        for q in Q_SMALL:
            s5 = FIVE.state_before_final_h(data_r, q)
            s6 = SIX.final_state_6q_noisy(rho, q)
            for lab in PAULI_LABELS:
                o = pauli_product(lab)
                worst = max(worst, abs(FIVE.delta_from_state_x(s5, o)
                                       - delta_6q(s6, o)[0]))
    assert worst < TOL


def test_h_random_hermitian_observables_agree(generic_states):
    rng = np.random.default_rng(909)
    obs = []
    for _ in range(12):
        m = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        h = 0.5 * (m + m.conj().T)
        obs.append(h / np.linalg.norm(h))
    # observables must genuinely carry imaginary off-diagonal weight
    assert max(float(np.max(np.abs(h.imag))) for h in obs) > 0.05
    worst = 0.0
    for rho in generic_states[:3]:
        data_r = np.kron(rho, rho)
        for q in (0.0, 0.05, 0.3):
            s5 = FIVE.state_before_final_h(data_r, q)
            s6 = SIX.final_state_6q_noisy(rho, q)
            for h in obs:
                worst = max(worst, abs(FIVE.delta_from_state_x(s5, h)
                                       - delta_6q(s6, h)[0]))
    assert worst < TOL


def test_identity_observable_raw_scale_agrees(generic_states):
    """Delta_5(I) vs Delta_6(I): rules out a hidden q-dependent prefactor."""
    i4 = np.eye(4, dtype=complex)
    worst = 0.0
    for rho in generic_states:
        for q in Q_SMALL:
            worst = max(worst, abs(_delta5(rho, i4, q) - _delta6(rho, i4, q)))
    assert worst < TOL


# ===========================================================================
# I, J: Pauli-tomographic consequence
# ===========================================================================

def test_i_reconstructed_operators_agree(generic_states):
    worst = 0.0
    for rho in generic_states:
        data_r = np.kron(rho, rho)
        for q in Q_SMALL:
            s5 = FIVE.state_before_final_h(data_r, q)
            s6 = SIX.final_state_6q_noisy(rho, q)
            d5 = {lab: FIVE.delta_from_state_x(s5, pauli_product(lab))
                  for lab in PAULI_LABELS}
            d6 = {lab: delta_6q(s6, pauli_product(lab))[0]
                  for lab in PAULI_LABELS}
            t5 = sum(d5[lab] * pauli_product(lab) for lab in PAULI_LABELS) / 4.0
            t6 = reconstruct_operator(d6)
            worst = max(worst, float(np.linalg.norm(t5 - t6)))
    assert worst < TOL


def test_j_normalised_operators_agree_where_denominator_is_safe(generic_states):
    worst = 0.0
    checked = 0
    for rho in generic_states:
        data_r = np.kron(rho, rho)
        for q in Q_SMALL:
            s5 = FIVE.state_before_final_h(data_r, q)
            s6 = SIX.final_state_6q_noisy(rho, q)
            d5 = {lab: FIVE.delta_from_state_x(s5, pauli_product(lab))
                  for lab in PAULI_LABELS}
            d6 = {lab: delta_6q(s6, pauli_product(lab))[0]
                  for lab in PAULI_LABELS}
            t5 = sum(d5[lab] * pauli_product(lab) for lab in PAULI_LABELS) / 4.0
            t6 = reconstruct_operator(d6)
            tr5 = float(np.real(np.trace(t5)))
            tr6 = float(np.real(np.trace(t6)))
            if min(abs(tr5), abs(tr6)) < 1e-09:
                continue
            checked += 1
            worst = max(worst, float(np.linalg.norm(t5 / tr5 - t6 / tr6)))
    assert checked > 0
    assert worst < TOL


# ===========================================================================
# K: arbitrary four-data-qubit input R
# ===========================================================================

def test_k_arbitrary_four_qubit_input_agrees():
    rng = np.random.default_rng(13131)
    r_states = [_random_mixed(rng, 16),
                _random_mixed(rng, 16, rank=3),
                np.kron(_random_mixed(rng), _random_mixed(rng, rank=1)),
                _dm(rng.normal(size=16) + 1j * rng.normal(size=16))]
    # at least one R must not be of the form rho (x) rho
    worst = 0.0
    for data_r in r_states:
        for q in (0.0, 1e-3, 0.1, 0.5, 1.0):
            s5 = FIVE.state_before_final_h(data_r, q)
            s6 = SCRIPT.main_state_6q(data_r, q)
            for lab in PAULI_LABELS:
                o = pauli_product(lab)
                worst = max(worst, abs(FIVE.delta_from_state_x(s5, o)
                                       - SCRIPT.delta_6_from_state(s6, o)[0]))
    assert worst < TOL


def test_general_r_runner_reduces_to_the_frozen_identical_copy_runner():
    rng = np.random.default_rng(777)
    rho = _random_mixed(rng)
    for q in (0.0, 0.08, 0.4):
        a = SCRIPT.main_state_6q(np.kron(rho, rho), q)
        b = SIX.final_state_6q_noisy(rho, q)
        assert np.max(np.abs(a - b)) < TOL


# ===========================================================================
# L: negative controls must be detectable
# ===========================================================================

def test_l_negative_controls_produce_clearly_nonzero_discrepancy(generic_states):
    rho = generic_states[3]
    data_r = np.kron(rho, rho)
    q = 0.1
    s5 = FIVE.state_before_final_h(data_r, q)
    o = pauli_product("ZZ")

    # control A: one 6Q CNOT noise strength changed from q to q + delta
    sched = [q] * SIX.N_CNOT
    sched[0] = q + 1e-3
    d_a = SCRIPT.ref_delta_6q(data_r, o, q, q_schedule=sched)
    # control B: one depolarizing channel removed
    d_b = SCRIPT.ref_delta_6q(data_r, o, q, skip_noise_after=(3,))
    # control D: C_XX alone
    s6 = SCRIPT.ref_state_6q(data_r, q)
    d_d = SCRIPT._tr(SCRIPT.ref_obs_6q(o, SCRIPT._X1), s6)

    d5 = FIVE.delta_from_state_x(s5, o)
    assert abs(d5 - d_a) > NC_MIN
    assert abs(d5 - d_b) > NC_MIN
    assert abs(d5 - d_d) > NC_MIN
    # and the unmodified circuit still agrees, so the metric is not broken
    assert abs(d5 - SCRIPT.ref_delta_6q(data_r, o, q)) < TOL


def test_negative_control_c_flipped_cnot_is_still_unitary_and_detected(generic_states):
    flipped = []
    done = False
    for kind, arg in SIX.L16_OPS:
        if kind == "CNOT" and not done and tuple(arg) == (SIX.A, SIX.A2):
            flipped.append((kind, (arg[1], arg[0])))
            done = True
        else:
            flipped.append((kind, arg))
    assert done
    u = SIX.circuit_unitary(flipped)
    assert np.max(np.abs(u.conj().T @ u - np.eye(64))) < TOL
    assert np.max(np.abs(u - SIX.circuit_unitary())) > 1e-03
    rho = generic_states[3]
    data_r = np.kron(rho, rho)
    q = 0.1
    d5 = FIVE.delta_5q_type3_noisy(rho, pauli_product("ZZ"), q)
    d6 = SCRIPT.ref_delta_6q(data_r, pauli_product("ZZ"), q, ops=flipped)
    assert abs(d5 - d6) > NC_MIN


# ===========================================================================
# independence of code paths
# ===========================================================================

def test_independent_reference_routes_agree_with_the_main_paths(generic_states):
    worst5 = worst6 = 0.0
    for rho in generic_states[:3]:
        data_r = np.kron(rho, rho)
        for q in (0.0, 0.07, 0.45):
            s5m = FIVE.state_before_final_h(data_r, q)
            s5r = SCRIPT.ref_state_5q(data_r, q)
            s6m = SIX.final_state_6q_noisy(rho, q)
            s6r = SCRIPT.ref_state_6q(data_r, q)
            for lab in PAULI_LABELS:
                o = pauli_product(lab)
                worst5 = max(worst5, abs(
                    FIVE.delta_from_state_x(s5m, o)
                    - SCRIPT._tr(SCRIPT.ref_obs_5q(o, SCRIPT._X1), s5r)))
                worst6 = max(worst6, abs(
                    delta_6q(s6m, o)[0]
                    - SCRIPT.ref_delta_6q(data_r, o, q)))
            assert np.max(np.abs(s6m - s6r)) < TOL
    assert worst5 < TOL
    assert worst6 < TOL


def test_local_coherence_mechanism_diagnostic(generic_states):
    """Mechanism check only: Omega_5 = <0|sigma_5|1> vs 2 Omega_6 = 2<00|sigma_6|11>."""
    worst = 0.0
    for rho in generic_states:
        data_r = np.kron(rho, rho)
        for q in (0.0, 0.05, 0.3, 1.0):
            om5 = FIVE.ancilla_01_block(FIVE.state_before_final_h(data_r, q))
            t = SIX.final_state_6q_noisy(rho, q).reshape([2] * 12)
            om6 = t[0, 0, :, :, :, :, 1, 1, :, :, :, :].reshape(16, 16)
            worst = max(worst, float(np.max(np.abs(om5 - 2.0 * om6))))
    assert worst < TOL


# ===========================================================================
# M: no analytic Type-3 recurrence anywhere in this comparison
# ===========================================================================

def test_m_no_analytic_recurrence_used():
    forbidden = ("analytic_exact_map", "analytic_map_type3", "analytic_map_type4",
                 "analytic_map_4q", "repeated_dynamics", "noisy_closed_form",
                 "bell_population_dynamics")
    for path in (MODULE_PATH, SCRIPT_PATH):
        text = path.read_text()
        code = "\n".join(l for l in text.splitlines()
                         if not re.match(r"\s*#", l))
        for name in forbidden:
            # the names may appear only inside the prose that says they are
            # NOT used; they must never appear in an import or a call
            assert not re.search(rf"import\s+.*{name}", code), (path.name, name)
            assert not re.search(rf"{name}\s*\(", code), (path.name, name)
    # and neither module exposes such a symbol in its namespace
    for mod in (FIVE, SIX):
        assert not [n for n in vars(mod) if "analytic" in n]
