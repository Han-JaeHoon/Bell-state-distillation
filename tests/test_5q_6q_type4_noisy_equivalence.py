"""Raw noisy estimator comparison: 5Q Type-4 against 6Q-L14.

The equality assertions here follow the residuals already observed by
``scripts/verify_5q_6q_type4_noisy_equivalence.py``; they are not assumptions
built into the experiment.  Nothing in the primary path uses the analytic
Type-4 recurrence, a closed-form noisy map, or any Bell-diagonal or isotropy
assumption.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

from pqec_distill import locc_vd_type4_noisy as SIX
from pqec_distill import type4_5q_raw_noisy as FIVE
from pqec_distill import type4_7cnot_fredkin as T4
from pqec_distill.locc_vd import (
    PAULI_LABELS, delta_6q, final_state_6q, pauli_product, reconstruct_operator,
)
from pqec_distill.noise import replacement_depolarizing

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "verify_5q_6q_type4_noisy_equivalence.py"
MODULE_PATHS = (
    ROOT / "src" / "pqec_distill" / "type4_5q_raw_noisy.py",
    ROOT / "src" / "pqec_distill" / "locc_vd_type4_noisy.py",
)

TOL = 1e-12
NC_MIN = 1e-06
Q_SMALL = (0.0, 1e-3, 0.05, 0.2, 0.6, 1.0)


def _load():
    spec = importlib.util.spec_from_file_location("_t4_eq_script", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCRIPT = _load()


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
    rng = np.random.default_rng(90210)
    return [_dm([1, 1, 0, 0]),
            _dm([1, 0, 0, 1j]),
            _dm([np.cos(0.4), 0.3j, -0.6, 0.2]),
            _random_mixed(rng),
            _random_mixed(rng, rank=2)]


def _delta6(data_r, o, q):
    s6 = SIX.final_state_6q_t4_noisy(data_r, q)
    return SCRIPT.delta_6_from_state(s6, o)[0]


# ===========================================================================
# F, G: 6Q-L14 structure
# ===========================================================================

def test_f_six_qubit_l14_has_exactly_14_cnots_split_7_7():
    assert SIX.N_CNOT == 14
    assert SIX.N_CNOT_ALICE == 7
    assert SIX.N_CNOT_BOB == 7
    assert SIX.WIRES_6Q == ("a", "b", "A1", "B1", "A2", "B2")
    assert SIX.ALICE_TRIPLE == (0, 2, 4)
    assert SIX.BOB_TRIPLE == (1, 3, 5)


def test_g_six_qubit_l14_has_no_cross_party_two_qubit_gate():
    assert SIX.crosses_party_cut() == []
    for kind, arg in SIX.L14_OPS:
        sup = T4.ops_support(kind, arg)
        assert sup <= SIX.ALICE_WIRES or sup <= SIX.BOB_WIRES


def test_six_qubit_local_blocks_are_fredkins_and_commute():
    for party in ("alice", "bob"):
        fit = T4.global_phase_fit(SIX.local_block_unitary(party),
                                  SIX.ideal_local_fredkin(party))
        assert fit["max_entry_residual"] < TOL
        assert abs(fit["global_phase_over_pi"] + 0.125) < 1e-09
    ua, ub = SIX.local_block_unitary("alice"), SIX.local_block_unitary("bob")
    assert np.max(np.abs(ua @ ub - ub @ ua)) < TOL
    fit = T4.global_phase_fit(SIX.circuit_unitary(),
                              SIX.ideal_local_fredkin("bob")
                              @ SIX.ideal_local_fredkin("alice"))
    assert fit["max_entry_residual"] < TOL
    assert abs(fit["global_phase_over_pi"] + 0.25) < 1e-09


def test_five_qubit_type4_has_exactly_14_cnots():
    assert FIVE.N_CNOT_5Q_T4 == 14
    assert len(T4.ops_cnot_pairs(FIVE.OPS_REGROUPED)) == 14
    assert FIVE.WIRES_5Q == ("c", "A1", "B1", "A2", "B2")


def test_same_replacement_depolarizing_convention():
    assert FIVE.replacement_depolarizing is replacement_depolarizing
    assert SIX.replacement_depolarizing is replacement_depolarizing
    assert FIVE.NOISE_CONVENTION == SIX.NOISE_CONVENTION
    for key, ops in (("regrouped", FIVE.OPS_REGROUPED),
                     ("original", FIVE.OPS_ORIGINAL)):
        locs = [tuple(p) for k, p in FIVE.PROGRAMS[key] if k == "D"]
        assert locs == T4.ops_cnot_pairs(ops)
    assert [tuple(p) for k, p in SIX.PROGRAM if k == "D"] == SIX.cnot_wire_pairs()
    rng = np.random.default_rng(31)
    for n in (5, 6):
        d = 2 ** n
        m = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        sigma = m @ m.conj().T
        sigma = sigma / np.trace(sigma).real
        for q in (0.0, 0.23, 1.0):
            a = replacement_depolarizing(sigma, (0, n - 1), q, n_qubits=n)
            b = SCRIPT.ref_depol(sigma, (0, n - 1), q, n)
            assert np.max(np.abs(a - b)) < TOL


# ===========================================================================
# H: q = 0 agrees with the frozen ideal 6Q estimator (and with Tr(O rho^2))
# ===========================================================================

def test_h_q0_six_qubit_l14_matches_frozen_ideal_estimator(generic_states):
    worst_ideal = worst_rho2 = 0.0
    for rho in generic_states:
        data_r = np.kron(rho, rho)
        s6_ideal = final_state_6q(rho)
        rho2 = rho @ rho
        for lab in PAULI_LABELS:
            o = pauli_product(lab)
            d6 = _delta6(data_r, o, 0.0)
            worst_ideal = max(worst_ideal, abs(d6 - delta_6q(s6_ideal, o)[0]))
            worst_rho2 = max(worst_rho2,
                             abs(d6 - float(np.real(np.trace(o @ rho2)))))
    assert worst_ideal < TOL
    assert worst_rho2 < TOL


# ===========================================================================
# I, J: read-out conventions and the original gate ordering
# ===========================================================================

def test_i_route_a_equals_route_b(generic_states):
    worst = 0.0
    for rho in generic_states:
        for q in Q_SMALL:
            for lab in PAULI_LABELS:
                o = pauli_product(lab)
                worst = max(worst, abs(
                    FIVE.delta_5q_type4_noisy(rho, o, q, "A")
                    - FIVE.delta_5q_type4_noisy(rho, o, q, "B")))
    assert worst < TOL


def test_j_regrouped_raw_signal_equals_original_frozen_ordering(generic_states):
    worst = 0.0
    for rho in generic_states:
        data_r = np.kron(rho, rho)
        for q in Q_SMALL:
            s5b = FIVE.state_after_final_h(data_r, q)
            s5o = FIVE.state_original(data_r, q)
            # the two orderings are genuinely different gate lists
            for lab in PAULI_LABELS:
                o = pauli_product(lab)
                worst = max(worst, abs(FIVE.delta_from_state_z(s5b, o)
                                       - FIVE.delta_from_state_z(s5o, o)))
    assert worst < TOL
    assert [(k, tuple(a)) for k, a in FIVE.OPS_ORIGINAL] != \
        [(k, tuple(a)) for k, a in FIVE.OPS_REGROUPED]


def test_original_ordering_rejects_route_a():
    with pytest.raises(ValueError):
        FIVE.delta_5q_type4_noisy(np.eye(4) / 4, np.eye(4), 0.1,
                                  route="A", ordering="original")


# ===========================================================================
# K, L: the primary equivalence
# ===========================================================================

def test_k_all_sixteen_pauli_estimators_agree(generic_states):
    worst = 0.0
    for rho in generic_states:
        data_r = np.kron(rho, rho)
        for q in Q_SMALL:
            s5 = FIVE.state_before_final_h(data_r, q)
            s6 = SIX.final_state_6q_t4_noisy(data_r, q)
            for lab in PAULI_LABELS:
                o = pauli_product(lab)
                worst = max(worst, abs(
                    FIVE.delta_from_state_x(s5, o)
                    - SCRIPT.delta_6_from_state(s6, o)[0]))
    assert worst < TOL


def test_l_random_hermitian_observables_agree(generic_states):
    rng = np.random.default_rng(606)
    obs = []
    for _ in range(12):
        m = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        h = 0.5 * (m + m.conj().T)
        obs.append(h / np.linalg.norm(h))
    assert max(float(np.max(np.abs(h.imag))) for h in obs) > 0.05
    worst = 0.0
    for rho in generic_states[:3]:
        data_r = np.kron(rho, rho)
        for q in (0.0, 0.05, 0.3):
            s5 = FIVE.state_before_final_h(data_r, q)
            s6 = SIX.final_state_6q_t4_noisy(data_r, q)
            for h in obs:
                worst = max(worst, abs(
                    FIVE.delta_from_state_x(s5, h)
                    - SCRIPT.delta_6_from_state(s6, h)[0]))
    assert worst < TOL


def test_identity_observable_raw_scale_agrees(generic_states):
    """Delta_5(I) vs Delta_6(I): rules out a hidden q-dependent prefactor."""
    i4 = np.eye(4, dtype=complex)
    worst = 0.0
    for rho in generic_states:
        data_r = np.kron(rho, rho)
        for q in Q_SMALL:
            worst = max(worst, abs(
                FIVE.delta_5q_type4_noisy(rho, i4, q)
                - _delta6(data_r, i4, q)))
    assert worst < TOL


# ===========================================================================
# M: Pauli-tomographic reconstruction
# ===========================================================================

def test_m_reconstructed_operators_agree(generic_states):
    worst_t = worst_r = 0.0
    checked = 0
    for rho in generic_states:
        data_r = np.kron(rho, rho)
        for q in Q_SMALL:
            s5 = FIVE.state_before_final_h(data_r, q)
            s6 = SIX.final_state_6q_t4_noisy(data_r, q)
            d5 = {lab: FIVE.delta_from_state_x(s5, pauli_product(lab))
                  for lab in PAULI_LABELS}
            d6 = {lab: SCRIPT.delta_6_from_state(s6, pauli_product(lab))[0]
                  for lab in PAULI_LABELS}
            t5 = sum(d5[lab] * pauli_product(lab) for lab in PAULI_LABELS) / 4.0
            t6 = reconstruct_operator(d6)
            worst_t = max(worst_t, float(np.linalg.norm(t5 - t6)))
            tr5 = float(np.real(np.trace(t5)))
            tr6 = float(np.real(np.trace(t6)))
            if min(abs(tr5), abs(tr6)) < 1e-09:
                continue
            checked += 1
            worst_r = max(worst_r, float(np.linalg.norm(t5 / tr5 - t6 / tr6)))
    assert worst_t < TOL
    assert checked > 0
    assert worst_r < TOL


# ===========================================================================
# N: arbitrary four-qubit input R
# ===========================================================================

def test_n_arbitrary_four_qubit_input_agrees():
    rng = np.random.default_rng(24680)
    r_states = [_random_mixed(rng, 16),
                _random_mixed(rng, 16, rank=3),
                np.kron(_random_mixed(rng), _random_mixed(rng, rank=1)),
                _dm(rng.normal(size=16) + 1j * rng.normal(size=16))]
    worst = 0.0
    for data_r in r_states:
        for q in (0.0, 1e-3, 0.1, 0.5, 1.0):
            s5 = FIVE.state_before_final_h(data_r, q)
            s6 = SIX.final_state_6q_t4_noisy(data_r, q)
            for lab in PAULI_LABELS:
                o = pauli_product(lab)
                worst = max(worst, abs(
                    FIVE.delta_from_state_x(s5, o)
                    - SCRIPT.delta_6_from_state(s6, o)[0]))
    assert worst < TOL


# ===========================================================================
# O: negative controls
# ===========================================================================

def test_o_negative_controls_produce_clearly_nonzero_discrepancy(generic_states):
    rho = generic_states[3]
    data_r = np.kron(rho, rho)
    q = 0.1
    o = pauli_product("ZZ")
    d5 = FIVE.delta_5q_type4_noisy(rho, o, q)

    sched = [q] * SIX.N_CNOT
    sched[0] = q + 1e-3
    d_a = SCRIPT.ref_delta_6q(data_r, o, q, q_schedule=sched)
    d_b = SCRIPT.ref_delta_6q(data_r, o, q, skip_noise_after=(3,))
    s6 = SCRIPT.ref_state_6q(data_r, q)
    d_d = SCRIPT._tr(SCRIPT.ref_obs_6q(o, SCRIPT._X1), s6)

    assert abs(d5 - d_a) > NC_MIN
    assert abs(d5 - d_b) > NC_MIN
    assert abs(d5 - d_d) > NC_MIN
    # the unperturbed comparison still agrees, so the metric is not broken
    assert abs(d5 - SCRIPT.ref_delta_6q(data_r, o, q)) < TOL


def test_negative_control_flipped_cnot_is_unitary_and_detected(generic_states):
    flipped, done = [], False
    for kind, arg in SIX.L14_OPS:
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
    d5 = FIVE.delta_5q_type4_noisy(rho, pauli_product("ZZ"), 0.1)
    d6 = SCRIPT.ref_delta_6q(data_r, pauli_product("ZZ"), 0.1, ops=flipped)
    assert abs(d5 - d6) > NC_MIN


# ===========================================================================
# independent routes and the mechanism diagnostic
# ===========================================================================

def test_independent_reference_routes_agree_with_the_main_paths(generic_states):
    worst5 = worst6 = 0.0
    for rho in generic_states[:3]:
        data_r = np.kron(rho, rho)
        for q in (0.0, 0.07, 0.45):
            s5m = FIVE.state_before_final_h(data_r, q)
            s5r = SCRIPT.ref_state_5q(data_r, q)
            s6m = SIX.final_state_6q_t4_noisy(data_r, q)
            s6r = SCRIPT.ref_state_6q(data_r, q)
            assert np.max(np.abs(s6m - s6r)) < TOL
            for lab in PAULI_LABELS:
                o = pauli_product(lab)
                worst5 = max(worst5, abs(
                    FIVE.delta_from_state_x(s5m, o)
                    - SCRIPT._tr(SCRIPT.ref_obs_5q(o, SCRIPT._X1), s5r)))
                worst6 = max(worst6, abs(
                    SCRIPT.delta_6_from_state(s6m, o)[0]
                    - SCRIPT.ref_delta_6q(data_r, o, q)))
    assert worst5 < TOL
    assert worst6 < TOL


def test_local_coherence_mechanism_diagnostic(generic_states):
    """Mechanism check only: Omega_5 = <0|sigma_5|1>_c vs 2 <00|sigma_6|11>."""
    worst = 0.0
    for rho in generic_states:
        data_r = np.kron(rho, rho)
        for q in (0.0, 0.05, 0.3, 1.0):
            om5 = FIVE.ancilla_01_block(FIVE.state_before_final_h(data_r, q))
            om6 = SIX.ancilla_0011_block(
                SIX.final_state_6q_t4_noisy(data_r, q))
            worst = max(worst, float(np.max(np.abs(om5 - 2.0 * om6))))
    assert worst < TOL


# ===========================================================================
# P: no Type-4 analytic recurrence in the primary path
# ===========================================================================

def test_p_no_analytic_recurrence_in_the_primary_path():
    forbidden = ("analytic_exact_map", "analytic_map_type3", "analytic_map_type4",
                 "analytic_map_4q", "repeated_dynamics", "noisy_closed_form",
                 "bell_population_dynamics")
    for path in MODULE_PATHS:
        code = "\n".join(l for l in path.read_text().splitlines()
                         if not re.match(r"\s*#", l))
        for name in forbidden:
            assert not re.search(rf"import\s+.*{name}", code), (path.name, name)
            assert not re.search(rf"{name}\s*\(", code), (path.name, name)
    for mod in (FIVE, SIX, T4):
        assert not [n for n in vars(mod) if "analytic" in n]
    # in the comparison script the only mention is inside the clearly
    # labelled SECONDARY function, which the primary path never calls
    src = SCRIPT_PATH.read_text()
    idx = src.index("def secondary_analytic_sanity_check")
    primary = src[:idx]
    for name in forbidden:
        assert not re.search(rf"import\s+.*{name}", primary), name
        assert not re.search(rf"{name}\s*\(", primary), name


def test_secondary_analytic_check_is_separate_and_consistent():
    """Clearly labelled secondary cross-check; informational only."""
    res = SCRIPT.secondary_analytic_sanity_check()
    if not res.get("available"):
        pytest.skip(f"analytic map unavailable: {res.get('reason')}")
    assert res["max_abs_diff"] < 1e-10
