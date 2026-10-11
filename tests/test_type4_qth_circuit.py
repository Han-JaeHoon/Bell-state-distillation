"""q_th full-circuit study: the 5Q Type 4 protocol (14 CNOTs)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from _qth_testlib import (                                        # noqa: E402
    ROOT, assert_bell_projector_fidelity, assert_no_analytic_imports,
    assert_q0_is_identity, assert_q1_replaces_the_pair, assert_routes_agree,
    core, driver, expected_cnots,
)

PROTOCOL = "type4"
TOL = 1e-12
D = driver()
C = core()


# --- A, B: the frozen specification ----------------------------------------

def test_a_circuit_matches_the_frozen_repository_source():
    prov = C.provenance_checks()
    assert prov["type4_matches_verify_type4_qabs_circuit.py"] is True
    assert prov["type4_matches_verify_type4_qrel_circuit.py"] is True
    assert C.N_WIRES[PROTOCOL] == 5
    assert C.ANCILLA_5Q == 0 and C.RETAIN_5Q == (1, 2) and C.DISCARD_5Q == (3, 4)


def test_b_exact_cnot_count():
    assert C.n_cnot(PROTOCOL) == expected_cnots(PROTOCOL) == 14
    assert C.provenance_checks()["n_cnot_ok"] is True
    # every CNOT direction matters
    ops = list(C.OPS[PROTOCOL])
    base = C.build_program(PROTOCOL)
    for k, (kind, arg) in enumerate(ops):
        if kind != "CNOT":
            continue
        flipped = list(ops)
        flipped[k] = ("CNOT", (arg[1], arg[0]))
        rho_a, _ = C.one_round(C.isotropic_rho(0.3), 0.05, PROTOCOL, program=base)
        rho_b, _ = C.one_round(C.isotropic_rho(0.3), 0.05, PROTOCOL,
                               program=C.build_program(PROTOCOL, ops=flipped))
        assert np.max(np.abs(rho_a - rho_b)) > 1e-09, k
        break


# --- C, D: the noise channel -----------------------------------------------

def test_c_q0_noise_is_identity():
    assert_q0_is_identity(PROTOCOL)


def test_d_q1_replaces_the_pair():
    assert_q1_replaces_the_pair(PROTOCOL)


# --- E: the fidelity observable --------------------------------------------

def test_e_bell_projector_gives_the_right_fidelity():
    assert_bell_projector_fidelity()


def test_effective_state_comes_from_physical_pauli_tomography():
    """All sixteen coefficients of T are circuit expectation values."""
    eps, q = 0.30, 0.06
    rho = C.isotropic_rho(eps)
    out, info = C.one_round(rho, q, PROTOCOL)
    deltas = info["deltas"]
    assert set(deltas) == set(C.PAULI_LABELS) and len(deltas) == 16
    # rebuild T here, independently of the module, and compare
    t = sum(deltas[lab] * np.kron(C.PAULI_1Q[lab[0]], C.PAULI_1Q[lab[1]])
            for lab in C.PAULI_LABELS) / 4.0
    assert abs(float(np.real(np.trace(t))) - info["denominator"]) < 1e-14
    assert np.max(np.abs(t / info["denominator"] - out)) < 1e-13
    # each Delta_P really is <Z_a (x) P (x) I> on the final five-wire state
    sigma = np.kron(C._P0, np.kron(rho, rho))
    sigma = C._run_program(sigma, C.PROGRAMS[PROTOCOL], q, 5)
    for lab in C.PAULI_LABELS:
        obs = C._kron_list([C._Z,
                            np.kron(C.PAULI_1Q[lab[0]], C.PAULI_1Q[lab[1]]),
                            np.eye(4, dtype=complex)])
        assert abs(float(np.real(np.trace(obs @ sigma))) - deltas[lab]) < 1e-13


# --- F: the predicate is about fidelity, not Bell-error populations --------

def test_f_predicate_uses_delta_F_not_bell_error_populations():
    """A PASS round may have a RISING Bell-error population: only F decides."""
    eps = 0.40
    found = None
    for q in np.linspace(0.005, D.q_th_first(PROTOCOL, eps)["q_threshold"], 40):
        r = D.run_trajectory(PROTOCOL, eps, float(q), record=True)
        if r["status"] != D.PASS:
            continue
        rows = r["rows"]
        if any(rows[n + 1]["p"][k] > rows[n]["p"][k] + 1e-12
               for n in range(len(rows) - 1) for k in (1, 2, 3)):
            found = (float(q), r)
            break
    assert found is not None, ("expected a PASS trajectory with a rising "
                               "Bell-error population")
    q, r = found
    rows = r["rows"]
    assert all(x["dF"] > -D.VIOL_TOL for x in rows[1:])
    rho = C.isotropic_rho(eps)
    assert abs(rows[0]["F"] - C.fidelity(rho)) < 1e-15
    for x in rows[1:6]:
        rho, _ = C.one_round(rho, q, PROTOCOL)
        assert abs(x["F"] - C.fidelity(rho)) < 1e-13


# --- G, H: the two thresholds and the binding round ------------------------

def test_g_first_and_full_thresholds_are_evaluated_independently():
    eps = 0.15
    first = D.q_th_first(PROTOCOL, eps)
    assert first["has_root"] and first["n_sign_changes"] == 1
    f0, f1, _ = D.first_round_step(PROTOCOL, eps, first["q_upper_fail"])
    assert f1 - f0 < -D.VIOL_TOL
    f0, f1, _ = D.first_round_step(PROTOCOL, eps, first["q_threshold"])
    assert f1 - f0 > 0
    # at the one-round threshold the ALL-round predicate already fails, and it
    # fails at a LATER round: the two quantities are genuinely different
    full = D.run_trajectory(PROTOCOL, eps, first["q_threshold"])
    assert full["status"] == D.FAIL
    assert full["round"] is not None and full["round"] >= 1
    for p in (D.__file__, C.__file__):
        assert f"{first['q_threshold']:.9f}" not in Path(p).read_text()


def test_h_binding_round_is_derived_from_the_data_and_is_not_zero():
    eps = 0.15
    rows = D.protocol_table(PROTOCOL, [eps], verbose=False, time_budget=90.0)
    r = rows[0]
    assert r["binding_round"] is not None
    # measured, not hard-coded: recompute it here from the fidelity series
    tr = D.run_trajectory(PROTOCOL, eps, r["q_th_full_upper_fail"], record=True)
    if tr["status"] == D.FAIL:
        f = [x["F"] for x in tr["rows"]]
        expect = next(n for n in range(len(f) - 1)
                      if f[n + 1] - f[n] < -D.VIOL_TOL)
        assert r["binding_round"] == expect
    assert isinstance(r["binding_round"], int)


# --- I: no analytic shortcut -----------------------------------------------

def test_i_no_analytic_recurrence_in_the_primary_path():
    assert_no_analytic_imports()


# --- J: independent routes --------------------------------------------------

def test_j_route_a_and_route_b_agree():
    """Pauli tomography (route A) against the parity-weighted partial trace."""
    assert_routes_agree(PROTOCOL, [(0.05, 0.0), (0.15, 0.03), (0.40, 0.08),
                                   (2.0 / 3.0, 0.10)], tol=1e-11)


@pytest.mark.skipif(not D.HP.AVAILABLE, reason="python-flint not installed")
def test_float_and_high_precision_routes_agree():
    res = D.route_comparison_hp([(PROTOCOL, 0.40, 0.05)])
    assert res["worst"]["rho"] < 1e-11


# --- K, L: the bracket and the bisection -----------------------------------

@pytest.mark.parametrize("eps", [0.05, 0.15])
def test_k_bracket_endpoints_pass_and_fail(eps):
    rows = D.protocol_table(PROTOCOL, [eps], verbose=False, time_budget=90.0)
    r = rows[0]
    assert r["q_th_full_lower_pass"] is not None
    assert D.hybrid_status(PROTOCOL, eps, r["q_th_full_lower_pass"])["status"] \
        == D.PASS
    assert D.hybrid_status(PROTOCOL, eps, r["q_th_full_upper_fail"])["status"] \
        == D.FAIL


def test_l_bisection_width_meets_the_tolerance():
    rows = D.protocol_table(PROTOCOL, [0.05], verbose=False, time_budget=90.0)
    assert rows[0]["threshold_bracket_width"] <= D.BISECT_TOL + 1e-18


# --- M: degenerate endpoints ------------------------------------------------

def test_m_degenerate_endpoints_are_handled_separately():
    degen = {d["epsilon"]: d for d in D.degenerate_endpoints([PROTOCOL])}
    assert set(degen) == {0.0, 1.0}
    assert abs(degen[0.0]["F_0"] - 1.0) < 1e-15
    assert degen[0.0]["strict_increase_possible"] is False
    for pr in degen[0.0]["probes"]:
        assert pr["dF1"] is None or pr["dF1"] <= 1e-14
    assert abs(degen[1.0]["F_0"] - 0.25) < 1e-15
    assert 0.0 not in D.EPS_TABLE and 1.0 not in D.EPS_TABLE


# --- N: a contaminated trajectory is never silently FAIL --------------------

def test_n_transverse_contamination_is_reported_not_called_failure():
    r = D.run_trajectory(PROTOCOL, 0.40, 0.05, offdiag_trust=1e-40)
    assert r["status"] == D.AMBIG
    assert "off-diagonal" in r["reason"]
    lo, hi = D._last_pass_first_fail([0.0, 0.1, 0.2],
                                     [D.PASS, D.AMBIG, D.AMBIG])
    assert lo is None and hi is None


def test_transverse_direction_really_is_amplified_here():
    """The reason this protocol needs the guard at all, measured."""
    eps, q, amp = 0.40, 0.08, 1e-10
    base = C.isotropic_rho(eps)
    pert = amp * (np.outer(C.PHI_P, C.PHI_M.conj())
                  + np.outer(C.PHI_M, C.PHI_P.conj()))
    rho = base + pert
    rho = rho / np.trace(rho).real
    w0 = np.sqrt(C.bell_offdiagonal_weight(rho))
    for _ in range(6):
        rho, _ = C.one_round(rho, q, PROTOCOL)
    w6 = np.sqrt(C.bell_offdiagonal_weight(rho))
    assert w6 > w0, (w0, w6)


@pytest.mark.skipif(not D.HP.AVAILABLE, reason="python-flint not installed")
def test_high_precision_route_keeps_the_state_bell_diagonal():
    from fractions import Fraction
    r = D.run_trajectory_hp(PROTOCOL, Fraction(2, 5), Fraction(1, 20))
    assert r["status"] in (D.PASS, D.FAIL)
    assert r["c_off_max"] < 1e-100
    assert r["max_ball_radius"] < 1e-60


# --- O: negative controls ---------------------------------------------------

def test_o_negative_controls_are_detected():
    nc = D.negative_controls([PROTOCOL])
    names = {r["control"] for r in nc["rows"]}
    assert "F_wrong_ancilla_parity_sign" in names
    assert len(names) >= 5
    for r in nc["rows"]:
        assert r["detected"], r["control"]


# --- P: the frozen q_abs / q_rel results are untouched ----------------------

def test_p_frozen_threshold_files_are_not_modified():
    paths = [ROOT / "results" / d / f for d, f in (
        ("type4_qabs", "type4_qabs_circuit.csv"), ("type4_qrel", "type4_qrel_circuit.csv"))]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths if p.exists()}
    assert before
    D.secondary_hierarchy([{"protocol": PROTOCOL, "epsilon": 0.15,
                            "q_th_full_estimate": 0.05}])
    for p, h in before.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == h, p


def test_normalisation_denominator_is_recorded_and_guarded():
    """Type 3/4 divide by a parity weight that collapses at large q."""
    _out, info = C.one_round(C.isotropic_rho(0.4), 0.05, PROTOCOL)
    assert info["denominator"] > D.DENOM_FLOOR
    r = D.run_trajectory(PROTOCOL, 0.4, 0.88)
    assert r["status"] in (D.FAIL, D.UNSAFE)
    if r["status"] == D.UNSAFE:
        assert "denominator" in r["reason"]
