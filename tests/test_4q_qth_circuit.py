"""q_th full-circuit study: the 4Q postselected protocol (5 CNOTs)."""

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

PROTOCOL = "4q"
TOL = 1e-12
D = driver()
C = core()


# --- A, B: the frozen specification ----------------------------------------

def test_a_circuit_matches_the_frozen_repository_source():
    prov = C.provenance_checks()
    assert prov["4q_matches_src/pqec_distill/circuit.py"] is True
    assert prov["4q_matches_scripts/verify_4q_qabs_circuit.py"] is True
    assert C.CNOT_SEQUENCE_4Q == [(2, 3), (1, 3), (0, 3), (2, 1), (2, 0)]
    assert C.RETAIN_4Q == (0, 1) and C.MEASURED_4Q == (2, 3)
    assert C.SUCCESS_OUTCOME_4Q == (0, 0)
    assert C.OPS["4q"][-1] == ("H", C.Q3)


def test_b_exact_cnot_count():
    assert C.n_cnot(PROTOCOL) == expected_cnots(PROTOCOL) == 5
    assert C.provenance_checks()["n_cnot_ok"] is True


# --- C, D: the noise channel -----------------------------------------------

def test_c_q0_noise_is_identity():
    assert_q0_is_identity(PROTOCOL)


def test_d_q1_replaces_the_pair():
    assert_q1_replaces_the_pair(PROTOCOL)


# --- E: the fidelity observable --------------------------------------------

def test_e_bell_projector_gives_the_right_fidelity():
    assert_bell_projector_fidelity()


# --- F: the predicate is about fidelity, not Bell-error populations --------

def test_f_predicate_uses_delta_F_not_bell_error_populations():
    # q chosen below q_th but high enough that the Bell-error populations are
    # no longer all decreasing: exactly the regime where the fidelity
    # criterion and a Bell-error criterion disagree
    eps, q = 0.40, 0.14
    r = D.run_trajectory(PROTOCOL, eps, q, record=True)
    assert r["status"] == D.PASS
    rows = r["rows"]
    # some non-target population rises on a round that the predicate accepts
    rose = any(rows[n + 1]["p"][k] > rows[n]["p"][k] + 1e-12
               for n in range(len(rows) - 1) for k in (1, 2, 3))
    assert rose, "expected at least one rising Bell-error population"
    # while every fidelity step is positive
    assert all(x["dF"] > -D.VIOL_TOL for x in rows[1:])
    # and the recorded fidelities are exactly <Phi+|rho|Phi+>
    rho = C.isotropic_rho(eps)
    assert abs(rows[0]["F"] - C.fidelity(rho)) < 1e-15
    for x in rows[1:]:
        rho, _ = C.one_round(rho, q, PROTOCOL)
        assert abs(x["F"] - C.fidelity(rho)) < 1e-14


# --- G, H: the two thresholds and the binding round ------------------------

def test_g_first_and_full_thresholds_are_evaluated_independently():
    eps = 0.15
    first = D.q_th_first(PROTOCOL, eps)
    assert first["has_root"] and first["n_sign_changes"] == 1
    # the one-round predicate really is one round
    f0, f1, _ = D.first_round_step(PROTOCOL, eps, first["q_upper_fail"])
    assert f1 - f0 < -D.VIOL_TOL
    f0, f1, _ = D.first_round_step(PROTOCOL, eps, first["q_threshold"])
    assert f1 - f0 > 0
    # the all-round predicate iterates
    full = D.run_trajectory(PROTOCOL, eps, first["q_threshold"])
    assert full["rounds_run"] > 1
    # neither number is hard-coded in the sources
    for p in (D.__file__, C.__file__):
        assert f"{first['q_threshold']:.9f}" not in Path(p).read_text()


def test_h_binding_round_is_derived_from_the_data():
    eps = 0.40
    first = D.q_th_first(PROTOCOL, eps)
    rec = D._binding_record(PROTOCOL, eps, first["q_upper_fail"])
    assert rec["status_at_upper_fail"] == D.FAIL
    assert rec["binding_round"] == 0
    # recomputed independently here
    f0, f1, _ = D.first_round_step(PROTOCOL, eps, first["q_upper_fail"])
    assert abs(rec["DeltaF_at_upper_fail"] - (f1 - f0)) < 1e-15
    assert rec["binding_round"] is not None


# --- I: no analytic shortcut -----------------------------------------------

def test_i_no_analytic_recurrence_in_the_primary_path():
    assert_no_analytic_imports()


# --- J: independent routes --------------------------------------------------

def test_j_route_a_and_route_b_agree():
    assert_routes_agree(PROTOCOL, [(0.05, 0.0), (0.15, 0.05), (0.40, 0.12),
                                   (2.0 / 3.0, 0.17), (0.90, 0.3)])


@pytest.mark.skipif(not D.HP.AVAILABLE, reason="python-flint not installed")
def test_float_and_high_precision_routes_agree():
    res = D.route_comparison_hp([(PROTOCOL, 0.40, 0.08)])
    assert res["worst"]["rho"] < 1e-12


# --- K, L: the bracket and the bisection -----------------------------------

@pytest.mark.parametrize("eps", [0.05, 0.40])
def test_k_bracket_endpoints_pass_and_fail(eps):
    rows = D.protocol_table(PROTOCOL, [eps], verbose=False, time_budget=60.0)
    r = rows[0]
    assert r["q_th_full_lower_pass"] is not None
    assert D.run_trajectory(PROTOCOL, eps, r["q_th_full_lower_pass"])["status"] \
        == D.PASS
    assert D.run_trajectory(PROTOCOL, eps, r["q_th_full_upper_fail"])["status"] \
        == D.FAIL
    assert r["predicate_sign_changes"] == 1


def test_l_bisection_width_meets_the_tolerance():
    rows = D.protocol_table(PROTOCOL, [0.15], verbose=False, time_budget=60.0)
    assert rows[0]["threshold_bracket_width"] <= D.BISECT_TOL + 1e-18
    assert rows[0]["trusted_flag"] is True


# --- M: degenerate endpoints ------------------------------------------------

def test_m_degenerate_endpoints_are_handled_separately():
    degen = {d["epsilon"]: d for d in D.degenerate_endpoints([PROTOCOL])}
    assert set(degen) == {0.0, 1.0}
    assert abs(degen[0.0]["F_0"] - 1.0) < 1e-15
    assert degen[0.0]["strict_increase_possible"] is False
    for pr in degen[0.0]["probes"]:
        assert pr["dF1"] is None or pr["dF1"] <= 1e-15
    assert abs(degen[1.0]["F_0"] - 0.25) < 1e-15
    for pr in degen[1.0]["probes"]:
        assert pr["dF1"] is None or abs(pr["dF1"]) < 1e-14
    assert 0.0 not in D.EPS_TABLE and 1.0 not in D.EPS_TABLE


# --- N: a contaminated trajectory is never silently FAIL --------------------

def test_n_transverse_contamination_is_reported_not_called_failure():
    """Forcing the trust threshold below the seed must give AMBIGUOUS."""
    r = D.run_trajectory(PROTOCOL, 0.40, 0.05, offdiag_trust=1e-40)
    assert r["status"] == D.AMBIG
    assert "off-diagonal" in r["reason"]
    assert r["status"] != D.FAIL
    # and an AMBIGUOUS label can never be read as a bracket endpoint
    lo, hi = D._last_pass_first_fail([0.0, 0.1, 0.2],
                                     [D.PASS, D.AMBIG, D.AMBIG])
    assert lo is None and hi is None


def test_4q_shows_no_transverse_contamination_in_practice():
    for eps, q in ((0.05, 0.02), (0.40, 0.05), (2.0 / 3.0, 0.05)):
        r = D.run_trajectory(PROTOCOL, eps, q)
        assert r["status"] == D.PASS
        assert r["c_off_max"] < D.OFFDIAG_TRUST


# --- O: negative controls ---------------------------------------------------

def test_o_negative_controls_are_detected():
    nc = D.negative_controls([PROTOCOL])
    names = {r["control"] for r in nc["rows"]}
    assert "E_wrong_postselection_branch" in names
    assert len(names) >= 5
    for r in nc["rows"]:
        assert r["detected"], r["control"]
        assert r["max_abs_rho_diff"] > 1e-09 or r["abs_F_diff"] > 1e-09, r


# --- P: the frozen q_abs / q_rel results are untouched ----------------------

def test_p_frozen_threshold_files_are_not_modified():
    paths = [ROOT / "results" / d / f for d, f in (
        ("4q_qabs", "4q_qabs_circuit.csv"),
        ("type3_qabs", "type3_qabs_circuit.csv"),
        ("type4_qabs", "type4_qabs_circuit.csv"),
        ("type3_qrel", "type3_qrel_circuit.csv"),
        ("type4_qrel", "type4_qrel_circuit.csv"))]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths if p.exists()}
    assert before, "expected at least one frozen threshold file"
    D.secondary_hierarchy([{"protocol": PROTOCOL, "epsilon": 0.15,
                            "q_th_full_estimate": 0.08}])
    for p, h in before.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == h, p


def test_secondary_hierarchy_is_not_used_by_the_search():
    """The hierarchy comparison is read-only and cannot steer the search."""
    rows = D.protocol_table(PROTOCOL, [0.15], verbose=False, time_budget=60.0)
    est = rows[0]["q_th_full_estimate"]
    hier = D.secondary_hierarchy(rows)
    assert hier["n_compared"] >= 1
    # calling it again changes nothing
    rows2 = D.protocol_table(PROTOCOL, [0.15], verbose=False, time_budget=60.0)
    assert rows2[0]["q_th_full_estimate"] == est
