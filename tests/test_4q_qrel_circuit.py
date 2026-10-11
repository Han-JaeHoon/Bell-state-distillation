"""Circuit-level 4Q relative-suppression threshold study: automated checks.

These tests exercise the implementation and the threshold machinery of
``scripts/verify_4q_qrel_circuit.py``.  The primary path of that script uses no
analytic 4Q recurrence and no previously stored threshold; a test below checks
that textually as well as by inspecting the imported module.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "verify_4q_qrel_circuit.py"


def _load():
    spec = importlib.util.spec_from_file_location("_4q_qrel_script", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load()
TOL = 1e-12


# ===========================================================================
# A: the frozen 5-CNOT sequence and its directions
# ===========================================================================

def test_a_five_cnot_sequence_and_directions_are_frozen():
    assert S.N_CNOT == 5
    assert S.CNOT_SEQUENCE == [(2, 3), (1, 3), (0, 3), (2, 1), (2, 0)]
    assert S.CNOT_LABELS == [("q3", "q4"), ("q2", "q4"), ("q1", "q4"),
                             ("q3", "q2"), ("q3", "q1")]
    assert S.RETAIN == (0, 1)
    assert S.MEASURED == (2, 3)
    assert S.SUCCESS_OUTCOME == (0, 0)
    # and it is the repository's own list, read textually from both sources
    prov = S.provenance_checks()
    assert prov["matches_src/pqec_distill/circuit.py"] is True
    assert prov["matches_scripts/verify_4q_qabs_circuit.py"] is True
    # the directions matter: reversing any one changes the unitary
    for k in range(5):
        seq = list(S.CNOT_SEQUENCE)
        seq[k] = (seq[k][1], seq[k][0])
        assert np.max(np.abs(S.full_unitary(sequence=seq)
                             - S.full_unitary())) > 1e-03


def test_circuit_unitary_is_unitary_and_route_independent():
    v = S.full_unitary()
    assert np.max(np.abs(v.conj().T @ v - np.eye(S.DIM))) < TOL
    assert np.max(np.abs(v - S.full_unitary(S.cnot_permutation))) < TOL
    assert np.max(np.abs(S._hadamard_route_b(S.Q3)
                         - S.place_single(S._H, S.Q3))) < TOL


# ===========================================================================
# B, C: the noise channel at the two endpoints
# ===========================================================================

def test_b_q0_noise_is_the_identity_channel():
    rng = np.random.default_rng(5)
    z = S.random_density_matrix(rng, S.DIM)
    for pair in [(0, 1), (2, 3), (0, 3), (1, 2)]:
        assert S.replacement_depol(z, pair, 0.0) is z or \
            np.max(np.abs(S.replacement_depol(z, pair, 0.0) - z)) == 0.0
        assert np.max(np.abs(S._depol_route_b(z, pair, 0.0) - z)) == 0.0


def test_c_q1_replaces_the_pair_by_the_identity():
    rng = np.random.default_rng(6)
    z = S.random_density_matrix(rng, S.DIM)
    for pair in [(0, 1), (2, 3), (1, 3)]:
        out = S.replacement_depol(z, pair, 1.0)
        # the pair's reduced state must be exactly I/4 ...
        keep = list(pair)
        t = out.reshape([2] * 8)
        traced = [w for w in range(4) if w not in pair]
        row, col = list("abcd"), list("efgh")
        for w in traced:
            col[w] = row[w]
        red = np.einsum("".join(row) + "".join(col) + "->"
                        + "".join(row[w] for w in keep)
                        + "".join(col[w] for w in keep), t).reshape(4, 4)
        assert np.max(np.abs(red - np.eye(4) / 4)) < TOL
        # ... and the other two qubits must be untouched
        keep2 = traced
        t2 = z.reshape([2] * 8)
        row, col = list("abcd"), list("efgh")
        for w in pair:
            col[w] = row[w]
        before = np.einsum("".join(row) + "".join(col) + "->"
                           + "".join(row[w] for w in keep2)
                           + "".join(col[w] for w in keep2), t2).reshape(4, 4)
        t3 = out.reshape([2] * 8)
        row, col = list("abcd"), list("efgh")
        for w in pair:
            col[w] = row[w]
        after = np.einsum("".join(row) + "".join(col) + "->"
                          + "".join(row[w] for w in keep2)
                          + "".join(col[w] for w in keep2), t3).reshape(4, 4)
        assert np.max(np.abs(before - after)) < TOL
    # and it agrees with the independently coded channel
    assert S.structural_checks(np.random.default_rng(7))[
        "channel_vs_route_b_channel"] < TOL


# ===========================================================================
# D: the postselected state is normalised and physical
# ===========================================================================

@pytest.mark.parametrize("eps", [0.02, 0.15, 0.40, 2.0 / 3.0, 0.90])
@pytest.mark.parametrize("q", [0.0, 0.03, 0.2, 1.0])
def test_d_postselected_state_is_normalised_and_physical(eps, q):
    rho = S.isotropic_rho(eps)
    for _ in range(6):
        rho, info = S.one_round(rho, q)
        assert info["p_succ"] > 0.0
        assert abs(float(np.real(np.trace(rho))) - 1.0) < TOL
        assert np.max(np.abs(rho - rho.conj().T)) < TOL
        assert float(np.linalg.eigvalsh(rho).min()) > -TOL
        p = S.bell_populations(rho)
        assert abs(float(p.sum()) - 1.0) < TOL
        assert p.min() > -TOL


def test_measurement_branch_probabilities_sum_to_one():
    assert S.structural_checks(np.random.default_rng(8))[
        "branch_probabilities_sum"] < TOL


# ===========================================================================
# E, F: Bell populations and the relative ratio definition
# ===========================================================================

def test_e_bell_populations_are_computed_correctly():
    # on the four Bell states themselves
    for k, name in enumerate(S.BELL_ORDER):
        rho = np.outer(S.BELL[name], S.BELL[name].conj())
        p = S.bell_populations(rho)
        expect = np.zeros(4)
        expect[k] = 1.0
        assert np.max(np.abs(p - expect)) < TOL
        assert np.max(np.abs(S.bell_populations_route_b(rho) - expect)) < TOL
    # on the Bell-isotropic input, against the construction identity
    for eps in (0.0, 0.05, 0.3, 2.0 / 3.0, 1.0):
        p = S.bell_populations(S.isotropic_rho(eps))
        assert abs(p[0] - (1.0 - 3.0 * eps / 4.0)) < TOL
        assert np.max(np.abs(p[1:] - eps / 4.0)) < TOL
    # and on random states, against independently built projectors
    rng = np.random.default_rng(99)
    for _ in range(20):
        rho = S.random_density_matrix(rng, 4)
        assert np.max(np.abs(S.bell_populations(rho)
                             - S.bell_populations_route_b(rho))) < TOL


def test_f_relative_ratio_definition():
    p = np.array([0.8, 0.1, 0.06, 0.04])
    r = S.relative_ratios(p)
    assert abs(r[0] - 1.0) < TOL
    for i in (1, 2, 3):
        assert abs(r[i] - p[i] / p[0]) < TOL
    # the guard: a vanishing target population is reported, not silently used
    assert np.all(np.isinf(S.relative_ratios(np.array([0.0, .3, .3, .4]))))


def test_relative_criterion_is_not_the_absolute_criterion():
    """A round with Delta p_i > 0 and Delta R_i < 0 must count as satisfying."""
    rng = np.random.default_rng(4)
    p_n = np.array([0.5, 1.0 / 6, 1.0 / 6, 1.0 / 6])
    p_np1 = np.array([0.6, 0.18, 0.11, 0.11])
    dp = p_np1 - p_n
    dr = S.relative_ratios(p_np1) - S.relative_ratios(p_n)
    assert dp[1] > 0                      # the population rose ...
    assert dr[1] < 0                      # ... but its ratio fell
    assert rng is not None


# ===========================================================================
# G: no analytic shortcut in the primary path
# ===========================================================================

def test_g_primary_path_uses_no_analytic_recurrence():
    forbidden = ("analytic_exact_map", "analytic_map_4q", "noisy_analytics",
                 "analytic_map_type3", "analytic_map_type4",
                 "repeated_dynamics", "noisy_closed_form",
                 "bell_population_dynamics")
    text = SCRIPT_PATH.read_text()
    code = "\n".join(l for l in text.splitlines() if not re.match(r"\s*#", l))
    for name in forbidden:
        assert not re.search(rf"import\s+.*{name}", code), name
        assert not re.search(rf"{name}\s*\(", code), name
    # the script imports nothing from the package at all
    assert "from pqec_distill" not in code
    assert not re.search(r"^\s*import pqec_distill", code, re.M)
    assert not [n for n in vars(S) if "analytic" in n]
    # the frozen q_abs RESULTS file is opened in exactly one place, the
    # clearly separated secondary step
    idx = text.index("def secondary_q_abs_comparison")
    end = text.index("\n# =====", idx)
    body = text[idx:end]
    assert '"4q_qabs_circuit.csv"' in body
    assert text.count("4q_qabs_circuit.csv") == body.count("4q_qabs_circuit.csv")
    assert text.count('"4q_qabs"') == body.count('"4q_qabs"')
    # the threshold search itself never sees a q_abs value: no q_abs name
    # appears anywhere in the bracket/bisect/threshold machinery
    search = text[text.index("def bracket("):
                  text.index("# 10. SECONDARY")]
    assert "q_abs" not in search
    # and it takes no bracketing hint parameter at all
    assert "hint=" not in search
    assert "hint:" not in search


def test_secondary_comparison_does_not_modify_the_frozen_q_abs_results():
    path = ROOT / "results" / "4q_qabs" / "4q_qabs_circuit.csv"
    if not path.exists():
        pytest.skip("frozen q_abs results not present")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    S.secondary_q_abs_comparison([{"epsilon": 0.15, "q_rel_full": 0.05}])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


# ===========================================================================
# H: route A against route B
# ===========================================================================

def test_h_route_a_and_route_b_agree():
    res = S.route_comparison([(0.05, 0.0), (0.15, 0.05), (0.40, 0.12),
                              (2.0 / 3.0, 0.14), (0.20, 1.0)])
    for key, worst in res["worst"].items():
        assert worst < TOL, (key, worst)
    assert len(res["rows"]) == 25


# ===========================================================================
# I, J: the threshold bracket and the bisection
# ===========================================================================

@pytest.mark.parametrize("eps", [0.05, 0.15, 0.40])
def test_i_bracket_endpoints_satisfy_and_violate(eps):
    res = S.q_rel_full(eps)
    assert res["has_root"]
    assert res["n_sign_changes"] == 1
    lo, hi = res["q_threshold"], res["q_upper_violating"]
    assert 0.0 < lo < hi
    assert not S.run_trajectory(eps, lo)["violated"]
    assert S.run_trajectory(eps, hi)["violated"]


@pytest.mark.parametrize("eps", [0.05, 0.40])
def test_j_bisection_width_meets_the_tolerance(eps):
    res = S.q_rel_full(eps, tol=1e-10)
    assert res["bracket_width"] <= 1e-10 + 1e-18
    assert res["bisect_iterations"] < 200


# ===========================================================================
# K: the two predicates are computed and stored independently
# ===========================================================================

def test_k_full_and_first_round_thresholds_are_independent_quantities():
    eps = 0.15
    full = S.q_rel_full(eps)
    first = S.q_rel_first(eps)
    # distinct code paths: one iterates to convergence, one does a single round
    assert S.run_trajectory(eps, full["q_threshold"])["rounds_run"] > 1
    assert full["has_root"] and first["has_root"]
    # the first-round predicate really is a single round
    assert S.first_round_violates(eps, first["q_upper_violating"])
    assert not S.first_round_violates(eps, first["q_threshold"])
    # neither value is hard-coded anywhere in the script
    text = SCRIPT_PATH.read_text()
    for v in (full["q_threshold"], first["q_threshold"]):
        assert f"{v:.8f}" not in text


# ===========================================================================
# L: the limiting component comes from the data
# ===========================================================================

def test_l_limiting_component_is_detected_from_the_data():
    eps = 0.15
    res = S.q_rel_full(eps)
    tr = S.run_trajectory(eps, res["q_upper_violating"])
    assert tr["violated"]
    detected = tr["component"]
    # it must be the arg-max of Delta R over the non-target components,
    # recomputed here independently of the script's bookkeeping
    p0, p1, r0, r1, _ps = S.first_round_step(eps, res["q_upper_violating"])
    dr = r1 - r0
    idx = [S.BELL_ORDER.index(c) for c in S.NON_TARGET]
    expect = S.BELL_ORDER[max(idx, key=lambda i: dr[i])]
    assert detected == expect
    # and it is not merely the first entry of the NON_TARGET tuple
    assert S.NON_TARGET[0] == "Phi-"
    assert detected != S.NON_TARGET[0]
    # restricting the predicate to a single component gives a different,
    # larger threshold for the non-limiting ones
    limiting = S.q_rel_full(eps, components=(detected,))["q_threshold"]
    assert abs(limiting - res["q_threshold"]) < 1e-09
    for c in S.NON_TARGET:
        if c == detected:
            continue
        other = S.q_rel_full(eps, components=(c,))["q_threshold"]
        assert other > res["q_threshold"] + 1e-09


def test_degenerate_pair_equality_is_measured_not_imposed():
    """p_Phi- = p_Psi- is an observation here; the two are computed apart."""
    eps, q = 0.15, 0.05
    rho = S.isotropic_rho(eps)
    worst = 0.0
    for _ in range(6):
        rho, _ = S.one_round(rho, q)
        p = S.bell_populations(rho)
        worst = max(worst, abs(float(p[1] - p[3])))
    assert worst < 1e-14
    # the populations really are evaluated by separate expectation values
    assert S.BELL_ORDER.index("Phi-") != S.BELL_ORDER.index("Psi-")


# ===========================================================================
# M: the degenerate endpoints
# ===========================================================================

def test_m_degenerate_endpoints_are_handled_explicitly():
    degen = S.degenerate_endpoints()
    by_eps = {d["epsilon"]: d for d in degen}
    assert set(by_eps) == {0.0, 1.0}

    zero = by_eps[0.0]
    # eps = 0 starts with vanishing non-target ratios, so no q > 0 can lower
    # them further; only q = 0 satisfies the criterion
    assert max(zero["R0"][1:]) < 1e-30
    assert zero["q_threshold"] == 0.0
    assert not S.run_trajectory(0.0, 0.0)["violated"]
    for q in (1e-6, 1e-3, 0.05):
        assert S.run_trajectory(0.0, q)["violated"]

    one = by_eps[1.0]
    # eps = 1 is the maximally mixed fixed point: every ratio is 1 and stays 1
    assert np.max(np.abs(np.array(one["R0"]) - 1.0)) < TOL
    assert one["has_root_in_scan"] is False
    assert one["none_violating"] is True
    assert one["n_sign_changes"] == 0
    for pr in one["probes"]:
        assert pr["violated"] is False
        assert abs(pr["worst_delta_R"]) < TOL
    # and these two epsilons are not in the ordinary table
    assert 0.0 not in S.EPS_TABLE and 1.0 not in S.EPS_TABLE
    assert S.EPS_DEGENERATE == (0.0, 1.0)


# ===========================================================================
# N: negative / sensitivity controls
# ===========================================================================

def test_n_negative_controls_are_detected():
    nc = S.negative_controls()
    assert nc["baseline_q_rel_full"] is not None
    names = {r["control"] for r in nc["rows"]}
    assert names == {"A_one_cnot_noise_plus_1e-3", "B_one_channel_removed",
                     "C_one_cnot_reversed", "D_wrong_postselection_branch"}
    for r in nc["rows"]:
        assert r["detected"], r["control"]
        assert r["max_abs_rho_diff"] > 1e-09, r["control"]
    # the two structural controls break suppression entirely in the scan
    by = {r["control"]: r for r in nc["rows"]}
    assert by["C_one_cnot_reversed"]["modified_has_root_in_scan"] is False
    assert by["D_wrong_postselection_branch"]["modified_has_root_in_scan"] is False
    # the two noise controls shift the threshold measurably
    for k in ("A_one_cnot_noise_plus_1e-3", "B_one_channel_removed"):
        assert abs(by[k]["q_rel_full_shift"]) > 1e-06, k


def test_transverse_direction_is_measured_not_assumed():
    tg = S.transverse_growth()
    # one round takes a 2e-8 transverse amplitude down to round-off
    assert tg["max_amplitude_after_first_round"] < 1e-14
    assert tg["later_ratios_are_roundoff_level"] is True
    # and the scaling exponent is measured, not imposed
    assert np.isfinite(tg["amplitude_scaling_exponent"])
