"""Guards for the Bell-population dynamics analysis.

The SWAP-test protocols (Type 3/4/5) need the parent research repository; those
tests skip cleanly when it is not present, and the rest still run.
"""

from __future__ import annotations

import numpy as np
import pytest

from pqec_distill.bell_population_dynamics import (
    BELL_NAMES, INVARIANT_PAIR, PROTOCOLS, bell_offdiagonal_c, bell_populations,
    cross_check_exact_vs_dm, isotropic_dm, isotropic_populations,
    plateau_and_escape, run_dynamics, state_diagnostics,
    verify_cnot_counts, verify_noise_convention,
)
from pqec_distill.bell_states import bell_state, bell_diagonal_state
from pqec_distill.bell_states import bell_populations as bs_bell_populations

EPS, Q = 0.15, 0.02
EXPECTED_CNOTS = {"Type3": 16, "Type4": 14, "Type5": 14, "4Q": 5}


def _parent_available() -> bool:
    try:
        from pqec_distill.swap_test_source import find_parent_repo
        find_parent_repo()
        import pennylane  # noqa: F401
    except Exception:
        return False
    return True


needs_parent = pytest.mark.skipif(
    not _parent_available(),
    reason="parent repository PQEC-Operational-Threshold (or pennylane) not available")


# ---------------------------------------------------------------------------
# initial state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("eps", [0.0, 0.15, 0.4, 1.0])
def test_isotropic_populations_formula(eps):
    p = isotropic_populations(eps)
    assert p[0] == pytest.approx(1 - 3 * eps / 4, abs=1e-15)
    for i in (1, 2, 3):
        assert p[i] == pytest.approx(eps / 4, abs=1e-15)
    assert p.sum() == pytest.approx(1.0, abs=1e-15)


def test_eps_015_matches_the_stated_numbers():
    p = isotropic_populations(0.15)
    assert p == pytest.approx([0.8875, 0.0375, 0.0375, 0.0375], abs=1e-15)


def test_isotropic_dm_is_a_valid_bell_diagonal_state():
    rho = isotropic_dm(EPS)
    d = state_diagnostics(rho)
    assert d["trace_err"] < 1e-15
    assert d["herm_err"] < 1e-15
    assert d["min_eig"] > 0
    assert d["pop_sum_err"] < 1e-15
    assert bell_offdiagonal_c(rho) < 1e-30
    assert bell_populations(rho) == pytest.approx(isotropic_populations(EPS), abs=1e-15)


def test_bell_populations_agrees_with_the_package_helper():
    rng = np.random.default_rng(3)
    for _ in range(10):
        p = rng.dirichlet(np.ones(4))
        rho = bell_diagonal_state(p)
        assert bell_populations(rho) == pytest.approx(bs_bell_populations(rho), abs=1e-14)


def test_bell_offdiagonal_c_on_a_known_coherent_state():
    """(|Phi+> + |Psi+>)/sqrt(2) has two off-diagonal entries of modulus 1/2."""
    v = (bell_state(0, 0) + bell_state(1, 0)) / np.sqrt(2)
    rho = np.outer(v, v.conj())
    assert bell_offdiagonal_c(rho) == pytest.approx(2 * 0.25, abs=1e-14)


# ---------------------------------------------------------------------------
# circuits and noise convention
# ---------------------------------------------------------------------------

@needs_parent
def test_cnot_counts():
    assert verify_cnot_counts() == EXPECTED_CNOTS


@needs_parent
def test_q_is_the_same_channel_on_both_sides():
    r = verify_noise_convention(n_trials=2)
    assert r["max_abs_diff_2q_vs_parent_kraus"] < 1e-13
    assert r["max_abs_diff_5q_vs_parent_replacement"] < 1e-13


# ---------------------------------------------------------------------------
# the round maps
# ---------------------------------------------------------------------------

@needs_parent
@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_every_iterate_is_a_valid_state(protocol):
    dyn = run_dynamics(protocol, EPS, Q, n_rounds=8, backend="dm")
    for d in dyn.diagnostics:
        assert d["trace_err"] < 1e-12
        assert d["herm_err"] < 1e-12
        assert d["min_eig"] > -1e-12
        assert d["pop_sum_err"] < 1e-12
    assert np.all(dyn.pops.sum(axis=1) == pytest.approx(1.0, abs=1e-12))
    assert dyn.fidelity[0] == pytest.approx(1 - 3 * EPS / 4, abs=1e-15)


@needs_parent
@pytest.mark.parametrize("protocol", ["Type3", "Type4", "4Q"])
def test_exact_map_and_full_density_matrix_agree(protocol):
    r = cross_check_exact_vs_dm(protocol, EPS, Q, n_rounds=12)
    assert r["max_abs_pop_diff"] < 1e-12
    assert r["max_abs_F_diff"] < 1e-12


@needs_parent
@pytest.mark.parametrize("protocol", ["Type3", "Type4", "4Q"])
def test_bell_diagonal_protocols_stay_bell_diagonal(protocol):
    dyn = run_dynamics(protocol, EPS, Q, n_rounds=12, backend="dm")
    assert dyn.c_bell.max() < 1e-20


@needs_parent
def test_type5_leaves_the_bell_diagonal_manifold():
    dyn = run_dynamics("Type5", EPS, Q, n_rounds=12, backend="dm")
    assert dyn.c_bell[0] < 1e-30            # the input is Bell-diagonal
    assert dyn.c_bell[1:].max() > 1e-9      # one round already breaks it


@needs_parent
def test_type5_refuses_the_exact_backend():
    with pytest.raises(ValueError, match="no verified Bell-diagonal exact map"):
        run_dynamics("Type5", EPS, Q, n_rounds=3, backend="exact")


def test_unknown_protocol_is_rejected():
    with pytest.raises(KeyError):
        run_dynamics("Type9", EPS, Q, n_rounds=1)


# ---------------------------------------------------------------------------
# the symmetries each protocol does and does not keep
# ---------------------------------------------------------------------------

@needs_parent
@pytest.mark.parametrize("protocol", ["Type3", "Type4"])
def test_swap_test_keeps_psi_plus_equal_psi_minus(protocol):
    dyn = run_dynamics(protocol, EPS, Q, n_rounds=12, backend="dm")
    i_p, i_m = BELL_NAMES.index("Psi+"), BELL_NAMES.index("Psi-")
    assert np.max(np.abs(dyn.pops[:, i_p] - dyn.pops[:, i_m])) < 1e-14


@needs_parent
def test_4q_keeps_phi_minus_equal_psi_minus():
    dyn = run_dynamics("4Q", EPS, Q, n_rounds=12, backend="dm")
    i_a, i_b = BELL_NAMES.index("Phi-"), BELL_NAMES.index("Psi-")
    assert np.max(np.abs(dyn.pops[:, i_a] - dyn.pops[:, i_b])) < 1e-14


@needs_parent
def test_type5_breaks_every_pairwise_equality():
    """Guard against carrying a Type 3/4 or 4Q symmetry over to Type 5."""
    dyn = run_dynamics("Type5", EPS, Q, n_rounds=12, backend="dm")
    p = dyn.pops[-1]
    for i, j in ((1, 2), (1, 3), (2, 3)):
        assert abs(p[i] - p[j]) > 1e-6


# ---------------------------------------------------------------------------
# plateau / escape detection
# ---------------------------------------------------------------------------

@needs_parent
def test_no_escape_reported_for_a_settling_protocol():
    dyn = run_dynamics("Type3", EPS, Q, n_rounds=60, backend="dm")
    assert plateau_and_escape(dyn)["n_escape"] is None


@needs_parent
@pytest.mark.slow
def test_type5_plateau_then_escape():
    dyn = run_dynamics("Type5", EPS, Q, n_rounds=1200, backend="dm")
    e = plateau_and_escape(dyn)
    assert e["n_escape"] is not None
    assert 50 < e["n_escape"] < 1200
    assert e["plateau_F"] > 0.9
    assert e["F_end"] < 0.6
    assert e["c_bell_end"] > 0.1
    assert e["c_bell_growth_per_round"] > 1.0


# ---------------------------------------------------------------------------
# the eps sweep along the Bell-isotropic initial line (scripts/epsilon_sweep.py)
# ---------------------------------------------------------------------------

SWEEP_EPS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
SWEEP_PROTOCOLS = ("Type3", "Type4", "4Q")

#: Fixed points at q = 0.02 as found INDEPENDENTLY in the repeated-dynamics
#: analysis (results/data/repeated_dynamics_independent/fixed_points.csv,
#: branch "main0"): target-branch fidelity and Jacobian spectral radius.
FIXED_POINT_Q002 = {
    "Type3": (0.95580256178356484048, 0.06514242096884506),
    "Type4": (0.96438111487671234217, 0.04276005297530256),
    "4Q": (0.97877575986304003999, 0.04082909309888361),
}

#: The pair of populations each protocol keeps exactly equal.
EXPECTED_INVARIANT_PAIR = {"Type3": ("Psi+", "Psi-"), "Type4": ("Psi+", "Psi-"),
                           "4Q": ("Phi-", "Psi-")}


def _residual_ratio(protocol, eps, rounds=20):
    """r_n / r_{n-1} at the last round above the round-off floor."""
    f_inf = FIXED_POINT_Q002[protocol][0]
    dyn = run_dynamics(protocol, eps, Q, n_rounds=rounds, backend="exact")
    r = np.abs(dyn.fidelity - f_inf)
    m = (r > 1e-13) & (dyn.n >= 3)
    j = int(np.where(m)[0][-1])
    return float(r[j] / r[j - 1])


def test_invariant_pair_table_matches_expectation():
    assert {k: tuple(v) for k, v in INVARIANT_PAIR.items()} == EXPECTED_INVARIANT_PAIR


@pytest.mark.parametrize("protocol", SWEEP_PROTOCOLS)
@pytest.mark.parametrize("eps", SWEEP_EPS)
def test_sweep_populations_stay_valid(protocol, eps):
    dyn = run_dynamics(protocol, eps, Q, n_rounds=50, backend="exact")
    assert np.all(np.abs(dyn.pops.sum(axis=1) - 1.0) < 1e-14)
    assert dyn.pops.min() >= 0.0
    assert dyn.fidelity[0] == pytest.approx(1 - 3 * eps / 4, abs=1e-15)


@pytest.mark.parametrize("protocol", SWEEP_PROTOCOLS)
@pytest.mark.parametrize("eps", SWEEP_EPS)
def test_each_protocol_keeps_its_own_invariant_pair(protocol, eps):
    dyn = run_dynamics(protocol, eps, Q, n_rounds=50, backend="exact")
    a, b = INVARIANT_PAIR[protocol]
    i_a, i_b = BELL_NAMES.index(a), BELL_NAMES.index(b)
    assert np.max(np.abs(dyn.pops[:, i_a] - dyn.pops[:, i_b])) < 1e-15


@pytest.mark.parametrize("protocol", SWEEP_PROTOCOLS)
def test_the_other_pairs_are_not_invariant(protocol):
    """Guard against carrying one protocol's symmetry over to another."""
    dyn = run_dynamics(protocol, 0.30, Q, n_rounds=50, backend="exact")
    keep = set(INVARIANT_PAIR[protocol])
    others = [(a, b) for a in BELL_NAMES[1:] for b in BELL_NAMES[1:]
              if a < b and {a, b} != keep]
    for a, b in others:
        i_a, i_b = BELL_NAMES.index(a), BELL_NAMES.index(b)
        assert np.max(np.abs(dyn.pops[:, i_a] - dyn.pops[:, i_b])) > 1e-6


@pytest.mark.parametrize("protocol", SWEEP_PROTOCOLS)
def test_every_epsilon_reaches_the_same_fixed_point(protocol):
    finals = [run_dynamics(protocol, e, Q, n_rounds=200, backend="exact",
                           record_every=200).pops[-1]
              for e in SWEEP_EPS]
    ref = finals[0]
    for p in finals[1:]:
        assert np.max(np.abs(p - ref)) < 1e-14
    assert ref[0] == pytest.approx(FIXED_POINT_Q002[protocol][0], abs=1e-13)


@pytest.mark.parametrize("protocol", SWEEP_PROTOCOLS)
def test_fixed_point_and_rate_match_the_independent_jacobian_analysis(protocol):
    f_expected, rho_expected = FIXED_POINT_Q002[protocol]
    f_inf = float(run_dynamics(protocol, 0.30, Q, n_rounds=200, backend="exact",
                               record_every=200).pops[-1][0])
    assert f_inf == pytest.approx(f_expected, abs=1e-13)
    # the per-round residual ratio approaches the spectral radius; it is still
    # a few percent above it by the time round-off stops the sequence
    assert _residual_ratio(protocol, 0.30) == pytest.approx(rho_expected, rel=0.05)


@needs_parent
@pytest.mark.parametrize("protocol", SWEEP_PROTOCOLS)
@pytest.mark.parametrize("eps", [0.02, 0.15, 0.30])
def test_sweep_exact_matches_full_circuit_simulation(protocol, eps):
    r = cross_check_exact_vs_dm(protocol, eps, Q, n_rounds=50)
    assert r["max_abs_pop_diff"] < 1e-12
    assert r["max_C_Bell_dm"] < 1e-20


@pytest.mark.parametrize("protocol", SWEEP_PROTOCOLS)
def test_break_even_epsilon_decides_whether_a_round_helps(protocol):
    """F_infty is the same for every eps, so whether repetition helps depends
    only on where F_0 starts relative to it."""
    f_inf = FIXED_POINT_Q002[protocol][0]
    break_even = 4 * (1 - f_inf) / 3
    for eps in SWEEP_EPS:
        assert (f_inf > 1 - 3 * eps / 4) == (eps > break_even)
    below = min(e for e in SWEEP_EPS if e < break_even)
    above = max(SWEEP_EPS)
    assert np.all(np.diff(run_dynamics(protocol, below, Q, n_rounds=20,
                                       backend="exact").fidelity) <= 1e-15)
    assert np.all(np.diff(run_dynamics(protocol, above, Q, n_rounds=20,
                                       backend="exact").fidelity) >= -1e-15)


def test_4q_success_probability_is_recorded_and_others_are_not():
    dyn = run_dynamics("4Q", 0.15, Q, n_rounds=10, backend="exact")
    assert np.isnan(dyn.weight[0])                 # undefined before any round
    assert np.all(np.isfinite(dyn.weight[1:]))
    assert np.all((dyn.weight[1:] > 0) & (dyn.weight[1:] <= 1))
    for protocol in ("Type3", "Type4"):
        other = run_dynamics(protocol, 0.15, Q, n_rounds=10, backend="exact")
        assert np.all(np.isnan(other.weight))


@needs_parent
def test_4q_success_probability_agrees_between_backends():
    """The exact map and the full circuit simulation must index P_succ the same
    way and give the same numbers."""
    a = run_dynamics("4Q", 0.15, Q, n_rounds=8, backend="dm")
    b = run_dynamics("4Q", 0.15, Q, n_rounds=8, backend="exact")
    assert np.isnan(a.weight[0]) and np.isnan(b.weight[0])
    assert a.weight[1:] == pytest.approx(b.weight[1:], abs=1e-13)


def test_4q_success_probability_converges_with_the_state():
    finals = [run_dynamics("4Q", e, Q, n_rounds=200, backend="exact",
                           record_every=200).weight[-1] for e in SWEEP_EPS]
    for p in finals[1:]:
        assert p == pytest.approx(finals[0], abs=1e-13)


def test_eps_ramp_refuses_to_invent_steps():
    from pqec_distill.bell_population_plots import eps_ramp
    assert len(eps_ramp(6)) == 6
    assert len(set(eps_ramp(6))) == 6
    with pytest.raises(ValueError, match="facet"):
        eps_ramp(9)
