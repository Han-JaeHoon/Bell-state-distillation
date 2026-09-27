"""Guards for the Bell-population dynamics analysis.

The SWAP-test protocols (Type 3/4/5) need the parent research repository; those
tests skip cleanly when it is not present, and the rest still run.
"""

from __future__ import annotations

import numpy as np
import pytest

from pqec_distill.bell_population_dynamics import (
    BELL_NAMES, PROTOCOLS, bell_offdiagonal_c, bell_populations,
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
