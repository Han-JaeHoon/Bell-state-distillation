"""One-round operational CNOT-noise threshold p*(eps)."""

import math

import numpy as np
import pytest

from pqec_distill.analytics import F_of_eps
from pqec_distill.noisy_analytics import one_round_fidelity, qbar_replace, threshold_p

# break-even values from the earlier 41-point grid sweep (linear interpolation)
GRID_BREAK_EVEN = {0.05: 0.033697, 0.10: 0.061156, 0.20: 0.102816,
                   0.30: 0.132114, 0.50: 0.166546}


@pytest.mark.parametrize("eps,p_grid", GRID_BREAK_EVEN.items())
def test_matches_grid_break_even(eps, p_grid):
    assert abs(threshold_p(eps) - p_grid) < 2e-5


@pytest.mark.parametrize("eps", [0.01, 0.1, 0.3, 0.5, 0.7, 0.9])
def test_root_and_sign_change(eps):
    p = threshold_p(eps)
    f_in = float(F_of_eps(eps))
    assert abs(one_round_fidelity(eps, qbar_replace(p)) - f_in) < 1e-12
    assert one_round_fidelity(eps, qbar_replace(p - 1e-4)) > f_in
    assert one_round_fidelity(eps, qbar_replace(p + 1e-4)) < f_in


@pytest.mark.parametrize("eps", [0.1, 0.5])
def test_pauli_convention_is_15_over_16(eps):
    assert abs(threshold_p(eps, "pauli") - 15 / 16 * threshold_p(eps)) < 1e-14


def test_threshold_is_not_monotone_in_eps():
    """p* peaks near eps ~ 2/3 and decreases for noisier inputs."""
    assert threshold_p(2 / 3) > threshold_p(0.5)
    assert threshold_p(2 / 3) > threshold_p(0.9)


def test_pure_bell_input_has_no_threshold():
    assert math.isnan(threshold_p(0.0))


def test_threshold_grows_with_input_noise_at_small_eps():
    vals = [threshold_p(e) for e in (0.01, 0.05, 0.1, 0.2, 0.3)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
