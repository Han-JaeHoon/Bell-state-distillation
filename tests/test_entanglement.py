"""Concurrence / negativity, and the re-entangling window."""

import numpy as np
import pytest
import sympy as sp

from _helpers import random_bell_diagonal_populations
from pqec_distill.analytics import (
    EPS_INPUT_SEPARABLE, EPS_OUTPUT_SEPARABLE, input_concurrence_isotropic,
    isotropic_populations, output_concurrence_isotropic,
)
from pqec_distill.bell_states import bell_diagonal_state, bell_state
from pqec_distill.entanglement import (
    bell_diagonal_concurrence, concurrence, negativity,
)
from pqec_distill.measurement import run_circuit_success

TOL = 1e-10
RNG = np.random.default_rng(12345)


def test_bell_states_are_maximally_entangled():
    for (a, b) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        v = bell_state(a, b).reshape(4, 1)
        rho = v @ v.conj().T
        assert abs(concurrence(rho) - 1.0) < TOL
        assert abs(negativity(rho) - 0.5) < TOL


def test_product_state_has_zero_entanglement():
    rho = np.kron(np.diag([1.0, 0.0]), np.diag([0.5, 0.5])).astype(complex)
    assert concurrence(rho) < TOL
    assert negativity(rho) < TOL


@pytest.mark.parametrize("p", random_bell_diagonal_populations(RNG, 40))
def test_bell_diagonal_concurrence_formula(p):
    rho = bell_diagonal_state(p)
    assert abs(concurrence(rho) - bell_diagonal_concurrence(p)) < TOL


def test_input_separability_boundary():
    assert abs(EPS_INPUT_SEPARABLE - 2.0 / 3.0) < 1e-15
    assert input_concurrence_isotropic(EPS_INPUT_SEPARABLE - 1e-9) > 0
    assert input_concurrence_isotropic(EPS_INPUT_SEPARABLE + 1e-9) == 0


def test_output_separability_boundary_symbolic():
    """F_out(eps) = 1/2 at eps = 2 - 2 sqrt(3)/3."""
    e = sp.symbols("epsilon", positive=True)
    F = 1 - sp.Rational(3, 4) * e
    q = e / 4
    roots = sp.solve(sp.Eq(F ** 2 / (F ** 2 + 3 * q ** 2), sp.Rational(1, 2)), e)
    physical = [r for r in roots if 0 < float(r) <= 1]
    assert len(physical) == 1
    assert abs(float(physical[0]) - EPS_OUTPUT_SEPARABLE) < 1e-14


@pytest.mark.parametrize("eps", [0.67, 0.70, 0.75, 0.80, 0.84])
def test_re_entangling_window(eps):
    """Separable input, entangled postselected output: 2/3 <= eps < 0.84530."""
    assert EPS_INPUT_SEPARABLE <= eps < EPS_OUTPUT_SEPARABLE
    rho_in = bell_diagonal_state(isotropic_populations(eps))
    rho_out, _ = run_circuit_success(rho_in)
    assert concurrence(rho_in) < TOL
    assert concurrence(rho_out) > 1e-6


@pytest.mark.parametrize("eps", [0.86, 0.9, 0.95, 1.0])
def test_no_output_entanglement_above_threshold(eps):
    rho_in = bell_diagonal_state(isotropic_populations(eps))
    rho_out, _ = run_circuit_success(rho_in)
    assert concurrence(rho_out) < TOL


@pytest.mark.parametrize("eps", [0.0, 0.1, 0.4, 0.7, 0.9])
def test_circuit_output_concurrence_matches_analytic(eps):
    rho_in = bell_diagonal_state(isotropic_populations(eps))
    rho_out, _ = run_circuit_success(rho_in)
    assert abs(concurrence(rho_out) - float(output_concurrence_isotropic(eps))) < TOL
