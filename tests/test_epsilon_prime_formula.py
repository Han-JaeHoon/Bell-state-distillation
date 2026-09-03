"""eps' = eps^2 / (4 - 6 eps + 3 eps^2), including a SymPy symbolic check."""

import numpy as np
import pytest
import sympy as sp

from pqec_distill.analytics import eps_prime, isotropic_populations
from pqec_distill.bell_states import bell_diagonal_state, bell_populations
from pqec_distill.measurement import run_circuit_success

TOL = 1e-12


@pytest.mark.parametrize("eps", [0.0, 0.01, 0.1, 0.4, 2.0 / 3.0, 0.7, 0.8, 0.9, 1.0])
def test_circuit_reproduces_eps_prime(eps):
    rho = bell_diagonal_state(isotropic_populations(eps))
    rho_out, _ = run_circuit_success(rho)
    # eps' read off from the output populations: p_Phi+ = 1 - 3 eps'/4
    eps_out = (1.0 - bell_populations(rho_out)[0]) * 4.0 / 3.0
    assert abs(eps_out - float(eps_prime(eps))) < TOL


def test_symbolic_eps_prime_from_F_and_q():
    """Symbolically derive eps' from F^2/(F^2+3q^2) and compare."""
    e = sp.symbols("epsilon", nonnegative=True)
    F = 1 - sp.Rational(3, 4) * e
    q = e / 4
    F_out = F ** 2 / (F ** 2 + 3 * q ** 2)
    eps_out = sp.simplify((1 - F_out) * sp.Rational(4, 3))
    expected = e ** 2 / (4 - 6 * e + 3 * e ** 2)
    assert sp.simplify(eps_out - expected) == 0


def test_symbolic_success_probability():
    e = sp.symbols("epsilon", nonnegative=True)
    F = 1 - sp.Rational(3, 4) * e
    q = e / 4
    expected = (4 - 6 * e + 3 * e ** 2) / 4
    assert sp.simplify(F ** 2 + 3 * q ** 2 - expected) == 0


def test_symbolic_small_eps_expansion():
    """eps' = eps^2/4 + O(eps^3)."""
    e = sp.symbols("epsilon", positive=True)
    expr = e ** 2 / (4 - 6 * e + 3 * e ** 2)
    series = sp.series(expr, e, 0, 4).removeO()
    assert sp.simplify(series.coeff(e, 2) - sp.Rational(1, 4)) == 0
    assert sp.simplify(series.coeff(e, 0)) == 0
    assert sp.simplify(series.coeff(e, 1)) == 0
    # the eps^3 coefficient is 3/8, so the correction is genuinely O(eps^3)
    assert sp.simplify(series.coeff(e, 3) - sp.Rational(3, 8)) == 0


def test_numeric_small_eps_ratio():
    for eps in (1e-3, 1e-4, 1e-5):
        ratio = float(eps_prime(eps)) / (eps ** 2 / 4.0)
        assert abs(ratio - 1.0) < 5e-3


def test_eps_prime_fixed_points():
    assert abs(float(eps_prime(0.0)) - 0.0) < 1e-15
    assert abs(float(eps_prime(1.0)) - 1.0) < 1e-15


@pytest.mark.parametrize("eps", [0.05, 0.2, 0.5, 0.66, 0.9])
def test_purification_strictly_improves_below_one(eps):
    assert float(eps_prime(eps)) < eps
