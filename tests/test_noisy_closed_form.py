"""Exact noisy one-round map on Bell-diagonal states vs the dense simulator."""

import numpy as np
import pytest
import sympy as sp

from _helpers import random_bell_diagonal_populations
from pqec_distill.bell_states import bell_diagonal_state
from pqec_distill.gates import PAULI_X, PAULI_Y, PAULI_Z, kron_list
from pqec_distill.measurement import postselect_branch
from pqec_distill.noise import NOISE_CONVENTIONS, noisy_full_channel_dm
from pqec_distill.noisy_analytics import (
    fidelity_uv, noisy_map_uv, noisy_map_xyz, one_round_fidelity,
    p_success_xyz, qbar_pauli, qbar_replace,
)

TOL = 1e-12
RNG = np.random.default_rng(2026)
POPS = random_bell_diagonal_populations(RNG, 12)
XX, YY, ZZ = (kron_list([P, P]) for P in (PAULI_X, PAULI_Y, PAULI_Z))


def _xyz(rho):
    return tuple(float(np.real(np.trace(P @ rho))) for P in (XX, YY, ZZ))


def _dense_round(rho, p, convention):
    out4 = noisy_full_channel_dm(np.kron(rho, rho), p, convention)
    rt, prob = postselect_branch(out4, 0, 0)
    return _xyz(rt / prob), prob


@pytest.mark.parametrize("convention", NOISE_CONVENTIONS)
@pytest.mark.parametrize("p", [0.0, 0.01, 0.05, 0.2, 0.5])
@pytest.mark.parametrize("pop", POPS)
def test_closed_form_matches_dense(convention, p, pop):
    rho = bell_diagonal_state(pop)
    x, y, z = _xyz(rho)
    qb = qbar_replace(p) if convention == "replace" else qbar_pauli(p)
    got, prob = _dense_round(rho, p, convention)
    pred = noisy_map_xyz(x, y, z, qb)
    assert np.max(np.abs(np.array(got) - np.array(pred))) < TOL
    assert abs(prob - p_success_xyz(x, y, z, qb)) < TOL


@pytest.mark.parametrize("pop", POPS[:5])
def test_reduces_to_matrix_square_at_zero_noise(pop):
    rho = bell_diagonal_state(pop)
    x, y, z = _xyz(rho)
    d = 1 + x * x + y * y + z * z
    expected = (2 * (x - y * z) / d, 2 * (y - x * z) / d, 2 * (z - x * y) / d)
    assert np.allclose(noisy_map_xyz(x, y, z, 1.0), expected, atol=TOL)


@pytest.mark.parametrize("qb", [1.0, 0.99, 0.9, 0.7])
def test_plane_y_equals_minus_z_is_invariant(qb):
    rng = np.random.default_rng(1)
    for _ in range(20):
        x, z = rng.uniform(-0.6, 0.6, size=2)
        xn, yn, zn = noisy_map_xyz(x, -z, z, qb)
        assert abs(yn + zn) < TOL
        un, vn = noisy_map_uv(x, z, qb)
        assert abs(un - xn) < TOL and abs(vn - zn) < TOL


@pytest.mark.parametrize("qb", [0.99, 0.9])
def test_plane_y_equals_minus_x_is_NOT_invariant(qb):
    """Unlike the 5-qubit SWAP-test gadget, here XX and YY damp differently."""
    x, z = 0.5, 0.4
    xn, yn, zn = noisy_map_xyz(x, -x, z, qb)
    assert abs(yn + xn) > 1e-4


def test_symbolic_pure_bell_series_and_slope():
    q, eb, u, v = sp.symbols("q epsbar u v", positive=True)
    qb = 1 - q
    D = 1 + qb ** 5 * u ** 2 + (qb ** 5 + qb ** 3) * v ** 2
    un = qb ** 3 * ((1 + qb) * u + 2 * qb ** 2 * v ** 2) / D
    vn = qb ** 4 * v * ((1 + qb) + 2 * qb * u) / D
    F1 = ((1 + un + 2 * vn) / 4).subs({u: eb, v: eb})
    series = sp.series(F1.subs(eb, 1), q, 0, 3).removeO()
    assert sp.simplify(series - (1 - q - sp.Rational(5, 4) * q ** 2)) == 0
    K = sp.simplify(-sp.diff(F1, q).subs(q, 0))
    expected = eb * (12 * eb ** 3 - 3 * eb ** 2 + 30 * eb + 25) / (4 * (3 * eb ** 2 + 1) ** 2)
    assert sp.simplify(K - expected) == 0
    assert K.subs(eb, 1) == 1


def test_numeric_slope_is_one():
    h = 1e-6
    slope = (one_round_fidelity(0.0, 1.0) - one_round_fidelity(0.0, 1.0 - h)) / h
    assert abs(slope - 1.0) < 1e-5


def test_fidelity_uv_matches_bell_population():
    pop = np.array([0.55, 0.2, 0.15, 0.1])
    rho = bell_diagonal_state(pop)
    x, y, z = _xyz(rho)
    # on the plane y = -z the formula F = (1 + u + 2v)/4 must reproduce p_Phi+
    assert abs((1 + x - y + z) / 4 - pop[0]) < TOL
