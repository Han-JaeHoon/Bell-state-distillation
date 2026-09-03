"""Repeated noisy purification: fixed points, stability, off-Bell decay."""

import numpy as np
import pytest

from _helpers import random_density_matrix
from pqec_distill.analytics import isotropic_populations
from pqec_distill.bell_states import bell_diagonal_state
from pqec_distill.gates import I2, PAULI_Z, kron_list
from pqec_distill.noisy_analytics import (
    asymptotic_threshold_p, bistability_onset_p, entanglement_limit_p,
    fidelity_uv, fixed_point_branch, fixed_point_uv, jacobian_uv, noisy_map_uv,
    saddle_node_p, threshold_p, v0_family_limit_p, v0_fixed_point_u,
    v0_transverse_eigenvalue,
)
from pqec_distill.repeated_noisy import (
    PAULI_LABELS_2Q, effective_map, fixed_point_dense, full_jacobian, iterate,
    off_bell_projection_norm, to_pauli_coords,
)

P_VALUES = [0.001, 0.01, 0.05, 0.1, 0.15]
IXX, IYY, IZZ = (PAULI_LABELS_2Q.index(k) for k in ("XX", "YY", "ZZ"))


def _fidelity(rho):
    r = to_pauli_coords(rho)
    return (1 + r[IXX] - r[IYY] + r[IZZ]) / 4


def _dense_fp(p):
    rho0 = bell_diagonal_state(isotropic_populations(0.1))
    rho, n, res = fixed_point_dense(rho0, p, "replace")
    assert res < 1e-13
    return rho


@pytest.mark.parametrize("p", P_VALUES)
def test_dense_fixed_point_matches_closed_form(p):
    rho = _dense_fp(p)
    u, v = fixed_point_branch(1 - p)
    r = to_pauli_coords(rho)
    assert abs(r[IXX] - u) < 1e-10
    assert abs(r[IZZ] - v) < 1e-10
    assert abs(r[IYY] + v) < 1e-10
    assert abs(_fidelity(rho) - fidelity_uv(u, v)) < 1e-10


def test_weak_noise_expansion_of_fixed_point():
    """u* = 1 - q - 13/4 q^2, v* = 1 - 3/2 q - 33/8 q^2, F* = 1 - q - 23/8 q^2."""
    for q in (1e-3, 3e-4):
        u, v = fixed_point_branch(1 - q)
        assert abs((1 - u - q) / q ** 2 - 13 / 4) < 0.05
        assert abs((1 - v - 1.5 * q) / q ** 2 - 33 / 8) < 0.05
        assert abs((1 - fidelity_uv(u, v) - q) / q ** 2 - 23 / 8) < 0.05


@pytest.mark.parametrize("p", P_VALUES)
def test_fixed_point_is_a_full_state_attractor(p):
    """Spectral radius of the exact 15-dim Jacobian is < 1 and EQUALS the
    Bell-sector value: no unstable off-Bell direction (unlike the 5-qubit
    SWAP-test fixed points of the parent project)."""
    rho = _dense_fp(p)
    lam15 = np.abs(np.linalg.eigvals(full_jacobian(rho, p, "replace")))
    u, v = fixed_point_branch(1 - p)
    lam2 = np.abs(np.linalg.eigvals(jacobian_uv(u, v, 1 - p)))
    assert lam15.max() < 1.0
    assert abs(lam15.max() - lam2.max()) < 1e-6


@pytest.mark.parametrize("p", [0.0, 0.01, 0.1])
def test_off_bell_directions_are_superattracting(p):
    """All 12 off-Bell Jacobian eigenvalues vanish: the map is a Bell-basis
    Schur square, so off-Bell coherences enter only at second order."""
    rho = _dense_fp(p) if p > 0 else bell_diagonal_state([1, 0, 0, 0])
    lam = np.sort(np.abs(np.linalg.eigvals(full_jacobian(rho, p, "replace"))))
    assert np.all(lam[:12] < 1e-12)


@pytest.mark.parametrize("p", [0.01, 0.05])
def test_seeded_off_bell_perturbation_decays_quadratically(p):
    rho_star = _dense_fp(p)
    seed = (kron_list([PAULI_Z, I2]) + kron_list([I2, PAULI_Z])) / np.sqrt(2)
    rho = rho_star + 1e-2 / 4 * seed
    d0 = off_bell_projection_norm(rho)
    rho, _ = effective_map(rho, p, "replace")
    d1 = off_bell_projection_norm(rho)
    rho, _ = effective_map(rho, p, "replace")
    d2 = off_bell_projection_norm(rho)
    assert d1 < 1e-2 * d0          # quadratic, not linear, contraction
    assert d2 < 1e-2 * d1


@pytest.mark.parametrize("eps0", [0.05, 0.3, 0.5, 0.8])
def test_bell_isotropic_inputs_converge_to_the_same_fixed_point(eps0):
    p = 0.05
    ref = _dense_fp(p)
    rho = fixed_point_dense(bell_diagonal_state(isotropic_populations(eps0)), p, "replace")[0]
    assert np.linalg.norm(rho - ref, "fro") < 1e-11


def test_four_bell_attractors_are_symmetric_images():
    """Each Bell state carries its own attracting fixed point; the four are
    related by the Pauli symmetry of the noise model and share F* and C*."""
    from pqec_distill.bell_states import bell_populations
    p = 0.05
    u, v = fixed_point_branch(1 - p)
    f_star = fidelity_uv(u, v)
    for k in range(4):
        pop = np.full(4, 0.1 / 4)
        pop[k] = 1 - 0.075
        rho, _, res = fixed_point_dense(bell_diagonal_state(pop), p, "replace")
        assert res < 1e-13
        pops = bell_populations(rho)
        assert int(np.argmax(pops)) == k
        assert abs(pops.max() - f_star) < 1e-10
        assert off_bell_projection_norm(rho) < 1e-12


def test_generic_non_bell_diagonal_inputs_converge_to_a_bell_attractor():
    """A generic input is purified toward whichever Bell state dominates it:
    the limit is Bell diagonal with maximal population F* (one of the four
    symmetric attractors), not necessarily the Phi+ one."""
    from pqec_distill.bell_states import bell_populations
    p = 0.05
    u, v = fixed_point_branch(1 - p)
    f_star = fidelity_uv(u, v)
    rng = np.random.default_rng(99)
    for _ in range(6):
        rho = iterate(random_density_matrix(rng), p, "replace", 80)[-1]
        assert off_bell_projection_norm(rho) < 1e-9
        assert abs(bell_populations(rho).max() - f_star) < 1e-9


def test_saddle_node_and_entanglement_limit():
    p_sn = saddle_node_p()
    p_ent = entanglement_limit_p()
    assert abs(p_sn - 0.180669725979) < 1e-9
    assert abs(p_ent - 0.179815332614) < 1e-9
    assert p_ent < p_sn
    assert fixed_point_branch(1 - p_sn - 1e-6) is None
    assert fixed_point_branch(1 - p_sn + 1e-6) is not None
    u, v = fixed_point_branch(1 - (p_ent + p_sn) / 2)
    assert fidelity_uv(u, v) < 0.5     # exists but already separable


@pytest.mark.parametrize("p", [0.19, 0.25])
def test_beyond_v0_limit_converges_to_maximally_mixed(p):
    """For p > p0 = 0.18083 the only attractor is I/4."""
    assert p > v0_family_limit_p()
    rho = iterate(bell_diagonal_state(isotropic_populations(0.05)), p, "replace", 400)[-1]
    assert _fidelity(rho) < 0.5
    assert np.linalg.norm(rho - np.eye(4) / 4, "fro") < 1e-6


def test_three_regime_structure_above_the_saddle_node():
    """p_B < p_SN < p0, with the v = 0 point  1/4(II + u0 XX)  the attractor in
    the narrow window p_SN < p < p0 (found in an external review of the
    analytic calculation; see report K.5)."""
    p_b, p_sn, p0 = bistability_onset_p(), saddle_node_p(), v0_family_limit_p()
    assert abs(p_b - 0.175833265266489) < 1e-12
    assert abs(p_sn - 0.180669725978882) < 1e-12
    assert abs(p0 - 0.180827486603836) < 1e-12
    assert p_b < p_sn < p0
    # the v = 0 point is transversally unstable below p_B and stable above
    assert v0_transverse_eigenvalue(1 - 0.17) > 1.0
    assert v0_transverse_eigenvalue(1 - 0.178) < 1.0
    # in the window it is an exact fixed point of the FULL noisy circuit
    from pqec_distill.gates import PAULI_X, kron_list
    p = 0.1807
    u0 = v0_fixed_point_u(1 - p)
    assert abs(u0 - 0.0381424) < 1e-6
    rho = (np.eye(4) + u0 * kron_list([PAULI_X, PAULI_X])) / 4
    out, _ = effective_map(rho, p, "replace")
    assert np.linalg.norm(out - rho, "fro") < 1e-15
    # ...and the Phi+ branch is gone there
    assert fixed_point_branch(1 - p) is None


def test_window_converges_to_v0_point_slowly():
    """Convergence in the window is governed by the saddle-node ghost, so it
    takes thousands of rounds; iterate the exact reduced map."""
    p = 0.1807
    u0 = v0_fixed_point_u(1 - p)
    u = v = 0.9
    for _ in range(60000):
        u, v = noisy_map_uv(u, v, 1 - p)
    assert abs(v) < 1e-6 and abs(u - u0) < 1e-6


def test_bistability_basin_at_p_0p18():
    """For p_B < p < p_SN both attractors coexist; at p = 0.18 the basin
    boundary on the isotropic line is near t_c = 0.2226, below the entangled
    inputs (t > 1/3), which all reach the Phi+ branch."""
    p = 0.18
    u_star, v_star = fixed_point_branch(1 - p)

    def endpoint(t):
        u = v = t
        for _ in range(3000):
            u, v = noisy_map_uv(u, v, 1 - p)
        return u, v
    u, v = endpoint(0.20)
    assert abs(v) < 1e-9 and abs(u - v0_fixed_point_u(1 - p)) < 1e-6
    for t in (0.25, 1 / 3, 0.5, 0.9):
        u, v = endpoint(t)
        assert abs(u - u_star) < 1e-6 and abs(v - v_star) < 1e-6


def test_asymptotic_threshold_slightly_above_one_round_threshold():
    """eps0 = 0.1: one-round break-even at p = 0.061160, but the fixed point
    still beats the input up to p = 0.061550."""
    p1, pinf = threshold_p(0.1), asymptotic_threshold_p(0.1)
    assert abs(p1 - 0.0611603568) < 1e-9
    assert abs(pinf - 0.0615498) < 1e-6
    assert pinf > p1
    # at exactly the one-round threshold the first round is break-even and
    # later rounds improve a little
    u = v = 0.9
    fs = [fidelity_uv(u, v)]
    for _ in range(3):
        u, v = noisy_map_uv(u, v, 1 - p1)
        fs.append(fidelity_uv(u, v))
    assert abs(fs[1] - 0.925) < 1e-9
    assert fs[3] > fs[2] > fs[1]


def test_low_fidelity_branch_is_unstable_in_bell_sector():
    for p in (0.05, 0.1):
        fps = fixed_point_uv(1 - p)
        assert len(fps) == 2
        lam = np.abs(np.linalg.eigvals(jacobian_uv(*fps[1], 1 - p)))
        assert lam.max() > 1.0
