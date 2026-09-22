"""Guards for the independently derived repeated-dynamics results.

Everything here is checked against ``pqec_distill.repeated_dynamics``, which
in turn uses only the verified one-round maps.  The reduced maps and the
closed-form eigenvalues are re-stated here in a couple of lines each, so a
regression in either place shows up as a disagreement rather than as two
copies of the same mistake.

The excluded modules (noisy_analytics, repeated_noisy, swap_test_reference)
are deliberately not imported.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pqec_distill.analytic_exact_map import pauli_from_pops, pops_from_pauli
from pqec_distill.repeated_dynamics import (
    CIRCUITS, classify_trajectory, isotropic_pops, one_round,
    success_probability, trajectory,
)

DATA = Path(__file__).resolve().parents[1] / "results" / "data" / "repeated_dynamics_independent"
K = {"Type3": 4, "Type4": 2}
QS = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]


# ---------------------------------------------------------------------------
# the reduced maps, restated
# ---------------------------------------------------------------------------

def reduced(s, t, q, circuit):
    qb = 1.0 - q
    if circuit == "4Q":
        d = 1 + qb**5 * s**2 + (qb**5 + qb**3) * t**2
        return (qb**3 * ((1 + qb) * s + 2 * qb**2 * t**2) / d,
                qb**4 * t * ((1 + qb) + 2 * qb * s) / d)
    k = K[circuit]
    d = 1 + qb**k * (2 * s**2 + t**2)
    return (2 * qb**(k + 2) * s * (1 + t) / d,
            qb**(k + 1) * (1 + qb) * (s**2 + t) / d)


def to_reduced(p):
    """coord1 = x, coord2 = z in BOTH charts."""
    x, _y, z = pauli_from_pops(p)
    return x, z


def from_reduced(s, t, circuit):
    return pops_from_pauli(s, -s, t) if circuit != "4Q" else pops_from_pauli(s, -t, t)


# ---------------------------------------------------------------------------
# initial state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("eps", [0.0, 0.1, 0.37, 0.5, 0.9, 1.0])
def test_isotropic_pauli_coordinates(eps):
    x, y, z = pauli_from_pops(isotropic_pops(eps))
    t = 1.0 - eps
    assert (x, y, z) == pytest.approx((t, -t, t), abs=1e-15)


def test_isotropic_lies_on_both_planes():
    for eps in np.linspace(0.0, 1.0, 21):
        x, y, z = pauli_from_pops(isotropic_pops(eps))
        assert abs(x + y) < 1e-15
        assert abs(y + z) < 1e-15


# ---------------------------------------------------------------------------
# invariance of the planes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("circuit", CIRCUITS)
@pytest.mark.parametrize("q", QS)
def test_plane_is_invariant(circuit, q):
    """The circuit's own plane is preserved exactly, for on-plane inputs that
    are NOT Bell-isotropic."""
    rng = np.random.default_rng(20260922)
    for _ in range(40):
        s, t = rng.uniform(-0.9, 0.9, 2)
        p = from_reduced(s, t, circuit)
        if p.min() < 0:
            continue
        out = one_round(p, q, circuit)
        x, y, z = pauli_from_pops(out)
        combo = (x + y) if circuit != "4Q" else (y + z)
        assert abs(combo) < 1e-14


@pytest.mark.parametrize("circuit", CIRCUITS)
def test_other_plane_is_not_invariant(circuit):
    """Guard against over-generalising: the OTHER combination, which also
    vanishes on the Bell-isotropic line, is not preserved."""
    worst = 0.0
    for eps in (0.1, 0.3, 0.5, 0.7):
        p = isotropic_pops(eps)
        for _ in range(30):
            p = one_round(p, 0.05, circuit)
            x, y, z = pauli_from_pops(p)
            worst = max(worst, abs(y + z) if circuit != "4Q" else abs(x + y))
    assert worst > 1e-3


# ---------------------------------------------------------------------------
# reduced map == full map on the plane
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("circuit", CIRCUITS)
@pytest.mark.parametrize("q", QS)
def test_reduced_map_matches_full_map(circuit, q):
    rng = np.random.default_rng(7)
    for _ in range(60):
        s, t = rng.uniform(-0.9, 0.9, 2)
        p = from_reduced(s, t, circuit)
        if p.min() < 0:
            continue
        got = to_reduced(one_round(p, q, circuit))
        assert got == pytest.approx(reduced(s, t, q, circuit), abs=1e-14)


@pytest.mark.parametrize("circuit", CIRCUITS)
def test_fidelity_from_reduced_coordinates(circuit):
    rng = np.random.default_rng(11)
    for _ in range(40):
        s, t = rng.uniform(-0.5, 0.5, 2)
        p = from_reduced(s, t, circuit)
        expected = (1 + s + 2 * t) / 4 if circuit == "4Q" else (1 + 2 * s + t) / 4
        assert p[0] == pytest.approx(expected, abs=1e-15)


# ---------------------------------------------------------------------------
# fixed-point equations
# ---------------------------------------------------------------------------

def offplane(q, circuit):
    qb = 1.0 - q
    if circuit == "4Q":
        sq = (qb**3 * (1 + qb) - 1) / qb**5
        return (np.sqrt(sq), 0.0) if sq > 0 else None
    k = K[circuit]
    sq = (qb**(k + 1) * (1 + qb) - 1) / qb**k
    return (0.0, np.sqrt(sq)) if sq > 0 else None


def main_roots(q, circuit):
    qb = 1.0 - q
    if circuit == "4Q":
        A = 2 * qb**3 * (2 * qb**2 + 1)
        B = -qb * (3 * qb**4 + 1)
        C = -2 * (qb**5 + qb**4 - 1)
    else:
        k = K[circuit]
        A = qb**k * (5 * qb + 1)
        B = -2 * qb**k * (qb**3 + qb**2 - qb + 1)
        C = -(qb + 1) * (2 * qb**(k + 2) - 1)
    disc = B * B - 4 * A * C
    if disc < 0:
        return []
    out = []
    for r in ((-B + np.sqrt(disc)) / (2 * A), (-B - np.sqrt(disc)) / (2 * A)):
        if circuit == "4Q":
            sq = r * (2 * r * qb**2 + qb**2 - 1) / (2 * qb**2)
            if sq >= 0:
                out.append((r, np.sqrt(sq)))
        else:
            sq = r * (2 * qb * r + qb - 1) / (1 + qb)
            if sq >= 0:
                out.append((np.sqrt(sq), r))
    return out


@pytest.mark.parametrize("circuit", CIRCUITS)
@pytest.mark.parametrize("q", [0.001, 0.01, 0.05, 0.1, 0.12, 0.15, 0.17, 0.18])
def test_branch_formulas_give_fixed_points(circuit, q):
    cands = [(0.0, 0.0)]
    off = offplane(q, circuit)
    if off is not None:
        cands.append(off)
    cands += main_roots(q, circuit)
    for s, t in cands:
        sp, tp = reduced(s, t, q, circuit)
        assert max(abs(sp - s), abs(tp - t)) < 1e-12


@pytest.mark.parametrize("circuit", CIRCUITS)
@pytest.mark.parametrize("q", [0.001, 0.05, 0.1, 0.15, 0.17])
def test_offplane_eigenvalues_closed_form(circuit, q):
    """Triangular Jacobian: lambda_parallel = (2-M)/M, and M is also an
    eigenvalue of the maximally mixed fixed point."""
    fp = offplane(q, circuit)
    if fp is None:
        pytest.skip("branch does not exist at this q")
    qb = 1.0 - q
    k = 3 if circuit == "4Q" else K[circuit] + 1
    M = qb**k * (1 + qb)
    lam_par = (2 - M) / M
    lam_perp = (qb + 2 * qb**5 * fp[0] / M) if circuit == "4Q" \
        else 2 * qb * (1 + fp[1]) / (1 + qb)

    h = 1e-6
    j = np.empty((2, 2))
    for col, d in enumerate(((h, 0.0), (0.0, h))):
        pp = reduced(fp[0] + d[0], fp[1] + d[1], q, circuit)
        pm = reduced(fp[0] - d[0], fp[1] - d[1], q, circuit)
        j[:, col] = [(a - b) / (2 * h) for a, b in zip(pp, pm)]
    num = sorted(abs(np.linalg.eigvals(j)))
    assert num == pytest.approx(sorted((abs(lam_par), abs(lam_perp))), abs=1e-7)

    origin = (qb**3 * (1 + qb), qb**4 * (1 + qb)) if circuit == "4Q" \
        else (2 * qb**(K[circuit] + 2), qb**(K[circuit] + 1) * (1 + qb))
    assert min(abs(e - M) for e in origin) < 1e-15


# ---------------------------------------------------------------------------
# trajectories and success probability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("circuit", CIRCUITS)
def test_trajectory_records_required_fields(circuit):
    tr = trajectory(circuit, 0.2, 0.02, max_rounds=30, tol=1e-15)
    row = tr["rows"][0]
    for key in ("n", "p_PhiP", "p_PhiM", "p_PsiP", "p_PsiM", "x", "y", "z",
                "fidelity", "step_inf", "ratio_PhiM", "ratio_PsiP", "ratio_PsiM"):
        assert key in row
    assert ("P_succ" in row) is (circuit == "4Q")
    for r in tr["rows"]:
        p = np.array([r["p_PhiP"], r["p_PhiM"], r["p_PsiP"], r["p_PsiM"]])
        assert p.sum() == pytest.approx(1.0, abs=1e-14)
        assert p.min() >= -1e-15


def test_success_probability_only_for_4q():
    p = isotropic_pops(0.2)
    assert success_probability(p, 0.05, "Type3") is None
    assert success_probability(p, 0.05, "Type4") is None
    ps = success_probability(p, 0.05, "4Q")
    assert 0.0 < ps <= 1.0


@pytest.mark.parametrize("circuit", CIRCUITS)
def test_no_short_cycles_at_moderate_noise(circuit):
    for eps in (0.1, 0.5, 0.9):
        for q in (0.02, 0.1, 0.17):
            c = classify_trajectory(circuit, eps, q, max_rounds=800)
            assert c["cycle_period"] in (None, 1)


def test_maximally_mixed_is_an_exact_fixed_point():
    p = isotropic_pops(1.0)
    for circuit in CIRCUITS:
        for q in QS:
            assert one_round(p, q, circuit) == pytest.approx(p, abs=1e-16)


# ---------------------------------------------------------------------------
# frozen files stay consistent with the map
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (DATA / "fixed_points.csv").exists(),
                    reason="frozen results not present")
def test_frozen_fixed_points_are_fixed_points():
    rows = list(csv.DictReader(open(DATA / "fixed_points.csv")))
    assert len(rows) > 100
    for r in rows:
        s, t, q = float(r["coord1"]), float(r["coord2"]), float(r["q"])
        sp, tp = reduced(s, t, q, r["circuit"])
        assert max(abs(sp - s), abs(tp - t)) < 1e-10


@pytest.mark.skipif(not (DATA / "endpoint_verification.csv").exists(),
                    reason="frozen results not present")
def test_frozen_endpoints_are_valid_states():
    rows = list(csv.DictReader(open(DATA / "endpoint_verification.csv")))
    assert rows
    for r in rows:
        p = np.array([float(r[k]) for k in ("p_PhiP", "p_PhiM", "p_PsiP", "p_PsiM")])
        assert p.sum() == pytest.approx(1.0, abs=1e-13)
        assert p.min() >= -1e-15
        resid = np.max(np.abs(one_round(p, float(r["q"]), r["circuit"]) - p))
        assert resid < 1e-9


@pytest.mark.skipif(not (DATA / "critical_points.json").exists(),
                    reason="frozen results not present")
def test_frozen_critical_points_satisfy_their_conditions():
    crit = json.loads((DATA / "critical_points.json").read_text())
    for circuit, d in crit.items():
        qb = 1.0 - float(d["q_offplane_branch_vanishes"])
        k = 3 if circuit == "4Q" else K[circuit] + 1
        assert qb**k * (1 + qb) == pytest.approx(1.0, abs=1e-12)
        # the target branch really does disappear just above the saddle node
        q_sn = float(d["q_saddle_node_discriminant_zero"])
        assert main_roots(q_sn - 1e-6, circuit)
        assert not main_roots(q_sn + 1e-6, circuit)
