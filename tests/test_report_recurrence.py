"""The report note's recurrence must agree with the repository's verified map.

``scripts/report_recurrence.py`` implements the recurrences exactly as written
in the report note, from the note alone.  These tests are the guard that keeps
the note and the code from drifting apart, on generic Bell-diagonal states --
not only on the Bell-isotropic line.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from report_recurrence import (  # noqa: E402
    BELL_ORDER, NOTE_PROTOCOLS, isotropic_pops, note_fidelity_in_plane,
    note_step, note_success, pops_to_xyz, xyz_to_pops,
)

from pqec_distill.analytic_exact_map import (  # noqa: E402
    analytic_map_4q, analytic_map_type3, analytic_map_type4, analytic_success_4q,
)
from pqec_distill.bell_population_dynamics import INVARIANT_PAIR  # noqa: E402

MAP = {"Type3": analytic_map_type3, "Type4": analytic_map_type4,
       "4Q": analytic_map_4q}
QS = (0.0, 0.005, 0.02, 0.08, 0.2, 0.5)


def random_states(n=120, seed=20260929):
    return np.random.default_rng(seed).dirichlet(np.ones(4), size=n)


def test_bell_order_is_the_canonical_one():
    assert BELL_ORDER == ("Phi+", "Phi-", "Psi+", "Psi-")


def test_pauli_round_trip():
    for p in random_states(40):
        assert xyz_to_pops(*pops_to_xyz(p)) == pytest.approx(p, abs=1e-15)


def test_isotropic_initial_vector():
    assert isotropic_pops(0.15) == pytest.approx([0.8875, 0.0375, 0.0375, 0.0375],
                                                 abs=1e-15)


@pytest.mark.parametrize("protocol", NOTE_PROTOCOLS)
@pytest.mark.parametrize("q", QS)
def test_note_recurrence_matches_the_verified_map(protocol, q):
    for p in random_states():
        assert note_step(p, q, protocol) == pytest.approx(MAP[protocol](p, q),
                                                          abs=1e-14)


@pytest.mark.parametrize("q", QS)
def test_note_success_probability_matches(q):
    for p in random_states(60):
        assert note_success(p, q) == pytest.approx(float(analytic_success_4q(p, q)),
                                                   abs=1e-14)


@pytest.mark.parametrize("protocol", NOTE_PROTOCOLS)
def test_fidelity_written_in_the_invariant_plane(protocol):
    """(1+2x+z)/4 for Type 3/4 and (1+x+2z)/4 for 4Q must equal p_Phi+ ALONG the
    trajectory, and must NOT be interchangeable."""
    p = isotropic_pops(0.15)
    worst_right = worst_wrong = 0.0
    for _ in range(40):
        worst_right = max(worst_right,
                          abs(note_fidelity_in_plane(p, protocol) - float(p[0])))
        x, _y, z = pops_to_xyz(p)
        other = (1 + 2 * x + z) / 4 if protocol == "4Q" else (1 + x + 2 * z) / 4
        worst_wrong = max(worst_wrong, abs(other - float(p[0])))
        p = MAP[protocol](p, 0.02)
    assert worst_right < 1e-14
    assert worst_wrong > 1e-3          # the other formula really is wrong here


@pytest.mark.parametrize("protocol", NOTE_PROTOCOLS)
def test_the_stated_invariant_plane_is_preserved(protocol):
    """y = -x for Type 3/4, y = -z for 4Q, starting from Bell-isotropic."""
    p = isotropic_pops(0.15)
    worst = 0.0
    for _ in range(40):
        x, y, z = pops_to_xyz(p)
        worst = max(worst, abs(x + y) if protocol != "4Q" else abs(y + z))
        p = note_step(p, 0.02, protocol)
    assert worst < 1e-14


@pytest.mark.parametrize("protocol", NOTE_PROTOCOLS)
def test_trajectory_stays_a_valid_distribution(protocol):
    p = isotropic_pops(0.15)
    for _ in range(60):
        assert p.sum() == pytest.approx(1.0, abs=1e-14)
        assert p.min() >= -1e-15
        p = note_step(p, 0.02, protocol)


@pytest.mark.parametrize("protocol", NOTE_PROTOCOLS)
def test_invariant_pair_agrees_with_the_package_table(protocol):
    a, b = INVARIANT_PAIR[protocol]
    i_a, i_b = BELL_ORDER.index(a), BELL_ORDER.index(b)
    p = isotropic_pops(0.15)
    worst = 0.0
    for _ in range(40):
        worst = max(worst, abs(float(p[i_a] - p[i_b])))
        p = note_step(p, 0.02, protocol)
    assert worst < 1e-15


def test_note_has_no_recurrence_for_type5():
    with pytest.raises(KeyError, match="Type5"):
        note_step(isotropic_pops(0.15), 0.02, "Type5")
