"""Comparison with the parent project's SWAP-test gadgets: frozen data integrity,
re-implemented recursions, and the ordering claims made in the docs."""

import csv
import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from pqec_distill.noisy_analytics import fidelity_uv, fixed_point_branch, saddle_node_p
from pqec_distill.swap_test_reference import (
    Q_ENT_STEP3, Q_ENT_STEP4, Q_SN_STEP3, Q_SN_STEP4, fidelity_34,
    fixed_point_34, step3_map, step4_map,
)

EXT = Path(__file__).resolve().parents[1] / "results" / "data" / "external"
CSV = EXT / "pqec_operational_threshold__fidelity_vs_round.csv"
META = EXT / "pqec_operational_threshold__fidelity_vs_round_metadata.json"
PROV = EXT / "PROVENANCE.md"


def _recorded_sha(label):
    text = PROV.read_text()
    m = re.search(rf"SHA-256 \({label}\): `([0-9a-f]{{64}})`", text)
    assert m, f"no SHA-256 for {label} in PROVENANCE.md"
    return m.group(1)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_files_match_provenance():
    assert _sha256(CSV) == _recorded_sha("CSV")
    assert _sha256(META) == _recorded_sha("metadata")


@pytest.fixture(scope="module")
def parent():
    out = {}
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["series"], {})[int(r["n"])] = float(r["F"])
    return out


@pytest.mark.parametrize("series,step", [("step3", step3_map), ("step4", step4_map)])
def test_reimplemented_recursion_matches_frozen_csv(parent, series, step):
    """The parent's dense 32x32 iteration (frozen) vs our (u,v) recursion."""
    qb, u, v = 0.99, 0.9, 0.9
    worst = abs(parent[series][0] - fidelity_34(u, v))
    for n in range(1, 5001):
        u, v = step(u, v, qb)
        if n in parent[series]:
            worst = max(worst, abs(parent[series][n] - fidelity_34(u, v)))
    assert worst < 1e-12


def test_parent_note_tabulated_fixed_points():
    """Representative values quoted in the parent notes 01/02."""
    for q, f3, f4 in [(0.01, 0.978346, 0.982351), (0.05, 0.881237, 0.908138)]:
        assert abs(fidelity_34(*fixed_point_34(q, 3)) - f3) < 1e-6
        assert abs(fidelity_34(*fixed_point_34(q, 4)) - f4) < 1e-6
    assert abs(fidelity_34(*fixed_point_34(0.10, 3)) - 0.712918) < 1e-6
    assert abs(fidelity_34(*fixed_point_34(0.12, 3)) - 0.600670) < 1e-6


def test_parent_saddle_node_and_entanglement_limits():
    """Discriminant vanishes at q_SN; F* = 1/2 at q_ent (as in the notes)."""
    for step, q_sn, q_ent in [(3, Q_SN_STEP3, Q_ENT_STEP3), (4, Q_SN_STEP4, Q_ENT_STEP4)]:
        assert not np.isnan(fixed_point_34(q_sn - 1e-7, step)[0])
        assert np.isnan(fixed_point_34(q_sn + 1e-7, step)[0])
        assert abs(fidelity_34(*fixed_point_34(q_ent, step)) - 0.5) < 1e-6


def test_first_order_coefficients_ordering():
    """1 - F* ~ A q with A = 17/8 (S3), 7/4 (S4), 1 (4-qubit)."""
    q = 1e-5
    a3 = (1 - fidelity_34(*fixed_point_34(q, 3))) / q
    a4 = (1 - fidelity_34(*fixed_point_34(q, 4))) / q
    a4q = (1 - fidelity_uv(*fixed_point_branch(1 - q))) / q
    assert abs(a3 - 17 / 8) < 1e-3 and abs(a4 - 7 / 4) < 1e-3 and abs(a4q - 1) < 1e-3


def test_4qubit_fixed_point_above_swap_test_fixed_points():
    """At equal per-CNOT noise, F*(4q) > F*(S4) > F*(S3) wherever all exist
    below q = 0.17 (the curves cross Step 4's only above ~0.175)."""
    for q in np.linspace(0.001, 0.17, 60):
        f4q = fidelity_uv(*fixed_point_branch(1 - q))
        f4 = fidelity_34(*fixed_point_34(q, 4))
        assert f4q > f4
        if q < Q_SN_STEP3:
            assert f4 > fidelity_34(*fixed_point_34(q, 3))


def test_step4_branch_outlives_4qubit_branch_slightly():
    """Honest ordering of the saddle nodes: S3 < 4q < S4."""
    assert Q_SN_STEP3 < saddle_node_p() < Q_SN_STEP4


def test_same_condition_as_parent_figure_1(parent):
    """eb = 0.9, q = 0.01: 4-qubit F_5000 exceeds S3, S4 and the escaped S5."""
    u = v = 0.9
    for _ in range(5000):
        u, v = __import__("pqec_distill.noisy_analytics", fromlist=["noisy_map_uv"]).noisy_map_uv(u, v, 0.99)
    f4q = fidelity_uv(u, v)
    assert f4q > parent["step3"][5000] and f4q > parent["step4"][5000]
    assert parent["step5"][10] > parent["step4"][10]        # Step 5's plateau is high...
    assert parent["step5"][5000] < 0.5                      # ...but it escapes
    assert f4q > parent["step5"][10]                        # and 4q beats even the plateau
