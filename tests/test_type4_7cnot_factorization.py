"""Does the frozen 14-CNOT Type-4 circuit factorise into two 7-CNOT Fredkins?

Nothing here uses the analytic Type-4 recurrence; ``analytic_exact_map`` is
never imported by the modules under test.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

from pqec_distill import type4_7cnot_fredkin as T4
from pqec_distill.noise import replacement_depolarizing

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "verify_type4_7cnot_factorization.py"
MODULE_PATH = ROOT / "src" / "pqec_distill" / "type4_7cnot_fredkin.py"

TOL = 1e-12
EXPECTED_CNOTS_A = [(1, 3), (0, 3), (1, 3), (0, 3), (0, 1), (0, 1), (3, 1)]
EXPECTED_CNOTS_B = [(2, 4), (0, 4), (2, 4), (0, 4), (0, 2), (0, 2), (4, 2)]
Q_GRID = (0.0, 1e-4, 0.01, 0.05, 0.2, 0.5, 1.0)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCRIPT = _load(SCRIPT_PATH, "_t4_factor_script")


@pytest.fixture(scope="module")
def split():
    return T4.split_type4_sequence()


def _key(ops):
    return [(k, tuple(a)) for k, a in ops]


# ---------------------------------------------------------------------------
# the gate list is the repository's own
# ---------------------------------------------------------------------------

def test_gate_list_matches_the_frozen_type4_scripts():
    prov = SCRIPT.provenance_checks()
    assert prov["n_operations"] == 36
    assert prov["n_cnot"] == 14
    assert prov["matches_verify_type4_qabs_circuit.py"] is True
    assert prov["matches_verify_type4_qrel_circuit.py"] is True
    assert prov["u3_vs_reference_u3"] < TOL


def test_cnot_pattern_matches_the_stated_structure(split):
    assert T4.ops_cnot_pairs(split["alice_ops"]) == EXPECTED_CNOTS_A
    assert T4.ops_cnot_pairs(split["bob_ops"]) == EXPECTED_CNOTS_B
    a_sup = {w for k, a in split["alice_ops"] for w in T4.ops_support(k, a)}
    b_sup = {w for k, a in split["bob_ops"] for w in T4.ops_support(k, a)}
    assert a_sup == {0, 1, 3}
    assert b_sup == {0, 2, 4}


# ===========================================================================
# A: G7 has exactly 7 CNOTs
# ===========================================================================

def test_a_g7_has_exactly_seven_cnots(split):
    assert T4.N_CNOT_G7 == 7
    assert len(T4.ops_cnot_pairs(T4.G7_OPS)) == 7
    assert len(T4.ops_cnot_pairs(split["alice_local"])) == 7
    assert len(T4.ops_cnot_pairs(split["bob_local"])) == 7


# ===========================================================================
# B: the two extracted blocks are identical after relabelling
# ===========================================================================

def test_b_extracted_blocks_identical_after_relabelling(split):
    assert _key(split["alice_local"]) == _key(split["bob_local"])
    assert _key(split["alice_local"]) == _key(T4.G7_OPS)
    assert _key(split["bob_local"]) == _key(T4.G7_OPS)
    for (k1, a1), (k2, a2) in zip(split["alice_local"], split["bob_local"]):
        assert k1 == k2
        assert all(float(x) == float(y) for x, y in zip(a1, a2))


# ===========================================================================
# C: each block is an ideal Fredkin up to a global phase
# ===========================================================================

@pytest.mark.parametrize("which", ["candidate", "alice", "bob"])
def test_c_block_is_ideal_fredkin_up_to_global_phase(split, which):
    ops = {"candidate": T4.G7_OPS, "alice": split["alice_local"],
           "bob": split["bob_local"]}[which]
    fit = T4.global_phase_fit(T4.block_unitary(ops), T4.fredkin_matrix())
    assert fit["max_entry_residual"] < TOL
    assert fit["frobenius_residual"] < TOL
    assert fit["unitarity_residual"] < TOL
    # the phase is a measured output; it is -pi/8 and is NOT 0, so the
    # comparison genuinely needed it
    assert abs(fit["global_phase_over_pi"] + 0.125) < 1e-09
    assert fit["residual_without_phase"] > 0.1


# ===========================================================================
# D: the ideal unitaries of the two orderings agree
# ===========================================================================

def test_d_original_and_regrouped_ideal_unitaries_agree():
    u_o = T4.ops_unitary(T4.TYPE4_5Q_OPS, 5)
    u_r = T4.ops_unitary(T4.REGROUPED_5Q_OPS, 5)
    assert np.max(np.abs(u_o - u_r)) < TOL
    assert np.max(np.abs(u_o.conj().T @ u_o - np.eye(32))) < TOL


def test_regrouping_is_a_disjoint_support_reordering():
    cert = T4.commutation_certificate(T4.TYPE4_5Q_OPS, T4.REGROUPED_5Q_OPS)
    assert cert["same_length"]
    assert cert["same_multiset"]
    assert cert["offending_pairs"] == []
    assert cert["is_certified"]
    # the reordering is not trivial: operations really did move
    assert cert["n_inversions"] > 0


def test_regrouped_circuit_is_the_ideal_two_fredkin_swap_test():
    ideal = SCRIPT.ideal_unitary_checks()
    assert ideal["regrouped_vs_ideal_swap_test"]["max_entry_residual"] < TOL
    assert abs(ideal["regrouped_vs_ideal_swap_test"]["global_phase_over_pi"]
               + 0.25) < 1e-09


# ===========================================================================
# E: the NOISY channels of the two orderings agree
# ===========================================================================

def test_e_original_and_regrouped_noisy_channels_agree():
    rng = np.random.default_rng(98765)
    states = []
    for _ in range(3):
        a = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
        m = a @ a.conj().T
        states.append(m / np.trace(m).real)
    v = rng.normal(size=32) + 1j * rng.normal(size=32)
    states.append(np.outer(v, v.conj()) / float(np.vdot(v, v).real))

    def channel(s, pair, q, n):
        return replacement_depolarizing(s, pair, q, n_qubits=n)

    prog_o = T4.build_program(T4.TYPE4_5Q_OPS, 5)
    prog_r = T4.build_program(T4.REGROUPED_5Q_OPS, 5)
    worst = 0.0
    for sigma in states:
        for q in Q_GRID:
            a = T4.run_noisy_ops(sigma, T4.TYPE4_5Q_OPS, q, 5,
                                 channel=channel, program=prog_o)
            b = T4.run_noisy_ops(sigma, T4.REGROUPED_5Q_OPS, q, 5,
                                 channel=channel, program=prog_r)
            worst = max(worst, float(np.max(np.abs(a - b))))
    assert worst < TOL


def test_reference_route_reproduces_the_module_channel():
    rng = np.random.default_rng(555)
    a = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
    sigma = a @ a.conj().T
    sigma = sigma / np.trace(sigma).real

    def channel(s, pair, q, n):
        return replacement_depolarizing(s, pair, q, n_qubits=n)

    for q in (0.0, 0.03, 0.4, 1.0):
        main = T4.run_noisy_ops(sigma, T4.REGROUPED_5Q_OPS, q, 5, channel=channel)
        ref = SCRIPT.ref_run(sigma, T4.REGROUPED_5Q_OPS, q, 5)
        assert np.max(np.abs(main - ref)) < TOL


def test_embed_and_relabel_are_inverse():
    for wires in (T4.ALICE_TRIPLE_5Q, T4.BOB_TRIPLE_5Q, (4, 0, 2)):
        back = T4._local_relabel(T4.embed_ops(T4.G7_OPS, wires), wires)
        assert _key(back) == _key(T4.G7_OPS)


# ===========================================================================
# the factorization script does not reach for the analytic recurrence
# ===========================================================================

def test_no_analytic_recurrence_in_the_factorization_path():
    forbidden = ("analytic_exact_map", "analytic_map_type3", "analytic_map_type4",
                 "analytic_map_4q", "repeated_dynamics", "noisy_closed_form",
                 "bell_population_dynamics")
    for path in (MODULE_PATH, SCRIPT_PATH):
        code = "\n".join(l for l in path.read_text().splitlines()
                         if not re.match(r"\s*#", l))
        for name in forbidden:
            assert not re.search(rf"import\s+.*{name}", code), (path.name, name)
            assert not re.search(rf"{name}\s*\(", code), (path.name, name)
    assert not [n for n in vars(T4) if "analytic" in n]
