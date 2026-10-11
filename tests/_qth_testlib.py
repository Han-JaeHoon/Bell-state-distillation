"""Shared helpers for the three q_th protocol test modules."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

DRIVER_PATH = SCRIPTS / "verify_qth_full_circuit.py"
CORE_PATH = SCRIPTS / "_qth_core.py"
HP_PATH = SCRIPTS / "_qth_highprec.py"

FORBIDDEN = ("analytic_exact_map", "analytic_map_4q", "analytic_map_type3",
             "analytic_map_type4", "noisy_analytics", "noisy_closed_form",
             "repeated_dynamics", "bell_population_dynamics")

_DRIVER = None


def driver():
    global _DRIVER
    if _DRIVER is None:
        spec = importlib.util.spec_from_file_location("_qth_driver", DRIVER_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _DRIVER = mod
    return _DRIVER


def core():
    return driver().C


def expected_cnots(protocol: str) -> int:
    return {"type3": 16, "type4": 14, "4q": 5}[protocol]


def assert_q0_is_identity(protocol: str):
    C = core()
    n = C.N_WIRES[protocol]
    rng = np.random.default_rng(3)
    d = 2 ** n
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    z = a @ a.conj().T
    z = z / np.trace(z).real
    for pair in C.cnot_pairs(protocol):
        assert np.max(np.abs(C.replacement_depol(z, pair, 0.0, n) - z)) == 0.0
        assert np.max(np.abs(C._depol_b(z, pair, 0.0, n) - z)) == 0.0


def assert_q1_replaces_the_pair(protocol: str):
    C = core()
    n = C.N_WIRES[protocol]
    rng = np.random.default_rng(4)
    d = 2 ** n
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    z = a @ a.conj().T
    z = z / np.trace(z).real
    letters_r, letters_c = "abcdefgh"[:n], "ijklmnop"[:n]

    def reduced(m, keep):
        row, col = list(letters_r), list(letters_c)
        for w in range(n):
            if w not in keep:
                col[w] = row[w]
        return np.einsum("".join(row) + "".join(col) + "->"
                         + "".join(row[w] for w in keep)
                         + "".join(col[w] for w in keep),
                         m.reshape([2] * (2 * n))).reshape(2 ** len(keep),
                                                           2 ** len(keep))

    for pair in set(C.cnot_pairs(protocol)):
        out = C.replacement_depol(z, pair, 1.0, n)
        assert np.max(np.abs(reduced(out, sorted(pair)) - np.eye(4) / 4)) < 1e-12
        others = [w for w in range(n) if w not in pair]
        assert np.max(np.abs(reduced(out, others) - reduced(z, others))) < 1e-12
        # and the independently coded channel agrees
        assert np.max(np.abs(out - C._depol_b(z, pair, 1.0, n))) < 1e-12


def assert_bell_projector_fidelity():
    C = core()
    assert np.max(np.abs(C.PHI_PLUS_PROJECTOR
                         - np.outer(C.PHI_P, C.PHI_P.conj()))) < 1e-15
    for name, vec in C.BELL.items():
        rho = np.outer(vec, vec.conj())
        assert abs(C.fidelity(rho) - (1.0 if name == "Phi+" else 0.0)) < 1e-14
    for eps in (0.0, 0.05, 0.4, 2.0 / 3.0, 1.0):
        assert abs(C.fidelity(C.isotropic_rho(eps)) - (1 - 0.75 * eps)) < 1e-14
    rng = np.random.default_rng(11)
    for _ in range(20):
        a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        rho = a @ a.conj().T
        rho = rho / np.trace(rho).real
        assert abs(C.fidelity(rho) - C.bell_populations(rho)[0]) < 1e-14


def assert_routes_agree(protocol: str, points, tol=1e-12):
    C = core()
    worst = 0.0
    for eps, q in points:
        rho_a = rho_b = C.isotropic_rho(eps)
        for _ in range(5):
            rho_a, ia = C.one_round(rho_a, q, protocol)
            rho_b, ib = C.one_round_b(rho_b, q, protocol)
            worst = max(worst, float(np.max(np.abs(rho_a - rho_b))),
                        abs(C.fidelity(rho_a) - C.fidelity(rho_b)),
                        abs(ia["denominator"] - ib["denominator"]))
    assert worst < tol, worst
    return worst


def assert_no_analytic_imports():
    for path in (CORE_PATH, HP_PATH, DRIVER_PATH):
        text = path.read_text()
        code = "\n".join(l for l in text.splitlines() if not re.match(r"\s*#", l))
        for name in FORBIDDEN:
            assert not re.search(rf"import\s+.*{name}", code), (path.name, name)
            assert not re.search(rf"{name}\s*\(", code), (path.name, name)
        assert "from pqec_distill" not in code, path.name
        assert not re.search(r"^\s*import pqec_distill", code, re.M), path.name
    d = driver()
    for mod in (d, d.C, d.HP):
        assert not [n for n in vars(mod) if "analytic" in n.lower()]
    # the threshold search never reads a q_abs or q_rel value: those names
    # appear only in the clearly separated secondary section
    text = DRIVER_PATH.read_text()
    search = text[text.index("def run_trajectory("):
                  text.index("# 8. SECONDARY comparisons")]
    for bad in ("q_abs", "q_rel", "_QABS_FILES", "_QREL_FILES"):
        assert bad not in search, bad
