#!/usr/bin/env python3
"""Circuit-level NUMERICAL comparison of the raw noisy 5Q Type-4 estimator and
the raw noisy 6Q-L14 estimator.

THE QUESTION (not assumed, measured)
------------------------------------
Five qubits, ordering ``[c, A1, B1, A2, B2]``, the regrouped Type-4 circuit
``H_c -> G7(c;A1,A2) -> G7(c;B1,B2) -> H_c`` (14 CNOTs):

    Delta_5_T4(q; O, R) = < X_c (x) O_(A1B1) (x) I_(A2B2) >     (route A)
                        = < Z_c (x) O_(A1B1) (x) I_(A2B2) >     (route B)

Six qubits, ordering ``[a, b, A1, B1, A2, B2]``, the distributed two-ancilla
circuit 6Q-L14 (7 CNOTs Alice + 7 CNOTs Bob, no cross-party gate):

    Delta_6_T4(q; O, R) = < X_a X_b (x) O (x) I > - < Y_a Y_b (x) O (x) I >

Both are raw and unnormalised.  The script measures whether they coincide for
broad families of two-qubit rho, arbitrary four-qubit data inputs R, arbitrary
Hermitian O, and q from 0 to 1, and separately whether ``Delta_5(I)`` equals
``Delta_6(I)`` so that a hidden q-dependent prefactor could not escape notice.

GUARD AGAINST SELF-COMPARISON
-----------------------------
The 5Q signal is also computed from the ORIGINAL frozen 14-CNOT Type-4 gate
ordering (route B), and that is compared with the regrouped one, so the new
circuit is never silently compared with itself.  Four negative controls are
included as well.

NOT USED
--------
No analytic Type-4 Bell-diagonal recurrence, no closed-form noisy map, no
previously derived Type-4 threshold.  ``pqec_distill.analytic_exact_map`` is
never imported.  Bell diagonality, isotropy and estimator equality are never
assumed.

Usage::

    python scripts/verify_5q_6q_type4_noisy_equivalence.py
    python scripts/verify_5q_6q_type4_noisy_equivalence.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pqec_distill import locc_vd_type4_noisy as SIX                # noqa: E402
from pqec_distill import type4_5q_raw_noisy as FIVE                # noqa: E402
from pqec_distill import type4_7cnot_fredkin as T4                 # noqa: E402
from pqec_distill.locc_vd import (                                 # noqa: E402
    PAULI_LABELS, correlator_6q, delta_6q, final_state_6q, pauli_product,
    reconstruct_operator,
)
from pqec_distill.noise import replacement_depolarizing            # noqa: E402

OUT = ROOT / "results" / "data" / "5q_6q_type4_noisy_equivalence"

TOL_SANITY = 1e-12
TOL_EQUIV = 1e-12
NC_MIN_SIGNAL = 1e-06

Q_GRID = (0.0, 1e-6, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.05,
          0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0)

SEEDS = {"pure": 20261011, "mixed": 5772156, "rank_def": 27182818,
         "observables": 31415926, "general_r": 16180339, "projectors": 14142135}


# ===========================================================================
# 1.  self-contained reference route (independent of both main paths)
# ===========================================================================

_I2 = np.eye(2, dtype=complex)
_X1 = np.array([[0, 1], [1, 0]], dtype=complex)
_Y1 = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z1 = np.array([[1, 0], [0, -1]], dtype=complex)
_ROW, _COL = "abcdefgh", "ijklmnop"


def ref_u3(theta, phi, lam) -> np.ndarray:
    """U3 from its matrix definition, written without np.exp."""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -(np.cos(lam) + 1j * np.sin(lam)) * s],
                     [(np.cos(phi) + 1j * np.sin(phi)) * s,
                      (np.cos(phi + lam) + 1j * np.sin(phi + lam)) * c]],
                    dtype=complex)


def ref_gate(kind, arg, n: int) -> np.ndarray:
    """CNOT as the computational-basis permutation it induces."""
    d = 2 ** n
    if kind == "CNOT":
        c, tgt = arg
        m = np.zeros((d, d), dtype=complex)
        for idx in range(d):
            bits = [(idx >> (n - 1 - w)) & 1 for w in range(n)]
            if bits[c] == 1:
                bits[tgt] ^= 1
            m[sum(b << (n - 1 - w) for w, b in enumerate(bits)), idx] = 1.0
        return m
    w, t, p, l = arg
    out = np.array([[1.0 + 0j]])
    for k in range(n):
        out = np.kron(out, ref_u3(t, p, l) if k == w else _I2)
    return out


def ref_depol(sigma, pair, q, n) -> np.ndarray:
    """Replacement depolarizing coded directly as two einsum contractions."""
    if q == 0.0:
        return sigma
    i, j = pair
    rest = [w for w in range(n) if w not in (i, j)]
    t = sigma.reshape([2] * (2 * n))
    row, col = list(_ROW[:n]), list(_COL[:n])
    for w in (i, j):
        col[w] = row[w]
    sub = "".join(row) + "".join(col)
    out = "".join(row[w] for w in rest) + "".join(col[w] for w in rest)
    reduced = np.einsum(f"{sub}->{out}", t)
    target = _ROW[:n] + _COL[:n]
    rebuilt = np.einsum(f"{out},{_ROW[i]}{_COL[i]},{_ROW[j]}{_COL[j]}->{target}",
                        reduced, _I2, _I2)
    d = 2 ** n
    return (1 - q) * sigma + q * 0.25 * rebuilt.reshape(d, d)


def _ref_run(ops, sigma, q, n, q_schedule=None, skip_noise_after=()):
    k = 0
    for kind, arg in ops:
        g = ref_gate(kind, arg, n)
        sigma = g @ sigma @ g.conj().T
        if kind == "CNOT":
            k += 1
            qk = q if q_schedule is None else q_schedule[k - 1]
            if k not in skip_noise_after:
                sigma = ref_depol(sigma, tuple(arg), qk, n)
    return sigma


def ref_state_5q(data_r, q, **kw) -> np.ndarray:
    """Reference 5Q route, stopped before the final ancilla H (route A)."""
    proj0 = np.array([[1, 0], [0, 0]], dtype=complex)
    sigma = np.kron(proj0, np.asarray(data_r, dtype=complex))
    return _ref_run(FIVE.OPS_REGROUPED_NO_FINAL_H, sigma, q, 5, **kw)


def ref_state_6q(data_r, q, ops=None, **kw) -> np.ndarray:
    """Reference 6Q-L14 route; both ancillas prepared in |+>."""
    plus = 0.5 * np.ones((2, 2), dtype=complex)
    sigma = np.kron(plus, np.kron(plus, np.asarray(data_r, dtype=complex)))
    return _ref_run(ops if ops is not None else SIX.L14_OPS, sigma, q, 6, **kw)


def ref_obs_5q(o, ancilla) -> np.ndarray:
    return np.kron(ancilla, np.kron(o, np.eye(4, dtype=complex)))


def ref_obs_6q(o, ancilla) -> np.ndarray:
    return np.kron(ancilla, np.kron(ancilla, np.kron(o, np.eye(4, dtype=complex))))


def _tr(obs, sigma) -> float:
    return float(np.real(np.sum(obs * sigma.T)))


def ref_delta_5q(data_r, o, q, **kw) -> float:
    return _tr(ref_obs_5q(o, _X1), ref_state_5q(data_r, q, **kw))


def ref_delta_6q(data_r, o, q, **kw) -> float:
    s = ref_state_6q(data_r, q, **kw)
    return _tr(ref_obs_6q(o, _X1), s) - _tr(ref_obs_6q(o, _Y1), s)


# ===========================================================================
# 2.  main paths
# ===========================================================================

def delta_6_from_state(sigma6, o):
    """``(Delta_6, C_XX, C_YY)`` through the frozen 6Q read-out."""
    c_xx = correlator_6q(sigma6, "X", o)
    c_yy = correlator_6q(sigma6, "Y", o)
    return c_xx - c_yy, c_xx, c_yy


# ===========================================================================
# 3.  structural and noise-convention checks
# ===========================================================================

def structural_checks() -> dict:
    u6 = SIX.circuit_unitary()
    ua, ub = SIX.local_block_unitary("alice"), SIX.local_block_unitary("bob")
    fa, fb = SIX.ideal_local_fredkin("alice"), SIX.ideal_local_fredkin("bob")
    out = {
        "wires_5q": list(FIVE.WIRES_5Q),
        "wires_6q": list(SIX.WIRES_6Q),
        "alice_triple_6q": list(SIX.ALICE_TRIPLE),
        "bob_triple_6q": list(SIX.BOB_TRIPLE),
        "n_cnot_5q": FIVE.N_CNOT_5Q_T4,
        "n_cnot_6q": SIX.N_CNOT,
        "n_cnot_6q_alice": SIX.N_CNOT_ALICE,
        "n_cnot_6q_bob": SIX.N_CNOT_BOB,
        "n_cnot_5q_equals_14": FIVE.N_CNOT_5Q_T4 == 14,
        "n_cnot_6q_equals_14": SIX.N_CNOT == 14,
        "cnot_pairs_5q_regrouped": [list(p) for p in
                                    T4.ops_cnot_pairs(FIVE.OPS_REGROUPED)],
        "cnot_pairs_5q_original": [list(p) for p in
                                   T4.ops_cnot_pairs(FIVE.OPS_ORIGINAL)],
        "cnot_pairs_6q": [list(p) for p in SIX.cnot_wire_pairs()],
        "cross_party_two_qubit_gates_6q": [list(p) for p in SIX.crosses_party_cut()],
        "unitarity_6q": float(np.max(np.abs(u6.conj().T @ u6 - np.eye(64)))),
        "local_blocks_commutator_6q": float(np.max(np.abs(ua @ ub - ub @ ua))),
        "alice_block_vs_ideal_fredkin": T4.global_phase_fit(ua, fa),
        "bob_block_vs_ideal_fredkin": T4.global_phase_fit(ub, fb),
        "full_6q_vs_product_of_ideal_fredkins": T4.global_phase_fit(u6, fb @ fa),
    }
    diffs = [float(np.max(np.abs(ref_gate(k, a, 5)
                                 - T4.gate_matrix(k, a, 5))))
             for k, a in FIVE.OPS_REGROUPED]
    out["ref_gate_vs_module_gate_5q"] = max(diffs)
    diffs6 = [float(np.max(np.abs(ref_gate(k, a, 6)
                                  - T4.gate_matrix(k, a, 6))))
              for k, a in SIX.L14_OPS]
    out["ref_gate_vs_module_gate_6q"] = max(diffs6)
    return out


def noise_convention_checks(rng, n_trials: int = 12) -> dict:
    out = {
        "same_function_object": (
            FIVE.replacement_depolarizing is replacement_depolarizing
            and SIX.replacement_depolarizing is replacement_depolarizing),
        "convention_string_5q": FIVE.NOISE_CONVENTION,
        "convention_string_6q": SIX.NOISE_CONVENTION,
        "convention_strings_equal": FIVE.NOISE_CONVENTION == SIX.NOISE_CONVENTION,
    }
    worst = {5: 0.0, 6: 0.0}
    for n in (5, 6):
        d = 2 ** n
        for _ in range(n_trials):
            m = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
            sigma = m @ m.conj().T
            sigma = sigma / np.trace(sigma).real
            i, j = rng.choice(n, size=2, replace=False)
            q = float(rng.uniform(0.0, 1.0))
            a = replacement_depolarizing(sigma, (int(i), int(j)), q, n_qubits=n)
            b = ref_depol(sigma, (int(i), int(j)), q, n)
            worst[n] = max(worst[n], float(np.max(np.abs(a - b))))
    out["package_channel_vs_ref_channel_5wires"] = worst[5]
    out["package_channel_vs_ref_channel_6wires"] = worst[6]
    for key, prog, ops in (
            ("5q_regrouped", FIVE.PROGRAMS["regrouped"], FIVE.OPS_REGROUPED),
            ("5q_original", FIVE.PROGRAMS["original"], FIVE.OPS_ORIGINAL),
            ("6q", SIX.PROGRAM, SIX.L14_OPS)):
        locs = [list(p) for k, p in prog if k == "D"]
        out[f"n_channels_{key}"] = len(locs)
        out[f"channels_match_cnots_{key}"] = (
            locs == [list(p) for p in T4.ops_cnot_pairs(ops)])
    z = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
    out["q0_is_identity_channel"] = float(np.max(np.abs(
        replacement_depolarizing(z, (0, 1), 0.0, n_qubits=5) - z)))
    return out


def ideal_6q_validation(states, observables) -> dict:
    """q = 0: 6Q-L14 against the frozen ideal 6Q estimator and Tr(O rho^2)."""
    worst = {"vs_frozen_ideal_6q": 0.0, "vs_trace_O_rho2": 0.0,
             "vs_5q_regrouped": 0.0, "worst_tag": None}
    rows = []
    for slabel, sfam, rho in states:
        data_r = np.kron(rho, rho)
        s6 = SIX.final_state_6q_t4_noisy(data_r, 0.0)
        s6_frozen_ideal = final_state_6q(rho)
        s5 = FIVE.state_before_final_h(data_r, 0.0)
        rho2 = rho @ rho
        for lab, fam, o in observables:
            d6 = delta_6_from_state(s6, o)[0]
            d6_ref = delta_6q(s6_frozen_ideal, o)[0]
            d_rho2 = float(np.real(np.trace(o @ rho2)))
            d5 = FIVE.delta_from_state_x(s5, o)
            a, b, c = abs(d6 - d6_ref), abs(d6 - d_rho2), abs(d6 - d5)
            if a > worst["vs_frozen_ideal_6q"]:
                worst["vs_frozen_ideal_6q"], worst["worst_tag"] = a, (slabel, lab)
            worst["vs_trace_O_rho2"] = max(worst["vs_trace_O_rho2"], b)
            worst["vs_5q_regrouped"] = max(worst["vs_5q_regrouped"], c)
            if fam == "F1_pauli_product":
                rows.append({"state": slabel, "state_family": sfam,
                             "observable": lab, "delta6_L14": d6,
                             "delta6_frozen_ideal": d6_ref,
                             "trace_O_rho_squared": d_rho2, "delta5_T4": d5,
                             "abs_diff_vs_frozen_ideal": a,
                             "abs_diff_vs_trace_O_rho2": b})
    return {"worst": worst, "rows": rows}


# ===========================================================================
# 4.  state and observable families
# ===========================================================================

PHI_P = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
PHI_M = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
PSI_P = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
PSI_M = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
_BELL_ROWS = np.stack([PHI_P, PHI_M, PSI_P, PSI_M])


def _dm(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=complex)
    return np.outer(v, v.conj()) / float(np.vdot(v, v).real)


def _isotropic(eps) -> np.ndarray:
    return (1 - eps) * _dm(PHI_P) + eps * np.eye(4, dtype=complex) / 4


def _bell_diagonal(p) -> np.ndarray:
    out = np.zeros((4, 4), dtype=complex)
    for w, v in zip(p, _BELL_ROWS):
        out = out + float(w) * np.outer(v, v.conj())
    return out


def _random_pure(rng, dim=4) -> np.ndarray:
    return _dm(rng.normal(size=dim) + 1j * rng.normal(size=dim))


def _random_mixed(rng, dim=4, rank=None) -> np.ndarray:
    r = dim if rank is None else rank
    a = rng.normal(size=(dim, r)) + 1j * rng.normal(size=(dim, r))
    m = a @ a.conj().T
    return m / np.trace(m).real


def rho_test_set(quick: bool = False) -> list:
    sts = []
    for name, v in (("Phi+", PHI_P), ("Phi-", PHI_M),
                    ("Psi+", PSI_P), ("Psi-", PSI_M)):
        sts.append((f"bell_{name}", "A_bell", _dm(v)))
    for k, lab in enumerate(("00", "01", "10", "11")):
        e = np.zeros(4, dtype=complex)
        e[k] = 1.0
        sts.append((f"comp_{lab}", "B_computational", _dm(e)))
    sts.append(("coh_00_plus_01", "F_non_bell_diagonal", _dm([1, 1, 0, 0])))
    sts.append(("coh_plus_zero", "F_non_bell_diagonal",
                _dm(np.kron([1, 1], [1, 0]))))
    sts.append(("coh_tilted_phi", "F_non_bell_diagonal",
                _dm([np.cos(0.3), 0.2, -0.5, np.sin(1.1)])))
    sts.append(("imag_00_plus_i11", "G_imaginary_coherence", _dm([1, 0, 0, 1j])))
    sts.append(("imag_01_plus_i10", "G_imaginary_coherence", _dm([0, 1, 1j, 0])))
    m = 0.6 * _dm([1, 0, 0, 1j]) + 0.4 * _dm([0.3, 1j * 0.5, 0.7, -0.2j])
    sts.append(("imag_mixed", "G_imaginary_coherence", m / np.trace(m).real))
    for e in (1e-6, 1e-3):
        sts.append((f"near_pure_eps{e:g}", "H_near_pure", _isotropic(e)))
    for e in (1 - 1e-6, 0.999):
        sts.append((f"near_mixed_eps{e:.6f}", "I_near_maximally_mixed",
                    _isotropic(e)))
    for p in ((0.5, 0.3, 0.15, 0.05), (0.7, 0.1, 0.1, 0.1),
              (0.4, 0.4, 0.15, 0.05), (0.25, 0.25, 0.25, 0.25)):
        sts.append((f"belldiag_{'_'.join(f'{x:g}' for x in p)}",
                    "J_bell_diagonal_asymmetric", _bell_diagonal(p)))
    for e in (0.05, 0.2, 0.4, 0.6, 0.8, 1.0):
        sts.append((f"isotropic_eps{e:g}", "K_bell_isotropic", _isotropic(e)))

    n_pure, n_mixed, n_rank = (6, 6, 4) if quick else (25, 25, 16)
    rng = np.random.default_rng(SEEDS["pure"])
    for k in range(n_pure):
        sts.append((f"rand_pure_{k:02d}", "C_random_pure", _random_pure(rng)))
    rng = np.random.default_rng(SEEDS["mixed"])
    for k in range(n_mixed):
        sts.append((f"rand_mixed_{k:02d}", "D_random_full_rank", _random_mixed(rng)))
    rng = np.random.default_rng(SEEDS["rank_def"])
    for k in range(n_rank):
        rank = 2 if k % 2 == 0 else 3
        sts.append((f"rand_rank{rank}_{k:02d}", "E_rank_deficient",
                    _random_mixed(rng, rank=rank)))
    return sts


def general_r_test_set(quick: bool = False) -> list:
    rng = np.random.default_rng(SEEDS["general_r"])
    out = []
    n = 2 if quick else 4
    for k in range(n):
        out.append((f"R_pure4q_{k:02d}", "random_pure_4q", _random_pure(rng, 16)))
    for k in range(n):
        out.append((f"R_mixed4q_{k:02d}", "random_full_rank_4q",
                    _random_mixed(rng, 16)))
    for k in range(n):
        rank = 3 + 4 * k
        out.append((f"R_rank{rank}_{k:02d}", "rank_deficient_4q",
                    _random_mixed(rng, 16, rank=rank)))
    for k in range(n):
        out.append((f"R_product_nonidentical_{k:02d}", "product_rho_tensor_sigma",
                    np.kron(_random_mixed(rng), _random_pure(rng))))
    for k in range(max(1, n // 2)):
        w = rng.dirichlet(np.ones(3))
        acc = np.zeros((16, 16), dtype=complex)
        for wi in w:
            acc = acc + wi * np.kron(_random_mixed(rng), _random_mixed(rng))
        out.append((f"R_correlated_{k:02d}", "correlated_across_copies", acc))
    for k in range(max(1, n // 2)):
        v = (rng.normal(size=16) + 1j * rng.normal(size=16)).reshape(4, 4)
        v[:, 0] *= 3.0
        out.append((f"R_entangled_cut_{k:02d}", "entangled_across_copy_cut",
                    _dm(v.reshape(16))))
    for k in range(max(1, n // 2)):
        base = _dm(rng.normal(size=16) + 1j * rng.normal(size=16))
        out.append((f"R_complex_coherent_{k:02d}", "complex_coherences",
                    0.85 * base + 0.15 * np.eye(16, dtype=complex) / 16))
    return out


def observable_set(quick: bool = False) -> list:
    obs = [(lab, "F1_pauli_product", pauli_product(lab)) for lab in PAULI_LABELS]
    rng = np.random.default_rng(SEEDS["observables"])
    for k in range(8 if quick else 30):
        m = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        h = 0.5 * (m + m.conj().T)
        obs.append((f"randherm_{k:02d}", "F2_random_hermitian",
                    h / np.linalg.norm(h)))
    for name, v in (("Phi+", PHI_P), ("Phi-", PHI_M),
                    ("Psi+", PSI_P), ("Psi-", PSI_M)):
        obs.append((f"proj_{name}", "F3_projector", _dm(v)))
    rng = np.random.default_rng(SEEDS["projectors"])
    for k in range(2 if quick else 6):
        obs.append((f"proj_rand1_{k:02d}", "F3_projector", _random_pure(rng)))
    return obs


# ===========================================================================
# 5.  comparison engine
# ===========================================================================

def _rel(a, b) -> float:
    return abs(a - b) / max(1.0, abs(a), abs(b))


class Tally:
    def __init__(self):
        self.n = 0
        self.total = 0.0
        self.sq = 0.0
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.worst = None
        self.worst_rel = None

    def add(self, diff, rel, tag):
        self.n += 1
        self.total += diff
        self.sq += diff * diff
        if diff > self.max_abs:
            self.max_abs, self.worst = diff, tag
        if rel > self.max_rel:
            self.max_rel, self.worst_rel = rel, tag

    def summary(self) -> dict:
        if self.n == 0:
            return {"n": 0}
        return {"n": self.n, "max_abs": self.max_abs,
                "mean_abs": self.total / self.n,
                "rms_abs": float(np.sqrt(self.sq / self.n)),
                "max_rel": self.max_rel, "worst": self.worst,
                "worst_rel": self.worst_rel}


def _obs_cache_5q(observables):
    return [(lab, fam, FIVE.observable_5q_x(o), FIVE.observable_5q_z(o), o)
            for lab, fam, o in observables]


def primary_comparison(states, observables, q_grid, verbose=True):
    cache = _obs_cache_5q(observables)
    pauli_rows, agg_rows, recon_rows = [], [], []
    t_all, t_pauli, t_rand = Tally(), Tally(), Tally()
    t_ab, t_id, t_orig = Tally(), Tally(), Tally()
    recon = {"T_fro": 0.0, "T_max": 0.0, "T_trace": 0.0, "R_fro": 0.0,
             "worst_T": None, "worst_R": None}
    t0 = time.time()
    for si, (slabel, sfam, rho) in enumerate(states):
        data_r = np.kron(rho, rho)
        for q in q_grid:
            s5a = FIVE.state_before_final_h(data_r, q)
            s5b = FIVE.FINAL_H @ s5a @ FIVE.FINAL_H.conj().T
            s5o = FIVE.state_original(data_r, q)
            s6 = SIX.final_state_6q_t4_noisy(data_r, q)
            d5_pauli, d6_pauli = {}, {}
            fam_tally = Tally()
            for lab, fam, obs_x, obs_z, o in cache:
                d5a = _tr(obs_x, s5a)
                d5b = _tr(obs_z, s5b)
                d5orig = _tr(obs_z, s5o)
                d6, c_xx, c_yy = delta_6_from_state(s6, o)
                diff, rel = abs(d5a - d6), _rel(d5a, d6)
                tag = (slabel, lab, q)
                t_all.add(diff, rel, tag)
                t_ab.add(abs(d5a - d5b), _rel(d5a, d5b), tag)
                t_orig.add(abs(d5b - d5orig), _rel(d5b, d5orig), tag)
                if fam == "F1_pauli_product":
                    t_pauli.add(diff, rel, tag)
                    d5_pauli[lab], d6_pauli[lab] = d5a, d6
                    pauli_rows.append({
                        "state": slabel, "state_family": sfam, "q": q,
                        "observable": lab, "delta5_routeA": d5a,
                        "delta5_routeB": d5b, "delta5_original_ordering": d5orig,
                        "delta6": d6, "c_xx": c_xx, "c_yy": c_yy,
                        "abs_diff": diff, "rel_diff": rel,
                        "route_AB_abs_diff": abs(d5a - d5b),
                        "regrouped_vs_original_abs_diff": abs(d5b - d5orig)})
                    if lab == "II":
                        t_id.add(diff, rel, tag)
                else:
                    t_rand.add(diff, rel, tag)
                    fam_tally.add(diff, rel, tag)
            s = fam_tally.summary()
            agg_rows.append({
                "state": slabel, "state_family": sfam, "q": q,
                "n_observables": s["n"], "max_abs_diff": s["max_abs"],
                "mean_abs_diff": s["mean_abs"], "rms_abs_diff": s["rms_abs"],
                "max_rel_diff": s["max_rel"],
                "worst_observable": s["worst"][1] if s["worst"] else ""})
            t5 = sum(d5_pauli[lab] * pauli_product(lab)
                     for lab in PAULI_LABELS) / 4.0
            t6 = reconstruct_operator(d6_pauli)
            fro = float(np.linalg.norm(t5 - t6))
            mx = float(np.max(np.abs(t5 - t6)))
            tr5 = float(np.real(np.trace(t5)))
            tr6 = float(np.real(np.trace(t6)))
            safe = min(abs(tr5), abs(tr6)) > 1e-09
            rfro = ""
            if safe:
                rfro = float(np.linalg.norm(t5 / tr5 - t6 / tr6))
                if rfro > recon["R_fro"]:
                    recon["R_fro"], recon["worst_R"] = rfro, (slabel, q)
            if fro > recon["T_fro"]:
                recon["T_fro"], recon["worst_T"] = fro, (slabel, q)
            recon["T_max"] = max(recon["T_max"], mx)
            recon["T_trace"] = max(recon["T_trace"], abs(tr5 - tr6))
            recon_rows.append({
                "state": slabel, "state_family": sfam, "q": q,
                "trace_T5": tr5, "trace_T6": tr6,
                "trace_abs_diff": abs(tr5 - tr6), "T_frobenius_diff": fro,
                "T_max_entry_diff": mx, "denominator_safe": int(safe),
                "R_frobenius_diff": rfro})
        if verbose and (si + 1) % 10 == 0:
            print(f"      ... {si + 1}/{len(states)} states "
                  f"({time.time() - t0:.0f} s)", flush=True)
    return {"pauli_rows": pauli_rows, "agg_rows": agg_rows,
            "recon_rows": recon_rows, "all": t_all.summary(),
            "pauli": t_pauli.summary(), "random": t_rand.summary(),
            "route_ab": t_ab.summary(), "identity": t_id.summary(),
            "regrouped_vs_original": t_orig.summary(), "recon": recon,
            "seconds": time.time() - t0}


def general_r_comparison(r_states, observables, q_grid):
    cache = _obs_cache_5q(observables)
    rows = []
    tally, t_id, t_ab, t_orig = Tally(), Tally(), Tally(), Tally()
    recon = {"T_fro": 0.0, "R_fro": 0.0, "worst_T": None}
    for rlabel, rfam, data_r in r_states:
        for q in q_grid:
            s5a = FIVE.state_before_final_h(data_r, q)
            s5b = FIVE.FINAL_H @ s5a @ FIVE.FINAL_H.conj().T
            s5o = FIVE.state_original(data_r, q)
            s6 = SIX.final_state_6q_t4_noisy(data_r, q)
            fam = Tally()
            d5_pauli, d6_pauli = {}, {}
            d5_i = d6_i = 0.0
            for lab, ofam, obs_x, obs_z, o in cache:
                d5a = _tr(obs_x, s5a)
                d6 = delta_6_from_state(s6, o)[0]
                tag = (rlabel, lab, q)
                fam.add(abs(d5a - d6), _rel(d5a, d6), tag)
                tally.add(abs(d5a - d6), _rel(d5a, d6), tag)
                t_ab.add(abs(d5a - _tr(obs_z, s5b)),
                         _rel(d5a, _tr(obs_z, s5b)), tag)
                t_orig.add(abs(_tr(obs_z, s5b) - _tr(obs_z, s5o)), 0.0, tag)
                if ofam == "F1_pauli_product":
                    d5_pauli[lab], d6_pauli[lab] = d5a, d6
                    if lab == "II":
                        d5_i, d6_i = d5a, d6
                        t_id.add(abs(d5a - d6), _rel(d5a, d6), tag)
            t5 = sum(d5_pauli[lab] * pauli_product(lab)
                     for lab in PAULI_LABELS) / 4.0
            t6 = reconstruct_operator(d6_pauli)
            fro = float(np.linalg.norm(t5 - t6))
            tr5 = float(np.real(np.trace(t5)))
            tr6 = float(np.real(np.trace(t6)))
            rfro = ""
            if min(abs(tr5), abs(tr6)) > 1e-09:
                rfro = float(np.linalg.norm(t5 / tr5 - t6 / tr6))
                recon["R_fro"] = max(recon["R_fro"], rfro)
            if fro > recon["T_fro"]:
                recon["T_fro"], recon["worst_T"] = fro, (rlabel, q)
            s = fam.summary()
            rows.append({
                "r_label": rlabel, "r_family": rfam, "q": q,
                "n_observables": s["n"], "max_abs_diff": s["max_abs"],
                "mean_abs_diff": s["mean_abs"], "rms_abs_diff": s["rms_abs"],
                "max_rel_diff": s["max_rel"],
                "worst_observable": s["worst"][1] if s["worst"] else "",
                "delta5_identity": d5_i, "delta6_identity": d6_i,
                "identity_abs_diff": abs(d5_i - d6_i),
                "T_frobenius_diff": fro, "R_frobenius_diff": rfro})
    return {"rows": rows, "all": tally.summary(), "identity": t_id.summary(),
            "route_ab": t_ab.summary(),
            "regrouped_vs_original": t_orig.summary(), "recon": recon}


# ===========================================================================
# 6.  negative controls
# ===========================================================================

def negative_controls(states, q_list=(0.05, 0.2)) -> dict:
    rows = []
    summary = {}
    chosen = [s for s in states if s[0] in
              ("rand_mixed_00", "bell_Phi+", "isotropic_eps0.4",
               "imag_00_plus_i11")] or states[:3]
    paulis = [(lab, pauli_product(lab)) for lab in PAULI_LABELS]

    def _record(control, desc, slabel, q, metric, d5, d6m):
        rows.append({"control": control, "description": desc, "state": slabel,
                     "q": q, "metric": metric, "delta5": d5,
                     "delta6_modified": d6m, "abs_diff": abs(d5 - d6m)})
        summary.setdefault(control, {"description": desc, "max_abs_diff": 0.0,
                                     "n": 0})
        summary[control]["max_abs_diff"] = max(summary[control]["max_abs_diff"],
                                               abs(d5 - d6m))
        summary[control]["n"] += 1

    delta_nc = 1e-03
    skip_index = 3
    # control C: exchange control/target of one Alice-block CNOT
    flipped, done = [], False
    for kind, arg in SIX.L14_OPS:
        if kind == "CNOT" and not done and tuple(arg) == (SIX.A, SIX.A2):
            flipped.append((kind, (arg[1], arg[0])))
            done = True
        else:
            flipped.append((kind, arg))
    if not done:
        flipped = list(SIX.L14_OPS)
        for i, (kind, arg) in enumerate(flipped):
            if kind == "CNOT":
                flipped[i] = (kind, (arg[1], arg[0]))
                break

    for slabel, sfam, rho in chosen:
        data_r = np.kron(rho, rho)
        for q in q_list:
            s5a = FIVE.state_before_final_h(data_r, q)
            sched = [q] * SIX.N_CNOT
            sched[0] = q + delta_nc
            s6_a = ref_state_6q(data_r, q, q_schedule=sched)
            s6_b = ref_state_6q(data_r, q, skip_noise_after=(skip_index,))
            s6_c = ref_state_6q(data_r, q, ops=flipped)
            s6_plain = ref_state_6q(data_r, q)
            for lab, o in paulis:
                d5 = FIVE.delta_from_state_x(s5a, o)
                ox, oy = ref_obs_6q(o, _X1), ref_obs_6q(o, _Y1)
                _record("A_one_cnot_noise_q_plus_delta",
                        f"first 6Q CNOT noise set to q+{delta_nc:g}",
                        slabel, q, lab, d5, _tr(ox, s6_a) - _tr(oy, s6_a))
                _record("B_one_channel_removed",
                        f"depolarizing channel after 6Q CNOT #{skip_index} removed",
                        slabel, q, lab, d5, _tr(ox, s6_b) - _tr(oy, s6_b))
                _record("C_one_cnot_orientation_flipped",
                        "one Alice-block CNOT control/target exchanged",
                        slabel, q, lab, d5, _tr(ox, s6_c) - _tr(oy, s6_c))
                _record("D_c_xx_alone",
                        "C_XX alone compared against Delta_5 (no -C_YY)",
                        slabel, q, lab, d5, _tr(ox, s6_plain))
    for k in summary:
        summary[k]["detected"] = summary[k]["max_abs_diff"] > NC_MIN_SIGNAL
    return {"rows": rows, "summary": summary}


# ===========================================================================
# 7.  independent routes and mechanism diagnostic
# ===========================================================================

def independent_route_check(states, observables, q_list, n_states=8) -> dict:
    subset = states[:n_states]
    paulis = [(lab, pauli_product(lab)) for lab in PAULI_LABELS]
    extra = [(lab, o) for lab, fam, o in observables
             if fam != "F1_pauli_product"][:6]
    rows = []
    worst = {"five": 0.0, "six": 0.0, "six_state": 0.0,
             "six_vs_frozen_ideal_q0": 0.0}
    for slabel, sfam, rho in subset:
        data_r = np.kron(rho, rho)
        for q in q_list:
            s5_main = FIVE.state_before_final_h(data_r, q)
            s5_ref = ref_state_5q(data_r, q)
            s6_main = SIX.final_state_6q_t4_noisy(data_r, q)
            s6_ref = ref_state_6q(data_r, q)
            worst["six_state"] = max(worst["six_state"],
                                     float(np.max(np.abs(s6_main - s6_ref))))
            for lab, o in paulis + extra:
                d5m = FIVE.delta_from_state_x(s5_main, o)
                d5r = _tr(ref_obs_5q(o, _X1), s5_ref)
                d6m = delta_6_from_state(s6_main, o)[0]
                d6r = (_tr(ref_obs_6q(o, _X1), s6_ref)
                       - _tr(ref_obs_6q(o, _Y1), s6_ref))
                worst["five"] = max(worst["five"], abs(d5m - d5r))
                worst["six"] = max(worst["six"], abs(d6m - d6r))
                if q == 0.0:
                    worst["six_vs_frozen_ideal_q0"] = max(
                        worst["six_vs_frozen_ideal_q0"],
                        abs(d6m - delta_6q(final_state_6q(rho), o)[0]))
                rows.append({"state": slabel, "q": q, "observable": lab,
                             "delta5_main": d5m, "delta5_ref": d5r,
                             "delta6_main": d6m, "delta6_ref": d6r,
                             "five_route_diff": abs(d5m - d5r),
                             "six_route_diff": abs(d6m - d6r)})
    return {"rows": rows, "worst": worst, "n_states": len(subset)}


def local_coherence_diagnostic(states, q_list, n_states=8) -> dict:
    rows, worst, worst_tag = [], 0.0, None
    for slabel, sfam, rho in states[:n_states]:
        data_r = np.kron(rho, rho)
        for q in q_list:
            om5 = FIVE.ancilla_01_block(FIVE.state_before_final_h(data_r, q))
            om6 = SIX.ancilla_0011_block(SIX.final_state_6q_t4_noisy(data_r, q))
            d = float(np.max(np.abs(om5 - 2.0 * om6)))
            rows.append({"state": slabel, "q": q,
                         "max_abs_Omega5_minus_2Omega6": d,
                         "fro_Omega5": float(np.linalg.norm(om5)),
                         "fro_Omega6": float(np.linalg.norm(om6))})
            if d > worst:
                worst, worst_tag = d, (slabel, q)
    return {"rows": rows, "max_abs_diff": worst, "worst": worst_tag}


# ===========================================================================
# 8.  secondary, clearly labelled: the known Type-4 Bell-diagonal map
# ===========================================================================

def secondary_analytic_sanity_check() -> dict:
    """SECONDARY ONLY -- never enters any primary conclusion.

    Imported here, inside the function, so that the primary verification path
    above never touches ``pqec_distill.analytic_exact_map``.
    """
    try:
        from pqec_distill.analytic_exact_map import analytic_map_type4
    except Exception as exc:                                      # pragma: no cover
        return {"available": False, "reason": str(exc)}
    worst = 0.0
    rows = []
    for eps in (0.05, 0.2, 0.5, 0.8):
        rho = _isotropic(eps)
        p_in = np.array([float(np.real(v.conj() @ rho @ v)) for v in _BELL_ROWS])
        for q in (0.0, 0.01, 0.05, 0.2):
            # measured: the normalised effective operator from the 6Q circuit
            s6 = SIX.final_state_6q_t4_noisy(np.kron(rho, rho), q)
            d6 = {lab: delta_6_from_state(s6, pauli_product(lab))[0]
                  for lab in PAULI_LABELS}
            t6 = reconstruct_operator(d6)
            tr = float(np.real(np.trace(t6)))
            meas = np.array([float(np.real(v.conj() @ (t6 / tr) @ v))
                             for v in _BELL_ROWS])
            pred = np.asarray(analytic_map_type4(p_in, q), dtype=float)
            d = float(np.max(np.abs(meas - pred)))
            worst = max(worst, d)
            rows.append({"eps": eps, "q": q,
                         "measured_populations": meas.tolist(),
                         "analytic_populations": pred.tolist(),
                         "max_abs_diff": d})
    return {"available": True, "max_abs_diff": worst, "rows": rows}


# ===========================================================================
# 9.  output
# ===========================================================================

def write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT)] + list(args),
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    q_grid = Q_GRID
    states = rho_test_set(args.quick)
    observables = observable_set(args.quick)
    r_states = general_r_test_set(args.quick)

    print("=" * 74)
    print("5Q Type-4  vs  6Q-L14 : raw noisy estimator comparison")
    print("=" * 74)
    print(f"  5Q ordering : {FIVE.WIRES_5Q}")
    print(f"  6Q ordering : {SIX.WIRES_6Q}")
    print(f"  states {len(states)}   observables {len(observables)}"
          f"   q values {len(q_grid)}   general R {len(r_states)}")

    print("\n[1] structural checks")
    struct = structural_checks()
    for k in ("n_cnot_5q", "n_cnot_6q", "n_cnot_6q_alice", "n_cnot_6q_bob",
              "cross_party_two_qubit_gates_6q", "unitarity_6q",
              "local_blocks_commutator_6q", "ref_gate_vs_module_gate_5q",
              "ref_gate_vs_module_gate_6q"):
        print(f"      {k:40s} {struct[k]}")
    for k in ("alice_block_vs_ideal_fredkin", "bob_block_vs_ideal_fredkin",
              "full_6q_vs_product_of_ideal_fredkins"):
        i = struct[k]
        print(f"      {k:40s} phase {i['global_phase_over_pi']:+.6f} pi, "
              f"max {i['max_entry_residual']:.3e}")

    print("\n[2] noise-convention checks")
    noise_chk = noise_convention_checks(np.random.default_rng(7))
    for k in ("same_function_object", "convention_strings_equal",
              "package_channel_vs_ref_channel_5wires",
              "package_channel_vs_ref_channel_6wires",
              "n_channels_5q_regrouped", "n_channels_5q_original",
              "n_channels_6q", "channels_match_cnots_5q_regrouped",
              "channels_match_cnots_5q_original", "channels_match_cnots_6q",
              "q0_is_identity_channel"):
        print(f"      {k:40s} {noise_chk[k]}")

    print("\n[3] ideal (q = 0) 6Q-L14 validation")
    ideal = ideal_6q_validation(states, observables)
    for k, v in ideal["worst"].items():
        print(f"      {k:40s} {v}")

    print("\n[4] primary comparison (rho (x) rho)")
    prim = primary_comparison(states, observables, q_grid)
    print(f"      comparisons                   {prim['all']['n']}")
    print(f"      max |Delta_5 - Delta_6|       {prim['all']['max_abs']:.6e}")
    print(f"      mean                          {prim['all']['mean_abs']:.6e}")
    print(f"      rms                           {prim['all']['rms_abs']:.6e}")
    print(f"      max relative                  {prim['all']['max_rel']:.6e}")
    print(f"      worst (rho, O, q)             {prim['all']['worst']}")
    print(f"      Delta_5(I) vs Delta_6(I)      {prim['identity']['max_abs']:.6e}")
    print(f"      route A vs route B            {prim['route_ab']['max_abs']:.6e}")
    print(f"      regrouped vs ORIGINAL 5Q      "
          f"{prim['regrouped_vs_original']['max_abs']:.6e}")
    print(f"      max ||T_5 - T_6||_F           {prim['recon']['T_fro']:.6e}")
    print(f"      max ||R_5 - R_6||_F           {prim['recon']['R_fro']:.6e}")
    print(f"      elapsed                       {prim['seconds']:.1f} s")

    print("\n[5] arbitrary four-qubit R")
    gen = general_r_comparison(r_states, observables, q_grid)
    print(f"      comparisons                   {gen['all']['n']}")
    print(f"      max |Delta_5 - Delta_6|       {gen['all']['max_abs']:.6e}")
    print(f"      worst (R, O, q)               {gen['all']['worst']}")
    print(f"      Delta_5(I) vs Delta_6(I)      {gen['identity']['max_abs']:.6e}")
    print(f"      regrouped vs ORIGINAL 5Q      "
          f"{gen['regrouped_vs_original']['max_abs']:.6e}")
    print(f"      max ||T_5 - T_6||_F           {gen['recon']['T_fro']:.6e}")

    print("\n[6] independent routes")
    indep = independent_route_check(states, observables, q_grid)
    for k, v in indep["worst"].items():
        print(f"      {k:40s} {v:.6e}")

    print("\n[7] negative controls")
    nc = negative_controls(states)
    for k, v in sorted(nc["summary"].items()):
        print(f"      {k:36s} max |diff| = {v['max_abs_diff']:.6e}  "
              f"detected = {v['detected']}")

    print("\n[8] local-coherence mechanism diagnostic")
    coh = local_coherence_diagnostic(states, q_grid)
    print(f"      max |Omega_5 - 2 Omega_6|     {coh['max_abs_diff']:.6e}"
          f"   (worst {coh['worst']})")

    print("\n[9] SECONDARY sanity check against the known Type-4 Bell-diagonal "
          "map (not part of any primary conclusion)")
    sec = secondary_analytic_sanity_check()
    print(f"      {sec}" if not sec.get("available")
          else f"      max |measured - analytic| = {sec['max_abs_diff']:.6e}")

    write_csv(OUT / "arbitrary_rho_pauli.csv", prim["pauli_rows"])
    write_csv(OUT / "arbitrary_rho_random_observable.csv", prim["agg_rows"])
    write_csv(OUT / "operator_reconstruction.csv", prim["recon_rows"])
    write_csv(OUT / "general_R.csv", gen["rows"])
    write_csv(OUT / "negative_controls.csv", nc["rows"])
    write_csv(OUT / "independent_routes.csv", indep["rows"])
    write_csv(OUT / "local_coherence_diagnostic.csv", coh["rows"])
    write_csv(OUT / "ideal_q0_validation.csv", ideal["rows"])
    if sec.get("available"):
        write_csv(OUT / "secondary_analytic_sanity.csv",
                  [{"eps": r["eps"], "q": r["q"],
                    "max_abs_diff": r["max_abs_diff"],
                    "measured": " ".join(f"{x:.12e}" for x in r["measured_populations"]),
                    "analytic": " ".join(f"{x:.12e}" for x in r["analytic_populations"])}
                   for r in sec["rows"]])

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/verify_5q_6q_type4_noisy_equivalence.py",
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_head": _git("rev-parse", "HEAD"),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "platform": platform.platform(), "quick_mode": bool(args.quick),
        "question": ("does the raw noisy 5Q Type-4 estimator "
                     "Delta_5 = <X_c (x) O (x) I> equal the raw noisy 6Q-L14 "
                     "estimator Delta_6 = <X_a X_b (x) O (x) I> "
                     "- <Y_a Y_b (x) O (x) I> ?"),
        "not_used_in_primary_path": [
            "analytic Type-4 Bell-diagonal recurrence",
            "pqec_distill.analytic_exact_map",
            "closed-form noisy map",
            "previously derived Type-4 thresholds",
            "any assumption of Bell diagonality, isotropy or estimator equality"],
        "circuit_source_5q": T4.TYPE4_SOURCE,
        "circuit_source_6q": SIX.CIRCUIT_SOURCE,
        "noise_convention": FIVE.NOISE_CONVENTION,
        "tolerances": {"sanity": TOL_SANITY, "equivalence": TOL_EQUIV,
                       "negative_control_min_signal": NC_MIN_SIGNAL},
        "seeds": SEEDS, "q_grid": list(q_grid),
        "n_states": len(states), "n_observables": len(observables),
        "n_general_r": len(r_states),
        "state_families": sorted({f for _, f, _ in states}),
        "observable_families": sorted({f for _, f, _ in observables}),
        "general_r_families": sorted({f for _, f, _ in r_states}),
        "structural": struct, "noise_checks": noise_chk,
        "ideal_q0_validation": ideal["worst"],
        "primary": {k: prim[k] for k in
                    ("all", "pauli", "random", "route_ab", "identity",
                     "regrouped_vs_original")},
        "primary_reconstruction": prim["recon"],
        "general_r": {"all": gen["all"], "identity": gen["identity"],
                      "route_ab": gen["route_ab"],
                      "regrouped_vs_original": gen["regrouped_vs_original"],
                      "reconstruction": gen["recon"]},
        "independent_routes": indep["worst"],
        "negative_controls": nc["summary"],
        "local_coherence": {"max_abs_diff": coh["max_abs_diff"],
                            "worst": coh["worst"]},
        "secondary_analytic_sanity": {k: v for k, v in sec.items() if k != "rows"},
        "total_seconds": time.time() - t_start,
    }
    with (OUT / "metadata.json").open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    print(f"\n  written to {OUT.relative_to(ROOT)}")
    print(f"  total {time.time() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
