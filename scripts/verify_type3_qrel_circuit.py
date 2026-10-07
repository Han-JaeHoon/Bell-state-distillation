"""Independent circuit-level verification of the Type-3 RELATIVE
Bell-error-suppression threshold q_rel^(3)(epsilon).

SELF-CONTAINED.  This script imports nothing from ``pqec_distill``, nothing
from the parent research repository and no analytic recurrence.  The only
prior information it uses is

  1. the canonical Type-3 physical gate sequence, transcribed below from
     PQEC-Operational-Threshold/verify_analytic_decomposed.py
     (functions ``_c2`` / ``_tof`` / ``_fred``, commit a4969947,
      sha256 dd8de89220cb0b53...),
  2. the Bell-isotropic input state,
  3. the per-CNOT two-qubit replacement depolarizing channel,
  4. the parity-weighted reconstruction rule of that source,
  5. the operational definition of q_rel given in the task.

NOT fully blind: see the report.  No q_rel formula, root, relative-ratio map,
threshold table or limiting component was available or used -- none exists in
either repository -- but the author of this script had previously computed the
Type-3 ABSOLUTE threshold q_abs and knows its values and limiting component.
Nothing from that work enters the search below.

DEFINITION BEING TESTED

    R_i^(n) = p_i^(n) / p_Phi+^(n)        for i in {Phi-, Psi+, Psi-}

    q_rel(eps) = sup { q : R_i^(n+1) < R_i^(n) for every i and every n >= 0 }

The failure condition is R_i^(n+1) >= R_i^(n).  It is NOT p_i^(n+1) >= p_i^(n):
a non-target population may grow and still pass, provided the target grows
faster.  Which component is limiting, and whether the first round is binding,
are OUTPUTS here, not assumptions.

Run:
    python scripts/verify_type3_qrel_circuit.py
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "type3_qrel"

N_WIRES = 5
DIM = 2 ** N_WIRES
ANCILLA = 0
RETAIN = (1, 2)          # register A, kept
DISCARD = (3, 4)         # register B, traced out

# ===========================================================================
# 1. gate primitives (dense, 5 qubits, |q0 q1 q2 q3 q4> with q0 most significant)
# ===========================================================================

_I2 = np.eye(2, dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
_TDG = _T.conj().T
_P0 = np.array([[1, 0], [0, 0]], dtype=complex)
_P1 = np.array([[0, 0], [0, 1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)

_ONE_QUBIT = {"H": _H, "T": _T, "Tdg": _TDG}


def _kron_list(mats):
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def single_qubit_gate(name: str, wire: int) -> np.ndarray:
    factors = [_I2] * N_WIRES
    factors[wire] = _ONE_QUBIT[name]
    return _kron_list(factors)


def cnot(control: int, target: int) -> np.ndarray:
    """|0><0|_c (x) I  +  |1><1|_c (x) X_t, built from the definition."""
    if control == target:
        raise ValueError("control and target must differ")
    a = [_I2] * N_WIRES
    a[control] = _P0
    b = [_I2] * N_WIRES
    b[control] = _P1
    b[target] = _X
    return _kron_list(a) + _kron_list(b)


# ===========================================================================
# 2. the canonical Type-3 gate sequence
# ===========================================================================

def fredkin_ops(q: int, a: int, b: int) -> list:
    """Transcription of ``_fred(q, a, b) = _c2(b,a); _tof(q,a,b); _c2(b,a)``.

    ``_tof`` is the Clifford+T Toffoli with control1 = q, control2 = a,
    target = b, six CNOTs, single-qubit gates ideal.  The single-qubit gates
    stay in their original positions: the decomposition, not just the unitary,
    is what the noise sees.
    """
    ops = [("CNOT", (b, a))]                                    # _c2(b, a)
    # ---- _tof(c1=q, c2=a, t=b) ----------------------------------------
    ops += [("H", b)]
    ops += [("CNOT", (a, b)), ("Tdg", b)]
    ops += [("CNOT", (q, b)), ("T", b)]
    ops += [("CNOT", (a, b)), ("Tdg", b)]
    ops += [("CNOT", (q, b)), ("T", b), ("T", a)]
    ops += [("CNOT", (q, a)), ("H", b)]
    ops += [("T", q), ("Tdg", a)]
    ops += [("CNOT", (q, a))]
    # -------------------------------------------------------------------
    ops += [("CNOT", (b, a))]                                   # _c2(b, a)
    return ops


#: H_a, CSWAP(a; A1,B1), CSWAP(a; A2,B2), H_a -- wires (a, A1, A2, B1, B2)
TYPE3_OPS = ([("H", ANCILLA)]
             + fredkin_ops(ANCILLA, 1, 3)
             + fredkin_ops(ANCILLA, 2, 4)
             + [("H", ANCILLA)])

N_CNOT = sum(1 for kind, _ in TYPE3_OPS if kind == "CNOT")

CIRCUIT_SOURCE = ("PQEC-Operational-Threshold/verify_analytic_decomposed.py "
                  "_c2/_tof/_fred, commit a4969947, sha256 "
                  "dd8de89220cb0b5315f0c5f7b9c2c8cf483980ac96d9ad9cccc1bd3f8e7a7837")


def format_sequence() -> str:
    lines, k = [], 0
    for kind, arg in TYPE3_OPS:
        if kind == "CNOT":
            k += 1
            lines.append(f"  {len(lines)+1:3d}.  CNOT({arg[0]} -> {arg[1]})"
                         f"      [CNOT #{k:2d}]  + depolarizing D_q on "
                         f"({arg[0]},{arg[1]})")
        else:
            lines.append(f"  {len(lines)+1:3d}.  {kind}({arg})")
    return "\n".join(lines)


def build_program():
    """Merge runs of unitaries into single 32x32 matrices; keep the channel
    locations exactly where the CNOTs are."""
    program, acc = [], np.eye(DIM, dtype=complex)
    for kind, arg in TYPE3_OPS:
        if kind == "CNOT":
            acc = cnot(*arg) @ acc
            program.append(("U", acc))
            acc = np.eye(DIM, dtype=complex)
            program.append(("D", tuple(arg)))
        else:
            acc = single_qubit_gate(kind, arg) @ acc
    program.append(("U", acc))
    return program


PROGRAM = build_program()


def circuit_unitary(ops=None) -> np.ndarray:
    u = np.eye(DIM, dtype=complex)
    for kind, arg in (ops or TYPE3_OPS):
        u = (cnot(*arg) if kind == "CNOT" else single_qubit_gate(kind, arg)) @ u
    return u


# ===========================================================================
# 3. the two-qubit replacement depolarizing channel, implemented directly
# ===========================================================================

_ROW = "abcde"
_COL = "fghij"


def replacement_depol(sigma: np.ndarray, pair, q: float) -> np.ndarray:
    """D_q^(ij)(sigma) = (1-q) sigma + q [ I_ij/4 (x) Tr_ij(sigma) ]."""
    if q == 0.0:
        return sigma
    i, j = pair
    rest = [w for w in range(N_WIRES) if w not in (i, j)]
    t = sigma.reshape([2] * (2 * N_WIRES))
    row, col = list(_ROW), list(_COL)
    for w in (i, j):
        col[w] = row[w]
    sub = "".join(row) + "".join(col)
    out = "".join(row[w] for w in rest) + "".join(col[w] for w in rest)
    reduced = np.einsum(f"{sub}->{out}", t)
    target = "".join(_ROW) + "".join(_COL)
    rebuilt = np.einsum(f"{out},{_ROW[i]}{_COL[i]},{_ROW[j]}{_COL[j]}->{target}",
                        reduced, _I2, _I2)
    return (1 - q) * sigma + q * 0.25 * rebuilt.reshape(DIM, DIM)


# ===========================================================================
# 4. one Type-3 round: parity-weighted reconstruction (NOT postselection)
# ===========================================================================

PHI_P = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
PHI_M = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
PSI_P = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
PSI_M = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
BELL = {"Phi+": PHI_P, "Phi-": PHI_M, "Psi+": PSI_P, "Psi-": PSI_M}
BELL_ORDER = ("Phi+", "Phi-", "Psi+", "Psi-")
NON_TARGET = ("Phi-", "Psi+", "Psi-")

_BELL_ROWS = None


def bell_offdiagonal_weight(rho: np.ndarray) -> float:
    """C_off = sum_{i != j} |<B_i| rho |B_j>|^2, measured on the actual state."""
    global _BELL_ROWS
    if _BELL_ROWS is None:
        _BELL_ROWS = np.stack([BELL[k] for k in BELL_ORDER])
    r = _BELL_ROWS.conj() @ rho @ _BELL_ROWS.T
    return float(np.sum(np.abs(r - np.diag(np.diag(r))) ** 2))


def bell_populations(rho: np.ndarray) -> np.ndarray:
    """p_i = <B_i| rho |B_i>, all four computed independently with no symmetry
    imposed between any of them."""
    return np.array([float(np.real(BELL[k].conj() @ rho @ BELL[k]))
                     for k in BELL_ORDER])


def relative_ratios(p: np.ndarray) -> np.ndarray:
    """R_i = p_i / p_Phi+ for i = Phi-, Psi+, Psi-  (index 1, 2, 3).

    Returned in the four-slot layout with slot 0 unused (set to 1 by
    definition) so the index convention matches ``bell_populations``.
    """
    out = np.full(4, np.nan)
    out[0] = 1.0
    if p[0] <= 0.0 or not np.isfinite(p[0]):
        return out
    out[1:] = p[1:] / p[0]
    return out


def isotropic_rho(eps: float) -> np.ndarray:
    """rho_0(eps) = (1-eps)|Phi+><Phi+| + eps I/4, built from the definition."""
    e = float(eps)
    return (1 - e) * np.outer(PHI_P, PHI_P.conj()) + e * np.eye(4, dtype=complex) / 4


def one_round(rho: np.ndarray, q: float, program=None):
    """rho_(n+1) = tau_A / Tr(tau_A) with
    tau_A = Tr_B[ <0|sigma_out|0>_a - <1|sigma_out|1>_a ].

    Parity-weighted virtual purification: every ancilla outcome is used and the
    outcome parity supplies the sign.  This is NOT a physical postselection on
    ancilla outcome 0, and no success probability is involved.
    """
    sigma = np.kron(np.array([[1, 0], [0, 0]], dtype=complex),
                    np.kron(rho, rho))
    for kind, payload in (program or PROGRAM):
        if kind == "U":
            sigma = payload @ sigma @ payload.conj().T
        else:
            sigma = replacement_depol(sigma, payload, q)

    t = sigma.reshape([2] * (2 * N_WIRES))
    tau = np.zeros((4, 4), dtype=complex)
    for m, sign in ((0, +1.0), (1, -1.0)):
        block = t[m, :, :, :, :, m, :, :, :, :]
        traced = np.einsum("ijklmnkl->ijmn", block)
        tau += sign * traced.reshape(4, 4)
    weight = float(np.real(np.trace(tau)))
    herm_residue = float(np.linalg.norm(tau - tau.conj().T))
    tau = 0.5 * (tau + tau.conj().T)          # exact-arithmetic identity
    if weight == 0.0 or not np.isfinite(weight):
        raise FloatingPointError(f"parity denominator vanished: {weight!r}")
    return tau / weight, {"parity_denominator": weight,
                          "herm_residue": herm_residue}


# ===========================================================================
# 5. independent sanity checks
# ===========================================================================

def random_density_matrix(rng, dim):
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


def _reduced(sigma: np.ndarray, keep) -> np.ndarray:
    keep = sorted(keep)
    traced = [w for w in range(N_WIRES) if w not in keep]
    t = sigma.reshape([2] * (2 * N_WIRES))
    row, col = list(_ROW), list(_COL)
    for w in traced:
        col[w] = row[w]
    sub = "".join(row) + "".join(col)
    out = "".join(row[w] for w in keep) + "".join(col[w] for w in keep)
    d = 2 ** len(keep)
    return np.einsum(f"{sub}->{out}", t).reshape(d, d)


def check_noise_channel(rng, n_trials=25) -> dict:
    worst = {"trace": 0.0, "herm": 0.0, "min_eig": 0.0, "identity_at_q0": 0.0,
             "q1_pair_is_I_over_4": 0.0, "q1_rest_preserved": 0.0}
    pairs = [(0, 1), (1, 3), (2, 4), (3, 1), (0, 2), (3, 4)]
    for _ in range(n_trials):
        sigma = random_density_matrix(rng, DIM)
        for pair in pairs:
            for q in (0.0, 0.1, 0.5, 0.9, 1.0):
                out = replacement_depol(sigma, pair, q)
                worst["trace"] = max(worst["trace"], abs(np.trace(out) - 1.0))
                worst["herm"] = max(worst["herm"],
                                    np.linalg.norm(out - out.conj().T))
                worst["min_eig"] = min(worst["min_eig"], float(
                    np.linalg.eigvalsh(0.5 * (out + out.conj().T)).min()))
            worst["identity_at_q0"] = max(
                worst["identity_at_q0"],
                float(np.max(np.abs(replacement_depol(sigma, pair, 0.0) - sigma))))
            out1 = replacement_depol(sigma, pair, 1.0)
            rest = [w for w in range(N_WIRES) if w not in pair]
            worst["q1_pair_is_I_over_4"] = max(
                worst["q1_pair_is_I_over_4"],
                float(np.max(np.abs(_reduced(out1, list(pair)) - np.eye(4) / 4))))
            worst["q1_rest_preserved"] = max(
                worst["q1_rest_preserved"],
                float(np.max(np.abs(_reduced(out1, rest) - _reduced(sigma, rest)))))
    return {k: float(v) for k, v in worst.items()}


def check_circuit_unitary() -> dict:
    """The noiseless 16-CNOT sequence must implement
    H_a CSWAP(a;A2,B2) CSWAP(a;A1,B1) H_a up to a global phase."""
    def cswap(control, x, y):
        a = [_I2] * N_WIRES
        a[control] = _P0
        keep = _kron_list(a)
        perm = np.arange(DIM)
        for idx in range(DIM):
            bits = [(idx >> (N_WIRES - 1 - w)) & 1 for w in range(N_WIRES)]
            bits[x], bits[y] = bits[y], bits[x]
            perm[idx] = sum(b << (N_WIRES - 1 - w) for w, b in enumerate(bits))
        swap = np.eye(DIM, dtype=complex)[:, perm]
        b = [_I2] * N_WIRES
        b[control] = _P1
        return keep + _kron_list(b) @ swap

    target = single_qubit_gate("H", ANCILLA)
    target = cswap(ANCILLA, 1, 3) @ target
    target = cswap(ANCILLA, 2, 4) @ target
    target = single_qubit_gate("H", ANCILLA) @ target
    v = circuit_unitary()
    phase = np.exp(-1j * np.angle(np.trace(target.conj().T @ v)))
    diff = phase * v - target
    return {"max_abs_unitary_diff": float(np.max(np.abs(diff))),
            "frobenius_unitary_diff": float(np.linalg.norm(diff)),
            "unitarity": float(np.max(np.abs(v.conj().T @ v - np.eye(DIM)))),
            "global_phase_over_pi": float(
                np.angle(np.trace(target.conj().T @ v)) / np.pi)}


def check_noiseless_purification(rng, n_trials=30) -> dict:
    """At q = 0 the parity-weighted output must be rho^2 / Tr(rho^2), and the
    parity denominator must be the purity.  This is an ideal-purification
    identity, not an analytic threshold result."""
    worst = worst_weight = worst_trace_dist = worst_herm = 0.0
    worst_eig = np.inf
    for _ in range(n_trials):
        rho = random_density_matrix(rng, 4)
        out, info = one_round(rho, 0.0)
        expected = rho @ rho / np.trace(rho @ rho)
        worst = max(worst, float(np.max(np.abs(out - expected))))
        d = out - expected
        worst_trace_dist = max(worst_trace_dist, 0.5 * float(
            np.sum(np.abs(np.linalg.eigvalsh(0.5 * (d + d.conj().T))))))
        worst_weight = max(worst_weight, abs(info["parity_denominator"]
                                             - float(np.real(np.trace(rho @ rho)))))
        worst_herm = max(worst_herm, float(np.linalg.norm(out - out.conj().T)))
        worst_eig = min(worst_eig, float(np.linalg.eigvalsh(out).min()))
    return {"max_abs_diff_vs_rho2_over_tr_rho2": worst,
            "max_trace_distance": worst_trace_dist,
            "max_abs_denominator_minus_purity": worst_weight,
            "max_hermiticity_error": worst_herm,
            "min_eigenvalue": float(worst_eig),
            "n_random_trials": n_trials}


def check_state_validity(eps, q, n_rounds=40) -> dict:
    rho = isotropic_rho(eps)
    worst = {"herm": 0.0, "trace": 0.0, "min_eig": np.inf,
             "pop_sum_minus_1": 0.0, "tau_herm_residue": 0.0,
             "min_parity_denominator": np.inf, "min_p_phi_plus": np.inf}
    for n in range(n_rounds + 1):
        p = bell_populations(rho)
        worst["herm"] = max(worst["herm"],
                            float(np.linalg.norm(rho - rho.conj().T)))
        worst["trace"] = max(worst["trace"], abs(float(np.trace(rho).real) - 1.0))
        worst["min_eig"] = min(worst["min_eig"],
                               float(np.linalg.eigvalsh(rho).min()))
        worst["pop_sum_minus_1"] = max(worst["pop_sum_minus_1"],
                                       abs(float(p.sum()) - 1.0))
        worst["min_p_phi_plus"] = min(worst["min_p_phi_plus"], float(p[0]))
        if n < n_rounds:
            rho, info = one_round(rho, q)
            worst["tau_herm_residue"] = max(worst["tau_herm_residue"],
                                            info["herm_residue"])
            worst["min_parity_denominator"] = min(
                worst["min_parity_denominator"], info["parity_denominator"])
    return {k: float(v) for k, v in worst.items()}


# ===========================================================================
# 6. the RELATIVE-suppression predicate
# ===========================================================================
#
# Tolerances.  The round-off floor on the ratios is MEASURED at run time by
# `measure_error_floor()` and printed with the results; it is not assumed.
#
#   STEP_TOL  = 1e-14   stop once max|rho_(n+1) - rho_n| falls below this: the
#                       orbit is then at the numerical fixed point and any
#                       further sign change in Delta R is round-off.
#   VIOL_TOL  = 1e-12   a round counts as violating only if some non-target
#                       Delta R_i exceeds +VIOL_TOL.
#
# OFFDIAG_TRUST is the round-off contamination guard.  The Bell-isotropic input
# is built exactly Bell-diagonal, so its Bell off-diagonal weight starts at
# round-off; if the circuit map amplifies that transverse direction the
# accumulated round-off can push the simulated orbit off a saddle and
# manufacture a ratio increase the exact dynamics does not have.  Whether this
# map amplifies it is MEASURED by `transverse_growth()` below, not assumed.

STEP_TOL = 1e-14
VIOL_TOL = 1e-12
N_MAX = 5000
OFFDIAG_TRUST = 1e-18


def run_trajectory(eps: float, q: float, step_tol=STEP_TOL, n_max=N_MAX,
                   viol_tol=VIOL_TOL, components=NON_TARGET, record=False,
                   offdiag_trust=OFFDIAG_TRUST):
    """Iterate the real circuit and look for the first RELATIVE violation.

    The failure condition is Delta R_i = R_i^(n+1) - R_i^(n) > viol_tol.  It is
    deliberately NOT a condition on Delta p_i: a non-target population may grow
    while its ratio to the target falls, and that passes.
    """
    idx = [BELL_ORDER.index(c) for c in components]
    rho = isotropic_rho(eps)
    p = bell_populations(rho)
    r = relative_ratios(p)
    c_off0 = bell_offdiagonal_weight(rho)
    rows = [{"n": 0, "p": p.copy(), "R": r.copy(), "dp": np.full(4, np.nan),
             "dR": np.full(4, np.nan), "step": np.nan, "c_off": c_off0,
             "parity_denominator": np.nan}] if record else []
    worst_dr, worst_at, worst_comp = -np.inf, None, None
    c_off_max = c_off0
    n, step, trusted, n_trusted, converged = 0, np.nan, True, None, False
    # does a raw population ever rise while its ratio falls?  (section 19)
    abs_rise_with_rel_fall = []
    while n < n_max:
        rho_next, info = one_round(rho, q)
        p_next = bell_populations(rho_next)
        r_next = relative_ratios(p_next)
        dp = p_next - p
        dr = r_next - r
        step = float(np.max(np.abs(rho_next - rho)))
        c_off = bell_offdiagonal_weight(rho_next)
        c_off_max = max(c_off_max, c_off)
        if record:
            rows.append({"n": n + 1, "p": p_next.copy(), "R": r_next.copy(),
                         "dp": dp.copy(), "dR": dr.copy(), "step": step,
                         "c_off": c_off,
                         "parity_denominator": info["parity_denominator"]})
        for i in idx:
            if dp[i] > viol_tol and dr[i] < -viol_tol:
                abs_rise_with_rel_fall.append(
                    {"round": n, "component": BELL_ORDER[i],
                     "delta_p": float(dp[i]), "delta_R": float(dr[i]),
                     "p_n": float(p[i]), "p_np1": float(p_next[i]),
                     "p_phi_plus_n": float(p[0]),
                     "p_phi_plus_np1": float(p_next[0])})
            if dr[i] > worst_dr:
                worst_dr, worst_at, worst_comp = float(dr[i]), n, BELL_ORDER[i]
        hits = [i for i in idx if dr[i] > viol_tol]
        if hits:
            first = max(hits, key=lambda i: dr[i])
            return {"violated": True, "round": n, "component": BELL_ORDER[first],
                    "p_n": p.copy(), "p_np1": p_next.copy(),
                    "R_n": r.copy(), "R_np1": r_next.copy(),
                    "delta_p": dp.copy(), "delta_R": dr.copy(),
                    "delta_R_violating": float(dr[first]),
                    "delta_p_of_violating": float(dp[first]),
                    "all_violating": [BELL_ORDER[i] for i in hits],
                    "parity_denominator": info["parity_denominator"],
                    "rounds_run": n + 1, "final_step": step, "trusted": True,
                    "n_trusted": None, "converged": False,
                    "c_off_initial": c_off0, "c_off_final": c_off,
                    "c_off_max": c_off_max, "worst_delta_R": worst_dr,
                    "worst_round": worst_at, "worst_component": worst_comp,
                    "abs_rise_with_rel_fall": abs_rise_with_rel_fall,
                    "rows": rows}
        rho, p, r = rho_next, p_next, r_next
        n += 1
        if step < step_tol:
            converged = True
            break
        if c_off > offdiag_trust:
            trusted, n_trusted = False, n
            break
    return {"violated": False, "round": None, "component": None,
            "rounds_run": n, "final_step": step, "trusted": trusted,
            "n_trusted": n_trusted, "converged": converged,
            "c_off_initial": c_off0, "c_off_final": bell_offdiagonal_weight(rho),
            "c_off_max": c_off_max, "worst_delta_R": worst_dr,
            "worst_round": worst_at, "worst_component": worst_comp,
            "p_final": p.copy(), "R_final": r.copy(),
            "abs_rise_with_rel_fall": abs_rise_with_rel_fall, "rows": rows}


def record_trajectory(eps: float, q: float, n_rounds: int,
                      viol_tol=VIOL_TOL, components=NON_TARGET):
    """Round-by-round record for plotting; does not stop at the first
    violation, so the figure can show what happens afterwards."""
    idx = [BELL_ORDER.index(c) for c in components]
    rho = isotropic_rho(eps)
    p = bell_populations(rho)
    r = relative_ratios(p)
    rows = [{"n": 0, "p": p.copy(), "R": r.copy(), "dp": np.full(4, np.nan),
             "dR": np.full(4, np.nan), "c_off": bell_offdiagonal_weight(rho),
             "parity_denominator": np.nan}]
    first_violation = None
    for n in range(n_rounds):
        rho, info = one_round(rho, q)
        p_next = bell_populations(rho)
        r_next = relative_ratios(p_next)
        dp, dr = p_next - p, r_next - r
        if first_violation is None and any(dr[i] > viol_tol for i in idx):
            k = max((i for i in idx if dr[i] > viol_tol), key=lambda i: dr[i])
            first_violation = {"round": n, "component": BELL_ORDER[k],
                               "delta_R": float(dr[k]),
                               "delta_p": float(dp[k])}
        rows.append({"n": n + 1, "p": p_next.copy(), "R": r_next.copy(),
                     "dp": dp.copy(), "dR": dr.copy(),
                     "c_off": bell_offdiagonal_weight(rho),
                     "parity_denominator": info["parity_denominator"]})
        p, r = p_next, r_next
    return rows, first_violation


def first_round_step(eps: float, q: float):
    """rho_0 -> rho_1 only; returns (p0, p1, R0, R1, parity denominator)."""
    rho = isotropic_rho(eps)
    p0 = bell_populations(rho)
    rho1, info = one_round(rho, q)
    p1 = bell_populations(rho1)
    return p0, p1, relative_ratios(p0), relative_ratios(p1), \
        info["parity_denominator"]


def first_round_violates(eps: float, q: float, viol_tol=VIOL_TOL,
                         components=NON_TARGET) -> bool:
    _p0, _p1, r0, r1, _w = first_round_step(eps, q)
    return any((r1 - r0)[BELL_ORDER.index(c)] > viol_tol for c in components)


def measure_error_floor(eps=0.15, q=0.02, n_settle=60, n_watch=40) -> dict:
    """Empirical round-off floor on the ratios once the orbit has settled."""
    rho = isotropic_rho(eps)
    for _ in range(n_settle):
        rho, _ = one_round(rho, q)
    worst_dr = worst_dp = worst_step = 0.0
    p = bell_populations(rho)
    r = relative_ratios(p)
    for _ in range(n_watch):
        rho_next, _ = one_round(rho, q)
        p_next = bell_populations(rho_next)
        r_next = relative_ratios(p_next)
        worst_dr = max(worst_dr, float(np.max(np.abs(r_next[1:] - r[1:]))))
        worst_dp = max(worst_dp, float(np.max(np.abs(p_next - p))))
        worst_step = max(worst_step, float(np.max(np.abs(rho_next - rho))))
        rho, p, r = rho_next, p_next, r_next
    return {"max_abs_delta_R_at_fixed_point": worst_dr,
            "max_abs_delta_p_at_fixed_point": worst_dp,
            "max_abs_step_at_fixed_point": worst_step, "eps": eps, "q": q}


# ===========================================================================
# 7. threshold search -- bracket, then bisect, on the circuit predicate only
# ===========================================================================

def bracket(predicate, q_lo=0.0, q_hi=0.9, n_coarse=90):
    grid = np.linspace(q_lo, q_hi, n_coarse + 1)
    flags = [predicate(float(q)) for q in grid]
    changes = [k for k in range(len(flags) - 1) if flags[k] != flags[k + 1]]
    if not changes:
        return None, None, {"monotone": True, "n_sign_changes": 0,
                            "all_violating": all(flags),
                            "none_violating": not any(flags)}
    k = changes[0]
    return float(grid[k]), float(grid[k + 1]), {
        "monotone": len(changes) == 1, "n_sign_changes": len(changes),
        "scan_grid_step": float(grid[1] - grid[0])}


def bisect(predicate, lo: float, hi: float, tol=1e-10, max_iter=200):
    """lo satisfies, hi violates; return the largest q known to satisfy."""
    assert not predicate(lo) and predicate(hi)
    it = 0
    while hi - lo > tol and it < max_iter:
        mid = 0.5 * (lo + hi)
        if predicate(mid):
            hi = mid
        else:
            lo = mid
        it += 1
    return lo, hi, it


def _hinted_bracket(pred, hint, q_hi, n_coarse):
    """Look near ``hint`` first, then fall back to the full scan.  The bracket
    is re-verified before bisecting, so a hint can never change the answer."""
    if hint is not None:
        lo_h, hi_h = max(0.0, hint - 0.03), min(q_hi, hint + 0.03)
        if not pred(lo_h) and pred(hi_h):
            return lo_h, hi_h, {"monotone": None, "n_sign_changes": None,
                                "from_hint": True}
    return bracket(pred, 0.0, q_hi, n_coarse)


def q_rel_full(eps: float, viol_tol=VIOL_TOL, tol=1e-10, components=NON_TARGET,
               q_hi=0.9, n_coarse=90, hint=None):
    pred = lambda q: run_trajectory(eps, q, viol_tol=viol_tol,
                                    components=components)["violated"]
    lo, hi, info = _hinted_bracket(pred, hint, q_hi, n_coarse)
    if lo is None:
        return {"q_rel": None, "bracket_info": info}
    a, b, it = bisect(pred, lo, hi, tol=tol)
    detail = run_trajectory(eps, b, viol_tol=viol_tol, components=components)
    at_lo = run_trajectory(eps, a, viol_tol=viol_tol, components=components)
    return {"q_rel": a, "q_upper": b, "bracket": (lo, hi), "iterations": it,
            "bracket_width": b - a, "bracket_info": info,
            "limiting_component": detail["component"],
            "first_violation_round": detail["round"],
            "delta_R_at_violation": detail.get("delta_R_violating"),
            "delta_p_of_violating": detail.get("delta_p_of_violating"),
            "parity_denominator_at_violation": detail.get("parity_denominator"),
            "all_violating_at_q_upper": detail.get("all_violating"),
            # NOTE: these belong to the q_upper run, NOT to q_rel itself.
            "p_n_at_q_upper": detail.get("p_n"),
            "p_np1_at_q_upper": detail.get("p_np1"),
            "R_n_at_q_upper": detail.get("R_n"),
            "R_np1_at_q_upper": detail.get("R_np1"),
            "below_converged": at_lo["converged"],
            "below_trusted": at_lo["trusted"],
            "below_n_trusted": at_lo["n_trusted"],
            "below_rounds_run": at_lo["rounds_run"],
            "below_c_off_max": at_lo["c_off_max"],
            "below_worst_delta_R": at_lo["worst_delta_R"],
            "below_worst_component": at_lo["worst_component"],
            "below_worst_round": at_lo["worst_round"],
            "below_abs_rise_with_rel_fall": at_lo["abs_rise_with_rel_fall"]}


def q_rel_first_round(eps: float, viol_tol=VIOL_TOL, tol=1e-10,
                      components=NON_TARGET, q_hi=0.9, n_coarse=90, hint=None):
    pred = lambda q: first_round_violates(eps, q, viol_tol, components)
    lo, hi, info = _hinted_bracket(pred, hint, q_hi, n_coarse)
    if lo is None:
        return {"q_rel": None, "bracket_info": info}
    a, b, it = bisect(pred, lo, hi, tol=tol)
    _p0, _p1, r0, r1, w = first_round_step(eps, b)
    dr = r1 - r0
    hits = [BELL_ORDER[i] for i in (1, 2, 3) if dr[i] > viol_tol]
    return {"q_rel": a, "q_upper": b, "iterations": it, "bracket_width": b - a,
            "bracket_info": info, "parity_denominator_at_q_upper": w,
            "limiting_component": (max(hits, key=lambda c: dr[BELL_ORDER.index(c)])
                                   if hits else None),
            "all_violating_components": hits}


def per_component_thresholds(eps: float, viol_tol=VIOL_TOL, tol=1e-10) -> dict:
    """Relative threshold for each non-target component on its own, for the
    first round alone and over all rounds.  The limiting component is the one
    with the smallest value -- an output, not an input."""
    first, full, detail = {}, {}, {}
    for c in NON_TARGET:
        first[c] = q_rel_first_round(eps, viol_tol=viol_tol, tol=tol,
                                     components=(c,))["q_rel"]
        r = q_rel_full(eps, viol_tol=viol_tol, tol=tol, components=(c,))
        full[c] = r["q_rel"]
        detail[c] = {"round": r.get("first_violation_round"),
                     "delta_R": r.get("delta_R_at_violation")}
    return {"first_round": first, "all_rounds": full, "all_round_detail": detail}


def tolerance_study(eps_values, tols=(1e-10, 1e-12, 1e-14), bisect_tol=1e-11) -> dict:
    out = {}
    for eps in eps_values:
        vals = {}
        for t in tols:
            vals[f"{t:g}"] = q_rel_full(eps, viol_tol=t, tol=bisect_tol)["q_rel"]
        finite = [v for v in vals.values() if v is not None]
        out[f"{eps:g}"] = {"values": vals,
                           "spread": (max(finite) - min(finite)) if finite else None}
    return out


def near_threshold_scan(eps: float, q_star: float,
                        deltas=(1e-7, 1e-6, 1e-4, 1e-3)) -> list:
    rows = []
    for d in deltas:
        for side, q in (("below", q_star - d), ("above", q_star + d)):
            if q < 0:
                continue
            r = run_trajectory(eps, float(q))
            row = {"epsilon": eps, "delta": d, "side": side, "q": float(q),
                   "violated": r["violated"], "violation_round": r["round"],
                   "violating_component": r["component"],
                   "delta_R_violating": r.get("delta_R_violating"),
                   "delta_p_of_violating": r.get("delta_p_of_violating"),
                   "parity_denominator": r.get("parity_denominator"),
                   "worst_delta_R": r["worst_delta_R"],
                   "worst_component": r["worst_component"],
                   "worst_round": r["worst_round"],
                   "rounds_run": r["rounds_run"], "converged": r["converged"],
                   "trusted": r["trusted"], "n_trusted": r["n_trusted"],
                   "c_off_max": r["c_off_max"],
                   "n_abs_rise_with_rel_fall": len(r["abs_rise_with_rel_fall"])}
            if r["violated"]:
                row.update({"p_phi_plus_n": float(r["p_n"][0]),
                            "p_violating_n": float(
                                r["p_n"][BELL_ORDER.index(r["component"])]),
                            "p_violating_np1": float(
                                r["p_np1"][BELL_ORDER.index(r["component"])]),
                            "R_violating_n": float(
                                r["R_n"][BELL_ORDER.index(r["component"])]),
                            "R_violating_np1": float(
                                r["R_np1"][BELL_ORDER.index(r["component"])])})
            else:
                row.update({"p_phi_plus_n": float(r["p_final"][0]),
                            "p_violating_n": None, "p_violating_np1": None,
                            "R_violating_n": None, "R_violating_np1": None})
            rows.append(row)
    return rows


def first_round_absolute_rise(eps: float, q: float, viol_tol=VIOL_TOL,
                             components=NON_TARGET) -> bool:
    """Does some non-target RAW population increase in the first round?

    This is the absolute-criterion failure condition, measured here only so the
    window where the absolute criterion has already failed while the relative
    one still holds can be located from this script's own circuit runs.  No
    previously computed absolute threshold is read or used.
    """
    _p0, _p1, _r0, _r1, _w = first_round_step(eps, q)
    p0, p1, _r0b, _r1b, _wb = first_round_step(eps, q)
    dp = p1 - p0
    return any(dp[BELL_ORDER.index(c)] > viol_tol for c in components)


def absolute_rise_onset(eps: float, viol_tol=VIOL_TOL, tol=1e-10,
                        q_hi=0.9, n_coarse=90) -> dict:
    """Largest q for which no non-target raw population rises in round 0."""
    pred = lambda q: first_round_absolute_rise(eps, q, viol_tol)
    lo, hi, info = bracket(pred, 0.0, q_hi, n_coarse)
    if lo is None:
        return {"q_onset": None, "bracket_info": info}
    a, b, it = bisect(pred, lo, hi, tol=tol)
    p0, p1, r0, r1, _w = first_round_step(eps, b)
    dp, dr = p1 - p0, r1 - r0
    rising = [BELL_ORDER[i] for i in (1, 2, 3) if dp[i] > viol_tol]
    return {"q_onset": a, "q_upper": b, "iterations": it,
            "bracket_info": info, "rising_components": rising,
            "first_rising_component": (max(rising,
                                           key=lambda c: dp[BELL_ORDER.index(c)])
                                       if rising else None),
            "delta_p_at_onset": float(max(dp[1:])),
            "delta_R_at_onset": float(max(dr[1:]))}


# ===========================================================================
# 8. transverse stability of the Bell-diagonal manifold (separate diagnostic)
# ===========================================================================

TRANSVERSE_SEED = 1e-12
TRANSVERSE_ROUNDS = 60
TRANSVERSE_FIT = (10, 40)
BELL_PAIRS = (("Phi+", "Phi-"), ("Phi+", "Psi+"), ("Phi-", "Psi-"),
              ("Psi+", "Psi-"))


def seeded_rho(eps: float, seed: float, pair=("Phi+", "Phi-")) -> np.ndarray:
    u, v = BELL[pair[0]], BELL[pair[1]]
    delta = np.outer(u, v.conj()) + np.outer(v, u.conj())
    return isotropic_rho(eps) + seed * delta


def transverse_growth(eps: float, q: float, seed=TRANSVERSE_SEED,
                      n_rounds=TRANSVERSE_ROUNDS, fit=TRANSVERSE_FIT,
                      pair=("Phi+", "Phi-")) -> dict:
    """Geometric-mean amplitude multiplier per round of the off-diagonal part,
    measured on the actual physical circuit."""
    rho = seeded_rho(eps, seed, pair)
    eig0 = float(np.linalg.eigvalsh(rho).min())
    amps = [float(np.sqrt(bell_offdiagonal_weight(rho)))]
    for _ in range(n_rounds):
        rho, _ = one_round(rho, q)
        amps.append(float(np.sqrt(bell_offdiagonal_weight(rho))))
    a = np.array(amps)
    lo, hi = fit[0], min(fit[1], len(a) - 1)
    rate = (float("nan") if a[lo] <= 0 or a[hi] <= 0 or hi <= lo
            else float((a[hi] / a[lo]) ** (1.0 / (hi - lo))))
    return {"epsilon": eps, "q": q, "seed": seed, "growth_rate": rate,
            "a_initial": a[0], "a_final": a[-1], "amps": a,
            "fp_locked": bool(a[hi] == a[lo]),
            "amplitude_at_floor": bool(a[hi] < 1e-15),
            "seeded_state_min_eig": eig0, "pair": f"{pair[0]}/{pair[1]}"}


# ===========================================================================
# 9. plotting (self-contained: only matplotlib, no project style module)
# ===========================================================================

BELL_COLOR = {"Phi+": "#2a78d6", "Phi-": "#eb6834",
              "Psi+": "#1baf7a", "Psi-": "#eda100"}
BELL_MARKER = {"Phi+": "o", "Phi-": "s", "Psi+": "^", "Psi-": "D"}
BELL_TEX = {"Phi+": r"$\Phi^{+}$", "Phi-": r"$\Phi^{-}$",
            "Psi+": r"$\Psi^{+}$", "Psi-": r"$\Psi^{-}$"}
INK, INK2, GRID = "#101010", "#4a4a4a", "#d9d9d9"


def use_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.bbox": "tight", "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11,
        "axes.titlesize": 11.5, "axes.labelsize": 11.5,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "axes.facecolor": "white", "axes.edgecolor": GRID,
        "axes.labelcolor": INK, "axes.linewidth": 0.9,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.8,
        "xtick.color": INK2, "ytick.color": INK2,
        "lines.linewidth": 1.8, "lines.markersize": 4.5,
        "legend.frameon": False,
    })


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300 if ext == "png" else None)
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_curve(dense, table):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    _despine(ax)
    ax.plot([r["epsilon"] for r in dense], [r["q_rel_full"] for r in dense],
            color=BELL_COLOR["Phi+"], linewidth=2.0)
    ax.plot([r["epsilon"] for r in table], [r["q_rel_full"] for r in table],
            linestyle="none", marker="o", markersize=6, markerfacecolor="white",
            markeredgecolor=BELL_COLOR["Phi+"], markeredgewidth=1.6,
            label="high-precision points")
    ax.set_xlabel(r"input mixing $\epsilon$")
    ax.set_ylabel(r"$q_{\mathrm{rel}}^{(3)}(\epsilon)$")
    ax.set_xlim(0, max(r["epsilon"] for r in dense))
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", labelcolor=INK)
    ax.set_title("Type 3: relative Bell-error-suppression threshold\n"
                 "(from the 16-CNOT circuit simulation)", color=INK, pad=8)
    save(fig, "type3_qrel_curve")


def plot_first_vs_full(dense):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7),
                             gridspec_kw=dict(wspace=0.30))
    for ax in axes:
        _despine(ax)
    e = np.array([r["epsilon"] for r in dense])
    full = np.array([r["q_rel_full"] for r in dense], dtype=float)
    first = np.array([r["q_rel_first"] for r in dense], dtype=float)
    axes[0].plot(e, full, color=BELL_COLOR["Phi+"], linewidth=3.4, alpha=0.85,
                 label=r"$q_{\mathrm{rel}}$  (all rounds)")
    axes[0].plot(e, first, color=BELL_COLOR["Phi-"], linewidth=1.4,
                 linestyle=(0, (5, 3)),
                 label=r"$q_{\mathrm{rel}}$  (first round only)")
    axes[0].set_ylabel(r"$q_{\mathrm{rel}}^{(3)}$")
    axes[0].legend(loc="upper left", labelcolor=INK)
    axes[0].set_ylim(0, None)
    d = full - first
    axes[1].axhline(0.0, color=INK2, linewidth=0.9)
    axes[1].plot(e, d, color=BELL_COLOR["Psi+"], marker="o", markersize=3.6,
                 markeredgecolor="white", linewidth=1.3)
    axes[1].set_ylabel(r"$q^{\rm all\ rounds} - q^{\rm first\ round}$")
    lim = max(float(np.max(np.abs(d))), 1e-12)
    axes[1].set_ylim(-1.6 * lim, 1.6 * lim)
    axes[1].annotate(rf"$\max_\epsilon |q^{{\rm all}} - q^{{\rm first}}|"
                     rf" = {np.max(np.abs(d)):.1e}$" f"\n({len(e)} grid points)",
                     xy=(0.04, 0.84), xycoords="axes fraction", fontsize=9.5,
                     color=INK, va="top")
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.set_xlabel(r"input mixing $\epsilon$")
        ax.set_xlim(0, float(e.max()))
        ax.text(-0.15, 1.05, tag, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=INK)
    fig.suptitle("Type 3: all-round versus first-round relative threshold",
                 fontsize=13, color=INK, y=1.02)
    save(fig, "type3_qrel_first_vs_full")


def plot_components(table):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7),
                             gridspec_kw=dict(wspace=0.30))
    for ax in axes:
        _despine(ax)
    e = [r["epsilon"] for r in table]
    firsts = {"Phi-": "q_Phi_minus_rel_first", "Psi+": "q_Psi_plus_rel_first",
              "Psi-": "q_Psi_minus_rel_first"}
    alls = {"Phi-": "q_Phi_minus_rel_all", "Psi+": "q_Psi_plus_rel_all",
            "Psi-": "q_Psi_minus_rel_all"}
    for ax, keys, title in ((axes[0], firsts, "first-round definition"),
                            (axes[1], alls, "standalone all-round definition")):
        for name, key in keys.items():
            style = dict(linewidth=1.4, linestyle=(0, (5, 3))) if name == "Psi-" \
                else dict(linewidth=1.9)
            ax.plot(e, [r[key] for r in table], color=BELL_COLOR[name],
                    marker=BELL_MARKER[name], markeredgecolor="white",
                    label=BELL_TEX[name], **style)
        ax.set_xlabel(r"input mixing $\epsilon$")
        ax.set_ylabel("per-component relative threshold")
        ax.set_ylim(0, None)
        ax.legend(loc="upper left", labelcolor=INK)
        ax.set_title(title, color=INK, fontsize=11)
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.text(-0.15, 1.05, tag, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=INK)
    fig.suptitle("Type 3: which Bell component sets the relative threshold",
                 fontsize=13, color=INK, y=1.02)
    save(fig, "type3_qrel_components")


def plot_eps015_ratios(cases, eps, q_star):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(cases), figsize=(4.1 * len(cases), 7.4),
                             gridspec_kw=dict(hspace=0.30, wspace=0.32))
    for col, (label, q, rows, viol) in enumerate(cases):
        n = np.array([r["n"] for r in rows])
        ratios = np.array([r["R"] for r in rows])
        top, bot = axes[0][col], axes[1][col]
        for k in (1, 2, 3):
            name = BELL_ORDER[k]
            style = dict(linewidth=1.5, linestyle=(0, (5, 3))) if name == "Psi-" \
                else dict(linewidth=1.8)
            top.plot(n, ratios[:, k], color=BELL_COLOR[name],
                     marker=BELL_MARKER[name], markevery=(k, max(1, len(n) // 8)),
                     markeredgecolor="white", label=BELL_TEX[name], **style)
            bot.plot(n[1:], np.diff(ratios[:, k]), color=BELL_COLOR[name],
                     marker=BELL_MARKER[name], markevery=(k, max(1, len(n) // 8)),
                     markeredgecolor="white", label=BELL_TEX[name], **style)
        bot.axhline(0.0, color=INK2, linewidth=0.9)
        top.set_yscale("log")
        dq = q - q_star
        top.set_title(f"{label}\n$q = q_{{\\mathrm{{rel}}}} {dq:+.1e}$",
                      color=INK, fontsize=10.5, pad=6)
        bot.set_xlabel(r"purification round $n$")
        if col == 0:
            top.set_ylabel(r"$R_i = p_i / p_{\Phi^+}$")
            bot.set_ylabel(r"$\Delta R_i$  (failure if $> 0$)")
        for ax in (top, bot):
            _despine(ax)
            ax.set_xlim(0, n[-1])
        if viol is not None:
            for ax in (top, bot):
                ax.axvline(viol["round"], color=INK2, linestyle=(0, (4, 3)),
                           linewidth=1.1)
            bot.annotate(f"first violation: {BELL_TEX[viol['component']]}"
                         f" at $n={viol['round']}$\n"
                         rf"$\Delta R = {viol['delta_R']:+.2e}$",
                         xy=(0.04, 0.92), xycoords="axes fraction",
                         fontsize=9, color=INK, va="top")
        else:
            bot.annotate("no violation:\nall three ratios fall\nat every round",
                         xy=(0.40, 0.42), xycoords="axes fraction",
                         fontsize=9, color=INK, va="top")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.03), labelcolor=INK)
    fig.suptitle(rf"Type 3 at $\epsilon = {eps:g}$: relative ratios around "
                 rf"$q_{{\mathrm{{rel}}}} = {q_star:.10f}$",
                 fontsize=13, color=INK, y=0.985)
    save(fig, "type3_qrel_eps015_ratios")


def plot_eps015_populations(cases, eps, q_star):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(cases), figsize=(4.1 * len(cases), 7.4),
                             gridspec_kw=dict(hspace=0.30, wspace=0.32))
    for col, (label, q, rows, viol) in enumerate(cases):
        n = np.array([r["n"] for r in rows])
        pops = np.array([r["p"] for r in rows])
        top, bot = axes[0][col], axes[1][col]
        for k, name in enumerate(BELL_ORDER):
            style = dict(linewidth=1.5, linestyle=(0, (5, 3))) if name == "Psi-" \
                else dict(linewidth=1.8)
            top.plot(n, pops[:, k], color=BELL_COLOR[name],
                     marker=BELL_MARKER[name], markevery=(k, max(1, len(n) // 8)),
                     markeredgecolor="white", label=BELL_TEX[name], **style)
            if name != "Phi+":
                bot.plot(n, pops[:, k], color=BELL_COLOR[name],
                         marker=BELL_MARKER[name],
                         markevery=(k, max(1, len(n) // 8)),
                         markeredgecolor="white", label=BELL_TEX[name], **style)
        top.set_ylim(0, 1)
        dq = q - q_star
        top.set_title(f"{label}\n$q = q_{{\\mathrm{{rel}}}} {dq:+.1e}$",
                      color=INK, fontsize=10.5, pad=6)
        bot.set_xlabel(r"purification round $n$")
        if col == 0:
            top.set_ylabel("Bell population")
            bot.set_ylabel("non-target populations")
        err = pops[:, 1:]
        lo, hi = float(err.min()), float(err.max())
        pad = max(0.08 * (hi - lo), 1e-4)
        bot.set_ylim(lo - pad, hi + pad)
        for ax in (top, bot):
            _despine(ax)
            ax.set_xlim(0, n[-1])
        if viol is not None:
            for ax in (top, bot):
                ax.axvline(viol["round"], color=INK2, linestyle=(0, (4, 3)),
                           linewidth=1.1)
            bot.annotate(rf"$\Delta p$ of the violating component:"
                         f"\n" rf"${viol['delta_p']:+.2e}$",
                         xy=(0.04, 0.92), xycoords="axes fraction",
                         fontsize=9, color=INK, va="top")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.03), labelcolor=INK)
    fig.suptitle(rf"Type 3 at $\epsilon = {eps:g}$: raw Bell populations for "
                 rf"the same three $q$", fontsize=13, color=INK, y=0.985)
    save(fig, "type3_qrel_eps015_populations")


def plot_transverse(cases, at_thr, in_q):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.5),
                             gridspec_kw=dict(wspace=0.34))
    for ax in axes:
        _despine(ax)
    palette = [BELL_COLOR["Phi+"], BELL_COLOR["Psi+"], BELL_COLOR["Phi-"]]
    for k, (label, q, rows, _v) in enumerate(cases):
        n = [r["n"] for r in rows[1:]]
        c = [max(r["c_off"], 1e-300) for r in rows[1:]]
        axes[0].semilogy(n, c, color=palette[k % 3], marker="o",
                         markevery=(k, 4), markeredgecolor="white",
                         linewidth=1.7, label=label)
    axes[0].axhline(OFFDIAG_TRUST, color=INK2, linestyle=(0, (4, 3)),
                    linewidth=1.1)
    axes[0].annotate("trust limit", xy=(0.04, 0.12), xycoords="axes fraction",
                     fontsize=9, color=INK2)
    axes[0].set_xlabel(r"purification round $n$")
    axes[0].set_ylabel(r"$C_{\rm off}(n)$")
    axes[0].legend(loc="upper left", labelcolor=INK, fontsize=8.5)
    axes[0].set_title(r"Bell off-diagonal weight along the actual trajectory",
                      color=INK, fontsize=10.5)

    axes[1].axhline(1.0, color=INK2, linewidth=0.9)
    axes[1].plot([r["epsilon"] for r in at_thr],
                 [r["growth_rate"] for r in at_thr], color=BELL_COLOR["Phi+"],
                 marker="o", markeredgecolor="white", linewidth=1.9)
    axes[1].set_xlabel(r"input mixing $\epsilon$")
    axes[1].set_ylabel(r"$\lambda_\perp$  (per round)")
    axes[1].set_title(r"seeded perturbation at $q = q_{\rm rel}(\epsilon) - 10^{-6}$",
                      color=INK, fontsize=10.5)

    axes[2].axhline(1.0, color=INK2, linewidth=0.9)
    eps_vals = sorted({r["epsilon"] for r in in_q})
    for k, ev in enumerate(eps_vals):
        sub = sorted((r for r in in_q if r["epsilon"] == ev),
                     key=lambda r: r["q"])
        style = dict(linewidth=4.0, alpha=0.5) if k == 0 else \
            dict(linewidth=1.4, linestyle=(0, (5, 3)))
        axes[2].plot([r["q"] for r in sub], [r["growth_rate"] for r in sub],
                     color=palette[k % 3], marker=BELL_MARKER[BELL_ORDER[1 + k % 3]],
                     markeredgecolor="white", label=rf"$\epsilon = {ev:g}$",
                     **style)
    axes[2].set_xlabel(r"per-CNOT noise $q$")
    axes[2].set_ylabel(r"$\lambda_\perp$  (per round)")
    axes[2].legend(loc="upper left", labelcolor=INK)
    axes[2].set_title(r"at fixed $\epsilon$, as a function of $q$",
                      color=INK, fontsize=10.5)
    for ax, tag in zip(axes, ("(a)", "(b)", "(c)")):
        ax.text(-0.18, 1.06, tag, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=INK)
    fig.suptitle("Type 3: Bell off-diagonal and transverse diagnostics "
                 "(separate from the q_rel definition)",
                 fontsize=13, color=INK, y=1.03)
    save(fig, "type3_qrel_transverse")


# ===========================================================================
# 10. driver
# ===========================================================================

EPS_TABLE = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.66)


def _fmt(v, spec=".12f"):
    return "n/a" if v is None else format(v, spec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bisect-tol", type=float, default=1e-11)
    ap.add_argument("--dense-points", type=int, default=65)
    ap.add_argument("--eps-min-dense", type=float, default=0.01)
    ap.add_argument("--eps-max-dense", type=float, default=0.65)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("=" * 78)
    print("TYPE-3 RELATIVE THRESHOLD q_rel -- INDEPENDENT CIRCUIT-LEVEL CHECK")
    print("=" * 78)
    print(f"  source : {CIRCUIT_SOURCE}")
    print("  wires  : (a, A1, A2, B1, B2) = (0, 1, 2, 3, 4)")
    print("  retained register A = wires (1, 2);  discarded B = wires (3, 4)")
    print("  readout: tau_A = Tr_B[<0|sigma|0>_a - <1|sigma|1>_a], "
          "rho_next = tau_A / Tr(tau_A)")
    print("           parity-weighted, NOT a postselection on outcome 0")
    print(f"  CNOT count = {N_CNOT}   (expected 16: two Fredkins x 8)")
    print("  failure condition: R_i^(n+1) - R_i^(n) > VIOL_TOL  "
          "(NOT p_i^(n+1) >= p_i^(n))\n")
    print(format_sequence())

    print("\n" + "=" * 78)
    print("SANITY CHECKS")
    print("=" * 78)
    rng = np.random.default_rng(20261007)
    checks = {
        "D_circuit_unitary": check_circuit_unitary(),
        "B_noise_channel": check_noise_channel(rng, 10),
        "C_noiseless_equals_rho2_over_purity": check_noiseless_purification(rng, 30),
        "A_state_validity_eps0.15_q0.02": check_state_validity(0.15, 0.02, 40),
        "error_floor": measure_error_floor(),
        "tolerances": {"STEP_TOL": STEP_TOL, "VIOL_TOL": VIOL_TOL,
                       "N_MAX": N_MAX, "OFFDIAG_TRUST": OFFDIAG_TRUST,
                       "bisect_tol": args.bisect_tol},
    }
    for group, d in checks.items():
        print(f"  {group}")
        for k, v in d.items():
            print(f"      {k:<44} {v:.4e}" if isinstance(v, float)
                  else f"      {k:<44} {v}")

    print("\n" + "=" * 78)
    print("THRESHOLD SEARCH  (predicate = relative ratios from the circuit)")
    print("=" * 78)
    table = []
    for eps in EPS_TABLE:
        t0 = time.time()
        full = q_rel_full(eps, tol=args.bisect_tol)
        first = q_rel_first_round(eps, tol=args.bisect_tol)
        comps = per_component_thresholds(eps, tol=args.bisect_tol)
        fr, fu, fd = comps["first_round"], comps["all_rounds"], comps["all_round_detail"]
        smallest = min((v for v in fr.values() if v is not None), default=None)
        limiting_first = next((c for c, v in fr.items() if v == smallest), None)
        row = {
            "epsilon": eps,
            "q_rel_full": full["q_rel"], "q_rel_first": first["q_rel"],
            "q_rel_full_minus_first": (None if full["q_rel"] is None
                                       or first["q_rel"] is None
                                       else full["q_rel"] - first["q_rel"]),
            "q_Phi_minus_rel_first": fr["Phi-"],
            "q_Psi_plus_rel_first": fr["Psi+"],
            "q_Psi_minus_rel_first": fr["Psi-"],
            "q_Psi_plus_minus_Psi_minus_first": (
                None if fr["Psi+"] is None or fr["Psi-"] is None
                else fr["Psi+"] - fr["Psi-"]),
            "q_Phi_minus_rel_all": fu["Phi-"], "q_Psi_plus_rel_all": fu["Psi+"],
            "q_Psi_minus_rel_all": fu["Psi-"],
            "binding_round_Phi_minus": fd["Phi-"]["round"],
            "binding_round_Psi_plus": fd["Psi+"]["round"],
            "binding_round_Psi_minus": fd["Psi-"]["round"],
            "limiting_component": limiting_first,
            "limiting_at_q_upper": full["limiting_component"],
            "first_violation_round_near_threshold": full["first_violation_round"],
            "threshold_bracket_width": full.get("bracket_width"),
            "q_upper_violating": full.get("q_upper"),
            "Delta_R_at_q_upper": full.get("delta_R_at_violation"),
            "Delta_p_of_violating_at_q_upper": full.get("delta_p_of_violating"),
            "parity_denominator_at_q_upper": full.get(
                "parity_denominator_at_violation"),
            "max_off_Bell_weight_before_convergence": full.get("below_c_off_max"),
            "below_threshold_converged": full.get("below_converged"),
            "below_threshold_trusted": full.get("below_trusted"),
            "below_threshold_n_trusted": full.get("below_n_trusted"),
            "below_threshold_rounds_run": full.get("below_rounds_run"),
            "below_worst_delta_R": full.get("below_worst_delta_R"),
            "below_worst_component": full.get("below_worst_component"),
            "below_worst_round": full.get("below_worst_round"),
            "below_n_abs_rise_with_rel_fall": len(
                full.get("below_abs_rise_with_rel_fall") or []),
            "bracket_monotone": full["bracket_info"].get("monotone"),
            "bracket_sign_changes": full["bracket_info"].get("n_sign_changes"),
            # NOTE: evaluated at q_upper_violating, NOT at q_rel_full.
            "p_n_at_q_upper": (None if full.get("p_n_at_q_upper") is None
                               else full["p_n_at_q_upper"].tolist()),
            "p_np1_at_q_upper": (None if full.get("p_np1_at_q_upper") is None
                                 else full["p_np1_at_q_upper"].tolist()),
            "R_n_at_q_upper": (None if full.get("R_n_at_q_upper") is None
                               else full["R_n_at_q_upper"].tolist()),
            "R_np1_at_q_upper": (None if full.get("R_np1_at_q_upper") is None
                                 else full["R_np1_at_q_upper"].tolist()),
            "seconds": round(time.time() - t0, 2),
        }
        table.append(row)
        print(f"  eps={eps:<5} q_rel_full={_fmt(row['q_rel_full'])}  "
              f"q_rel_first={_fmt(row['q_rel_first'])}  "
              f"diff={_fmt(row['q_rel_full_minus_first'], '+.2e')}  "
              f"limiting={str(row['limiting_component']):<5} "
              f"round={row['first_violation_round_near_threshold']}  "
              f"dR={_fmt(row['Delta_R_at_q_upper'], '.3e')}  "
              f"conv={row['below_threshold_converged']} "
              f"trusted={row['below_threshold_trusted']}  ({row['seconds']}s)")

    print("\n  per-component FIRST-ROUND relative thresholds (smallest = limiting)")
    for r in table:
        print(f"    eps={r['epsilon']:<5} Phi-={_fmt(r['q_Phi_minus_rel_first'])}  "
              f"Psi+={_fmt(r['q_Psi_plus_rel_first'])}  "
              f"Psi-={_fmt(r['q_Psi_minus_rel_first'])}  "
              f"Psi+ - Psi-={_fmt(r['q_Psi_plus_minus_Psi_minus_first'], '+.2e')}")

    print("\n  per-component STANDALONE ALL-ROUND relative thresholds (binding round)")
    for r in table:
        print(f"    eps={r['epsilon']:<5} "
              f"Phi-={_fmt(r['q_Phi_minus_rel_all'])} (n={r['binding_round_Phi_minus']})  "
              f"Psi+={_fmt(r['q_Psi_plus_rel_all'])} (n={r['binding_round_Psi_plus']})  "
              f"Psi-={_fmt(r['q_Psi_minus_rel_all'])} (n={r['binding_round_Psi_minus']})")

    print("\n  violation-tolerance sensitivity (VIOL_TOL 1e-10 / 1e-12 / 1e-14)")
    tol_study = tolerance_study((0.05, 0.15, 0.40, 0.65), bisect_tol=args.bisect_tol)
    for eps, d in tol_study.items():
        vals = "  ".join(f"{k}:{_fmt(v)}" for k, v in d["values"].items())
        print(f"    eps={eps:<5} {vals}   spread={_fmt(d['spread'], '.2e')}")

    print("\n  near-threshold check at delta = 1e-7, 1e-6, 1e-4, 1e-3")
    near = []
    for r in table:
        if r["q_rel_full"] is not None:
            near += near_threshold_scan(r["epsilon"], r["q_rel_full"])
    bad = [r for r in near
           if (r["side"] == "below" and r["violated"])
           or (r["side"] == "above" and not r["violated"])]
    print(f"    {len(near)} tests; unexpected outcomes = {len(bad)}")
    for r in bad:
        print(f"      UNEXPECTED  eps={r['epsilon']} delta={r['delta']:g} "
              f"{r['side']} q={r['q']:.12f} violated={r['violated']}")
    print(f"    violation rounds on the above side: "
          f"{sorted({r['violation_round'] for r in near if r['side'] == 'above' and r['violated']})}")
    print(f"    violating components on the above side:  "
          f"{sorted({r['violating_component'] for r in near if r['side'] == 'above' and r['violated']})}")

    print("\n  dense epsilon grid ...")
    dense = []
    grid = np.linspace(args.eps_min_dense, args.eps_max_dense, args.dense_points)
    hint_f = hint_g = None
    for eps in grid:
        f = q_rel_full(float(eps), tol=args.bisect_tol, hint=hint_f)
        g = q_rel_first_round(float(eps), tol=args.bisect_tol, hint=hint_g)
        hint_f, hint_g = f["q_rel"], g["q_rel"]
        dense.append({"epsilon": float(eps), "q_rel_full": f["q_rel"],
                      "q_rel_first": g["q_rel"],
                      "limiting_component": f["limiting_component"],
                      "first_violation_round_near_threshold": f["first_violation_round"],
                      "threshold_bracket_width": f.get("bracket_width")})
    diffs = [abs(r["q_rel_full"] - r["q_rel_first"]) for r in dense
             if r["q_rel_full"] is not None and r["q_rel_first"] is not None]
    limiting_set = sorted({r["limiting_component"] for r in dense})
    rounds_set = sorted({r["first_violation_round_near_threshold"] for r in dense})
    print(f"    {len(dense)} points;  max |full - first| = {max(diffs):.3e}")
    print(f"    limiting components over the grid: {limiting_set}")
    print(f"    first-violation rounds over the grid: {rounds_set}")

    # ---- eps = 0.15 detail ---------------------------------------------
    eps_d = 0.15
    q_star = next(r["q_rel_full"] for r in table if r["epsilon"] == eps_d)
    cases, traj_rows = [], []
    for label, q in (("well below threshold", q_star - 5e-3),
                     ("just below threshold", q_star - 1e-7),
                     ("just above threshold", q_star + 1e-3)):
        rows, viol = record_trajectory(eps_d, q, 20)
        cases.append((label, q, rows, viol))
        res = run_trajectory(eps_d, q)
        v = (f"violation: {res['component']} at n={res['round']}, "
             f"Delta R={res['delta_R_violating']:+.4e}, "
             f"Delta p of that component={res['delta_p_of_violating']:+.4e}") \
            if res["violated"] else \
            (f"no violation; worst Delta R = {res['worst_delta_R']:+.3e} "
             f"({res['worst_component']} at n={res['worst_round']}), "
             f"converged={res['converged']}, trusted={res['trusted']}")
        n_mix = len(res["abs_rise_with_rel_fall"])
        print(f"    eps=0.15, q={q:.12f}  [{label}]  {v}")
        print(f"        rounds where a raw p_i ROSE while its ratio FELL: {n_mix}")
        for ex in res["abs_rise_with_rel_fall"][:3]:
            print(f"          n={ex['round']} {ex['component']}: "
                  f"dp={ex['delta_p']:+.3e} (p {ex['p_n']:.8f} -> {ex['p_np1']:.8f}), "
                  f"dR={ex['delta_R']:+.3e}, "
                  f"p_Phi+ {ex['p_phi_plus_n']:.8f} -> {ex['p_phi_plus_np1']:.8f}")
        for r in rows:
            traj_rows.append({
                "case": label, "q": q, "round": r["n"],
                "p_phi_plus": r["p"][0], "p_phi_minus": r["p"][1],
                "p_psi_plus": r["p"][2], "p_psi_minus": r["p"][3],
                "R_phi_minus": r["R"][1], "R_psi_plus": r["R"][2],
                "R_psi_minus": r["R"][3],
                "d_p_phi_minus": r["dp"][1], "d_p_psi_plus": r["dp"][2],
                "d_p_psi_minus": r["dp"][3],
                "d_R_phi_minus": r["dR"][1], "d_R_psi_plus": r["dR"][2],
                "d_R_psi_minus": r["dR"][3],
                "C_offdiag": r["c_off"],
                "parity_denominator": r["parity_denominator"]})

    # the window where the ABSOLUTE criterion has already failed while the
    # RELATIVE one still holds -- both edges measured by this script
    print("\n  window where a raw p_i already rises but every ratio still falls")
    onsets = []
    for r in table:
        o = absolute_rise_onset(r["epsilon"], tol=args.bisect_tol)
        width = (None if o["q_onset"] is None or r["q_rel_full"] is None
                 else r["q_rel_full"] - o["q_onset"])
        onsets.append({"epsilon": r["epsilon"],
                       "q_absolute_rise_onset": o["q_onset"],
                       "first_rising_component": o.get("first_rising_component"),
                       "q_rel_full": r["q_rel_full"], "window_width": width,
                       "window_relative_size": (None if width is None
                                                else width / r["q_rel_full"])})
        print(f"    eps={r['epsilon']:<5} raw rise starts at q="
              f"{_fmt(o['q_onset'])}  ({o.get('first_rising_component')})"
              f"   q_rel={_fmt(r['q_rel_full'])}   window width="
              f"{_fmt(width, '.3e')}")

    eps_window = next(w for w in onsets if w["epsilon"] == eps_d)
    mixed = []
    lo_w = eps_window["q_absolute_rise_onset"]
    for q in np.linspace(lo_w, q_star, 25)[1:-1]:
        res = run_trajectory(eps_d, float(q))
        if not res["violated"] and res["abs_rise_with_rel_fall"]:
            ex = res["abs_rise_with_rel_fall"][0]
            mixed.append({"epsilon": eps_d, "q": float(q), **ex})
    print(f"\n    inside that window at eps=0.15: {len(mixed)} of 23 sampled q "
          f"show a raw rise with every ratio still falling")
    if mixed:
        m = mixed[len(mixed) // 2]
        print(f"      representative q = {m['q']:.9f}")
        print(f"        n={m['round']}, {m['component']}: "
              f"p {m['p_n']:.9f} -> {m['p_np1']:.9f}  "
              f"(dp={m['delta_p']:+.3e}, RISES)")
        print(f"        R {m['p_n']/m['p_phi_plus_n']:.9f} -> "
              f"{m['p_np1']/m['p_phi_plus_np1']:.9f}  "
              f"(dR={m['delta_R']:+.3e}, FALLS)")
        print(f"        because p_Phi+ grew {m['p_phi_plus_n']:.9f} -> "
              f"{m['p_phi_plus_np1']:.9f}")

    # ---- transverse diagnostics ----------------------------------------
    print("\n" + "=" * 78)
    print("BELL OFF-DIAGONAL AND TRANSVERSE DIAGNOSTICS (separate from q_rel)")
    print("=" * 78)
    at_thr = []
    for r in table:
        if r["q_rel_full"] is None:
            continue
        q = max(r["q_rel_full"] - 1e-6, 0.0)
        g = transverse_growth(r["epsilon"], q)
        at_thr.append({"epsilon": r["epsilon"], "q_rel": r["q_rel_full"], "q": q,
                       "growth_rate": g["growth_rate"], "a_initial": g["a_initial"],
                       "a_final": g["a_final"], "fp_locked": g["fp_locked"],
                       "amplitude_at_floor": g["amplitude_at_floor"],
                       "seeded_state_min_eig": g["seeded_state_min_eig"]})
        print(f"    eps={r['epsilon']:<5} q={q:.12f}  "
              f"lambda_perp={g['growth_rate']:.6f}  "
              f"a: {g['a_initial']:.3e} -> {g['a_final']:.3e}")
    in_q = []
    print("\n    lambda_perp as a function of q at fixed eps")
    for ev in (0.15, 0.50):
        for q in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
            g = transverse_growth(ev, q)
            in_q.append({"epsilon": ev, "q": q, "growth_rate": g["growth_rate"],
                         "fp_locked": g["fp_locked"]})
            print(f"      eps={ev:<5} q={q:<6g} lambda_perp={g['growth_rate']:.6f}"
                  + ("   [fp-locked]" if g["fp_locked"] else ""))
    print("\n    per Bell pair at eps=0.15, q=0.02")
    pair_rows = []
    for pair in BELL_PAIRS:
        g = transverse_growth(0.15, 0.02, pair=pair)
        pair_rows.append({"epsilon": 0.15, "q": 0.02, "pair": g["pair"],
                          "growth_rate": g["growth_rate"]})
        print(f"      {g['pair']:<12} lambda_perp={g['growth_rate']:.6f}")

    # ---- outputs --------------------------------------------------------
    main_cols = ["epsilon", "q_rel_full", "q_rel_first", "q_rel_full_minus_first",
                 "q_Phi_minus_rel_first", "q_Psi_plus_rel_first",
                 "q_Psi_minus_rel_first", "limiting_component",
                 "first_violation_round_near_threshold",
                 "threshold_bracket_width", "q_upper_violating",
                 "Delta_R_at_q_upper", "Delta_p_of_violating_at_q_upper",
                 "parity_denominator_at_q_upper", "seconds"]
    comp_cols = ["epsilon", "q_Phi_minus_rel_first", "q_Psi_plus_rel_first",
                 "q_Psi_minus_rel_first", "q_Psi_plus_minus_Psi_minus_first",
                 "q_Phi_minus_rel_all", "q_Psi_plus_rel_all",
                 "q_Psi_minus_rel_all", "binding_round_Phi_minus",
                 "binding_round_Psi_plus", "binding_round_Psi_minus"]
    diag_cols = ["epsilon", "max_off_Bell_weight_before_convergence",
                 "below_threshold_converged", "below_threshold_trusted",
                 "below_threshold_n_trusted", "below_threshold_rounds_run",
                 "below_worst_delta_R", "below_worst_component",
                 "below_worst_round", "below_n_abs_rise_with_rel_fall",
                 "limiting_at_q_upper", "bracket_monotone",
                 "bracket_sign_changes"]
    for name, cols in (("type3_qrel_circuit.csv", main_cols),
                       ("type3_qrel_component_thresholds.csv", comp_cols),
                       ("type3_qrel_diagnostics.csv", diag_cols)):
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(table)
    for name, rowset in (("type3_qrel_dense.csv", dense),
                         ("type3_qrel_near_threshold.csv", near),
                         ("type3_qrel_eps015_trajectories.csv", traj_rows),
                         ("type3_qrel_transverse.csv", at_thr),
                         ("type3_qrel_transverse_vs_q.csv", in_q),
                         ("type3_qrel_transverse_pairs.csv", pair_rows),
                         ("type3_qrel_absolute_rise_window.csv", onsets)):
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rowset[0]))
            w.writeheader()
            w.writerows(rowset)
    if mixed:
        with open(OUT / "type3_qrel_abs_vs_rel_examples.csv", "w",
                  newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(mixed[0]))
            w.writeheader()
            w.writerows(mixed)

    meta = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "quantity": "q_rel: relative Bell-error-suppression threshold",
        "failure_condition": "R_i^(n+1) - R_i^(n) > VIOL_TOL, R_i = p_i / p_Phi+",
        "blindness": ("independent circuit-level implementation verification; "
                      "no q_rel formula, root, map, table or limiting component "
                      "existed in either repository or was used"),
        "circuit_source": CIRCUIT_SOURCE,
        "n_cnot": N_CNOT,
        "wire_order": "(a, A1, A2, B1, B2) = (0, 1, 2, 3, 4)",
        "gate_sequence": [[k, list(a) if isinstance(a, tuple) else a]
                          for k, a in TYPE3_OPS],
        "noise_model": "two-qubit replacement depolarizing after every CNOT",
        "readout": ("tau_A = Tr_B[<0|sigma|0>_a - <1|sigma|1>_a]; "
                    "rho_next = tau_A/Tr(tau_A); parity-weighted, "
                    "NOT a postselection"),
        "no_twirling": True, "used_analytic_input": False,
        "seed": 20261007,
        "checks": {k: {kk: (float(vv) if isinstance(vv, float) else vv)
                       for kk, vv in v.items()} for k, v in checks.items()},
        "tolerance_study": tol_study,
        "table": table,
        "dense_max_abs_full_minus_first": float(max(diffs)),
        "dense_limiting_components": limiting_set,
        "dense_first_violation_rounds": [int(r) if r is not None else None
                                         for r in rounds_set],
        "near_threshold_unexpected": bad,
        "absolute_rise_window": onsets,
        "abs_rises_while_rel_falls": mixed[:20],
        "n_q_values_with_abs_rise_and_rel_fall": len(mixed),
        "transverse_at_threshold": at_thr, "transverse_vs_q": in_q,
        "transverse_pairs": pair_rows,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    (OUT / "type3_qrel_verification.json").write_text(
        json.dumps(meta, indent=2, default=str))

    if not args.no_figures:
        use_style()
        plot_curve(dense, table)
        plot_first_vs_full(dense)
        plot_components(table)
        plot_eps015_ratios(cases, eps_d, q_star)
        plot_eps015_populations(cases, eps_d, q_star)
        plot_transverse(cases, at_thr, in_q)
    print("\n" + "=" * 78)
    print("FILES")
    print("=" * 78)
    for p in sorted(OUT.glob("*")):
        print(f"  {p.relative_to(ROOT)}")
    print(f"\n  elapsed {meta['elapsed_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
