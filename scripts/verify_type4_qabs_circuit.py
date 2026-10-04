"""Blind circuit-level verification of the Type-4 absolute Bell-error-suppression
threshold q_abs^(4)(epsilon).

SELF-CONTAINED ON PURPOSE.  This script imports nothing from ``pqec_distill``,
nothing from the parent research repository, and no analytic recurrence.  The
only prior information it uses is

  1. the canonical Type-4 physical gate sequence, transcribed verbatim below
     from PQEC-Operational-Threshold/pqec_resynth_noise.py (the module-level
     ``GATES`` list; commit 621d593c, sha256 8d66cea64f494443...),
  2. the Bell-isotropic input state,
  3. the per-CNOT two-qubit replacement depolarizing channel,
  4. the operational definition of q_abs given in the task.

No Bell-diagonal reduced map, no (x,y,z) or (u,v) recurrence, no closed-form
population map and no previously computed Type-4 threshold is used anywhere.
Every density matrix is propagated in full, and every round feeds its exact
output state into the next round -- no twirling, no re-isotropisation, no
Bell-diagonal projection, no symmetrisation, no off-diagonal truncation.

DEFINITION BEING TESTED

    q_abs(eps) = sup { q : p_i^(n+1) < p_i^(n)
                           for every i in {Phi-, Psi+, Psi-} and every n >= 0 }

with p_i^(n) = <B_i| rho_n |B_i> read off the full two-qubit density matrix.
Which component is limiting, whether Psi+ and Psi- agree, and whether the first
round is the binding one are OUTPUTS here, not assumptions.

Run:
    python scripts/verify_type4_qabs_circuit.py
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
OUT = ROOT / "results" / "type4_qabs"

N_WIRES = 5
DIM = 2 ** N_WIRES
ANCILLA = 0
RETAIN = (1, 2)          # register A, kept
DISCARD = (3, 4)         # register B, traced out

# ===========================================================================
# 1. gate primitives (dense, 5 qubits, |q0 q1 q2 q3 q4> with q0 most significant)
# ===========================================================================

_I2 = np.eye(2, dtype=complex)
_P0 = np.array([[1, 0], [0, 0]], dtype=complex)
_P1 = np.array([[0, 0], [0, 1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)


def _kron_list(mats):
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def u3_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    """U3(theta, phi, lambda), the Qiskit ``u`` / PennyLane ``U3`` convention:

        [[      cos(t/2)        ,  -e^{i lam}      sin(t/2) ],
         [ e^{i phi} sin(t/2)   ,   e^{i(phi+lam)} cos(t/2) ]]
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
                    dtype=complex)


def place_single(mat: np.ndarray, wire: int) -> np.ndarray:
    factors = [_I2] * N_WIRES
    factors[wire] = mat
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


def hadamard_on(wire: int) -> np.ndarray:
    return place_single(np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
                        wire)


# ===========================================================================
# 2. the canonical Type-4 gate sequence
# ===========================================================================
#
# Transcribed verbatim, in order and with the original angles, from the
# module-level ``GATES`` list of
#     PQEC-Operational-Threshold/pqec_resynth_noise.py
# (Qiskit transpile of H_a . CSWAP(a;A1,B1) . CSWAP(a;A2,B2) . H_a into the
# basis {u, cx}, optimization_level 3, seed_transpiler 7; global phase pi/4).
#
# The decomposition itself -- not merely the unitary it implements -- is what
# the noise sees, so the single-qubit gates stay exactly where they are and the
# CNOTs keep their original order and orientation.
#
# Wire convention of that source file:  0 = ancilla a, 1 = A1, 2 = A2,
# 3 = B1, 4 = B2;  the two controlled swaps are (a; A1, B1) and (a; A2, B2).

_P4 = np.pi / 4
_P2 = np.pi / 2

TYPE4_OPS = [
    ("U3", (0, 1.5707963267948966, 0.0, 3.141592653589793)),
    ("U3", (1, 1.5707963267948966, -1.5707963267948966, 1.5707963267948966)),
    ("U3", (2, 1.5707963267948966, -1.5707963267948966, 1.5707963267948966)),
    ("U3", (3, 1.5707963267948968, -3.141592653589793, -2.356194490192345)),
    ("CNOT", (1, 3)),
    ("U3", (1, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966)),
    ("U3", (3, 0.7853981633974475, -2.356194490192345, -1.5707963267948966)),
    ("CNOT", (0, 3)),
    ("U3", (3, 0.0, 0.0, 0.7853981633974483)),
    ("CNOT", (1, 3)),
    ("U3", (1, 0.0, 0.0, 0.7853981633974483)),
    ("U3", (3, 0.0, 0.0, -0.7853981633974483)),
    ("CNOT", (0, 3)),
    ("CNOT", (0, 1)),
    ("U3", (0, 0.0, 0.0, 0.7853981633974483)),
    ("U3", (1, 0.0, 0.0, -0.7853981633974483)),
    ("CNOT", (0, 1)),
    ("U3", (3, 1.5707963267948966, 0.0, -2.3561944901923453)),
    ("CNOT", (3, 1)),
    ("U3", (4, 1.5707963267948968, -3.141592653589793, -2.356194490192345)),
    ("CNOT", (2, 4)),
    ("U3", (2, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966)),
    ("U3", (4, 0.7853981633974475, -2.356194490192345, -1.5707963267948966)),
    ("CNOT", (0, 4)),
    ("U3", (4, 0.0, 0.0, 0.7853981633974483)),
    ("CNOT", (2, 4)),
    ("U3", (2, 0.0, 0.0, 0.7853981633974483)),
    ("U3", (4, 0.0, 0.0, -0.7853981633974483)),
    ("CNOT", (0, 4)),
    ("CNOT", (0, 2)),
    ("U3", (0, 0.0, 0.0, 0.7853981633974483)),
    ("U3", (2, 0.0, 0.0, -0.7853981633974483)),
    ("CNOT", (0, 2)),
    ("U3", (0, 1.5707963267948966, 0.0, 3.141592653589793)),
    ("U3", (4, 1.5707963267948966, 0.0, -2.3561944901923453)),
    ("CNOT", (4, 2)),
]

N_CNOT = sum(1 for kind, _ in TYPE4_OPS if kind == "CNOT")


def format_sequence() -> str:
    lines, k = [], 0
    for kind, arg in TYPE4_OPS:
        if kind == "CNOT":
            k += 1
            lines.append(f"  {len(lines)+1:3d}.  CNOT({arg[0]} -> {arg[1]})"
                         f"           [CNOT #{k:2d}]  + depolarizing D_q on "
                         f"({arg[0]},{arg[1]})")
        else:
            w, t, p, l = arg
            lines.append(f"  {len(lines)+1:3d}.  U3(wire {w}; theta={t/np.pi:+.4f}pi"
                         f", phi={p/np.pi:+.4f}pi, lam={l/np.pi:+.4f}pi)")
    return "\n".join(lines)


def op_matrix(kind, arg) -> np.ndarray:
    if kind == "CNOT":
        return cnot(*arg)
    w, t, p, l = arg
    return place_single(u3_matrix(t, p, l), w)


def build_program():
    """Merge runs of unitaries into single 32x32 matrices; keep the channel
    locations exactly where the CNOTs are.  Returns [("U", M) | ("D", (i,j))]."""
    program, acc = [], np.eye(DIM, dtype=complex)
    for kind, arg in TYPE4_OPS:
        acc = op_matrix(kind, arg) @ acc
        if kind == "CNOT":
            program.append(("U", acc))
            acc = np.eye(DIM, dtype=complex)
            program.append(("D", tuple(arg)))
    program.append(("U", acc))
    return program


PROGRAM = build_program()


def circuit_unitary(ops=None) -> np.ndarray:
    """Noiseless 32x32 unitary of the transcribed sequence."""
    u = np.eye(DIM, dtype=complex)
    for kind, arg in (ops or TYPE4_OPS):
        u = op_matrix(kind, arg) @ u
    return u


# ===========================================================================
# 3. the two-qubit replacement depolarizing channel, implemented directly
# ===========================================================================

_ROW = "abcde"
_COL = "fghij"


def replacement_depol(sigma: np.ndarray, pair, q: float) -> np.ndarray:
    """D_q^(ij)(sigma) = (1-q) sigma + q [ I_ij/4 (x) Tr_ij(sigma) ].

    Implemented from the definition on the FULL five-qubit density matrix:
    partial trace over the two wires, then re-insert I/4 in their slots.
    """
    if q == 0.0:
        return sigma
    i, j = pair
    rest = [w for w in range(N_WIRES) if w not in (i, j)]
    t = sigma.reshape([2] * (2 * N_WIRES))
    row, col = list(_ROW), list(_COL)
    for w in (i, j):                      # equal indices -> traced
        col[w] = row[w]
    sub = "".join(row) + "".join(col)
    out = "".join(row[w] for w in rest) + "".join(col[w] for w in rest)
    reduced = np.einsum(f"{sub}->{out}", t)
    target = "".join(_ROW) + "".join(_COL)
    rebuilt = np.einsum(f"{out},{_ROW[i]}{_COL[i]},{_ROW[j]}{_COL[j]}->{target}",
                        reduced, _I2, _I2)
    return (1 - q) * sigma + q * 0.25 * rebuilt.reshape(DIM, DIM)


# ===========================================================================
# 4. one Type-4 round on a full two-qubit density matrix
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
    """p_i = <B_i| rho |B_i>, computed independently for all four, with no
    equality assumed between any of them."""
    return np.array([float(np.real(BELL[k].conj() @ rho @ BELL[k]))
                     for k in BELL_ORDER])


def isotropic_rho(eps: float) -> np.ndarray:
    """rho_0(eps) = (1-eps)|Phi+><Phi+| + eps I/4, built from the definition."""
    e = float(eps)
    return (1 - e) * np.outer(PHI_P, PHI_P.conj()) + e * np.eye(4, dtype=complex) / 4


def one_round(rho: np.ndarray, q: float, program=None):
    """rho_(n+1) = tau_A / Tr(tau_A) with
    tau_A = Tr_B[ <0|sigma_out|0>_a - <1|sigma_out|1>_a ].

    Parity-weighted virtual purification -- NOT a physical postselection on
    ancilla outcome 0.
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
        # <m|_a sigma |m>_a  -> axes (A1, A2, B1, B2 | A1', A2', B1', B2')
        block = t[m, :, :, :, :, m, :, :, :, :]
        # trace out register B (wires 3, 4): B1 = B1' and B2 = B2'
        traced = np.einsum("ijklmnkl->ijmn", block)
        tau += sign * traced.reshape(4, 4)
    weight = float(np.real(np.trace(tau)))
    herm_residue = float(np.linalg.norm(tau - tau.conj().T))
    tau = 0.5 * (tau + tau.conj().T)          # exact-arithmetic identity
    if weight == 0.0 or not np.isfinite(weight):
        raise FloatingPointError(f"parity weight vanished: {weight!r}")
    return tau / weight, {"weight": weight, "herm_residue": herm_residue}


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
    """(B) trace preservation, hermiticity, positivity, q=0 identity, and the
    q=1 limit: the chosen pair becomes I/4 and the other wires' reduced state
    is untouched."""
    worst = {"trace": 0.0, "herm": 0.0, "min_eig": 0.0, "identity_at_q0": 0.0,
             "q1_pair_is_I_over_4": 0.0, "q1_rest_preserved": 0.0}
    pairs = [(0, 1), (1, 3), (2, 4), (3, 1), (4, 2), (0, 4)]
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
    """(D) the noiseless 14-CNOT sequence must implement
    H_a CSWAP(a;A2,B2) CSWAP(a;A1,B1) H_a, up to a global phase."""
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

    target = hadamard_on(ANCILLA)
    target = cswap(ANCILLA, 1, 3) @ target
    target = cswap(ANCILLA, 2, 4) @ target
    target = hadamard_on(ANCILLA) @ target

    v = circuit_unitary()
    phase = np.exp(-1j * np.angle(np.trace(target.conj().T @ v)))
    diff = phase * v - target
    hs = abs(np.trace(target.conj().T @ v)) / DIM      # 1 iff equal up to phase
    rng = np.random.default_rng(7)
    worst_state = 0.0
    for _ in range(40):
        psi = rng.normal(size=DIM) + 1j * rng.normal(size=DIM)
        psi /= np.linalg.norm(psi)
        worst_state = max(worst_state,
                          float(np.max(np.abs(np.abs(v @ psi)
                                              - np.abs(target @ psi)))))
    return {"max_abs_unitary_diff": float(np.max(np.abs(diff))),
            "frobenius_unitary_diff": float(np.linalg.norm(diff)),
            "normalised_HS_overlap_minus_1": float(hs - 1.0),
            "unitarity": float(np.max(np.abs(v.conj().T @ v - np.eye(DIM)))),
            "max_abs_amplitude_diff_random_states": worst_state,
            "global_phase_over_pi": float(
                np.angle(np.trace(target.conj().T @ v)) / np.pi)}


def check_noiseless_purification(rng, n_trials=25) -> dict:
    """(C) at q = 0 the parity-weighted output must be rho^2 / Tr(rho^2)."""
    worst = worst_weight = worst_trace_dist = 0.0
    for _ in range(n_trials):
        rho = random_density_matrix(rng, 4)
        out, info = one_round(rho, 0.0)
        expected = rho @ rho / np.trace(rho @ rho)
        worst = max(worst, float(np.max(np.abs(out - expected))))
        d = out - expected
        worst_trace_dist = max(worst_trace_dist, 0.5 * float(
            np.sum(np.abs(np.linalg.eigvalsh(0.5 * (d + d.conj().T))))))
        worst_weight = max(worst_weight,
                           abs(info["weight"] - float(np.real(np.trace(rho @ rho)))))
    return {"max_abs_diff_vs_rho2_over_tr_rho2": worst,
            "max_trace_distance_vs_rho2_over_tr_rho2": worst_trace_dist,
            "max_abs_weight_minus_purity": worst_weight,
            "n_random_trials": n_trials}


def check_state_validity(eps, q, n_rounds=40) -> dict:
    """(A) hermiticity, trace, positivity along an actual trajectory."""
    rho = isotropic_rho(eps)
    worst = {"herm": 0.0, "trace": 0.0, "min_eig": np.inf,
             "pop_sum_minus_1": 0.0, "tau_herm_residue": 0.0}
    for n in range(n_rounds + 1):
        worst["herm"] = max(worst["herm"],
                            float(np.linalg.norm(rho - rho.conj().T)))
        worst["trace"] = max(worst["trace"], abs(float(np.trace(rho).real) - 1.0))
        worst["min_eig"] = min(worst["min_eig"],
                               float(np.linalg.eigvalsh(rho).min()))
        worst["pop_sum_minus_1"] = max(worst["pop_sum_minus_1"],
                                       abs(float(bell_populations(rho).sum()) - 1.0))
        if n < n_rounds:
            rho, info = one_round(rho, q)
            worst["tau_herm_residue"] = max(worst["tau_herm_residue"],
                                            info["herm_residue"])
    return {k: float(v) for k, v in worst.items()}


# ===========================================================================
# 6. the absolute-suppression predicate
# ===========================================================================
#
# Tolerances, and why they are what they are.
#
# At a converged orbit the floating-point map settles onto a numerical fixed
# point whose residual |Delta p| is the round-off floor.  That floor is
# MEASURED by `measure_error_floor()` at run time and printed with the results;
# it is not assumed here.
#
#   STEP_TOL  = 1e-14   iteration stops once max|rho_(n+1) - rho_n| falls below
#                       this: beyond it the orbit sits at the numerical fixed
#                       point and any further sign change is round-off.
#   VIOL_TOL  = 1e-12   a round counts as violating only if some non-target
#                       Delta p_i exceeds +VIOL_TOL, i.e. ~1e4 times the
#                       measured floor.
#
# Because d(Delta p)/dq is O(1) near the crossing, VIOL_TOL = 1e-12 localises
# the threshold to ~1e-12 in q, far inside the 1e-8 target.  The insensitivity
# is checked explicitly by `tolerance_study()`.

STEP_TOL = 1e-14
VIOL_TOL = 1e-12
N_MAX = 5000

# ---------------------------------------------------------------------------
# Round-off contamination guard.
#
# The Bell-isotropic initial state is built exactly Bell-diagonal, so its Bell
# off-diagonal weight
#     C_off(rho) = sum_{i != j} |<B_i| rho |B_j>|^2
# starts at round-off (~1e-34).  If the circuit map amplifies that transverse
# direction, the accumulated round-off can push the simulated orbit off a
# saddle and manufacture a population increase that the exact dynamics does not
# have; the escape round then diverges as the seed goes to zero, so it is an
# artefact of finite precision rather than a feature of the map.  Whether the
# Type-4 map amplifies or contracts that direction is measured separately by
# `transverse_growth()` below -- it is not assumed here either way.
#
# The guard stops trusting a trajectory once C_off exceeds OFFDIAG_TRUST.  At
# that level the induced population error is ~1e-18, six orders below
# VIOL_TOL, so the guard can never hide a genuine violation -- it only refuses
# to certify one that round-off manufactured.
OFFDIAG_TRUST = 1e-18


def run_trajectory(eps: float, q: float, step_tol=STEP_TOL, n_max=N_MAX,
                   viol_tol=VIOL_TOL, components=NON_TARGET, record=False,
                   offdiag_trust=OFFDIAG_TRUST):
    """Iterate the real circuit and look for the first absolute-suppression
    violation.  Every round's exact output state is the next round's input.

    ``violated`` is True only when some tracked non-target population rose by
    more than ``viol_tol`` *while the trajectory was still trustworthy*, i.e.
    before amplified round-off (measured by the Bell off-diagonal weight) could
    have moved the simulated state away from the exact orbit.
    """
    idx = [BELL_ORDER.index(c) for c in components]
    rho = isotropic_rho(eps)
    p = bell_populations(rho)
    c_off0 = bell_offdiagonal_weight(rho)
    rows = [{"n": 0, "p": p.copy(), "dp": np.full(4, np.nan), "step": np.nan,
             "c_off": c_off0}] if record else []
    worst_dp, worst_at, worst_comp = -np.inf, None, None
    c_off_max = c_off0
    n, step, trusted, n_trusted, converged = 0, np.nan, True, None, False
    while n < n_max:
        rho_next, _ = one_round(rho, q)
        p_next = bell_populations(rho_next)
        dp = p_next - p
        step = float(np.max(np.abs(rho_next - rho)))
        c_off = bell_offdiagonal_weight(rho_next)
        c_off_max = max(c_off_max, c_off)
        if record:
            rows.append({"n": n + 1, "p": p_next.copy(), "dp": dp.copy(),
                         "step": step, "c_off": c_off})
        for i in idx:
            if dp[i] > worst_dp:
                worst_dp, worst_at, worst_comp = float(dp[i]), n, BELL_ORDER[i]
        hits = [i for i in idx if dp[i] > viol_tol]
        if hits:
            first = max(hits, key=lambda i: dp[i])
            return {"violated": True, "round": n, "component": BELL_ORDER[first],
                    "p_n": p.copy(), "p_np1": p_next.copy(),
                    "delta_p": dp.copy(), "delta_p_violating": float(dp[first]),
                    "all_violating": [BELL_ORDER[i] for i in hits],
                    "rounds_run": n + 1, "final_step": step, "trusted": True,
                    "n_trusted": None, "converged": False,
                    "c_off_initial": c_off0, "c_off_final": c_off,
                    "c_off_max": c_off_max,
                    "worst_delta_p": worst_dp, "worst_round": worst_at,
                    "worst_component": worst_comp, "rows": rows}
        rho, p = rho_next, p_next
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
            "c_off_max": c_off_max,
            "worst_delta_p": worst_dp, "worst_round": worst_at,
            "worst_component": worst_comp, "p_final": p.copy(), "rows": rows}


def record_trajectory(eps: float, q: float, n_rounds: int,
                      viol_tol=VIOL_TOL, components=NON_TARGET):
    """Round-by-round record for plotting.  Unlike run_trajectory this does not
    stop at the first violation -- it reports where the violation happened and
    keeps going, so the figure can show what the populations do afterwards."""
    idx = [BELL_ORDER.index(c) for c in components]
    rho = isotropic_rho(eps)
    p = bell_populations(rho)
    rows = [{"n": 0, "p": p.copy(), "dp": np.full(4, np.nan),
             "c_off": bell_offdiagonal_weight(rho)}]
    first_violation = None
    for n in range(n_rounds):
        rho, _ = one_round(rho, q)
        p_next = bell_populations(rho)
        dp = p_next - p
        if first_violation is None and any(dp[i] > viol_tol for i in idx):
            k = max((i for i in idx if dp[i] > viol_tol), key=lambda i: dp[i])
            first_violation = {"round": n, "component": BELL_ORDER[k],
                               "delta_p": float(dp[k])}
        rows.append({"n": n + 1, "p": p_next.copy(), "dp": dp.copy(),
                     "c_off": bell_offdiagonal_weight(rho)})
        p = p_next
    return rows, first_violation


def first_round_step(eps: float, q: float):
    """rho_0 -> rho_1 only; returns (p0, p1, dp)."""
    rho = isotropic_rho(eps)
    p0 = bell_populations(rho)
    rho1, _ = one_round(rho, q)
    p1 = bell_populations(rho1)
    return p0, p1, p1 - p0


def first_round_violates(eps: float, q: float, viol_tol=VIOL_TOL,
                         components=NON_TARGET) -> bool:
    _p0, _p1, dp = first_round_step(eps, q)
    return any(dp[BELL_ORDER.index(c)] > viol_tol for c in components)


def measure_error_floor(eps=0.15, q=0.02, n_settle=60, n_watch=40) -> dict:
    """Empirical round-off floor: the residual |Delta p| once the orbit has
    settled onto the floating-point fixed point."""
    rho = isotropic_rho(eps)
    for _ in range(n_settle):
        rho, _ = one_round(rho, q)
    worst_dp = worst_step = 0.0
    p = bell_populations(rho)
    for _ in range(n_watch):
        rho_next, _ = one_round(rho, q)
        p_next = bell_populations(rho_next)
        worst_dp = max(worst_dp, float(np.max(np.abs(p_next - p))))
        worst_step = max(worst_step, float(np.max(np.abs(rho_next - rho))))
        rho, p = rho_next, p_next
    return {"max_abs_delta_p_at_fixed_point": worst_dp,
            "max_abs_step_at_fixed_point": worst_step,
            "eps": eps, "q": q}


# ===========================================================================
# 7. threshold search -- bracket, then bisect, on the circuit predicate only
# ===========================================================================

def bracket(predicate, q_lo=0.0, q_hi=0.5, n_coarse=64):
    """Find (q_low satisfying, q_high violating) by a coarse scan.  Also reports
    whether the predicate looked monotone across the scan."""
    grid = np.linspace(q_lo, q_hi, n_coarse + 1)
    flags = [predicate(float(q)) for q in grid]
    changes = [k for k in range(len(flags) - 1) if flags[k] != flags[k + 1]]
    if not changes:
        return None, None, {"monotone": True, "n_sign_changes": 0,
                            "all_violating": all(flags),
                            "none_violating": not any(flags)}
    k = changes[0]
    info = {"monotone": len(changes) == 1, "n_sign_changes": len(changes),
            "scan_grid_step": float(grid[1] - grid[0])}
    return float(grid[k]), float(grid[k + 1]), info


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
    """Look near ``hint`` first, then fall back to the full scan.  The hint can
    only change where we look: the bracket is re-verified before bisecting, so
    it can never change the answer."""
    if hint is not None:
        lo_h, hi_h = max(0.0, hint - 0.02), min(q_hi, hint + 0.02)
        if not pred(lo_h) and pred(hi_h):
            return lo_h, hi_h, {"monotone": None, "n_sign_changes": None,
                                "from_hint": True}
    return bracket(pred, 0.0, q_hi, n_coarse)


def q_abs_full(eps: float, viol_tol=VIOL_TOL, tol=1e-10, components=NON_TARGET,
               q_hi=0.5, n_coarse=40, hint=None):
    pred = lambda q: run_trajectory(eps, q, viol_tol=viol_tol,
                                    components=components)["violated"]
    lo, hi, info = _hinted_bracket(pred, hint, q_hi, n_coarse)
    if lo is None:
        return {"q_abs": None, "bracket_info": info}
    a, b, it = bisect(pred, lo, hi, tol=tol)
    detail = run_trajectory(eps, b, viol_tol=viol_tol, components=components)
    at_lo = run_trajectory(eps, a, viol_tol=viol_tol, components=components)
    return {"q_abs": a, "q_upper": b, "bracket": (lo, hi), "iterations": it,
            "bracket_width": b - a, "bracket_info": info,
            "limiting_component": detail["component"],
            "first_violation_round": detail["round"],
            "delta_p_at_violation": detail.get("delta_p_violating"),
            "all_violating_at_q_upper": detail.get("all_violating"),
            # NOTE: these populations belong to the q_upper run, not to q_abs.
            "p_n_at_q_upper": detail.get("p_n"),
            "p_np1_at_q_upper": detail.get("p_np1"),
            "below_converged": at_lo["converged"],
            "below_trusted": at_lo["trusted"],
            "below_n_trusted": at_lo["n_trusted"],
            "below_rounds_run": at_lo["rounds_run"],
            "below_c_off_max": at_lo["c_off_max"],
            "below_c_off_final": at_lo["c_off_final"],
            "below_worst_delta_p": at_lo["worst_delta_p"],
            "below_worst_component": at_lo["worst_component"],
            "below_worst_round": at_lo["worst_round"],
            "below_final_step": at_lo["final_step"]}


def q_abs_first_round(eps: float, viol_tol=VIOL_TOL, tol=1e-10,
                      components=NON_TARGET, q_hi=0.5, n_coarse=40, hint=None):
    pred = lambda q: first_round_violates(eps, q, viol_tol, components)
    lo, hi, info = _hinted_bracket(pred, hint, q_hi, n_coarse)
    if lo is None:
        return {"q_abs": None, "bracket_info": info}
    a, b, it = bisect(pred, lo, hi, tol=tol)
    _p0, _p1, dp = first_round_step(eps, b)
    hits = [BELL_ORDER[i] for i in (1, 2, 3) if dp[i] > viol_tol]
    return {"q_abs": a, "q_upper": b, "iterations": it, "bracket_width": b - a,
            "bracket_info": info,
            "limiting_component": (max(hits, key=lambda c: dp[BELL_ORDER.index(c)])
                                   if hits else None),
            "all_violating_components": hits}


def per_component_thresholds(eps: float, viol_tol=VIOL_TOL, tol=1e-10) -> dict:
    """Threshold for each non-target component on its own, both for the first
    round alone (the definition in the task) and over all rounds.  The limiting
    component is the one with the smallest value -- an output, not an input."""
    first, full = {}, {}
    for c in NON_TARGET:
        first[c] = q_abs_first_round(eps, viol_tol=viol_tol, tol=tol,
                                     components=(c,))["q_abs"]
        full[c] = q_abs_full(eps, viol_tol=viol_tol, tol=tol,
                             components=(c,))["q_abs"]
    return {"first_round": first, "all_rounds": full}


def psi_component_study(eps: float, tols=(1e-10, 1e-12, 1e-14),
                        bisect_tol=1e-11, q_values=None) -> dict:
    """Why the Psi-only ALL-ROUND threshold is not sharply resolved.

    The Phi- violation near q_abs happens at n = 0 and its Delta p grows
    linearly in (q - q_abs), so the overall threshold is localised to ~1e-10.
    The Psi+/Psi- violation, taken on its own, instead happens at a later
    round and becomes tangential as q decreases: the violating round grows and
    Delta p shrinks towards the detection tolerance.  This records both effects
    rather than quoting a single number for them.
    """
    thresholds = {}
    for c in ("Psi+", "Psi-"):
        vals = {}
        for t in tols:
            vals[f"{t:g}"] = q_abs_full(eps, viol_tol=t, tol=bisect_tol,
                                        components=(c,))["q_abs"]
        finite = [v for v in vals.values() if v is not None]
        thresholds[c] = {"values": vals,
                         "spread": (max(finite) - min(finite)) if finite else None}
    rounds = []
    if q_values is None:
        base = thresholds["Psi+"]["values"][f"{VIOL_TOL:g}"]
        q_values = [] if base is None else [base + d for d in
                                            (1e-9, 2e-6, 4e-5, 2.5e-4, 6e-3, 1.1e-2)]
    for q in q_values:
        r = run_trajectory(eps, float(q), components=("Psi+",))
        rounds.append({"epsilon": eps, "q": float(q), "violated": r["violated"],
                       "violation_round": r["round"],
                       "delta_p_violating": r.get("delta_p_violating"),
                       "c_off_at_stop": r["c_off_final"],
                       "c_off_max": r["c_off_max"], "trusted": r["trusted"]})
    return {"thresholds": thresholds, "violation_round_vs_q": rounds}


def tolerance_study(eps_values, tols=(1e-10, 1e-12, 1e-14), bisect_tol=1e-11) -> dict:
    """How much does q_abs move when the violation tolerance moves by 1e4?"""
    out = {}
    for eps in eps_values:
        vals = {}
        for t in tols:
            vals[f"{t:g}"] = q_abs_full(eps, viol_tol=t, tol=bisect_tol)["q_abs"]
        finite = [v for v in vals.values() if v is not None]
        out[f"{eps:g}"] = {"values": vals,
                           "spread": (max(finite) - min(finite)) if finite else None}
    return out


def near_threshold_scan(eps: float, q_star: float,
                        deltas=(1e-7, 1e-6, 1e-4, 1e-3)) -> list:
    """Section 15: check both sides of the located threshold at several scales."""
    rows = []
    for d in deltas:
        for side, q in (("below", q_star - d), ("above", q_star + d)):
            if q < 0:
                continue
            r = run_trajectory(eps, float(q))
            rows.append({
                "epsilon": eps, "delta": d, "side": side, "q": float(q),
                "violated": r["violated"],
                "violation_round": r["round"],
                "violating_component": r["component"],
                "delta_p_violating": r.get("delta_p_violating"),
                "worst_delta_p": r["worst_delta_p"],
                "worst_component": r["worst_component"],
                "worst_round": r["worst_round"],
                "rounds_run": r["rounds_run"], "converged": r["converged"],
                "trusted": r["trusted"], "n_trusted": r["n_trusted"],
                "c_off_max": r["c_off_max"]})
    return rows


# ===========================================================================
# 8. transverse stability of the Bell-diagonal manifold (separate diagnostic)
# ===========================================================================
#
# This does NOT enter the threshold definition.  It answers a different
# question: if the state carries a small Bell off-diagonal component, does the
# Type-4 round contract it or amplify it?  The answer sets how many rounds a
# finite-precision repeated simulation can be trusted for, because the exactly
# Bell-diagonal initial state only ever leaves the manifold through round-off.

TRANSVERSE_SEED = 1e-12
TRANSVERSE_ROUNDS = 60
TRANSVERSE_FIT = (10, 40)


def seeded_rho(eps: float, seed: float, pair=("Phi+", "Phi-")) -> np.ndarray:
    """rho_0(eps) plus a Hermitian, trace-zero Bell off-diagonal perturbation."""
    u, v = BELL[pair[0]], BELL[pair[1]]
    delta = np.outer(u, v.conj()) + np.outer(v, u.conj())
    return isotropic_rho(eps) + seed * delta


def transverse_growth(eps: float, q: float, seed=TRANSVERSE_SEED,
                      n_rounds=TRANSVERSE_ROUNDS, fit=TRANSVERSE_FIT) -> dict:
    """Geometric-mean amplitude multiplier per round for the off-diagonal part,

        lambda_perp = ( a_hi / a_lo ) ^ (1 / (hi - lo)),
        a_n         = sqrt( C_off(rho_n) ),

    taken over a window after the transient so the number is asymptotic."""
    rho = seeded_rho(eps, seed)
    eig0 = float(np.linalg.eigvalsh(rho).min())
    amps = [float(np.sqrt(bell_offdiagonal_weight(rho)))]
    for _ in range(n_rounds):
        rho, _ = one_round(rho, q)
        amps.append(float(np.sqrt(bell_offdiagonal_weight(rho))))
    a = np.array(amps)
    lo, hi = fit[0], min(fit[1], len(a) - 1)
    if a[lo] <= 0.0 or a[hi] <= 0.0 or hi <= lo:
        rate = float("nan")
    else:
        rate = float((a[hi] / a[lo]) ** (1.0 / (hi - lo)))
    return {"epsilon": eps, "q": q, "seed": seed, "growth_rate": rate,
            "a_initial": a[0], "a_final": a[-1], "amps": a,
            "fp_locked": bool(a[hi] == a[lo]),
            "seeded_state_min_eig": eig0}


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
    paths = []
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        paths.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


def plot_curve(dense, table):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    _despine(ax)
    ax.plot([r["epsilon"] for r in dense], [r["q_abs_full"] for r in dense],
            color=BELL_COLOR["Phi+"], linewidth=2.0)
    ax.plot([r["epsilon"] for r in table], [r["q_abs_full"] for r in table],
            linestyle="none", marker="o", markersize=6,
            markerfacecolor="white", markeredgecolor=BELL_COLOR["Phi+"],
            markeredgewidth=1.6, label="high-precision points")
    ax.set_xlabel(r"input mixing $\epsilon$")
    ax.set_ylabel(r"$q_{\mathrm{abs}}^{(4)}(\epsilon)$")
    ax.set_xlim(0, max(r["epsilon"] for r in dense))
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", labelcolor=INK)
    ax.set_title("Type 4: absolute Bell-error-suppression threshold\n"
                 "(from the 14-CNOT circuit simulation)", color=INK, pad=8)
    return save(fig, "type4_qabs_curve")


def plot_first_vs_full(dense, table):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7),
                             gridspec_kw=dict(wspace=0.30))
    for ax in axes:
        _despine(ax)
    e = np.array([r["epsilon"] for r in dense])
    full = np.array([r["q_abs_full"] for r in dense], dtype=float)
    first = np.array([r["q_abs_first_round"] for r in dense], dtype=float)
    axes[0].plot(e, full, color=BELL_COLOR["Phi+"], linewidth=3.4, alpha=0.85,
                 label=r"$q_{\mathrm{abs}}$  (all rounds)")
    axes[0].plot(e, first, color=BELL_COLOR["Phi-"], linewidth=1.4,
                 linestyle=(0, (5, 3)),
                 label=r"$q_{\mathrm{abs}}$  (first round only)")
    axes[0].set_ylabel(r"$q_{\mathrm{abs}}^{(4)}$")
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
    fig.suptitle("Type 4: all-round versus first-round threshold",
                 fontsize=13, color=INK, y=1.02)
    return save(fig, "type4_qabs_first_vs_full")


def plot_components(table):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7),
                             gridspec_kw=dict(wspace=0.30))
    for ax in axes:
        _despine(ax)
    e = [r["epsilon"] for r in table]
    keys = {"Phi-": "q_Phi_minus", "Psi+": "q_Psi_plus", "Psi-": "q_Psi_minus"}
    for name, key in keys.items():
        style = dict(linewidth=1.4, linestyle=(0, (5, 3))) if name == "Psi-" \
            else dict(linewidth=1.9)
        axes[0].plot(e, [r[key] for r in table], color=BELL_COLOR[name],
                     marker=BELL_MARKER[name], markeredgecolor="white",
                     label=BELL_TEX[name], **style)
    axes[0].set_ylabel(r"per-component threshold")
    axes[0].legend(loc="upper left", labelcolor=INK)
    axes[0].set_ylim(0, None)

    base = np.array([r[keys[r["limiting_Bell_component"]]]
                     if r["limiting_Bell_component"] in keys
                     else min(r[k] for k in keys.values()) for r in table],
                    dtype=float)
    for name, key in keys.items():
        v = np.array([r[key] for r in table], dtype=float) - base
        style = dict(linewidth=1.4, linestyle=(0, (5, 3))) if name == "Psi-" \
            else dict(linewidth=1.9)
        axes[1].plot(e, v, color=BELL_COLOR[name], marker=BELL_MARKER[name],
                     markeredgecolor="white", label=BELL_TEX[name], **style)
    axes[1].set_ylabel("threshold minus the smallest of the three")
    axes[1].legend(loc="upper left", labelcolor=INK)
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.set_xlabel(r"input mixing $\epsilon$")
        ax.text(-0.15, 1.05, tag, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=INK)
    fig.suptitle("Type 4: which Bell component sets the threshold "
                 "(first-round definition)", fontsize=13, color=INK, y=1.02)
    return save(fig, "type4_qabs_components")


def plot_eps015_trajectories(cases, eps, q_star):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(cases), figsize=(4.1 * len(cases), 7.6),
                             gridspec_kw=dict(hspace=0.30, wspace=0.30))
    for col, (label, q, rows, viol) in enumerate(cases):
        n = np.array([r["n"] for r in rows])
        pops = np.array([r["p"] for r in rows])
        top, bot = axes[0][col], axes[1][col]
        for k, name in enumerate(BELL_ORDER):
            style = dict(linewidth=1.5, linestyle=(0, (5, 3))) if name == "Psi-" \
                else dict(linewidth=1.8)
            top.plot(n, pops[:, k], color=BELL_COLOR[name],
                     marker=BELL_MARKER[name], markevery=(k, max(1, len(n) // 10)),
                     markeredgecolor="white", label=BELL_TEX[name], **style)
            if name != "Phi+":
                bot.plot(n, pops[:, k], color=BELL_COLOR[name],
                         marker=BELL_MARKER[name],
                         markevery=(k, max(1, len(n) // 10)),
                         markeredgecolor="white", label=BELL_TEX[name], **style)
        top.set_ylim(0, 1)
        dq = q - q_star
        top.set_title(f"{label}\n$q = q_{{\\mathrm{{abs}}}} {dq:+.1e}$",
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
            bot.annotate(f"first violation: {BELL_TEX[viol['component']]}"
                         f" at $n={viol['round']}$\n"
                         rf"$\Delta p = {viol['delta_p']:+.2e}$",
                         xy=(0.04, 0.80), xycoords="axes fraction",
                         fontsize=9, color=INK, va="top")
        else:
            bot.annotate("no violation:\nall three decrease\nat every round",
                         xy=(0.50, 0.66), xycoords="axes fraction",
                         fontsize=9, color=INK, va="top")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.03), labelcolor=INK)
    fig.suptitle(rf"Type 4 at $\epsilon = {eps:g}$: below and above "
                 rf"$q_{{\mathrm{{abs}}}} = {q_star:.10f}$",
                 fontsize=13, color=INK, y=0.985)
    return save(fig, "type4_qabs_eps015_trajectories")


def plot_transverse(at_threshold, in_q):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7),
                             gridspec_kw=dict(wspace=0.30))
    for ax in axes:
        _despine(ax)
    e = [r["epsilon"] for r in at_threshold]
    g = [r["growth_rate"] for r in at_threshold]
    axes[0].axhline(1.0, color=INK2, linewidth=0.9)
    axes[0].plot(e, g, color=BELL_COLOR["Phi+"], marker="o",
                 markeredgecolor="white", linewidth=1.9)
    axes[0].set_xlabel(r"input mixing $\epsilon$")
    axes[0].set_ylabel(r"$\lambda_\perp$  (per round)")
    axes[0].set_title(r"at $q = q_{\mathrm{abs}}(\epsilon) - 10^{-6}$",
                      color=INK, fontsize=11)
    eps_vals = sorted({r["epsilon"] for r in in_q})
    palette = [BELL_COLOR["Phi-"], BELL_COLOR["Psi+"], BELL_COLOR["Psi-"]]
    axes[1].axhline(1.0, color=INK2, linewidth=0.9)
    series = {}
    for k, ev in enumerate(eps_vals):
        sub = sorted((r for r in in_q if r["epsilon"] == ev),
                     key=lambda r: r["q"])
        series[ev] = np.array([r["growth_rate"] for r in sub], dtype=float)
        # the curves coincide, so draw the first one wide and the rest on top
        style = dict(linewidth=4.0, alpha=0.5) if k == 0 else \
            dict(linewidth=1.4, linestyle=(0, (5, 3)))
        axes[1].plot([r["q"] for r in sub], [r["growth_rate"] for r in sub],
                     color=palette[k % len(palette)],
                     marker=BELL_MARKER[BELL_ORDER[1 + k % 3]],
                     markeredgecolor="white",
                     label=rf"$\epsilon = {ev:g}$", **style)
    if len(eps_vals) == 2:
        a, b = (series[ev] for ev in eps_vals)
        axes[1].annotate(rf"the two curves coincide:  $\max_q |\lambda_\perp"
                         rf"(\epsilon_1) - \lambda_\perp(\epsilon_2)|"
                         rf" = {float(np.max(np.abs(a - b))):.1e}$",
                         xy=(0.04, 0.72), xycoords="axes fraction",
                         fontsize=9.5, color=INK)
    axes[1].set_xlabel(r"per-CNOT noise $q$")
    axes[1].set_ylabel(r"$\lambda_\perp$  (per round)")
    axes[1].set_title(r"at fixed $\epsilon$, as a function of $q$",
                      color=INK, fontsize=11)
    axes[1].legend(loc="upper left", labelcolor=INK)
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.text(-0.15, 1.05, tag, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=INK)
    fig.suptitle("Type 4: transverse growth of a Bell off-diagonal "
                 "perturbation (separate diagnostic)",
                 fontsize=13, color=INK, y=1.02)
    return save(fig, "type4_transverse_growth")


# ===========================================================================
# 10. driver
# ===========================================================================

EPS_TABLE = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.65)
CIRCUIT_SOURCE = ("PQEC-Operational-Threshold/pqec_resynth_noise.py, "
                  "module-level GATES list, commit 621d593c, sha256 "
                  "8d66cea64f494443ba55983a8b82f220e27184710298e75fba4df313d56078ee")


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
    print("TYPE-4 CIRCUIT: exact gate sequence actually used")
    print("=" * 78)
    print(f"  source : {CIRCUIT_SOURCE}")
    print("  wires  : (a, A1, A2, B1, B2) = (0, 1, 2, 3, 4)")
    print(f"  CNOT count = {N_CNOT}   (expected 14)\n")
    print(format_sequence())

    print("\n" + "=" * 78)
    print("SANITY CHECKS")
    print("=" * 78)
    rng = np.random.default_rng(20261004)
    checks = {
        "D_circuit_unitary_vs_H_CSWAP_CSWAP_H": check_circuit_unitary(),
        "B_noise_channel": check_noise_channel(rng, 10),
        "C_noiseless_equals_rho2_over_purity": check_noiseless_purification(rng, 25),
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
    print("THRESHOLD SEARCH  (predicate = actual circuit repeated simulation)")
    print("=" * 78)
    table = []
    for eps in EPS_TABLE:
        t0 = time.time()
        full = q_abs_full(eps, tol=args.bisect_tol)
        first = q_abs_first_round(eps, tol=args.bisect_tol)
        comps = per_component_thresholds(eps, tol=args.bisect_tol)
        fr, fu = comps["first_round"], comps["all_rounds"]
        smallest = min((v for v in fr.values() if v is not None), default=None)
        limiting_from_components = next(
            (c for c, v in fr.items() if v == smallest), None)
        row = {
            "epsilon": eps,
            "q_abs_full": full["q_abs"],
            "q_abs_first_round": first["q_abs"],
            "q_abs_full_minus_first": (None if full["q_abs"] is None
                                       or first["q_abs"] is None
                                       else full["q_abs"] - first["q_abs"]),
            "q_Phi_minus": fr["Phi-"], "q_Psi_plus": fr["Psi+"],
            "q_Psi_minus": fr["Psi-"],
            "q_Psi_plus_minus_Psi_minus": (None if fr["Psi+"] is None
                                           or fr["Psi-"] is None
                                           else fr["Psi+"] - fr["Psi-"]),
            "q_Phi_minus_all_rounds": fu["Phi-"],
            "q_Psi_plus_all_rounds": fu["Psi+"],
            "q_Psi_minus_all_rounds": fu["Psi-"],
            "limiting_Bell_component": limiting_from_components,
            "limiting_at_q_upper": full["limiting_component"],
            "first_violation_round_near_threshold": full["first_violation_round"],
            "threshold_bracket_width": full.get("bracket_width"),
            "q_upper_violating": full.get("q_upper"),
            "Delta_p_at_q_upper": full.get("delta_p_at_violation"),
            "max_off_Bell_weight_before_convergence": full.get("below_c_off_max"),
            "below_threshold_converged": full.get("below_converged"),
            "below_threshold_trusted": full.get("below_trusted"),
            "below_threshold_n_trusted": full.get("below_n_trusted"),
            "below_threshold_rounds_run": full.get("below_rounds_run"),
            "below_worst_delta_p": full.get("below_worst_delta_p"),
            "below_worst_component": full.get("below_worst_component"),
            "below_worst_round": full.get("below_worst_round"),
            "bracket_monotone": full["bracket_info"].get("monotone"),
            "bracket_sign_changes": full["bracket_info"].get("n_sign_changes"),
            # NOTE: p_n / p_np1 below are evaluated at q_upper_violating (the
            # smallest q the bisection knows to violate), NOT at q_abs_full.
            "p_n_at_q_upper": (None if full.get("p_n_at_q_upper") is None
                               else full["p_n_at_q_upper"].tolist()),
            "p_np1_at_q_upper": (None if full.get("p_np1_at_q_upper") is None
                                 else full["p_np1_at_q_upper"].tolist()),
            "seconds": round(time.time() - t0, 2),
        }
        table.append(row)
        print(f"  eps={eps:<5} q_abs_full={_fmt(row['q_abs_full'])}  "
              f"q_abs_first={_fmt(row['q_abs_first_round'])}  "
              f"diff={_fmt(row['q_abs_full_minus_first'], '+.2e')}  "
              f"limiting={str(row['limiting_Bell_component']):<5} "
              f"round={row['first_violation_round_near_threshold']}  "
              f"dp={_fmt(row['Delta_p_at_q_upper'], '.3e')}  "
              f"conv={row['below_threshold_converged']} "
              f"trusted={row['below_threshold_trusted']}  ({row['seconds']}s)")

    print("\n  per-component first-round thresholds (smallest = limiting)")
    for row in table:
        print(f"    eps={row['epsilon']:<5} Phi-={_fmt(row['q_Phi_minus'])}  "
              f"Psi+={_fmt(row['q_Psi_plus'])}  Psi-={_fmt(row['q_Psi_minus'])}  "
              f"Psi+ - Psi-={_fmt(row['q_Psi_plus_minus_Psi_minus'], '+.2e')}")

    print("\n  per-component all-round thresholds")
    for row in table:
        print(f"    eps={row['epsilon']:<5} "
              f"Phi-={_fmt(row['q_Phi_minus_all_rounds'])}  "
              f"Psi+={_fmt(row['q_Psi_plus_all_rounds'])}  "
              f"Psi-={_fmt(row['q_Psi_minus_all_rounds'])}")

    print("\n  violation-tolerance sensitivity (VIOL_TOL 1e-10 / 1e-12 / 1e-14)")
    tol_study = tolerance_study((0.05, 0.15, 0.40, 0.65), bisect_tol=args.bisect_tol)
    for eps, d in tol_study.items():
        vals = "  ".join(f"{k}:{_fmt(v)}" for k, v in d["values"].items())
        print(f"    eps={eps:<5} {vals}   spread={_fmt(d['spread'], '.2e')}")

    print("\n  Psi-only ALL-ROUND threshold: tolerance sensitivity at eps=0.15")
    psi = psi_component_study(0.15, bisect_tol=args.bisect_tol)
    for c, d in psi["thresholds"].items():
        vals = "  ".join(f"{k}:{_fmt(v)}" for k, v in d["values"].items())
        print(f"    {c:<5} {vals}   spread={_fmt(d['spread'], '.2e')}")
    print("    violating round as q approaches that threshold from above")
    for r in psi["violation_round_vs_q"]:
        print(f"      q={r['q']:.9f} round={r['violation_round']} "
              f"dp={_fmt(r['delta_p_violating'], '.3e')} "
              f"C_off={r['c_off_at_stop']:.2e} trusted={r['trusted']}")

    # ---- near-threshold scan -------------------------------------------
    print("\n  near-threshold check at delta = 1e-7, 1e-6, 1e-4, 1e-3")
    near = []
    for row in table:
        if row["q_abs_full"] is None:
            continue
        near += near_threshold_scan(row["epsilon"], row["q_abs_full"])
    bad = [r for r in near
           if (r["side"] == "below" and r["violated"])
           or (r["side"] == "above" and not r["violated"])]
    print(f"    {len(near)} tests; unexpected outcomes = {len(bad)}")
    for r in bad:
        print(f"      UNEXPECTED  eps={r['epsilon']} delta={r['delta']:g} "
              f"{r['side']} q={r['q']:.12f} violated={r['violated']}")
    rounds_above = sorted({r["violation_round"] for r in near
                           if r["side"] == "above" and r["violated"]})
    comps_above = sorted({r["violating_component"] for r in near
                          if r["side"] == "above" and r["violated"]})
    print(f"    violation rounds seen on the above side: {rounds_above}")
    print(f"    violating components on the above side:  {comps_above}")

    # ---- dense curve ---------------------------------------------------
    print("\n  dense epsilon grid ...")
    dense = []
    grid = np.linspace(args.eps_min_dense, args.eps_max_dense, args.dense_points)
    hint_f = hint_g = None
    for eps in grid:
        f = q_abs_full(float(eps), tol=args.bisect_tol, hint=hint_f)
        g = q_abs_first_round(float(eps), tol=args.bisect_tol, hint=hint_g)
        hint_f, hint_g = f["q_abs"], g["q_abs"]
        dense.append({"epsilon": float(eps), "q_abs_full": f["q_abs"],
                      "q_abs_first_round": g["q_abs"],
                      "limiting_Bell_component": f["limiting_component"],
                      "first_violation_round_near_threshold": f["first_violation_round"],
                      "threshold_bracket_width": f.get("bracket_width")})
    diffs = [abs(r["q_abs_full"] - r["q_abs_first_round"]) for r in dense
             if r["q_abs_full"] is not None and r["q_abs_first_round"] is not None]
    limiting_set = sorted({r["limiting_Bell_component"] for r in dense})
    rounds_set = sorted({r["first_violation_round_near_threshold"] for r in dense})
    print(f"    {len(dense)} points;  max |full - first| over the grid = "
          f"{max(diffs):.3e}")
    print(f"    limiting components over the grid: {limiting_set}")
    print(f"    first-violation rounds over the grid: {rounds_set}")

    # ---- eps = 0.15 trajectories ---------------------------------------
    eps_d = 0.15
    q_star = next(r["q_abs_full"] for r in table if r["epsilon"] == eps_d)
    cases = []
    for label, q in (("well below threshold", q_star - 5e-3),
                     ("just below threshold", q_star - 1e-7),
                     ("just above threshold", q_star + 1e-3)):
        rows, viol = record_trajectory(eps_d, q, 20)
        cases.append((label, q, rows, viol))
        res = run_trajectory(eps_d, q)
        v = (f"violation: {res['component']} at n={res['round']}, "
             f"Delta p={res['delta_p_violating']:+.4e}") if res["violated"] \
            else (f"no violation; worst Delta p = {res['worst_delta_p']:+.3e} "
                  f"({res['worst_component']} at n={res['worst_round']}), "
                  f"converged={res['converged']}, trusted={res['trusted']}")
        print(f"    eps=0.15, q={q:.12f}  [{label}]  {v}")
        with open(OUT / f"type4_qabs_eps015_q{q:.8f}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["round", "p_phi_plus", "p_phi_minus", "p_psi_plus",
                        "p_psi_minus", "d_phi_minus", "d_psi_plus", "d_psi_minus",
                        "C_offdiag"])
            for r in rows:
                w.writerow([r["n"], *r["p"], r["dp"][1], r["dp"][2], r["dp"][3],
                            r["c_off"]])

    # ---- transverse stability (separate diagnostic) ---------------------
    print("\n" + "=" * 78)
    print("TRANSVERSE STABILITY OF THE BELL-DIAGONAL MANIFOLD")
    print("  (separate diagnostic; does not enter the q_abs definition)")
    print("=" * 78)
    at_thr = []
    for row in table:
        if row["q_abs_full"] is None:
            continue
        q = max(row["q_abs_full"] - 1e-6, 0.0)
        g = transverse_growth(row["epsilon"], q)
        at_thr.append({"epsilon": row["epsilon"], "q_abs": row["q_abs_full"],
                       "q": q, "growth_rate": g["growth_rate"],
                       "a_initial": g["a_initial"], "a_final": g["a_final"],
                       "fp_locked": g["fp_locked"],
                       "seeded_state_min_eig": g["seeded_state_min_eig"]})
        print(f"    eps={row['epsilon']:<5} q={q:.12f}  "
              f"lambda_perp={g['growth_rate']:.6f}  "
              f"a: {g['a_initial']:.3e} -> {g['a_final']:.3e}"
              + ("   [fp-locked]" if g["fp_locked"] else ""))
    in_q = []
    print("\n    lambda_perp as a function of q at fixed eps")
    for ev in (0.15, 0.50):
        for q in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
            g = transverse_growth(ev, q)
            in_q.append({"epsilon": ev, "q": q, "growth_rate": g["growth_rate"],
                         "fp_locked": g["fp_locked"]})
            print(f"      eps={ev:<5} q={q:<6g} lambda_perp={g['growth_rate']:.6f}"
                  + ("   [fp-locked]" if g["fp_locked"] else ""))

    # ---- outputs --------------------------------------------------------
    main_cols = ["epsilon", "q_abs_full", "q_abs_first_round",
                 "q_abs_full_minus_first", "q_Phi_minus", "q_Psi_plus",
                 "q_Psi_minus", "q_Psi_plus_minus_Psi_minus",
                 "limiting_Bell_component",
                 "first_violation_round_near_threshold",
                 "threshold_bracket_width", "q_upper_violating",
                 "Delta_p_at_q_upper", "seconds"]
    diag_cols = ["epsilon", "max_off_Bell_weight_before_convergence",
                 "below_threshold_converged", "below_threshold_trusted",
                 "below_threshold_n_trusted", "below_threshold_rounds_run",
                 "below_worst_delta_p", "below_worst_component",
                 "below_worst_round", "q_Phi_minus_all_rounds",
                 "q_Psi_plus_all_rounds", "q_Psi_minus_all_rounds",
                 "limiting_at_q_upper", "bracket_monotone",
                 "bracket_sign_changes"]
    with open(OUT / "type4_qabs_circuit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=main_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(table)
    with open(OUT / "type4_qabs_diagnostics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=diag_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(table)
    with open(OUT / "type4_qabs_dense.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dense[0]))
        w.writeheader()
        w.writerows(dense)
    with open(OUT / "type4_qabs_near_threshold.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(near[0]))
        w.writeheader()
        w.writerows(near)
    with open(OUT / "type4_qabs_psi_component_study.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(psi["violation_round_vs_q"][0]))
        w.writeheader()
        w.writerows(psi["violation_round_vs_q"])
    with open(OUT / "type4_transverse_growth.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(at_thr[0]))
        w.writeheader()
        w.writerows(at_thr)
    with open(OUT / "type4_transverse_growth_vs_q.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(in_q[0]))
        w.writeheader()
        w.writerows(in_q)

    meta = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "circuit_source": CIRCUIT_SOURCE,
        "n_cnot": N_CNOT,
        "wire_order": "(a, A1, A2, B1, B2) = (0, 1, 2, 3, 4)",
        "gate_sequence": [[k, list(a)] for k, a in TYPE4_OPS],
        "noise_model": "two-qubit replacement depolarizing after every CNOT",
        "readout": ("tau_A = Tr_B[<0|sigma|0>_a - <1|sigma|1>_a]; "
                    "rho_next = tau_A/Tr(tau_A)  (parity-weighted, not postselected)"),
        "no_twirling": True,
        "used_analytic_input": False,
        "checks": {k: {kk: (float(vv) if isinstance(vv, float) else vv)
                       for kk, vv in v.items()} for k, v in checks.items()},
        "tolerance_study": tol_study,
        "psi_component_study_eps015": psi,
        "table": table,
        "dense_max_abs_full_minus_first": float(max(diffs)),
        "dense_limiting_components": limiting_set,
        "dense_first_violation_rounds": [int(r) if r is not None else None
                                         for r in rounds_set],
        "near_threshold_unexpected": bad,
        "transverse_at_threshold": at_thr,
        "transverse_vs_q": in_q,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    (OUT / "type4_qabs_verification.json").write_text(json.dumps(meta, indent=2))

    if not args.no_figures:
        use_style()
        plot_curve(dense, table)
        plot_first_vs_full(dense, table)
        plot_components(table)
        plot_eps015_trajectories(cases, eps_d, q_star)
        plot_transverse(at_thr, in_q)
    print("\n" + "=" * 78)
    print("FILES")
    print("=" * 78)
    for p in sorted(OUT.glob("*")):
        print(f"  {p.relative_to(ROOT)}")
    print(f"\n  elapsed {meta['elapsed_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
