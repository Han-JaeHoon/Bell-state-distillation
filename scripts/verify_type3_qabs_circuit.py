"""Blind circuit-level verification of the Type-3 absolute Bell-error-suppression
threshold q_abs^(3)(epsilon).

SELF-CONTAINED ON PURPOSE.  This script imports nothing from ``pqec_distill``,
nothing from the parent research repository, and no analytic recurrence.  The
only prior information it uses is

  1. the canonical Type-3 physical gate sequence, transcribed below from
     PQEC-Operational-Threshold/verify_analytic_decomposed.py
     (functions ``_c2`` / ``_tof`` / ``_fred``, commit a4969947,
      sha256 dd8de89220cb0b53...),
  2. the Bell-isotropic input state,
  3. the per-CNOT two-qubit replacement depolarizing channel,
  4. the operational definition of q_abs given in the task.

No Bell-diagonal reduced map, no (x,y,z) or (u,v) recurrence, no closed-form
population map, no previously computed threshold is used anywhere.  Every
density matrix is propagated in full, and every round feeds its exact output
state into the next round -- no twirling, no re-isotropisation, no Bell-diagonal
projection, no symmetrisation, no off-diagonal truncation.

DEFINITION BEING TESTED

    q_abs(eps) = sup { q : p_i^(n+1) < p_i^(n)
                           for every i in {Phi-, Psi+, Psi-} and every n >= 0 }

with p_i^(n) = <B_i| rho_n |B_i> read off the full two-qubit density matrix.
Which component is limiting, and whether the first round is the binding one,
are OUTPUTS here, not assumptions.

Run:
    python scripts/verify_type3_qabs_circuit.py
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
OUT = ROOT / "results" / "type3_qabs"

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
    target = b, six CNOTs, single-qubit gates ideal.  The single-qubit gates are
    kept in their original positions: the decomposition, not just the unitary,
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


def format_sequence() -> str:
    lines, k = [], 0
    for kind, arg in TYPE3_OPS:
        if kind == "CNOT":
            k += 1
            lines.append(f"  {len(lines)+1:3d}.  CNOT({arg[0]} -> {arg[1]})"
                         f"      [CNOT #{k}]  + depolarizing D_q on "
                         f"({arg[0]},{arg[1]})")
        else:
            lines.append(f"  {len(lines)+1:3d}.  {kind}({arg})")
    return "\n".join(lines)


def build_program():
    """Merge runs of unitaries into single 32x32 matrices; keep the channel
    locations exactly where the CNOTs are.  Returns [("U", M) | ("D", (i,j))]."""
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
    """Noiseless 32x32 unitary of the transcribed sequence."""
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
# 4. one Type-3 round on a full two-qubit density matrix
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


def check_noise_channel(rng, n_trials=25) -> dict:
    """(B) trace preservation, hermiticity, positivity, q=0 identity, and the
    q=1 limit: the chosen pair becomes I/4 and the other wires' reduced state
    is untouched."""
    worst = {"trace": 0.0, "herm": 0.0, "min_eig": 0.0, "identity_at_q0": 0.0,
             "q1_pair_is_I_over_4": 0.0, "q1_rest_preserved": 0.0}
    pairs = [(0, 1), (1, 3), (2, 4), (3, 4), (0, 4)]
    for _ in range(n_trials):
        sigma = random_density_matrix(rng, DIM)
        for pair in pairs:
            for q in (0.0, 0.1, 0.5, 0.9, 1.0):
                out = replacement_depol(sigma, pair, q)
                worst["trace"] = max(worst["trace"],
                                     abs(np.trace(out) - 1.0))
                worst["herm"] = max(worst["herm"],
                                    np.linalg.norm(out - out.conj().T))
                worst["min_eig"] = min(worst["min_eig"], float(
                    np.linalg.eigvalsh(0.5 * (out + out.conj().T)).min()))
            worst["identity_at_q0"] = max(
                worst["identity_at_q0"],
                float(np.max(np.abs(replacement_depol(sigma, pair, 0.0) - sigma))))

            out1 = replacement_depol(sigma, pair, 1.0)
            # the pair's reduced state must be exactly I/4
            rest = [w for w in range(N_WIRES) if w not in pair]
            worst["q1_pair_is_I_over_4"] = max(
                worst["q1_pair_is_I_over_4"],
                float(np.max(np.abs(_reduced(out1, list(pair))
                                    - np.eye(4) / 4))))
            # the remaining three wires must be untouched
            worst["q1_rest_preserved"] = max(
                worst["q1_rest_preserved"],
                float(np.max(np.abs(_reduced(out1, rest) - _reduced(sigma, rest)))))
    return {k: float(v) for k, v in worst.items()}


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


def check_circuit_unitary() -> dict:
    """(D) the noiseless 16-CNOT sequence must implement
    H_a CSWAP(a;A2,B2) CSWAP(a;A1,B1) H_a, up to a global phase."""
    def cswap(control, x, y):
        a = [_I2] * N_WIRES
        a[control] = _P0
        keep = _kron_list(a)
        # |1><1|_c (x) SWAP_xy
        swap = np.eye(DIM, dtype=complex)
        perm = np.arange(DIM)
        for idx in range(DIM):
            bits = [(idx >> (N_WIRES - 1 - w)) & 1 for w in range(N_WIRES)]
            bits[x], bits[y] = bits[y], bits[x]
            perm[idx] = sum(b << (N_WIRES - 1 - w) for w, b in enumerate(bits))
        swap = np.eye(DIM, dtype=complex)[:, perm]
        b = [_I2] * N_WIRES
        b[control] = _P1
        proj1 = _kron_list(b)
        return keep + proj1 @ swap

    target = single_qubit_gate("H", ANCILLA)
    target = cswap(ANCILLA, 1, 3) @ target
    target = cswap(ANCILLA, 2, 4) @ target
    target = single_qubit_gate("H", ANCILLA) @ target

    v = circuit_unitary()
    phase = np.exp(-1j * np.angle(np.trace(target.conj().T @ v)))
    diff = phase * v - target
    # also a direct basis-state test, independent of the phase alignment
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
            "unitarity": float(np.max(np.abs(v.conj().T @ v - np.eye(DIM)))),
            "max_abs_amplitude_diff_random_states": worst_state,
            "global_phase": float(np.angle(np.trace(target.conj().T @ v)))}


def check_noiseless_purification(rng, n_trials=25) -> dict:
    """(C) at q = 0 the parity-weighted output must be rho^2 / Tr(rho^2)."""
    worst = 0.0
    worst_weight = 0.0
    for _ in range(n_trials):
        rho = random_density_matrix(rng, 4)
        out, info = one_round(rho, 0.0)
        expected = rho @ rho / np.trace(rho @ rho)
        worst = max(worst, float(np.max(np.abs(out - expected))))
        worst_weight = max(worst_weight,
                           abs(info["weight"] - float(np.real(np.trace(rho @ rho)))))
    return {"max_abs_diff_vs_rho2_over_tr_rho2": worst,
            "max_abs_weight_minus_purity": worst_weight}


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
# At a converged orbit the floating-point map settles into a period-2 limit
# cycle whose Delta p oscillates at ~6.3e-17 -- that is the numerical error
# floor, measured by `measure_error_floor()` below, not assumed.
#
#   STEP_TOL  = 1e-14   iteration stops once max|rho_(n+1) - rho_n| falls below
#                       this: beyond it the orbit is at the numerical fixed
#                       point and any further sign change is round-off.
#   VIOL_TOL  = 1e-12   a round counts as violating only if some non-target
#                       Delta p_i exceeds +VIOL_TOL, i.e. ~1.6e4 times the
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
# MEASURED, not assumed: the Bell-isotropic initial state is built exactly
# Bell-diagonal, and its Bell off-diagonal weight
#     C_off(rho) = sum_{i != j} |<B_i| rho |B_j>|^2
# starts at ~3e-34, i.e. at round-off.  In part of the (eps, q) plane that
# round-off is EXPONENTIALLY AMPLIFIED by the circuit map (measured growth
# ~1.55 per round at eps = 0.5), and once it reaches O(1e-12) it pushes the
# simulated orbit off a saddle and produces a population increase that the
# exact dynamics does not have.  A seed-injection scan confirms the mechanism:
# injecting a deliberate off-diagonal seed s at n = 0 delays the escape by
# +10.4 rounds per decade of s, exactly ln(10)/ln(growth), so the escape round
# diverges as s -> 0.
#
# The guard therefore stops trusting the simulated trajectory once C_off
# exceeds OFFDIAG_TRUST.  At that level the induced population error is
# ~1e-18, six orders below VIOL_TOL, so the guard can never hide a genuine
# violation -- it only refuses to certify one that round-off manufactured.
OFFDIAG_TRUST = 1e-18


def run_trajectory(eps: float, q: float, step_tol=STEP_TOL, n_max=N_MAX,
                   viol_tol=VIOL_TOL, components=NON_TARGET, record=False,
                   offdiag_trust=OFFDIAG_TRUST):
    """Iterate the real circuit and look for the first absolute-suppression
    violation.  Every round's exact output state is the next round's input.

    ``violated`` is True only when some tracked non-target population rose by
    more than ``viol_tol`` *while the trajectory was still trustworthy*, i.e.
    before amplified round-off (measured by the Bell off-diagonal weight)
    could have moved the simulated state away from the exact orbit.
    """
    idx = [BELL_ORDER.index(c) for c in components]
    rho = isotropic_rho(eps)
    p = bell_populations(rho)
    rows = [{"n": 0, "p": p.copy(), "dp": np.full(4, np.nan), "step": np.nan,
             "c_off": bell_offdiagonal_weight(rho)}] if record else []
    worst_dp, worst_at, worst_comp = -np.inf, None, None
    c_off0 = bell_offdiagonal_weight(rho)
    n, step, trusted, n_trusted, converged = 0, np.nan, True, None, False
    while n < n_max:
        rho_next, _ = one_round(rho, q)
        p_next = bell_populations(rho_next)
        dp = p_next - p
        step = float(np.max(np.abs(rho_next - rho)))
        c_off = bell_offdiagonal_weight(rho_next)
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


def first_round_violates(eps: float, q: float, viol_tol=VIOL_TOL,
                         components=NON_TARGET) -> bool:
    """Only rho_0 -> rho_1."""
    rho = isotropic_rho(eps)
    p = bell_populations(rho)
    rho1, _ = one_round(rho, q)
    dp = bell_populations(rho1) - p
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
            "bracket_info": info,
            "limiting_component": detail["component"],
            "first_violation_round": detail["round"],
            "delta_p_at_violation": detail.get("delta_p_violating"),
            "p_n_at_q_upper": detail.get("p_n"),
            "p_np1_at_q_upper": detail.get("p_np1"),
            "all_violating_components": detail.get("all_violating"),
            "below_converged": at_lo["converged"],
            "below_trusted": at_lo["trusted"],
            "below_n_trusted": at_lo["n_trusted"],
            "below_c_off_final": at_lo["c_off_final"]}


def q_abs_first_round(eps: float, viol_tol=VIOL_TOL, tol=1e-10,
                      components=NON_TARGET, q_hi=0.5, n_coarse=40, hint=None):
    pred = lambda q: first_round_violates(eps, q, viol_tol, components)
    lo, hi, info = _hinted_bracket(pred, hint, q_hi, n_coarse)
    if lo is None:
        return {"q_abs": None, "bracket_info": info}
    a, b, it = bisect(pred, lo, hi, tol=tol)
    rho = isotropic_rho(eps)
    rho1, _ = one_round(rho, b)
    dp = bell_populations(rho1) - bell_populations(rho)
    hits = [BELL_ORDER[i] for i in (1, 2, 3) if dp[i] > viol_tol]
    return {"q_abs": a, "q_upper": b, "iterations": it, "bracket_info": info,
            "limiting_component": max(hits, key=lambda c: dp[BELL_ORDER.index(c)])
            if hits else None,
            "all_violating_components": hits}


def per_component_thresholds(eps: float, viol_tol=VIOL_TOL, tol=1e-10) -> dict:
    """q_abs for each non-target component on its own.  The limiting component
    is the one with the smallest of these -- an output, not an assumption."""
    out = {}
    for c in NON_TARGET:
        r = q_abs_full(eps, viol_tol=viol_tol, tol=tol, components=(c,))
        out[c] = r["q_abs"]
    return out


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


# ===========================================================================
# 8. plotting (self-contained: only matplotlib, no project style module)
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
    ax.set_ylabel(r"$q_{\mathrm{abs}}^{(3)}(\epsilon)$")
    ax.set_xlim(0, max(r["epsilon"] for r in dense))
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", labelcolor=INK)
    ax.set_title("Type 3: absolute Bell-error-suppression threshold\n"
                 "(from the 16-CNOT circuit simulation)", color=INK, pad=8)
    return save(fig, "type3_qabs_curve")


def plot_first_vs_full(dense, table):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7),
                             gridspec_kw=dict(wspace=0.28))
    for ax in axes:
        _despine(ax)
    e = [r["epsilon"] for r in dense]
    full = np.array([r["q_abs_full"] for r in dense])
    first = np.array([r["q_abs_first_round"] for r in dense])
    axes[0].plot(e, full, color=BELL_COLOR["Phi+"], linewidth=3.4, alpha=0.85,
                 label=r"$q_{\mathrm{abs}}$  (all rounds)")
    axes[0].plot(e, first, color=BELL_COLOR["Phi-"], linewidth=1.4,
                 linestyle=(0, (5, 3)), label=r"$q_{\mathrm{abs}}$  (first round only)")
    axes[0].set_ylabel(r"$q_{\mathrm{abs}}^{(3)}$")
    axes[0].legend(loc="upper left", labelcolor=INK)
    axes[0].set_ylim(0, None)
    d = np.abs(full - first)
    axes[0].annotate(rf"the two curves coincide:  $\max_\epsilon |q^{{\rm all}} - "
                     rf"q^{{\rm first}}| = {d.max():.1e}$"
                     f"\n({len(e)} grid points)",
                     xy=(0.04, 0.62), xycoords="axes fraction", fontsize=9.5,
                     color=INK)

    # (b) the three per-component thresholds: which one is limiting
    et = [r["epsilon"] for r in table]
    for name, key in (("Phi-", "q_abs_Phi_minus"), ("Psi+", "q_abs_Psi_plus"),
                      ("Psi-", "q_abs_Psi_minus")):
        style = dict(linewidth=1.5, linestyle=(0, (5, 3))) if name == "Psi-" \
            else dict(linewidth=1.8)
        axes[1].plot(et, [r[key] for r in table], color=BELL_COLOR[name],
                     marker=BELL_MARKER[name], markeredgecolor="white",
                     label=BELL_TEX[name], **style)
    axes[1].set_ylabel(r"per-component $q_{\mathrm{abs}}$")
    axes[1].legend(loc="upper left", labelcolor=INK)
    axes[1].set_ylim(0, None)
    axes[1].annotate(r"$\Phi^{-}$ is the smallest at every tested $\epsilon$,"
                     "\nso it sets the threshold",
                     xy=(0.33, 0.12), xycoords="axes fraction", fontsize=9.5,
                     color=INK)
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.set_xlabel(r"input mixing $\epsilon$")
        ax.set_xlim(0, max(e))
        ax.text(-0.14, 1.04, tag, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=INK)
    fig.suptitle("Type 3: all-round vs first-round threshold, and the limiting "
                 "Bell component", fontsize=13, color=INK, y=1.0)
    return save(fig, "type3_qabs_first_vs_full")


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
                         xy=(0.04, 0.78), xycoords="axes fraction",
                         fontsize=9, color=INK)
        else:
            bot.annotate("no violation:\nall three decrease\nat every round",
                         xy=(0.52, 0.62), xycoords="axes fraction",
                         fontsize=9, color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), labelcolor=INK)
    fig.suptitle(rf"Type 3 at $\epsilon = {eps:g}$: below and above "
                 rf"$q_{{\mathrm{{abs}}}} = {q_star:.10f}$",
                 fontsize=13, color=INK, y=0.985)
    return save(fig, "type3_qabs_eps015_trajectories")


# ===========================================================================
# 9. driver
# ===========================================================================

EPS_TABLE = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bisect-tol", type=float, default=1e-11)
    ap.add_argument("--dense-points", type=int, default=61)
    ap.add_argument("--eps-max-dense", type=float, default=0.64)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("=" * 78)
    print("TYPE-3 CIRCUIT: exact gate sequence actually used")
    print("=" * 78)
    print("  source : PQEC-Operational-Threshold/verify_analytic_decomposed.py")
    print("           functions _c2 / _tof / _fred  (commit a4969947,")
    print("           sha256 dd8de89220cb0b53...), transcribed verbatim")
    print("  wires  : (a, A1, A2, B1, B2) = (0, 1, 2, 3, 4)")
    print(f"  CNOT count = {N_CNOT}   (expected 16: two Fredkins x 8)\n")
    print(format_sequence())

    print("\n" + "=" * 78)
    print("SANITY CHECKS")
    print("=" * 78)
    rng = np.random.default_rng(20261002)
    checks = {
        "D_circuit_unitary": check_circuit_unitary(),
        "B_noise_channel": check_noise_channel(rng, 10),
        "C_noiseless_equals_rho2_over_purity": check_noiseless_purification(rng, 20),
        "A_state_validity_eps0.15_q0.02": check_state_validity(0.15, 0.02, 40),
        "error_floor": measure_error_floor(),
        "tolerances": {"STEP_TOL": STEP_TOL, "VIOL_TOL": VIOL_TOL,
                       "N_MAX": N_MAX, "bisect_tol": args.bisect_tol},
    }
    for group, d in checks.items():
        print(f"  {group}")
        for k, v in d.items():
            print(f"      {k:<42} {v:.4e}" if isinstance(v, float)
                  else f"      {k:<42} {v}")

    print("\n" + "=" * 78)
    print("THRESHOLD SEARCH  (predicate = actual circuit repeated simulation)")
    print("=" * 78)
    table = []
    for eps in EPS_TABLE:
        t0 = time.time()
        full = q_abs_full(eps, tol=args.bisect_tol)
        first = q_abs_first_round(eps, tol=args.bisect_tol)
        comps = per_component_thresholds(eps, tol=args.bisect_tol)
        dpv = full.get("delta_p_at_violation")
        row = {
            "epsilon": eps,
            "q_abs_full": full["q_abs"],
            "q_upper_violating": full.get("q_upper"),
            "q_abs_first_round": first["q_abs"],
            "q_abs_full_minus_first": (None if full["q_abs"] is None
                                       or first["q_abs"] is None
                                       else full["q_abs"] - first["q_abs"]),
            "limiting_Bell_component": full["limiting_component"],
            "first_violation_round_near_threshold": full["first_violation_round"],
            "Delta_p_at_violation": dpv,
            "q_abs_Phi_minus": comps["Phi-"], "q_abs_Psi_plus": comps["Psi+"],
            "q_abs_Psi_minus": comps["Psi-"],
            "below_threshold_converged": full.get("below_converged"),
            "below_threshold_trusted": full.get("below_trusted"),
            "below_threshold_n_trusted": full.get("below_n_trusted"),
            "bracket_monotone": full["bracket_info"].get("monotone"),
            "bracket_sign_changes": full["bracket_info"].get("n_sign_changes"),
            # NOTE: these populations are evaluated at q_upper_violating (the
            # smallest q the bisection knows to violate), NOT at q_abs_full and
            # NOT at any other q.  Delta_p_at_violation belongs to the same run.
            "p_n_at_q_upper": (None if full.get("p_n_at_q_upper") is None
                               else full["p_n_at_q_upper"].tolist()),
            "p_np1_at_q_upper": (None if full.get("p_np1_at_q_upper") is None
                                 else full["p_np1_at_q_upper"].tolist()),
            "seconds": round(time.time() - t0, 2),
        }
        table.append(row)
        print(f"  eps={eps:<5} q_abs_full={row['q_abs_full']:.12f}  "
              f"q_abs_first={row['q_abs_first_round']:.12f}  "
              f"diff={row['q_abs_full_minus_first']:+.2e}  "
              f"limiting={row['limiting_Bell_component']:<5} "
              f"round={row['first_violation_round_near_threshold']}  "
              f"dp={row['Delta_p_at_violation']:.3e}  "
              f"conv={row['below_threshold_converged']} "
              f"trusted={row['below_threshold_trusted']}  ({row['seconds']}s)")

    print("\n  per-component thresholds (smallest = limiting)")
    for row in table:
        print(f"    eps={row['epsilon']:<5} Phi-={row['q_abs_Phi_minus']:.12f}  "
              f"Psi+={row['q_abs_Psi_plus']:.12f}  "
              f"Psi-={row['q_abs_Psi_minus']:.12f}")

    print("\n  violation-tolerance sensitivity (VIOL_TOL 1e-10 / 1e-12 / 1e-14)")
    tol_study = tolerance_study((0.05, 0.15, 0.40), bisect_tol=args.bisect_tol)
    for eps, d in tol_study.items():
        vals = "  ".join(f"{k}:{v:.12f}" for k, v in d["values"].items())
        print(f"    eps={eps:<5} {vals}   spread={d['spread']:.2e}")

    # ---- dense curve ---------------------------------------------------
    print("\n  dense epsilon grid ...")
    dense = []
    grid = np.linspace(0.005, args.eps_max_dense, args.dense_points)
    hint_f = hint_g = None
    for eps in grid:
        f = q_abs_full(float(eps), tol=args.bisect_tol, hint=hint_f)
        g = q_abs_first_round(float(eps), tol=args.bisect_tol, hint=hint_g)
        hint_f, hint_g = f["q_abs"], g["q_abs"]
        dense.append({"epsilon": float(eps), "q_abs_full": f["q_abs"],
                      "q_abs_first_round": g["q_abs"],
                      "limiting_Bell_component": f["limiting_component"],
                      "first_violation_round_near_threshold": f["first_violation_round"]})
    diffs = [abs(r["q_abs_full"] - r["q_abs_first_round"]) for r in dense
             if r["q_abs_full"] is not None and r["q_abs_first_round"] is not None]
    print(f"    {len(dense)} points;  max |full - first| over the grid = "
          f"{max(diffs):.3e}")

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
        with open(OUT / f"type3_qabs_eps015_q{q:.8f}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["round", "p_phi_plus", "p_phi_minus", "p_psi_plus",
                        "p_psi_minus", "d_phi_minus", "d_psi_plus", "d_psi_minus",
                        "C_offdiag"])
            for r in rows:
                w.writerow([r["n"], *r["p"], r["dp"][1], r["dp"][2], r["dp"][3],
                            r["c_off"]])

    # ---- outputs --------------------------------------------------------
    with open(OUT / "type3_qabs_circuit.csv", "w", newline="") as fh:
        cols = ["epsilon", "q_abs_full", "q_upper_violating", "q_abs_first_round",
                "q_abs_full_minus_first", "limiting_Bell_component",
                "first_violation_round_near_threshold", "Delta_p_at_violation",
                "q_abs_Phi_minus", "q_abs_Psi_plus", "q_abs_Psi_minus",
                "below_threshold_converged", "below_threshold_trusted",
                "below_threshold_n_trusted",
                "bracket_monotone", "bracket_sign_changes", "seconds"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(table)
    with open(OUT / "type3_qabs_curve_dense.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dense[0]))
        w.writeheader()
        w.writerows(dense)

    meta = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "circuit_source": ("PQEC-Operational-Threshold/verify_analytic_decomposed.py "
                           "_c2/_tof/_fred, commit a4969947, "
                           "sha256 dd8de89220cb0b5315f0c5f7b9c2c8cf483980ac96d9ad9cccc1bd3f8e7a7837"),
        "n_cnot": N_CNOT,
        "gate_sequence": [[k, list(a) if isinstance(a, tuple) else a]
                          for k, a in TYPE3_OPS],
        "noise_model": "two-qubit replacement depolarizing after every CNOT",
        "readout": "tau_A = Tr_B[<0|sigma|0>_a - <1|sigma|1>_a]; rho_next = tau_A/Tr(tau_A)",
        "no_twirling": True,
        "checks": {k: {kk: (vv if not isinstance(vv, float) else float(vv))
                       for kk, vv in v.items()} for k, v in checks.items()},
        "tolerance_study": tol_study,
        "table": table,
        "dense_max_abs_full_minus_first": float(max(diffs)),
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    (OUT / "type3_qabs_verification.json").write_text(json.dumps(meta, indent=2))

    written = []
    if not args.no_figures:
        use_style()
        written += plot_curve(dense, table)
        written += plot_first_vs_full(dense, table)
        written += plot_eps015_trajectories(cases, eps_d, q_star)
    print("\n" + "=" * 78)
    print("FILES")
    print("=" * 78)
    for p in sorted(OUT.glob("*")):
        print(f"  {p.relative_to(ROOT)}")
    print(f"\n  elapsed {meta['elapsed_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
