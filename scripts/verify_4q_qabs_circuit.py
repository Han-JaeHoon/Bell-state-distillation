"""Blind circuit-level verification of the 4Q absolute Bell-error-suppression
threshold q_abs^(4Q)(epsilon).

SELF-CONTAINED ON PURPOSE.  This script imports nothing from ``pqec_distill``
and no analytic recurrence.  The only prior information it uses is

  1. the canonical 4Q physical circuit specification, transcribed below from
     the docstring and ``CNOT_SEQUENCE`` of src/pqec_distill/circuit.py
     (commit 69248179, sha256 e475a54da16a3d1c...),
  2. the Bell-isotropic input state,
  3. the per-CNOT two-qubit replacement depolarizing channel,
  4. the physical postselection rule of that specification,
  5. the operational definition of q_abs given in the task.

No Bell-diagonal reduced map, no (x,y,z) recurrence, no closed-form population
or success-probability map, and no previously computed 4Q threshold is used
anywhere.  Every density matrix is propagated in full, and every round feeds
its exact conditional output state into the next round -- no twirling, no
re-isotropisation, no Bell-diagonal projection, no symmetrisation, no
off-diagonal truncation.

DEFINITION BEING TESTED

    q_abs(eps) = sup { q : p_i^(n+1) < p_i^(n)
                           for every i in {Phi-, Psi+, Psi-} and every n >= 0 }

with p_i^(n) = <B_i| rho_n |B_i> read off the full conditional two-qubit
density matrix.  Which component is limiting, whether any two populations
coincide, and whether the first round is the binding one are OUTPUTS here,
not assumptions.  The success probability is recorded but never enters the
threshold definition.

Run:
    python scripts/verify_4q_qabs_circuit.py
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
OUT = ROOT / "results" / "4q_qabs"

# ---------------------------------------------------------------------------
# Canonical 4Q specification, transcribed verbatim:
#
#   ordering  |q1 q2 q3 q4>, q1 most significant; 0-based q1->0 ... q4->3
#   inputs    rho (x) rho, i.e. copy A on (q1,q2), copy B on (q3,q4)
#   gates     CNOT q3->q4, CNOT q2->q4, CNOT q1->q4, CNOT q3->q2, CNOT q3->q1
#             then H on q3
#   measure   q3 and q4 in the computational basis
#   success   outcome (m3, m4) = (0, 0)
#   retained  (q1, q2)
# ---------------------------------------------------------------------------

N_WIRES = 4
DIM = 2 ** N_WIRES
Q1, Q2, Q3, Q4 = 0, 1, 2, 3
RETAIN = (Q1, Q2)
MEASURED = (Q3, Q4)
SUCCESS_OUTCOME = (0, 0)

CNOT_SEQUENCE = [(Q3, Q4), (Q2, Q4), (Q1, Q4), (Q3, Q2), (Q3, Q1)]
CNOT_LABELS = [("q3", "q4"), ("q2", "q4"), ("q1", "q4"), ("q3", "q2"), ("q3", "q1")]
N_CNOT = len(CNOT_SEQUENCE)

CIRCUIT_SOURCE = ("pqec-bell-distillation/src/pqec_distill/circuit.py, "
                  "CNOT_SEQUENCE + module docstring, commit 69248179, sha256 "
                  "e475a54da16a3d1cb33b2e5f545fbbfcdbc06b6e43ebfc0aeb68a38029e03969")

# ===========================================================================
# 1. gate primitives (dense, 4 qubits, |q1 q2 q3 q4> with q1 most significant)
# ===========================================================================

_I2 = np.eye(2, dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_P0 = np.array([[1, 0], [0, 0]], dtype=complex)
_P1 = np.array([[0, 0], [0, 1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)


def _kron_list(mats):
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


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


def cnot_permutation(control: int, target: int) -> np.ndarray:
    """The same gate built a second, independent way: as the permutation of
    computational basis states that flips the target when the control is 1.
    Used only by the sanity check, so the check does not reuse ``cnot``."""
    perm = np.zeros(DIM, dtype=int)
    for idx in range(DIM):
        bits = [(idx >> (N_WIRES - 1 - w)) & 1 for w in range(N_WIRES)]
        if bits[control] == 1:
            bits[target] ^= 1
        perm[idx] = sum(b << (N_WIRES - 1 - w) for w, b in enumerate(bits))
    m = np.zeros((DIM, DIM), dtype=complex)
    for col, row in enumerate(perm):
        m[row, col] = 1.0
    return m


def five_cnot_unitary(builder=cnot) -> np.ndarray:
    u = np.eye(DIM, dtype=complex)
    for control, target in CNOT_SEQUENCE:
        u = builder(control, target) @ u
    return u


def full_unitary(builder=cnot) -> np.ndarray:
    """V = H_(q3) @ U_CNOT, the complete unitary before measurement."""
    return place_single(_H, Q3) @ five_cnot_unitary(builder)


def format_sequence() -> str:
    lines = []
    for k, ((c, t), (cl, tl)) in enumerate(zip(CNOT_SEQUENCE, CNOT_LABELS), 1):
        lines.append(f"  {k}.  CNOT({cl} -> {tl})   wires ({c} -> {t})"
                     f"      [CNOT #{k}]  + depolarizing D_q on ({c},{t})")
    lines.append(f"  {N_CNOT + 1}.  H(q3)   wire {Q3}      [ideal, no noise]")
    lines.append(f"  {N_CNOT + 2}.  measure (q3, q4) in the computational basis;"
                 f" success = {SUCCESS_OUTCOME}; retain (q1, q2)")
    return "\n".join(lines)


def build_program():
    """[('U', M) | ('D', (i,j))] with the channels exactly where the CNOTs are
    and the final Hadamard noiseless."""
    program = []
    for control, target in CNOT_SEQUENCE:
        program.append(("U", cnot(control, target)))
        program.append(("D", (control, target)))
    program.append(("U", place_single(_H, Q3)))
    return program


PROGRAM = build_program()

# ===========================================================================
# 2. the two-qubit replacement depolarizing channel, implemented directly
# ===========================================================================

_ROW = "abcd"
_COL = "efgh"


def replacement_depol(sigma: np.ndarray, pair, q: float) -> np.ndarray:
    """D_q^(ij)(sigma) = (1-q) sigma + q [ I_ij/4 (x) Tr_ij(sigma) ].

    Implemented from the definition on the FULL four-qubit density matrix:
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
# 3. one 4Q round: circuit, physical measurement, postselection
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


def project_success(sigma: np.ndarray, outcome=SUCCESS_OUTCOME):
    """Projective measurement of (q3, q4), keep the ``outcome`` branch, trace
    the measured wires out.  Returns (unnormalised 4x4 block, P_succ).

    For a rank-one projector on the measured wires, projecting and then tracing
    them out is exactly the corresponding diagonal sub-block, which is what is
    taken here; P_succ is that block's trace, as required.
    """
    m3, m4 = outcome
    t = sigma.reshape([2] * (2 * N_WIRES))
    block = t[:, :, m3, m4, :, :, m3, m4]          # (q1,q2 | q1',q2')
    rho_tilde = block.reshape(4, 4)
    return rho_tilde, float(np.real(np.trace(rho_tilde)))


def one_round(rho: np.ndarray, q: float, program=None, outcome=SUCCESS_OUTCOME):
    """rho_(n+1) = (projected, traced block) / P_succ.

    Physical postselection on the measurement outcome -- NOT a parity-weighted
    reconstruction.
    """
    sigma = np.kron(rho, rho)
    for kind, payload in (program or PROGRAM):
        if kind == "U":
            sigma = payload @ sigma @ payload.conj().T
        else:
            sigma = replacement_depol(sigma, payload, q)
    rho_tilde, p_succ = project_success(sigma, outcome)
    herm_residue = float(np.linalg.norm(rho_tilde - rho_tilde.conj().T))
    rho_tilde = 0.5 * (rho_tilde + rho_tilde.conj().T)   # exact-arithmetic identity
    if p_succ <= 0.0 or not np.isfinite(p_succ):
        raise FloatingPointError(f"success probability vanished: {p_succ!r}")
    return rho_tilde / p_succ, {"p_succ": p_succ, "herm_residue": herm_residue}


def one_round_direct(rho: np.ndarray, outcome=SUCCESS_OUTCOME):
    """Independent noiseless implementation (route A): build V once from the
    permutation-matrix CNOTs, apply it in a single conjugation, then project.
    Used only by the q = 0 sanity check."""
    v = full_unitary(cnot_permutation)
    sigma = v @ np.kron(rho, rho) @ v.conj().T
    rho_tilde, p_succ = project_success(sigma, outcome)
    return rho_tilde / p_succ, p_succ


# ===========================================================================
# 4. independent sanity checks
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
    q=1 limit: the chosen pair becomes I/4 and the other wires are untouched."""
    worst = {"trace": 0.0, "herm": 0.0, "min_eig": 0.0, "identity_at_q0": 0.0,
             "q1_pair_is_I_over_4": 0.0, "q1_rest_preserved": 0.0}
    pairs = list({tuple(p) for p in CNOT_SEQUENCE}) + [(0, 1), (1, 2)]
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


def check_noiseless_two_implementations(rng, n_trials=25) -> dict:
    """(C) at q = 0, the gate-by-gate route and the single-unitary route built
    from independently constructed CNOTs must agree, in both the conditional
    state and the success probability."""
    worst_state = worst_prob = worst_trace_dist = 0.0
    worst_cnot = float(np.max(np.abs(five_cnot_unitary(cnot)
                                     - five_cnot_unitary(cnot_permutation))))
    for _ in range(n_trials):
        rho = random_density_matrix(rng, 4)
        out_b, info = one_round(rho, 0.0)
        out_a, p_a = one_round_direct(rho)
        worst_state = max(worst_state, float(np.max(np.abs(out_a - out_b))))
        d = out_a - out_b
        worst_trace_dist = max(worst_trace_dist, 0.5 * float(
            np.sum(np.abs(np.linalg.eigvalsh(0.5 * (d + d.conj().T))))))
        worst_prob = max(worst_prob, abs(p_a - info["p_succ"]))
    return {"max_abs_state_diff": worst_state,
            "max_trace_distance": worst_trace_dist,
            "max_abs_P_succ_diff": worst_prob,
            "max_abs_unitary_diff_two_builders": worst_cnot,
            "n_random_trials": n_trials}


def check_branch_probabilities(rng, n_trials=15) -> dict:
    """(E) the four measurement branches must carry total probability 1 and
    each conditional state must be a valid density matrix."""
    worst_sum = worst_trace = 0.0
    worst_eig = np.inf
    for _ in range(n_trials):
        rho = random_density_matrix(rng, 4)
        for q in (0.0, 0.05, 0.2):
            sigma = np.kron(rho, rho)
            for kind, payload in PROGRAM:
                sigma = (payload @ sigma @ payload.conj().T) if kind == "U" \
                    else replacement_depol(sigma, payload, q)
            total = 0.0
            for m3 in (0, 1):
                for m4 in (0, 1):
                    tilde, p = project_success(sigma, (m3, m4))
                    total += p
                    if p > 1e-14:
                        r = tilde / p
                        worst_trace = max(worst_trace,
                                          abs(float(np.real(np.trace(r))) - 1.0))
                        worst_eig = min(worst_eig, float(
                            np.linalg.eigvalsh(0.5 * (r + r.conj().T)).min()))
            worst_sum = max(worst_sum, abs(total - 1.0))
    return {"max_abs_branch_probability_sum_minus_1": worst_sum,
            "max_abs_conditional_trace_minus_1": worst_trace,
            "min_conditional_eigenvalue": float(worst_eig),
            "n_random_trials": n_trials}


def check_state_validity(eps, q, n_rounds=40) -> dict:
    """(A) hermiticity, trace, positivity and P_succ along an actual trajectory."""
    rho = isotropic_rho(eps)
    worst = {"herm": 0.0, "trace": 0.0, "min_eig": np.inf,
             "pop_sum_minus_1": 0.0, "tilde_herm_residue": 0.0,
             "min_P_succ": np.inf}
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
            worst["tilde_herm_residue"] = max(worst["tilde_herm_residue"],
                                              info["herm_residue"])
            worst["min_P_succ"] = min(worst["min_P_succ"], info["p_succ"])
    return {k: float(v) for k, v in worst.items()}


# ===========================================================================
# 5. the absolute-suppression predicate
# ===========================================================================
#
# Tolerances.  The round-off floor is MEASURED by `measure_error_floor()` at
# run time and printed with the results; it is not assumed here.
#
#   STEP_TOL  = 1e-14   stop once max|rho_(n+1) - rho_n| falls below this: the
#                       orbit is then at the numerical fixed point and any
#                       further sign change is round-off.
#   VIOL_TOL  = 1e-12   a round counts as violating only if some non-target
#                       Delta p_i exceeds +VIOL_TOL.
#   P_SUCC_FLOOR        below this the conditional state is dominated by
#                       round-off in a vanishing branch, so the trajectory
#                       stops being trusted.
#
# OFFDIAG_TRUST is the round-off contamination guard: the Bell-isotropic input
# is built exactly Bell-diagonal, so its Bell off-diagonal weight starts at
# round-off; if the map amplifies that direction, accumulated round-off can
# manufacture a population increase the exact dynamics does not have.  Whether
# this 4Q map amplifies or contracts it is MEASURED separately by
# `transverse_growth()`; it is not assumed here either way.  At C_off = 1e-18
# the induced population error is ~1e-18, six orders below VIOL_TOL, so the
# guard cannot hide a genuine violation.

STEP_TOL = 1e-14
VIOL_TOL = 1e-12
N_MAX = 5000
OFFDIAG_TRUST = 1e-18
P_SUCC_FLOOR = 1e-12


def run_trajectory(eps: float, q: float, step_tol=STEP_TOL, n_max=N_MAX,
                   viol_tol=VIOL_TOL, components=NON_TARGET, record=False,
                   offdiag_trust=OFFDIAG_TRUST, p_succ_floor=P_SUCC_FLOOR):
    """Iterate the real postselected circuit and look for the first absolute-
    suppression violation.  Every round's exact conditional output state is the
    next round's input.

    ``violated`` is True only when some tracked non-target population rose by
    more than ``viol_tol`` while the trajectory was still trustworthy.
    """
    idx = [BELL_ORDER.index(c) for c in components]
    rho = isotropic_rho(eps)
    p = bell_populations(rho)
    c_off0 = bell_offdiagonal_weight(rho)
    rows = [{"n": 0, "p": p.copy(), "dp": np.full(4, np.nan), "step": np.nan,
             "c_off": c_off0, "p_succ": np.nan}] if record else []
    worst_dp, worst_at, worst_comp = -np.inf, None, None
    c_off_max, p_succ_min, p_succ_last = c_off0, np.inf, np.nan
    n, step, trusted, n_trusted, converged = 0, np.nan, True, None, False
    while n < n_max:
        rho_next, info = one_round(rho, q)
        p_next = bell_populations(rho_next)
        dp = p_next - p
        step = float(np.max(np.abs(rho_next - rho)))
        c_off = bell_offdiagonal_weight(rho_next)
        c_off_max = max(c_off_max, c_off)
        p_succ_last = info["p_succ"]
        p_succ_min = min(p_succ_min, p_succ_last)
        if record:
            rows.append({"n": n + 1, "p": p_next.copy(), "dp": dp.copy(),
                         "step": step, "c_off": c_off, "p_succ": p_succ_last})
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
                    "p_succ_at_violation": p_succ_last,
                    "rounds_run": n + 1, "final_step": step, "trusted": True,
                    "n_trusted": None, "converged": False,
                    "c_off_initial": c_off0, "c_off_final": c_off,
                    "c_off_max": c_off_max, "p_succ_min": p_succ_min,
                    "p_succ_final": p_succ_last,
                    "worst_delta_p": worst_dp, "worst_round": worst_at,
                    "worst_component": worst_comp, "rows": rows}
        rho, p = rho_next, p_next
        n += 1
        if step < step_tol:
            converged = True
            break
        if c_off > offdiag_trust or p_succ_last < p_succ_floor:
            trusted, n_trusted = False, n
            break
    return {"violated": False, "round": None, "component": None,
            "rounds_run": n, "final_step": step, "trusted": trusted,
            "n_trusted": n_trusted, "converged": converged,
            "c_off_initial": c_off0, "c_off_final": bell_offdiagonal_weight(rho),
            "c_off_max": c_off_max, "p_succ_min": p_succ_min,
            "p_succ_final": p_succ_last,
            "worst_delta_p": worst_dp, "worst_round": worst_at,
            "worst_component": worst_comp, "p_final": p.copy(), "rows": rows}


def record_trajectory(eps: float, q: float, n_rounds: int,
                      viol_tol=VIOL_TOL, components=NON_TARGET):
    """Round-by-round record for plotting; unlike run_trajectory this does not
    stop at the first violation."""
    idx = [BELL_ORDER.index(c) for c in components]
    rho = isotropic_rho(eps)
    p = bell_populations(rho)
    rows = [{"n": 0, "p": p.copy(), "dp": np.full(4, np.nan),
             "c_off": bell_offdiagonal_weight(rho), "p_succ": np.nan}]
    first_violation = None
    for n in range(n_rounds):
        rho, info = one_round(rho, q)
        p_next = bell_populations(rho)
        dp = p_next - p
        if first_violation is None and any(dp[i] > viol_tol for i in idx):
            k = max((i for i in idx if dp[i] > viol_tol), key=lambda i: dp[i])
            first_violation = {"round": n, "component": BELL_ORDER[k],
                               "delta_p": float(dp[k]),
                               "p_succ": info["p_succ"]}
        rows.append({"n": n + 1, "p": p_next.copy(), "dp": dp.copy(),
                     "c_off": bell_offdiagonal_weight(rho),
                     "p_succ": info["p_succ"]})
        p = p_next
    return rows, first_violation


def first_round_step(eps: float, q: float):
    """rho_0 -> rho_1 only; returns (p0, p1, dp, P_succ)."""
    rho = isotropic_rho(eps)
    p0 = bell_populations(rho)
    rho1, info = one_round(rho, q)
    p1 = bell_populations(rho1)
    return p0, p1, p1 - p0, info["p_succ"]


def first_round_violates(eps: float, q: float, viol_tol=VIOL_TOL,
                         components=NON_TARGET) -> bool:
    _p0, _p1, dp, _ps = first_round_step(eps, q)
    return any(dp[BELL_ORDER.index(c)] > viol_tol for c in components)


def measure_error_floor(eps=0.15, q=0.02, n_settle=60, n_watch=40) -> dict:
    """Empirical round-off floor once the orbit has settled."""
    rho = isotropic_rho(eps)
    for _ in range(n_settle):
        rho, _ = one_round(rho, q)
    worst_dp = worst_step = worst_ps = 0.0
    p = bell_populations(rho)
    _r, info = one_round(rho, q)
    ps_ref = info["p_succ"]
    for _ in range(n_watch):
        rho_next, info = one_round(rho, q)
        p_next = bell_populations(rho_next)
        worst_dp = max(worst_dp, float(np.max(np.abs(p_next - p))))
        worst_step = max(worst_step, float(np.max(np.abs(rho_next - rho))))
        worst_ps = max(worst_ps, abs(info["p_succ"] - ps_ref))
        rho, p = rho_next, p_next
    return {"max_abs_delta_p_at_fixed_point": worst_dp,
            "max_abs_step_at_fixed_point": worst_step,
            "max_abs_P_succ_drift_at_fixed_point": worst_ps,
            "eps": eps, "q": q}


# ===========================================================================
# 6. threshold search -- bracket, then bisect, on the circuit predicate only
# ===========================================================================

def bracket(predicate, q_lo=0.0, q_hi=0.5, n_coarse=64):
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
    """Look near ``hint`` first, then fall back to the full scan.  The bracket
    is re-verified before bisecting, so a hint can never change the answer."""
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
            "p_succ_at_violation": detail.get("p_succ_at_violation"),
            "all_violating_at_q_upper": detail.get("all_violating"),
            # NOTE: these populations belong to the q_upper run, not to q_abs.
            "p_n_at_q_upper": detail.get("p_n"),
            "p_np1_at_q_upper": detail.get("p_np1"),
            "below_converged": at_lo["converged"],
            "below_trusted": at_lo["trusted"],
            "below_n_trusted": at_lo["n_trusted"],
            "below_rounds_run": at_lo["rounds_run"],
            "below_c_off_max": at_lo["c_off_max"],
            "below_p_succ_min": at_lo["p_succ_min"],
            "below_p_succ_final": at_lo["p_succ_final"],
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
    _p0, _p1, dp, ps = first_round_step(eps, b)
    hits = [BELL_ORDER[i] for i in (1, 2, 3) if dp[i] > viol_tol]
    return {"q_abs": a, "q_upper": b, "iterations": it, "bracket_width": b - a,
            "bracket_info": info, "p_succ_at_q_upper": ps,
            "limiting_component": (max(hits, key=lambda c: dp[BELL_ORDER.index(c)])
                                   if hits else None),
            "all_violating_components": hits}


def per_component_thresholds(eps: float, viol_tol=VIOL_TOL, tol=1e-10) -> dict:
    """Threshold for each non-target component on its own, for the first round
    alone and over all rounds.  The limiting component is the one with the
    smallest value -- an output, not an input."""
    first, full, detail = {}, {}, {}
    for c in NON_TARGET:
        first[c] = q_abs_first_round(eps, viol_tol=viol_tol, tol=tol,
                                     components=(c,))["q_abs"]
        r = q_abs_full(eps, viol_tol=viol_tol, tol=tol, components=(c,))
        full[c] = r["q_abs"]
        detail[c] = {"round": r.get("first_violation_round"),
                     "delta_p": r.get("delta_p_at_violation")}
    return {"first_round": first, "all_rounds": full, "all_round_detail": detail}


def tolerance_study(eps_values, tols=(1e-10, 1e-12, 1e-14), bisect_tol=1e-11) -> dict:
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
    """Section 17: check both sides of the located threshold at several scales."""
    rows = []
    for d in deltas:
        for side, q in (("below", q_star - d), ("above", q_star + d)):
            if q < 0:
                continue
            r = run_trajectory(eps, float(q))
            rows.append({
                "epsilon": eps, "delta": d, "side": side, "q": float(q),
                "violated": r["violated"], "violation_round": r["round"],
                "violating_component": r["component"],
                "delta_p_violating": r.get("delta_p_violating"),
                "P_succ_at_violation": r.get("p_succ_at_violation"),
                "worst_delta_p": r["worst_delta_p"],
                "worst_component": r["worst_component"],
                "worst_round": r["worst_round"],
                "rounds_run": r["rounds_run"], "converged": r["converged"],
                "trusted": r["trusted"], "n_trusted": r["n_trusted"],
                "c_off_max": r["c_off_max"], "P_succ_min": r["p_succ_min"],
                "P_succ_final": r["p_succ_final"]})
    return rows


# ===========================================================================
# 7. transverse stability of the Bell-diagonal manifold (separate diagnostic)
# ===========================================================================

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
    measured through the full postselected round."""
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
            # the fitted rate is meaningless once the amplitude has fallen to
            # round-off: the ratio then measures noise, not the map
            "amplitude_at_floor": bool(a[hi] < 1e-15),
            "seeded_state_min_eig": eig0}


BELL_PAIRS = (("Phi+", "Phi-"), ("Phi+", "Psi+"), ("Phi+", "Psi-"),
              ("Phi-", "Psi+"), ("Phi-", "Psi-"), ("Psi+", "Psi-"))


def transverse_one_round(eps: float, q: float, seed: float, pair) -> dict:
    """One round applied to a seeded state: how much off-diagonal amplitude
    survives, and how that survival scales with the seed.

    If the round's linearisation annihilates the transverse direction, the
    surviving amplitude is second order, so a1 / a0^2 is constant in the seed
    while a1 / a0 is proportional to it.  Both ratios are reported and the
    conclusion is read off the measurement, not assumed.
    """
    rho = seeded_rho(eps, seed, pair=pair)
    a0 = float(np.sqrt(bell_offdiagonal_weight(rho)))
    rho1, info = one_round(rho, q)
    a1 = float(np.sqrt(bell_offdiagonal_weight(rho1)))
    return {"epsilon": eps, "q": q, "seed": seed,
            "pair": f"{pair[0]}/{pair[1]}", "a0": a0, "a1": a1,
            "ratio_a1_over_a0": (a1 / a0) if a0 > 0 else float("nan"),
            "ratio_a1_over_a0_squared": (a1 / a0 ** 2) if a0 > 0 else float("nan"),
            "P_succ": info["p_succ"],
            "at_round_off": bool(a1 < 1e-15)}


def transverse_order_test(eps_q_pairs, seeds=(1e-3, 1e-4, 1e-5, 1e-6),
                          pairs=BELL_PAIRS) -> list:
    rows = []
    for eps, q in eps_q_pairs:
        for pair in pairs:
            for seed in seeds:
                rows.append(transverse_one_round(eps, q, seed, pair))
    return rows


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
            linestyle="none", marker="o", markersize=6, markerfacecolor="white",
            markeredgecolor=BELL_COLOR["Phi+"], markeredgewidth=1.6,
            label="high-precision points")
    ax.set_xlabel(r"input mixing $\epsilon$")
    ax.set_ylabel(r"$q_{\mathrm{abs}}^{(4\mathrm{Q})}(\epsilon)$")
    ax.set_xlim(0, max(r["epsilon"] for r in dense))
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", labelcolor=INK)
    ax.set_title("4Q: absolute Bell-error-suppression threshold\n"
                 "(from the 5-CNOT postselected circuit simulation)",
                 color=INK, pad=8)
    return save(fig, "4q_qabs_curve")


def plot_first_vs_full(dense):
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
    axes[0].set_ylabel(r"$q_{\mathrm{abs}}^{(4\mathrm{Q})}$")
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
    fig.suptitle("4Q: all-round versus first-round threshold",
                 fontsize=13, color=INK, y=1.02)
    return save(fig, "4q_qabs_first_vs_full")


def plot_components(table):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7),
                             gridspec_kw=dict(wspace=0.30))
    for ax in axes:
        _despine(ax)
    e = [r["epsilon"] for r in table]
    first_keys = {"Phi-": "q_Phi_minus_first", "Psi+": "q_Psi_plus_first",
                  "Psi-": "q_Psi_minus_first"}
    all_keys = {"Phi-": "q_Phi_minus_all", "Psi+": "q_Psi_plus_all",
                "Psi-": "q_Psi_minus_all"}
    for ax, keys, title in ((axes[0], first_keys, "first-round definition"),
                            (axes[1], all_keys, "standalone all-round definition")):
        for name, key in keys.items():
            style = dict(linewidth=1.4, linestyle=(0, (5, 3))) if name == "Psi-" \
                else dict(linewidth=1.9)
            ax.plot(e, [r[key] for r in table], color=BELL_COLOR[name],
                    marker=BELL_MARKER[name], markeredgecolor="white",
                    label=BELL_TEX[name], **style)
        ax.set_xlabel(r"input mixing $\epsilon$")
        ax.set_ylabel("per-component threshold")
        ax.set_ylim(0, None)
        ax.legend(loc="upper left", labelcolor=INK)
        ax.set_title(title, color=INK, fontsize=11)
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.text(-0.15, 1.05, tag, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=INK)
    fig.suptitle("4Q: which Bell component sets the threshold",
                 fontsize=13, color=INK, y=1.02)
    return save(fig, "4q_qabs_components")


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
                         rf"$\Delta p = {viol['delta_p']:+.2e}$"
                         f"\n$P_{{\\rm succ}} = {viol['p_succ']:.4f}$",
                         xy=(0.04, 0.80), xycoords="axes fraction",
                         fontsize=9, color=INK, va="top")
        else:
            bot.annotate("no violation:\nall three decrease\nat every round",
                         xy=(0.50, 0.66), xycoords="axes fraction",
                         fontsize=9, color=INK, va="top")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.03), labelcolor=INK)
    fig.suptitle(rf"4Q at $\epsilon = {eps:g}$: below and above "
                 rf"$q_{{\mathrm{{abs}}}} = {q_star:.10f}$",
                 fontsize=13, color=INK, y=0.985)
    return save(fig, "4q_qabs_eps015_trajectories")


def plot_success(cases, eps, q_star, succ_vs_eps):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7),
                             gridspec_kw=dict(wspace=0.30))
    for ax in axes:
        _despine(ax)
    palette = [BELL_COLOR["Phi+"], BELL_COLOR["Psi+"], BELL_COLOR["Phi-"]]
    for k, (label, q, rows, _viol) in enumerate(cases):
        n = [r["n"] for r in rows[1:]]
        ps = [r["p_succ"] for r in rows[1:]]
        axes[0].plot(n, ps, color=palette[k % len(palette)], marker="o",
                     markevery=(k, 3), markeredgecolor="white", linewidth=1.8,
                     label=f"{label}  ($q = q_{{\\rm abs}} {q - q_star:+.0e}$)")
    axes[0].set_xlabel(r"purification round $n$")
    axes[0].set_ylabel(r"$P_{\rm succ}^{(n)}$")
    axes[0].legend(loc="lower right", labelcolor=INK, fontsize=9)
    axes[0].set_title(rf"$\epsilon = {eps:g}$, round by round",
                      color=INK, fontsize=11)

    e = [r["epsilon"] for r in succ_vs_eps]
    axes[1].plot(e, [r["P_succ_round1"] for r in succ_vs_eps],
                 color=BELL_COLOR["Phi+"], marker="o", markeredgecolor="white",
                 linewidth=1.9, label=r"$P_{\rm succ}^{(1)}$")
    axes[1].plot(e, [r["P_succ_converged"] for r in succ_vs_eps],
                 color=BELL_COLOR["Phi-"], marker="s", markeredgecolor="white",
                 linewidth=1.9, linestyle=(0, (5, 3)),
                 label=r"$P_{\rm succ}$ at the end of the trusted run")
    axes[1].set_xlabel(r"input mixing $\epsilon$")
    axes[1].set_ylabel(r"$P_{\rm succ}$")
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="lower left", labelcolor=INK, fontsize=9)
    axes[1].set_title(r"at $q = q_{\rm abs}(\epsilon) - 10^{-6}$",
                      color=INK, fontsize=11)
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.text(-0.15, 1.05, tag, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=INK)
    fig.suptitle("4Q: postselection success probability (recorded, not part of "
                 "the threshold definition)", fontsize=13, color=INK, y=1.02)
    return save(fig, "4q_success_eps015")


def plot_transverse(at_threshold, in_q, order):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7),
                             gridspec_kw=dict(wspace=0.30))
    for ax in axes:
        _despine(ax)
    # (a) surviving amplitude versus seed amplitude, for the three (eps, q)
    keys = sorted({(r["epsilon"], r["q"]) for r in order})
    palette = [BELL_COLOR["Phi+"], BELL_COLOR["Psi+"], BELL_COLOR["Phi-"]]
    for k, (ev, qv) in enumerate(keys):
        sub = sorted((r for r in order
                      if r["epsilon"] == ev and r["q"] == qv
                      and r["pair"] == "Phi+/Phi-" and not r["at_round_off"]),
                     key=lambda r: r["a0"])
        axes[0].plot([r["a0"] for r in sub], [r["a1"] for r in sub],
                     color=palette[k % len(palette)], marker="o",
                     markeredgecolor="white", linewidth=1.8,
                     label=rf"$\epsilon = {ev:g}$, $q = {qv:.4g}$")
    lo = min(r["a0"] for r in order if not r["at_round_off"])
    hi = max(r["a0"] for r in order if not r["at_round_off"])
    ref = np.array([lo, hi])
    axes[0].plot(ref, ref, color=INK2, linestyle=(0, (4, 3)), linewidth=1.0,
                 label=r"slope 1 (first order)")
    axes[0].plot(ref, ref ** 2, color=INK2, linestyle=(0, (1, 2)), linewidth=1.3,
                 label=r"slope 2 (second order)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"seeded off-diagonal amplitude  $a_0$")
    axes[0].set_ylabel(r"amplitude after one round  $a_1$")
    axes[0].legend(loc="upper left", labelcolor=INK, fontsize=8.5)
    axes[0].set_title("the surviving amplitude is second order in the seed",
                      color=INK, fontsize=11)

    # (b) the quadratic coefficient across epsilon
    axes[1].plot([r["epsilon"] for r in at_threshold],
                 [r["quadratic_coefficient_a1_over_a0_squared"]
                  for r in at_threshold],
                 color=BELL_COLOR["Phi+"], marker="o", markeredgecolor="white",
                 linewidth=1.9, label=r"at $q = q_{\rm abs}(\epsilon) - 10^{-6}$")
    axes[1].set_xlabel(r"input mixing $\epsilon$")
    axes[1].set_ylabel(r"$a_1 / a_0^{2}$")
    axes[1].legend(loc="lower right", labelcolor=INK, fontsize=9)
    axes[1].set_title("the second-order coefficient", color=INK, fontsize=11)
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.text(-0.15, 1.05, tag, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=INK)
    fig.suptitle("4Q: a Bell off-diagonal perturbation is annihilated to first "
                 "order by one postselected round", fontsize=13, color=INK, y=1.02)
    return save(fig, "4q_transverse_growth")


# ===========================================================================
# 9. driver
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
    print("4Q CIRCUIT: exact specification actually used")
    print("=" * 78)
    print(f"  source   : {CIRCUIT_SOURCE}")
    print("  ordering : |q1 q2 q3 q4>, q1 most significant; q1->0 ... q4->3")
    print("  inputs   : rho_n (x) rho_n   -- copy A on (q1,q2), copy B on (q3,q4)")
    print("  retained : (q1, q2)          measured: (q3, q4)")
    print(f"  success  : (m3, m4) = {SUCCESS_OUTCOME}")
    print(f"  CNOT count = {N_CNOT}   (expected 5)\n")
    print(format_sequence())

    print("\n" + "=" * 78)
    print("SANITY CHECKS")
    print("=" * 78)
    rng = np.random.default_rng(20261004)
    checks = {
        "C_noiseless_two_independent_implementations":
            check_noiseless_two_implementations(rng, 25),
        "B_noise_channel": check_noise_channel(rng, 10),
        "E_measurement_branches": check_branch_probabilities(rng, 15),
        "A_state_validity_eps0.15_q0.02": check_state_validity(0.15, 0.02, 40),
        "error_floor": measure_error_floor(),
        "tolerances": {"STEP_TOL": STEP_TOL, "VIOL_TOL": VIOL_TOL,
                       "N_MAX": N_MAX, "OFFDIAG_TRUST": OFFDIAG_TRUST,
                       "P_SUCC_FLOOR": P_SUCC_FLOOR,
                       "bisect_tol": args.bisect_tol},
    }
    for group, d in checks.items():
        print(f"  {group}")
        for k, v in d.items():
            print(f"      {k:<46} {v:.4e}" if isinstance(v, float)
                  else f"      {k:<46} {v}")

    print("\n" + "=" * 78)
    print("THRESHOLD SEARCH  (predicate = actual postselected circuit simulation)")
    print("=" * 78)
    table = []
    for eps in EPS_TABLE:
        t0 = time.time()
        full = q_abs_full(eps, tol=args.bisect_tol)
        first = q_abs_first_round(eps, tol=args.bisect_tol)
        comps = per_component_thresholds(eps, tol=args.bisect_tol)
        fr, fu, fd = comps["first_round"], comps["all_rounds"], comps["all_round_detail"]
        smallest = min((v for v in fr.values() if v is not None), default=None)
        limiting_first = next((c for c, v in fr.items() if v == smallest), None)
        row = {
            "epsilon": eps,
            "q_abs_full": full["q_abs"],
            "q_abs_first_round": first["q_abs"],
            "q_abs_full_minus_first": (None if full["q_abs"] is None
                                       or first["q_abs"] is None
                                       else full["q_abs"] - first["q_abs"]),
            "q_Phi_minus_first": fr["Phi-"], "q_Psi_plus_first": fr["Psi+"],
            "q_Psi_minus_first": fr["Psi-"],
            "q_Phi_minus_all": fu["Phi-"], "q_Psi_plus_all": fu["Psi+"],
            "q_Psi_minus_all": fu["Psi-"],
            "all_round_binding_round_Phi_minus": fd["Phi-"]["round"],
            "all_round_binding_round_Psi_plus": fd["Psi+"]["round"],
            "all_round_binding_round_Psi_minus": fd["Psi-"]["round"],
            "limiting_Bell_component": limiting_first,
            "limiting_at_q_upper": full["limiting_component"],
            "first_violation_round_near_threshold": full["first_violation_round"],
            "threshold_bracket_width": full.get("bracket_width"),
            "q_upper_violating": full.get("q_upper"),
            "Delta_p_at_q_upper": full.get("delta_p_at_violation"),
            "P_succ_at_q_upper": full.get("p_succ_at_violation"),
            "max_off_Bell_weight_before_convergence": full.get("below_c_off_max"),
            "below_threshold_converged": full.get("below_converged"),
            "below_threshold_trusted": full.get("below_trusted"),
            "below_threshold_n_trusted": full.get("below_n_trusted"),
            "below_threshold_rounds_run": full.get("below_rounds_run"),
            "below_P_succ_min": full.get("below_p_succ_min"),
            "below_P_succ_final": full.get("below_p_succ_final"),
            "below_worst_delta_p": full.get("below_worst_delta_p"),
            "below_worst_component": full.get("below_worst_component"),
            "below_worst_round": full.get("below_worst_round"),
            "bracket_monotone": full["bracket_info"].get("monotone"),
            "bracket_sign_changes": full["bracket_info"].get("n_sign_changes"),
            # NOTE: p_n / p_np1 are evaluated at q_upper_violating, NOT at q_abs.
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

    print("\n  per-component FIRST-ROUND thresholds (smallest = limiting)")
    for r in table:
        print(f"    eps={r['epsilon']:<5} Phi-={_fmt(r['q_Phi_minus_first'])}  "
              f"Psi+={_fmt(r['q_Psi_plus_first'])}  "
              f"Psi-={_fmt(r['q_Psi_minus_first'])}")

    print("\n  per-component STANDALONE ALL-ROUND thresholds (binding round)")
    for r in table:
        print(f"    eps={r['epsilon']:<5} "
              f"Phi-={_fmt(r['q_Phi_minus_all'])} (n={r['all_round_binding_round_Phi_minus']})  "
              f"Psi+={_fmt(r['q_Psi_plus_all'])} (n={r['all_round_binding_round_Psi_plus']})  "
              f"Psi-={_fmt(r['q_Psi_minus_all'])} (n={r['all_round_binding_round_Psi_minus']})")

    print("\n  violation-tolerance sensitivity (VIOL_TOL 1e-10 / 1e-12 / 1e-14)")
    tol_study = tolerance_study((0.05, 0.15, 0.40, 0.65), bisect_tol=args.bisect_tol)
    for eps, d in tol_study.items():
        vals = "  ".join(f"{k}:{_fmt(v)}" for k, v in d["values"].items())
        print(f"    eps={eps:<5} {vals}   spread={_fmt(d['spread'], '.2e')}")

    # ---- near-threshold scan -------------------------------------------
    print("\n  near-threshold check at delta = 1e-7, 1e-6, 1e-4, 1e-3")
    near = []
    for r in table:
        if r["q_abs_full"] is None:
            continue
        near += near_threshold_scan(r["epsilon"], r["q_abs_full"])
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

    # ---- eps = 0.15 trajectories + success probability ------------------
    eps_d = 0.15
    q_star = next(r["q_abs_full"] for r in table if r["epsilon"] == eps_d)
    cases, succ_rows = [], []
    for label, q in (("well below threshold", q_star - 5e-3),
                     ("just below threshold", q_star - 1e-7),
                     ("just above threshold", q_star + 1e-3)):
        rows, viol = record_trajectory(eps_d, q, 20)
        cases.append((label, q, rows, viol))
        res = run_trajectory(eps_d, q)
        v = (f"violation: {res['component']} at n={res['round']}, "
             f"Delta p={res['delta_p_violating']:+.4e}, "
             f"P_succ={res['p_succ_at_violation']:.6f}") if res["violated"] \
            else (f"no violation; worst Delta p = {res['worst_delta_p']:+.3e} "
                  f"({res['worst_component']} at n={res['worst_round']}), "
                  f"converged={res['converged']}, trusted={res['trusted']}, "
                  f"P_succ_final={res['p_succ_final']:.6f}")
        print(f"    eps=0.15, q={q:.12f}  [{label}]  {v}")
        with open(OUT / f"4q_qabs_eps015_q{q:.8f}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["round", "p_phi_plus", "p_phi_minus", "p_psi_plus",
                        "p_psi_minus", "d_phi_minus", "d_psi_plus", "d_psi_minus",
                        "C_offdiag", "P_succ"])
            for r in rows:
                w.writerow([r["n"], *r["p"], r["dp"][1], r["dp"][2], r["dp"][3],
                            r["c_off"], r["p_succ"]])
        for r in rows[1:]:
            succ_rows.append({"epsilon": eps_d, "case": label, "q": q,
                              "round": r["n"], "P_succ": r["p_succ"]})

    # P_succ versus epsilon, just below each threshold
    succ_vs_eps = []
    for r in table:
        if r["q_abs_full"] is None:
            continue
        q = max(r["q_abs_full"] - 1e-6, 0.0)
        res = run_trajectory(r["epsilon"], q, record=True)
        ps = [x["p_succ"] for x in res["rows"][1:]]
        succ_vs_eps.append({"epsilon": r["epsilon"], "q": q,
                            "P_succ_round1": ps[0] if ps else float("nan"),
                            "P_succ_converged": ps[-1] if ps else float("nan"),
                            "P_succ_min": res["p_succ_min"],
                            "rounds_run": res["rounds_run"],
                            "converged": res["converged"],
                            "trusted": res["trusted"]})
    print("\n  P_succ at q = q_abs - 1e-6")
    for r in succ_vs_eps:
        print(f"    eps={r['epsilon']:<5} P_succ^(1)={r['P_succ_round1']:.6f}  "
              f"final={r['P_succ_converged']:.6f}  min={r['P_succ_min']:.6f}  "
              f"rounds={r['rounds_run']} conv={r['converged']} "
              f"trusted={r['trusted']}")

    # ---- transverse stability (separate diagnostic) ---------------------
    print("\n" + "=" * 78)
    print("TRANSVERSE STABILITY OF THE BELL-DIAGONAL MANIFOLD")
    print("  (separate diagnostic; does not enter the q_abs definition)")
    print("=" * 78)
    at_thr = []
    for r in table:
        if r["q_abs_full"] is None:
            continue
        q = max(r["q_abs_full"] - 1e-6, 0.0)
        g = transverse_growth(r["epsilon"], q)
        one = transverse_one_round(r["epsilon"], q, 1e-6, BELL_PAIRS[0])
        at_thr.append({"epsilon": r["epsilon"], "q_abs": r["q_abs_full"], "q": q,
                       "one_round_ratio_a1_over_a0": one["ratio_a1_over_a0"],
                       "quadratic_coefficient_a1_over_a0_squared":
                           one["ratio_a1_over_a0_squared"],
                       "a0": one["a0"], "a1": one["a1"],
                       "asymptotic_growth_rate": g["growth_rate"],
                       "asymptotic_rate_is_meaningless_amplitude_at_floor":
                           g["amplitude_at_floor"],
                       "a_final_after_60_rounds": g["a_final"],
                       "seeded_state_min_eig": g["seeded_state_min_eig"]})
        print(f"    eps={r['epsilon']:<5} q={q:.12f}  "
              f"a1/a0={one['ratio_a1_over_a0']:.3e}  "
              f"a1/a0^2={one['ratio_a1_over_a0_squared']:.4f}  "
              f"a after 60 rounds: {g['a_final']:.2e}"
              + ("   [at round-off]" if g["amplitude_at_floor"] else ""))

    print("\n    order test: is the surviving amplitude first or second order "
          "in the seed?")
    order = transverse_order_test([(0.15, 0.02), (0.15, 0.060664417271),
                                   (0.50, 0.129399238959)])
    for eps, q in sorted({(r["epsilon"], r["q"]) for r in order}):
        sub = [r for r in order if r["epsilon"] == eps and r["q"] == q
               and r["pair"] == "Phi+/Phi-" ]
        line = "  ".join(f"seed={r['seed']:.0e}: a1/a0={r['ratio_a1_over_a0']:.2e}"
                         for r in sub)
        coef = {round(r["ratio_a1_over_a0_squared"], 4) for r in sub
                if not r["at_round_off"]}
        print(f"      eps={eps:<5} q={q:.9f}  {line}")
        print(f"        a1/a0^2 over those seeds: {sorted(coef)}")
    pair_coef = {}
    for r in order:
        if r["epsilon"] == 0.15 and r["q"] == 0.02 and not r["at_round_off"]:
            pair_coef.setdefault(r["pair"], set()).add(
                round(r["ratio_a1_over_a0_squared"], 4))
    print("      per-pair a1/a0^2 at eps=0.15, q=0.02:")
    for k, v in pair_coef.items():
        print(f"        {k:<12} {sorted(v)}")

    in_q = []
    print("\n    one-round ratio as a function of q at fixed eps (seed 1e-6)")
    for ev in (0.15, 0.50):
        for q in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
            one = transverse_one_round(ev, q, 1e-6, BELL_PAIRS[0])
            in_q.append({"epsilon": ev, "q": q,
                         "ratio_a1_over_a0": one["ratio_a1_over_a0"],
                         "quadratic_coefficient": one["ratio_a1_over_a0_squared"],
                         "at_round_off": one["at_round_off"]})
            print(f"      eps={ev:<5} q={q:<6g} a1/a0={one['ratio_a1_over_a0']:.3e}"
                  f"  a1/a0^2={one['ratio_a1_over_a0_squared']:.4f}")

    # ---- outputs --------------------------------------------------------
    main_cols = ["epsilon", "q_abs_full", "q_abs_first_round",
                 "q_abs_full_minus_first", "q_Phi_minus_first",
                 "q_Psi_plus_first", "q_Psi_minus_first",
                 "limiting_Bell_component",
                 "first_violation_round_near_threshold",
                 "threshold_bracket_width", "q_upper_violating",
                 "Delta_p_at_q_upper", "P_succ_at_q_upper", "seconds"]
    comp_cols = ["epsilon", "q_Phi_minus_first", "q_Psi_plus_first",
                 "q_Psi_minus_first", "q_Phi_minus_all", "q_Psi_plus_all",
                 "q_Psi_minus_all", "all_round_binding_round_Phi_minus",
                 "all_round_binding_round_Psi_plus",
                 "all_round_binding_round_Psi_minus"]
    diag_cols = ["epsilon", "max_off_Bell_weight_before_convergence",
                 "below_threshold_converged", "below_threshold_trusted",
                 "below_threshold_n_trusted", "below_threshold_rounds_run",
                 "below_P_succ_min", "below_P_succ_final",
                 "below_worst_delta_p", "below_worst_component",
                 "below_worst_round", "limiting_at_q_upper",
                 "bracket_monotone", "bracket_sign_changes"]
    for name, cols in (("4q_qabs_circuit.csv", main_cols),
                       ("4q_component_thresholds.csv", comp_cols),
                       ("4q_qabs_diagnostics.csv", diag_cols)):
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(table)
    for name, rowset in (("4q_qabs_dense.csv", dense),
                         ("4q_qabs_near_threshold.csv", near),
                         ("4q_success_probability.csv", succ_rows),
                         ("4q_success_vs_epsilon.csv", succ_vs_eps),
                         ("4q_transverse_growth.csv", at_thr),
                         ("4q_transverse_growth_vs_q.csv", in_q),
                         ("4q_transverse_order_test.csv", order)):
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rowset[0]))
            w.writeheader()
            w.writerows(rowset)

    meta = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "circuit_source": CIRCUIT_SOURCE,
        "n_cnot": N_CNOT,
        "qubit_order": "|q1 q2 q3 q4>, q1 most significant; q1->0 ... q4->3",
        "inputs": "rho_n (x) rho_n; copy A on (q1,q2), copy B on (q3,q4)",
        "cnot_sequence": [list(p) for p in CNOT_SEQUENCE],
        "cnot_labels": [list(p) for p in CNOT_LABELS],
        "final_single_qubit_gate": "H on q3 (ideal)",
        "measured": list(MEASURED), "success_outcome": list(SUCCESS_OUTCOME),
        "retained_pair": list(RETAIN),
        "noise_model": "two-qubit replacement depolarizing after every CNOT",
        "readout": ("physical projective measurement of (q3,q4), postselect on "
                    "(0,0), trace out, normalise by P_succ"),
        "no_twirling": True, "used_analytic_input": False,
        "checks": {k: {kk: (float(vv) if isinstance(vv, float) else vv)
                       for kk, vv in v.items()} for k, v in checks.items()},
        "tolerance_study": tol_study,
        "table": table,
        "dense_max_abs_full_minus_first": float(max(diffs)),
        "dense_limiting_components": limiting_set,
        "dense_first_violation_rounds": [int(r) if r is not None else None
                                         for r in rounds_set],
        "near_threshold_unexpected": bad,
        "success_vs_epsilon": succ_vs_eps,
        "transverse_at_threshold": at_thr,
        "transverse_vs_q": in_q,
        "transverse_order_test": order,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    (OUT / "4q_qabs_verification.json").write_text(json.dumps(meta, indent=2))

    if not args.no_figures:
        use_style()
        plot_curve(dense, table)
        plot_first_vs_full(dense)
        plot_components(table)
        plot_eps015_trajectories(cases, eps_d, q_star)
        plot_success(cases, eps_d, q_star, succ_vs_eps)
        plot_transverse(at_thr, in_q, order)
    print("\n" + "=" * 78)
    print("FILES")
    print("=" * 78)
    for p in sorted(OUT.glob("*")):
        print(f"  {p.relative_to(ROOT)}")
    print(f"\n  elapsed {meta['elapsed_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
