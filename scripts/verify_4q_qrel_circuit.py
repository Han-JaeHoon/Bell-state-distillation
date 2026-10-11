#!/usr/bin/env python3
"""Circuit-level numerical study of the 4Q RELATIVE Bell-error-suppression
threshold q_rel(epsilon).

SELF-CONTAINED ON PURPOSE.  This script imports nothing from ``pqec_distill``
and no analytic recurrence.  The only prior information it uses is

  1. the canonical 4Q physical circuit specification, transcribed below from
     the docstring and ``CNOT_SEQUENCE`` of ``src/pqec_distill/circuit.py``
     (the repository's single source of truth for that gate list; the
     transcription is verified against that file and against
     ``scripts/verify_4q_qabs_circuit.py`` at run time),
  2. the Bell-isotropic input state,
  3. the per-CNOT two-qubit replacement depolarizing channel,
  4. the physical postselection rule of that specification,
  5. the operational definition of q_rel given in the task.

DEFINITION BEING TESTED

    R_i^(n) = p_i^(n) / p_Phi+^(n),      i in {Phi-, Psi+, Psi-}

    q_rel(eps) = sup { q : R_i^(n+1) < R_i^(n)
                           for every non-target i and every n >= 0 }

with p_B^(n) = <B| rho_n |B> read off the FULL conditional two-qubit density
matrix.  q_rel does NOT require the absolute populations to fall: a round in
which p_i rises while R_i falls still passes, and such events are searched for
explicitly.

Which Bell component is limiting, whether the first round binds, whether the
output stays Bell diagonal, and how q_rel compares with q_abs are OUTPUTS here,
not inputs.  The success probability is recorded but never enters the
threshold definition.

NO ANALYTIC SHORTCUTS IN THE PRIMARY PATH.  No Bell-diagonal reduced map, no
(x,y,z) recurrence, no closed-form population or success-probability map, no
previously computed 4Q threshold of any kind -- in particular the frozen q_abs
values are NOT used as bracketing hints.  Every density matrix is propagated in
full and every round feeds its exact conditional output into the next round: no
twirling, no re-isotropisation, no Bell-diagonal projection, no symmetrisation,
no off-diagonal truncation.  The comparison with q_abs happens in a single
clearly separated SECONDARY step, after the primary table has been computed.

Run:
    python scripts/verify_4q_qrel_circuit.py
    python scripts/verify_4q_qrel_circuit.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "4q_qrel"

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
                  "CNOT_SEQUENCE + module docstring (the repository's single "
                  "source of truth); identical to CNOT_SEQUENCE in "
                  "scripts/verify_4q_qabs_circuit.py")

NOISE_CONVENTION = ("two-qubit REPLACEMENT depolarizing after every CNOT, on "
                    "that CNOT's two wires: D_q(sigma) = (1-q) sigma + "
                    "q [I_ij/4 (x) Tr_ij(sigma)]  -- NOT the 15-Pauli "
                    "convention")

# ===========================================================================
# 1. gate primitives (dense, 4 qubits, |q1 q2 q3 q4> with q1 most significant)
# ===========================================================================

_I2 = np.eye(2, dtype=complex)
_P0 = np.array([[1, 0], [0, 0]], dtype=complex)
_P1 = np.array([[0, 0], [0, 1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


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
    """|0><0|_c (x) I + |1><1|_c (x) X_t, built from the definition."""
    if control == target:
        raise ValueError("control and target must differ")
    a = [_I2] * N_WIRES
    a[control] = _P0
    b = [_I2] * N_WIRES
    b[control] = _P1
    b[target] = _X
    return _kron_list(a) + _kron_list(b)


def cnot_permutation(control: int, target: int) -> np.ndarray:
    """The same CNOT written as the computational-basis permutation it induces
    (route B): no projectors, no Kronecker product."""
    m = np.zeros((DIM, DIM), dtype=complex)
    for idx in range(DIM):
        bits = [(idx >> (N_WIRES - 1 - w)) & 1 for w in range(N_WIRES)]
        if bits[control] == 1:
            bits[target] ^= 1
        m[sum(b << (N_WIRES - 1 - w) for w, b in enumerate(bits)), idx] = 1.0
    return m


def five_cnot_unitary(builder=cnot, sequence=None) -> np.ndarray:
    u = np.eye(DIM, dtype=complex)
    for control, target in (sequence or CNOT_SEQUENCE):
        u = builder(control, target) @ u
    return u


def full_unitary(builder=cnot, sequence=None) -> np.ndarray:
    """V = H_(q3) @ U_CNOT, the complete unitary before measurement."""
    return place_single(_H, Q3) @ five_cnot_unitary(builder, sequence)


def format_sequence(sequence=None) -> str:
    lines = []
    seq = sequence or CNOT_SEQUENCE
    for k, (c, t) in enumerate(seq, 1):
        cl, tl = f"q{c + 1}", f"q{t + 1}"
        lines.append(f"  {k}.  CNOT({cl} -> {tl})   wires ({c} -> {t})"
                     f"      [CNOT #{k}]  + depolarizing D_q on ({c},{t})")
    lines.append(f"  {len(seq) + 1}.  H(q3)   wire {Q3}      [ideal, no noise]")
    lines.append(f"  {len(seq) + 2}.  measure (q3, q4) in the computational "
                 f"basis; success = {SUCCESS_OUTCOME}; retain (q1, q2)")
    return "\n".join(lines)


def build_program(sequence=None, skip_noise_after=(), q_offsets=None):
    """``[("U", M) | ("D", (i, j), dq)]`` with the channels exactly where the
    CNOTs are and the final Hadamard noiseless.

    ``skip_noise_after`` and ``q_offsets`` exist ONLY for the negative controls
    of section 16; the primary path uses neither.
    """
    program = []
    for k, (control, target) in enumerate((sequence or CNOT_SEQUENCE), 1):
        program.append(("U", cnot(control, target)))
        if k not in skip_noise_after:
            dq = 0.0 if q_offsets is None else float(q_offsets.get(k, 0.0))
            program.append(("D", (control, target), dq))
    program.append(("U", place_single(_H, Q3)))
    return program


PROGRAM = build_program()

# ===========================================================================
# 2. the two-qubit replacement depolarizing channel, implemented directly
# ===========================================================================

_ROW = "abcd"
_COL = "efgh"


def replacement_depol(sigma: np.ndarray, pair, q: float) -> np.ndarray:
    """``D_q^(ij)(sigma) = (1-q) sigma + q [ I_ij/4 (x) Tr_ij(sigma) ]``.

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
# 3. Bell basis, input family, one 4Q round
# ===========================================================================

PHI_P = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
PHI_M = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
PSI_P = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
PSI_M = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
BELL = {"Phi+": PHI_P, "Phi-": PHI_M, "Psi+": PSI_P, "Psi-": PSI_M}
BELL_ORDER = ("Phi+", "Phi-", "Psi+", "Psi-")
NON_TARGET = ("Phi-", "Psi+", "Psi-")
_BELL_ROWS = np.stack([BELL[k] for k in BELL_ORDER])


def bell_populations(rho: np.ndarray) -> np.ndarray:
    """``p_i = <B_i| rho |B_i>``, each computed independently, with no equality
    assumed between any of them."""
    return np.array([float(np.real(BELL[k].conj() @ rho @ BELL[k]))
                     for k in BELL_ORDER])


def bell_offdiagonal_weight(rho: np.ndarray) -> float:
    """``C_off = sum_{i != j} |<B_i| rho |B_j>|^2``, measured on the state."""
    r = _BELL_ROWS.conj() @ rho @ _BELL_ROWS.T
    return float(np.sum(np.abs(r - np.diag(np.diag(r))) ** 2))


def isotropic_rho(eps: float) -> np.ndarray:
    """``rho_0(eps) = (1-eps)|Phi+><Phi+| + eps I/4``, from the definition."""
    e = float(eps)
    return (1 - e) * np.outer(PHI_P, PHI_P.conj()) + e * np.eye(4, dtype=complex) / 4


#: below this target population the ratios are not trusted; recorded, not hidden
RATIO_FLOOR = 1e-14


def relative_ratios(p: np.ndarray) -> np.ndarray:
    """``R_i = p_i / p_Phi+``.  ``R_Phi+`` is 1 by construction and unused."""
    denom = float(p[0])
    if not np.isfinite(denom) or denom <= RATIO_FLOOR:
        return np.full(4, np.inf)
    return p / denom


def project_success(sigma: np.ndarray, outcome=SUCCESS_OUTCOME):
    """Projective measurement of ``(q3, q4)``, keep the ``outcome`` branch,
    trace the measured wires out.  Returns ``(unnormalised 4x4 block, P_succ)``.

    For a rank-one projector on the measured wires, projecting and then tracing
    them out is exactly the corresponding diagonal sub-block, which is what is
    taken here; ``P_succ`` is that block's trace.
    """
    m3, m4 = outcome
    t = sigma.reshape([2] * (2 * N_WIRES))
    block = t[:, :, m3, m4, :, :, m3, m4]          # (q1,q2 | q1',q2')
    rho_tilde = block.reshape(4, 4)
    return rho_tilde, float(np.real(np.trace(rho_tilde)))


def one_round(rho: np.ndarray, q: float, program=None, outcome=SUCCESS_OUTCOME):
    """``rho_(n+1) = (projected, traced block) / P_succ``.

    Physical postselection on the measurement outcome -- NOT a parity-weighted
    reconstruction.  Route A.
    """
    sigma = np.kron(rho, rho)
    for item in (program if program is not None else PROGRAM):
        if item[0] == "U":
            m = item[1]
            sigma = m @ sigma @ m.conj().T
        else:
            sigma = replacement_depol(sigma, item[1], q + item[2])
    rho_tilde, p_succ = project_success(sigma, outcome)
    herm_residue = float(np.linalg.norm(rho_tilde - rho_tilde.conj().T))
    rho_tilde = 0.5 * (rho_tilde + rho_tilde.conj().T)   # exact-arithmetic id.
    if p_succ <= 0.0 or not np.isfinite(p_succ):
        raise FloatingPointError(f"success probability vanished: {p_succ!r}")
    out = rho_tilde / p_succ
    return out, {"p_succ": p_succ, "herm_residue": herm_residue,
                 "trace_error": abs(float(np.real(np.trace(out))) - 1.0),
                 "min_eig": float(np.linalg.eigvalsh(
                     0.5 * (out + out.conj().T)).min())}


# ===========================================================================
# 4. ROUTE B: an independent implementation of the same physical round
# ===========================================================================

_ROWB = "wxyz"
_COLB = "pqrs"


def _depol_route_b(sigma: np.ndarray, pair, q: float) -> np.ndarray:
    """Replacement depolarizing coded independently (own index letters, own
    contraction order)."""
    if q == 0.0:
        return sigma
    i, j = pair
    keep = [w for w in range(N_WIRES) if w not in (i, j)]
    t = sigma.reshape([2] * (2 * N_WIRES))
    row, col = list(_ROWB), list(_COLB)
    for w in (i, j):
        col[w] = row[w]
    reduced = np.einsum("".join(row) + "".join(col) + "->"
                        + "".join(row[w] for w in keep)
                        + "".join(col[w] for w in keep), t)
    out_idx = "".join(row[w] for w in keep) + "".join(col[w] for w in keep)
    rebuilt = np.einsum(f"{out_idx},{_ROWB[i]}{_COLB[i]},{_ROWB[j]}{_COLB[j]}"
                        f"->{_ROWB}{_COLB}", reduced, _I2, _I2)
    return (1.0 - q) * sigma + 0.25 * q * rebuilt.reshape(DIM, DIM)


def _hadamard_route_b(wire: int) -> np.ndarray:
    """H on one wire, built by explicit index arithmetic rather than kron."""
    m = np.zeros((DIM, DIM), dtype=complex)
    s = 1.0 / np.sqrt(2.0)
    shift = N_WIRES - 1 - wire
    for col in range(DIM):
        bit = (col >> shift) & 1
        base = col & ~(1 << shift)
        m[base, col] += s                                   # |0>
        m[base | (1 << shift), col] += s if bit == 0 else -s
    return m


def _bell_projectors_route_b():
    """The four Bell projectors built from their stabilizer form,
    ``|B><B| = (I + s1 P1)(I + s2 P2)/4`` with P1 = XX, P2 = ZZ."""
    i4 = np.eye(4, dtype=complex)
    xx = np.kron(_X, _X)
    zz = np.kron(np.array([[1, 0], [0, -1]], dtype=complex),
                 np.array([[1, 0], [0, -1]], dtype=complex))
    signs = {"Phi+": (+1, +1), "Phi-": (-1, +1),
             "Psi+": (+1, -1), "Psi-": (-1, -1)}
    return {k: 0.25 * (i4 + s1 * xx) @ (i4 + s2 * zz)
            for k, (s1, s2) in signs.items()}


_PROJ_B = _bell_projectors_route_b()


def bell_populations_route_b(rho: np.ndarray) -> np.ndarray:
    """``p_i = Tr(|B_i><B_i| rho)`` from independently built projectors."""
    return np.array([float(np.real(np.trace(_PROJ_B[k] @ rho)))
                     for k in BELL_ORDER])


def _postselect_route_b(sigma: np.ndarray, outcome=SUCCESS_OUTCOME):
    """Project with an explicit 16x16 projector, then partial-trace by einsum.

    No sub-block shortcut, so this path checks that shortcut rather than
    reusing it.
    """
    m3, m4 = outcome
    proj = _kron_list([_I2, _I2,
                       _P0 if m3 == 0 else _P1,
                       _P0 if m4 == 0 else _P1])
    projected = proj @ sigma @ proj.conj().T
    t = projected.reshape([2] * (2 * N_WIRES))
    reduced = np.einsum("abcdefcd->abef", t).reshape(4, 4)
    return reduced, float(np.real(np.trace(reduced)))


def one_round_route_b(rho: np.ndarray, q: float, sequence=None,
                      outcome=SUCCESS_OUTCOME, skip_noise_after=(),
                      q_offsets=None):
    """Gate-by-gate independent route: permutation CNOTs, index-arithmetic H,
    independent channel, independent projection and partial trace."""
    sigma = np.kron(rho, rho)
    for k, (control, target) in enumerate((sequence or CNOT_SEQUENCE), 1):
        g = cnot_permutation(control, target)
        sigma = g @ sigma @ g.conj().T
        if k not in skip_noise_after:
            dq = 0.0 if q_offsets is None else float(q_offsets.get(k, 0.0))
            sigma = _depol_route_b(sigma, (control, target), q + dq)
    h = _hadamard_route_b(Q3)
    sigma = h @ sigma @ h.conj().T
    rho_tilde, p_succ = _postselect_route_b(sigma, outcome)
    if p_succ <= 0.0 or not np.isfinite(p_succ):
        raise FloatingPointError(f"success probability vanished: {p_succ!r}")
    return rho_tilde / p_succ, {"p_succ": p_succ}


# ===========================================================================
# 5. provenance and independent sanity checks
# ===========================================================================

def _extract_sequence(path: Path, name: str):
    """Read a ``(control, target)`` list straight out of a source file.

    The file is parsed textually, not imported, so no module-level code of the
    frozen scripts runs and no stored threshold can leak in.  The list taken is
    the one on the right-hand side of the top-level assignment to ``name``, so
    a mention of the name inside a docstring or a type annotation such as
    ``list[tuple[int, int]]`` cannot be picked up by mistake.
    """
    text = path.read_text()
    m = re.search(rf"^{re.escape(name)}[^\n=]*=\s*", text, re.M)
    if m is None:
        raise ValueError(f"no top-level assignment to {name} in {path.name}")
    start = text.index("[", m.end() - 1)
    end = text.index("]", start) + 1
    body = text[start:end]
    pairs = re.findall(r"\(\s*Q(\d)\s*,\s*Q(\d)\s*\)", body)
    if pairs:
        return [(int(a) - 1, int(b) - 1) for a, b in pairs]
    return [(int(a), int(b)) for a, b in
            re.findall(r"\(\s*(\d)\s*,\s*(\d)\s*\)", body)]


def provenance_checks() -> dict:
    res = {"cnot_sequence_used": [list(p) for p in CNOT_SEQUENCE],
           "n_cnot": N_CNOT,
           "success_outcome": list(SUCCESS_OUTCOME),
           "retained": list(RETAIN), "measured": list(MEASURED)}
    for label, path, name in (
            ("src/pqec_distill/circuit.py",
             ROOT / "src" / "pqec_distill" / "circuit.py", "CNOT_SEQUENCE"),
            ("scripts/verify_4q_qabs_circuit.py",
             ROOT / "scripts" / "verify_4q_qabs_circuit.py", "CNOT_SEQUENCE")):
        try:
            res[f"matches_{label}"] = _extract_sequence(path, name) == CNOT_SEQUENCE
        except Exception as exc:                                  # pragma: no cover
            res[f"matches_{label}"] = f"unavailable: {exc}"
    return res


def random_density_matrix(rng, dim):
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = a @ a.conj().T
    return rho / np.trace(rho).real


def structural_checks(rng) -> dict:
    v = full_unitary()
    vp = full_unitary(cnot_permutation)
    res = {
        "unitarity": float(np.max(np.abs(v.conj().T @ v - np.eye(DIM)))),
        "dense_vs_permutation_cnots": float(np.max(np.abs(v - vp))),
        "hadamard_route_b_vs_dense": float(np.max(np.abs(
            _hadamard_route_b(Q3) - place_single(_H, Q3)))),
        "n_channels": sum(1 for item in PROGRAM if item[0] == "D"),
        "channel_locations": [list(item[1]) for item in PROGRAM
                              if item[0] == "D"],
    }
    res["channels_match_cnots"] = (res["channel_locations"]
                                   == [list(p) for p in CNOT_SEQUENCE])
    # Bell projectors: independent construction vs outer products
    res["bell_projectors_route_b"] = max(
        float(np.max(np.abs(_PROJ_B[k] - np.outer(BELL[k], BELL[k].conj()))))
        for k in BELL_ORDER)
    # q = 0 is the exact identity channel; q = 1 fully replaces the pair
    z = random_density_matrix(rng, DIM)
    res["q0_identity_channel"] = float(np.max(np.abs(
        replacement_depol(z, (0, 1), 0.0) - z)))
    worst_q1 = 0.0
    for pair in [(0, 1), (2, 3), (0, 3), (1, 2)]:
        full = replacement_depol(z, pair, 1.0)
        i, j = pair
        keep = [w for w in range(N_WIRES) if w not in pair]
        t = z.reshape([2] * (2 * N_WIRES))
        row, col = list(_ROW), list(_COL)
        for w in pair:
            col[w] = row[w]
        red = np.einsum("".join(row) + "".join(col) + "->"
                        + "".join(row[w] for w in keep)
                        + "".join(col[w] for w in keep), t)
        out_idx = "".join(row[w] for w in keep) + "".join(col[w] for w in keep)
        expect = np.einsum(f"{out_idx},{_ROW[i]}{_COL[i]},{_ROW[j]}{_COL[j]}"
                           f"->{_ROW}{_COL}", red, _I2, _I2)
        worst_q1 = max(worst_q1, float(np.max(np.abs(
            full - 0.25 * expect.reshape(DIM, DIM)))))
    res["q1_pair_replacement"] = worst_q1
    res["channel_vs_route_b_channel"] = max(
        float(np.max(np.abs(replacement_depol(z, pair, qq)
                            - _depol_route_b(z, pair, qq))))
        for pair in [(0, 1), (2, 3), (1, 3)] for qq in (0.0, 0.37, 1.0))
    # measurement-branch probabilities must sum to one
    sigma = full_unitary() @ np.kron(isotropic_rho(0.2), isotropic_rho(0.2)) \
        @ full_unitary().conj().T
    res["branch_probabilities_sum"] = abs(sum(
        project_success(sigma, (a, b))[1] for a in (0, 1) for b in (0, 1)) - 1.0)
    return res


def route_comparison(points) -> dict:
    """Route A against route B on the full retained state and every derived
    quantity, at representative ``(eps, q)`` points and over several rounds."""
    rows = []
    worst = {"rho": 0.0, "populations": 0.0, "p_succ": 0.0, "ratios": 0.0,
             "populations_projector_vs_expectation": 0.0}
    for eps, q in points:
        rho_a = rho_b = isotropic_rho(eps)
        for n in range(1, 6):
            rho_a, info_a = one_round(rho_a, q)
            rho_b, info_b = one_round_route_b(rho_b, q)
            d_rho = float(np.max(np.abs(rho_a - rho_b)))
            pa, pb = bell_populations(rho_a), bell_populations_route_b(rho_b)
            d_pop = float(np.max(np.abs(pa - pb)))
            d_ps = abs(info_a["p_succ"] - info_b["p_succ"])
            d_rat = float(np.max(np.abs(
                relative_ratios(pa)[1:] - relative_ratios(pb)[1:])))
            d_proj = float(np.max(np.abs(pa - bell_populations_route_b(rho_a))))
            worst["rho"] = max(worst["rho"], d_rho)
            worst["populations"] = max(worst["populations"], d_pop)
            worst["p_succ"] = max(worst["p_succ"], d_ps)
            worst["ratios"] = max(worst["ratios"], d_rat)
            worst["populations_projector_vs_expectation"] = max(
                worst["populations_projector_vs_expectation"], d_proj)
            rows.append({"epsilon": eps, "q": q, "round": n,
                         "max_abs_rho_diff": d_rho,
                         "max_abs_population_diff": d_pop,
                         "abs_p_succ_diff": d_ps,
                         "max_abs_ratio_diff": d_rat,
                         "p_succ_route_a": info_a["p_succ"],
                         "p_succ_route_b": info_b["p_succ"]})
    return {"rows": rows, "worst": worst}


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
#   N_MAX     = 5000    maximum rounds; the number actually used is reported.
#
# OFFDIAG_TRUST is the round-off contamination guard.  The Bell-isotropic input
# is built exactly Bell-diagonal, so its Bell off-diagonal weight starts at
# round-off; if the circuit map amplified that transverse direction the
# accumulated round-off could push the simulated orbit off a saddle and
# manufacture a ratio increase the exact dynamics does not have.  Whether this
# map amplifies it is MEASURED by `transverse_growth()`, not assumed.

STEP_TOL = 1e-14
VIOL_TOL = 1e-12
N_MAX = 5000
OFFDIAG_TRUST = 1e-18


def run_trajectory(eps: float, q: float, step_tol=STEP_TOL, n_max=N_MAX,
                   viol_tol=VIOL_TOL, components=NON_TARGET, record=False,
                   offdiag_trust=OFFDIAG_TRUST, program=None,
                   outcome=SUCCESS_OUTCOME):
    """Iterate the real circuit and look for the first RELATIVE violation.

    The failure condition is ``Delta R_i = R_i^(n+1) - R_i^(n) > viol_tol``.
    It is deliberately NOT a condition on ``Delta p_i``: a non-target
    population may grow while its ratio to the target falls, and that passes.
    """
    idx = [BELL_ORDER.index(c) for c in components]
    rho = isotropic_rho(eps)
    p = bell_populations(rho)
    r = relative_ratios(p)
    c_off0 = bell_offdiagonal_weight(rho)
    rows = [{"n": 0, "p": p.copy(), "R": r.copy(), "dp": np.full(4, np.nan),
             "dR": np.full(4, np.nan), "step": np.nan, "c_off": c_off0,
             "p_succ": np.nan, "trace_error": 0.0, "herm_residue": 0.0,
             "min_eig": float(np.linalg.eigvalsh(rho).min())}] if record else []
    worst_dr, worst_at, worst_comp = -np.inf, None, None
    c_off_max, min_eig, max_trace_err = c_off0, np.inf, 0.0
    max_herm, min_p_succ, min_p_target = 0.0, np.inf, float(p[0])
    n, step, trusted, n_trusted, converged = 0, np.nan, True, None, False
    abs_rise_with_rel_fall = []
    while n < n_max:
        rho_next, info = one_round(rho, q, program=program, outcome=outcome)
        p_next = bell_populations(rho_next)
        r_next = relative_ratios(p_next)
        dp = p_next - p
        dr = r_next - r
        step = float(np.max(np.abs(rho_next - rho)))
        c_off = bell_offdiagonal_weight(rho_next)
        c_off_max = max(c_off_max, c_off)
        min_eig = min(min_eig, info["min_eig"])
        max_trace_err = max(max_trace_err, info["trace_error"])
        max_herm = max(max_herm, info["herm_residue"])
        min_p_succ = min(min_p_succ, info["p_succ"])
        min_p_target = min(min_p_target, float(p_next[0]))
        if record:
            rows.append({"n": n + 1, "p": p_next.copy(), "R": r_next.copy(),
                         "dp": dp.copy(), "dR": dr.copy(), "step": step,
                         "c_off": c_off, "p_succ": info["p_succ"],
                         "trace_error": info["trace_error"],
                         "herm_residue": info["herm_residue"],
                         "min_eig": info["min_eig"]})
        for i in idx:
            if dp[i] > viol_tol and dr[i] < -viol_tol:
                abs_rise_with_rel_fall.append(
                    {"round": n, "component": BELL_ORDER[i],
                     "delta_p": float(dp[i]), "delta_R": float(dr[i]),
                     "p_n": float(p[i]), "p_np1": float(p_next[i]),
                     "p_phi_plus_n": float(p[0]),
                     "p_phi_plus_np1": float(p_next[0]),
                     "R_n": float(r[i]), "R_np1": float(r_next[i])})
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
                    "p_succ": info["p_succ"], "rounds_run": n + 1,
                    "final_step": step, "trusted": True, "n_trusted": None,
                    "converged": False, "c_off_initial": c_off0,
                    "c_off_final": c_off, "c_off_max": c_off_max,
                    "min_eig": min_eig, "max_trace_error": max_trace_err,
                    "max_herm_residue": max_herm, "min_p_succ": min_p_succ,
                    "min_p_target": min_p_target, "worst_delta_R": worst_dr,
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
            "c_off_max": c_off_max, "min_eig": min_eig,
            "max_trace_error": max_trace_err, "max_herm_residue": max_herm,
            "min_p_succ": min_p_succ, "min_p_target": min_p_target,
            "worst_delta_R": worst_dr, "worst_round": worst_at,
            "worst_component": worst_comp, "p_final": p.copy(),
            "R_final": r.copy(),
            "abs_rise_with_rel_fall": abs_rise_with_rel_fall, "rows": rows}


def record_trajectory(eps: float, q: float, n_rounds: int,
                      viol_tol=VIOL_TOL, components=NON_TARGET):
    """Round-by-round record for plotting; does NOT stop at the first
    violation, so the figure shows what happens afterwards."""
    idx = [BELL_ORDER.index(c) for c in components]
    rho = isotropic_rho(eps)
    p = bell_populations(rho)
    r = relative_ratios(p)
    rows = [{"n": 0, "p": p.copy(), "R": r.copy(), "dp": np.full(4, np.nan),
             "dR": np.full(4, np.nan), "c_off": bell_offdiagonal_weight(rho),
             "p_succ": np.nan}]
    first_violation = None
    for n in range(n_rounds):
        rho, info = one_round(rho, q)
        p_next = bell_populations(rho)
        r_next = relative_ratios(p_next)
        dp, dr = p_next - p, r_next - r
        if first_violation is None and any(dr[i] > viol_tol for i in idx):
            k = max((i for i in idx if dr[i] > viol_tol), key=lambda i: dr[i])
            first_violation = {"round": n, "component": BELL_ORDER[k],
                               "delta_R": float(dr[k]), "delta_p": float(dp[k])}
        rows.append({"n": n + 1, "p": p_next.copy(), "R": r_next.copy(),
                     "dp": dp.copy(), "dR": dr.copy(),
                     "c_off": bell_offdiagonal_weight(rho),
                     "p_succ": info["p_succ"]})
        p, r = p_next, r_next
    return rows, first_violation


def first_round_step(eps: float, q: float):
    """``rho_0 -> rho_1`` only; returns ``(p0, p1, R0, R1, P_succ)``."""
    rho = isotropic_rho(eps)
    p0 = bell_populations(rho)
    rho1, info = one_round(rho, q)
    p1 = bell_populations(rho1)
    return p0, p1, relative_ratios(p0), relative_ratios(p1), info["p_succ"]


def first_round_violates(eps: float, q: float, viol_tol=VIOL_TOL,
                         components=NON_TARGET) -> bool:
    _p0, _p1, r0, r1, _ps = first_round_step(eps, q)
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


def _perturbed_rho(eps: float, amp: float) -> np.ndarray:
    """Bell-isotropic state plus a Hermitian Bell-OFF-diagonal perturbation."""
    pert = amp * (np.outer(PHI_P, PHI_M.conj()) + np.outer(PHI_M, PHI_P.conj()))
    pert = pert + 1j * amp * (np.outer(PSI_P, PSI_M.conj())
                              - np.outer(PSI_M, PSI_P.conj()))
    rho = isotropic_rho(eps) + pert
    return rho / np.trace(rho).real


def transverse_growth(eps=0.15, q=0.05, amp=1e-08, n_rounds=12) -> dict:
    """Does the map amplify the Bell off-diagonal (transverse) direction?

    A small Hermitian transverse perturbation is added to the Bell-diagonal
    input and its off-diagonal amplitude is followed.  Measured, not assumed.

    Two things are reported separately, because they mean different things:

    * ``first_round_ratio`` -- the only round whose input amplitude is above
      round-off, so the only one that measures the map;
    * ``later_ratios`` -- once the amplitude has collapsed to ~1e-17 the
      sequence is pure round-off and its ratios fluctuate around one; they say
      nothing about the map and must not be read as growth.

    ``amplitude_scaling_exponent`` fits ``out ~ in^k`` from two input
    amplitudes: ``k ~ 1`` would be linear amplification/damping, ``k ~ 2`` an
    annihilation of the transverse direction to first order.
    """
    rho = _perturbed_rho(eps, amp)
    weights = [np.sqrt(bell_offdiagonal_weight(rho))]
    for _ in range(n_rounds):
        rho, _ = one_round(rho, q)
        weights.append(np.sqrt(bell_offdiagonal_weight(rho)))
    ratios = [weights[k + 1] / weights[k] for k in range(len(weights) - 1)
              if weights[k] > 0]
    # amplitude-scaling exponent from a single round at two input amplitudes
    pairs = []
    for a in (amp, amp / 10.0):
        r0 = _perturbed_rho(eps, a)
        w_in = np.sqrt(bell_offdiagonal_weight(r0))
        r1, _ = one_round(r0, q)
        pairs.append((w_in, np.sqrt(bell_offdiagonal_weight(r1))))
    (i1, o1), (i2, o2) = pairs
    exponent = (float(np.log(o1 / o2) / np.log(i1 / i2))
                if min(i1, i2, o1, o2) > 0 else float("nan"))
    return {"eps": eps, "q": q, "amplitude": amp,
            "offdiag_amplitude_sequence": [float(w) for w in weights],
            "growth_ratios": [float(x) for x in ratios],
            "first_round_ratio": float(ratios[0]) if ratios else float("nan"),
            "later_ratios_are_roundoff_level": bool(
                len(weights) > 2 and max(weights[2:]) < 1e-14),
            "max_amplitude_after_first_round": float(max(weights[1:]))
            if len(weights) > 1 else float("nan"),
            "max_later_ratio_roundoff": float(max(ratios[1:]))
            if len(ratios) > 1 else float("nan"),
            "amplitude_scaling_exponent": exponent,
            "scaling_probe": [[float(x) for x in pr] for pr in pairs]}


# ===========================================================================
# 7. threshold search -- bracket, then bisect, on the circuit predicate only
# ===========================================================================

def bracket(predicate, q_lo=0.0, q_hi=0.9, n_coarse=90):
    grid = np.linspace(q_lo, q_hi, n_coarse + 1)
    flags = [predicate(float(q)) for q in grid]
    changes = [k for k in range(len(flags) - 1) if flags[k] != flags[k + 1]]
    info = {"n_sign_changes": len(changes),
            "monotone": len(changes) == 1,
            "all_violating": bool(all(flags)),
            "none_violating": bool(not any(flags)),
            "scan_grid_step": float(grid[1] - grid[0]),
            "n_coarse": n_coarse}
    if not changes:
        return None, None, info
    k = changes[0]
    return float(grid[k]), float(grid[k + 1]), info


def bisect(predicate, lo: float, hi: float, tol=1e-10, max_iter=200):
    """``lo`` satisfies, ``hi`` violates; return the largest q known to satisfy."""
    if predicate(lo) or not predicate(hi):
        raise ValueError(f"bad bracket: predicate({lo}) / predicate({hi})")
    it = 0
    while hi - lo > tol and it < max_iter:
        mid = 0.5 * (lo + hi)
        if predicate(mid):
            hi = mid
        else:
            lo = mid
        it += 1
    return lo, hi, it


def _threshold(pred, q_hi=0.9, n_coarse=90, tol=1e-10) -> dict:
    """The full bracket-then-bisect record for one predicate.  No hints."""
    lo, hi, info = bracket(pred, 0.0, q_hi, n_coarse)
    out = dict(info)
    if lo is None:
        out.update({"q_threshold": None, "q_upper_violating": None,
                    "bracket_width": None, "bisect_iterations": None,
                    "has_root": False})
        return out
    a, b, it = bisect(pred, lo, hi, tol=tol)
    out.update({"q_threshold": a, "q_upper_violating": b,
                "bracket_width": b - a, "bisect_iterations": it,
                "has_root": True, "coarse_bracket": [lo, hi]})
    return out


def q_rel_full(eps: float, viol_tol=VIOL_TOL, tol=1e-10, components=NON_TARGET,
               q_hi=0.9, n_coarse=90) -> dict:
    pred = lambda q: run_trajectory(eps, q, viol_tol=viol_tol,              # noqa: E731
                                    components=components)["violated"]
    return _threshold(pred, q_hi, n_coarse, tol)


def q_rel_first(eps: float, viol_tol=VIOL_TOL, tol=1e-10,
                components=NON_TARGET, q_hi=0.9, n_coarse=90) -> dict:
    pred = lambda q: first_round_violates(eps, q, viol_tol=viol_tol,        # noqa: E731
                                          components=components)
    return _threshold(pred, q_hi, n_coarse, tol)


def per_component_thresholds(eps: float, viol_tol=VIOL_TOL, tol=1e-10,
                             q_hi=0.9, n_coarse=90) -> dict:
    """First-round AND all-round threshold for each non-target component."""
    out = {}
    for c in NON_TARGET:
        out[f"{c}_first"] = q_rel_first(eps, viol_tol, tol, (c,), q_hi, n_coarse)
        out[f"{c}_all"] = q_rel_full(eps, viol_tol, tol, (c,), q_hi, n_coarse)
    return out


# ===========================================================================
# 8. the main table
# ===========================================================================

EPS_TABLE = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60,
             2.0 / 3.0, 0.75, 0.90)
EPS_DEGENERATE = (0.0, 1.0)
NEAR_EPS = (0.05, 0.15, 0.40, 2.0 / 3.0)


def main_table(eps_values, tol=1e-10, q_hi=0.9, n_coarse=90, verbose=True):
    rows, extra = [], {}
    for eps in eps_values:
        t0 = time.time()
        full = q_rel_full(eps, tol=tol, q_hi=q_hi, n_coarse=n_coarse)
        first = q_rel_first(eps, tol=tol, q_hi=q_hi, n_coarse=n_coarse)
        comps = per_component_thresholds(eps, tol=tol, q_hi=q_hi,
                                         n_coarse=n_coarse)
        q_star = full["q_threshold"]
        q_up = full["q_upper_violating"]
        # what happens just above the threshold
        up = run_trajectory(eps, q_up, record=False) if q_up is not None else {}
        # the trajectory at the largest satisfying q: diagnostics + events
        at = run_trajectory(eps, q_star, record=False) if q_star is not None else {}
        # absolute-rise / relative-fall census, sampled below the threshold
        events = []
        if q_star is not None:
            for frac in (0.25, 0.5, 0.75, 0.9, 0.99, 1.0):
                qq = q_star * frac
                tr = run_trajectory(eps, qq)
                for ev in tr["abs_rise_with_rel_fall"]:
                    events.append(dict(epsilon=eps, q=qq, **ev))
        rho0 = isotropic_rho(eps)
        p0 = bell_populations(rho0)
        row = {
            "epsilon": eps,
            "F_in": float(p0[0]),
            "q_rel_full": q_star,
            "q_upper_violating": q_up,
            "q_rel_first": first["q_threshold"],
            "q_rel_full_minus_first": (
                None if q_star is None or first["q_threshold"] is None
                else q_star - first["q_threshold"]),
            "q_rel_Phi_minus_first": comps["Phi-_first"]["q_threshold"],
            "q_rel_Psi_plus_first": comps["Psi+_first"]["q_threshold"],
            "q_rel_Psi_minus_first": comps["Psi-_first"]["q_threshold"],
            "q_rel_Phi_minus_all": comps["Phi-_all"]["q_threshold"],
            "q_rel_Psi_plus_all": comps["Psi+_all"]["q_threshold"],
            "q_rel_Psi_minus_all": comps["Psi-_all"]["q_threshold"],
            "limiting_component": up.get("component"),
            "all_violating_components_at_upper": " ".join(
                up.get("all_violating", [])),
            "binding_round": up.get("round"),
            "Delta_R_at_upper": up.get("delta_R_violating"),
            "Delta_p_of_violating_at_upper": up.get("delta_p_of_violating"),
            "success_probability_at_relevant_step": up.get("p_succ"),
            "min_success_probability_at_q_rel": at.get("min_p_succ"),
            "max_Bell_offdiagonal_weight": max(
                at.get("c_off_max", 0.0), up.get("c_off_max", 0.0)),
            "n_absolute_rise_relative_fall_events": len(events),
            "rounds_run_at_q_rel": at.get("rounds_run"),
            "converged_at_q_rel": bool(at.get("converged", False)),
            "trusted_at_q_rel": bool(at.get("trusted", False)),
            "min_eigenvalue_at_q_rel": at.get("min_eig"),
            "max_trace_error_at_q_rel": at.get("max_trace_error"),
            "max_hermiticity_residue_at_q_rel": at.get("max_herm_residue"),
            "min_p_target_at_q_rel": at.get("min_p_target"),
            "bracket_sign_changes": full["n_sign_changes"],
            "bracket_width": full["bracket_width"],
            "bisect_iterations": full["bisect_iterations"],
            "seconds": time.time() - t0,
        }
        rows.append(row)
        extra[eps] = {"components": comps, "events": events}
        if verbose:
            print(f"      eps={eps:<9.6f} q_rel_full={q_star!r:<22} "
                  f"first={first['q_threshold']!r:<22} "
                  f"limiting={row['limiting_component']} "
                  f"round={row['binding_round']} "
                  f"sign_changes={full['n_sign_changes']} "
                  f"({row['seconds']:.1f} s)", flush=True)
    return rows, extra


def degenerate_endpoints(tol=1e-10, q_hi=0.9, n_coarse=90) -> list:
    """eps = 0 and eps = 1 are reported separately, never forced into a root."""
    out = []
    q_probe = (0.0, 1e-6, 1e-3, 0.01, 0.05, 0.2, 0.5, 0.9)
    for eps in EPS_DEGENERATE:
        rho0 = isotropic_rho(eps)
        p0 = bell_populations(rho0)
        r0 = relative_ratios(p0)
        probes = []
        for q in q_probe:
            tr = run_trajectory(eps, q)
            p0b, p1b, r0b, r1b, ps = first_round_step(eps, q)
            probes.append({"q": q, "violated": bool(tr["violated"]),
                           "first_violating_round": tr["round"],
                           "component": tr["component"],
                           "rounds_run": tr["rounds_run"],
                           "max_abs_delta_R_first_round": float(
                               np.max(np.abs((r1b - r0b)[1:]))
                               if np.all(np.isfinite(r1b)) else np.inf),
                           "worst_delta_R": tr["worst_delta_R"],
                           "p_succ_first_round": ps,
                           "p1": [float(x) for x in p1b]})
        thr = q_rel_full(eps, tol=tol, q_hi=q_hi, n_coarse=n_coarse)
        out.append({"epsilon": eps, "p0": [float(x) for x in p0],
                    "R0": [float(x) for x in r0],
                    "has_root_in_scan": thr["has_root"],
                    "q_threshold": thr["q_threshold"],
                    "q_upper_violating": thr["q_upper_violating"],
                    "n_sign_changes": thr["n_sign_changes"],
                    "all_violating": thr["all_violating"],
                    "none_violating": thr["none_violating"],
                    "probes": probes})
    return out


def tolerance_sensitivity(eps_values, tolerances=(1e-10, 1e-12, 1e-14),
                          tol=1e-10) -> list:
    rows = []
    for eps in eps_values:
        row = {"epsilon": eps}
        for vt in tolerances:
            r = q_rel_full(eps, viol_tol=vt, tol=tol)
            row[f"q_rel_full_viol_tol_{vt:g}"] = r["q_threshold"]
        vals = [row[k] for k in row if k.startswith("q_rel_full_viol")
                and row[k] is not None]
        row["spread"] = (max(vals) - min(vals)) if len(vals) > 1 else None
        rows.append(row)
    return rows


# ===========================================================================
# 9. negative / sensitivity controls   (never used to compute q_rel)
# ===========================================================================

def negative_controls(eps=0.15, q=0.05) -> dict:
    """Four deliberate deviations from the frozen specification."""
    base_rho, base_info = one_round(isotropic_rho(eps), q)
    base_p = bell_populations(base_rho)
    base_r = relative_ratios(base_p)
    base_thr = q_rel_full(eps)["q_threshold"]
    rows = []

    def _record(name, desc, rho_mod, info_mod, thr_info=None):
        p = bell_populations(rho_mod)
        r = relative_ratios(p)
        thr = None if thr_info is None else thr_info["q_threshold"]
        rows.append({
            "control": name, "description": desc, "epsilon": eps, "q": q,
            "max_abs_rho_diff": float(np.max(np.abs(rho_mod - base_rho))),
            "max_abs_population_diff": float(np.max(np.abs(p - base_p))),
            "max_abs_ratio_diff": float(np.max(np.abs(r[1:] - base_r[1:]))),
            "abs_p_succ_diff": abs(info_mod["p_succ"] - base_info["p_succ"]),
            "q_rel_full_modified": thr,
            "q_rel_full_shift": (None if thr is None or base_thr is None
                                 else thr - base_thr),
            "modified_has_root_in_scan": (None if thr_info is None
                                          else thr_info["has_root"]),
            "modified_all_violating": (None if thr_info is None
                                       else thr_info["all_violating"]),
            "modified_none_violating": (None if thr_info is None
                                        else thr_info["none_violating"])})

    # A: noise after one CNOT raised by +1e-3
    prog_a = build_program(q_offsets={3: 1e-03})
    rho_a, info_a = one_round(isotropic_rho(eps), q, program=prog_a)
    thr_a = _threshold(lambda qq: run_trajectory(                           # noqa: E731
        eps, qq, program=build_program(q_offsets={3: 1e-03}))["violated"])
    _record("A_one_cnot_noise_plus_1e-3",
            "replacement-depolarizing strength after CNOT #3 set to q + 1e-3",
            rho_a, info_a, thr_a)

    # B: one depolarizing channel removed
    prog_b = build_program(skip_noise_after=(2,))
    rho_b, info_b = one_round(isotropic_rho(eps), q, program=prog_b)
    thr_b = _threshold(lambda qq: run_trajectory(                           # noqa: E731
        eps, qq, program=build_program(skip_noise_after=(2,)))["violated"])
    _record("B_one_channel_removed",
            "depolarizing channel after CNOT #2 removed",
            rho_b, info_b, thr_b)

    # C: one CNOT reversed
    seq_c = list(CNOT_SEQUENCE)
    seq_c[3] = (seq_c[3][1], seq_c[3][0])
    prog_c = build_program(sequence=seq_c)
    rho_c, info_c = one_round(isotropic_rho(eps), q, program=prog_c)
    thr_c = _threshold(lambda qq: run_trajectory(                           # noqa: E731
        eps, qq, program=build_program(sequence=seq_c))["violated"])
    _record("C_one_cnot_reversed",
            f"CNOT #4 reversed: {CNOT_SEQUENCE[3]} -> {tuple(seq_c[3])}",
            rho_c, info_c, thr_c)

    # D: the wrong postselection branch
    rho_d, info_d = one_round(isotropic_rho(eps), q, outcome=(0, 1))
    thr_d = _threshold(lambda qq: run_trajectory(                           # noqa: E731
        eps, qq, outcome=(0, 1))["violated"])
    _record("D_wrong_postselection_branch",
            "postselect (q3,q4) = (0,1) instead of (0,0)",
            rho_d, info_d, thr_d)

    for r in rows:
        r["detected"] = bool(r["max_abs_rho_diff"] > 1e-09
                             or (r["q_rel_full_shift"] is not None
                                 and abs(r["q_rel_full_shift"]) > 1e-09)
                             or r["modified_has_root_in_scan"] is False)
    return {"rows": rows, "baseline_q_rel_full": base_thr,
            "baseline_p_succ": base_info["p_succ"]}


# ===========================================================================
# 10. SECONDARY: comparison with the frozen q_abs values
# ===========================================================================

def secondary_q_abs_comparison(table) -> dict:
    """Read the already-frozen 4Q q_abs results and compare.

    Called ONCE, after the primary table above is complete.  Nothing here feeds
    back into q_rel.
    """
    path = ROOT / "results" / "4q_qabs" / "4q_qabs_circuit.csv"
    if not path.exists():
        return {"available": False, "reason": f"{path} not present"}
    with path.open() as fh:
        q_abs = {}
        for rec in csv.DictReader(fh):
            try:
                q_abs[round(float(rec["epsilon"]), 12)] = float(rec["q_abs_full"])
            except (KeyError, TypeError, ValueError):
                continue
    rows, worst_viol = [], None
    for r in table:
        key = round(float(r["epsilon"]), 12)
        qa = q_abs.get(key)
        if qa is None or r["q_rel_full"] is None:
            continue
        diff = r["q_rel_full"] - qa
        rows.append({"epsilon": r["epsilon"], "q_abs": qa,
                     "q_rel": r["q_rel_full"], "q_rel_minus_q_abs": diff,
                     "q_abs_le_q_rel": bool(qa <= r["q_rel_full"] + 1e-12)})
        if qa > r["q_rel_full"] + 1e-12 and (worst_viol is None
                                             or diff < worst_viol["q_rel_minus_q_abs"]):
            worst_viol = rows[-1]
    return {"available": True, "rows": rows, "n": len(rows),
            "q_abs_le_q_rel_everywhere": all(x["q_abs_le_q_rel"] for x in rows),
            "min_difference": min((x["q_rel_minus_q_abs"] for x in rows),
                                  default=None),
            "max_difference": max((x["q_rel_minus_q_abs"] for x in rows),
                                  default=None),
            "worst_ordering_violation": worst_viol,
            "source": str(path.relative_to(ROOT))}


# ===========================================================================
# 11. figures
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
    import matplotlib.pyplot as plt
    paths = []
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        paths.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return paths


def plot_curve(table, dense):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    if dense:
        ax.plot([r["epsilon"] for r in dense], [r["q_rel_full"] for r in dense],
                color=BELL_COLOR["Phi+"], linewidth=2.0, alpha=0.8,
                label="dense sweep")
    ax.plot([r["epsilon"] for r in table], [r["q_rel_full"] for r in table],
            linestyle="none", marker="o", markersize=5.2,
            markerfacecolor="white", markeredgecolor=BELL_COLOR["Phi+"],
            markeredgewidth=1.6, label="mandatory table")
    ax.set_xlabel(r"input noise $\varepsilon$")
    ax.set_ylabel(r"$q_{\mathrm{rel}}$")
    ax.set_title(r"4Q relative Bell-error-suppression threshold"
                 "\n" r"(full repeated-circuit predicate)", color=INK)
    ax.legend(loc="lower right")
    _despine(ax)
    fig.tight_layout()
    return save(fig, "4q_qrel_curve")


def plot_first_vs_full(table):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    e = [r["epsilon"] for r in table]
    axes[0].plot(e, [r["q_rel_full"] for r in table],
                 color=BELL_COLOR["Phi+"], linewidth=3.4, alpha=0.85,
                 label=r"$q_{\mathrm{rel}}$, all rounds")
    axes[0].plot(e, [r["q_rel_first"] for r in table],
                 color=BELL_COLOR["Phi-"], linewidth=1.4, linestyle=(0, (5, 2)),
                 label=r"$q_{\mathrm{rel}}$, first round only")
    axes[0].set_xlabel(r"$\varepsilon$")
    axes[0].set_ylabel(r"$q_{\mathrm{rel}}$")
    axes[0].set_title("all-round versus first-round predicate", color=INK)
    axes[0].legend(loc="lower right")
    d = [(r["q_rel_full_minus_first"] or 0.0) for r in table]
    axes[1].plot(e, d, color=BELL_COLOR["Psi+"], marker="o", markersize=3.6,
                 markerfacecolor="white", markeredgewidth=1.3)
    axes[1].axhline(0.0, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
    axes[1].set_xlabel(r"$\varepsilon$")
    axes[1].set_ylabel(r"$q_{\mathrm{rel}}^{\mathrm{full}} - "
                       r"q_{\mathrm{rel}}^{\mathrm{first}}$")
    axes[1].set_title("difference (measured, not assumed)", color=INK)
    for ax in axes:
        _despine(ax)
    fig.tight_layout()
    return save(fig, "4q_qrel_first_vs_full")


def plot_components(table):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    e = [r["epsilon"] for r in table]
    keys = {"Phi-": ("q_rel_Phi_minus_first", "q_rel_Phi_minus_all"),
            "Psi+": ("q_rel_Psi_plus_first", "q_rel_Psi_plus_all"),
            "Psi-": ("q_rel_Psi_minus_first", "q_rel_Psi_minus_all")}
    for k, (kf, ka) in keys.items():
        axes[0].plot(e, [r[kf] for r in table], color=BELL_COLOR[k],
                     marker=BELL_MARKER[k], markersize=3.8,
                     markerfacecolor="white", markeredgewidth=1.2,
                     label=BELL_TEX[k])
        axes[1].plot(e, [r[ka] for r in table], color=BELL_COLOR[k],
                     marker=BELL_MARKER[k], markersize=3.8,
                     markerfacecolor="white", markeredgewidth=1.2,
                     label=BELL_TEX[k])
    axes[0].set_title("component threshold, first round", color=INK)
    axes[1].set_title("component threshold, all rounds", color=INK)
    for ax in axes:
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel(r"$q$")
        ax.legend(loc="lower right", ncol=3)
        _despine(ax)
    fig.tight_layout()
    return save(fig, "4q_qrel_components")


def plot_trajectories(traj_blocks, stem, quantity):
    """3 rows (below / near / above) x len(NEAR_EPS) columns."""
    import matplotlib.pyplot as plt
    n_col = len(traj_blocks)
    fig, axes = plt.subplots(3, n_col, figsize=(3.3 * n_col, 8.2),
                             sharex=True)
    axes = np.atleast_2d(axes)
    for j, blk in enumerate(traj_blocks):
        for i, (tag, q, rows) in enumerate(blk["runs"]):
            ax = axes[i][j]
            n = [r["n"] for r in rows]
            for k, name in enumerate(BELL_ORDER):
                if quantity == "R" and name == "Phi+":
                    continue
                y = [r[quantity][k] for r in rows]
                ax.plot(n, y, color=BELL_COLOR[name], marker=BELL_MARKER[name],
                        markersize=3.0, markerfacecolor="white",
                        markeredgewidth=1.0, label=BELL_TEX[name])
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(rf"$\varepsilon = {blk['eps']:.4f}$", color=INK)
            ax.text(0.97, 0.95, f"{tag}\nq = {q:.8f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8.5, color=INK2)
            if j == 0:
                ax.set_ylabel(r"$R_i^{(n)}$" if quantity == "R"
                              else r"$p_i^{(n)}$")
            if i == 2:
                ax.set_xlabel("round $n$")
            _despine(ax)
    axes[0][0].legend(loc="lower left", ncol=3, fontsize=8)
    fig.suptitle("relative error ratios" if quantity == "R"
                 else "absolute Bell populations", color=INK, y=0.995)
    fig.tight_layout()
    return save(fig, stem)


def plot_abs_vs_rel(comparison):
    import matplotlib.pyplot as plt
    rows = comparison["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    e = [r["epsilon"] for r in rows]
    axes[0].plot(e, [r["q_abs"] for r in rows], color=BELL_COLOR["Phi-"],
                 marker="s", markersize=3.8, markerfacecolor="white",
                 markeredgewidth=1.2, label=r"$q_{\mathrm{abs}}$ (frozen)")
    axes[0].plot(e, [r["q_rel"] for r in rows], color=BELL_COLOR["Phi+"],
                 marker="o", markersize=3.8, markerfacecolor="white",
                 markeredgewidth=1.2, label=r"$q_{\mathrm{rel}}$ (this study)")
    axes[0].set_title("4Q absolute versus relative threshold", color=INK)
    axes[0].legend(loc="lower right")
    axes[1].plot(e, [r["q_rel_minus_q_abs"] for r in rows],
                 color=BELL_COLOR["Psi+"], marker="o", markersize=3.8,
                 markerfacecolor="white", markeredgewidth=1.2)
    axes[1].axhline(0.0, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
    axes[1].set_title(r"$q_{\mathrm{rel}} - q_{\mathrm{abs}}$", color=INK)
    for ax in axes:
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel(r"$q$")
        _despine(ax)
    fig.tight_layout()
    return save(fig, "4q_qrel_abs_vs_rel")


# ===========================================================================
# 12. output helpers
# ===========================================================================

def write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
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
    ap.add_argument("--quick", action="store_true",
                    help="fewer epsilon values and a coarser dense sweep")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    eps_values = (0.05, 0.15, 0.40, 2.0 / 3.0) if args.quick else EPS_TABLE

    print("=" * 76)
    print("4Q RELATIVE Bell-error-suppression threshold  q_rel(epsilon)")
    print("=" * 76)
    print("  failure condition:  R_i^(n+1) - R_i^(n) > VIOL_TOL,  "
          "R_i = p_i / p_Phi+")
    print("  (NOT a condition on Delta p_i)")
    print(f"\n  circuit source: {CIRCUIT_SOURCE}")
    print(f"  noise: {NOISE_CONVENTION}")
    print("\n  gate sequence:")
    print(format_sequence())

    print("\n[1] provenance of the gate list")
    prov = provenance_checks()
    for k, v in prov.items():
        print(f"      {k:46s} {v}")

    print("\n[2] structural / channel checks")
    struct = structural_checks(np.random.default_rng(11))
    for k, v in struct.items():
        print(f"      {k:46s} {v}")

    print("\n[3] route A vs route B (independent implementation)")
    route_points = [(0.05, 0.0), (0.05, 0.02), (0.15, 0.05), (0.40, 0.12),
                    (2.0 / 3.0, 0.14), (0.90, 0.3), (0.20, 1.0)]
    routes = route_comparison(route_points)
    for k, v in routes["worst"].items():
        print(f"      max {k:42s} {v:.6e}")

    print("\n[4] measured round-off floor and transverse growth")
    floor = measure_error_floor()
    for k, v in floor.items():
        print(f"      {k:46s} {v}")
    tg = transverse_growth()
    print(f"      transverse first-round amplitude ratio         "
          f"{tg['first_round_ratio']:.6e}   (eps={tg['eps']}, q={tg['q']}, "
          f"input amp {tg['amplitude']:g})")
    print(f"      transverse amplitude-scaling exponent         "
          f"{tg['amplitude_scaling_exponent']:.6f}")
    print(f"      amplitude after one round                     "
          f"{tg['max_amplitude_after_first_round']:.6e}  "
          f"later ratios are round-off level: "
          f"{tg['later_ratios_are_roundoff_level']} "
          f"(max later ratio {tg['max_later_ratio_roundoff']:.3f})")
    print(f"      tolerances: STEP_TOL={STEP_TOL:g} VIOL_TOL={VIOL_TOL:g} "
          f"N_MAX={N_MAX} OFFDIAG_TRUST={OFFDIAG_TRUST:g} RATIO_FLOOR="
          f"{RATIO_FLOOR:g}")

    print("\n[5] primary threshold table")
    table, extra = main_table(eps_values)

    print("\n[6] degenerate endpoints eps = 0 and eps = 1")
    degen = degenerate_endpoints()
    for d in degen:
        print(f"      eps={d['epsilon']}: R0={d['R0']}  "
              f"has_root_in_scan={d['has_root_in_scan']}  "
              f"q_threshold={d['q_threshold']!r}  "
              f"sign_changes={d['n_sign_changes']}  "
              f"none_violating={d['none_violating']}")
        for pr in d["probes"]:
            print(f"          q={pr['q']:<8g} violated={pr['violated']!r:<6} "
                  f"round={pr['first_violating_round']!r:<5} "
                  f"comp={pr['component']!r:<8} "
                  f"worst_dR={pr['worst_delta_R']:+.6e}")

    print("\n[7] violation-tolerance sensitivity")
    sens = tolerance_sensitivity(eps_values if args.quick
                                 else (0.05, 0.15, 0.40, 2.0 / 3.0))
    for r in sens:
        print(f"      eps={r['epsilon']:<9.6f} " + "  ".join(
            f"{k.split('viol_tol_')[1]}: {r[k]!r}" for k in r
            if k.startswith("q_rel_full_viol")) + f"   spread={r['spread']!r}")

    print("\n[8] negative / sensitivity controls")
    nc = negative_controls()
    print(f"      baseline q_rel_full(eps=0.15) = {nc['baseline_q_rel_full']!r}")
    for r in nc["rows"]:
        print(f"      {r['control']:34s} d|rho|={r['max_abs_rho_diff']:.3e}  "
              f"dR={r['max_abs_ratio_diff']:.3e}  "
              f"q_rel_shift={r['q_rel_full_shift']!r:<24} "
              f"has_root={r['modified_has_root_in_scan']!r:<6} "
              f"all_viol={r['modified_all_violating']!r:<6} "
              f"detected={r['detected']}")

    print("\n[9] near-threshold trajectories")
    traj_rows, traj_blocks = [], []
    small_delta = 5e-04
    for eps in (NEAR_EPS if not args.quick else eps_values):
        q_star = next((r["q_rel_full"] for r in table
                       if abs(r["epsilon"] - eps) < 1e-12), None)
        if q_star is None:
            q_star = q_rel_full(eps)["q_threshold"]
        runs = []
        for tag, q in (("below", max(0.0, q_star - small_delta)),
                       ("near", q_star),
                       ("above", q_star + small_delta)):
            rows, fv = record_trajectory(eps, q, 40)
            runs.append((tag, q, rows))
            for r in rows:
                traj_rows.append({
                    "epsilon": eps, "q": q, "tag": tag, "n": r["n"],
                    **{f"p_{BELL_ORDER[k]}": float(r["p"][k]) for k in range(4)},
                    **{f"R_{BELL_ORDER[k]}": float(r["R"][k]) for k in range(4)},
                    **{f"dp_{BELL_ORDER[k]}": float(r["dp"][k]) for k in range(4)},
                    **{f"dR_{BELL_ORDER[k]}": float(r["dR"][k]) for k in range(4)},
                    "p_succ": float(r["p_succ"]),
                    "bell_offdiagonal_weight": float(r["c_off"]),
                    "first_violation_round": (fv or {}).get("round"),
                    "first_violation_component": (fv or {}).get("component")})
            print(f"      eps={eps:<9.6f} {tag:<6} q={q:.10f}  "
                  f"first violation: {fv}")
        traj_blocks.append({"eps": eps, "q_star": q_star, "runs": runs})

    print("\n[10] absolute-rise / relative-fall census")
    all_events = [ev for eps in extra for ev in extra[eps]["events"]]
    print(f"      total events found: {len(all_events)}")
    if all_events:
        ex = max(all_events, key=lambda e: e["delta_p"])
        print(f"      example: eps={ex['epsilon']:.6f} q={ex['q']:.10f} "
              f"round={ex['round']} component={ex['component']}")
        print(f"               p_i: {ex['p_n']:.12f} -> {ex['p_np1']:.12f}  "
              f"(delta_p = {ex['delta_p']:+.6e})")
        print(f"               p_Phi+: {ex['p_phi_plus_n']:.12f} -> "
              f"{ex['p_phi_plus_np1']:.12f}")
        print(f"               R_i: {ex['R_n']:.12f} -> {ex['R_np1']:.12f}  "
              f"(delta_R = {ex['delta_R']:+.6e})")

    print("\n[11] dense epsilon sweep (secondary)")
    dense = []
    if not args.quick:
        for eps in np.round(np.arange(0.02, 0.961, 0.02), 6):
            r = q_rel_full(float(eps))
            dense.append({"epsilon": float(eps), "q_rel_full": r["q_threshold"],
                          "q_upper_violating": r["q_upper_violating"],
                          "n_sign_changes": r["n_sign_changes"]})
        print(f"      {len(dense)} points; "
              f"sign-change counts observed: "
              f"{sorted({d['n_sign_changes'] for d in dense})}")

    print("\n[12] SECONDARY comparison with the frozen q_abs values")
    comparison = secondary_q_abs_comparison(table)
    if comparison.get("available"):
        print(f"      source: {comparison['source']}   "
              f"matched epsilon values: {comparison['n']}")
        print(f"      q_abs <= q_rel everywhere: "
              f"{comparison['q_abs_le_q_rel_everywhere']}")
        print(f"      q_rel - q_abs in "
              f"[{comparison['min_difference']:.10f}, "
              f"{comparison['max_difference']:.10f}]")
        for r in comparison["rows"]:
            print(f"          eps={r['epsilon']:<9.6f} q_abs={r['q_abs']:.10f} "
                  f"q_rel={r['q_rel']:.10f} "
                  f"diff={r['q_rel_minus_q_abs']:+.10f}")
    else:
        print(f"      {comparison}")

    # ---- files -------------------------------------------------------
    write_csv(OUT / "4q_qrel_circuit.csv", table)
    if dense:
        write_csv(OUT / "4q_qrel_dense.csv", dense)
    write_csv(OUT / "4q_qrel_trajectories.csv", traj_rows)
    if comparison.get("available"):
        write_csv(OUT / "4q_qrel_abs_vs_rel.csv", comparison["rows"])
    write_csv(OUT / "4q_qrel_route_comparison.csv", routes["rows"])
    write_csv(OUT / "4q_qrel_negative_controls.csv", nc["rows"])
    write_csv(OUT / "4q_qrel_tolerance_sensitivity.csv", sens)
    if all_events:
        write_csv(OUT / "4q_qrel_abs_rise_rel_fall.csv", all_events)
    write_csv(OUT / "4q_qrel_component_thresholds.csv",
              [{"epsilon": eps, "component": c, "scope": scope,
                "q_threshold": extra[eps]["components"][f"{c}_{scope}"]["q_threshold"],
                "q_upper_violating": extra[eps]["components"][f"{c}_{scope}"]["q_upper_violating"],
                "n_sign_changes": extra[eps]["components"][f"{c}_{scope}"]["n_sign_changes"]}
               for eps in extra for c in NON_TARGET for scope in ("first", "all")])
    write_csv(OUT / "4q_qrel_degenerate_endpoints.csv",
              [{"epsilon": d["epsilon"], "q": pr["q"],
                "violated": pr["violated"],
                "first_violating_round": pr["first_violating_round"],
                "component": pr["component"], "rounds_run": pr["rounds_run"],
                "worst_delta_R": pr["worst_delta_R"],
                "p_succ_first_round": pr["p_succ_first_round"],
                "R0": " ".join(f"{x:.12e}" for x in d["R0"]),
                "p1": " ".join(f"{x:.12e}" for x in pr["p1"])}
               for d in degen for pr in d["probes"]])

    figures = []
    if not args.no_figures:
        print("\n[13] figures")
        try:
            import matplotlib
            matplotlib.use("Agg")
            use_style()
            figures += plot_curve(table, dense)
            figures += plot_first_vs_full(table)
            figures += plot_components(table)
            figures += plot_trajectories(traj_blocks, "4q_qrel_trajectories_R", "R")
            figures += plot_trajectories(traj_blocks, "4q_qrel_trajectories_p", "p")
            if comparison.get("available"):
                figures += plot_abs_vs_rel(comparison)
            for f in figures:
                print(f"      {f}")
        except Exception as exc:                                  # pragma: no cover
            print(f"      figures skipped: {exc}")

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/verify_4q_qrel_circuit.py",
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_head": _git("rev-parse", "HEAD"),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "platform": platform.platform(), "quick_mode": bool(args.quick),
        "definition": ("q_rel(eps) = sup { q : R_i^(n+1) < R_i^(n) for every "
                       "i in {Phi-,Psi+,Psi-} and every n >= 0 }, "
                       "R_i = p_i / p_Phi+"),
        "failure_condition": "R_i^(n+1) - R_i^(n) > VIOL_TOL",
        "circuit_source": CIRCUIT_SOURCE,
        "noise_convention": NOISE_CONVENTION,
        "gate_sequence": format_sequence(),
        "repeated_round_rule": ("rho_n (x) rho_n -> noisy 5-CNOT circuit -> H(q3)"
                                " -> postselect (q3,q4) = (0,0) -> trace out "
                                "(q3,q4) -> normalise; the exact full 4x4 "
                                "conditional state is fed into the next round"),
        "not_used_in_primary_path": [
            "pqec_distill.analytic_exact_map", "analytic_map_4q",
            "pqec_distill.noisy_analytics",
            "any closed-form 4Q Bell-diagonal recurrence",
            "frozen q_abs values as bracketing hints",
            "any inferred formula for q_rel"],
        "tolerances": {"STEP_TOL": STEP_TOL, "VIOL_TOL": VIOL_TOL,
                       "N_MAX": N_MAX, "OFFDIAG_TRUST": OFFDIAG_TRUST,
                       "RATIO_FLOOR": RATIO_FLOOR, "bisection_tol": 1e-10,
                       "coarse_scan_points": 91, "q_scan_upper": 0.9,
                       "near_threshold_delta": small_delta},
        "epsilon_table": list(eps_values),
        "epsilon_degenerate": list(EPS_DEGENERATE),
        "provenance": prov, "structural": struct,
        "route_comparison": routes["worst"],
        "error_floor": floor, "transverse_growth": tg,
        "table": table,
        "degenerate_endpoints": degen,
        "tolerance_sensitivity": sens,
        "negative_controls": {"baseline_q_rel_full": nc["baseline_q_rel_full"],
                              "rows": nc["rows"]},
        "n_absolute_rise_relative_fall_events": len(all_events),
        "absolute_rise_relative_fall_example": (
            max(all_events, key=lambda e: e["delta_p"]) if all_events else None),
        "secondary_q_abs_comparison": comparison,
        "figures": figures,
        "total_seconds": time.time() - t_start,
    }
    with (OUT / "4q_qrel_verification.json").open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    print(f"\n  written to {OUT.relative_to(ROOT)}")
    print(f"  total {time.time() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
