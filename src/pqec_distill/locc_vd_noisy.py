"""Noisy 6Q-L16: the distributed LOCC virtual-distillation circuit with the two
local Fredkins decomposed into the canonical Type-3 gate sequence and a
two-qubit replacement depolarizing channel after every CNOT.

WHAT IS GIVEN (not derived here)
--------------------------------
* the 6Q LOCC architecture and wire ordering of :mod:`pqec_distill.locc_vd`,
  ``[a, b, A1, B1, A2, B2]`` with indices 0..5;
* the canonical Type-3 Fredkin decomposition used by the existing 5Q Type-3
  study -- ``_c2(b,a); _tof(q,a,b); _c2(b,a)`` with the Clifford+T Toffoli, 8
  CNOTs per Fredkin -- transcribed below from
  PQEC-Operational-Threshold/verify_analytic_decomposed.py (``_c2``/``_tof``/
  ``_fred``, commit a4969947, sha256 dd8de89220cb0b53...), the same sequence
  that ``scripts/verify_type3_qabs_circuit.py`` uses;
* the per-CNOT two-qubit REPLACEMENT depolarizing channel, taken from this
  repository's own :func:`pqec_distill.noise.replacement_depolarizing` rather
  than reimplemented here.

Alice applies ``Fredkin(a; A1, A2)`` on wires (0, 2, 4) and Bob applies
``Fredkin(b; B1, B2)`` on wires (1, 3, 5).  No gate touches both parties, so
all 16 CNOTs are party-local: 8 on Alice's side, 8 on Bob's.

Noise is applied ONLY after CNOTs.  Single-qubit gates, the ancilla ``|+>``
preparation and the read-out basis rotations are ideal, so the noisy quantum
core is identical for the XX and YY measurement settings; those settings differ
only in which ideal ancilla observable is read.

WHAT IS NOT ASSUMED
-------------------
Nothing here assumes Bell-diagonal preservation, isotropy preservation, any
closed-form noisy map, or any threshold.  The reconstructed operator is called
``T_q(rho)`` and is NOT called ``rho^2``; at ``q > 0`` it is whatever the
circuit produces, and its hermiticity, trace, positivity and Bell off-diagonal
content are measured, never imposed.
"""

from __future__ import annotations

import numpy as np

from .locc_vd import (
    A, A1, A2, B, B1, B2, N_QUBITS_6Q, PAULI_LABELS, correlator_6q,
    data_input, embed_three_qubit_gate, fredkin_matrix, pauli_product,
    plus_state_dm, reconstruct_operator,
)
from .noise import replacement_depolarizing

__all__ = [
    "N_WIRES", "DIM", "ALICE_TRIPLE", "BOB_TRIPLE", "ALICE_WIRES", "BOB_WIRES",
    "fredkin_ops", "L16_OPS", "N_CNOT_ALICE", "N_CNOT_BOB", "N_CNOT",
    "CIRCUIT_SOURCE", "format_sequence", "cnot_wire_pairs",
    "crosses_party_cut", "local_block_unitary", "circuit_unitary",
    "build_program", "PROGRAM", "final_state_6q_noisy",
    "deltas_all_paulis_noisy", "effective_operator", "BELL", "BELL_ORDER",
    "bell_populations", "bell_offdiagonal_frobenius", "bell_matrix",
    "isotropic_rho", "bell_diagonal_rho", "phi_plus_fidelity",
]

N_WIRES = N_QUBITS_6Q           # 6
DIM = 2 ** N_WIRES              # 64

#: Alice's three wires (ancilla, both of her data qubits)
ALICE_TRIPLE = (A, A1, A2)      # (0, 2, 4)
#: Bob's three wires
BOB_TRIPLE = (B, B1, B2)        # (1, 3, 5)
ALICE_WIRES = frozenset(ALICE_TRIPLE)
BOB_WIRES = frozenset(BOB_TRIPLE)

CIRCUIT_SOURCE = (
    "canonical Type-3 Fredkin decomposition: "
    "PQEC-Operational-Threshold/verify_analytic_decomposed.py _c2/_tof/_fred, "
    "commit a4969947, sha256 "
    "dd8de89220cb0b5315f0c5f7b9c2c8cf483980ac96d9ad9cccc1bd3f8e7a7837; "
    "same sequence as scripts/verify_type3_qabs_circuit.py")

NOISE_CONVENTION = (
    "two-qubit REPLACEMENT depolarizing applied after every CNOT on that "
    "CNOT's two wires, via pqec_distill.noise.replacement_depolarizing")


# ---------------------------------------------------------------------------
# the canonical Type-3 Fredkin decomposition, remapped onto a wire triple
# ---------------------------------------------------------------------------

def fredkin_ops(q: int, a: int, b: int) -> list:
    """``_fred(q, a, b) = _c2(b,a); _tof(q,a,b); _c2(b,a)``.

    ``_tof`` is the Clifford+T Toffoli with control1 = q, control2 = a,
    target = b: six CNOTs, single-qubit gates ideal and left in their original
    positions, because the noise sees the decomposition and not merely the
    unitary.  Eight CNOTs in total.
    """
    ops = [("CNOT", (b, a))]                                   # _c2(b, a)
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
    ops += [("CNOT", (b, a))]                                  # _c2(b, a)
    return ops


#: Alice's block then Bob's block.  They act on disjoint wires and commute; the
#: simulator keeps this fixed order.
L16_OPS = fredkin_ops(*ALICE_TRIPLE) + fredkin_ops(*BOB_TRIPLE)

_ALICE_BLOCK = fredkin_ops(*ALICE_TRIPLE)
_BOB_BLOCK = fredkin_ops(*BOB_TRIPLE)
N_CNOT_ALICE = sum(1 for kind, _ in _ALICE_BLOCK if kind == "CNOT")
N_CNOT_BOB = sum(1 for kind, _ in _BOB_BLOCK if kind == "CNOT")
N_CNOT = sum(1 for kind, _ in L16_OPS if kind == "CNOT")


def cnot_wire_pairs(ops=None) -> list:
    return [tuple(arg) for kind, arg in (ops or L16_OPS) if kind == "CNOT"]


def crosses_party_cut(ops=None) -> list:
    """Two-qubit gates with one wire on each side of the Alice/Bob cut."""
    bad = []
    for pair in cnot_wire_pairs(ops):
        sides = {("A" if w in ALICE_WIRES else "B") for w in pair}
        if len(sides) != 1:
            bad.append(pair)
    return bad


def format_sequence(ops=None) -> str:
    lines, k = [], 0
    for kind, arg in (ops or L16_OPS):
        if kind == "CNOT":
            k += 1
            side = "Alice" if arg[0] in ALICE_WIRES else "Bob"
            lines.append(f"  {len(lines)+1:3d}.  CNOT({arg[0]} -> {arg[1]})"
                         f"   [{side}, CNOT #{k:2d}]  + D_q on "
                         f"({arg[0]},{arg[1]})")
        else:
            side = "Alice" if arg in ALICE_WIRES else "Bob"
            lines.append(f"  {len(lines)+1:3d}.  {kind}({arg})   [{side}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# dense gates on the six-wire register
# ---------------------------------------------------------------------------

_I2 = np.eye(2, dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
_P0 = np.array([[1, 0], [0, 0]], dtype=complex)
_P1 = np.array([[0, 0], [0, 1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_ONE_QUBIT = {"H": _H, "T": _T, "Tdg": _T.conj().T}


def _kron_list(mats):
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def _single(name: str, wire: int) -> np.ndarray:
    factors = [_I2] * N_WIRES
    factors[wire] = _ONE_QUBIT[name]
    return _kron_list(factors)


def _cnot(control: int, target: int) -> np.ndarray:
    a = [_I2] * N_WIRES
    a[control] = _P0
    b = [_I2] * N_WIRES
    b[control] = _P1
    b[target] = _X
    return _kron_list(a) + _kron_list(b)


def _op_matrix(kind, arg) -> np.ndarray:
    return _cnot(*arg) if kind == "CNOT" else _single(kind, arg)


def circuit_unitary(ops=None) -> np.ndarray:
    """Noiseless 64x64 unitary of the decomposed sequence."""
    u = np.eye(DIM, dtype=complex)
    for kind, arg in (ops or L16_OPS):
        u = _op_matrix(kind, arg) @ u
    return u


def local_block_unitary(party: str) -> np.ndarray:
    """The 64x64 unitary of one party's 8-CNOT block."""
    if party not in ("alice", "bob"):
        raise ValueError("party must be 'alice' or 'bob'")
    return circuit_unitary(_ALICE_BLOCK if party == "alice" else _BOB_BLOCK)


def ideal_local_fredkin(party: str) -> np.ndarray:
    """The ideal Fredkin on that party's triple, for comparison."""
    triple = ALICE_TRIPLE if party == "alice" else BOB_TRIPLE
    return embed_three_qubit_gate(fredkin_matrix(), triple, N_WIRES)


def build_program(ops=None):
    """[('U', M) | ('D', (i,j))]: runs of unitaries merged, channel locations
    left exactly where the CNOTs are."""
    program, acc = [], np.eye(DIM, dtype=complex)
    for kind, arg in (ops or L16_OPS):
        acc = _op_matrix(kind, arg) @ acc
        if kind == "CNOT":
            program.append(("U", acc))
            acc = np.eye(DIM, dtype=complex)
            program.append(("D", tuple(arg)))
    program.append(("U", acc))
    return program


PROGRAM = build_program()


# ---------------------------------------------------------------------------
# the noisy circuit
# ---------------------------------------------------------------------------

def initial_state(rho: np.ndarray) -> np.ndarray:
    """|+><+|_a (x) |+><+|_b (x) rho_(A1B1) (x) rho_(A2B2) in [a,b,A1,B1,A2,B2]."""
    plus = plus_state_dm()
    return np.kron(plus, np.kron(plus, data_input(rho)))


def final_state_6q_noisy(rho: np.ndarray, q: float, program=None) -> np.ndarray:
    """Run the decomposed circuit with replacement depolarizing after each CNOT."""
    sigma = initial_state(rho)
    for kind, payload in (program or PROGRAM):
        if kind == "U":
            sigma = payload @ sigma @ payload.conj().T
        else:
            sigma = replacement_depolarizing(sigma, payload, q, n_qubits=N_WIRES)
    return sigma


def deltas_all_paulis_noisy(rho: np.ndarray, q: float, program=None) -> dict:
    """{label: (d_P, C_XX, C_YY)} with d_P = C_XX - C_YY, from the noisy state.

    The noisy quantum core is run once; the XX and YY settings differ only in
    the ideal ancilla observable, so no extra noise enters the read-out.
    """
    sigma = final_state_6q_noisy(rho, q, program)
    out = {}
    for lab in PAULI_LABELS:
        o = pauli_product(lab)
        c_xx = correlator_6q(sigma, "X", o)
        c_yy = correlator_6q(sigma, "Y", o)
        out[lab] = (c_xx - c_yy, c_xx, c_yy)
    return out


def effective_operator(rho: np.ndarray, q: float, program=None) -> dict:
    """T_q, D_q = Tr T_q, and R_q = T_q / D_q, with their measured properties.

    No interpretation of T_q is made and nothing is symmetrised or truncated.
    """
    deltas = deltas_all_paulis_noisy(rho, q, program)
    d = {k: v[0] for k, v in deltas.items()}
    t = reconstruct_operator(d)
    denom = float(np.real(np.trace(t)))
    herm = float(np.linalg.norm(t - t.conj().T))
    r = t / denom if denom != 0.0 and np.isfinite(denom) else None
    info = {"T": t, "D": denom, "R": r, "deltas": d, "raw": deltas,
            "T_hermiticity": herm, "D_equals_d_II": abs(denom - d["II"])}
    if r is not None:
        rh = 0.5 * (r + r.conj().T)
        eig = np.linalg.eigvalsh(rh)
        info.update({"R_hermiticity": float(np.linalg.norm(r - r.conj().T)),
                     "R_trace": float(np.real(np.trace(r))),
                     "R_min_eig": float(eig.min()),
                     "R_max_eig": float(eig.max()),
                     "R_is_psd_1e-10": bool(eig.min() > -1e-10)})
    return info


# ---------------------------------------------------------------------------
# Bell-basis diagnostics (measured, never imposed)
# ---------------------------------------------------------------------------

PHI_P = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
PHI_M = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
PSI_P = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
PSI_M = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
BELL = {"Phi+": PHI_P, "Phi-": PHI_M, "Psi+": PSI_P, "Psi-": PSI_M}
BELL_ORDER = ("Phi+", "Phi-", "Psi+", "Psi-")
_BELL_ROWS = np.stack([BELL[k] for k in BELL_ORDER])


def bell_matrix(m: np.ndarray) -> np.ndarray:
    """<B_i| m |B_j> for the four Bell states, in BELL_ORDER."""
    return _BELL_ROWS.conj() @ m @ _BELL_ROWS.T


def bell_populations(m: np.ndarray) -> np.ndarray:
    """The four Bell diagonal entries, each computed independently."""
    return np.array([float(np.real(v.conj() @ m @ v)) for v in _BELL_ROWS])


def bell_offdiagonal_frobenius(m: np.ndarray) -> float:
    """Frobenius norm of the off-diagonal block of m in the Bell basis."""
    r = bell_matrix(m)
    return float(np.linalg.norm(r - np.diag(np.diag(r))))


def isotropic_rho(eps: float) -> np.ndarray:
    e = float(eps)
    return (1 - e) * np.outer(PHI_P, PHI_P.conj()) + e * np.eye(4, dtype=complex) / 4


def bell_diagonal_rho(p) -> np.ndarray:
    out = np.zeros((4, 4), dtype=complex)
    for w, v in zip(p, _BELL_ROWS):
        out = out + float(w) * np.outer(v, v.conj())
    return out


def phi_plus_fidelity(m: np.ndarray) -> float:
    return float(np.real(PHI_P.conj() @ m @ PHI_P))
