"""6Q-L14: the distributed two-ancilla Type-4 virtual-distillation circuit.

CONSTRUCTION
------------
Take the standalone 7-CNOT local block ``G7`` that the 14-CNOT Type-4 sequence
factorises into (verified in :mod:`pqec_distill.type4_7cnot_fredkin` and by
``scripts/verify_type4_7cnot_factorization.py``) and give each party its own
ancilla and its own copy of ``G7``:

    ordering  [a, b, A1, B1, A2, B2]  ->  0, 1, 2, 3, 4, 5
    Alice     G7(a; A1, A2) = G7(0; 2, 4)          7 CNOTs
    Bob       G7(b; B1, B2) = G7(1; 3, 5)          7 CNOTs

Both ancillas start in ``|+>``; the four-data-qubit input ``R`` lives on
``(A1, B1, A2, B2)``.  No gate touches both parties, so all 14 CNOTs are
party-local and there are no cross-party two-qubit gates.  Noise is applied
only after CNOTs, so the noisy quantum core is identical for the XX and YY
read-out settings.

The estimator is

    Delta_6_T4(q; O, R) = < X_a X_b (x) O (x) I > - < Y_a Y_b (x) O (x) I >

evaluated on the full 64x64 noisy density matrix.  No Bell projection, no
twirling, no reduced analytic map, no re-isotropisation.

NOT USED
--------
No analytic Type-4 recurrence and no closed-form noisy map; nothing from
:mod:`pqec_distill.analytic_exact_map` is imported.  The reconstructed operator
is called ``T_q`` and is not called ``rho^2``.
"""

from __future__ import annotations

import numpy as np

from .noise import replacement_depolarizing
from .type4_7cnot_fredkin import (
    G7_OPS, build_program, embed_ops, fredkin_matrix, gate_matrix,
    ops_cnot_pairs, ops_support,
)

__all__ = [
    "N_WIRES", "DIM", "WIRES_6Q", "A", "B", "A1", "B1", "A2", "B2",
    "ALICE_TRIPLE", "BOB_TRIPLE", "ALICE_WIRES", "BOB_WIRES", "L14_OPS",
    "ALICE_BLOCK", "BOB_BLOCK", "N_CNOT_ALICE", "N_CNOT_BOB", "N_CNOT",
    "CIRCUIT_SOURCE", "NOISE_CONVENTION", "PROGRAM", "cnot_wire_pairs",
    "crosses_party_cut", "format_sequence", "circuit_unitary",
    "local_block_unitary", "ideal_local_fredkin", "embed_three_qubit_gate",
    "initial_state", "data_input", "final_state_6q_t4_noisy",
    "ancilla_0011_block",
]

N_WIRES = 6
DIM = 2 ** N_WIRES
WIRES_6Q = ("a", "b", "A1", "B1", "A2", "B2")
A, B, A1, B1, A2, B2 = 0, 1, 2, 3, 4, 5
ALICE_TRIPLE = (A, A1, A2)        # (0, 2, 4)
BOB_TRIPLE = (B, B1, B2)          # (1, 3, 5)
ALICE_WIRES = frozenset(ALICE_TRIPLE)
BOB_WIRES = frozenset(BOB_TRIPLE)

CIRCUIT_SOURCE = (
    "the standalone 7-CNOT local block G7 extracted and verified from the "
    "canonical 14-CNOT Type-4 sequence "
    "(pqec_distill.type4_7cnot_fredkin.G7_OPS), applied once per party")

NOISE_CONVENTION = (
    "two-qubit REPLACEMENT depolarizing applied after every CNOT on that "
    "CNOT's two wires, via pqec_distill.noise.replacement_depolarizing")

ALICE_BLOCK = embed_ops(G7_OPS, ALICE_TRIPLE)
BOB_BLOCK = embed_ops(G7_OPS, BOB_TRIPLE)
#: Alice's block then Bob's block; they act on disjoint wires and commute, and
#: the simulator keeps this fixed order.
L14_OPS = ALICE_BLOCK + BOB_BLOCK

N_CNOT_ALICE = len(ops_cnot_pairs(ALICE_BLOCK))
N_CNOT_BOB = len(ops_cnot_pairs(BOB_BLOCK))
N_CNOT = len(ops_cnot_pairs(L14_OPS))

PROGRAM = build_program(L14_OPS, N_WIRES)


def cnot_wire_pairs(ops=None) -> list:
    return ops_cnot_pairs(ops if ops is not None else L14_OPS)


def crosses_party_cut(ops=None) -> list:
    """Two-qubit gates with one wire on each side of the Alice/Bob cut."""
    bad = []
    for kind, arg in (ops if ops is not None else L14_OPS):
        if kind != "CNOT":
            continue
        sides = {("A" if w in ALICE_WIRES else "B") for w in ops_support(kind, arg)}
        if len(sides) != 1:
            bad.append(tuple(arg))
    return bad


def format_sequence(ops=None) -> str:
    lines, k = [], 0
    for kind, arg in (ops if ops is not None else L14_OPS):
        sup = ops_support(kind, arg)
        side = "Alice" if sup <= ALICE_WIRES else ("Bob" if sup <= BOB_WIRES
                                                   else "CROSS-PARTY")
        if kind == "CNOT":
            k += 1
            lines.append(f"  {len(lines)+1:3d}.  CNOT({arg[0]} -> {arg[1]})"
                         f"   [{side}, CNOT #{k:2d}]  + D_q on "
                         f"({arg[0]},{arg[1]})")
        else:
            w, t, p, l = arg
            lines.append(f"  {len(lines)+1:3d}.  U3(wire {w}; "
                         f"theta={t/np.pi:+.4f}pi, phi={p/np.pi:+.4f}pi, "
                         f"lam={l/np.pi:+.4f}pi)   [{side}]")
    return "\n".join(lines)


def circuit_unitary(ops=None) -> np.ndarray:
    """Noiseless 64x64 unitary of the decomposed sequence."""
    u = np.eye(DIM, dtype=complex)
    for kind, arg in (ops if ops is not None else L14_OPS):
        u = gate_matrix(kind, arg, N_WIRES) @ u
    return u


def local_block_unitary(party: str) -> np.ndarray:
    if party not in ("alice", "bob"):
        raise ValueError("party must be 'alice' or 'bob'")
    return circuit_unitary(ALICE_BLOCK if party == "alice" else BOB_BLOCK)


def embed_three_qubit_gate(gate8: np.ndarray, wires) -> np.ndarray:
    """Embed an 8x8 three-qubit gate on arbitrary wires of the 6-wire register.

    Built here by tensor-leg transposition, independently of
    :mod:`pqec_distill.locc_vd`.
    """
    wires = tuple(int(w) for w in wires)
    rest = [w for w in range(N_WIRES) if w not in wires]
    big = np.kron(gate8, np.eye(2 ** len(rest), dtype=complex))
    t = big.reshape([2] * (2 * N_WIRES))
    order = list(wires) + rest
    perm = np.argsort(order)
    axes = list(perm) + [N_WIRES + p for p in perm]
    return t.transpose(axes).reshape(DIM, DIM)


def ideal_local_fredkin(party: str) -> np.ndarray:
    """The ideal Fredkin on that party's triple, for comparison."""
    triple = ALICE_TRIPLE if party == "alice" else BOB_TRIPLE
    return embed_three_qubit_gate(fredkin_matrix(), triple)


# ---------------------------------------------------------------------------
# states
# ---------------------------------------------------------------------------

def data_input(rho: np.ndarray) -> np.ndarray:
    """``rho_(A1B1) (x) rho_(A2B2)`` in the order ``(A1, B1, A2, B2)``."""
    rho = np.asarray(rho, dtype=complex)
    return np.kron(rho, rho)


def initial_state(data_r: np.ndarray) -> np.ndarray:
    """``|+><+|_a (x) |+><+|_b (x) R`` on ``[a, b, A1, B1, A2, B2]``."""
    data_r = np.asarray(data_r, dtype=complex)
    if data_r.shape != (16, 16):
        raise ValueError(f"R must be 16x16, got {data_r.shape}")
    plus = 0.5 * np.ones((2, 2), dtype=complex)
    return np.kron(plus, np.kron(plus, data_r))


def final_state_6q_t4_noisy(data_r: np.ndarray, q: float,
                            program=None) -> np.ndarray:
    """Run 6Q-L14 with replacement depolarizing after each of the 14 CNOTs."""
    sigma = initial_state(data_r)
    for kind, payload in (program if program is not None else PROGRAM):
        if kind == "U":
            sigma = payload @ sigma @ payload.conj().T
        else:
            sigma = replacement_depolarizing(sigma, payload, float(q),
                                             n_qubits=N_WIRES)
    return sigma


def ancilla_0011_block(sigma6: np.ndarray) -> np.ndarray:
    """``<00|_(ab) sigma |11>_(ab)``: a 16x16 operator on ``(A1, B1, A2, B2)``."""
    t = np.asarray(sigma6, dtype=complex).reshape([2] * (2 * N_WIRES))
    return t[0, 0, :, :, :, :, 1, 1, :, :, :, :].reshape(16, 16)
